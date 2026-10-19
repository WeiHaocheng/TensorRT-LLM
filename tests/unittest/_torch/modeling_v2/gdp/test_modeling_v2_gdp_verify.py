# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdp_verify catalog entry, at nemotron4-nano-preview's tp1 and tp4-rank shapes."""

import math
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_verify import gdp_verify as op
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

__extra_import_path__ = [".."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gdp_verify requires a CUDA device"

#: A v2 (`MambaHybridCacheManagerV2`) slot holds the ssm states of all 32 GDP
#: layers back to back; the view under test is this one's.
_V2_LAYERS, _V2_LAYER = 32, 13


def _conv_dim(spec) -> int:
    """The GDP conv's channels at a cell's geometry, [V0..V2 | K0..K2 | Q]: 16896 at
    24 heads, 4224 at 6."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    return nh * h * dv + (nh + 1) * h * dk


def _conv_state_floats(spec) -> int:
    """What a v1 (`CppMambaHybridCacheManager`) slot carries after a GDP layer's
    ssm state: its conv state, conv_dim x (kernel 4 - 1) bf16 values, in floats."""
    return _conv_dim(spec) * 3 * 2 // 4


def _rows_view(proj: torch.Tensor, start: int, inner: tuple, tokens: int | None) -> torch.Tensor:
    """Columns ``start:`` of the in_proj-shaped ``proj`` as ``[rows, *inner]`` (``tokens`` None)
    or ``[rows / tokens, tokens, *inner]``, every view keeping ``proj``'s row stride for a token
    -- the views the targets hand the GDP entries."""
    strides, size = [], 1
    for n in reversed(inner):
        strides.insert(0, size)
        size *= n
    rows, row = proj.shape[0], proj.stride(0)
    if tokens is None:
        shape, stride = (rows, *inner), (row, *strides)
    else:
        shape, stride = (rows // tokens, tokens, *inner), (tokens * row, row, *strides)
    return proj.as_strided(shape, stride, proj.storage_offset() + start)


def _compare_out(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The output band of `gdp_recurrent`'s test, for the same reason: both
    sides compute each token's output in fp32 and round it to bf16 once, so
    what separates them is which way an element on a rounding boundary goes --
    one bf16 step at most, which is up to 2 ulp of its row's scale. The fp32
    state's drift (below) is far too small to move anything but such a case.

    Measured over 32 seeds of every cell up to B = 64 and 8 above: each side
    within 0.996 ulp of an fp64 transcription (one rounding); kernel against
    `reference` 1.97 ulp element-wise and 0.045 of relative RMS, 0.065% of the
    outputs differing at all.

    So: 2 ulp element-wise, the bound itself, and 0.25 of relative RMS, 5x the
    measured worst case. The mistakes only this gate sees are far out: the
    decay applied on a request's first token only, or each token restarted
    from the slot, both at least 5.2e2 ulp element-wise and 16 of relative RMS
    from fp64.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)


def _compare_states(states: torch.Tensor, ref: torch.Tensor) -> None:
    """Snapshots `[n, T, H, V, K]` against `reference`'s: `gdp_recurrent`'s
    state band for the first, widened by sqrt(t + 1) for the one after token t.

    Both sides keep the state in fp32 and differ only in the order fp32 rounds
    each token's decay, dot products, deltas and rank-one updates, so the
    snapshot after token t carries t + 1 tokens of those roundings. Measured
    over 32 seeds of every cell up to B = 64 and 8 above, kernel against
    `reference`, snapshot by snapshot (t = 0, 1, 2): 3.66 / 5.69 / 6.68 fp32
    ulp of the row scale element-wise, 0.63 / 0.91 / 1.17 of relative RMS --
    one token's error (`gdp_recurrent`'s 3.74 / 0.575) growing as independent
    roundings add, about as sqrt(t + 1). Each side is that far from an fp64
    transcription too (kernel 5.46 / 0.874, `reference` 5.08 / 0.790 over all
    snapshots).

    So snapshot t gets 8 sqrt(t + 1) ulp element-wise and sqrt(t + 1) of
    relative RMS: `gdp_recurrent`'s 8 / 1 for the first; about 2x the measured
    worst case element-wise and 1.5x in RMS at every depth. The vendored FLA
    verify kernel lands inside (6.52 / 0.849 over all snapshots); the nearest
    real mistakes are three orders out or more: snapshots rounded to bf16, 3.2e4 /
    1.4e4 from fp64; the snapshot taken before a token's last Householder
    step, 9.6e6 / 6.4e5; position t holding the state before token t, 7.5e7 /
    1.6e6.
    """
    for t in range(states.shape[1]):
        depth = math.sqrt(t + 1)
        assert_within_ulp(
            states[:, t],
            ref[:, t],
            element_ulp=8.0 * depth,
            rms_ulp=depth,
            ulp=ULP[torch.float32],
        )


def _num_slots(spec) -> int:
    """v1: two slots per row plus three, so the rows' slots are scattered through a
    pool they do not fill. v2: one per row plus three -- a v2 slot is 25 MB."""
    return spec["batch"] + 3 if spec["pool"] == "v2" else 2 * spec["batch"] + 3


def _raw_pool(spec, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """The memory a GDP layer's pool view lives in, every float of it a non-zero
    state the op must leave alone.

    v1: a guard block (the previous layer's last), then per slot this layer's
    ssm state followed by its conv state. v2: per slot the 32 layers' ssm states
    back to back, the view taking layer 13's, so the other 31 of every slot
    surround it."""
    ssm = spec["heads"] * spec["value_dim"] * spec["key_dim"]
    if spec["pool"] == "v2":
        shape = (num_slots, _V2_LAYERS * ssm)
    else:
        shape = (num_slots + 1, ssm + _conv_state_floats(spec))
    return torch.randn(shape, generator=g, device="cuda") * 0.5


def _pool_view(raw: torch.Tensor, spec) -> torch.Tensor:
    """`[num_slots, H, V, K]` over `raw`, slot stride = the block (v1) or the 32 layers (v2)."""
    h, dv, dk = spec["heads"], spec["value_dim"], spec["key_dim"]
    ssm = h * dv * dk
    if spec["pool"] == "v2":
        return raw[:, _V2_LAYER * ssm : (_V2_LAYER + 1) * ssm].view(-1, h, dv, dk)
    return raw[1:, :ssm].view(-1, h, dv, dk)


def _raw_intermediate(spec, g: torch.Generator) -> torch.Tensor:
    """What a layer's view of `intermediate_ssm_states` lives in: a guard row
    (the previous layer's last), the batch's rows, and two rows past the batch,
    as a max_batch_size above the graph batch leaves them. Every float non-zero."""
    shape = (spec["batch"] + 3, spec["tokens"], spec["heads"], spec["value_dim"], spec["key_dim"])
    return torch.randn(shape, generator=g, device="cuda")


def _inputs(spec, batch: int, g: torch.Generator):
    """(q, k, v, g, beta, proj) as the target holds them after the conv of a
    verify step, which writes its post-SiLU output in place into the in_proj
    output's conv columns: the generation rows are request-major, one in_proj row
    per token, so q, k and v are `[B, T, ...]` views of `proj`, packed [V0..V2 |
    K0..K2 | Q] as `Nemotron4GDPMixer` splits it, a token's row stride the in_proj
    width and a request's T times it. The rest of each row is NaN, which a read
    outside the views would carry into the result. beta is the bf16 sigmoid of an
    in_proj slice; g is HF's fp32 decay over the checkpoint's parameter ranges
    (exp(A_log) in [0.0061, 18.3], dt_bias in [-7.03, -1.94] across its 32 GDP
    layers), so some heads forget within a token and others barely at all."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    tokens = spec["tokens"]
    conv_dim = _conv_dim(spec)
    mixed = F.silu(torch.randn(batch * tokens, conv_dim, generator=g, device="cuda"))
    proj = torch.full(
        (batch * tokens, spec["proj_width"]), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    lo = spec["conv_offset"]
    proj[:, lo : lo + conv_dim].copy_(mixed)
    v = _rows_view(proj, lo, (nh, h, dv), tokens)
    k = _rows_view(proj, lo + nh * h * dv, (nh, h, dk), tokens)
    q = _rows_view(proj, lo + nh * h * (dv + dk), (h, dk), tokens)
    b = torch.randn(batch, tokens, nh * h, generator=g, device="cuda").to(torch.bfloat16)
    beta = b.sigmoid().view(batch, tokens, nh, h)
    a_log = torch.empty(h, device="cuda").uniform_(math.log(0.0061), math.log(18.3), generator=g)
    dt_bias = torch.empty(h, device="cuda").uniform_(-7.03, -1.94, generator=g).to(torch.bfloat16)
    a = torch.randn(batch, tokens, h, generator=g, device="cuda").to(torch.bfloat16)
    decay = -a_log.exp() * F.softplus(a.float() + dt_bias.float())
    return q, k, v, decay, beta, proj


def _slots(batch: int, pad_rows: int, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """Distinct random slots for the real rows, PAD_SLOT_ID for the padding rows
    at the end of the batch -- where a CUDA-graph batch puts them."""
    real = torch.randperm(num_slots, generator=g, device="cuda")[: batch - pad_rows]
    pad = torch.full((pad_rows,), PAD_SLOT_ID, device="cuda")
    return torch.cat([real, pad]).to(torch.int32)


def _build(spec, seed: int):
    """One cell: (args, the pool's raw memory, the intermediate's raw memory). The in_proj
    buffer q, k and v view is `args["q"]`'s base."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_slots = _num_slots(spec)
    pool_raw = _raw_pool(spec, num_slots, g)
    inter_raw = _raw_intermediate(spec, g)
    q, k, v, decay, beta, _ = _inputs(spec, spec["batch"], g)
    args = dict(
        q=q,
        k=k,
        v=v,
        g=decay,
        beta=beta,
        state_pool=_pool_view(pool_raw, spec),
        slots=_slots(spec["batch"], spec["pad_rows"], num_slots, g),
        intermediate=inter_raw[1:],
        scale=spec["scale"],
    )
    return args, pool_raw, inter_raw


def _real_rows(slots: torch.Tensor) -> torch.Tensor:
    return (slots >= 0).nonzero().squeeze(-1)


def _check(out, ref_out, ref_inter, args, pool_raw, pool_before, inter_raw, inter_before) -> None:
    """Outputs and every real request's T snapshots against `reference`; the
    pool's memory bit-identical to before, and so is every float of the
    intermediate's memory but those snapshots -- padding rows, rows past the
    batch and the guard row where a write one row early would land."""
    _compare_out(out, ref_out)
    real = _real_rows(args["slots"])
    tokens = args["q"].shape[1]
    _compare_states(args["intermediate"][real, :tokens], ref_inter[real, :tokens])
    assert torch.equal(pool_raw, pool_before), "the op wrote to the pool"
    owned = torch.zeros(inter_raw.shape[:2], dtype=torch.bool, device=inter_raw.device)
    owned[1:][real, :tokens] = True
    assert torch.equal(inter_raw[~owned], inter_before[~owned]), (
        "the op wrote outside its requests' snapshots"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args, pool_raw, inter_raw = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    pool_before, inter_before = pool_raw.clone(), inter_raw.clone()
    ref_out, ref_inter = op.reference(**args)
    out = op(**args)
    assert out.shape == (spec["batch"], spec["tokens"], spec["heads"], spec["value_dim"])
    assert out.dtype == args["q"].dtype
    _check(out, ref_out, ref_inter, args, pool_raw, pool_before, inter_raw, inter_before)


def test_cuda_graph_replays_verify_commit_verify() -> None:
    """One capture, two replays with the engine's commit between them: each
    replay reads whatever the input buffers, the slot list and the pool hold
    at replay time, and the second continues from the snapshots the commit
    promoted -- the same slots in another row order."""
    spec = next(c.spec for c in op.CELLS if c.spec["batch"] == 64 and c.spec["tokens"] == 3)
    args, pool_raw, inter_raw = _build(spec, seed=64)
    g = torch.Generator(device="cuda").manual_seed(65)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**args)  # compiles the kernel outside the capture
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**args)

    tokens = spec["tokens"]
    n_real = spec["batch"] - spec["pad_rows"]
    real_slots = args["slots"][:n_real].clone()
    for replay in range(2):
        fresh = _inputs(spec, spec["batch"], g)
        for name, value in zip(("q", "k", "v", "g", "beta"), fresh[:5]):
            args[name].copy_(value)
        order = real_slots[torch.randperm(n_real, generator=g, device="cuda")]
        args["slots"][:n_real].copy_(order)
        pool_before, inter_before = pool_raw.clone(), inter_raw.clone()
        ref_out, ref_inter = op.reference(**args)
        graph.replay()
        _check(out, ref_out, ref_inter, args, pool_raw, pool_before, inter_raw, inter_before)

        # `update_mamba_states`: each request's accepted snapshot into its slot.
        accepted = torch.randint(1, tokens + 1, (n_real,), generator=g, device="cuda")
        rows = torch.arange(n_real, device="cuda")
        args["state_pool"][order.long()] = args["intermediate"][rows, accepted - 1]


def test_rows_may_share_a_slot() -> None:
    """The slot is only read, so two rows on one slot -- how the cache managers
    pad a CUDA-graph batch, every padding row on the dummy request's slot --
    each continue that slot's state: the guard admits it and both rows get the
    first row's bits."""
    spec = next(c.spec for c in op.CELLS if c.spec["batch"] == 4 and c.spec["tokens"] == 3)
    args, _, _ = _build(spec, seed=4)
    slots = args["slots"].clone()
    slots[1] = slots[0]
    for name in ("q", "k", "v", "g", "beta"):
        args[name][1].copy_(args[name][0])
    with validating(op):
        out = op(**dict(args, slots=slots))
    assert torch.equal(out[1], out[0])
    assert torch.equal(args["intermediate"][1], args["intermediate"][0])


def test_reference_agrees_with_the_vendored_fla_verify_kernel() -> None:
    """`reference`'s step order, per-token snapshots and padding against an
    implementation written by someone else: the vendored fused-recurrent kernel
    the engine's GDN verify path runs, with its intermediate-state buffer and
    its pool update disabled, on the tokens' Householder steps expanded the way
    `Nemotron4GDPMixer.forward` expands them. Its snapshot after each token's
    last step is the token's state; its output there is the token's output."""
    from tensorrt_llm._torch.modules.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )

    spec = next(c.spec for c in op.CELLS if c.spec["batch"] == 16 and c.spec["tokens"] == 3)
    args, _, _ = _build(spec, seed=16)
    ref_out, ref_inter = op.reference(**args)

    batch, tokens, nh, h, dk = args["k"].shape
    dv = args["v"].shape[-1]
    steps = tokens * nh
    query = torch.zeros_like(args["k"])
    query[:, :, -1] = args["q"]
    decay = torch.zeros_like(args["beta"], dtype=torch.float32)
    decay[:, :, 0] = args["g"]
    slots = args["slots"]
    pad = slots < 0
    pool = args["state_pool"][torch.where(pad, 0, slots).long()].contiguous()
    rows = torch.where(pad, PAD_SLOT_ID, torch.arange(batch, device="cuda")).to(torch.int32)
    buffer = torch.zeros(batch, steps, h, dv, dk, device="cuda")
    fla_out = fused_recurrent_gated_delta_rule_update(
        q=query.reshape(batch, steps, h, dk),
        k=args["k"].reshape(batch, steps, h, dk),
        v=args["v"].reshape(batch, steps, h, dv),
        g=decay.reshape(batch, steps, h),
        beta=args["beta"].reshape(batch, steps, h),
        scale=args["scale"],
        initial_state_source=pool,
        initial_state_indices=rows,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        intermediate_states_buffer=buffer,
        cache_steps=steps,
    )
    real = _real_rows(slots)
    _compare_out(fla_out.view(batch, tokens, nh, h, dv)[:, :, -1], ref_out)
    snapshots = buffer.view(batch, tokens, nh, h, dv, dk)[:, :, -1]
    _compare_states(snapshots[real], ref_inter[real, :tokens])


def _ok(**over):
    """The draft-2 batch-4 cell's arguments with some of them replaced."""
    spec = next(c.spec for c in op.CELLS if c.spec["batch"] == 4 and c.spec["tokens"] == 3)
    args, _, _ = _build(spec, seed=4)
    args.update(over)
    return args


def test_guard_refuses_a_pool_in_hf_state_layout() -> None:
    """HF's [H, K, V] state has the pool's element count; read through the
    [H, V, K] strides the kernel takes, it comes back transposed."""
    args = _ok()
    hf_layout = args["state_pool"].transpose(-1, -2).contiguous()
    unguarded = op(**dict(args, state_pool=hf_layout))
    assert unguarded.isfinite().all(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="state_pool must be"):
        with validating(op):
            op(**dict(args, state_pool=hf_layout))


def test_guard_refuses_a_bf16_pool() -> None:
    """Loaded without complaint, every verify starting from a rounded state."""
    args = _ok()
    bf16_pool = args["state_pool"].to(torch.bfloat16)
    unguarded = op(**dict(args, state_pool=bf16_pool))
    assert unguarded.isfinite().all(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="recurrent state in float32"):
        with validating(op):
            op(**dict(args, state_pool=bf16_pool))


def test_guard_refuses_a_bf16_intermediate() -> None:
    """Stored without complaint; the commit would copy rounded states into the pool."""
    args = _ok()
    bf16_inter = torch.zeros_like(args["intermediate"], dtype=torch.bfloat16)
    op(**dict(args, intermediate=bf16_inter))
    assert bf16_inter.abs().sum() > 0, "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="snapshots the commit copies"):
        with validating(op):
            op(**dict(args, intermediate=bf16_inter))


def test_guard_refuses_an_intermediate_with_fewer_rows_than_requests() -> None:
    """Not run unguarded: the stores would run past the buffer."""
    args = _ok()
    short = args["intermediate"][: args["slots"].numel() - 1]
    with pytest.raises(AssertionError, match="intermediate must be"):
        with validating(op):
            op(**dict(args, intermediate=short))


def test_guard_refuses_an_intermediate_with_fewer_positions_than_tokens() -> None:
    """A draft-1 buffer under a draft-2 verify: each request's third snapshot
    would land on the next request's first."""
    args = _ok()
    short = torch.zeros_like(args["intermediate"][:, :2])
    with pytest.raises(AssertionError, match="intermediate must be"):
        with validating(op):
            op(**dict(args, intermediate=short))


def test_guard_refuses_a_bf16_decay() -> None:
    args = _ok()
    with pytest.raises(AssertionError, match="log decay. must be float32"):
        with validating(op):
            op(**dict(args, g=args["g"].to(torch.bfloat16)))


def test_guard_refuses_a_decay_that_grows_the_state() -> None:
    args = _ok()
    with pytest.raises(AssertionError, match="must be the log decay"):
        with validating(op):
            op(**dict(args, g=-args["g"]))


def test_guard_refuses_a_pre_sigmoid_beta() -> None:
    args = _ok()
    logits = torch.logit(args["beta"].float()).to(torch.bfloat16)
    with pytest.raises(AssertionError, match="post-sigmoid"):
        with validating(op):
            op(**dict(args, beta=logits))


def test_guard_refuses_a_head_major_k() -> None:
    """[B, T, H, n_h, K] has k's rank and size: read through the [B, T, n_h, H,
    K] strides the kernel takes, every step normalizes some other head's key."""
    args = _ok()
    head_major = args["k"].transpose(2, 3).contiguous()
    unguarded = op(**dict(args, k=head_major))
    assert unguarded.isfinite().all(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="k must be"):
        with validating(op):
            op(**dict(args, k=head_major))


def test_guard_refuses_a_householder_major_beta_read_head_major() -> None:
    args = _ok()
    with pytest.raises(AssertionError, match="beta must be"):
        with validating(op):
            op(**dict(args, beta=args["beta"].transpose(2, 3)))


def test_guard_refuses_a_slot_past_the_pool() -> None:
    args = _ok()
    slots = args["slots"].clone()
    slots[0] = args["state_pool"].shape[0]
    with pytest.raises(AssertionError, match="slots must be in"):
        with validating(op):
            op(**dict(args, slots=slots))
