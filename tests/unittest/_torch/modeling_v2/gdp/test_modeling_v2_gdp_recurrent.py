# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdp_recurrent catalog entry, at nemotron4-nano-preview's tp1 and tp4-rank shapes."""

import math
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_recurrent import (
    gdp_recurrent as op,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

__extra_import_path__ = [".."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gdp_recurrent requires a CUDA device"

#: The head counts the cells certify: tp1's 24 and a tp4 rank's 6.
HEADS = sorted({c.spec["heads"] for c in op.CELLS}, reverse=True)


def _conv_dim(spec) -> int:
    """The GDP conv's channels at a cell's geometry, [V0..V2 | K0..K2 | Q]: 16896 at
    24 heads, 4224 at 6."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    return nh * h * dv + (nh + 1) * h * dk


def _conv_state_floats(spec) -> int:
    """What a slot of the hybrid cache manager's packed pool carries after its ssm
    state: the GDP conv state, conv_dim x (kernel 4 - 1) bf16 values, in floats."""
    return _conv_dim(spec) * 3 * 2 // 4


def _cell_spec(heads: int, batch: int):
    """The spec of the cell at this head count and batch."""
    return next(c.spec for c in op.CELLS if c.spec["heads"] == heads and c.spec["batch"] == batch)


def _compare_out(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides compute the output in fp32 and round it to bf16 once.

    Each lands within one bf16 rounding of an fp64 transcription -- 0.996 ulp
    of the row scale at worst, the kernel and `reference` alike, over 8 seeds
    of every cell -- so all that separates them is which way an element on a
    rounding boundary goes: one bf16 step at most, which is up to 2 ulp of the
    element's own magnitude and so of its row's. Measured: 1.47 ulp
    element-wise, 0.022 of relative RMS (0.065% of the outputs differ at all).
    On a tp4 rank's 6 heads, over 8 seeds of each of its cells: 1.87 ulp
    element-wise, a flip on an element in its row's top binade; 0.016 of
    relative RMS; at most 0.26% of the outputs differ, one of batch 1's 384.

    So: 2 ulp element-wise, the bound itself, and 0.25 of relative RMS, 10x
    the measured worst case. The mistake only this gate can see -- the output
    read one Householder step early, which leaves the state right -- sits 32
    ulp of relative RMS from fp64; reversed steps 6, the decay on every step 34.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)


def _compare_state(state: torch.Tensor, ref: torch.Tensor) -> None:
    """The state stays fp32 on both sides, so they differ only in the order fp32
    rounds one token: the decay, then per Householder step a 128-term dot
    product, the delta and the rank-one update. Each side sits ~3 fp32 ulp of
    its row's scale from an fp64 transcription (kernel 3.10, `reference` 3.16
    element-wise; 0.56 / 0.55 of relative RMS) and 3.74 / 0.575 from the other;
    on a tp4 rank's 6 heads, 3.84 / 0.62 from the other at worst over 8 seeds of
    each cell.

    So: 8 ulp element-wise and 1 of relative RMS, about 2x the worst case of
    every comparison here, the vendored FLA kernel's included (5.29 / 0.608;
    its l2 norm puts the eps outside the root, ~2 ulp away and immaterial).
    The nearest real mistakes sit three orders out in relative RMS: a
    bf16-rounded decay 1.5e3 ulp from fp64, a bf16 state 2.0e4, reversed
    Householder steps 1.2e5.
    """
    assert_within_ulp(state, ref, element_ulp=8.0, rms_ulp=1.0, ulp=ULP[torch.float32])


def _raw_pool(spec, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """The memory one layer's pool lives in: a guard block, then one block per
    slot, every float of it a non-zero state the op must leave alone unless it
    owns it. A packed block is ssm state followed by conv state, the hybrid
    cache manager's per-layer layout; a dense block is the ssm state alone."""
    ssm = spec["heads"] * spec["value_dim"] * spec["key_dim"]
    block = ssm + (_conv_state_floats(spec) if spec["pool"] == "packed" else 0)
    return torch.randn(num_slots + 1, block, generator=g, device="cuda") * 0.5


def _pool_view(raw: torch.Tensor, spec) -> torch.Tensor:
    """`[num_slots, H, V, K]` over `raw` past its guard block, slot stride = the block."""
    h, dv, dk = spec["heads"], spec["value_dim"], spec["key_dim"]
    return raw[1:, : h * dv * dk].view(-1, h, dv, dk)


def _inputs(spec, batch: int, g: torch.Generator):
    """(q, k, v, g, beta) as the target holds them after the decode conv, which
    writes its post-SiLU output in place into the in_proj output's conv columns:
    q, k and v are views of those columns, one in_proj row per request, packed
    [V0..V2 | K0..K2 | Q] as `Nemotron4GDPMixer` splits it, so their row stride
    is the in_proj width. The rest of each row is NaN, which a read outside the
    views would carry into the result. beta is the bf16 sigmoid of an in_proj
    slice; g is HF's fp32 decay over the checkpoint's parameter ranges
    (exp(A_log) in [0.0061, 18.3], dt_bias in [-7.03, -1.94] across its 32
    GDP layers), so some heads forget within a step and others barely at all."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    conv_dim = _conv_dim(spec)
    mixed = F.silu(torch.randn(batch, conv_dim, generator=g, device="cuda")).to(torch.bfloat16)
    proj = torch.full(
        (batch, spec["proj_width"]), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    conv = proj[:, spec["conv_offset"] : spec["conv_offset"] + conv_dim]
    conv.copy_(mixed)
    v = conv[:, : nh * h * dv].view(batch, nh, h, dv)
    k = conv[:, nh * h * dv : nh * h * (dv + dk)].view(batch, nh, h, dk)
    q = conv[:, nh * h * (dv + dk) :].view(batch, h, dk)
    b = torch.randn(batch, nh * h, generator=g, device="cuda").to(torch.bfloat16)
    beta = b.sigmoid().view(batch, nh, h)
    a_log = torch.empty(h, device="cuda").uniform_(math.log(0.0061), math.log(18.3), generator=g)
    dt_bias = torch.empty(h, device="cuda").uniform_(-7.03, -1.94, generator=g).to(torch.bfloat16)
    a = torch.randn(batch, h, generator=g, device="cuda").to(torch.bfloat16)
    decay = -a_log.exp() * F.softplus(a.float() + dt_bias.float())
    return q, k, v, decay, beta


def _slots(batch: int, pad_rows: int, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """Distinct random slots for the real rows, PAD_SLOT_ID for the padding rows
    at the end of the batch -- where a CUDA-graph batch puts them."""
    real = torch.randperm(num_slots, generator=g, device="cuda")[: batch - pad_rows]
    pad = torch.full((pad_rows,), PAD_SLOT_ID, device="cuda")
    return torch.cat([real, pad]).to(torch.int32)


def _build(spec, seed: int):
    """One cell: (args, the raw pool memory). Two slots per row plus three, so
    the real rows' slots are scattered through a pool they do not fill."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = spec["batch"]
    num_slots = 2 * batch + 3
    raw = _raw_pool(spec, num_slots, g)
    q, k, v, decay, beta = _inputs(spec, batch, g)
    args = dict(
        q=q,
        k=k,
        v=v,
        g=decay,
        beta=beta,
        state_pool=_pool_view(raw, spec),
        slots=_slots(batch, spec["pad_rows"], num_slots, g),
        scale=spec["scale"],
    )
    return args, raw


def _check(out, ref_out, ref_pool, args, raw, raw_before, spec) -> None:
    """Output and advanced state against `reference`; every other float of the
    pool's memory -- other slots, conv state, the guard block where a padding
    row would land if it wrote -- bit-identical to before."""
    _compare_out(out, ref_out)
    slots = args["slots"]
    rows = slots[slots >= 0].long()
    pool = args["state_pool"]
    _compare_state(pool[rows], ref_pool[rows])
    owned = torch.zeros_like(raw)
    _pool_view(owned, spec)[rows] = 1.0
    untouched = owned == 0
    assert torch.equal(raw[untouched], raw_before[untouched]), "the op wrote outside its slots"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, raw = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    raw_before = raw.clone()
    ref_out, ref_pool = op.reference(**args)  # before the call: the op advances the pool
    out = op(**args)
    assert out.shape == args["q"].shape[:2] + (cell.spec["value_dim"],)
    assert out.dtype == args["q"].dtype
    _check(out, ref_out, ref_pool, args, raw, raw_before, cell.spec)


@pytest.mark.parametrize("heads", HEADS)
def test_cuda_graph_replays_on_new_inputs(heads: int) -> None:
    """One capture, two replays: each replay reads whatever the input buffers,
    the slot list and the pool hold at replay time, and the second continues
    the states the first wrote -- the same slots in another row order."""
    spec = _cell_spec(heads, 64)
    args, raw = _build(spec, seed=64)
    g = torch.Generator(device="cuda").manual_seed(65)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**args)  # compiles the kernel outside the capture
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**args)

    real = args["slots"][: spec["batch"] - spec["pad_rows"]].clone()
    for replay in range(2):
        fresh = _inputs(spec, spec["batch"], g)
        for name, value in zip(("q", "k", "v", "g", "beta"), fresh):
            args[name].copy_(value)
        order = real[torch.randperm(real.numel(), generator=g, device="cuda")]
        args["slots"][: real.numel()].copy_(order)
        raw_before = raw.clone()
        ref_out, ref_pool = op.reference(**args)
        graph.replay()
        _check(out, ref_out, ref_pool, args, raw, raw_before, spec)


@pytest.mark.parametrize("heads", HEADS)
def test_a_row_does_not_depend_on_its_batch(heads: int) -> None:
    """Bit-identical output and state for a request whether it runs alone, in a
    batch of three or in the padded graph batch of 1024: a row's arithmetic is
    fixed by the kernel's tile, never by the batch around it. That is what lets
    a request decode to the same bits with CUDA graphs on (padded batches) and
    off (eager batches of whatever is running)."""
    spec = _cell_spec(heads, 1024)
    args, raw = _build(spec, seed=1024)
    raw_before = raw.clone()
    full_out = op(**args)
    full_pool = args["state_pool"].clone()
    for rows in (slice(0, 1), slice(1, 4)):
        raw.copy_(raw_before)
        sub = {name: args[name][rows] for name in ("q", "k", "v", "g", "beta", "slots")}
        out = op(**sub, state_pool=args["state_pool"], scale=args["scale"])
        assert torch.equal(out, full_out[rows]), f"rows {rows}: out depends on the batch"
        mine = args["slots"][rows].long()
        assert torch.equal(args["state_pool"][mine], full_pool[mine]), (
            f"rows {rows}: state depends on the batch"
        )


def test_reference_agrees_with_the_vendored_fla_kernel() -> None:
    """`reference`'s pool layout, step order and padding against an
    implementation written by someone else: the vendored fused-recurrent
    gated-delta-rule kernel run on the token's Householder steps expanded the
    way `Nemotron4GDPMixer.forward` expands them, over a dense pool."""
    from tensorrt_llm._torch.modules.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )

    spec = dict(_cell_spec(24, 16), pool="dense")
    args, _ = _build(spec, seed=16)
    ref_out, ref_pool = op.reference(**args)

    q, k, v = args["q"], args["k"], args["v"]
    query = torch.zeros_like(k)
    query[:, -1] = q
    decay = torch.zeros_like(args["beta"], dtype=torch.float32)
    decay[:, 0] = args["g"]
    pool = args["state_pool"]
    fla_out = fused_recurrent_gated_delta_rule_update(
        q=query,
        k=k,
        v=v,
        g=decay,
        beta=args["beta"],
        scale=args["scale"],
        initial_state_source=pool,
        initial_state_indices=args["slots"],
        use_qk_l2norm_in_kernel=True,
    )[:, -1]
    rows = args["slots"][args["slots"] >= 0].long()
    _compare_out(fla_out, ref_out)
    _compare_state(pool[rows], ref_pool[rows])


def _ok(**over):
    """The tp1 batch-4 cell's arguments with one of them replaced."""
    spec = _cell_spec(24, 4)
    args, _ = _build(spec, seed=4)
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
    """Loaded and stored without complaint, rounding the state every step."""
    args = _ok()
    bf16_pool = args["state_pool"].to(torch.bfloat16)
    unguarded = op(**dict(args, state_pool=bf16_pool))
    assert unguarded.isfinite().all(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="recurrent state in float32"):
        with validating(op):
            op(**dict(args, state_pool=bf16_pool))


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
    """[B, H, n_h, K] has k's rank and size: read through the [B, n_h, H, K]
    strides the kernel takes, every step normalizes some other head's key."""
    args = _ok()
    head_major = args["k"].transpose(1, 2).contiguous()
    unguarded = op(**dict(args, k=head_major))
    assert unguarded.isfinite().all(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="k must be"):
        with validating(op):
            op(**dict(args, k=head_major))


def test_guard_refuses_a_householder_major_beta_read_head_major() -> None:
    args = _ok()
    with pytest.raises(AssertionError, match="beta must be"):
        with validating(op):
            op(**dict(args, beta=args["beta"].transpose(1, 2)))


def test_guard_refuses_two_rows_on_one_slot() -> None:
    """How the cache managers pad a CUDA-graph batch: every padding row gets
    the dummy request's slot. The rows race on it."""
    args = _ok()
    slots = args["slots"].clone()
    slots[-1] = slots[0]
    with pytest.raises(AssertionError, match="share a slot"):
        with validating(op):
            op(**dict(args, slots=slots))


def test_guard_refuses_a_slot_past_the_pool() -> None:
    args = _ok()
    slots = args["slots"].clone()
    slots[0] = args["state_pool"].shape[0]
    with pytest.raises(AssertionError, match="slots must be in"):
        with validating(op):
            op(**dict(args, slots=slots))
