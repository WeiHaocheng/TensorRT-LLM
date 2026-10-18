# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdp_chunk entry of nemotron4-nano-preview's tp1 target."""

import itertools
import math
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1.ops.gdp.gdp_chunk import (  # noqa: E501
    gdp_chunk as op,
)
from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from tensorrt_llm._torch.modules.fla.cumsum import chunk_local_cumsum

__extra_import_path__ = ["../../../.."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gdp_chunk requires a CUDA device"

#: What a slot of the hybrid cache manager's packed pool carries after its ssm
#: state: the GDP conv state, conv_dim 16896 x (kernel 4 - 1) bf16 values.
_CONV_STATE_FLOATS = 16896 * 3 * 2 // 4


_OUT_ELEMENT_ULP, _OUT_RMS_ULP = 32.0, 2.0
_STATE_ELEMENT_ULP, _STATE_RMS_ULP = 8.0, 2.0


def _compare_out(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The chunked algorithm against a per-token fp32 recurrence.

    The kernel carries the l2-normalized q and k, the WY factors, the per-chunk
    states and the corrected values in bf16 between its stages, and only the
    running state in fp32; `reference` is fp32 throughout and rounds once. The
    fp32 reference sits within one bf16 rounding of an fp64 transcription (at
    most 1.0 ulp element-wise, 0.44 of RMS), so the gap below is the kernel's.

    Measured over every cell at six seeds (the cell's own and five more, 108
    runs): relative RMS 0.80 to 1.32 ulp; element-wise at most 7.9 ulp against
    the row scale at the cells' own seeds and 12.3 at any seed. The large ones
    sit on rows 1% to 8% of the median row's size: a strong decay has just
    wiped the state, the row is what is left of the token's own three updates
    after they cancel, and its error scales with those updates rather than
    with the row.

    So: 32 ulp element-wise, 2 of relative RMS. The vendored wrapper, which
    drops the coupling of a token's Householder steps, misses by 234 to 288 ulp
    element-wise and 15 to 29 of RMS on the 7- to 2048-token cells.
    """
    assert_within_ulp(out, ref, element_ulp=_OUT_ELEMENT_ULP, rms_ulp=_OUT_RMS_ULP)


def _compare_state(state: torch.Tensor, ref: torch.Tensor) -> None:
    """Final states, scored in bf16 ulps though the pool is fp32: the chunk
    states and corrected values they are built from are bf16, so fp32's own ulp
    would measure the container. A row is one (head, value) column of K
    entries.

    Measured over the same 108 runs: element-wise at most 3.6 ulp against the
    row scale, relative RMS at most 1.16; the fp32 reference's states sit
    within 30 fp32 ulps (about 0.001 bf16 ulp) of fp64's.

    So: 8 ulp element-wise, 2 of RMS. The vendored wrapper misses by 108 to 272
    ulp element-wise and 9 to 33 of RMS.
    """
    assert_within_ulp(
        state,
        ref,
        element_ulp=_STATE_ELEMENT_ULP,
        rms_ulp=_STATE_RMS_ULP,
        ulp=ULP[torch.bfloat16],
    )


def _raw_pool(spec, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """The memory one layer's pool lives in: a guard block, then one block per
    slot, every float of it non-zero -- a saved state where a sequence resumes,
    a previous request's leftovers where it starts fresh, and memory the op must
    leave alone everywhere else. A block is ssm state followed by conv state,
    the hybrid cache manager's per-layer layout."""
    ssm = spec["heads"] * spec["value_dim"] * spec["key_dim"]
    return torch.randn(num_slots + 1, ssm + _CONV_STATE_FLOATS, generator=g, device="cuda") * 0.5


def _pool_view(raw: torch.Tensor, spec) -> torch.Tensor:
    """`[num_slots, H, V, K]` over `raw` past its guard block, slot stride = the block."""
    h, dv, dk = spec["heads"], spec["value_dim"], spec["key_dim"]
    return raw[1:, : h * dv * dk].view(-1, h, dv, dk)


def _inputs(spec, num_tokens: int, g: torch.Generator):
    """(q, k, v, g, beta) as the target holds them after the conv: q, k and v
    are views of one post-SiLU conv-output row per token, packed [V0..V2 |
    K0..K2 | Q] as `Nemotron4GDPMixer` splits it; beta is the bf16 sigmoid of an
    in_proj slice; g is HF's fp32 decay over the checkpoint's parameter ranges
    (exp(A_log) in [0.0061, 18.3], dt_bias in [-7.03, -1.94] across its 32
    GDP layers), so some heads forget within a token and others barely at all."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    conv_dim = nh * h * dv + (nh + 1) * h * dk
    mixed = F.silu(torch.randn(num_tokens, conv_dim, generator=g, device="cuda")).to(torch.bfloat16)
    v = mixed[:, : nh * h * dv].view(num_tokens, nh, h, dv)
    k = mixed[:, nh * h * dv : nh * h * (dv + dk)].view(num_tokens, nh, h, dk)
    q = mixed[:, nh * h * (dv + dk) :].view(num_tokens, h, dk)
    b = torch.randn(num_tokens, nh * h, generator=g, device="cuda").to(torch.bfloat16)
    beta = b.sigmoid().view(num_tokens, nh, h)
    a_log = torch.empty(h, device="cuda").uniform_(math.log(0.0061), math.log(18.3), generator=g)
    dt_bias = torch.empty(h, device="cuda").uniform_(-7.03, -1.94, generator=g).to(torch.bfloat16)
    a = torch.randn(num_tokens, h, generator=g, device="cuda").to(torch.bfloat16)
    decay = -a_log.exp() * F.softplus(a.float() + dt_bias.float())
    return q, k, v, decay, beta


def _build(spec, seed: int):
    """One cell: (args, the raw pool memory). Two slots per sequence plus three,
    so the sequences' slots are scattered through a pool they do not fill."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens = spec["seq_lens"]
    num_seqs = len(seq_lens)
    num_slots = 2 * num_seqs + 3
    raw = _raw_pool(spec, num_slots, g)
    q, k, v, decay, beta = _inputs(spec, sum(seq_lens), g)
    resume = {
        "fresh": [False] * num_seqs,
        "continued": [True] * num_seqs,
        "mixed": [n % 2 == 0 for n in range(num_seqs)],
    }[spec["init"]]
    args = dict(
        q=q,
        k=k,
        v=v,
        g=decay,
        beta=beta,
        state_pool=_pool_view(raw, spec),
        slots=torch.randperm(num_slots, generator=g, device="cuda")[:num_seqs].to(torch.int32),
        cu_seqlens=torch.tensor(
            [0, *itertools.accumulate(seq_lens)], dtype=torch.int32, device="cuda"
        ),
        has_initial_state=torch.tensor(resume, dtype=torch.bool, device="cuda"),
        scale=spec["scale"],
    )
    return args, raw


def _check(out, ref_out, ref_pool, args, raw, raw_before, spec) -> None:
    """Output and final states against `reference`; every other float of the
    pool's memory -- other slots, conv state, the guard block -- bit-identical
    to before."""
    _compare_out(out, ref_out)
    rows = args["slots"].long()
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
    ref_out, ref_pool = op.reference(**args)  # before the call: the op writes the pool
    out = op(**args)
    assert out.shape == args["q"].shape[:2] + (cell.spec["value_dim"],)
    assert out.dtype == args["q"].dtype
    _check(out, ref_out, ref_pool, args, raw, raw_before, cell.spec)


def _spec(seq_lens, init="fresh"):
    base = next(c.spec for c in op.CELLS)
    return dict(base, seq_lens=seq_lens, init=init)


def test_guard_refuses_pad_slot_id() -> None:
    """A negative slot indexes from the end: the op would resume from, and
    overwrite, the pool's last slot."""
    args, _ = _build(_spec((5, 7)), seed=1)
    args["slots"][1] = -1
    with pytest.raises(AssertionError, match="prefill rows have no padding slot"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_shared_slot() -> None:
    args, _ = _build(_spec((5, 7)), seed=2)
    args["slots"][1] = args["slots"][0]
    with pytest.raises(AssertionError, match="share a slot"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_positive_decay() -> None:
    args, _ = _build(_spec((9,)), seed=3)
    args["g"] = args["g"].abs()
    with pytest.raises(AssertionError, match="log decay"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_pre_sigmoid_beta() -> None:
    args, _ = _build(_spec((9,)), seed=4)
    args["beta"] = torch.logit(args["beta"].float()).to(torch.bfloat16) * 4
    with pytest.raises(AssertionError, match="post-sigmoid"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_bf16_decay() -> None:
    args, _ = _build(_spec((9,)), seed=5)
    args["g"] = args["g"].to(torch.bfloat16)
    with pytest.raises(AssertionError, match="must be float32"):
        with validating(op):
            op(**args)


def test_guard_refuses_an_hf_layout_pool() -> None:
    """HF's `[H, K, V]` state is the pool's `[H, V, K]` transposed."""
    args, _ = _build(_spec((9,)), seed=6)
    args["state_pool"] = args["state_pool"].transpose(-1, -2)
    with pytest.raises(AssertionError, match="K innermost"):
        with validating(op):
            op(**args)


def test_guard_refuses_offsets_that_miss_the_tokens() -> None:
    args, _ = _build(_spec((5, 7)), seed=7)
    args["cu_seqlens"] = args["cu_seqlens"].clone()
    args["cu_seqlens"][-1] -= 1
    with pytest.raises(AssertionError, match="run from 0 to the token count"):
        with validating(op):
            op(**args)


def _expanded(args):
    """The HF expansion as the vendored wrapper takes it: `[1, T * n_h, H, *]`,
    the query on each token's last sub-step and the decay on its first, q and k
    left for the wrapper to normalize."""
    q, k, v, g, beta = (args[n] for n in ("q", "k", "v", "g", "beta"))
    num_tokens, nh, h, dk = k.shape
    steps = num_tokens * nh
    q_steps = q.new_zeros(num_tokens, nh, h, dk)
    q_steps[:, -1] = q
    g_steps = g.new_zeros(num_tokens, nh, h)
    g_steps[:, 0] = g
    return (
        q_steps.view(1, steps, h, dk),
        k.contiguous().view(1, steps, h, dk),
        v.contiguous().view(1, steps, h, -1),
        g_steps.view(1, steps, h),
        beta.contiguous().view(1, steps, h),
    )


def _rms_ulp(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative RMS error in bf16 ulps, as `assert_within_ulp` scores it."""
    diff = (out.float() - ref.float()).pow(2).mean().sqrt()
    return (diff / ref.float().pow(2).mean().sqrt()).item() / ULP[torch.bfloat16]


def test_the_vendored_wrapper_drops_the_householder_coupling() -> None:
    """Why the entry calls the vendored chunk stages itself instead of handing
    `chunk_gated_delta_rule` the expanded sequence.

    The wrapper's `chunk_local_cumsum` is a parallel scan. Over a decay that is
    non-zero on every third sub-step its prefix sums are not flat across a
    token's sub-steps but wander by an ulp, and the stages weight each pair of
    sub-steps by `safe_exp(later - earlier)`: a one-ulp rise becomes a zero, and
    the coupling between the token's Householder steps is dropped. Measured on
    this cell: 15.2 ulp of relative RMS, 7.6 times the band -- and the entry's
    own stages fed that cumsum miss by the same amount, so it is the cumsum
    alone. If the wrapper comes back near the reference, the hazard is gone and
    with it the reason for `_decay_cumsum_kernel`.
    """
    cell = next(c for c in op.CELLS if c.spec["seq_lens"] == (512,) and c.spec["init"] == "fresh")
    args, _ = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref_out, _ = op.reference(**args)
    nh, h, dv, dk = (cell.spec[n] for n in ("householder", "heads", "value_dim", "key_dim"))
    q_steps, k_steps, v_steps, g_steps, beta_steps = _expanded(args)
    cu_steps = args["cu_seqlens"].long() * nh

    g_cumsum = chunk_local_cumsum(g_steps, chunk_size=64, cu_seqlens=cu_steps)[0]
    step = torch.arange(g_cumsum.shape[0], device="cuda")
    flat = (step % nh != 0) & (step % 64 != 0)  # adds zero, and is not a chunk's first
    rises = (g_cumsum[1:] > g_cumsum[:-1])[flat[1:]]
    assert rises.any(), "the vendored scan is now flat across a token's sub-steps"

    out, _ = chunk_gated_delta_rule(
        q_steps,
        k_steps,
        v_steps,
        g_steps,
        beta_steps,
        scale=cell.spec["scale"],
        initial_state=torch.zeros(1, h, dv, dk, device="cuda"),
        output_final_state=True,
        cu_seqlens=cu_steps,
        use_qk_l2norm_in_kernel=True,
    )
    rival = _rms_ulp(out.view(-1, nh, h, dv)[:, -1], ref_out)
    assert rival > 4 * _OUT_RMS_ULP, (
        f"the vendored wrapper now lands {rival:.2f} ulp of RMS from the reference: "
        "the cumsum hazard may be gone"
    )


def test_a_cold_autotune_cache_leaves_the_states_right() -> None:
    """The vendored indexed state path (`initial_state_indices` with
    `inplace_indexed_state_update`) writes the final states into the pool from
    inside `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, which is autotuned
    without restoring its arguments: the first call per shape runs it once per
    benchmarked config, each run advancing the pool again. The entry gathers the
    states into a buffer the kernel only reads and scatters the finals back after
    it, so a re-tune changes nothing. Both halves: the vendored path, cold and
    warm, disagrees with itself (measured: cold, the slot comes back NaN, the
    repeated updates having blown it up; warm, it repeats bit for bit); the
    entry, cold, matches the reference."""
    kernel = chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    while not hasattr(kernel, "cache"):  # the autotuner sits under the heuristics wrapper
        kernel = kernel.fn
    g = torch.Generator(device="cuda").manual_seed(17)
    h, dk, dv, steps = 24, 128, 64, 192
    k = F.normalize(torch.randn(1, steps, h, dk, generator=g, device="cuda"), dim=-1)
    w = torch.randn(1, steps, h, dk, generator=g, device="cuda") * 0.05
    u = torch.randn(1, steps, h, dv, generator=g, device="cuda")
    pool = torch.randn(4, h, dv, dk, generator=g, device="cuda")
    slots = torch.tensor([2], device="cuda")

    def vendored_indexed(cold: bool) -> torch.Tensor:
        state = pool.clone()
        if cold:
            kernel.cache.clear()
        chunk_gated_delta_rule_fwd_h(
            k=k.to(torch.bfloat16),
            w=w.to(torch.bfloat16),
            u=u.to(torch.bfloat16),
            initial_state=state,
            initial_state_indices=slots,
            inplace_indexed_state_update=True,
        )
        return state[2]

    cold, warm = vendored_indexed(cold=True), vendored_indexed(cold=False)
    gap = _rms_ulp(cold, warm)
    assert math.isnan(gap) or gap > 64, (  # repeated updates may also blow the state up
        f"the vendored indexed path, cold and warm, now agrees to {gap:.2f} ulp of RMS: "
        "it may no longer advance the state while it autotunes"
    )

    cell = next(
        c for c in op.CELLS if c.spec["seq_lens"] == (512,) and c.spec["init"] == "continued"
    )
    args, raw = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    raw_before = raw.clone()
    ref_out, ref_pool = op.reference(**args)
    kernel.cache.clear()
    out = op(**args)
    _check(out, ref_out, ref_pool, args, raw, raw_before, cell.spec)
