# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_gdp_chunk catalog entry."""

import itertools
import math
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp import (
    flashinfer_gdp_chunk as entry_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.flashinfer_gdp_chunk import (
    flashinfer_gdp_chunk as op,
)

__extra_import_path__ = [".."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_gdp_chunk requires a CUDA device"


def _conv_dim(spec) -> int:
    """The GDP conv's channels at a cell's geometry, [V0..V2 | K0..K2 | Q]: 16896 at 24 heads,
    4224 at 6."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    return nh * h * dv + (nh + 1) * h * dk


def _conv_state_floats(spec) -> int:
    """What a slot of the hybrid cache manager's packed pool carries after its ssm state: the
    GDP conv state, conv_dim x (kernel 4 - 1) bf16 values, in floats."""
    return _conv_dim(spec) * 3 * 2 // 4


_OUT_ELEMENT_ULP, _OUT_RMS_ULP = 32.0, 2.0
_STATE_ELEMENT_ULP, _STATE_RMS_ULP = 8.0, 2.0


def _compare_out(out: torch.Tensor, ref: torch.Tensor) -> None:
    """FlashInfer's chunked recurrence against a per-token fp32 one.

    The l2-normalized q and k, the padded v and the output are bf16; the gates and the state
    fp32. `reference` is fp32 throughout and rounds once, within one bf16 rounding of an fp64
    transcription, so the gap below is the kernel's.

    Measured over every cell at six seeds (the cell's own and five more: 240 runs over both
    geometries) with the kernel `_use_cp` picks: relative RMS 0.75 to 1.24 ulp; element-wise at
    most 11.5 ulp against the row scale at the cells' own seeds and 13.1 at any seed. Each
    kernel forced on the cells `test_both_kernels_meet_the_bands` drives (60 runs each): at most
    11.5 element-wise, RMS 1.00 to 1.25. `gdp_chunk` on the same 240 inputs: 13.8
    element-wise, RMS 0.80 to 1.33. The large ones sit, as for `gdp_chunk`, on rows a strong
    decay has just wiped, where the row is what is left of the token's three updates after they
    cancel and its error scales with those updates.

    So the bands are `gdp_chunk`'s: 32 ulp element-wise, 2 of relative RMS.
    """
    assert_within_ulp(out, ref, element_ulp=_OUT_ELEMENT_ULP, rms_ulp=_OUT_RMS_ULP)


def _compare_state(state: torch.Tensor, ref: torch.Tensor) -> None:
    """Final states, scored in bf16 ulps though the pool is fp32: the states are built from
    bf16 keys and values, so fp32's own ulp would measure the container. A row is one (head,
    value) column of K entries.

    Measured over the same 240 runs: element-wise at most 3.3 ulp against the row scale (2.5
    at the cells' own seeds), relative RMS at most 1.09; each kernel forced, 3.1 / 0.95 (CP)
    and 2.8 / 1.08 (non-CP); `gdp_chunk` on the same inputs, 3.5 / 1.18. So: 8 ulp
    element-wise, 2 of RMS, `gdp_chunk`'s bands.
    """
    assert_within_ulp(
        state,
        ref,
        element_ulp=_STATE_ELEMENT_ULP,
        rms_ulp=_STATE_RMS_ULP,
        ulp=ULP[torch.bfloat16],
    )


def _raw_pool(spec, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """The memory one layer's pool lives in: a guard block, then one block per slot, every
    float of it non-zero -- a saved state where a sequence resumes, a previous request's
    leftovers where it starts fresh, and memory the op must leave alone everywhere else. A block
    is ssm state followed by conv state, the hybrid cache manager's per-layer layout."""
    ssm = spec["heads"] * spec["value_dim"] * spec["key_dim"]
    block = ssm + _conv_state_floats(spec)
    return torch.randn(num_slots + 1, block, generator=g, device="cuda") * 0.5


def _pool_view(raw: torch.Tensor, spec) -> torch.Tensor:
    """`[num_slots, H, V, K]` over `raw` past its guard block, slot stride = the block."""
    h, dv, dk = spec["heads"], spec["value_dim"], spec["key_dim"]
    return raw[1:, : h * dv * dk].view(-1, h, dv, dk)


def _inputs(spec, num_tokens: int, g: torch.Generator):
    """(q, k, v, g, beta) as the target holds them after the conv: q, k and v are views of
    one post-SiLU conv-output row per token, packed [V0..V2 | K0..K2 | Q] as
    `Nemotron4GDPMixer` splits it; beta is the bf16 sigmoid of an in_proj slice; g is HF's fp32
    decay over the checkpoint's parameter ranges (exp(A_log) in [0.0061, 18.3], dt_bias in
    [-7.03, -1.94] across its 32 GDP layers), so some heads forget within a token and others
    barely at all."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    conv_dim = _conv_dim(spec)
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
    """One cell: (args, the raw pool memory). Two slots per sequence plus three, so the
    sequences' slots are scattered through a pool they do not fill."""
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
        "first+tail": [n == num_seqs - 1 for n in range(num_seqs)],
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
        max_seqlen=max(seq_lens),
    )
    return args, raw


def _check(out, ref_out, ref_pool, args, raw, raw_before, spec) -> None:
    """Output and final states against `reference`; every other float of the pool's memory --
    other slots, conv state, the guard block -- bit-identical to before."""
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


#: Cells both of FlashInfer's kernels can take, the CP choice aside: short and long single
#: sequences, both multi-sequence mixes and the protocol's two batches, at both geometries.
_BOTH_PATHS = [
    c
    for c in op.CELLS
    if c.spec["seq_lens"]
    in ((64,), (2048,), (3, 509, 2600, 5080), (1, 2, 17, 64, 129, 300, 511, 1024), (8160, 32))
    and c.spec["init"] != "fresh"
]


@pytest.mark.parametrize("use_cp", [True, False], ids=["cp", "non-cp"])
@pytest.mark.parametrize("cell", _BOTH_PATHS, ids=[c.why[:44] for c in _BOTH_PATHS])
def test_both_kernels_meet_the_bands(cell, use_cp) -> None:
    """`_use_cp` picks a kernel for speed alone, so each kernel answers within the bands on
    every batch, whichever one the rule would pick there."""
    args, raw = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31 + 1)
    raw_before = raw.clone()
    ref_out, ref_pool = op.reference(**args)
    call = {k: v for k, v in args.items() if k != "max_seqlen"}
    out = entry_module._flashinfer_gdp_chunk_fwd(**call, use_cp=use_cp)
    _check(out, ref_out, ref_pool, args, raw, raw_before, cell.spec)


#: (max_seqlen, num_tokens, num_seqs, heads) -> the faster kernel, measured on this machine:
#: CP / non-CP kernel time in ms beside each.
_MEASURED_CHOICES = [
    ((32, 32, 1, 24), False),  # 0.039 / 0.016
    ((512, 512, 1, 24), False),  # 0.073 / 0.053
    ((1536, 1536, 1, 24), True),  # 0.120 / 0.140
    ((8192, 8192, 1, 24), True),  # 0.334 / 0.675
    ((8160, 8192, 2, 24), True),  # (8160, 32): 0.351 / 0.674
    ((8192, 16384, 2, 24), True),  # (8192, 8192): 0.619 / 0.675
    ((5080, 8192, 4, 24), True),  # (3, 509, 2600, 5080): 0.439 / 0.430, a wash
    ((4000, 4192, 4, 24), True),  # (4000, 64, 64, 64): 0.241 / 0.340
    ((2048, 8192, 4, 24), False),  # 4 x 2048: 0.471 / 0.183
    ((1024, 8192, 8, 24), False),  # 8 x 1024: 0.512 / 0.200
    ((1024, 2048, 8, 24), False),  # (1, 2, ..., 1024): 0.245 / 0.119
    ((4096, 8192, 2, 6), True),  # (4096, 4096): 0.152 / 0.343
    ((8192, 16384, 2, 6), True),  # (8192, 8192): 0.222 / 0.671
    ((1024, 8192, 8, 6), False),  # 8 x 1024: 0.219 / 0.099
    ((512, 8192, 16, 6), False),  # 16 x 512: 0.277 / 0.060
]


@pytest.mark.parametrize("shape,cp", _MEASURED_CHOICES, ids=[str(s) for s, _ in _MEASURED_CHOICES])
def test_the_kernel_choice_follows_the_measurements(shape, cp) -> None:
    assert entry_module._use_cp(*shape) is cp


def test_repeat_calls_give_identical_bits() -> None:
    """Neither kernel autotunes, and CP's chunk length is a function of the shapes."""
    cell = next(c for c in op.CELLS if c.spec["seq_lens"] == (8160, 32))
    outs, states = [], []
    for _ in range(2):
        args, _ = _build(cell.spec, seed=11)
        outs.append(op(**args))
        states.append(args["state_pool"][args["slots"].long()].clone())
    assert torch.equal(outs[0], outs[1]) and torch.equal(states[0], states[1])


def _spec(seq_lens, init="fresh"):
    base = next(c.spec for c in op.CELLS)
    return dict(base, seq_lens=seq_lens, init=init)


def test_guard_refuses_pad_slot_id() -> None:
    """A negative slot indexes from the end: the op would resume from, and overwrite, the
    pool's last slot."""
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


def test_guard_refuses_values_wider_than_the_kernel_head() -> None:
    """The padding stops at 128: a wider value head would be cut, not refused."""
    args, _ = _build(dict(_spec((9,)), value_dim=192), seed=8)
    with pytest.raises(AssertionError, match="value_dim must be at most"):
        with validating(op):
            op(**args)
