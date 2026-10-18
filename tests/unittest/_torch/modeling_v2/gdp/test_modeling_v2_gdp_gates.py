# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdp_gates entry of nemotron4_nano_preview__sm_103__tp1.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the in_proj output a target hands the op (b and a as column
slices of one [T, 18528] buffer), drives every bf16 bit pattern through each gate, and holds the
inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_gates import gdp_gates as op

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdp_gates requires a CUDA device"

BF16 = torch.bfloat16

#: Scale of the cells' Gaussian in_proj outputs. The b and a columns of the checkpoint's in_proj
#: output sit within a few units of zero; 4 also reaches sigmoid's saturated tails and softplus'
#: linear stretch above the threshold (a + dt_bias > 20) on every large cell. Every bf16 value
#: each gate can take is driven by `test_every_bf16_input` regardless.
INPUT_SCALE = 4.0


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _flushed(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Where HF's value is an fp32 subnormal and the kernel's a zero of the same sign.

    The kernel's libdevice calls flush subnormal results to zero. In g that happens only where
    HF's decay is itself subnormal -- below 2^-126 in magnitude, a + dt_bias under about -84 --
    and the recurrences take g through exp(), which maps both to exactly 1.0.
    """
    tiny = torch.finfo(torch.float32).tiny
    subnormal = (ref != 0) & (ref.abs() < tiny)
    return subnormal & (out == 0) & (torch.signbit(out) == torch.signbit(ref))


def _assert_bitwise(name: str, x: torch.Tensor, out: torch.Tensor, ref: torch.Tensor) -> int:
    """The same bits as HF's ops wherever they give a number, and NaN wherever they give NaN;
    returns how many fp32 subnormals the kernel flushed to zero (`_flushed`), the one exception.

    There is no band to derive: kernel and reference run the same fp32 operations in the same
    order -- `1 / (1 + expf(-x))` rounded to bf16 for beta; the fp32 sum `a + dt_bias`, the
    threshold test and `log1pf(expf(.))` for softplus, then one fp32 product with `-exp(A_log)`
    for g -- so any other difference is a different computation. NaN payloads are left out of
    the claim; NaN-ness is not.
    """
    assert out.shape == ref.shape and out.dtype == ref.dtype, (name, out.shape, ref.shape)
    ref_nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(out), ref_nan), f"{name}: NaN where HF has a number, or reverse"
    flushed = _flushed(out, ref) if out.dtype == torch.float32 else torch.zeros_like(ref_nan)
    if bool(flushed.any()):
        assert torch.equal(torch.exp(out[flushed]), torch.exp(ref[flushed])), name
    differ = (_bits(out) != _bits(ref)) & ~ref_nan & ~flushed
    count = int(differ.sum())
    if count:
        at = differ.reshape(-1).nonzero()[:4].squeeze(-1)
        pairs = zip(
            x.reshape(-1)[at % x.numel()].tolist(),
            out.reshape(-1)[at].tolist(),
            ref.reshape(-1)[at].tolist(),
        )
        raise AssertionError(
            f"{name}: {count} of {out.numel()} outputs differ from HF's; (x, out, ref): "
            f"{list(pairs)}"
        )
    return int(flushed.sum())


def _parameters(spec, g):
    """`-exp(A_log)` and `dt_bias` at the checkpoint's ranges: A_log in [-5.1, 2.9] (A from 0.006
    to 18.3) and dt_bias in [-7.0, -1.9] over its 32 GDP layers."""
    heads = spec["heads"]
    a_log = torch.empty(heads, device="cuda").uniform_(-5.1, 2.9, generator=g)
    neg_a = -a_log.exp()
    dt_bias = torch.empty(heads, device="cuda").uniform_(-7.0, -1.9, generator=g).to(BF16)
    return neg_a, dt_bias


def _build(spec, seed):
    """b and a as the in_proj output's column slices: row stride 18528, no copy."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    proj = torch.randn(spec["tokens"], spec["proj_width"], generator=g, device="cuda")
    proj = proj.mul_(INPUT_SCALE).to(BF16)
    heads, householder = spec["heads"], spec["householder"]
    b = proj[:, spec["b_offset"] : spec["b_offset"] + householder * heads]
    a = proj[:, spec["a_offset"] : spec["a_offset"] + heads]
    return (b, a, *_parameters(spec, g))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = [t.clone() for t in args]
    beta, g = op(*args)
    assert all(torch.equal(t, b) for t, b in zip(args, before)), "the op must not write its inputs"
    ref_beta, ref_g = op.reference(*args)
    _assert_bitwise("beta", args[0], beta, ref_beta)
    _assert_bitwise("g", args[1], g, ref_g)


def _every_bf16() -> torch.Tensor:
    """All 65536 bf16 bit patterns: zeros, subnormals, normals, both infinities, 254 NaNs."""
    return (
        torch.arange(-(2**15), 2**15, dtype=torch.int32, device="cuda").to(torch.int16).view(BF16)
    )


def _rows(values: torch.Tensor, width: int, rows: int) -> torch.Tensor:
    """`values` laid out as `rows` rows of `width`, repeated from the start to fill them."""
    repeats = -(-rows * width // values.numel())
    return values.repeat(repeats)[: rows * width].view(rows, width)


def test_every_bf16_input(certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """The whole input domain of each gate, through the op and through HF's ops.

    beta depends on b alone, so every b value is driven once. g depends on a and on the head's
    dt_bias and -exp(A_log); every a value is driven once, at one of 24 heads with parameters
    spread over the checkpoint's ranges, so a + dt_bias crosses the softplus threshold, its log1p
    stretch and its underflow -- where HF's decay goes subnormal and the kernel's flushes to zero
    (`_flushed`).

    The controls show that bitwise agreement is a claim these inputs can refute: sigmoid kept in
    fp32 to the end (no bf16 rounding of beta) and softplus' threshold dropped (log1p(exp(.))
    everywhere) each move outputs that the kernel matches.
    """
    certifying(op)
    heads, householder = 24, 3
    every = _every_bf16()
    rows = -(-every.numel() // heads)  # every a value once; b repeats from the start
    b = _rows(every, householder * heads, rows)
    a = _rows(every, heads, rows)
    a_log = torch.linspace(-5.1, 2.9, heads, device="cuda")
    neg_a = -a_log.exp()
    dt_bias = torch.linspace(-7.0, -1.9, heads, device="cuda").to(BF16)
    beta, g = op(b, a, neg_a, dt_bias)
    ref_beta, ref_g = op.reference(b, a, neg_a, dt_bias)
    _assert_bitwise("beta", b, beta, ref_beta)
    # Measured: 24 of the 65544 decays are subnormal in HF and flushed here.
    assert _assert_bitwise("g", a, g, ref_g) <= 64

    finite_beta = torch.isfinite(ref_beta)
    unrounded = b.float().sigmoid().reshape(ref_beta.shape)
    beta_moves = int(((unrounded != ref_beta.float()) & finite_beta).sum())
    no_threshold = neg_a * torch.log1p(torch.exp(a.float() + dt_bias.float()))
    finite_g = torch.isfinite(ref_g)
    g_moves = int(((_bits(no_threshold) != _bits(ref_g)) & finite_g).sum())
    assert beta_moves >= 1000, f"beta's bf16 rounding changed only {beta_moves} outputs"
    assert g_moves >= 100, f"softplus' threshold changed only {g_moves} outputs"


def test_rows_do_not_depend_on_the_batch() -> None:
    """A token's gates come back as the same bits whatever batch it rides in.

    CUDA-graph padding and MTP verify change T for the same request, and the identity gates
    compare such runs bitwise. One program per token, this holds by construction; it is pinned.
    """
    spec = dict(op.CELLS[3].spec)
    b, a, neg_a, dt_bias = _build(spec, seed=5)
    full_beta, full_g = op(b, a, neg_a, dt_bias)
    for start, stop in ((0, 1), (37, 40), (1021, 1024)):
        beta, g = op(b[start:stop], a[start:stop], neg_a, dt_bias)
        assert torch.equal(beta, full_beta[start:stop]), (start, stop)
        assert torch.equal(g, full_g[start:stop]), (start, stop)


def test_a_captured_graph_replays_bitwise() -> None:
    """Captured once, replayed on new inputs, the same bits as eager and as HF's ops."""
    spec = dict(op.CELLS[0].spec, tokens=64)
    static = _build(spec, seed=6)
    op(*static)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_beta, static_g = op(*static)
    for seed in (7, 8):
        fresh = _build(spec, seed=seed)
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        graph.replay()
        beta, g = op(*fresh)
        assert torch.equal(static_beta, beta) and torch.equal(static_g, g), f"replay (seed {seed})"
        ref_beta, ref_g = op.reference(*fresh)
        _assert_bitwise("beta", fresh[0], static_beta, ref_beta)
        _assert_bitwise("g", fresh[1], static_g, ref_g)


def test_zero_tokens_launch_nothing() -> None:
    spec = dict(op.CELLS[0].spec, tokens=0)
    beta, g = op(*_build(spec, seed=9))
    assert beta.shape == (0, 3, 24) and beta.dtype == BF16
    assert g.shape == (0, 24) and g.dtype == torch.float32


def _hf_gates(b, a, neg_a, dt_bias):
    """HF's expressions on the columns, written out as `Nemotron4GDPMixer.forward` has them."""
    beta = b.sigmoid().reshape(b.shape[0], 3, 24)
    return beta, neg_a * F.softplus(a.float() + dt_bias.float())


def test_guard_refuses_a_transposed_gate_slice() -> None:
    """A column-strided b is read as if dense: the op answers for other elements."""
    b, a, neg_a, dt_bias = _build(dict(op.CELLS[2].spec), seed=10)
    b_t = b.t().contiguous().t()  # same values, column-major
    beta, _ = op(b_t, a, neg_a, dt_bias)
    ref_beta, _ = _hf_gates(b, a, neg_a, dt_bias)
    assert not torch.equal(beta, ref_beta), "the hazard is gone: a strided b reads right"
    with pytest.raises(AssertionError, match="unit-stride rows"), validating(op):
        op(b_t, a, neg_a, dt_bias)


def test_guard_refuses_b_that_is_not_whole_groups() -> None:
    """A b one head short is read with the head count a gives, and the last group runs short."""
    b, a, neg_a, dt_bias = _build(dict(op.CELLS[2].spec), seed=11)
    with pytest.raises(AssertionError, match=r"b must be \[T, n_h \* H\]"), validating(op):
        op(b[:, :71], a, neg_a, dt_bias)


def test_guard_refuses_per_head_vectors_of_another_length() -> None:
    b, a, neg_a, dt_bias = _build(dict(op.CELLS[2].spec), seed=12)
    with pytest.raises(AssertionError, match="neg_a and dt_bias must be"), validating(op):
        op(b, a, neg_a[:23], dt_bias)


def test_guard_refuses_gates_in_another_dtype() -> None:
    """fp32 columns load and convert silently, and beta comes back fp32, not HF's bf16 rounding."""
    b, a, neg_a, dt_bias = _build(dict(op.CELLS[2].spec), seed=13)
    beta, _ = op(b.float(), a.float(), neg_a, dt_bias)
    assert beta.dtype == torch.float32, "the hazard is gone: fp32 b gives a bf16 beta"
    with pytest.raises(AssertionError, match="b and a must be bfloat16"), validating(op):
        op(b.float(), a.float(), neg_a, dt_bias)


def test_guard_refuses_a_rounded_decay_factor() -> None:
    b, a, neg_a, dt_bias = _build(dict(op.CELLS[2].spec), seed=14)
    with pytest.raises(AssertionError, match="neg_a must be float32"), validating(op):
        op(b, a, neg_a.to(BF16), dt_bias)


def test_guard_refuses_a_log_passed_as_the_decay_factor() -> None:
    """A_log in place of -exp(A_log) gives a positive log decay wherever A_log > 0."""
    b, a, neg_a, dt_bias = _build(dict(op.CELLS[2].spec), seed=15)
    a_log = torch.log(-neg_a)
    _, g = op(b, a, a_log, dt_bias)
    assert bool((g > 0).any()), "the hazard is gone: A_log gives no growing decay"
    with pytest.raises(AssertionError, match="pass -exp\\(A_log\\), not A_log"), validating(op):
        op(b, a, a_log, dt_bias)
