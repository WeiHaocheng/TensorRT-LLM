# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_fused_add_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file owns only what a cell cannot carry -- turning a spec into real tensors and
stating how close the kernel has to come -- plus the inputs the guard exists to refuse, which
are claims about inputs no cell describes.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_fused_add_rmsnorm requires a CUDA device"

#: Bound on the fraction of normalized outputs that may differ from the reference at all; see
#: `_compare`.
MAX_MISMATCH_FRACTION = 1e-3

#: Cells with enough rows to call on a part of them.
_MULTI_ROW = [c for c in op.CELLS if c.spec["shape"][0] > 3]


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise distance between two bf16 tensors in ulps, counted on the bit patterns."""

    def index(t: torch.Tensor) -> torch.Tensor:
        # Adjacent bf16 values map to adjacent integers, and +0 and -0 to the same one.
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (index(a) - index(b)).abs()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The normalized output's final rounding may differ, and nothing else.

    Kernel and reference normalize the same fp32 sum and round once to bf16; only the order of
    the row's sum of squares and the `1 / sqrt` differ, by a few fp32 ulps. So an output can
    only land on the neighbouring bf16 value, and only when its fp32 value sits on a rounding
    boundary. Measured over every cell at five seeds each: at most 1.1e-5 of the outputs differ
    in the cells of 1024 rows and more, and at most one output in the smaller ones, each by
    exactly one ulp.

    Two gates:
      * every output within one bf16 ulp of its own value, counted on the bit patterns. A wrong
        eps -- x10, /10 or 0 -- misses by 31 ulps or more on every cell of 1024 rows and more,
        because `_build` puts rows on both sides of eps.
      * at most `MAX_MISMATCH_FRACTION` of the outputs not bit-identical, and never fewer than
        one allowed, so that a single boundary flip passes in a cell of any size. The HF order
        -- normalize the rounded sum, round, scale, round -- changes 33 to 37% of the outputs
        (`test_hf_order_is_two_ulps_away`): this gate is what tells it apart from the op.
    """
    assert out.dtype == ref.dtype and out.shape == ref.shape, (out.dtype, out.shape, ref.shape)
    distance = _ulp_distance(out, ref)
    worst = int(distance.max().item())
    assert worst <= 1, f"an output is {worst} bf16 ulps from the reference"
    mismatches, total = int((distance != 0).sum().item()), distance.numel()
    allowed = max(1, int(MAX_MISMATCH_FRACTION * total))
    assert mismatches <= allowed, (
        f"{mismatches} of {total} outputs differ from the reference at all; at most {allowed} may"
    )


def _build(spec, seed):
    """Materialize one cell.

    Every row of `x` and of `residual` gets its own magnitude, log-uniform over [1e-3, 3], drawn
    independently: some sums are dominated by one side, and the rows where both are small have a
    mean square below eps, where eps sets the output. At unit scale eps moves nothing a bf16
    output can show, and a wrong eps would pass.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    rows, hidden = math.prod(shape[:-1]), shape[-1]

    def rows_at_random_scales() -> torch.Tensor:
        log_scale = torch.empty(rows, 1, device="cuda").uniform_(
            math.log(1e-3), math.log(3.0), generator=g
        )
        t = torch.randn(rows, hidden, generator=g, device="cuda") * log_scale.exp()
        return t.reshape(shape).to(dtype)

    x = rows_at_random_scales()
    residual = rows_at_random_scales()
    weight = torch.randn(hidden, generator=g, device="cuda").to(dtype)
    return x, residual, weight, spec["eps"]


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every certified configuration, kernel against reference.

    The residual output is one fp32 add of two bf16 values rounded once, which both sides
    compute exactly alike, so it is gated bit for bit.
    """
    certifying(op)
    x, residual, weight, eps = _build(cell.spec, seed=_seed(cell))
    ref_x, ref_residual = op.reference(x, residual, weight, eps)
    out_x, out_residual = op(x, residual, weight, eps)
    assert out_x is x and out_residual is residual, "the op writes in place"
    assert torch.equal(residual, ref_residual), "the residual sum is not bit-identical"
    _compare(x, ref_x)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_hf_order_is_two_ulps_away(cell) -> None:
    """The rival `reference` rejects: the order an HF residual block computes in.

    HF writes the residual as a bf16 add, `s = x + residual`, normalizes that rounded sum and
    rounds the normalized value before scaling by the weight (`Nemotron4RMSNorm`,
    `LlamaRMSNorm`): `weight * (s * rsqrt(mean(s^2) + eps)).to(input_dtype)`. The residual it
    writes is this op's bit for bit. The normalized output is not: the sum's rounding and the
    extra one each move the product by less than one ulp of its own value, so the two orders
    land at most two ulps apart -- and disagree on about a third of the outputs (33 to 37%
    measured, 34% in every cell of 1024 rows and more), which is what keeps `_compare` from
    mistaking one for the other, and why a target that needs that order bit for bit needs a
    kernel of its own.
    """
    x, residual, weight, eps = _build(cell.spec, seed=_seed(cell))
    rounded_sum = x + residual
    sf = rounded_sum.float()
    hf_order = weight * (sf * torch.rsqrt(sf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)

    out_x, out_residual = op(x, residual, weight, eps)
    assert torch.equal(out_residual, rounded_sum), "the residual is not HF's bf16 add"
    distance = _ulp_distance(out_x, hf_order)
    assert int(distance.max().item()) <= 2, "the two orders are more than two ulps apart"
    mismatch = (distance != 0).float().mean().item()
    assert mismatch >= 100 * MAX_MISMATCH_FRACTION, (
        f"only {mismatch:.2e} of the outputs differ: `_compare` no longer tells the orders apart"
    )


@pytest.mark.parametrize("cell", _MULTI_ROW, ids=[c.why[:40] for c in _MULTI_ROW])
def test_a_row_does_not_depend_on_its_batch(cell) -> None:
    """A row's bits -- both outputs -- do not move with the number of rows beside it.

    CUDA-graph padding and speculative verify change how many rows share a call, and a
    target's greedy output stays token-identical across them only if each row comes out the
    same either way. Each row is normalized by its own sum, so it does: the first row, the
    first three and all but the last, called alone, are bit-identical to the same rows of the
    whole call.
    """
    x, residual, weight, eps = _build(cell.spec, seed=_seed(cell))
    whole_x, whole_residual = op(x.clone(), residual.clone(), weight, eps)
    for n in (1, 3, x.shape[0] - 1):
        part_x, part_residual = op(x[:n].clone(), residual[:n].clone(), weight, eps)
        assert torch.equal(part_x, whole_x[:n]), f"rows [0, {n}) moved"
        assert torch.equal(part_residual, whole_residual[:n]), f"residual rows [0, {n}) moved"


def _one(**over):
    g = torch.Generator(device="cuda").manual_seed(7)
    kw = dict(rows=4, hidden=2880, dtype=torch.bfloat16)
    kw.update(over)
    shape = (kw["rows"], kw["hidden"])
    return (
        torch.randn(shape, generator=g, device="cuda").to(kw["dtype"]),
        torch.randn(shape, generator=g, device="cuda").to(kw["dtype"]),
        torch.randn(shape[-1], generator=g, device="cuda").to(kw["dtype"]),
    )


def test_guard_refuses_fp8() -> None:
    """Accepted, and silently plausible: the case a caller cannot detect."""
    x, residual, weight = _one(dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, residual, weight, 1e-6)


def test_guard_refuses_a_strided_normalized_dim() -> None:
    """`stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 2880, 2, dtype=torch.bfloat16, device="cuda")
    x, residual = buf[..., 0], buf[..., 1]
    assert x.stride(-1) != 1
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, residual, weight, 1e-6)


def test_guard_refuses_aliased_buffers() -> None:
    """The op writes both tensors; one buffer passed twice yields neither."""
    x, _, weight = _one()
    with pytest.raises(AssertionError, match="distinct buffers"):
        with validating(op):
            op(x, x, weight, 1e-6)


def test_the_guard_is_not_on_the_hot_path() -> None:
    """Outside `validating`, the wrapper goes straight to the kernel.

    Stated as a test because it is the property that lets the guard be as
    strict as it likes: a served engine never runs it.
    """
    x, residual, weight = _one(dtype=torch.float8_e4m3fn)
    # fp8 is what the guard refuses; unguarded, the kernel answers anyway.
    op(x, residual, weight, 1e-6)
