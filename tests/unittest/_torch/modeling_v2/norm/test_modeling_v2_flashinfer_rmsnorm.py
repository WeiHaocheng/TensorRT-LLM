# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file owns only what a cell cannot carry -- turning a spec into real tensors and
stating how close the kernel has to come -- plus the inputs the guard exists to refuse.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"

#: Bound on the fraction of outputs that may differ from the reference at all; see `_compare`.
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
    """The final rounding may differ, and nothing else.

    Kernel and reference compute the same fp32 expression from the same bf16 inputs and round
    once to bf16; only the order of the row's sum of squares and the `1 / sqrt` differ, by a
    few fp32 ulps. So an output can only land on the neighbouring bf16 value, and only when its
    fp32 value sits on a rounding boundary. Measured over every cell at five seeds each: at
    most 1.3e-5 of the outputs differ in the cells of 1024 rows and more, and at most one
    output in the smaller ones, each by exactly one ulp; no output is more than 0.5001 ulp
    from the fp64 value.

    Two gates:
      * every output within one bf16 ulp of its own value, counted on the bit patterns. Each
        rival misses by far more: eps x10, /10 or 0 by 66 ulps or more on every cell of 1024
        rows and more, because `_build` puts rows on both sides of eps; the gemma `(1 + w)`
        weight by 32000 or more.
      * at most `MAX_MISMATCH_FRACTION` of the outputs not bit-identical, and never fewer than
        one allowed: the smallest cell has 768 outputs, where a single boundary flip is
        1.3e-3 of them. The two-rounding order of the HF norms stays within one ulp of the op
        but changes 19 to 30% of the outputs (`test_two_rounding_order_is_one_ulp_away`):
        this gate is what tells it apart from the op.
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

    Every row gets its own magnitude, log-uniform over [1e-3, 3]. The smallest rows' mean
    square sits below eps, where eps sets the output; at unit scale eps moves nothing a bf16
    output can show, and a wrong eps would pass. Real inputs reach that range: nemotron4's
    embedding rows, which its MTP layer's `enorm` normalizes, have a mean square of about 3e-3,
    and some of 2.5e-5.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    rows, hidden = math.prod(shape[:-1]), shape[-1]
    log_scale = torch.empty(rows, 1, device="cuda").uniform_(
        math.log(1e-3), math.log(3.0), generator=g
    )
    x = torch.randn(rows, hidden, generator=g, device="cuda") * log_scale.exp()
    weight = torch.randn(hidden, generator=g, device="cuda").to(dtype)
    return x.reshape(shape).to(dtype), weight, spec["eps"]


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, weight, eps = _build(cell.spec, seed=_seed(cell))
    out = op(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    _compare(out, op.reference(x, weight, eps))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_two_rounding_order_is_one_ulp_away(cell) -> None:
    """The rival `reference` rejects: round the normalized value, then scale by the weight.

    That is how the HF RMSNorms are written (`Nemotron4RMSNorm`, `LlamaRMSNorm`):
    `weight * normed.to(input_dtype)`, two roundings where this op makes one. The extra
    rounding moves the product by less than one ulp of its own value, so the two orders land
    on the same or neighbouring bf16 values -- but they disagree on about a quarter of the
    outputs (19 to 30% measured, 26% in every cell of 1024 rows and more), which is what keeps
    `_compare` from mistaking one for the other, and why a target that needs that order bit for
    bit needs a kernel of its own.
    """
    x, weight, eps = _build(cell.spec, seed=_seed(cell))
    out = op(x, weight, eps)
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    two_roundings = weight * normed.to(x.dtype)

    distance = _ulp_distance(out, two_roundings)
    assert int(distance.max().item()) <= 1, "the two orders are more than one ulp apart"
    mismatch = (distance != 0).float().mean().item()
    assert mismatch >= 100 * MAX_MISMATCH_FRACTION, (
        f"only {mismatch:.2e} of the outputs differ: `_compare` no longer tells the orders apart"
    )


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """The hazard `is_valid` exists for: accepted, and silently plausible.

    Driving the op directly shows why the guard is not cosmetic -- it returns a
    tensor rather than raising, so a caller who forgot to dequantize gets a
    number back.
    """
    x = torch.randn(4, 2880, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(2880, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_rmsnorm(x, weight, 1e-6)
    assert unguarded.dtype is torch.float8_e4m3fn, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, weight, 1e-6)


def test_guard_refuses_a_strided_normalized_dim() -> None:
    """`x.stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 2880, 2, dtype=torch.bfloat16, device="cuda")
    x = buf[..., 0]
    assert x.stride(-1) != 1
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, weight, 1e-6)
