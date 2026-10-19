# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as op

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, in the output's own precision.

    Kernel and reference consume bit-identical operands, and the product of
    two bf16 values is exact in fp32, so the two differ only in how they sum
    over K: cuBLAS's tensor-core fp32 accumulation against torch's IEEE fp32
    one. Over random operands a few outputs land near zero, where any
    reordering reads as a large relative difference, so both gates are scaled
    by the row rather than by the element.

    bf16 out: 4 ulp (2^-8) element-wise against the row scale, 2 of relative
    RMS. The two fp32 sums agree to ~1e-6 relative, so what remains visible
    is a rounding flip -- one bf16 spacing, at most 2 of these ulps at the top
    binade of a row. Measured worst across the cells, a tp4 rank's slices
    included: 1.98 element (that flip), 0.068 RMS. A wrong product misses by
    orders.

    fp32 out (the routers and the lm_head): 1024 ulp (2^-23) element-wise,
    512 of relative RMS. Here the fp32 sums are the output. Measured worst:
    62 element and 28 RMS at k 3072 and 7168, with the kernel the less
    accurate side (31-42 ulp from a float64 product, the reference within
    21). A result rounded to bf16 and widened scores at least 16484 and 13819,
    so this gate refuses it by 16x; a band in bf16 ulp would let it through.
    """
    if out.dtype is torch.float32:
        assert_within_ulp(out, ref, element_ulp=1024.0, rms_ulp=512.0, ulp=ULP[torch.float32])
    else:
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(spec, seed):
    """Materialize one cell.

    `mat_b` is built as [N, K] and transposed, because that is the one thing a
    spec cannot say: the op reads it column-major, and a stored weight's `.t()`
    already is that, which is why no transpose kernel runs in a target.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, dt = spec["m"], spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(dt).t()
    out_dtype = spec["out_dtype"]
    bias = None
    if spec["bias"]:
        bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt)
    return mat_a, mat_b, bias, out_dtype


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    _compare(out, op.reference(mat_a, mat_b, bias, out_dtype))


def test_fp32_out_refuses_a_bf16_rounded_result() -> None:
    """fp32 out has to carry fp32's precision, and the gate has to see it.

    A product rounded to bf16 and widened is an fp32 tensor of the right shape
    whose every value sits within half a bf16 ulp of the truth: inside any band
    scored in bf16 ulp. The router and lm_head cells ask for fp32 out because
    the checkpoint computes them in fp32, so the gate is in fp32 ulp, and this
    is the control that it separates the two.
    """
    cell = next(c for c in op.CELLS if c.why.startswith("nemotron4 MoE router m=1024"))
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(mat_a, mat_b, bias, out_dtype)
    _compare(op(mat_a, mat_b, bias, out_dtype), ref)
    with pytest.raises(AssertionError, match="not close"):
        _compare(ref.to(torch.bfloat16).float(), ref)


def _ok(dtype=torch.bfloat16):
    a = torch.randn(8, 64, dtype=dtype, device="cuda")
    b = torch.randn(32, 64, dtype=dtype, device="cuda").t()
    return a, b


def test_guard_refuses_a_row_major_mat_b() -> None:
    """Read column-major regardless; a row-major b is a different product."""
    a, _ = _ok()
    b = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="column-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_strided_mat_a() -> None:
    a = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")[:, :64]
    _, b = _ok()
    with pytest.raises(AssertionError, match="row-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_bias_in_the_wrong_dtype() -> None:
    """Accepted by the op, and the result is silently wrong."""
    a, b = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="bias dtype"):
        with validating(op):
            op(a, b, bias)


def test_guard_refuses_a_bias_on_fp32_inputs() -> None:
    """The op takes it and never adds it."""
    a, b = _ok(torch.float32)
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="silently ignored"):
        with validating(op):
            op(a, b, bias)
