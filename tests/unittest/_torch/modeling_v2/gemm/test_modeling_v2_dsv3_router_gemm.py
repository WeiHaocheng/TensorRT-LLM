# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dsv3_router_gemm catalog entry."""

import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.dsv3_router_gemm import (
    dsv3_router_gemm as op,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "dsv3_router_gemm requires a CUDA device"

_KERNEL = "router_gemm_kernel"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Router logits in fp32: the sum over K in another order, gated in fp32 ulp.

    The product of two bf16 values is exact in fp32, so kernel and reference differ only in summation order:
    per-thread fp32 sums over 8-element strides, warp shuffles and a cross-warp sum, against torch's IEEE fp32
    product. 1024 ulp (2^-23) element-wise against the row scale and 512 of relative RMS, cublas_mm's fp32 band:
    a result rounded to bf16 and widened scores above 16000 there, so the band refuses it by 16x. Measured worst
    over every cell and four extra seeds per cell: 3.6 element and 1.3 RMS, with the kernel 1.4 and 0.8 from a
    float64 product, so element-wise the kernel is the more accurate side (survey-gemm's `measure_entries.py`).
    """
    assert_within_ulp(out, ref, element_ulp=1024.0, rms_ulp=512.0, ulp=ULP[torch.float32])


def _build(spec, seed):
    """Materialize one cell: `mat_b` is built as [N, K] and transposed, a stored weight's `.t()`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n = spec["m"], spec["k"], spec["n"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(torch.bfloat16).t()
    return mat_a, mat_b


def _kernels(fn) -> list[str]:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    mat_a, mat_b = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, None, torch.float32)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is torch.float32
    _compare(out, op.reference(mat_a, mat_b, None, torch.float32))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_the_cells_run_the_kernel_not_the_fallback(cell) -> None:
    """A build without the (512, 3072) instantiation answers every cell correctly through cuBLAS; only the
    kernel's name tells the two apart."""
    mat_a, mat_b = _build(cell.spec, seed=3)
    op(mat_a, mat_b, None, torch.float32)  # warm
    names = _kernels(lambda: op(mat_a, mat_b, None, torch.float32))
    assert any(_KERNEL in name for name in names), f"no {_KERNEL} launched: {names}"


def test_more_than_8_tokens_take_the_cublas_fallback() -> None:
    """At (512, 3072) the kernel stops at 8 tokens; 12 is cuBLAS's, and still the right product."""
    spec = dict(op.CELLS[0].spec, m=12)
    mat_a, mat_b = _build(spec, seed=5)
    out = op(mat_a, mat_b, None, torch.float32)
    _compare(out, op.reference(mat_a, mat_b, None, torch.float32))
    names = _kernels(lambda: op(mat_a, mat_b, None, torch.float32))
    assert not any(_KERNEL in name for name in names), f"{_KERNEL} ran at 12 tokens: {names}"


def test_guard_refuses_a_row_major_mat_a_with_a_longer_row() -> None:
    """The op checks only the unit stride, then reads row i at i * K: a padded row reads the wrong elements."""
    spec = dict(op.CELLS[0].spec, m=4)
    _, mat_b = _build(spec, seed=9)
    wide = torch.randn(4, 3072 + 64, device="cuda").to(torch.bfloat16)
    mat_a = wide[:, :3072]
    unguarded = op(mat_a, mat_b, None, torch.float32)
    ref = op.reference(mat_a.contiguous(), mat_b, None, torch.float32)
    assert not torch.allclose(unguarded[1:], ref[1:], rtol=1e-2, atol=1e-2), (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="dense row-major"):
        with validating(op):
            op(mat_a, mat_b, None, torch.float32)


def test_guard_refuses_a_column_major_mat_b_with_a_longer_column() -> None:
    """Column j is read at j * K: a weight view with a longer column reads the wrong elements."""
    spec = dict(op.CELLS[0].spec, m=4)
    mat_a, _ = _build(spec, seed=13)
    wide = torch.randn(512, 3072 + 64, device="cuda").to(torch.bfloat16)
    mat_b = wide[:, :3072].t()
    assert mat_b.stride(0) == 1
    unguarded = op(mat_a, mat_b, None, torch.float32)
    ref = op.reference(mat_a, mat_b.contiguous(), None, torch.float32)
    assert not torch.allclose(unguarded[:, 1:], ref[:, 1:], rtol=1e-2, atol=1e-2), (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="dense column-major"):
        with validating(op):
            op(mat_a, mat_b, None, torch.float32)
