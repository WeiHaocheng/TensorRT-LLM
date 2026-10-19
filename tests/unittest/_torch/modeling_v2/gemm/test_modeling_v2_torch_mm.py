# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the torch_mm catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.torch_mm import torch_mm as op

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "torch_mm requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, in the output's own precision: cublas_mm's bands.

    The product of two bf16 values is exact in fp32, so the cuBLAS kernel and torch's IEEE fp32 product differ
    only in summation order. bf16 out: what remains visible after both round is a rounding flip, at most 2 ulp
    (2^-8) of the row scale, so 4 element-wise and 2 of relative RMS. fp32 out (the router): the sums are the
    output, gated at 1024 and 512 fp32 ulp (2^-23), which refuses a bf16-rounded result by 16x. Over survey-gemm's
    334 (shape, rows) points this op's error against an fp32 product equals cublas_mm's, point for point.
    """
    if out.dtype is torch.float32:
        assert_within_ulp(out, ref, element_ulp=1024.0, rms_ulp=512.0, ulp=ULP[torch.float32])
    else:
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(spec, seed):
    """Materialize one cell: `mat_b` is built as [N, K] and transposed, a stored weight's `.t()`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n = spec["m"], spec["k"], spec["n"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(torch.bfloat16).t()
    return mat_a, mat_b, spec["out_dtype"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    mat_a, mat_b, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    _compare(out, op.reference(mat_a, mat_b, out_dtype))
