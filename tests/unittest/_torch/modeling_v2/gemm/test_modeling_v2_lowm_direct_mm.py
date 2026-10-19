# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the lowm_direct_mm catalog entry, and for `lowm_plan.lowm_entry`, which picks between it,
lowm_splitk_mm and cublas_mm."""

import zlib

import pytest
import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, validate_tactic

import tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_direct_mm as entry_module
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_direct_mm import (
    NEMOTRON4_TACTICS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_direct_mm import (
    lowm_direct_mm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_plan import lowm_entry
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_splitk_mm import (
    NEMOTRON4_TACTICS as SPLITK_TACTICS,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "lowm_direct_mm requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, in bf16.

    Kernel and reference consume bit-identical operands, and the product of two bf16 values is exact in fp32,
    so the two differ only in how they sum over K: the kernel's per-thread fp32 FMA chains, warp shuffles and a
    cross-warp sum against torch's IEEE fp32 product. Both round once to bf16, so what remains visible is a
    rounding flip -- one bf16 spacing, at most 2 ulp (2^-8) of the row scale at the top binade of a row.

    4 ulp element-wise against the row scale and 2 of relative RMS: cublas_mm's band, for the same reason.
    Measured worst over every cell and four extra seeds per cell: 0.58 element and 0.042 RMS (survey-gemm's
    `measure_entries.py`); a wrong product misses by orders.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(spec, seed):
    """Materialize one cell: `mat_b` is built as [N, K] and transposed, a stored weight's `.t()`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n = spec["m"], spec["k"], spec["n"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(torch.bfloat16).t()
    return mat_a, mat_b


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    mat_a, mat_b = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is torch.bfloat16
    _compare(out, op.reference(mat_a, mat_b))


def test_every_row_is_a_tactic_the_kernel_takes() -> None:
    """A row the kernel would refuse fails at the first call of a served engine; refuse it here instead."""
    for (k, n, m), tactic in NEMOTRON4_TACTICS.items():
        validate_tactic(DirectTactic(*tactic), m, n, k)


def test_the_call_uses_the_row_for_its_shape(monkeypatch) -> None:
    """The table is the certified claim, so the tactic the kernel receives must be the row's."""
    seen = []
    real = entry_module.run_direct_dense

    def spy(a, b, out, pdl, tactic):
        seen.append(tactic)
        return real(a, b, out, pdl, tactic)

    monkeypatch.setattr(entry_module, "run_direct_dense", spy)
    for spec in (op.CELLS[0].spec, op.CELLS[-1].spec):
        mat_a, mat_b = _build(spec, seed=7)
        op(mat_a, mat_b)
        assert seen[-1] == DirectTactic(*NEMOTRON4_TACTICS[(spec["k"], spec["n"], spec["m"])])


def test_pdl_and_graph_replay_change_no_bits() -> None:
    """The entry is for CUDA-graph-captured steps, with or without PDL: the replayed product is the eager one."""
    spec = next(c.spec for c in op.CELLS if c.spec["m"] == 3)
    mat_a, mat_b = _build(spec, seed=11)
    eager = op(mat_a, mat_b)
    assert torch.equal(op(mat_a, mat_b, pdl=True), eager)
    static_a = torch.zeros_like(mat_a)
    op(static_a, mat_b)  # compiles before capture, as the engine's warm-up does
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        static_out = op(static_a, mat_b, pdl=True)
    static_a.copy_(mat_a)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, eager)


def test_the_plan_picks_one_entry_per_shape_and_rows() -> None:
    """lowm_entry: this entry's rows, lowm_splitk_mm's rows, and cublas_mm (None) everywhere else."""
    assert not set(NEMOTRON4_TACTICS) & set(SPLITK_TACTICS), "a (k, n, m) row in both low-M tables"
    for k, n, m in NEMOTRON4_TACTICS:
        assert lowm_entry(m, k, n) is op
    assert lowm_entry(64, 3072, 4608) is None, "64 rows is cublas_mm's"
    assert lowm_entry(1, 3072, 512) is None, "the router asks for fp32 logits"
