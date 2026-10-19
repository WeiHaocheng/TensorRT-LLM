# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense bf16 GEMM for a few rows: FlashInfer's CuTe-DSL split-K kernel, as TensorRT-LLM's low-M path runs it."""

from __future__ import annotations

import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    SplitKTactic,
    default_tactic,
    run_splitk_dense,
)

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: nemotron4-nano-preview's bf16-out GEMMs, at the row counts where this kernel beat cuBLAS by at least 7% in
#: survey-gemm's sweep (sm_103, CUDA-graph replay, weights streamed from HBM) and beat `lowm_direct_mm`:
#: (call sites, k, n, m) -> (mma_m, mma_n, split_k, ab_stages), the fastest of the tactics TensorRT-LLM's low-M
#: dispatcher races (`modules/low_m_gemm.py`, `_SplitKGemmRunner`). Row counts are the decode batches and the
#: 3-token MTP verify batches a CUDA graph captures up to 32 rows. tp4 rows are a tensor-parallel rank's slices.
_NEMOTRON4: tuple[tuple[str, int, int, int, tuple[int, int, int, int]], ...] = (
    ("GDP out_proj", 1536, 3072, 8, (64, 8, 2, 10)),
    ("fc1_latent_proj", 3072, 768, 16, (64, 8, 4, 10)),
    ("fc1_latent_proj", 3072, 768, 24, (64, 8, 4, 10)),
    ("fc1_latent_proj", 3072, 768, 32, (64, 16, 4, 8)),
    ("tp4 shared expert up", 3072, 1024, 12, (64, 8, 4, 10)),
    ("tp4 shared expert up", 3072, 1024, 16, (64, 8, 4, 10)),
    ("tp4 shared expert up", 3072, 1024, 24, (64, 16, 3, 9)),
    ("tp4 shared expert up", 3072, 1024, 32, (64, 16, 3, 9)),
    ("tp4 attention qkv", 3072, 1536, 8, (64, 8, 4, 10)),
    ("tp4 MLP up and down", 3072, 3072, 6, (128, 8, 4, 5)),
    ("tp4 MLP up and down", 3072, 3072, 8, (128, 8, 4, 5)),
    ("tp4 MLP up and down", 3072, 3072, 12, (64, 16, 2, 9)),
    ("tp4 MLP up and down", 3072, 3072, 16, (64, 16, 2, 9)),
    ("shared expert up", 3072, 4096, 4, (128, 8, 3, 5)),
    ("shared expert up", 3072, 4096, 6, (128, 8, 3, 5)),
    ("shared expert up", 3072, 4096, 8, (128, 8, 3, 5)),
    ("shared expert up", 3072, 4096, 12, (64, 16, 2, 9)),
    ("shared expert up", 3072, 4096, 16, (64, 16, 2, 9)),
    ("attention qkv", 3072, 4608, 4, (128, 8, 3, 5)),
    ("attention qkv", 3072, 4608, 6, (128, 8, 3, 5)),
    ("attention qkv", 3072, 4608, 8, (64, 8, 2, 10)),
    ("attention qkv", 3072, 4608, 12, (64, 16, 2, 9)),
    ("attention qkv", 3072, 4608, 16, (64, 16, 2, 9)),
    ("tp4 GDP in_proj", 3072, 4632, 4, (128, 8, 3, 5)),
    ("tp4 GDP in_proj", 3072, 4632, 6, (128, 8, 3, 5)),
    ("tp4 GDP in_proj", 3072, 4632, 8, (64, 8, 2, 10)),
    ("tp4 GDP in_proj", 3072, 4632, 12, (64, 16, 2, 9)),
    ("tp4 GDP in_proj", 3072, 4632, 16, (64, 16, 2, 9)),
    ("dense MLP up", 3072, 12288, 3, (128, 8, 1, 6)),
    ("dense MLP up", 3072, 12288, 4, (128, 8, 1, 6)),
    ("dense MLP up", 3072, 12288, 6, (128, 8, 1, 6)),
    ("dense MLP up", 3072, 12288, 8, (128, 8, 1, 6)),
    ("dense MLP up", 3072, 12288, 12, (128, 16, 1, 6)),
    ("dense MLP up", 3072, 12288, 16, (128, 16, 1, 6)),
    ("o_proj and shared down", 4096, 3072, 6, (64, 8, 2, 10)),
    ("o_proj and shared down", 4096, 3072, 8, (64, 8, 2, 10)),
    ("o_proj and shared down", 4096, 3072, 12, (64, 16, 2, 9)),
    ("o_proj and shared down", 4096, 3072, 16, (64, 16, 2, 9)),
    ("MTP eh_proj", 6144, 3072, 6, (128, 8, 4, 6)),
    ("MTP eh_proj", 6144, 3072, 8, (128, 8, 4, 6)),
    ("MTP eh_proj", 6144, 3072, 12, (64, 16, 2, 11)),
    ("MTP eh_proj", 6144, 3072, 16, (64, 16, 2, 11)),
)

#: (k, n, m) -> tactic: the rows above, which `__call__` uses. Any other (k, n, m) takes FlashInfer's default
#: tactic, which this entry does not certify.
NEMOTRON4_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (k, n, m): tactic for _, k, n, m, tactic in _NEMOTRON4
}

#: The CUDA-graph decode batch sizes up to 32 rows; an MTP verify step runs 3 rows per request.
_GRAPH_BATCHES = (1, 2, 4, 8, 16, 24, 32)


def _rows(m: int) -> str:
    """Which steps run a GEMM with `m` rows."""
    steps = []
    if m in _GRAPH_BATCHES:
        steps.append(f"decode of {m}")
    if m % 3 == 0 and m // 3 in _GRAPH_BATCHES:
        steps.append(f"MTP verify of {m // 3}")
    return " or ".join(steps)


class LowmSplitkMm(OpWrapper):
    """`mat_a @ mat_b` for 1 to 32 rows, bf16 out, fp32 accumulation, one launch.

    Tensor-core (tcgen05) tiles with the problem swapped so the weight's N fills the MMA's M; a cluster of
    `split_k` CTAs each accumulates one K slice in fp32, the peers hand their partials to rank 0 through
    distributed shared memory, and rank 0 adds them, rounds and stores once. `mat_b` is read column-major, which
    is what a stored weight's `.t()` already is.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per table row. The kernel is compiled per tactic, not per shape, but each row is a (shape, rows)
    # point where this kernel and its tactic were chosen, and what CI must keep proving.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"nemotron4 {site} m={m}: {k} -> {n}, {_rows(m)}",
            spec=dict(m=m, k=k, n=n),
        )
        for site, k, n, m, _ in _NEMOTRON4
    )

    note = """
    The kernel is compiled per tactic on first use, a fraction of a second each, so a target warms every row count
    it captures before capturing a CUDA graph (the engine's runner does). An eager call costs ~80-100 us of host
    time, against ~25 us for `cublas_mm`: the entry is for CUDA-graph-captured decode and MTP verify steps.

    bf16 out only, and no bias (the op's fused bias is not a cell axis: no nemotron4 GEMM has one). The router
    and the lm_head ask for fp32 logits, so neither is a cell.

    `pdl` (programmatic dependent launch) changes scheduling, not results; the cells run it off. TensorRT-LLM's
    low-M dispatcher passes `TRTLLM_ENABLE_PDL` (default on).

    Rows of the table are the measured winners at the CUDA-graph decode batches 1, 2, 4, 8, 16, 24, 32 and the
    MTP verify batches 3, 6, 12, 24; a row count outside the table takes FlashInfer's `default_tactic` and is
    uncertified. survey-gemm's result records the times and the rival kernels.
    """

    def __call__(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = False) -> torch.Tensor:
        m, k = mat_a.shape
        n = mat_b.shape[1]
        tactic = NEMOTRON4_TACTICS.get((k, n, m))
        tactic = SplitKTactic(*tactic) if tactic is not None else default_tactic(m, n, k)
        out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
        return run_splitk_dense(mat_a, mat_b, None, out, pdl, tactic)

    def reference(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = False
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off, rounded once to bf16; `pdl` is ignored."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        return ref.to(mat_a.dtype)


lowm_splitk_mm = LowmSplitkMm()
