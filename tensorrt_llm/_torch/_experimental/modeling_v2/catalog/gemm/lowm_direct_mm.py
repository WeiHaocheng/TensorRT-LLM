# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense bf16 GEMM for a few rows: FlashInfer's CuTe-DSL direct kernel, as TensorRT-LLM's low-M path runs it."""

from __future__ import annotations

import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
    DirectTactic,
    default_tactic,
    run_direct_dense,
)

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: nemotron4-nano-preview's bf16-out GEMMs, at the row counts where this kernel beat cuBLAS by at least 7% in
#: survey-gemm's sweep (sm_103, CUDA-graph replay, weights streamed from HBM): (call sites, k, n, m) ->
#: (block_size, outputs_per_block, rows_per_block), the fastest of the tactics TensorRT-LLM's low-M dispatcher
#: races (`modules/low_m_gemm.py`, `_DirectGemmRunner`). Row counts are the decode batches and the 3-token MTP verify
#: batches a CUDA graph captures up to 32 rows. tp4 rows are a tensor-parallel rank's slices.
_NEMOTRON4: tuple[tuple[str, int, int, int, tuple[int, int, int]], ...] = (
    ("fc2_latent_proj", 768, 3072, 1, (96, 2, 1)),
    ("fc2_latent_proj", 768, 3072, 2, (32, 2, 2)),
    ("fc2_latent_proj", 768, 3072, 3, (32, 2, 3)),
    ("fc2_latent_proj", 768, 3072, 4, (32, 2, 4)),
    ("fc2_latent_proj", 768, 3072, 6, (32, 2, 6)),
    ("tp4 o_proj and shared down", 1024, 3072, 1, (128, 2, 1)),
    ("tp4 o_proj and shared down", 1024, 3072, 2, (64, 2, 2)),
    ("tp4 o_proj and shared down", 1024, 3072, 3, (32, 1, 3)),
    ("tp4 o_proj and shared down", 1024, 3072, 4, (32, 1, 4)),
    ("tp4 o_proj and shared down", 1024, 3072, 6, (32, 2, 6)),
    ("GDP out_proj", 1536, 3072, 1, (192, 2, 1)),
    ("GDP out_proj", 1536, 3072, 2, (96, 1, 2)),
    ("GDP out_proj", 1536, 3072, 3, (96, 2, 3)),
    ("GDP out_proj", 1536, 3072, 4, (64, 2, 4)),
    ("GDP out_proj", 1536, 3072, 6, (32, 2, 6)),
    ("fc1_latent_proj", 3072, 768, 1, (192, 1, 1)),
    ("fc1_latent_proj", 3072, 768, 2, (192, 1, 2)),
    ("fc1_latent_proj", 3072, 768, 3, (128, 1, 3)),
    ("fc1_latent_proj", 3072, 768, 4, (128, 1, 4)),
    ("fc1_latent_proj", 3072, 768, 6, (128, 2, 6)),
    ("fc1_latent_proj", 3072, 768, 8, (128, 2, 8)),
    ("fc1_latent_proj", 3072, 768, 12, (32, 2, 4)),
    ("tp4 shared expert up", 3072, 1024, 1, (192, 1, 1)),
    ("tp4 shared expert up", 3072, 1024, 2, (192, 1, 2)),
    ("tp4 shared expert up", 3072, 1024, 3, (128, 1, 3)),
    ("tp4 shared expert up", 3072, 1024, 4, (128, 1, 4)),
    ("tp4 shared expert up", 3072, 1024, 6, (32, 1, 6)),
    ("tp4 shared expert up", 3072, 1024, 8, (96, 2, 8)),
    ("tp4 attention qkv", 3072, 1536, 1, (192, 1, 1)),
    ("tp4 attention qkv", 3072, 1536, 2, (192, 1, 2)),
    ("tp4 attention qkv", 3072, 1536, 3, (128, 2, 3)),
    ("tp4 attention qkv", 3072, 1536, 4, (64, 2, 4)),
    ("tp4 attention qkv", 3072, 1536, 6, (64, 2, 6)),
    ("tp4 MLP up and down", 3072, 3072, 1, (128, 1, 1)),
    ("tp4 MLP up and down", 3072, 3072, 2, (128, 2, 2)),
    ("tp4 MLP up and down", 3072, 3072, 3, (32, 2, 3)),
    ("tp4 MLP up and down", 3072, 3072, 4, (32, 2, 4)),
    ("shared expert up", 3072, 4096, 1, (128, 1, 1)),
    ("shared expert up", 3072, 4096, 2, (128, 2, 2)),
    ("shared expert up", 3072, 4096, 3, (128, 1, 3)),
    ("attention qkv", 3072, 4608, 1, (192, 2, 1)),
    ("attention qkv", 3072, 4608, 2, (128, 2, 2)),
    ("attention qkv", 3072, 4608, 3, (32, 4, 3)),
    ("tp4 GDP in_proj", 3072, 4632, 1, (128, 1, 1)),
    ("tp4 GDP in_proj", 3072, 4632, 2, (128, 2, 2)),
    ("tp4 GDP in_proj", 3072, 4632, 3, (32, 4, 3)),
    ("dense MLP up", 3072, 12288, 1, (96, 2, 1)),
    ("dense MLP up", 3072, 12288, 2, (128, 2, 2)),
    ("GDP in_proj", 3072, 18528, 1, (128, 2, 1)),
    ("o_proj and shared down", 4096, 3072, 1, (256, 2, 1)),
    ("o_proj and shared down", 4096, 3072, 2, (128, 2, 2)),
    ("o_proj and shared down", 4096, 3072, 3, (32, 2, 3)),
    ("o_proj and shared down", 4096, 3072, 4, (32, 2, 4)),
    ("MTP eh_proj", 6144, 3072, 1, (256, 2, 1)),
    ("MTP eh_proj", 6144, 3072, 2, (256, 2, 2)),
    ("MTP eh_proj", 6144, 3072, 3, (32, 2, 3)),
    ("MTP eh_proj", 6144, 3072, 4, (32, 2, 4)),
    ("dense MLP down", 12288, 3072, 1, (384, 2, 1)),
    ("dense MLP down", 12288, 3072, 2, (64, 4, 2)),
    ("dense MLP down", 12288, 3072, 3, (64, 4, 3)),
)

#: (k, n, m) -> tactic: the rows above, which `__call__` uses. Any other (k, n, m) takes FlashInfer's default
#: tactic, which this entry does not certify.
NEMOTRON4_TACTICS: dict[tuple[int, int, int], tuple[int, int, int]] = {
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


class LowmDirectMm(OpWrapper):
    """`mat_a @ mat_b` for 1 to 32 rows, bf16 out, fp32 accumulation, one launch.

    Each CTA prefetches whole K-extent weight rows into registers, reuses each value across `rows_per_block` rows
    of `mat_a`, accumulates with scalar fp32 FMAs, and reduces across warps through shared memory: no tensor cores,
    no split of K across CTAs. `mat_b` is read column-major, which is what a stored weight's `.t()` already is.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per table row: each (k, n, m) compiles its own kernel (the kernel is specialized on all three),
    # so each is a configuration of its own rather than a size within one.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"nemotron4 {site} m={m}: {k} -> {n}, {_rows(m)}",
            spec=dict(m=m, k=k, n=n),
        )
        for site, k, n, m, _ in _NEMOTRON4
    )

    note = """
    The kernel is compiled per (m, n, k, tactic, pdl) on first use, a fraction of a second each, so a target warms
    every row count it captures before capturing a CUDA graph (the engine's runner does: it runs each graph size
    eagerly first). An eager call costs ~60 us of host time, against ~25 us for `cublas_mm`: the entry is for
    CUDA-graph-captured decode and MTP verify steps.

    bf16 out only. The router asks for fp32 logits and the lm_head for fp32 logits, so neither is a cell.

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
        tactic = DirectTactic(*tactic) if tactic is not None else default_tactic(m, n, k)
        out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
        return run_direct_dense(mat_a, mat_b, out, pdl, tactic)

    def reference(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = False
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off, rounded once to bf16; `pdl` is ignored."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        return ref.to(mat_a.dtype)


lowm_direct_mm = LowmDirectMm()
