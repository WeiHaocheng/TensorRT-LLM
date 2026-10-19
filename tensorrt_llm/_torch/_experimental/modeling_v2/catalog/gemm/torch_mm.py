# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul through `torch.mm`: cuBLAS through ATen, the GEMM of TensorRT-LLM's default bf16 Linear."""

from __future__ import annotations

import torch

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: nemotron4-nano-preview's GEMMs in a tp4 rank's eager steps (prefill, mixed and the prompt's tail step), where
#: the step is host-bound: (call sites, k, n, out_dtype). Column-parallel projections keep k and take a quarter of
#: n, row-parallel ones a quarter of k; the router, both latent projections and MTP eh_proj are replicated.
_NEMOTRON4_TP4: tuple[tuple[str, int, int, torch.dtype | None], ...] = (
    ("GDP in_proj", 3072, 4632, None),
    ("GDP out_proj", 384, 3072, None),
    ("attention qkv", 3072, 1536, None),
    ("o_proj and shared down", 1024, 3072, None),
    ("MLP up and down", 3072, 3072, None),
    ("shared expert up", 3072, 1024, None),
    ("MoE router", 3072, 512, torch.float32),
    ("fc1_latent_proj", 3072, 768, None),
    ("fc2_latent_proj", 768, 3072, None),
    ("MTP eh_proj", 6144, 3072, None),
)

#: A prompt's tail step beside decode rows, an 8192-token prefill chunk, a 16384-token step.
_ROWS = ((64, "a tail step's rows"), (8192, "an 8192-token prefill"), (16384, "a 16384-token step"))


class TorchMm(OpWrapper):
    """`mat_a @ mat_b`, fp32-accumulated, one `torch.mm` call; fp32 out on request.

    Same cuBLAS GEMMs as `cublas_mm` at these shapes, issued through ATen: ~15 us of host time per eager call
    against `cublas_mm`'s ~25 us, which builds its cuBLASLt descriptors on every call. That matters only where a
    step is host-bound, as a tp4 rank's eager steps are.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"nemotron4 tp4 {site} m={m}: {k} -> {n}{', fp32 out' if out_dtype else ''}, {rows}",
            spec=dict(m=m, k=k, n=n, out_dtype=out_dtype),
        )
        for site, k, n, out_dtype in _NEMOTRON4_TP4
        for m, rows in _ROWS
    )

    note = """
    No bias argument: torch.mm has none, and no nemotron4 GEMM carries one. A CUDA-graph-captured decode step
    gains nothing here (host time is not in a graph replay); there the faster GEMMs are `cublas_mm` and the low-M
    entries (`lowm_plan`).
    """

    def __call__(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # torch.mm's out_dtype overload takes a dtype, not None.
        if out_dtype is None:
            return torch.mm(mat_a, mat_b)
        return torch.mm(mat_a, mat_b, out_dtype=out_dtype)

    def reference(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)


torch_mm = TorchMm()
