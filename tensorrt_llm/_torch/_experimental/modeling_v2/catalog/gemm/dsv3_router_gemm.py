# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE router logits for a few tokens: fp32 out of bf16 operands, one CTA per expert (`dsv3_router_gemm_op`)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: The CUDA-graph decode batch sizes up to 8 rows; an MTP verify step runs 3 rows per request.
_GRAPH_BATCHES = (1, 2, 4, 8)


def _rows(m: int) -> str:
    """Which steps run the router with `m` rows."""
    steps = []
    if m in _GRAPH_BATCHES:
        steps.append(f"decode of {m}")
    if m % 3 == 0 and m // 3 in _GRAPH_BATCHES:
        steps.append(f"MTP verify of {m // 3}")
    return " or ".join(steps)


class Dsv3RouterGemm(OpWrapper):
    """`mat_a @ mat_b (+ bias)` with fp32 out: router logits as the checkpoint's own fp32 GEMM computes them.

    For 512 experts over hidden 3072 and 1 to 8 tokens the op launches its own kernel: one CTA per expert reads
    that expert's weight row once, 128 threads take 8 bf16 values per K step for every token, accumulate in fp32,
    and reduce with warp shuffles then across warps. Every other shape, more tokens, a bias, or a non-fp32 output
    goes to the op's cuBLAS fallback (`cublas_mm_out`). `mat_b` is read column-major, a stored weight's `.t()`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # nemotron4's router (3072 -> 512, fp32 logits) at every row count a decode or MTP verify CUDA graph reaches
    # up to 8: the range the op's own kernel serves at this shape. The kernel is instantiated per token count, so
    # each is a configuration of its own.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"nemotron4 MoE router m={m}: 3072 -> 512, fp32 logits, {_rows(m)}",
            spec=dict(m=m, k=3072, n=512),
        )
        for m in (1, 2, 3, 4, 6, 8)
    )

    note = """
    The op's kernel path takes exactly: bf16 operands, fp32 out, no bias, and (num_experts, hidden) = (256,
    7168 / 6144 / 4096) for 1 to 16 tokens or (512, 3072) for 1 to 8 tokens; anything else is its cuBLAS
    fallback, which this entry does not certify. At (512, 3072) cuBLAS is the faster GEMM from about 12 tokens,
    which is why the kernel stops at 8 there; a target calls `cublas_mm` for larger router batches.

    PDL is controlled by `TRTLLM_ENABLE_PDL` (default on) inside the op; it changes scheduling, not results.
    """

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        return torch.ops.trtllm.dsv3_router_gemm_op(mat_a, mat_b, bias, out_dtype)

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        if bias is not None:
            ref = ref + bias.float()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)

    def is_valid(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        # The op checks only the unit stride of each operand; the kernel then reads row i of mat_a at i * K and
        # column j of mat_b at j * K, so a row-major view with a longer row, or a column-major one with a longer
        # column, is read as if it were dense: plausible logits from the wrong elements.
        k = mat_a.shape[1]
        assert mat_a.stride() == (k, 1), (
            f"mat_a must be dense row-major [M, K]: stride (K, 1) = ({k}, 1), got {mat_a.stride()}"
        )
        assert mat_b.stride() == (1, k), (
            f"mat_b must be dense column-major [K, N] (e.g. weight.t()): stride (1, K) = (1, {k}), "
            f"got {mat_b.stride()}"
        )


dsv3_router_gemm = Dsv3RouterGemm()
