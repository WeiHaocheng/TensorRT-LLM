# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: nemotron4-nano-preview's dense GEMMs, one row per distinct shape:
#: (call sites, k, n, out_dtype). All are bias-free on bf16 operands. The router
#: and the lm_head ask for fp32 out because the checkpoint's own code runs both
#: as fp32 GEMMs on bf16 values; their products are exact in fp32, so fp32
#: accumulation with fp32 out is the same math up to summation order. Attention's
#: o_proj and the shared expert's down projection are both 4096 -> 3072.
_NEMOTRON4_GEMMS: tuple[tuple[str, int, int, torch.dtype | None], ...] = (
    ("GDP in_proj", 3072, 18528, None),
    ("GDP out_proj", 1536, 3072, None),
    ("attention qkv", 3072, 4608, None),
    ("o_proj and shared down", 4096, 3072, None),
    ("dense MLP up", 3072, 12288, None),
    ("dense MLP down", 12288, 3072, None),
    ("MoE router", 3072, 512, torch.float32),
    ("fc1_latent_proj", 3072, 768, None),
    ("fc2_latent_proj", 768, 3072, None),
    ("shared expert up", 3072, 4096, None),
    ("MTP eh_proj", 6144, 3072, None),
)

#: The row counts every GEMM above runs at. cuBLASLt picks its kernel per
#: (m, n, k), so each row count is a claim of its own rather than a size.
_NEMOTRON4_LAYER_ROWS = (
    (1, "decode, one token"),
    (1024, "decode, the largest CUDA-graph batch"),
    (8192, "prefill, an 8192-token chunk"),
)

#: The lm_head only sees each request's last token plus its draft tokens.
_NEMOTRON4_LM_HEAD = ("lm_head", 3072, 262272, torch.float32)
_NEMOTRON4_LM_HEAD_ROWS = (
    (1, "one request"),
    (1024, "1024 requests"),
    (3072, "1024 requests verifying two drafts each"),
)

#: The tp4 target's per-rank slices of those GEMMs. Column-parallel projections
#: keep k and take a quarter of n, row-parallel ones a quarter of k; GDP in_proj
#: keeps its six heads of every segment ([z | V | K | Q | b | a] at 6 of 24
#: heads). The single KV head is replicated, so the fused qkv is 4 query heads
#: plus the whole k and v: 4 * 256 + 256 + 256. The dense MLP's up and down
#: projections are both 3072 -> 3072 per rank, and o_proj and the shared
#: expert's down projection both 1024 -> 3072. The router, both latent
#: projections and MTP eh_proj are replicated, at the shapes above.
_NEMOTRON4_TP4_GEMMS: tuple[tuple[str, int, int, torch.dtype | None], ...] = (
    ("GDP in_proj", 3072, 4632, None),
    ("GDP out_proj", 384, 3072, None),
    ("attention qkv", 3072, 1536, None),
    ("o_proj and shared down", 1024, 3072, None),
    ("MLP up and down", 3072, 3072, None),
    ("shared expert up", 3072, 1024, None),
)

#: tp4's lm_head is vocab-parallel: each rank computes its quarter of the
#: vocabulary's fp32 logits, at the row counts tp1's lm_head runs plus 8192
#: rows, above any batch the target verifies.
_NEMOTRON4_TP4_LM_HEAD = ("lm_head", 3072, 65568, torch.float32)
_NEMOTRON4_TP4_LM_HEAD_ROWS = (
    *_NEMOTRON4_LM_HEAD_ROWS,
    (8192, "8192 rows, above any verify batch"),
)


def _nemotron4_cells(
    target: str,
    gemms: tuple[tuple[str, int, int, torch.dtype | None], ...],
    lm_head: tuple[str, int, int, torch.dtype | None],
    lm_head_rows: tuple[tuple[int, str], ...],
) -> tuple[Cell, ...]:
    """One cell per (GEMM, row count) that a nemotron4 target's forward runs."""
    plan = [(gemm, _NEMOTRON4_LAYER_ROWS) for gemm in gemms]
    plan.append((lm_head, lm_head_rows))
    return tuple(
        Cell(
            why=f"{target} {site} m={m}: {k} -> {n}{', fp32 out' if out_dtype else ''}, {rows}",
            spec=dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=False, out_dtype=out_dtype),
        )
        for (site, k, n, out_dtype), row_counts in plan
        for m, rows in row_counts
    )


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Every target uses this for every dense projection. The first four cells
    # are the argument shapes that differ in kind rather than in size: fused bias
    # (gpt-oss carries one on qkv, o and the router), no bias (deepseek's
    # projections), and an out_dtype that is not the input's (deepseek's router
    # asks for fp32 logits). The nemotron4 cells after them differ in size: each
    # of its distinct GEMMs at each row count its forward runs, for the tp1
    # target and then for a tp4 rank's slices.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gpt-oss qkv with fused bias: hidden 2880 -> (64 + 8 + 8) * 64",
            spec=dict(m=16, k=2880, n=5120, dtype=torch.bfloat16, bias=True, out_dtype=None),
        ),
        Cell(
            why="deepseek q_a, no bias: hidden 7168 -> q_lora 1536",
            spec=dict(m=16, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek router: bf16 in, fp32 logits out, 256 experts",
            spec=dict(
                m=16, k=7168, n=256, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32
            ),
        ),
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        *_nemotron4_cells(
            "nemotron4", _NEMOTRON4_GEMMS, _NEMOTRON4_LM_HEAD, _NEMOTRON4_LM_HEAD_ROWS
        ),
        *_nemotron4_cells(
            "nemotron4 tp4",
            _NEMOTRON4_TP4_GEMMS,
            _NEMOTRON4_TP4_LM_HEAD,
            _NEMOTRON4_TP4_LM_HEAD_ROWS,
        ),
    )

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. Both shipped targets leave them at their defaults, so neither is
    a cell axis and neither is certified here.
    """

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cublas_mm(mat_a, mat_b, bias, out_dtype, output_buffer_kind, group)

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
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
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> None:
        # The kernel reads mat_a as dense row-major and mat_b as dense
        # column-major; other layouts produce silently wrong results.
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        if bias is not None:
            out_dt = out_dtype if out_dtype is not None else mat_a.dtype
            # A bias in the wrong dtype or shape is accepted by the op and
            # produces silently wrong results; with fp32 inputs the bias is
            # accepted but silently ignored.
            assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
            assert bias.dtype == out_dt, "bias dtype must equal the output dtype"
            assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
                "bias must be a contiguous [N] tensor"
            )


cublas_mm = CublasMm()
