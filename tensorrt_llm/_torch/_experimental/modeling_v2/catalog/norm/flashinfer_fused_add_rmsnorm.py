# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place fused residual add + RMS normalization via the flashinfer kernel.

Everything a target author needs in order to decide whether this op covers
their case is below, and all of it is checked: `reference` is what CI compares
the kernel against, and `CELLS` is what CI drives.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class FlashinferFusedAddRmsnorm(OpWrapper):
    """`residual += x; x = rmsnorm(residual) * weight`, in place, returns
    `(x, residual)` -- the same two tensors, mutated and handed back so a call
    site can show what it touched without allocating.

    Both tensors are written. The residual output is the fp32 sum rounded once
    to its own dtype; the normalization reads that sum in fp32, not the rounded
    value, which is what `reference` mirrors.

    Fusion boundary: add, normalization and weight scaling. No gemma `(1 + w)`
    form, no quantization.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The hidden sizes the targets normalize over, at a decode shape and a
    # prefill shape. Every target calls this op only on a residual stream, so
    # there is no 3-D case: unlike plain rmsnorm this one never runs on a
    # per-head view.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="decode-shaped, deepseek-r1 hidden",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        # nemotron4's MTP layer runs its block on an ordinary bf16 residual: the
        # attention output is added before `pre_moe_layernorm` and the MoE output
        # before `final_layernorm`, both at hidden 3072 (the main stack's wide
        # residual streams are not this op). The block sees one token per request
        # at depth 2, three per request at depth 1, and every context token of a
        # prefill chunk.
        Cell(
            why="nemotron4 3072, decode: one MTP draft token",
            spec=dict(shape=(1, 3072), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="nemotron4 3072 x1024: MTP depth 2 at the largest CUDA-graph batch",
            spec=dict(shape=(1024, 3072), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="nemotron4 3072 x3072: MTP depth 1, 1024 requests x 3 verified tokens",
            spec=dict(shape=(3072, 3072), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="nemotron4 3072 x8192: MTP layer over an 8192-token prefill chunk",
            spec=dict(shape=(8192, 3072), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="nemotron4 3072 x16384: MTP layer over the 16384-token prefill budget",
            spec=dict(shape=(16384, 3072), dtype=torch.bfloat16, eps=1e-5),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    PDL is controlled by `TRTLLM_ENABLE_PDL` (default on) inside the trtllm
    custom op; it changes scheduling, not results, and is not a cell axis.

    Every target passes contiguous tensors here -- the residual stream is a
    freshly allocated buffer -- so the strided-row path this kernel also has
    is not certified by this entry.

    The residual it writes is a bf16 add of the two inputs, bit for bit what
    an HF block's `hidden_states + update` writes. The normalized output is
    not HF's: HF normalizes that rounded sum and rounds twice (`weight *
    normed.to(input_dtype)`, as `Nemotron4RMSNorm` and `LlamaRMSNorm` do),
    this op normalizes the fp32 sum and rounds once. The two land at most two
    ulps apart and differ on about a third of the outputs
    (`test_hf_order_is_two_ulps_away`); a target that has to match HF's order
    bit for bit needs a kernel of its own.

    A row's results do not depend on how many rows share the call
    (`test_a_row_does_not_depend_on_its_batch`), so CUDA-graph padding and
    speculative verify leave every real row's bits alone.
    """

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes `x` and `residual` in place and returns both, in that order
        -- the op destroys each argument's prior contents, so a caller that
        wants them has to have kept its own copy beforehand."""
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(x, residual, weight, eps)
        return x, residual

    def reference(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two values the op writes, as (x_out, residual_out).

        Returns rather than mutates so a cell can drive both sides from the
        same inputs; the kernel's in-place write is what `__call__` does.
        """
        h = x.float() + residual.float()
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype), h.to(residual.dtype)

    def is_valid(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> None:
        # Only the cases the op takes and answers wrongly. Rank and width
        # disagreements it rejects itself, loudly.
        assert x.dtype is not torch.float8_e4m3fn, (
            "float8_e4m3fn is accepted by the kernel's dtype table and returns "
            "a plausible fp8 result; dequantize before normalizing"
        )
        assert x.stride(-1) == 1 and residual.stride(-1) == 1, (
            f"the normalized dim must be contiguous; got strides "
            f"{x.stride(-1)} and {residual.stride(-1)}"
        )
        # Aliasing is the trap specific to the in-place form: the kernel writes
        # both tensors, so one buffer passed twice is overwritten mid-kernel
        # and the result is neither value.
        assert x.data_ptr() != residual.data_ptr(), (
            "x and residual must be distinct buffers; the op writes both"
        )


flashinfer_fused_add_rmsnorm = FlashinferFusedAddRmsnorm()
