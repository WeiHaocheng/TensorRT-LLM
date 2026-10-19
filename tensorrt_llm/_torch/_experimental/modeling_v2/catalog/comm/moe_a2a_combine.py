# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum each token's per-rank expert partials back on its own rank, via the trtllm one-sided MoE all-to-all."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper
from .moe_a2a_dispatch import _SIZES

# fp32: each rank's window partial crosses the wire unrounded and the token's sum is rounded once by the
# caller (nemotron4's dep4 design, HF's single fp32 sum over a token's ten experts). bf16: the partials are
# rounded per rank first, as TensorRT-LLM's MoE backends send them.
_DTYPES = ((torch.float32, "fp32 partials"), (torch.bfloat16, "bf16 partials"))


class MoeA2aCombine(OpWrapper):
    """`out[t] = sum over the ranks that received token t of their partial for it`, on token t's own rank.

    The inverse of `moe_a2a_dispatch`, which must have run on the same workspace just before: it left there
    which ranks hold which of this rank's tokens. `payload` is this rank's `[ep_size,
    runtime_max_tokens_per_rank, hidden]` partials in the dispatch's receive layout (row `[s, j]` is the
    partial for the token received in that row), fp32 or bf16; the output is `[local_num_tokens, hidden]` in
    `payload`'s dtype. The kernel gathers a token's partials from at most `ep_size` ranks and adds them in fp32
    -- a pairwise tree, `(p0 + p1) + (p2 + p3)`, the ranks taken in the order the token's top-k list first names
    them -- then rounds once to the output dtype.

    `combine_payload_offset` is what `moe_a2a_dispatch` returned; `payload_in_workspace` says `payload` is the
    workspace's own combine region (`moe_a2a_get_combine_payload_tensor`), which skips the op's staging copy.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"per-rank tokens {list(sizes)}, {dname}: {why}",
            spec=dict(sizes=sizes, latent=768, top_k=10, num_experts=512, dtype=dtype),
        )
        for sizes, why in _SIZES
        for dtype, dname in _DTYPES
    )

    note = """
    fp32 payloads need this branch's `moe_a2a_combine` kernel: upstream instantiates it for fp16 and bf16 only
    ("Unsupported dtype for moe_a2a_combine"); the op already mapped fp32.

    bf16 partials round each rank's partial before the sum. Measured against the fp64 sum on nemotron4's dep4
    shapes, that leaves 40% of outputs off the correctly rounded value and a relative RMS error of 2.4e-3, where
    fp32 partials give 1.7e-3, the bf16 output's own rounding.

    `use_low_precision` (FP8 on the wire) is not a cell: its relative RMS error measured 2.7e-2.

    `local_num_tokens` must be the token count this rank dispatched; the op cannot check it, and a different
    count returns rows that are not this rank's tokens. Every rank must call with the same
    `runtime_max_tokens_per_rank` as its dispatch.
    """

    def __call__(
        self,
        payload: torch.Tensor,
        local_num_tokens: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        use_low_precision: bool = False,
        use_cft_counted_writes: bool = False,
        enable_rank_mask: bool = False,
        active_rank_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.moe_a2a_combine(
            payload=payload,
            local_num_tokens=local_num_tokens,
            workspace=workspace,
            metainfo=metainfo,
            runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            top_k=top_k,
            combine_payload_offset=combine_payload_offset,
            payload_in_workspace=payload_in_workspace,
            use_low_precision=use_low_precision,
            use_cft_counted_writes=use_cft_counted_writes,
            enable_rank_mask=enable_rank_mask,
            active_rank_mask=active_rank_mask,
        )

    def reference(
        self,
        per_rank_partials: Sequence[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """The collective exception to mirroring `__call__`: this rank's output is a function of every rank's
        partials. `per_rank_partials[r]` is `[local_num_tokens, hidden]`, rank `r`'s partial for each of this
        rank's tokens (zero where the token routes no expert to `r`), in `dtype` as the ranks send them.

        The sum in fp64, rounded once to `dtype`.
        """
        acc = per_rank_partials[0].double()
        for p in per_rank_partials[1:]:
            acc = acc + p.double()
        return acc.to(dtype)


moe_a2a_combine = MoeA2aCombine()
