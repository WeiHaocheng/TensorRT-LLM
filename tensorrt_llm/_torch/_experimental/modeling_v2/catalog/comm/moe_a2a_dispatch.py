# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send each token to the ranks that own its experts, via the trtllm one-sided MoE all-to-all."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

# A dep4 rank's tokens per MoE block: decode batches (every rank padded to the same count under CUDA graphs),
# MTP verify batches (3 rows a request), prefill steps, and ragged eager steps where ranks hold different counts.
_SIZES = (
    ((1, 1, 1, 1), "decode, one token per rank: the latency floor every block pays"),
    ((4, 4, 4, 4), "decode at C = 16"),
    ((16, 16, 16, 16), "decode at C = 64"),
    ((64, 64, 64, 64), "decode at C = 256"),
    ((256, 256, 256, 256), "decode at C = 1024"),
    ((1024, 1024, 1024, 1024), "decode at C = 4096, the largest CUDA-graph batch a rank captures"),
    ((3, 3, 3, 3), "MTP verify, one request per rank (1 + 2 drafts)"),
    ((48, 48, 48, 48), "MTP verify at C = 64"),
    ((768, 768, 768, 768), "MTP verify at C = 1024"),
    ((3072, 3072, 3072, 3072), "MTP verify at C = 4096"),
    ((8192, 8192, 8192, 8192), "one 8192-token prompt on every rank"),
    ((16384, 16384, 16384, 16384), "a 16384-token step on every rank"),
    (
        (1, 1, 1, 8192),
        "one rank prefills a prompt while the others run their one-token dummy request",
    ),
    ((9088, 896, 1, 8193), "a mixed eager step: prefill + decode, decode, idle, prefill"),
)

# The geometry of nemotron4's latent MoE: a token's latent, its ten expert ids and their routing weights.
_LATENT, _TOP_K, _EXPERTS = 768, 10, 512


class MoeA2aDispatch(OpWrapper):
    """Each rank's tokens, sent once to every rank that owns at least one of their `top_k` experts.

    `input_payloads` are per-token rows -- nemotron4 sends its latent (bf16 [T, 768]), its expert ids
    (int32 [T, 10]) and its routing weights (fp32 [T, 10]) -- and come back as one receive buffer per payload,
    `[ep_size, runtime_max_tokens_per_rank, ...]`, views into `workspace`. Rows `[s, :]` hold the tokens source
    rank `s` routed here, compacted: which row a token lands in is the kernel's choice, so a caller finds a
    token by its contents (the routing weights a row carries), never by position. Every received row is a
    bit-exact copy of the sent one. Rows no token was written to get expert ids `invalid_token_expert_id` in
    payload `expert_id_payload_index`, so an experts call skips them (`latent_moe_experts` treats any id
    outside its window as absent).

    `workspace` and `metainfo` are the group's all-to-all workspace: `NVLinkOneSided(mapping, num_experts,
    top_k, max_num_tokens_per_rank, hidden_size=..., dtype=...)`, built once at target init, outside any
    graph capture; its `workspace` and `moe_a2a_metainfo`. Every rank calls with the same
    `runtime_max_tokens_per_rank`: the most tokens any rank holds this step.

    Returns the receive buffers and `combine_payload_offset`, the offset `moe_a2a_combine` takes.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"per-rank tokens {list(sizes)}: {why}",
            spec=dict(sizes=sizes, latent=_LATENT, top_k=_TOP_K, num_experts=_EXPERTS),
        )
        for sizes, why in _SIZES
    )

    note = """
    The receive layout pads every source rank to `runtime_max_tokens_per_rank`. A step whose ranks hold very
    different token counts (one prompt on one rank) therefore hands the experts call ep_size times the busiest
    rank's rows, mostly sanitized: measured, the window call then costs 805 us instead of 590 us per block at
    [1, 1, 1, 8192], which outweighs the all-to-all's gain over the all-gather. Balanced steps (every CUDA-graph
    step, which pads every rank to one count) lose nothing.

    The expert-id fix-up runs as its own op (`moe_a2a_sanitize_expert_ids`) unless the dispatch uses CFT
    counted writes (`use_cft_counted_writes`, driver 615+), which fuse it into the dispatch. Certified on the
    fence path, the one this pool's driver runs.

    No `is_valid`: the op refuses loudly a rank holding more tokens than `runtime_max_tokens_per_rank`, a payload
    whose rows are not one per token, and a workspace of another layout. What it cannot see is the other
    ranks: every rank must pass the same `runtime_max_tokens_per_rank` and the same payload list, and nothing
    on one rank can check that.

    Not cells: EPLB statistics (`eplb_local_stats`), rank masks (`enable_rank_mask`), quantized payloads, and
    world sizes other than 4.
    """

    def __call__(
        self,
        token_selected_experts: torch.Tensor,
        input_payloads: List[torch.Tensor],
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        num_experts: int,
        eplb_local_stats: Optional[torch.Tensor] = None,
        use_cft_counted_writes: bool = False,
        expert_id_payload_index: Optional[int] = None,
        invalid_token_expert_id: Optional[int] = None,
        enable_rank_mask: bool = False,
        active_rank_mask: Optional[torch.Tensor] = None,
    ) -> tuple[List[torch.Tensor], int]:
        recv, combine_payload_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts=token_selected_experts,
            input_payloads=input_payloads,
            workspace=workspace,
            metainfo=metainfo,
            runtime_max_tokens_per_rank=runtime_max_tokens_per_rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            top_k=top_k,
            num_experts=num_experts,
            eplb_local_stats=eplb_local_stats,
            use_cft_counted_writes=use_cft_counted_writes,
            expert_id_payload_index=expert_id_payload_index,
            invalid_token_expert_id=invalid_token_expert_id,
            enable_rank_mask=enable_rank_mask,
            active_rank_mask=active_rank_mask,
        )
        if expert_id_payload_index is not None and not use_cft_counted_writes:
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                expert_ids=recv[expert_id_payload_index],
                workspace=workspace,
                metainfo=metainfo,
                ep_rank=ep_rank,
                invalid_expert_id=invalid_token_expert_id,
            )
        return recv, int(combine_payload_offset)

    def reference(
        self,
        per_rank_experts: Sequence[torch.Tensor],
        per_rank_payloads: Sequence[Sequence[torch.Tensor]],
        ep_rank: int,
        ep_size: int,
        num_experts: int,
    ) -> List[List[torch.Tensor]]:
        """The collective exception to mirroring `__call__`: what rank `ep_rank` receives is a function of every
        rank's tokens, so this takes them in rank order (`per_rank_experts[s]`, `per_rank_payloads[s]`).

        Returns, per source rank `s` and per payload, the rows of `s`'s tokens that route at least one expert to
        `ep_rank` (`experts // (num_experts // ep_size) == ep_rank`), in token order. The op's receive rows
        `[s, :count_s]` hold the same rows in an order of its choosing.
        """
        per_rank = num_experts // ep_size
        out = []
        for experts, payloads in zip(per_rank_experts, per_rank_payloads):
            routed = (experts.long() // per_rank == ep_rank).any(dim=1)
            out.append([p[routed] for p in payloads])
        return out


moe_a2a_dispatch = MoeA2aDispatch()
