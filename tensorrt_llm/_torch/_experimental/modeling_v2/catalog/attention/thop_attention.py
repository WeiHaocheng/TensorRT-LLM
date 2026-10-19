# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused attention core with fully explicit state: paged KV-cache append + masked FMHA.

Wraps the pybind binding ``tensorrt_llm.bindings.internal.thop.attention``
(approved policy exception to the ``torch.ops.trtllm.*`` entry shape): the same
C++ attention op behind the TRTLLM backend, but every piece of batch state and
layer config arrives as an explicit argument -- no registered layers, no
thread-local metadata.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm.bindings.internal import thop

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: attention_input_type. 0 is the standard append-and-attend path; 1 and 2 are
#: MLA's context and generation phases.
STANDARD, MLA_CONTEXT, MLA_GENERATION = 0, 1, 2

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128


class ThopAttention(OpWrapper):
    """Append this step's K/V to the paged cache, then attend over it.

    The most load-bearing entry in the catalog and the only one wrapping a
    pybind binding rather than a registered torch op. Both shipped targets go
    through it, by different paths: gpt-oss on the standard path with attention
    sinks and an alternating sliding window, deepseek on MLA's two phases.

    Writes rows `[:num_tokens]` of `output` and returns `output` -- the same
    buffer, mutated and handed back so a call site can show what it touched
    without allocating.
    """

    ARCHS = frozenset({Arch.SM_103})

    # nemotron4-nano-preview: 16 query heads over one 256-wide KV head, no
    # positional encoding (the op's rope off), q_scaling 1 (scale 1/16), a bf16
    # pool of 32-token pages, histories up to its max_seq_len of 9344. Its
    # engine runs chunked prefill, which turns use_paged_context_fmha on for
    # every call, decode and verify included. An MTP verify step is plain
    # generation with three query tokens per request: sm_103 runs linear-tree
    # spec-dec with is_spec_decoding_enabled off, and the op splits the
    # generation tokens evenly among the generation rows.
    #
    # Each of its cells states its rows: `contexts` as (cached, new) per
    # context row, then `generations` rows of `gen_tokens` each, whose
    # histories the test draws below max_seq_len (one at the limit, a few short
    # enough that a verify row's own siblings carry visible weight).
    _NEMOTRON4 = dict(
        input_type=STANDARD,
        num_heads=16,
        num_kv_heads=1,
        head_size=256,
        v_head_dim=256,
        sinks=False,
        window=None,
        quant_mode=0,
        tokens_per_block=32,
        max_seq_len=9344,
    )

    # Its tp4 target makes the same calls on a rank's 4 query heads over the
    # replicated KV head: the same pool and batches, a quarter of the query
    # heads per KV head.
    _NEMOTRON4_TP4 = dict(_NEMOTRON4, num_heads=4)

    # The shapes the shipped targets reach, which are different code paths
    # rather than different sizes. gpt-oss alternates its two window classes
    # inside one engine, so both of its cells are live every step; deepseek runs
    # context and generation against the same pool; nemotron4 is single-KV-head
    # GQA, decoded at batches of 1 to 1024 and verified at 1 to 512, where the
    # kernel the op picks changes with batch size and history length -- and
    # with the query heads per KV head, so tp4's 4 have cells of their own. Its
    # steps reach 16384 tokens per rank: a fresh 8192-token prompt beside the
    # largest decode or verify batch, or two fresh prompts in one step.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gpt-oss sliding layer: GQA 64q/8kv d64, sinks, window 128",
            spec=dict(
                input_type=STANDARD,
                num_heads=64,
                num_kv_heads=8,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=128,
                quant_mode=0,
            ),
        ),
        Cell(
            why="gpt-oss full layer: the same heads with no window",
            spec=dict(
                input_type=STANDARD,
                num_heads=64,
                num_kv_heads=8,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=None,
                quant_mode=0,
            ),
        ),
        Cell(
            why="deepseek MLA context: explicit k/v, qk 576 -> v 128, fp8 pool",
            spec=dict(
                input_type=MLA_CONTEXT,
                num_heads=128,
                num_kv_heads=128,
                head_size=576,
                v_head_dim=128,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="deepseek MLA generation: one kv head, latent 576 -> 512, fp8 pool",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=128,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="nemotron4 prefill: one fresh 8192-token prompt",
            spec=dict(contexts=((0, 8192),), generations=0, gen_tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 chunk: 2048 new tokens on 6144 cached",
            spec=dict(contexts=((6144, 2048),), generations=0, gen_tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 mixed: a fresh and a chunked context beside 6 decodes",
            spec=dict(
                contexts=((0, 1000), (4096, 2048)), generations=6, gen_tokens=1, **_NEMOTRON4
            ),
        ),
        Cell(
            why="nemotron4 decode B=1: one request at the longest history",
            spec=dict(contexts=(), generations=1, gen_tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 decode B=64: histories spread up to max_seq_len",
            spec=dict(contexts=(), generations=64, gen_tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 decode B=1024: the largest batch a DP rank decodes",
            spec=dict(contexts=(), generations=1024, gen_tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 verify B=1: 3 query tokens per request",
            spec=dict(contexts=(), generations=1, gen_tokens=3, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 verify B=64: 3 query tokens per request",
            spec=dict(contexts=(), generations=64, gen_tokens=3, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 verify B=512: 3 query tokens per request",
            spec=dict(contexts=(), generations=512, gen_tokens=3, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 mixed verify: two contexts beside 6 three-token rows",
            spec=dict(
                contexts=((0, 1000), (4096, 2048)), generations=6, gen_tokens=3, **_NEMOTRON4
            ),
        ),
        Cell(
            why="nemotron4 prefill beside decodes: a fresh 8192-token prompt and B=1024",
            spec=dict(contexts=((0, 8192),), generations=1024, gen_tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 prefill beside verifies: a fresh 8192-token prompt and 512 x 3",
            spec=dict(contexts=((0, 8192),), generations=512, gen_tokens=3, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 16384-token step: two fresh 8192-token prompts",
            spec=dict(contexts=((0, 8192), (0, 8192)), generations=0, gen_tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 tp4 prefill: one fresh 8192-token prompt, 4 q heads",
            spec=dict(contexts=((0, 8192),), generations=0, gen_tokens=1, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 chunk: 2048 new tokens on 6144 cached",
            spec=dict(contexts=((6144, 2048),), generations=0, gen_tokens=1, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 mixed: a fresh and a chunked context beside 6 decodes",
            spec=dict(
                contexts=((0, 1000), (4096, 2048)), generations=6, gen_tokens=1, **_NEMOTRON4_TP4
            ),
        ),
        Cell(
            why="nemotron4 tp4 decode B=1: one request at the longest history",
            spec=dict(contexts=(), generations=1, gen_tokens=1, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 decode B=64: histories spread up to max_seq_len",
            spec=dict(contexts=(), generations=64, gen_tokens=1, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 decode B=1024: the largest CUDA-graph batch",
            spec=dict(contexts=(), generations=1024, gen_tokens=1, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 verify B=1: 3 query tokens per request",
            spec=dict(contexts=(), generations=1, gen_tokens=3, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 verify B=64: 3 query tokens per request",
            spec=dict(contexts=(), generations=64, gen_tokens=3, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 verify B=512: 3 query tokens per request",
            spec=dict(contexts=(), generations=512, gen_tokens=3, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 mixed verify: two contexts beside 6 three-token rows",
            spec=dict(
                contexts=((0, 1000), (4096, 2048)), generations=6, gen_tokens=3, **_NEMOTRON4_TP4
            ),
        ),
        Cell(
            why="nemotron4 tp4 prefill beside decodes: a fresh 8192-token prompt and B=1024",
            spec=dict(contexts=((0, 8192),), generations=1024, gen_tokens=1, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 prefill beside verifies: a fresh 8192-token prompt and 512 x 3",
            spec=dict(contexts=((0, 8192),), generations=512, gen_tokens=3, **_NEMOTRON4_TP4),
        ),
        Cell(
            why="nemotron4 tp4 16384-token step: two fresh 8192-token prompts",
            spec=dict(
                contexts=((0, 8192), (0, 8192)), generations=0, gen_tokens=1, **_NEMOTRON4_TP4
            ),
        ),
    )

    note = """
    KV addressing over one pool and two is certified. gpt-oss alternates
    sliding-window and full attention, and KVCacheManagerV2 gives each window
    class its own layer group, so its mapping carries two pool ids; the op reads
    the pool column itself and this entry passes the mapping through untouched.
    A third pool is outside what was measured.

    `reference` covers the masked-softmax half. The append half -- that this
    step's K/V land in the right pool slots -- is what the paged-cache entries
    beside this one certify, and what the matrix here checks by reading the pool
    back rather than by a reference.

    The softmax statistics the MLA chunked-prefill path writes are a second
    output; `softmax_stats` states them separately so `reference` stays one
    thing.

    nemotron4's decode and verify rows run on trtllm-gen generation kernels
    (XqaDispatcher), whose variant changes with batch size and history
    length: SwapsAb kernels, compiled through NVRTC on first use in every
    process (6-12 s each), and precompiled GroupedKeepsAb ones. A shape the
    engine has not warmed (`trtllm_gen_jit_warmup`) stalls its first step by
    that much. Its context rows run the paged-KV context kernel, which exists
    for head 256 at 32-token pages only. The packed-QKV context path runs the
    geometry too, but no nemotron4 call takes it and it is not certified.

    Speculative decoding is certified as the linear chain sm_103 runs: verify
    rows with is_spec_decoding_enabled off, causal within their group. Tree
    and dynamic-tree masks (is_spec_decoding_enabled on, the packed mask
    arguments) are not driven here.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        output_sf: Optional[torch.Tensor],
        workspace_: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_total_kv_lens: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        host_request_types: torch.Tensor,
        max_context_q_len_override: Optional[int],
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        cache_indirection: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        rotary_inv_freq: Optional[torch.Tensor],
        rotary_cos_sin: Optional[torch.Tensor],
        latent_cache: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        attention_sinks: Optional[torch.Tensor],
        is_fused_qkv: bool,
        update_kv_cache: bool,
        predicted_tokens_per_seq: int,
        local_layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: Optional[int],
        max_num_requests: int,
        max_context_length: int,
        max_seq_len: int,
        attention_window_size: int,
        beam_width: int,
        mask_type: int,
        quant_mode: int,
        q_scaling: float,
        position_embedding_type: int,
        rope_dim: int,
        rope_base: float,
        rope_scale_type: int,
        rope_scale: float,
        rope_short_m_scale: float,
        rope_long_m_scale: float,
        rope_max_positions: int,
        rope_original_max_positions: int,
        use_paged_context_fmha: bool,
        attention_input_type: Optional[int],
        is_mla_enable: bool,
        chunked_prefill_buffer_batch_size: Optional[int],
        q_lora_rank: Optional[int],
        kv_lora_rank: Optional[int],
        qk_nope_head_dim: Optional[int],
        qk_rope_head_dim: Optional[int],
        v_head_dim: Optional[int],
        rope_append: Optional[bool],
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        helix_position_offsets: Optional[torch.Tensor],
        helix_is_inactive_rank: Optional[torch.Tensor],
        attention_chunk_size: Optional[int],
        softmax_stats_tensor: Optional[torch.Tensor],
        is_spec_decoding_enabled: bool,
        use_spec_decoding: bool,
        is_spec_dec_tree: bool,
        spec_decoding_generation_lengths: Optional[torch.Tensor],
        spec_decoding_position_offsets_for_cpp: Optional[torch.Tensor],
        spec_decoding_packed_mask: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask: Optional[torch.Tensor],
        spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor],
        sparse_kv_indices: Optional[torch.Tensor],
        sparse_kv_offsets: Optional[torch.Tensor],
        sparse_attn_indices: Optional[torch.Tensor],
        sparse_attn_offsets: Optional[torch.Tensor],
        sparse_attn_indices_block_size: int,
        num_sparse_topk: Optional[int] = None,
        sparse_attn_kv_lens: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor_prefill: Optional[float] = None,
        skip_softmax_threshold_scale_factor_decode: Optional[float] = None,
        skip_softmax_stat: Optional[torch.Tensor] = None,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None,
        flash_mla_num_splits: Optional[torch.Tensor] = None,
        sage_attn_num_elts_per_blk_q: int = 0,
        sage_attn_num_elts_per_blk_k: int = 0,
        sage_attn_num_elts_per_blk_v: int = 0,
        sage_attn_qk_int8: bool = False,
        num_contexts: int = 0,
        num_ctx_tokens: int = 0,
        trtllm_gen_jit_warmup: bool = False,
        aux_kv_cache_pool_ptr: Optional[int] = None,
        is_cross: bool = False,
        cross_kv: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        spec_decoding_target_max_draft_tokens: Optional[int] = None,
        quant_scale_qkv: Optional[torch.Tensor] = None,
        dsv4_inv_rope_cos_sin_cache: Optional[torch.Tensor] = None,
        enable_dsv4_epilogue_fusion: bool = False,
        # Defaults match what the in-tree caller
        # (attention/backends/fmha/fallback.py) passes on a dense, non-sparse,
        # non-folded path -- which is the path both migrated targets are on.
        max_num_sequences: Optional[int] = None,
        force_prepare_spec_dec_tree_mask: bool = False,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        skip_correction_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Writes `output` in place and returns it -- the op destroys its
        prior contents in rows `[:num_tokens]`, so a caller that wants them
        has to have kept its own copy beforehand."""
        thop.attention(
            q=q,
            k=k,
            v=v,
            output=output,
            output_sf=output_sf,
            workspace_=workspace_,
            sequence_length=sequence_length,
            host_past_key_value_lengths=host_past_key_value_lengths,
            host_total_kv_lens=host_total_kv_lens,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            host_request_types=host_request_types,
            max_context_q_len_override=max_context_q_len_override,
            kv_cache_block_offsets=kv_cache_block_offsets,
            host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            cache_indirection=cache_indirection,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            out_scale=out_scale,
            rotary_inv_freq=rotary_inv_freq,
            rotary_cos_sin=rotary_cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=block_ids_per_seq,
            attention_sinks=attention_sinks,
            is_fused_qkv=is_fused_qkv,
            update_kv_cache=update_kv_cache,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            local_layer_idx=local_layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tokens_per_block=tokens_per_block,
            max_num_requests=max_num_requests,
            max_context_length=max_context_length,
            max_seq_len=max_seq_len,
            attention_window_size=attention_window_size,
            beam_width=beam_width,
            mask_type=mask_type,
            quant_mode=quant_mode,
            q_scaling=q_scaling,
            position_embedding_type=position_embedding_type,
            rope_dim=rope_dim,
            rope_base=rope_base,
            rope_scale_type=rope_scale_type,
            rope_scale=rope_scale,
            rope_short_m_scale=rope_short_m_scale,
            rope_long_m_scale=rope_long_m_scale,
            rope_max_positions=rope_max_positions,
            rope_original_max_positions=rope_original_max_positions,
            use_paged_context_fmha=use_paged_context_fmha,
            attention_input_type=attention_input_type,
            is_mla_enable=is_mla_enable,
            chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_append=rope_append,
            mrope_rotary_cos_sin=mrope_rotary_cos_sin,
            mrope_position_deltas=mrope_position_deltas,
            helix_position_offsets=helix_position_offsets,
            helix_is_inactive_rank=helix_is_inactive_rank,
            attention_chunk_size=attention_chunk_size,
            softmax_stats_tensor=softmax_stats_tensor,
            is_spec_decoding_enabled=is_spec_decoding_enabled,
            use_spec_decoding=use_spec_decoding,
            is_spec_dec_tree=is_spec_dec_tree,
            spec_decoding_generation_lengths=spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp=spec_decoding_position_offsets_for_cpp,
            spec_decoding_packed_mask=spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=spec_decoding_bl_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=spec_bl_tree_first_sparse_mask_offset_kv,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
            sparse_attn_indices_block_size=sparse_attn_indices_block_size,
            num_sparse_topk=num_sparse_topk,
            sparse_attn_kv_lens=sparse_attn_kv_lens,
            skip_softmax_threshold_scale_factor_prefill=skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode=skip_softmax_threshold_scale_factor_decode,
            skip_softmax_stat=skip_softmax_stat,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
            flash_mla_tile_scheduler_metadata=flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=flash_mla_num_splits,
            sage_attn_num_elts_per_blk_q=sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=sage_attn_qk_int8,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            trtllm_gen_jit_warmup=trtllm_gen_jit_warmup,
            aux_kv_cache_pool_ptr=aux_kv_cache_pool_ptr,
            is_cross=is_cross,
            cross_kv=cross_kv,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
            spec_decoding_target_max_draft_tokens=spec_decoding_target_max_draft_tokens,
            quant_scale_qkv=quant_scale_qkv,
            dsv4_inv_rope_cos_sin_cache=dsv4_inv_rope_cos_sin_cache,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            max_num_sequences=max_num_sequences,
            force_prepare_spec_dec_tree_mask=force_prepare_spec_dec_tree_mask,
            kv_norm_weight=kv_norm_weight,
            kv_norm_eps=kv_norm_eps,
            skip_correction_threshold=skip_correction_threshold,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Sequence[int],
        kv_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        softmax_scale: float,
        causal: bool = True,
        attention_sinks: Optional[torch.Tensor] = None,
        window: Optional[int] = None,
        e4m3_inputs: bool = False,
        out_scale: float = 1.0,
    ) -> torch.Tensor:
        """Masked softmax attention over explicit per-sequence K/V, in fp32
        with TF32 off (`true_fp32_matmul`).

        The paged-cache exception to mirroring `__call__`, and the largest one:
        this op's output is a function of what the pool holds, and its 115
        arguments mostly say where to look. `k` and `v` are the gathered
        histories the call should have attended over; the test reads them out of
        the environment, which is the only place they exist.

        Causal is bottom-right aligned -- query `i` of a sequence sits at
        absolute position `kv_len - q_len + i`, which is what makes a decode
        step's single query attend to everything before it.

        `attention_sinks` adds one per-head logit to the softmax *denominator*
        and drops it from the numerator, so the attention weights sum to less
        than one. It is not scaled by `softmax_scale`: the rival hypothesis that
        it is was measured and rejected, and the test still drives both so the
        distinction cannot quietly stop holding.

        `window` keeps only the newest `window` keys per query row. It is a pure
        mask -- the pool is shared with the full-attention layers and nothing is
        evicted -- which is why one cell can differ from another by this alone.

        `e4m3_inputs` rounds q, k and v through e4m3, which is what the fp8 path
        quantizes them to; `out_scale` is the second of that path's two
        dequantization factors, the first riding in `softmax_scale`.
        """
        rep = num_heads // num_kv_heads
        total_q = sum(q_lens)
        out = torch.full(
            (total_q, num_heads * v_head_dim), float("nan"), dtype=torch.float32, device=q.device
        )
        with true_fp32_matmul():
            q_start = kv_start = 0
            for q_len, kv_len in zip(q_lens, kv_lens):
                if kv_len == 0:
                    # The op leaves these rows undefined and the merge plans skip
                    # them; saying so here keeps a caller from reading them.
                    q_start += q_len
                    continue
                q_seq = q[q_start : q_start + q_len].reshape(q_len, num_heads, qk_head_dim).float()
                k_seq = k[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, qk_head_dim)
                v_seq = v[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, v_head_dim)
                k_seq, v_seq = k_seq.float(), v_seq.float()
                if rep > 1:
                    k_seq = k_seq.repeat_interleave(rep, dim=1)
                    v_seq = v_seq.repeat_interleave(rep, dim=1)
                if e4m3_inputs:
                    q_seq = q_seq.to(torch.float8_e4m3fn).float()
                    k_seq = k_seq.to(torch.float8_e4m3fn).float()
                    v_seq = v_seq.to(torch.float8_e4m3fn).float()

                scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq) * softmax_scale
                keep = self._mask(q_len, kv_len, causal, window, q.device)
                scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

                if attention_sinks is None:
                    probs = torch.softmax(scores, dim=-1)
                else:
                    sink = attention_sinks.float().view(num_heads, 1)
                    peak = torch.maximum(scores.max(dim=-1).values, sink)
                    weights = torch.exp(scores - peak.unsqueeze(-1))
                    probs = weights / (weights.sum(-1) + torch.exp(sink - peak)).unsqueeze(-1)

                out[q_start : q_start + q_len] = (
                    torch.einsum("hij,jhd->ihd", probs, v_seq).reshape(q_len, -1) * out_scale
                )
                q_start += q_len
                kv_start += kv_len
        return out.to(q.dtype)

    def softmax_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_lens: Sequence[int],
        kv_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        softmax_scale: float,
        causal: bool = True,
        window: Optional[int] = None,
    ) -> torch.Tensor:
        """The second output the MLA chunked-prefill path writes, `[Tq, H, 2]`.

        Per (token, head): the row max of the scaled logits and the sum of
        `exp(logit - max)` over *this pass's* KV range. Separate from
        `reference` because it is one path's extra product, not part of what
        attention computes, and folding it in would make every other cell carry
        a return value it has no claim about.

        Rows of zero-KV sequences stay NaN, as the op leaves them.
        """
        rep = num_heads // num_kv_heads
        total_q = sum(q_lens)
        stats = torch.full(
            (total_q, num_heads, 2), float("nan"), dtype=torch.float32, device=q.device
        )
        with true_fp32_matmul():
            q_start = kv_start = 0
            for q_len, kv_len in zip(q_lens, kv_lens):
                if kv_len == 0:
                    q_start += q_len
                    continue
                q_seq = q[q_start : q_start + q_len].reshape(q_len, num_heads, qk_head_dim).float()
                k_seq = k[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, qk_head_dim)
                k_seq = k_seq.float()
                if rep > 1:
                    k_seq = k_seq.repeat_interleave(rep, dim=1)
                scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq) * softmax_scale
                keep = self._mask(q_len, kv_len, causal, window, q.device)
                scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
                row_max = scores.max(dim=-1).values
                row_sum = torch.exp(scores - row_max.unsqueeze(-1)).sum(dim=-1)
                stats[q_start : q_start + q_len, :, 0] = row_max.transpose(0, 1)
                stats[q_start : q_start + q_len, :, 1] = row_sum.transpose(0, 1)
                q_start += q_len
                kv_start += kv_len
        return stats

    @staticmethod
    def _mask(q_len: int, kv_len: int, causal: bool, window: Optional[int], device) -> torch.Tensor:
        """`[q_len, kv_len]` keep-mask, bottom-right aligned."""
        if not causal and window is None:
            return torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        keep = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
        offset = kv_len - q_len
        for i in range(q_len):
            high = offset + i + 1 if causal else kv_len
            low = 0 if window is None else max(0, high - window)
            keep[i, low:high] = True
        return keep

    def is_valid(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        attention_sinks: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        host_context_lengths: Optional[torch.Tensor] = None,
        num_contexts: int = 0,
        num_ctx_tokens: int = 0,
        tokens_per_block: Optional[int] = None,
        is_mla_enable: bool = False,
        **unused_kwargs: object,
    ) -> None:
        """Keyword-only, because every call site passes keywords.

        The one entry that does not mirror `__call__`. This op takes 115
        arguments; naming all of them here would restate the signature without
        adding a claim, and `reference` cannot mirror it either -- it is the
        paged-cache exception and takes gathered K/V -- so there is no shared
        argument list left for the mirroring to preserve. It names the
        arguments it reads, defaulted so that one claim can be probed at a
        time, and `**unused_kwargs` absorbs the rest.
        """
        # A context-only call with a non-multiple returns without raising:
        # measured at 6q/4kv d128, only the first
        # `(num_heads // num_kv_heads) * num_kv_heads` head columns of output are
        # computed and the rest come back all-zero. The geometry cannot be left
        # for the op to reject.
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
        # attention_sinks is consumed as a raw buffer of exactly num_heads fp32
        # values read from data_ptr(); only its dtype is validated by the op.
        # Measured: a stride-2 view whose values were correct produced silently
        # wrong output (the underlying memory is what gets read), a 32-element
        # tensor was read 64 elements deep past its end, a 128-element one had
        # its tail silently ignored, and an empty tensor silently disabled the
        # sink.
        if attention_sinks is not None:
            assert attention_sinks.is_contiguous() and attention_sinks.numel() == num_heads, (
                f"attention_sinks must be contiguous with exactly num_heads "
                f"({num_heads}) elements, got shape "
                f"{tuple(attention_sinks.shape)} contiguous="
                f"{attention_sinks.is_contiguous()}"
            )
        # Generation rows carrying several query tokens each (an MTP verify
        # step) come out right only from the trtllm-gen generation kernels.
        # Where XqaDispatcher hands generation to MMHA instead -- more than 32
        # q heads per KV head, pages under 8 tokens -- the op still runs them and
        # answers wrongly without raising. Measured with 3 tokens per row: NaNs
        # at 64q/1kv d128, finite values 630-700 ulp off at 4-token pages
        # (16q/1kv d256 and 32q/8kv d128); 1 token per row is right on the same
        # path. Beam search falls back to MMHA as well; nothing here drives it.
        # MLA generation serves its draft rows through its own kernels and is
        # not this case.
        if q is not None and host_context_lengths is not None and not is_mla_enable:
            num_generations = host_context_lengths.shape[0] - num_contexts
            if num_generations > 0 and q.shape[0] - num_ctx_tokens > num_generations:
                small_pages = tokens_per_block is not None and tokens_per_block < 8
                assert num_heads <= 32 * num_kv_heads and not small_pages, (
                    f"generation rows of several query tokens need the trtllm-gen "
                    f"kernels; {num_heads}q/{num_kv_heads}kv heads at tokens_per_block "
                    f"{tokens_per_block} fall back to MMHA, which computes them wrongly"
                )


thop_attention = ThopAttention()
