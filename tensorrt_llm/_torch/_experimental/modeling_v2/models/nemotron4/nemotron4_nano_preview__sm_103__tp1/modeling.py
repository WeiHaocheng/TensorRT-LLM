# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: nemotron4-nano-preview (text) / sm_103 / tp1 — self-contained modeling code.

The text model of the Nemotron4ForConditionalGeneration checkpoint: 44 blocks
of hidden 3072 on three wide-residual streams -- 12 GDP mixers alone, 4 dense
relu2 MLPs, 8 attention and 20 GDP mixers each beside a latent MoE (512 routed
experts at top-10 in a 768-wide latent space, plus a shared expert) -- and the
checkpoint's MTP layer, live only under a `speculative_config`. Every weight is
the checkpoint's bf16, declared in the sibling weights.py.

Runtime state. The engine recognizes the checkpoint as a hybrid
(`config_utils.is_nemotron4`) and allocates, per request slot, the 32 GDP
mixers' recurrent state ([24, 64, 128] fp32 as [heads, value, key], key
innermost) and conv state ([16896, 3] bf16, oldest input first), and paged bf16
KV for the 8 attention blocks (1 KV head of 256). `Nemotron4StateMetadata`
turns the engine's per-step slot list into the slots the state entries read;
each target's `step_args` projects the prepared metadata onto the explicit
arguments its forward passes.

The forward is a probe of that state, not the model: it reads every step
argument, keeps a step counter in each request's slot the way the GDP entries
read and write a slot, and reports what it read as the step's sampled token.
"""

import itertools
import json
import os
import weakref
from typing import NamedTuple

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._utils import prefer_pinned

from . import weights as _weights

#: Block types by the state they keep: a GDP mixer keeps recurrent and conv
#: state, an attention mixer keeps paged KV, a dense MLP keeps nothing.
_GDP_BLOCKS = ("linear_attention", "shortcut_moe_gdp")
_ATTENTION_BLOCKS = ("shortcut_moe_attn",)


class Nemotron4StateMetadata(Mamba2Metadata):
    """`Mamba2Metadata` plus what the GDP and conv entries need from a step.

    `state_slots` (int32 [rows], contexts first): each row's state slot, and
    `PAD_SLOT_ID` on the trailing CUDA-graph padding rows. The engine gives every
    padding row the padding dummy's own slot, repeated, so several rows of one
    step would write one slot; the state entries read zeros for `PAD_SLOT_ID`
    and write nothing.

    Fresh slots. A slot handed to a new request still holds the previous
    owner's state: the cache managers never clear it. A context row honors
    that through `has_initial_states` (false: the entries do not read the
    slot). A generation row always reads its slot, and the runner turns every
    final one-token context chunk into a generation row of a CUDA-graph batch
    (`_make_single_token_context_graph_batch`). For a fresh one-token prompt
    that row has no cached token -- the only generation row without one -- so
    `prepare` zeroes its recurrent and conv state in every GDP layer; a chunked
    prompt's promoted last token keeps its state.

    `prepare` runs eagerly before every forward, CUDA-graph replays included
    (`AttentionMetadata.prepare`), on the forward's stream: it fills
    `state_slots` in place, so a captured graph reads the current step's slots,
    and the zeroing lands before the step that reads the slot.
    """

    def __init__(
        self,
        max_batch_size: int,
        chunk_size: int,
        max_num_tokens: int | None = None,
    ) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens=max_num_tokens)
        self.state_slots = torch.full(
            (max_batch_size,), PAD_SLOT_ID, dtype=torch.int32, device="cuda"
        )

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        super().prepare(attn_metadata)
        num_rows = attn_metadata.seq_lens.shape[0]
        num_real = self._num_real_rows(attn_metadata, num_rows)
        self.state_slots[:num_rows].copy_(self.state_indices[:num_rows])
        if num_real < num_rows:
            self.state_slots[num_real:num_rows].fill_(PAD_SLOT_ID)
        fresh_rows = self._fresh_generation_rows(attn_metadata, num_real)
        if fresh_rows:
            self._zero_slots(attn_metadata.kv_cache_manager, fresh_rows)

    @staticmethod
    def _num_real_rows(attn_metadata: AttentionMetadata, num_rows: int) -> int:
        """Rows before the trailing CUDA-graph padding rows.

        The runner appends padding rows after every real row; they carry the
        padding dummy's request id, `CUDA_GRAPH_DUMMY_REQUEST_ID` less the
        draft length (the test `Mamba2Metadata.prepare` uses).
        """
        request_ids = attn_metadata.request_ids
        if request_ids is None:
            return num_rows
        max_draft_len = (
            getattr(attn_metadata.kv_cache_manager, "speculative_num_draft_tokens", 0) or 0
        )
        first_padding_id = CUDA_GRAPH_DUMMY_REQUEST_ID - max_draft_len
        num_real = num_rows
        while num_real > 0 and request_ids[num_real - 1] >= first_padding_id:
            num_real -= 1
        # Without draft tokens every padding row is a generation row appended after
        # the batch; with them the runner may order padding rows among the real ones.
        if max_draft_len and any(rid >= first_padding_id for rid in request_ids[:num_real]):
            raise NotImplementedError(
                "a CUDA-graph padding row precedes a real row; Nemotron4StateMetadata marks "
                "only trailing padding rows with PAD_SLOT_ID"
            )
        return num_real

    @staticmethod
    def _fresh_generation_rows(attn_metadata: AttentionMetadata, num_real: int) -> list[int]:
        """Real generation rows with no cached token: fresh one-token prompts."""
        num_contexts = attn_metadata.num_contexts
        if num_real <= num_contexts:
            return []
        cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        if isinstance(cached, torch.Tensor):
            cached = cached[:num_real].tolist()
        if 0 not in cached[num_contexts:num_real]:
            return []
        return [row for row in range(num_contexts, num_real) if cached[row] == 0]

    def _zero_slots(self, kv_cache_manager: MambaHybridCacheManager, rows: list[int]) -> None:
        """Zero the recurrent and conv state of these rows' slots, in every GDP layer."""
        # Both hybrid managers hand `get_state_indices` back as a host list, which
        # `Mamba2Metadata.prepare` stages in `state_indices_cpu`.
        if self._state_indices_aliased_ptr is not None:
            raise RuntimeError(
                "state slots arrived as a device tensor; the fresh-slot reset reads the host copy"
            )
        # A fresh pinned tensor keeps the copy asynchronous: the host allocator does not
        # hand its block out again before the copy has run.
        index = torch.tensor(
            self.state_indices_cpu[rows].tolist(), dtype=torch.long, pin_memory=prefer_pinned()
        ).to(self.state_slots.device, non_blocking=True)
        for layer in kv_cache_manager.mamba_layer_offsets:
            kv_cache_manager.get_ssm_states(layer).index_fill_(0, index, 0)
            kv_cache_manager.get_conv_states(layer).index_fill_(0, index, 0)


class _StateViews(NamedTuple):
    """One cache manager's per-layer addresses.

    `kv_layer` maps an attention layer to its index in the paged KV pools. The
    hybrid managers number every stateful layer, recurrent and attention alike,
    so the eight attention layers are not 0..7. `ssm_states` / `conv_states` map
    a GDP layer to its pool view: [slots, 24, 64, 128] fp32 and [slots, 16896,
    3] bf16, any slot stride, inner dims dense.
    """

    kv_layer: dict[int, int]
    ssm_states: dict[int, torch.Tensor]
    conv_states: dict[int, torch.Tensor]


class Nemotron4NanoPreviewTp1ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        d = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(d)

        self.gdp_layers = tuple(i for i, t in enumerate(d.layer_types) if t in _GDP_BLOCKS)
        # The trunk's attention layers. Under one-model MTP the engine keeps the
        # MTP layer's KV (global layer 44) in its separate draft KV cache
        # manager, not in the manager that prepares the step.
        self.attention_layers = tuple(
            i for i, t in enumerate(d.layer_types) if t in _ATTENTION_BLOCKS
        )
        self._checked_manager: weakref.ref | None = None

        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None

    def post_load_weights(self) -> None:
        """Construct the two targets, now that the weights are real.

        Must only be called after meta init is over, never from `__init__`, where the
        shell's containers are still meta.

        Named for the shell hook that drives it. Not an override: that hook lives on
        DecoderModelForCausalLM, and a core derives from DecoderModel, which has no
        such method -- so there is no super() to call here.
        """
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def state_views(self, kv_cache_manager: MambaHybridCacheManager) -> _StateViews:
        """The per-layer cache addresses of the manager that prepared this step.

        Read off the manager on every call and never kept: the engine sizes its
        caches with one manager, shuts it down and then builds the one it serves
        with, and a manager drops its views at shutdown so that none outlives the
        pool it points into. Only the manager itself is remembered, weakly, to
        check its layer layout once.
        """
        if self._checked_manager is None or self._checked_manager() is not kv_cache_manager:
            gdp_in_cache = sorted(getattr(kv_cache_manager, "mamba_layer_offsets", {}))
            if gdp_in_cache != list(self.gdp_layers):
                raise RuntimeError(
                    f"{type(kv_cache_manager).__name__} keeps recurrent state for layers "
                    f"{gdp_in_cache}; this target runs GDP mixers on {list(self.gdp_layers)}. "
                    "The engine must recognize the checkpoint as a Nemotron 4 hybrid "
                    "(config_utils.is_nemotron4)."
                )
            self._checked_manager = weakref.ref(kv_cache_manager)
        return _StateViews(
            kv_layer={i: kv_cache_manager.layer_offsets[i] for i in self.attention_layers},
            ssm_states={i: kv_cache_manager.get_ssm_states(i) for i in self.gdp_layers},
            conv_states={i: kv_cache_manager.get_conv_states(i) for i in self.gdp_layers},
        )

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment, and a predicate that reached for a second field
        would be reading one this call is not guaranteed to carry. A mixed batch -- context
        and generation rows in one step -- goes to prefill, so prefill is the general case
        and decode the specialization that may assume it has no context rows.
        """
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Route the step, and nothing else."""
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


def _state_metadata(md: AttentionMetadata) -> Nemotron4StateMetadata:
    """The step's state metadata, which only a Nemotron 4 hybrid cache manager prepares."""
    state = md.mamba_metadata
    if not isinstance(state, Nemotron4StateMetadata):
        raise RuntimeError(
            f"the step carries {type(state).__name__} state metadata under "
            f"{type(md.kv_cache_manager).__name__}; this target needs Nemotron4StateMetadata, "
            "which the engine prepares only with a hybrid recurrent-state cache "
            "(config_utils.is_nemotron4)"
        )
    return state


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: Nemotron4NanoPreviewTp1ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config.text_config
        self._hidden_size = cfg.hidden_size
        self._dtype = core.w["final_norm"].dtype

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes, so a rename upstream surfaces here rather than mid-forward. Everything
        checked is fixed at engine construction, so once per target instance is sound,
        and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto the explicit batch state of thop_attention
        and the recurrent-state entries, once per forward.

        Rows are requests, contexts first; tokens are packed row by row. The state
        entries take the context rows' slices: `query_start_loc[: num_contexts + 1]`,
        `state_slots[:num_contexts]`, `has_initial_state`; the generation rows that
        follow take one token and one slot each (routing admits no speculative
        decoding, so no row carries draft tokens).
        """
        state = _state_metadata(md)
        views = self.core.state_views(md.kv_cache_manager)
        num_rows = md.num_seqs
        num_contexts = md.num_contexts
        return dict(
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            workspace_=md.effective_workspace,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            num_contexts=num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
            beam_width=md.effective_beam_width,
            cache_indirection=md.cache_indirection,
            block_ids_per_seq=md.block_ids_per_seq,
            max_context_q_len_override=md.max_context_q_len_override,
            is_cross=md.is_cross,
            is_spec_decoding_enabled=md.is_spec_decoding_enabled,
            use_spec_decoding=md.use_spec_decoding,
            is_spec_dec_tree=md.is_spec_dec_tree,
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
            spec_decoding_packed_mask=md.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=md.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=md.flash_mla_num_splits,
            max_num_sequences=md.max_num_sequences,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
            # Recurrent state: rows, their slots and where each row's tokens start.
            num_rows=num_rows,
            num_generations=num_rows - num_contexts,
            context_lens=md.seq_lens[:num_contexts],
            query_start_loc=state.query_start_loc,
            query_start_loc_long=state.query_start_loc_long,
            has_initial_state=state.has_initial_states[:num_contexts],
            state_slots=state.state_slots[:num_rows],
            kv_layer=views.kv_layer,
            ssm_states=views.ssm_states,
            conv_states=views.conv_states,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        num_rows = step["num_rows"]
        row_ends = step["query_start_loc"][1 : num_rows + 1].long() - 1
        return _probe_step(
            self.core,
            step,
            num_tokens=(input_ids if inputs_embeds is None else inputs_embeds).shape[0],
            num_rows=num_rows,
            num_contexts=step["num_contexts"],
            row_ends=row_ends,
            hidden_size=self._hidden_size,
            dtype=self._dtype,
        )


class DecodeTarget:
    """The specialization: routed to only when there are no context rows."""

    def __init__(self, core: Nemotron4NanoPreviewTp1ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config.text_config
        self._hidden_size = cfg.hidden_size
        self._dtype = core.w["final_norm"].dtype

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes, so a rename upstream surfaces here rather than mid-forward. Everything
        checked is fixed at engine construction, so once per target instance is sound,
        and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto the explicit batch state of thop_attention
        and the recurrent-state entries, once per forward.

        Every row is one generation token. Under a CUDA graph the rows include the
        trailing padding rows, whose `state_slots` entry is `PAD_SLOT_ID`.
        """
        state = _state_metadata(md)
        views = self.core.state_views(md.kv_cache_manager)
        num_rows = md.num_seqs
        return dict(
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            workspace_=md.effective_workspace,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            num_contexts=0,
            num_ctx_tokens=0,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
            beam_width=md.effective_beam_width,
            cache_indirection=md.cache_indirection,
            block_ids_per_seq=md.block_ids_per_seq,
            max_context_q_len_override=md.max_context_q_len_override,
            is_cross=md.is_cross,
            is_spec_decoding_enabled=md.is_spec_decoding_enabled,
            use_spec_decoding=md.use_spec_decoding,
            is_spec_dec_tree=md.is_spec_dec_tree,
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
            spec_decoding_packed_mask=md.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=md.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=md.flash_mla_num_splits,
            max_num_sequences=md.max_num_sequences,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
            # Recurrent state: one row per token, padding rows on PAD_SLOT_ID.
            num_rows=num_rows,
            state_slots=state.state_slots[:num_rows],
            kv_layer=views.kv_layer,
            ssm_states=views.ssm_states,
            conv_states=views.conv_states,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        num_rows = step["num_rows"]
        return _probe_step(
            self.core,
            step,
            num_tokens=(input_ids if inputs_embeds is None else inputs_embeds).shape[0],
            num_rows=num_rows,
            num_contexts=0,
            row_ends=torch.arange(num_rows, device=step["state_slots"].device),
            hidden_size=self._hidden_size,
            dtype=self._dtype,
        )


# --- Probe: stands in for the model forward and its logits ------------------------------------
#
# Each slot carries a step counter in element [slot, 0, 0, 0] of the first GDP
# layer's recurrent state, and its low 7 bits in element [slot, 0, 0] of the
# last GDP layer's conv state. A step reads a row's counter the way the GDP
# entries read state -- a context row only if it has initial state, a
# generation row always, a padding row never -- and writes the count plus one
# back, never for a padding row. Each row's last hidden state carries what the
# step read; the shell turns it into a one-hot logit row, so the sampled token
# is the record, through CUDA-graph replays and the overlap scheduler alike.
# The token is _PROBE_TOKEN_BASE + code, code = count % 64 | (slot + 1) << 6 |
# read << 11 | repeated slot in the step << 12 | conv bits agree << 13 |
# padding rows in the step (at most 7) << 14.

_PROBE_TOKEN_BASE = 16
_PROBE_FIELDS = 6
_PROBE_LOG_ENV = "NEMOTRON4_TP1_PROBE_LOG"
_probe_records = itertools.count(1)


def _probe_counters(
    core: Nemotron4NanoPreviewTp1ModelingV2, step: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """The counter and its conv copy, as [slots] views; zeroed once per manager.

    The pools start as whatever the allocator left, so the probe clears its
    two elements of every slot the first time it sees a manager (an eager
    warmup step); after that only the steps write them.
    """
    first, last = core.gdp_layers[0], core.gdp_layers[-1]
    counter = step["ssm_states"][first][:, 0, 0, 0]
    low_bits = step["conv_states"][last][:, 0, 0]
    owner = core._checked_manager
    if getattr(core, "_probe_zeroed_for", None) is not owner:
        assert not torch.cuda.is_current_stream_capturing(), "probe counters cleared in a capture"
        counter.zero_()
        low_bits.zero_()
        core._probe_zeroed_for = owner
    return counter, low_bits


def _probe_step(
    core: Nemotron4NanoPreviewTp1ModelingV2,
    step: dict,
    *,
    num_tokens: int,
    num_rows: int,
    num_contexts: int,
    row_ends: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read and advance every row's counter; return [num_tokens, hidden] with each
    row's record in the first `_PROBE_FIELDS` columns of its last token."""
    counter, low_bits = _probe_counters(core, step)
    slots = step["state_slots"]
    valid = slots >= 0
    index = slots.clamp(min=0).long()
    reads = valid.clone()
    if num_contexts:
        reads[:num_contexts] &= step["has_initial_state"]
    zero = torch.zeros(num_rows, dtype=torch.float32, device=slots.device)
    stale = counter[index]
    seen = torch.where(reads, stale, zero)
    # Writes are additions, so a padding row adds zero to slot 0 instead of racing a real row there.
    counter.scatter_add_(0, index, torch.where(valid, seen + 1 - stale, zero))
    stale_low = low_bits[index].float()
    low_agrees = ~reads | (stale_low == torch.remainder(seen, 128))
    low_delta = torch.where(valid, torch.remainder(seen + 1, 128) - stale_low, zero)
    low_bits.scatter_add_(0, index, low_delta.to(low_bits.dtype))
    # A real slot on two rows: more equal (row, row) pairs than real rows.
    pairs = (slots.unsqueeze(1) == slots.unsqueeze(0)) & valid.unsqueeze(1)
    repeated = pairs.sum() > valid.sum()
    padding = (~valid).sum().clamp(max=7)
    fields = torch.stack(
        [
            torch.remainder(seen, 64),
            (slots + 1).float(),
            reads.float(),
            repeated.float().expand(num_rows),
            low_agrees.float(),
            padding.float().expand(num_rows),
        ],
        dim=1,
    )
    hidden = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=slots.device)
    hidden[row_ends, :_PROBE_FIELDS] = fields.to(dtype)
    return hidden


class _ProbeStateMetadata(Nemotron4StateMetadata):
    """`Nemotron4StateMetadata` that appends every prepared step to the JSON-lines
    file `NEMOTRON4_TP1_PROBE_LOG` names (nothing when it is unset)."""

    def __init__(
        self,
        max_batch_size: int,
        chunk_size: int,
        max_num_tokens: int | None = None,
    ) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens=max_num_tokens)
        path = os.environ.get(_PROBE_LOG_ENV)
        self._log = open(path, "a", buffering=1, encoding="utf-8") if path else None
        self._managers = weakref.WeakSet()
        self._zeroed: list = []

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        self._zeroed = []
        super().prepare(attn_metadata)
        if self._log is None:
            return
        manager = attn_metadata.kv_cache_manager
        if manager not in self._managers:
            self._managers.add(manager)
            self._write(dict(kind="manager", **_describe_manager(manager)))
        num_rows = attn_metadata.seq_lens.shape[0]
        num_contexts = attn_metadata.num_contexts
        cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        if isinstance(cached, torch.Tensor):
            cached = cached.tolist()
        slots = self.state_indices_cpu[:num_rows].tolist()
        num_real = self._num_real_rows(attn_metadata, num_rows)
        self._write(
            dict(
                kind="step",
                step=next(_probe_records),
                metadata=id(self),
                manager=id(manager),
                graph=bool(attn_metadata.is_cuda_graph),
                num_contexts=num_contexts,
                request_ids=[int(r) for r in (attn_metadata.request_ids or [])[:num_rows]],
                seq_lens=attn_metadata.seq_lens[:num_rows].tolist(),
                cached=[int(c) for c in cached[:num_rows]],
                slots=slots,
                state_slots=slots[:num_real] + [PAD_SLOT_ID] * (num_rows - num_real),
                has_initial_state=self.has_initial_states_cpu[:num_contexts].tolist(),
                zeroed=self._zeroed,
            )
        )

    def _zero_slots(self, kv_cache_manager: MambaHybridCacheManager, rows: list[int]) -> None:
        if self._log is not None:
            # Probe only: what each fresh row's slot held before the reset (a sync).
            first = min(kv_cache_manager.mamba_layer_offsets)
            counter = kv_cache_manager.get_ssm_states(first)[:, 0, 0, 0]
            slots = self.state_indices_cpu[rows].tolist()
            held = counter[torch.tensor(slots, device=counter.device)].tolist()
            self._zeroed = [dict(row=r, slot=s, held=h) for r, s, h in zip(rows, slots, held)]
        super()._zero_slots(kv_cache_manager, rows)

    def _write(self, record: dict) -> None:
        self._log.write(json.dumps(record) + "\n")


def _describe_manager(manager: MambaHybridCacheManager) -> dict:
    """What a cache manager allocated for this model: the probe's memory record."""
    gdp = sorted(manager.mamba_layer_offsets)
    ssm = manager.get_ssm_states(gdp[0])
    conv = manager.get_conv_states(gdp[0])
    attention = sorted(set(manager.layer_offsets) - set(gdp))
    return dict(
        manager_id=id(manager),
        cls=type(manager).__name__,
        gdp_layers=gdp,
        attention_layers=attention,
        kv_layer={str(i): manager.layer_offsets[i] for i in attention},
        kv_bytes_per_token=manager.get_cache_bytes_per_token(),
        tokens_per_block=manager.tokens_per_block,
        kv_pool=list(manager.get_buffers(attention[0]).shape),
        num_state_slots=ssm.shape[0],
        ssm=dict(shape=list(ssm.shape), stride=list(ssm.stride()), dtype=str(ssm.dtype)),
        conv=dict(shape=list(conv.shape), stride=list(conv.stride()), dtype=str(conv.dtype)),
        state_bytes_per_slot=len(gdp)
        * (ssm[0].numel() * ssm.element_size() + conv[0].numel() * conv.element_size()),
    )


class ModelingV2Nemotron4NanoPreviewSm103Tp1(
    DecoderModelForCausalLM[Nemotron4NanoPreviewTp1ModelingV2, PretrainedConfig]
):
    #: Built by the engine for every step (`resolve_mamba_metadata_cls`).
    mamba_metadata_cls = _ProbeStateMetadata

    def __init__(self, model_config: ModelConfig):
        # The composite config keeps the text model's widths under text_config.
        cfg = model_config.pretrained_config
        text = cfg.text_config
        # The engine reads the depth off the composite config when it sizes its caches, but
        # text_config derives it (a property over layer_types) and ModelConfig's mirror of
        # text_config onto the composite copies stored fields only.
        cfg.num_hidden_layers = text.num_hidden_layers
        super().__init__(
            Nemotron4NanoPreviewTp1ModelingV2(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        self._probe_vocab = text.vocab_size
        assert _PROBE_TOKEN_BASE + (1 << 17) <= self._probe_vocab, "probe codes exceed the vocab"

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Probe logits: one row per request (every token with `return_context_logits`),
        1.0 at the token that encodes the row's record, 0 elsewhere."""
        hidden = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )
        if not return_context_logits:
            last = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            hidden = hidden[last]
        f = hidden[:, :_PROBE_FIELDS].float().long()
        code = (
            (f[:, 0] & 63)
            | (f[:, 1] & 31) << 6
            | (f[:, 2] & 1) << 11
            | (f[:, 3] & 1) << 12
            | (f[:, 4] & 1) << 13
            | (f[:, 5] & 7) << 14
        )
        logits = torch.zeros(
            hidden.shape[0], self._probe_vocab, dtype=torch.float32, device=hidden.device
        )
        logits.scatter_(1, (code + _PROBE_TOKEN_BASE).unsqueeze(1), 1.0)
        return logits
