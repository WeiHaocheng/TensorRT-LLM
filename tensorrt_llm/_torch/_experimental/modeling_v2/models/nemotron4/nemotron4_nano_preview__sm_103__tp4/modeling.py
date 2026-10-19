# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: nemotron4-nano-preview (text) / sm_103 / tp4 — self-contained modeling code.

The text model of the Nemotron4ForConditionalGeneration checkpoint: 44 blocks
of hidden 3072 on three wide-residual streams -- 12 GDP mixers alone, 4 dense
relu2 MLPs, 8 attention and 20 GDP mixers each beside a latent MoE (512 routed
experts at top-10 in a 768-wide latent space, plus a shared expert) -- read out
of the streams, normed, and projected by an fp32-output lm_head. The
checkpoint's MTP layer is declared and loaded only under a `speculative_config`
(weights.py); this target has no draft forward yet. Every weight is the
checkpoint's bf16, laid out for the entries at load.

`tp4` is tensor parallel over the 4 GPUs of one node, without attention DP:
every rank runs every request. A rank holds 6 of the 24 GDP heads, 4 of the 16
query heads over the one KV head, which every rank keeps whole, and a quarter
of every MLP's intermediate width -- the dense MLPs', the shared experts' and
each of the 512 routed experts'. The wide residual streams, the norms, the
router, the latent projections and lm_head are replicated.

So every branch's projection back to the model width is a partial sum over the
ranks, and an all-reduce (`allreduce`, the TP group's MNNVL workspace) makes
it whole before the branch writes the streams: after a GDP mixer's `out_proj`,
an attention mixer's `o_proj` and a dense MLP's down projection. A latent MoE
reduces its routed experts' 768-wide partial before `fc2_norm`, which is not
linear, and its shared expert's partial in the same call. That is 72
all-reduces a forward. Each rank's GEMMs round their partial to bf16, and the
all-reduce sums the four partials in fp32, in rank order, and rounds once, so
every rank holds the same bits of every stream: the replicated work after it
needs no collective, and every rank forms the same logits.

The streams are `[T, 3 * 3072]` bf16, the embedding repeated three times. One
`wide_residual` call sits between every two branches: it writes the finished
branch's update into the streams and makes the next branch's reads, each with
its RMSNorm -- 73 calls a forward. A shortcut block reads its routed input and
its mixer input in the call before its mixer, from the same streams, and the
shared expert's input in the call after it, which is HF's order.

Runtime state, per rank. The engine recognizes the checkpoint as a hybrid
(`config_utils.is_nemotron4`) and gives every rank its own cache manager over
the rank's part of each request's state: per request slot, the 32 GDP mixers'
recurrent state of the rank's 6 heads ([6, 64, 128] fp32 as [heads, value,
key], key innermost) and conv state of its 4224 conv channels ([4224, 3] bf16,
oldest input first: its heads' blocks of V0..V2, K0..K2 and Q), and paged bf16
KV for the 8 attention blocks' one KV head of 256. Every rank runs the same
rows in the same order, each on its own slots. `Nemotron4StateMetadata` turns
the engine's per-step slot list into the slots the state entries read; each
target's `step_args` projects the prepared metadata onto the explicit
arguments its forward passes.

Batch invariance. A decode step computes every row from that row alone, in an
order that does not depend on how many rows share the step, so a request's
logits and the state it writes are bitwise the same alone, in any eager batch
and in a padded CUDA-graph batch, on every rank. cuBLAS picks its kernel by the
row count, and at three of a rank's GEMM shapes -- its slice of the GDP
`in_proj`, the router and `fc1_latent_proj` -- the kernels it picks sum
differently: decode runs those through `invariant_mm`, and generation attention
through `invariant_gen_attention`, whose KV split is fixed. The all-reduce
moves nothing: it forms every element's sum by one formula -- fp32, rank order,
one rounding -- whatever the row count or the row's place in the message.
Prefill and mixed steps keep cuBLAS and `thop_attention`; CUDA graphs do not
change their batches.
"""

import weakref
from typing import NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.softclamp_relu2 import (
    softclamp_relu2,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.invariant_gen_attention import (
    invariant_gen_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import allreduce
from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_update import (
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_varlen import (
    causal_conv1d_varlen,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_chunk import gdp_chunk
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_gates import gdp_gates
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_recurrent import gdp_recurrent
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.invariant_mm import invariant_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
    latent_moe_experts,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.gated_rmsnorm import gated_rmsnorm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.wide_residual import (
    wide_residual,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._utils import prefer_pinned

from . import weights as _weights

#: Block types by what they run: a GDP mixer keeps recurrent and conv state, an
#: attention mixer keeps paged KV, a dense MLP keeps nothing; the shortcut blocks
#: run their mixer beside a latent MoE.
_GDP_BLOCKS = ("linear_attention", "shortcut_moe_gdp")
_ATTENTION_BLOCKS = ("shortcut_moe_attn",)
_SHORTCUT_BLOCKS = ("shortcut_moe_attn", "shortcut_moe_gdp")

# thop_attention's arguments that do not move with the step. NoPE: no positional
# encoding at all (`Nemotron4Attention` applies none), so the rope_* values are
# inert placeholders; q_scaling 1 keeps the softmax scale at head_dim**-0.5 = 1/16.
_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    attention_sinks=None,
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)


def _row_runs(rows: list[int]) -> list[tuple[int, int]]:
    """Ascending row indices grouped into maximal `[start, stop)` runs."""
    runs: list[tuple[int, int]] = []
    for row in rows:
        if runs and runs[-1][1] == row:
            runs[-1] = (runs[-1][0], row + 1)
        else:
            runs.append((row, row + 1))
    return runs


class Nemotron4StateMetadata(Mamba2Metadata):
    """`Mamba2Metadata` plus what the GDP and conv entries need from a step, on one
    tensor-parallel rank.

    Every rank prepares every step, on its own cache manager, whose slots hold
    the rank's heads of each request's state.

    `state_slots` (int32 [rows], contexts first): each row's state slot, and
    `PAD_SLOT_ID` on every CUDA-graph padding row. The engine gives every
    padding row the padding dummy's own slot, repeated, so several rows of one
    step would write one slot; the state entries read zeros for `PAD_SLOT_ID`
    and write nothing. Padding rows are generation rows -- the context entries
    have no padding slot -- and are marked wherever they sit, although the
    runner appends them after every real row (`CUDAGraphRunner._get_padded_batch`).

    Fresh slots. A slot handed to a new request still holds the previous
    owner's state: the cache managers never clear it. A context row honors
    that through `has_initial_states` (false: the entries do not read the
    slot). A generation row always reads its slot, and the runner turns a final
    one-token context chunk into a generation row whenever the step runs as a
    CUDA graph (`_make_single_token_context_graph_batch`). For a fresh
    one-token prompt that generation row has no cached token -- the only real
    generation row without one -- so `prepare` zeroes its recurrent and conv
    state in every GDP layer; a chunked prompt's promoted last token keeps its
    state.

    `prepare` runs eagerly before every forward, CUDA-graph replays included
    (`AttentionMetadata.prepare`), on the forward's stream: it fills
    `state_slots` in place, so a captured graph reads the current step's slots,
    and the zeroing lands before the step that reads the slot.

    Speculative decoding. A padding row's request id is the padding dummy's,
    `CUDA_GRAPH_DUMMY_REQUEST_ID` less the step's draft length, which is at most
    the target's `max_draft_len`. The engine builds this metadata itself and
    hands it no speculative config, and the hybrid managers do not state the
    draft length either, so the shell binds the target's own
    (`for_draft_len`). `state_indices` keeps the engine's slots, the padding
    dummy's included.
    """

    #: The longest draft a generation row carries: the target's `max_draft_len`
    #: under speculative decoding, 0 without it.
    max_draft_len: int = 0

    @classmethod
    def for_draft_len(cls, max_draft_len: int) -> type["Nemotron4StateMetadata"]:
        """This class with the target's draft length bound, for the shell's `mamba_metadata_cls`."""
        return type(cls.__name__, (cls,), {"max_draft_len": max_draft_len})

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
        padding = self._padding_rows(attn_metadata, num_rows)
        self.state_slots[:num_rows].copy_(self.state_indices[:num_rows])
        for start, stop in _row_runs(padding):
            self.state_slots[start:stop].fill_(PAD_SLOT_ID)
        fresh_rows = self._fresh_generation_rows(attn_metadata, num_rows, padding)
        if fresh_rows:
            self._zero_slots(attn_metadata.kv_cache_manager, fresh_rows)

    def _padding_rows(self, attn_metadata: AttentionMetadata, num_rows: int) -> list[int]:
        """The CUDA-graph padding rows, ascending: generation rows whose request id is
        at or above `CUDA_GRAPH_DUMMY_REQUEST_ID - max_draft_len` (the test
        `Mamba2Metadata.prepare` uses)."""
        request_ids = attn_metadata.request_ids
        if request_ids is None:
            return []
        first_padding_id = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len
        return [
            row
            for row in range(attn_metadata.num_contexts, num_rows)
            if request_ids[row] >= first_padding_id
        ]

    @staticmethod
    def _fresh_generation_rows(
        attn_metadata: AttentionMetadata, num_rows: int, padding: list[int]
    ) -> list[int]:
        """Real generation rows with no cached token: fresh one-token prompts."""
        num_contexts = attn_metadata.num_contexts
        if num_rows <= num_contexts:
            return []
        cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        if isinstance(cached, torch.Tensor):
            cached = cached[:num_rows].tolist()
        if 0 not in cached[num_contexts:num_rows]:
            return []
        skip = set(padding)
        return [
            row for row in range(num_contexts, num_rows) if cached[row] == 0 and row not in skip
        ]

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


def _rank_state_shapes(model_config: ModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """One slot's state per GDP layer on this rank: the recurrent state of the rank's
    heads, [heads, value, key], and the conv state of its channels -- the packed
    [V0..V2 | K0..K2 | Q] conv input cut to its heads -- by the conv's kept inputs."""
    text = model_config.pretrained_config.text_config
    tp = model_config.mapping.tp_size
    value = text.gdp_num_heads * text.gdp_head_dim
    key = text.gdp_num_groups * text.gdp_key_head_dim
    conv_width = text.gdp_num_householder * (value + key) + key
    return (
        (text.gdp_num_heads // tp, text.gdp_head_dim, text.gdp_key_head_dim),
        (conv_width // tp, text.gdp_conv_kernel - 1),
    )


class _StateViews(NamedTuple):
    """One cache manager's per-layer addresses, on this rank.

    `kv_layer` maps an attention layer to its index in the paged KV pools. The
    hybrid managers number every stateful layer, recurrent and attention alike,
    so the eight attention layers are not 0..7. `ssm_states` / `conv_states` map
    a GDP layer to its pool view: [slots, 6, 64, 128] fp32 and [slots, 4224, 3]
    bf16 -- the rank's heads and conv channels -- any slot stride, inner dims
    dense.
    """

    kv_layer: dict[int, int]
    ssm_states: dict[int, torch.Tensor]
    conv_states: dict[int, torch.Tensor]


class _AllreduceWorkspace(NamedTuple):
    """The TP group's MNNVL workspace, as `allreduce` takes it: the unicast buffer viewed
    as its 3 Lamport stages, and the protocol's control words."""

    buffer: torch.Tensor
    flags: torch.Tensor
    rows: int  # the most rows one call may carry, the workspace's size


def _allreduce_workspace(model_config: ModelConfig, widths: tuple[int, ...]) -> _AllreduceWorkspace:
    """Build (or find) the TP group's MNNVL workspace, sized for a step of
    `max_num_tokens` rows at the widest width any all-reduce sends.

    Collective over the group: every rank builds it, eagerly, before any CUDA graph
    is captured. Nothing else in the engine grows it, so the views taken here stay
    the workspace's record (`allreduce`'s `is_valid` checks that they do).
    """
    mapping = model_config.mapping
    dtype = model_config.torch_dtype
    rows = model_config.max_num_tokens
    need = max(
        MNNVLAllReduce.get_required_workspace_size(rows, width, mapping.tp_size, dtype)
        for width in widths
    )
    workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, dtype, buffer_size_bytes=need)
    return _AllreduceWorkspace(
        buffer=workspace["uc_buffer"].view(dtype).view(3, -1),
        flags=workspace["buffer_flags"],
        rows=rows,
    )


# --- Per-layer weight views ---------------------------------------------------------------------
#
# Built once per target off the loaded weights: each GEMM operand as the `.t()` view cublas_mm
# reads, each wide-residual coefficient as the bf16 [3] vector wide_residual reads. Shapes are a
# rank's. The decode target runs the operands marked "invariant_mm" through that entry instead,
# which reads the stored [out, in] weight; its views keep that layout for them.


class _Read(NamedTuple):
    """One read of the streams and the RMSNorm that follows it."""

    factors: torch.Tensor  # f [streams]
    norm: torch.Tensor  # [hidden]


class _Write(NamedTuple):
    """One branch's write into the streams."""

    retention: torch.Tensor  # r [streams]
    write: torch.Tensor  # w [streams]


class _Gdp(NamedTuple):
    """A GDP mixer (`Nemotron4GDPMixer`) on this rank's 6 heads."""

    layer: int  # global layer index: its state pools are step["ssm_states"/"conv_states"][layer]
    in_proj: torch.Tensor  # [hidden, 4632]; invariant_mm: [4632, hidden]
    conv: torch.Tensor  # [4224, 4]
    neg_a: torch.Tensor  # -exp(A_log) [6] fp32
    dt_bias: torch.Tensor  # [6]
    norm: torch.Tensor  # [384]
    out_proj: torch.Tensor  # [384, hidden], row-parallel


class _Attention(NamedTuple):
    """An attention mixer (`Nemotron4Attention`) on this rank's 4 query heads: fused qkv
    over the whole KV head, NoPE."""

    layer: int  # global layer index: its KV pool index is step["kv_layer"][layer]
    qkv: torch.Tensor  # [hidden, 1536]
    o: torch.Tensor  # [1024, hidden], row-parallel


class _Mlp(NamedTuple):
    """A dense soft-clamped relu2 MLP (`Nemotron4MLP`) on this rank's quarter of its width."""

    up: torch.Tensor  # [hidden, 3072]
    down: torch.Tensor  # [3072, hidden], row-parallel


class _Moe(NamedTuple):
    """A latent MoE (`Nemotron4MoE`): router, routed latent experts at this rank's quarter
    of every expert's intermediate width, shared expert at a quarter of its width."""

    router: torch.Tensor  # [hidden, 512]; invariant_mm: [512, hidden]
    router_bias: torch.Tensor  # [512] fp32
    fc1_latent: torch.Tensor  # [hidden, 768]; invariant_mm: [768, hidden]
    experts_up: torch.Tensor  # [512, 1024, 768]
    experts_down: torch.Tensor  # [512, 768, 1024]
    fc2_norm: torch.Tensor  # [768]
    fc2_latent: torch.Tensor  # [768, hidden]
    shared_up: torch.Tensor  # [hidden, 1024]
    shared_down: torch.Tensor  # [1024, hidden], row-parallel


class _Layer(NamedTuple):
    """One block and the streams around it.

    `write` is a plain block's one write, or a shortcut block's mixer write; a
    shortcut block then reads `shared_read` for its shared expert and writes
    the MoE's output with `moe_write`. `next_reads` is what the call after the
    block's last write reads: the next block's inputs (a shortcut block's
    routed input first, then its mixer input), or, after the last block, the
    readout with the final norm.
    """

    mixer: _Gdp | _Attention | _Mlp
    write: _Write
    moe: _Moe | None
    shared_read: _Read | None
    moe_write: _Write | None
    next_reads: tuple[_Read, ...]


def _block_reads(w, layer_types: tuple[str, ...], i: int) -> tuple[_Read, ...]:
    """What block `i` reads off the streams before it runs; the readout past the last block."""
    if i == len(layer_types):
        return (_Read(w["readout"], w["final_norm"]),)
    if layer_types[i] in _SHORTCUT_BLOCKS:
        return (
            _Read(w[f"l{i}_shortcut_read"], w[f"l{i}_shortcut_norm"]),
            _Read(w[f"l{i}_mix_read"], w[f"l{i}_input_norm"]),
        )
    return (_Read(w[f"l{i}_res_read"], w[f"l{i}_input_norm"]),)


def _layer_views(w, layer_types: tuple[str, ...], row_invariant: bool) -> tuple[_Layer, ...]:
    """Every block's operands. With `row_invariant`, the GEMMs whose cuBLAS kernel changes with
    the row count at a rank's shapes (its slice of the GDP `in_proj`, the router and
    `fc1_latent_proj`) keep the stored [out, in] weight `invariant_mm` reads; every other GEMM
    operand is the `.t()` view `cublas_mm` reads."""

    def mm(name: str) -> torch.Tensor:
        return w[name] if row_invariant else w[name].t()

    layers = []
    for i, kind in enumerate(layer_types):
        if kind in _GDP_BLOCKS:
            mixer = _Gdp(
                layer=i,
                in_proj=mm(f"l{i}_gdp_in"),
                conv=w[f"l{i}_gdp_conv"],
                neg_a=w[f"l{i}_gdp_neg_a"],
                dt_bias=w[f"l{i}_gdp_dt_bias"],
                norm=w[f"l{i}_gdp_norm"],
                out_proj=w[f"l{i}_gdp_out"].t(),
            )
        elif kind in _ATTENTION_BLOCKS:
            mixer = _Attention(layer=i, qkv=w[f"l{i}_qkv"].t(), o=w[f"l{i}_o"].t())
        else:
            mixer = _Mlp(up=w[f"l{i}_mlp_up"].t(), down=w[f"l{i}_mlp_down"].t())
        if kind in _SHORTCUT_BLOCKS:
            write = _Write(w[f"l{i}_mix_retain"], w[f"l{i}_mix_write"])
            moe = _Moe(
                router=mm(f"l{i}_router"),
                router_bias=w[f"l{i}_router_bias"],
                fc1_latent=mm(f"l{i}_fc1_latent"),
                experts_up=w[f"l{i}_experts_up"],
                experts_down=w[f"l{i}_experts_down"],
                fc2_norm=w[f"l{i}_fc2_norm"],
                fc2_latent=w[f"l{i}_fc2_latent"].t(),
                shared_up=w[f"l{i}_shared_up"].t(),
                shared_down=w[f"l{i}_shared_down"].t(),
            )
            shared_read = _Read(w[f"l{i}_moe_read"], w[f"l{i}_pre_moe_norm"])
            moe_write = _Write(w[f"l{i}_moe_retain"], w[f"l{i}_moe_write"])
        else:
            write = _Write(w[f"l{i}_res_retain"], w[f"l{i}_res_write"])
            moe = shared_read = moe_write = None
        layers.append(
            _Layer(
                mixer=mixer,
                write=write,
                moe=moe,
                shared_read=shared_read,
                moe_write=moe_write,
                next_reads=_block_reads(w, layer_types, i + 1),
            )
        )
    return tuple(layers)


class _Dims(NamedTuple):
    """The text config's numbers the forward reads, at one rank's part, and where the rank's
    GDP in_proj packs its parts."""

    hidden: int
    streams: int
    eps: float
    latent: int
    q_width: int  # the rank's query heads' width
    heads_q: int  # the rank's query heads
    heads_kv: int  # every rank's: the one KV head is whole
    head_dim: int
    gdp_heads: int  # the rank's GDP heads
    gdp_householder: int
    gdp_key: int
    gdp_value: int
    gdp_value_dim: int  # the rank's heads' value width
    gdp_eps: float
    gdp_scale: float
    conv: slice  # the conv input [V0..V2 | K0..K2 | Q] within the rank's in_proj output
    beta: slice  # b0..b2
    decay: slice  # a
    conv_v: slice  # V0..V2 within the conv output
    conv_k: slice  # K0..K2
    conv_q: slice  # Q
    top_k: int
    num_experts: int
    routed_scale: float
    clamp: float


def _dims(cfg, tp: int) -> _Dims:
    """The rank's widths: a 1/tp of the heads of every mixer, each packed segment of the GDP
    in_proj at its heads' block, as weights.py lays them out."""
    n_h, heads = cfg.gdp_num_householder, cfg.gdp_num_heads // tp
    value_dim = heads * cfg.gdp_head_dim
    key_dim = cfg.gdp_num_groups // tp * cfg.gdp_key_head_dim
    conv_dim = n_h * value_dim + (n_h + 1) * key_dim
    beta_start = value_dim + conv_dim
    return _Dims(
        hidden=cfg.hidden_size,
        streams=cfg.wide_residual_num_streams,
        eps=cfg.rms_norm_eps,
        latent=cfg.moe_latent_size,
        q_width=cfg.num_attention_heads // tp * cfg.head_dim,
        heads_q=cfg.num_attention_heads // tp,
        heads_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        gdp_heads=heads,
        gdp_householder=n_h,
        gdp_key=cfg.gdp_key_head_dim,
        gdp_value=cfg.gdp_head_dim,
        gdp_value_dim=value_dim,
        gdp_eps=cfg.gdp_norm_eps,
        gdp_scale=cfg.gdp_key_head_dim**-0.5,
        conv=slice(value_dim, beta_start),
        beta=slice(beta_start, beta_start + n_h * heads),
        decay=slice(beta_start + n_h * heads, beta_start + (n_h + 1) * heads),
        conv_v=slice(0, n_h * value_dim),
        conv_k=slice(n_h * value_dim, n_h * (value_dim + key_dim)),
        conv_q=slice(n_h * (value_dim + key_dim), conv_dim),
        top_k=cfg.num_experts_per_tok,
        num_experts=cfg.n_routed_experts,
        routed_scale=cfg.routed_scaling_factor,
        clamp=cfg.activation_clamp_value,
    )


class Nemotron4NanoPreviewTp4ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        dims = _weights.MODEL_WEIGHTS.dims(self)
        mapping = model_config.mapping
        # The tensor-parallel group: every rank of the node, in rank order. Every rank runs
        # every request, and the ranks' partial sums meet in this group.
        self.rank = mapping.rank
        self.tp_rank = dims.tp_rank
        self.tp_size = dims.tp
        self.tp_group = list(mapping.tp_group)
        self.w = _weights.MODEL_WEIGHTS.declare(dims)

        self.layer_types = dims.layer_types
        self.gdp_layers = tuple(i for i, t in enumerate(dims.layer_types) if t in _GDP_BLOCKS)
        # The trunk's attention layers. With one-model MTP the engine keeps the MTP layer's
        # KV (global layer 44) in a separate draft KV cache manager.
        self.attention_layers = tuple(
            i for i, t in enumerate(dims.layer_types) if t in _ATTENTION_BLOCKS
        )
        self.state_shapes = _rank_state_shapes(model_config)
        self._checked_manager: weakref.ref | None = None

        self.allreduce_workspace: _AllreduceWorkspace | None = None
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None

    def post_load_weights(self) -> None:
        """Build the all-reduce workspace and construct the two targets, now that the
        weights are real.

        Must only be called after meta init is over, never from `__init__`, where the
        shell's containers are still meta. Every rank calls it at the same point of the
        load, which the workspace's collective construction needs.

        Named for the shell hook that drives it. Not an override: that hook lives on
        DecoderModelForCausalLM, and a core derives from DecoderModel, which has no
        such method -- so there is no super() to call here.
        """
        text = self.model_config.pretrained_config.text_config
        # Every all-reduce is [rows, hidden], or a latent MoE's two partials packed into
        # [rows, latent + hidden].
        self.allreduce_workspace = _allreduce_workspace(
            self.model_config, (text.hidden_size, text.moe_latent_size + text.hidden_size)
        )
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def state_views(self, kv_cache_manager: MambaHybridCacheManager) -> _StateViews:
        """The per-layer cache addresses of the manager that prepared this step.

        Read off the manager on every call and never kept: the engine sizes its
        caches with one manager, shuts it down and then builds the one it serves
        with, and a manager drops its views at shutdown so that none outlives the
        pool it points into. Only the manager itself is remembered, weakly, to
        check its layer layout and its per-rank state shapes once.
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
            first = self.gdp_layers[0]
            shapes = (
                tuple(kv_cache_manager.get_ssm_states(first).shape[1:]),
                tuple(kv_cache_manager.get_conv_states(first).shape[1:]),
            )
            if shapes != self.state_shapes:
                raise RuntimeError(
                    f"{type(kv_cache_manager).__name__} holds ssm {shapes[0]} and conv "
                    f"{shapes[1]} per slot; TP rank {self.tp_rank} of {self.tp_size} runs "
                    f"ssm {self.state_shapes[0]} and conv {self.state_shapes[1]}. The manager "
                    "must split the state over the tensor-parallel ranks "
                    "(config_utils.mamba_effective_tp_size)."
                )
            self._checked_manager = weakref.ref(kv_cache_manager)
        return _StateViews(
            kv_layer={i: kv_cache_manager.layer_offsets[i] for i in self.attention_layers},
            ssm_states={i: kv_cache_manager.get_ssm_states(i) for i in self.gdp_layers},
            conv_states={i: kv_cache_manager.get_conv_states(i) for i in self.gdp_layers},
        )

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        Host-side reads only: a decode-only CUDA graph is a per-capture constant
        environment. A mixed batch -- context and generation rows in one step -- goes to
        prefill, so prefill is the general case and decode the specialization that may
        assume it has no context rows. Every rank runs the same rows, so every rank picks
        the same target and issues the same all-reduces.

        The engine's TRTLLM-Gen JIT warmup goes to prefill too. It is one eager step of
        generation rows, run with `trtllm_gen_jit_warmup` raised so that thop_attention
        compiles its generation kernels for every batch size and length up front. Decode
        runs no thop_attention, but the generation rows of mixed steps do, and without the
        warmup each kernel would compile at its first mixed step, mid-serve. The engine
        sets the flag on every step and raises it only there, never under a CUDA graph,
        and on every rank alike.
        """
        if attn_metadata.num_contexts or attn_metadata.trtllm_gen_jit_warmup:
            return self._prefill
        return self._decode

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


def _heads(x: torch.Tensor, cols: slice, *inner: int) -> torch.Tensor:
    """`x[:, cols]` as `[rows, *inner]`, keeping `x`'s own row stride at every row count.

    `x[:, cols].view(rows, *inner)` gives a one-row batch the inner block's size as its row
    stride -- 768, 2304 or 1152 for the GDP q, k and v -- where every larger batch keeps the
    in_proj output's 4632. gdp_recurrent's kernel is specialized on its strides' divisibility by
    16, on which those disagree, so a request alone would run another compiled variant of it,
    whose reductions round otherwise, than the same request inside any batch.
    """
    strides, size = [], 1
    for n in reversed(inner):
        strides.insert(0, size)
        size *= n
    return x.as_strided(
        (x.shape[0], *inner), (x.stride(0), *strides), x.storage_offset() + cols.start
    )


def _check_rows(num_tokens: int, workspace: _AllreduceWorkspace) -> None:
    """A step's all-reduces carry one row per token; the workspace holds `max_num_tokens`,
    and the op does not check what it is handed."""
    if num_tokens > workspace.rows:
        raise RuntimeError(
            f"a step of {num_tokens} tokens exceeds the all-reduce workspace, sized for "
            f"max_num_tokens={workspace.rows} rows"
        )


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: Nemotron4NanoPreviewTp4ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        self._d = _dims(core.model_config.pretrained_config.text_config, core.tp_size)
        self._embed = core.w["embed"]
        self._first_reads = _block_reads(core.w, core.layer_types, 0)
        self._layers = _layer_views(core.w, core.layer_types, row_invariant=False)
        self._ar = core.allreduce_workspace

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
        (`attention`) and of the recurrent-state entries, once per forward.

        Rows are the step's requests, contexts first, the same rows in the same order on
        every rank; tokens are packed row by row. The state entries take the context rows'
        slices: `query_start_loc[: num_contexts + 1]`, `state_slots[:num_contexts]`,
        `has_initial_state`; the generation rows that follow take one token and one slot
        each (draft tokens come only with MTP, which this target does not run).
        """
        state = _state_metadata(md)
        views = self.core.state_views(md.kv_cache_manager)
        num_rows = md.num_seqs
        num_contexts = md.num_contexts
        return dict(
            attention=dict(
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
                attention_window_size=md.max_seq_len,
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
            ),
            # Recurrent state: rows, their slots and where each context row's tokens start.
            num_rows=num_rows,
            num_contexts=num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
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
        """The final normed hidden state of every token: `norm(residual_readout(streams))`."""
        d = self._d
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)

        h = (
            nn.functional.embedding(input_ids, self._embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        _check_rows(h.shape[0], self._ar)
        streams = h.repeat(1, d.streams)
        attn_out = torch.empty(h.shape[0], d.q_width, dtype=h.dtype, device=h.device)

        routed, x = self._streams_step(streams, None, None, self._first_reads)
        for L in self._layers:
            if isinstance(L.mixer, _Gdp):
                update = self._gdp(x, L.mixer, step)
            elif isinstance(L.mixer, _Attention):
                qkv = cublas_mm(x, L.mixer.qkv)
                attn_out = thop_attention(
                    q=qkv,
                    output=attn_out,
                    local_layer_idx=step["kv_layer"][L.mixer.layer],
                    num_heads=d.heads_q,
                    num_kv_heads=d.heads_kv,
                    head_size=d.head_dim,
                    **step["attention"],
                    **_CALL_CONSTANTS,
                )
                update = allreduce(cublas_mm(attn_out, L.mixer.o), self._ar.buffer, self._ar.flags)
            else:
                update = allreduce(
                    cublas_mm(softclamp_relu2(cublas_mm(x, L.mixer.up), d.clamp), L.mixer.down),
                    self._ar.buffer,
                    self._ar.flags,
                )
            if L.moe is None:
                routed, x = self._streams_step(streams, update, L.write, L.next_reads)
                continue
            _, shared = self._streams_step(streams, update, L.write, (L.shared_read,))
            update = self._moe(routed, shared, L.moe)
            routed, x = self._streams_step(streams, update, L.moe_write, L.next_reads)
        return x

    def _streams_step(
        self,
        streams: torch.Tensor,
        update: torch.Tensor | None,
        write: _Write | None,
        reads: tuple[_Read, ...],
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Write `update` into the streams in place (when given), then make one or two normed
        reads: `(None, read)`, or `(first, second)` -- a shortcut block's routed input, then
        its mixer input."""
        second = reads[1] if len(reads) == 2 else None
        _, normed, normed2 = wide_residual(
            streams,
            update,
            None if write is None else write.retention,
            None if write is None else write.write,
            reads[0].factors,
            reads[0].norm,
            self._d.eps,
            None if second is None else second.factors,
            None if second is None else second.norm,
        )
        return (None, normed) if second is None else (normed, normed2)

    def _gdp(self, x: torch.Tensor, g: _Gdp, step: dict) -> torch.Tensor:
        """`Nemotron4GDPMixer` on this rank's heads over a mixed batch: the context rows'
        tokens through the varlen conv and the chunked recurrence, each from zero or from
        its slot; then each generation row's one token through the decode conv and
        recurrence, from its slot. The heads' `out_proj` partials are all-reduced."""
        d = self._d
        num_ctx_tokens, num_contexts, num_rows = (
            step["num_ctx_tokens"],
            step["num_contexts"],
            step["num_rows"],
        )
        ssm_states, conv_states = step["ssm_states"][g.layer], step["conv_states"][g.layer]
        proj = cublas_mm(x, g.in_proj)  # [T, 4632] = [z | V0..V2 K0..K2 Q | b0..b2 | a]
        beta, decay = gdp_gates(proj[:, d.beta], proj[:, d.decay], g.neg_a, g.dt_bias)
        num_tokens = proj.shape[0]
        outs = []
        if num_contexts:
            ctx = slice(0, num_ctx_tokens)
            slots = step["state_slots"][:num_contexts]
            # Channel-last: the in_proj output's conv columns transposed in, and out.
            conv_out = torch.empty(
                num_ctx_tokens, d.conv.stop - d.conv.start, dtype=proj.dtype, device=proj.device
            ).t()
            causal_conv1d_varlen(
                proj[ctx, d.conv].t(),
                g.conv,
                None,
                conv_states,
                step["query_start_loc"][: num_contexts + 1],
                slots,
                step["has_initial_state"],
                True,
                PAD_SLOT_ID,
                conv_out,
            )
            vkq = conv_out.t()
            outs.append(
                gdp_chunk(
                    vkq[:, d.conv_q].view(num_ctx_tokens, d.gdp_heads, d.gdp_key),
                    vkq[:, d.conv_k].view(
                        num_ctx_tokens, d.gdp_householder, d.gdp_heads, d.gdp_key
                    ),
                    vkq[:, d.conv_v].view(
                        num_ctx_tokens, d.gdp_householder, d.gdp_heads, d.gdp_value
                    ),
                    decay[ctx],
                    beta[ctx],
                    ssm_states,
                    slots,
                    step["query_start_loc_long"][: num_contexts + 1],
                    step["has_initial_state"],
                    d.gdp_scale,
                )
            )
        if num_rows > num_contexts:
            gen = slice(num_ctx_tokens, num_tokens)
            slots = step["state_slots"][num_contexts:num_rows]
            num_gen = num_tokens - num_ctx_tokens
            # In place: the conv columns of the in_proj output become the conv output.
            causal_conv1d_update(
                proj[gen, d.conv].unsqueeze(-1),
                conv_states,
                g.conv,
                None,
                True,
                None,
                slots,
                PAD_SLOT_ID,
            )
            vkq = proj[gen, d.conv]
            outs.append(
                gdp_recurrent(
                    vkq[:, d.conv_q].view(num_gen, d.gdp_heads, d.gdp_key),
                    vkq[:, d.conv_k].view(num_gen, d.gdp_householder, d.gdp_heads, d.gdp_key),
                    vkq[:, d.conv_v].view(num_gen, d.gdp_householder, d.gdp_heads, d.gdp_value),
                    decay[gen],
                    beta[gen],
                    ssm_states,
                    slots,
                    d.gdp_scale,
                )
            )
        out = outs[0] if len(outs) == 1 else torch.cat(outs)
        y = gated_rmsnorm(
            out.view(num_tokens, d.gdp_value_dim),
            g.norm,
            None,
            d.gdp_eps,
            z=proj[:, : d.gdp_value_dim],
            group_size=d.gdp_value,
            norm_before_gate=False,
            is_rms_norm=True,
        )
        return allreduce(cublas_mm(y, g.out_proj), self._ar.buffer, self._ar.flags)

    def _moe(self, routed: torch.Tensor, shared: torch.Tensor, m: _Moe) -> torch.Tensor:
        """`Nemotron4MoE`: route at model width (fp32 logits), the routed experts in the
        latent space at this rank's slice of every expert, `fc2_norm` and the projection
        back up, plus the shared expert at this rank's slice.

        The routed experts' latent partial is all-reduced before `fc2_norm`, which is not
        linear, and the shared expert's partial joins it in one call: each element is
        summed on its own, so the two lie end to end in one `[T, latent + hidden]`
        message and come back as two dense blocks.
        """
        d = self._d
        weights, ids = noaux_tc_op(
            cublas_mm(routed, m.router, None, torch.float32),
            m.router_bias,
            1,
            1,
            d.top_k,
            d.routed_scale,
        )
        experts = latent_moe_experts(
            cublas_mm(routed, m.fc1_latent),
            ids,
            weights,
            m.experts_up,
            m.experts_down,
            d.clamp,
            0,
            d.num_experts,
        )
        shared = cublas_mm(softclamp_relu2(cublas_mm(shared, m.shared_up), d.clamp), m.shared_down)
        num_tokens, split = experts.shape[0], experts.numel()
        summed = allreduce(
            torch.cat((experts.view(-1), shared.view(-1))).view(num_tokens, -1),
            self._ar.buffer,
            self._ar.flags,
        ).view(-1)
        experts = flashinfer_rmsnorm(summed[:split].view(num_tokens, d.latent), m.fc2_norm, d.eps)
        return torch.add(
            cublas_mm(experts, m.fc2_latent), summed[split:].view(num_tokens, d.hidden)
        )


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    Batch-invariant: every entry it calls computes a row from that row alone, in an order
    the row count does not change, so each request's logits and the state it writes are
    the same bits whatever batch, padded or not, the step runs in. The GEMMs whose cuBLAS
    kernel changes with the row count at a rank's shapes run through `invariant_mm`;
    attention runs through `invariant_gen_attention`, whose KV split is fixed; the
    all-reduce sums every element by one formula at any row count.
    """

    def __init__(self, core: Nemotron4NanoPreviewTp4ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()
        self._d = _dims(core.model_config.pretrained_config.text_config, core.tp_size)
        self._embed = core.w["embed"]
        self._first_reads = _block_reads(core.w, core.layer_types, 0)
        self._layers = _layer_views(core.w, core.layer_types, row_invariant=True)
        self._ar = core.allreduce_workspace

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
        """Project the prepared metadata onto the explicit batch state of
        invariant_gen_attention (`attention`) and of the recurrent-state entries, once per
        forward.

        Every row is one generation token, the same rows in the same order on every rank.
        Under a CUDA graph the rows include the trailing padding rows, whose `state_slots`
        entry is `PAD_SLOT_ID`; their attention reads the padding dummy's own pages.
        """
        state = _state_metadata(md)
        views = self.core.state_views(md.kv_cache_manager)
        num_rows = md.num_seqs
        return dict(
            attention=dict(
                # Lengths count this step's token and are read on the device only, where the
                # overlap scheduler corrects them.
                sequence_length=md.kv_lens_cuda_runtime,
                kv_cache_block_offsets=md.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
                num_contexts=0,
                tokens_per_block=md.tokens_per_block,
                # Sizes the entry's grid: the cache manager's bound on every length.
                max_seq_len=md.max_seq_len,
            ),
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
        """The final normed hidden state of every row: `norm(residual_readout(streams))`."""
        d = self._d
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)

        h = (
            nn.functional.embedding(input_ids, self._embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        _check_rows(h.shape[0], self._ar)
        streams = h.repeat(1, d.streams)
        attn_out = torch.empty(h.shape[0], d.q_width, dtype=h.dtype, device=h.device)

        routed, x = self._streams_step(streams, None, None, self._first_reads)
        for L in self._layers:
            if isinstance(L.mixer, _Gdp):
                update = self._gdp(x, L.mixer, step)
            elif isinstance(L.mixer, _Attention):
                # Appends the row's K/V to its request's pages, then attends over them. NoPE;
                # q_scaling 1 keeps the softmax scale at head_dim**-0.5 = 1/16.
                attn_out = invariant_gen_attention(
                    q=cublas_mm(x, L.mixer.qkv),
                    output=attn_out,
                    local_layer_idx=step["kv_layer"][L.mixer.layer],
                    tokens_per_seq=1,
                    num_heads=d.heads_q,
                    num_kv_heads=d.heads_kv,
                    head_size=d.head_dim,
                    q_scaling=1.0,
                    **step["attention"],
                )
                update = allreduce(cublas_mm(attn_out, L.mixer.o), self._ar.buffer, self._ar.flags)
            else:
                update = allreduce(
                    cublas_mm(softclamp_relu2(cublas_mm(x, L.mixer.up), d.clamp), L.mixer.down),
                    self._ar.buffer,
                    self._ar.flags,
                )
            if L.moe is None:
                routed, x = self._streams_step(streams, update, L.write, L.next_reads)
                continue
            _, shared = self._streams_step(streams, update, L.write, (L.shared_read,))
            update = self._moe(routed, shared, L.moe)
            routed, x = self._streams_step(streams, update, L.moe_write, L.next_reads)
        return x

    def _streams_step(
        self,
        streams: torch.Tensor,
        update: torch.Tensor | None,
        write: _Write | None,
        reads: tuple[_Read, ...],
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Write `update` into the streams in place (when given), then make one or two normed
        reads: `(None, read)`, or `(first, second)` -- a shortcut block's routed input, then
        its mixer input."""
        second = reads[1] if len(reads) == 2 else None
        _, normed, normed2 = wide_residual(
            streams,
            update,
            None if write is None else write.retention,
            None if write is None else write.write,
            reads[0].factors,
            reads[0].norm,
            self._d.eps,
            None if second is None else second.factors,
            None if second is None else second.norm,
        )
        return (None, normed) if second is None else (normed, normed2)

    def _gdp(self, x: torch.Tensor, g: _Gdp, step: dict) -> torch.Tensor:
        """`Nemotron4GDPMixer` on this rank's heads for one token per row, each continuing
        from its slot; padding rows (`PAD_SLOT_ID`) read zeros and write nothing. The heads'
        `out_proj` partials are all-reduced."""
        d = self._d
        slots = step["state_slots"]
        proj = invariant_mm(x, g.in_proj)  # [B, 4632] = [z | V0..V2 K0..K2 Q | b0..b2 | a]
        num_rows = proj.shape[0]
        # In place: the conv columns of the in_proj output become the conv output.
        causal_conv1d_update(
            proj[:, d.conv].unsqueeze(-1),
            step["conv_states"][g.layer],
            g.conv,
            None,
            True,
            None,
            slots,
            PAD_SLOT_ID,
        )
        vkq = proj[:, d.conv]
        beta, decay = gdp_gates(proj[:, d.beta], proj[:, d.decay], g.neg_a, g.dt_bias)
        out = gdp_recurrent(
            _heads(vkq, d.conv_q, d.gdp_heads, d.gdp_key),
            _heads(vkq, d.conv_k, d.gdp_householder, d.gdp_heads, d.gdp_key),
            _heads(vkq, d.conv_v, d.gdp_householder, d.gdp_heads, d.gdp_value),
            decay,
            beta,
            step["ssm_states"][g.layer],
            slots,
            d.gdp_scale,
        )
        y = gated_rmsnorm(
            out.view(num_rows, d.gdp_value_dim),
            g.norm,
            None,
            d.gdp_eps,
            z=proj[:, : d.gdp_value_dim],
            group_size=d.gdp_value,
            norm_before_gate=False,
            is_rms_norm=True,
        )
        return allreduce(cublas_mm(y, g.out_proj), self._ar.buffer, self._ar.flags)

    def _moe(self, routed: torch.Tensor, shared: torch.Tensor, m: _Moe) -> torch.Tensor:
        """`Nemotron4MoE`: route at model width (fp32 logits), the routed experts in the
        latent space at this rank's slice of every expert, `fc2_norm` and the projection
        back up, plus the shared expert at this rank's slice.

        The routed experts' latent partial is all-reduced before `fc2_norm`, which is not
        linear, and the shared expert's partial joins it in one call: each element is
        summed on its own, so the two lie end to end in one `[B, latent + hidden]`
        message and come back as two dense blocks.
        """
        d = self._d
        weights, ids = noaux_tc_op(
            invariant_mm(routed, m.router, torch.float32),
            m.router_bias,
            1,
            1,
            d.top_k,
            d.routed_scale,
        )
        experts = latent_moe_experts(
            invariant_mm(routed, m.fc1_latent),
            ids,
            weights,
            m.experts_up,
            m.experts_down,
            d.clamp,
            0,
            d.num_experts,
        )
        shared = cublas_mm(softclamp_relu2(cublas_mm(shared, m.shared_up), d.clamp), m.shared_down)
        num_rows, split = experts.shape[0], experts.numel()
        summed = allreduce(
            torch.cat((experts.view(-1), shared.view(-1))).view(num_rows, -1),
            self._ar.buffer,
            self._ar.flags,
        ).view(-1)
        experts = flashinfer_rmsnorm(summed[:split].view(num_rows, d.latent), m.fc2_norm, d.eps)
        return torch.add(cublas_mm(experts, m.fc2_latent), summed[split:].view(num_rows, d.hidden))


class ModelingV2Nemotron4NanoPreviewSm103Tp4(
    DecoderModelForCausalLM[Nemotron4NanoPreviewTp4ModelingV2, PretrainedConfig]
):
    #: Built by the engine for every step (`resolve_mamba_metadata_cls`).
    mamba_metadata_cls = Nemotron4StateMetadata

    def __init__(self, model_config: ModelConfig):
        # The composite config keeps the text model's widths under text_config.
        cfg = model_config.pretrained_config
        text = cfg.text_config
        # The engine reads the depth off the composite config when it sizes its caches, but
        # text_config derives it (a property over layer_types) and ModelConfig's mirror of
        # text_config onto the composite copies stored fields only.
        cfg.num_hidden_layers = text.num_hidden_layers
        super().__init__(
            Nemotron4NanoPreviewTp4ModelingV2(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # Without attention DP the base class builds lm_head split over the vocabulary.
        # Every rank here forms every logit itself from the same hidden state, so lm_head
        # is whole on every rank, built as the base class builds it under attention DP.
        self.lm_head = LMHead(text.vocab_size, text.hidden_size, dtype=model_config.torch_dtype)
        self.epilogue = [self.lm_head]
        spec_config = getattr(model_config, "spec_config", None)
        if spec_config is not None:
            self.mamba_metadata_cls = Nemotron4StateMetadata.for_draft_len(
                spec_config.max_draft_len
            )

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
        """fp32 logits over the whole vocabulary, one row per request (every token with
        `return_context_logits`), the same bits on every rank.

        HF's lm_head is an fp32 GEMM over the bf16 hidden state and weight
        (`Nemotron4ForCausalLM.forward`); `cublas_mm` with an fp32 output computes the same
        products, exact in fp32, and keeps the fp32 sum, where the default logits processor
        would round the logits to bf16 first.
        """
        if spec_metadata is not None:
            raise NotImplementedError(
                "nemotron4-nano-preview / sm_103 / tp4 has no MTP draft forward yet"
            )
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
        return cublas_mm(hidden, self.lm_head.weight.t(), None, torch.float32)
