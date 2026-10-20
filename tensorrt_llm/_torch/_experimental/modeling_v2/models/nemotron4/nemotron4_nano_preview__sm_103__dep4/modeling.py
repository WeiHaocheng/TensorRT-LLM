# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: nemotron4-nano-preview (text) / sm_103 / dep4 — self-contained modeling code.

The text model of the Nemotron4ForConditionalGeneration checkpoint: 44 blocks
of hidden 3072 on three wide-residual streams -- 12 GDP mixers alone, 4 dense
relu2 MLPs, 8 attention and 20 GDP mixers each beside a latent MoE (512 routed
experts at top-10 in a 768-wide latent space, plus a shared expert) -- read out
of the streams, normed, and projected by an fp32-output lm_head. Every weight
is the checkpoint's bf16, laid out for the entries at load.

`dep4` is attention DP over the 4 GPUs of one node with the routed experts
split four ways. Every rank holds the whole model except the routed experts, of
which it holds one contiguous window of 128 of every MoE block's 512, and runs
its own requests through the embedding, every mixer, the dense MLPs, the shared
experts, the readout and the lm_head. Only the routed MoE crosses ranks, once
per MoE block:

1. the router and `fc1_latent_proj` run on the rank's own tokens;
2. each token's latent, expert ids and routing weights travel as one packed
   byte row, all-gathered in rank order;
3. `latent_moe_experts` runs the rank's window over every rank's tokens and
   returns the window's partial sums unrounded, in fp32;
4. the partial sums are reduce-scattered in fp32, so every token's ten experts
   are added before the one rounding to bf16 HF makes;
5. `fc2_norm` and `fc2_latent_proj` run on the rank's own tokens.

Speculative decoding. Under one-model MTP-Eagle (the one mode routing admits)
the checkpoint's MTP layer -- global layer 44, declared and loaded only then
(weights.py) -- drafts every depth, driven by the engine's worker on each rank
for its own requests. Its MoE crosses the ranks as the trunk's does, with its
own window of the MTP layer's experts; each draft step's collectives are sized
by the counts the worker hands that step, not by the trunk's. Its KV lives in
the rank's one cache manager beside the trunk's, at that manager's index for
layer 44. A generation row carries 1 + D tokens: its last accepted token and D
drafts. The trunk verifies them in one step: the GDP mixers run each row's
tokens from its slot through the verify entries, which leave the slot alone and
store the state after every token; the worker's commit (`update_mamba_states`)
promotes the accepted one into the slot, so a rejected draft never reaches it.

The streams are `[T, 3 * 3072]` bf16, the embedding repeated three times. One
`wide_residual` call sits between every two branches: it writes the finished
branch's update into the streams and makes the next branch's reads, each with
its RMSNorm -- 73 calls a forward. A shortcut block reads its routed input and
its mixer input in the call before its mixer, from the same streams, and the
shared expert's input in the call after it, which is HF's order.

Runtime state, per rank. The engine recognizes the checkpoint as a hybrid
(`config_utils.is_nemotron4`) and gives every rank its own cache manager:
per request slot, the 32 GDP mixers' recurrent state ([24, 64, 128] fp32 as
[heads, value, key], key innermost -- all 24 heads, since attention DP splits
requests, not heads) and conv state ([16896, 3] bf16, oldest input first), and
paged bf16 KV for the 8 attention blocks (1 KV head of 256).
`Nemotron4StateMetadata` turns the engine's per-step slot list into the slots
the state entries read; each target's `step_args` projects the prepared
metadata onto the explicit arguments its forward passes, including every
rank's token count, which the routed MoE's collectives are sized by.

Batch invariance. A decode step computes every row from that row alone, in an
order that depends neither on how many rows share the rank's step nor on what
the other ranks run beside it, so a request's logits and the state it writes
are bitwise the same alone, in any eager batch and in a padded CUDA-graph
batch, whether the other ranks are idle, decoding, verifying or prefilling. An
MTP verify is such a step with 1 + D tokens a row, and its token j gets the
bits that token gets as a decode step's one token. On its own rank, cuBLAS
picks its kernel by the row count, and on six of the GEMM shapes the kernels it
picks split K differently: decode and verify run those through `invariant_mm`,
and generation attention through `invariant_gen_attention`, whose KV split is
fixed and whose kernels compute a verify token with the arithmetic that decodes
it. Across
ranks, a token's routed experts see every rank's tokens: `latent_moe_experts`
gives a token the same bits whatever rows share its window call, and the
reduce-scatter adds a token's four fp32 partials in one fixed order whatever
the ranks' counts, padded or ragged -- measured as NCCL's ring chain that ends
at the token's own rank, which is NCCL's choice on this node rather than a
promise of the op. Prefill and mixed steps keep cuBLAS and `thop_attention`;
CUDA graphs do not change their batches. So do the MTP layer's draft steps: a
draft's bits change how many drafts the trunk accepts, never the tokens it
emits.
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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_update import (
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_varlen import (
    causal_conv1d_varlen,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_verify import (
    causal_conv1d_verify,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_chunk import gdp_chunk
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_gates import gdp_gates
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_recurrent import gdp_recurrent
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_verify import gdp_verify
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.invariant_mm import invariant_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
    latent_moe_experts,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.gated_rmsnorm import gated_rmsnorm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.wide_residual import (
    wide_residual,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.speculative import get_spec_worker
from tensorrt_llm._utils import prefer_pinned

from . import weights as _weights

#: Block types by what they run: a GDP mixer keeps recurrent and conv state, an
#: attention mixer keeps paged KV, a dense MLP keeps nothing; the shortcut blocks
#: run their mixer beside a latent MoE.
_GDP_BLOCKS = ("linear_attention", "shortcut_moe_gdp")
_ATTENTION_BLOCKS = ("shortcut_moe_attn",)
_SHORTCUT_BLOCKS = ("shortcut_moe_attn", "shortcut_moe_gdp")

#: What one routed token carries across the ranks, in this order in its packed byte row:
#: its latent (`fc1_latent_proj`'s output), its expert ids and its routing weights, each in
#: the dtype `latent_moe_experts` takes it in.
_DISPATCH_DTYPES = (torch.bfloat16, torch.int32, torch.float32)

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
    attention-DP rank.

    `state_slots` (int32 [rows], contexts first): each row's state slot, and
    `PAD_SLOT_ID` on every placeholder row. Two kinds of generation row stand in
    for no request, and both hold a real slot:

    * CUDA-graph padding rows. The ranks pad to one common graph batch, so a
      rank pads up to the largest rank's batch, not its own; the engine gives
      every padding row the padding dummy's own slot, repeated, so several rows
      of one step would write one slot.
    * The attention-DP dummy (`ATTENTION_DP_DUMMY_REQUEST_ID`): a rank with no
      request of its own runs one so that its collectives meet the other
      ranks'. A new one is made every step and handed a recycled slot that is
      never cleared, so it would read another request's stale state.

    The state entries read zeros for `PAD_SLOT_ID` and write nothing. A context
    row is never a placeholder -- the context entries have no padding slot --
    and does not need to be: a context dummy, which the engine makes only in
    disaggregated serving (with a KV cache transceiver, the dummy follows the
    group's phase), starts from zero by `has_initial_states` and writes only its
    own slot.

    Fresh slots. A slot handed to a new request still holds the previous
    owner's state: the cache managers never clear it. A context row honors
    that through `has_initial_states` (false: the entries do not read the
    slot). A generation row always reads its slot, and the runner turns a final
    one-token context chunk into a generation row whenever the step runs as a
    CUDA graph -- under attention DP, when every rank's batch can
    (`_make_single_token_context_graph_batch`); otherwise it stays a context row.
    For a fresh one-token prompt that generation row has no cached token -- the
    only real generation row without one -- so `prepare` zeroes its recurrent
    and conv state in every GDP layer; a chunked prompt's promoted last token
    keeps its state.

    `prepare` runs eagerly before every forward, CUDA-graph replays included
    (`AttentionMetadata.prepare`), on the forward's stream: it fills
    `state_slots` in place, so a captured graph reads the current step's slots,
    and the zeroing lands before the step that reads the slot.

    Speculative decoding. A padding row's request id is the padding dummy's,
    `CUDA_GRAPH_DUMMY_REQUEST_ID` less the step's draft length, which is at most
    the target's `max_draft_len`; the attention-DP dummy keeps its own id and
    carries the step's drafts like any generation row. The engine builds this
    metadata itself and hands it no speculative config, and the default hybrid
    manager does not state the draft length either, so the shell binds the
    target's own (`for_draft_len`). `state_indices` keeps the engine's slots,
    both dummies' included: the worker's commit (`update_mamba_states`) writes
    through it and does not skip a negative slot, and the slots it writes for the
    dummies hold no request's state.
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
        placeholders = self._placeholder_rows(attn_metadata, num_rows)
        self.state_slots[:num_rows].copy_(self.state_indices[:num_rows])
        for start, stop in _row_runs(placeholders):
            self.state_slots[start:stop].fill_(PAD_SLOT_ID)
        fresh_rows = self._fresh_generation_rows(attn_metadata, num_rows, placeholders)
        if fresh_rows:
            self._zero_slots(attn_metadata.kv_cache_manager, fresh_rows)

    def _placeholder_rows(self, attn_metadata: AttentionMetadata, num_rows: int) -> list[int]:
        """The generation rows that stand in for no request, ascending.

        The CUDA-graph padding rows carry the padding dummy's request id,
        `CUDA_GRAPH_DUMMY_REQUEST_ID` less the step's draft length: an id at or
        above `CUDA_GRAPH_DUMMY_REQUEST_ID - max_draft_len` is a padding row's.
        The attention-DP dummy carries `ATTENTION_DP_DUMMY_REQUEST_ID`, which no
        real request is given (the engine numbers requests from
        `max_batch_size` up).
        """
        request_ids = attn_metadata.request_ids
        if request_ids is None:
            return []
        first_padding_id = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len
        return [
            row
            for row in range(attn_metadata.num_contexts, num_rows)
            if request_ids[row] == ATTENTION_DP_DUMMY_REQUEST_ID
            or request_ids[row] >= first_padding_id
        ]

    @staticmethod
    def _fresh_generation_rows(
        attn_metadata: AttentionMetadata, num_rows: int, placeholders: list[int]
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
        skip = set(placeholders)
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


class _StateViews(NamedTuple):
    """One cache manager's per-layer addresses.

    `kv_layer` maps an attention layer to its index in the paged KV pools. The
    hybrid managers number every stateful layer, recurrent and attention alike,
    so the eight attention layers are not 0..7. `ssm_states` / `conv_states` map
    a GDP layer to its pool view: [slots, 24, 64, 128] fp32 and [slots, 16896,
    3] bf16, any slot stride, inner dims dense.

    `intermediate_ssm` / `intermediate_conv` map a GDP layer to the per-position
    buffers an MTP verify writes ([max_batch, 1 + D, 24, 64, 128] fp32 and
    [max_batch, 1 + D, 16896, 3] bf16; row g is the step's generation row g), or
    to None when the manager keeps none (no speculative decoding).
    """

    kv_layer: dict[int, int]
    ssm_states: dict[int, torch.Tensor]
    conv_states: dict[int, torch.Tensor]
    intermediate_ssm: dict[int, torch.Tensor | None]
    intermediate_conv: dict[int, torch.Tensor | None]


# --- Per-layer weight views ---------------------------------------------------------------------
#
# Built once per target off the loaded weights: each GEMM operand as the `.t()` view cublas_mm
# reads, each wide-residual coefficient as the bf16 [3] vector wide_residual reads. The decode
# target runs the operands marked "invariant_mm" through that entry instead, which reads the
# checkpoint's own [out, in] weight; its views keep that layout for them.


class _Read(NamedTuple):
    """One read of the streams and the RMSNorm that follows it."""

    factors: torch.Tensor  # f [streams]
    norm: torch.Tensor  # [hidden]


class _Write(NamedTuple):
    """One branch's write into the streams."""

    retention: torch.Tensor  # r [streams]
    write: torch.Tensor  # w [streams]


class _Gdp(NamedTuple):
    """A GDP mixer (`Nemotron4GDPMixer`)."""

    layer: int  # global layer index: its state pools are step["ssm_states"/"conv_states"][layer]
    in_proj: torch.Tensor  # [hidden, 18528]
    conv: torch.Tensor  # [16896, 4]
    neg_a: torch.Tensor  # -exp(A_log) [24] fp32
    dt_bias: torch.Tensor  # [24]
    norm: torch.Tensor  # [1536]
    out_proj: torch.Tensor  # [1536, hidden]


class _Attention(NamedTuple):
    """An attention mixer (`Nemotron4Attention`): fused qkv, NoPE, one KV head."""

    layer: int  # global layer index: its KV pool index is step["kv_layer"][layer]
    qkv: torch.Tensor  # [hidden, 4608]
    o: torch.Tensor  # [4096, hidden]; invariant_mm: [hidden, 4096]


class _Mlp(NamedTuple):
    """A dense soft-clamped relu2 MLP (`Nemotron4MLP`)."""

    up: torch.Tensor  # [hidden, 12288]; invariant_mm: [12288, hidden]
    down: torch.Tensor  # [12288, hidden]; invariant_mm: [hidden, 12288]


class _Moe(NamedTuple):
    """A latent MoE (`Nemotron4MoE`): router, this rank's window of the routed latent experts,
    shared expert."""

    router: torch.Tensor  # [hidden, 512]; invariant_mm: [512, hidden]
    router_bias: torch.Tensor  # [512] fp32
    fc1_latent: torch.Tensor  # [hidden, 768]; invariant_mm: [768, hidden]
    experts_up: torch.Tensor  # [128, 4096, 768], this rank's window as stored
    experts_down: torch.Tensor  # [128, 768, 4096], this rank's window as stored
    fc2_norm: torch.Tensor  # [768]
    fc2_latent: torch.Tensor  # [768, hidden]
    shared_up: torch.Tensor  # [hidden, 4096]
    shared_down: torch.Tensor  # [4096, hidden]; invariant_mm: [hidden, 4096]


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
    the row count (attention `o_proj`, the dense MLP, the router, `fc1_latent_proj` and the
    shared expert's down projection) keep the stored [out, in] weight `invariant_mm` reads;
    every other GEMM operand is the `.t()` view `cublas_mm` reads."""

    def mm(name: str) -> torch.Tensor:
        return w[name] if row_invariant else w[name].t()

    layers = []
    for i, kind in enumerate(layer_types):
        if kind in _GDP_BLOCKS:
            mixer = _Gdp(
                layer=i,
                in_proj=w[f"l{i}_gdp_in"].t(),
                conv=w[f"l{i}_gdp_conv"],
                neg_a=w[f"l{i}_gdp_neg_a"],
                dt_bias=w[f"l{i}_gdp_dt_bias"],
                norm=w[f"l{i}_gdp_norm"],
                out_proj=w[f"l{i}_gdp_out"].t(),
            )
        elif kind in _ATTENTION_BLOCKS:
            mixer = _Attention(layer=i, qkv=w[f"l{i}_qkv"].t(), o=mm(f"l{i}_o"))
        else:
            mixer = _Mlp(up=mm(f"l{i}_mlp_up"), down=mm(f"l{i}_mlp_down"))
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
                shared_down=mm(f"l{i}_shared_down"),
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


class _Mtp(NamedTuple):
    """The MTP layer (`Nemotron4MultiTokenPredictionLayer`): its front end, a shortcut attention
    + MoE block on an ordinary residual, and its final norm. Its output head is the trunk's
    lm_head."""

    attention: _Attention  # global layer `num_layers`; its KV lives in the rank's one manager
    enorm: torch.Tensor  # [hidden], the next token's embedding
    hnorm: torch.Tensor  # [hidden], the previous step's hidden state
    eh: torch.Tensor  # [2 * hidden, hidden]: the embedding half first
    shortcut_norm: torch.Tensor  # [hidden], the routed input
    input_norm: torch.Tensor  # [hidden], the attention input
    pre_moe_norm: torch.Tensor  # [hidden], the shared expert's input
    moe: _Moe  # this rank's window of the layer's routed experts
    final_norm: torch.Tensor  # [hidden]


def _mtp_view(w, i: int) -> _Mtp:
    """The MTP layer's operands, global layer `i`, laid out as PrefillTarget's blocks are: every
    GEMM operand the `.t()` view `cublas_mm` reads, in both targets."""
    return _Mtp(
        attention=_Attention(layer=i, qkv=w[f"l{i}_qkv"].t(), o=w[f"l{i}_o"].t()),
        enorm=w[f"l{i}_enorm"],
        hnorm=w[f"l{i}_hnorm"],
        eh=w[f"l{i}_eh"].t(),
        shortcut_norm=w[f"l{i}_shortcut_norm"],
        input_norm=w[f"l{i}_input_norm"],
        pre_moe_norm=w[f"l{i}_pre_moe_norm"],
        moe=_Moe(
            router=w[f"l{i}_router"].t(),
            router_bias=w[f"l{i}_router_bias"],
            fc1_latent=w[f"l{i}_fc1_latent"].t(),
            experts_up=w[f"l{i}_experts_up"],
            experts_down=w[f"l{i}_experts_down"],
            fc2_norm=w[f"l{i}_fc2_norm"],
            fc2_latent=w[f"l{i}_fc2_latent"].t(),
            shared_up=w[f"l{i}_shared_up"].t(),
            shared_down=w[f"l{i}_shared_down"].t(),
        ),
        final_norm=w[f"l{i}_mtp_norm"],
    )


class _Dims(NamedTuple):
    """The text config's numbers the forward reads, where the GDP in_proj packs its parts, and
    where a routed token's packed dispatch row keeps its fields."""

    hidden: int
    streams: int
    eps: float
    q_width: int
    heads_q: int
    heads_kv: int
    head_dim: int
    gdp_heads: int
    gdp_householder: int
    gdp_key: int
    gdp_value: int
    gdp_value_dim: int
    gdp_eps: float
    gdp_scale: float
    conv: slice  # the conv input [V0..V2 | K0..K2 | Q] within the in_proj output
    beta: slice  # b0..b2
    decay: slice  # a
    conv_v: slice  # V0..V2 within the conv output
    conv_k: slice  # K0..K2
    conv_q: slice  # Q
    top_k: int
    routed_scale: float
    clamp: float
    dispatch_latent: slice  # the latent's bf16 bytes within a packed dispatch row
    dispatch_ids: slice  # the expert ids' int32 bytes
    dispatch_weights: slice  # the routing weights' fp32 bytes


def _dims(cfg) -> _Dims:
    n_h, heads = cfg.gdp_num_householder, cfg.gdp_num_heads
    value_dim = heads * cfg.gdp_head_dim
    key_dim = cfg.gdp_num_groups * cfg.gdp_key_head_dim
    conv_dim = n_h * value_dim + (n_h + 1) * key_dim
    beta_start = value_dim + conv_dim
    latent_dtype, ids_dtype, weights_dtype = _DISPATCH_DTYPES
    ids_start = cfg.moe_latent_size * latent_dtype.itemsize
    weights_start = ids_start + cfg.num_experts_per_tok * ids_dtype.itemsize
    return _Dims(
        hidden=cfg.hidden_size,
        streams=cfg.wide_residual_num_streams,
        eps=cfg.rms_norm_eps,
        q_width=cfg.num_attention_heads * cfg.head_dim,
        heads_q=cfg.num_attention_heads,
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
        routed_scale=cfg.routed_scaling_factor,
        clamp=cfg.activation_clamp_value,
        dispatch_latent=slice(0, ids_start),
        dispatch_ids=slice(ids_start, weights_start),
        dispatch_weights=slice(
            weights_start, weights_start + cfg.num_experts_per_tok * weights_dtype.itemsize
        ),
    )


def _moe_sizes(all_rank_num_tokens) -> list[int] | None:
    """The per-rank row counts the routed MoE's two collectives take: None when every rank
    holds the same count (the padded form: every CUDA-graph step, where the ranks run one
    common batch), else every rank's count in rank order (the ragged form: eager steps, where
    ranks prefill independently)."""
    counts = [int(n) for n in all_rank_num_tokens]
    return None if len(set(counts)) == 1 else counts


class Nemotron4NanoPreviewDep4ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        d = _weights.MODEL_WEIGHTS.dims(self)
        mapping = model_config.mapping
        # The attention-DP group: every rank, in rank order. Each rank runs its own
        # requests, and every routed MoE block's collectives span the whole group.
        self.rank = mapping.rank
        self.dp_size = mapping.world_size
        self.dp_group = list(range(mapping.world_size))
        # This rank's window of every MoE block's routed experts:
        # global ids [expert_offset, expert_offset + local_experts).
        self.expert_offset = d.expert_offset
        self.local_experts = d.local_experts
        self.w = _weights.MODEL_WEIGHTS.declare(d)

        self.layer_types = d.layer_types
        self.gdp_layers = tuple(i for i, t in enumerate(d.layer_types) if t in _GDP_BLOCKS)
        # The trunk's attention layers. Under one-model MTP the engine keeps the MTP
        # layer's KV (global layer 44) in the same manager under attention DP; the MTP
        # layer reads its own index off the manager on every draft step.
        self.attention_layers = tuple(
            i for i, t in enumerate(d.layer_types) if t in _ATTENTION_BLOCKS
        )
        # The MTP layer, global layer `num_layers`, exists only under speculative
        # decoding (weights.py declares it then).
        self.mtp_layer = len(d.layer_types) if d.mtp_layers else None
        # Tokens a generation row carries: its last token, and under one-model MTP the
        # drafts it verifies.
        spec_config = getattr(model_config, "spec_config", None)
        self.tokens_per_gen_step = 1 if spec_config is None else spec_config.tokens_per_gen_step
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
            intermediate_ssm={
                i: kv_cache_manager.get_intermediate_ssm_states(i) for i in self.gdp_layers
            },
            intermediate_conv={
                i: kv_cache_manager.get_intermediate_conv_states(i) for i in self.gdp_layers
            },
        )

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        Host-side reads only: a decode-only CUDA graph is a per-capture constant
        environment. A mixed batch -- context and generation rows in one step -- goes to
        prefill, so prefill is the general case and decode the specialization that may
        assume it has no context rows.

        The engine's TRTLLM-Gen JIT warmup goes to prefill too. It is one eager step of
        generation rows, run with `trtllm_gen_jit_warmup` raised so that thop_attention
        compiles its generation kernels for every batch size and length up front. Decode
        runs no thop_attention, but the generation rows of mixed steps do, and without the
        warmup each kernel would compile at its first mixed step, mid-serve. The engine
        sets the flag on every step and raises it only there, never under a CUDA graph;
        every rank runs that step together, each with its own generation request.

        Under attention DP the ranks of one step may run different targets -- a rank
        prefilling beside ranks that decode, or run the attention-DP dummy -- so both
        targets make the same collectives in the same order: one all-gather and one
        reduce-scatter per MoE block, sized by the step's `all_rank_num_tokens`. The
        MTP layer's draft steps are routed the same way, per call, and its one MoE block
        makes the same pair in both targets.
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


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: Nemotron4NanoPreviewDep4ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist. The MTP layer's draft
        # steps read another projection, off the metadata the worker rewrites between them.
        self._contract_pending = step_contract_enabled()
        self._mtp_contract_pending = step_contract_enabled()
        self._d = _dims(core.model_config.pretrained_config.text_config)
        self._embed = core.w["embed"]
        self._first_reads = _block_reads(core.w, core.layer_types, 0)
        self._layers = _layer_views(core.w, core.layer_types, row_invariant=False)
        self._mtp = None if core.mtp_layer is None else _mtp_view(core.w, core.mtp_layer)
        self._tokens_per_gen_step = core.tokens_per_gen_step

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

    def _check_mtp_contract(self, md, all_rank_num_tokens) -> None:
        """The same opt-in fail-fast for the MTP layer's draft steps, whose metadata the
        worker rewrites between them and whose collectives the worker sizes."""
        if not self._mtp_contract_pending:
            return
        try:
            self._mtp_step_args(md, all_rank_num_tokens)
        except AttributeError as exc:
            raise AssertionError(f"draft-step metadata surface drifted: {exc}") from exc
        self._mtp_contract_pending = False

    def _attention_args(self, md: TrtllmAttentionMetadata) -> dict:
        """thop_attention's batch state, off the rank's one cache manager, for the trunk's
        steps and the MTP layer's draft steps alike.

        A generation row may carry 1 + D tokens (an MTP verify, or the MTP layer's first
        draft step). thop sizes the XQA workspace as if each of `max_num_sequences` rows
        carried one token, so `max_num_sequences` counts every token a row may carry.
        """
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
            attention_window_size=md.max_seq_len,
            num_contexts=md.num_contexts,
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
            max_num_sequences=md.max_num_sequences * self._tokens_per_gen_step,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        )

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto the explicit batch state of thop_attention
        (`attention`), of the recurrent-state entries and of the attention-DP group, once
        per forward.

        Rows are this rank's requests, contexts first; tokens are packed row by row. The
        state entries take the context rows' slices: `query_start_loc[: num_contexts + 1]`,
        `state_slots[:num_contexts]`, `has_initial_state`. The generation rows that follow
        take one slot each, and every one of them the same number of tokens: one, or 1 + D
        under MTP, whose states go to the `intermediate_*` buffers. `query_start_loc` counts
        one token per generation row, so only its first `num_contexts + 1` entries hold
        under MTP. `moe_sizes` is what the routed MoE's collectives take: every rank's token
        count this step, in rank order, ragged here since ranks prefill independently -- or
        None when they agree.
        """
        state = _state_metadata(md)
        views = self.core.state_views(md.kv_cache_manager)
        num_rows = md.num_seqs
        num_contexts = md.num_contexts
        return dict(
            attention=self._attention_args(md),
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
            intermediate_ssm=views.intermediate_ssm,
            intermediate_conv=views.intermediate_conv,
            # The attention-DP group: how many rows every rank feeds the routed MoE.
            moe_sizes=_moe_sizes(md.all_rank_num_tokens),
        )

    def _mtp_step_args(self, md: TrtllmAttentionMetadata, all_rank_num_tokens) -> dict:
        """The MTP layer's batch state, read off the metadata the worker hands each draft
        step: every call, since the worker rewrites that metadata between draft steps.

        Its KV is global layer `num_layers` in the rank's one cache manager, at that
        manager's own index. Its MoE's collectives take the counts the worker hands the
        step -- the step's tokens for the first, one row per request after it -- never the
        metadata's `all_rank_num_tokens`, which stays the trunk's through the draft loop.
        """
        return dict(
            attention=self._attention_args(md),
            kv_layer=md.kv_cache_manager.layer_offsets[self._mtp.attention.layer],
            moe_sizes=_moe_sizes(all_rank_num_tokens),
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
                update = cublas_mm(attn_out, L.mixer.o)
            else:
                update = cublas_mm(softclamp_relu2(cublas_mm(x, L.mixer.up), d.clamp), L.mixer.down)
            if L.moe is None:
                routed, x = self._streams_step(streams, update, L.write, L.next_reads)
                continue
            _, shared = self._streams_step(streams, update, L.write, (L.shared_read,))
            update = self._moe(routed, shared, L.moe, step["moe_sizes"])
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
        """`Nemotron4GDPMixer` over a mixed batch: the context rows' tokens through the
        varlen conv and the chunked recurrence, each from zero or from its slot; then the
        generation rows' tokens, each row from its slot (`_gdp_generation`)."""
        d = self._d
        num_ctx_tokens, num_contexts, num_rows = (
            step["num_ctx_tokens"],
            step["num_contexts"],
            step["num_rows"],
        )
        ssm_states, conv_states = step["ssm_states"][g.layer], step["conv_states"][g.layer]
        proj = cublas_mm(x, g.in_proj)  # [T, 18528] = [z | V0..V2 K0..K2 Q | b0..b2 | a]
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
            outs.append(
                self._gdp_generation(
                    proj[gen],
                    beta[gen],
                    decay[gen],
                    g,
                    step["state_slots"][num_contexts:num_rows],
                    step,
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
        return cublas_mm(y, g.out_proj)

    def _gdp_generation(
        self,
        proj: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        g: _Gdp,
        slots: torch.Tensor,
        step: dict,
    ) -> torch.Tensor:
        """The GDP recurrence of the generation rows, request-major, each row from its slot;
        placeholder rows (`PAD_SLOT_ID`) read zeros and write nothing.

        Every row carries the same number of tokens, read off host shapes: the engine pads or
        truncates each row's drafts to the step's draft length (`resolve_draft_len`), the
        attention-DP dummy's included, and a step with none skips the worker's commit. One
        token goes through the decode entries, which update the slot in place. The 1 + D
        tokens of an MTP verify go through the verify entries: they run the row's tokens from
        its slot without writing it, and store the state after each token in the manager's
        per-position buffers, row g for generation row g. The worker's commit
        (`update_mamba_states`) then promotes the last accepted token's state into the slot,
        so a rejected draft never reaches it.
        """
        d = self._d
        num_rows, num_tokens = slots.shape[0], proj.shape[0]
        tokens = num_tokens // num_rows
        ssm_states, conv_states = step["ssm_states"][g.layer], step["conv_states"][g.layer]
        # In place: the conv columns of the in_proj output become the conv output.
        vkq = proj[:, d.conv]
        if tokens == 1:
            causal_conv1d_update(
                vkq.unsqueeze(-1), conv_states, g.conv, None, True, None, slots, PAD_SLOT_ID
            )
            return gdp_recurrent(
                vkq[:, d.conv_q].view(num_rows, d.gdp_heads, d.gdp_key),
                vkq[:, d.conv_k].view(num_rows, d.gdp_householder, d.gdp_heads, d.gdp_key),
                vkq[:, d.conv_v].view(num_rows, d.gdp_householder, d.gdp_heads, d.gdp_value),
                decay,
                beta,
                ssm_states,
                slots,
                d.gdp_scale,
            )
        causal_conv1d_verify(
            vkq.view(num_rows, tokens, -1).transpose(1, 2),
            conv_states,
            g.conv,
            slots,
            step["intermediate_conv"][g.layer],
            PAD_SLOT_ID,
        )
        out = gdp_verify(
            vkq[:, d.conv_q].view(num_rows, tokens, d.gdp_heads, d.gdp_key),
            vkq[:, d.conv_k].view(num_rows, tokens, d.gdp_householder, d.gdp_heads, d.gdp_key),
            vkq[:, d.conv_v].view(num_rows, tokens, d.gdp_householder, d.gdp_heads, d.gdp_value),
            decay.view(num_rows, tokens, d.gdp_heads),
            beta.view(num_rows, tokens, d.gdp_householder, d.gdp_heads),
            ssm_states,
            slots,
            step["intermediate_ssm"][g.layer],
            d.gdp_scale,
        )
        return out.view(num_tokens, d.gdp_heads, d.gdp_value)

    def _moe(
        self, routed: torch.Tensor, shared: torch.Tensor, m: _Moe, sizes: list[int] | None
    ) -> torch.Tensor:
        """`Nemotron4MoE` with its routed experts split over the ranks.

        This rank routes its own tokens at model width (fp32 logits) and projects them
        into the latent space; every rank's tokens are gathered, each rank runs its window
        of the experts over all of them, and the windows' fp32 partial sums come back to
        the tokens' own ranks summed, then rounded once to bf16 -- HF's single fp32 sum
        over a token's ten experts. `fc2_norm`, the projection back up and the shared
        expert run on this rank's tokens.
        """
        d, core = self._d, self.core
        weights, ids = noaux_tc_op(
            cublas_mm(routed, m.router, None, torch.float32),
            m.router_bias,
            1,
            1,
            d.top_k,
            d.routed_scale,
        )
        latent = cublas_mm(routed, m.fc1_latent)
        # One byte row per token -- latent | ids | weights -- so one all-gather moves all three.
        packed = torch.cat(
            [latent.view(torch.uint8), ids.view(torch.uint8), weights.view(torch.uint8)], dim=1
        )
        gathered = allgather(packed, sizes, core.dp_group)
        latent_dtype, ids_dtype, weights_dtype = _DISPATCH_DTYPES
        partial = latent_moe_experts(
            gathered[:, d.dispatch_latent].view(latent_dtype),
            gathered[:, d.dispatch_ids].view(ids_dtype),
            gathered[:, d.dispatch_weights].view(weights_dtype),
            m.experts_up,
            m.experts_down,
            d.clamp,
            core.expert_offset,
            core.local_experts,
            torch.float32,
        )
        # The four windows' partial sums added in fp32, then the one rounding HF makes.
        experts = reducescatter(partial, sizes, core.dp_group).to(latent.dtype)
        experts = cublas_mm(flashinfer_rmsnorm(experts, m.fc2_norm, d.eps), m.fc2_latent)
        shared = cublas_mm(softclamp_relu2(cublas_mm(shared, m.shared_up), d.clamp), m.shared_down)
        return torch.add(experts, shared)

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """`Nemotron4MultiTokenPredictionLayer` over one draft step's rows: here the first,
        over a mixed step's context rows and generation rows alike.

        Returns the layer's `final_layernorm`-ed output, the hidden state HF hands the next
        depth and the head: the worker feeds each request's row back as the next step's
        `hidden_states`, and the shared head projects it with no norm of its own. Its MoE
        crosses the ranks as the trunk's does, sized by `all_rank_num_tokens`, the counts the
        worker hands this step.
        """
        d, m = self._d, self._mtp
        self._check_mtp_contract(attn_metadata, all_rank_num_tokens)
        step = self._mtp_step_args(attn_metadata, all_rank_num_tokens)

        # The next token's embedding first, then the previous step's hidden state.
        e = flashinfer_rmsnorm(nn.functional.embedding(input_ids, embed_tokens), m.enorm, d.eps)
        h = flashinfer_rmsnorm(hidden_states, m.hnorm, d.eps)
        x = cublas_mm(torch.cat((e, h), dim=-1), m.eh)
        # The shortcut block on an ordinary residual: routing reads the block input.
        routed = flashinfer_rmsnorm(x, m.shortcut_norm, d.eps)
        attn_out = thop_attention(
            q=cublas_mm(flashinfer_rmsnorm(x, m.input_norm, d.eps), m.attention.qkv),
            output=torch.empty(x.shape[0], d.q_width, dtype=x.dtype, device=x.device),
            local_layer_idx=step["kv_layer"],
            num_heads=d.heads_q,
            num_kv_heads=d.heads_kv,
            head_size=d.head_dim,
            **step["attention"],
            **_CALL_CONSTANTS,
        )
        # In place: the attention output becomes the shared expert's input, `x` the residual.
        shared, x = flashinfer_fused_add_rmsnorm(
            cublas_mm(attn_out, m.attention.o), x, m.pre_moe_norm, d.eps
        )
        out, _ = flashinfer_fused_add_rmsnorm(
            self._moe(routed, shared, m.moe, step["moe_sizes"]), x, m.final_norm, d.eps
        )
        return out


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    Batch-invariant: every entry it calls computes a row from that row alone, in an order
    neither the rank's row count nor the other ranks' change, so each request's logits and
    the state it writes are the same bits whatever batch, padded or not, the step runs in,
    and whatever the other ranks run beside it -- and token j of an MTP verify's 1 + D gets
    the bits that token gets as a decode step's one token. The GEMMs whose cuBLAS kernel
    changes with the row count run through `invariant_mm`; attention runs through
    `invariant_gen_attention`, whose KV split is fixed. The routed MoE's collectives are
    the ones PrefillTarget makes: the window call is row-invariant over every rank's
    tokens, and the reduce-scatter adds a token's partials in a fixed order. The MTP
    layer's draft steps are not part of this (`mtp_forward`).
    """

    def __init__(self, core: Nemotron4NanoPreviewDep4ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist. The MTP layer's draft
        # steps read another projection, off the metadata the worker rewrites between them.
        self._contract_pending = step_contract_enabled()
        self._mtp_contract_pending = step_contract_enabled()
        self._d = _dims(core.model_config.pretrained_config.text_config)
        self._embed = core.w["embed"]
        self._first_reads = _block_reads(core.w, core.layer_types, 0)
        self._layers = _layer_views(core.w, core.layer_types, row_invariant=True)
        # The MTP layer keeps cuBLAS's layout and kernels here too (`_mtp_view`, `_mtp_moe`).
        self._mtp = None if core.mtp_layer is None else _mtp_view(core.w, core.mtp_layer)
        self._tokens_per_gen_step = core.tokens_per_gen_step

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

    def _check_mtp_contract(self, md, all_rank_num_tokens) -> None:
        """The same opt-in fail-fast for the MTP layer's draft steps, whose metadata the
        worker rewrites between them and whose collectives the worker sizes."""
        if not self._mtp_contract_pending:
            return
        try:
            self._mtp_step_args(md, all_rank_num_tokens)
        except AttributeError as exc:
            raise AssertionError(f"draft-step metadata surface drifted: {exc}") from exc
        self._mtp_contract_pending = False

    def _attention_args(self, md: TrtllmAttentionMetadata) -> dict:
        """thop_attention's batch state for the MTP layer's draft steps, off the rank's one
        cache manager. The trunk's attention runs invariant_gen_attention, whose batch state
        is `step_args`'.

        A row may carry 1 + D tokens (the MTP layer's first draft step). thop sizes the XQA
        workspace as if each of `max_num_sequences` rows carried one token, so
        `max_num_sequences` counts every token a row may carry.
        """
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
            attention_window_size=md.max_seq_len,
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
            max_num_sequences=md.max_num_sequences * self._tokens_per_gen_step,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        )

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto the explicit batch state of
        invariant_gen_attention (`attention`), of the recurrent-state entries and of the
        attention-DP group, once per forward.

        Every row is one request carrying the same number of generation tokens: one, or
        1 + D under MTP, whose states go to the `intermediate_*` buffers. Under a CUDA graph
        the rows include the placeholder rows -- the padding up to the ranks' common graph
        batch and, on a rank with no request, the attention-DP dummy -- whose `state_slots`
        entry is `PAD_SLOT_ID`, and every rank runs the same batch, so `moe_sizes` is None.
        The padding rows' attention reads the padding dummy's own pages, the attention-DP
        dummy's its own. An eager decode step can meet ranks that prefill, and then carries
        their counts.
        """
        state = _state_metadata(md)
        views = self.core.state_views(md.kv_cache_manager)
        num_rows = md.num_seqs
        return dict(
            attention=dict(
                # Lengths count this step's tokens and are read on the device only, where
                # the overlap scheduler corrects them.
                sequence_length=md.kv_lens_cuda_runtime,
                kv_cache_block_offsets=md.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
                num_contexts=0,
                tokens_per_block=md.tokens_per_block,
                # Sizes the entry's grid: the cache manager's bound on every length, the
                # drafts a verify appends included.
                max_seq_len=md.max_seq_len,
            ),
            # Recurrent state: one slot per row, placeholder rows on PAD_SLOT_ID.
            num_rows=num_rows,
            state_slots=state.state_slots[:num_rows],
            kv_layer=views.kv_layer,
            ssm_states=views.ssm_states,
            conv_states=views.conv_states,
            intermediate_ssm=views.intermediate_ssm,
            intermediate_conv=views.intermediate_conv,
            # The attention-DP group: how many rows every rank feeds the routed MoE.
            moe_sizes=_moe_sizes(md.all_rank_num_tokens),
        )

    def _mtp_step_args(self, md: TrtllmAttentionMetadata, all_rank_num_tokens) -> dict:
        """The MTP layer's batch state, read off the metadata the worker hands each draft
        step: every call, since the worker rewrites that metadata between draft steps.

        Its KV is global layer `num_layers` in the rank's one cache manager, at that
        manager's own index. Its MoE's collectives take the counts the worker hands the
        step -- the step's tokens for the first, one row per request after it -- never the
        metadata's `all_rank_num_tokens`, which stays the trunk's through the draft loop.
        """
        return dict(
            attention=self._attention_args(md),
            kv_layer=md.kv_cache_manager.layer_offsets[self._mtp.attention.layer],
            moe_sizes=_moe_sizes(all_rank_num_tokens),
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
        """The final normed hidden state of every token: `norm(residual_readout(streams))`.

        A token of an MTP verify gets the bits it gets as a decode step's one token: every
        entry here computes a row from that row alone, and the attention computes token j of
        a row's 1 + D with the program that decodes it alone over the same keys.
        """
        d = self._d
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)

        h = (
            nn.functional.embedding(input_ids, self._embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        # Every row carries the same number of tokens, read off host shapes (`_gdp_generation`).
        tokens_per_seq = h.shape[0] // step["num_rows"]
        streams = h.repeat(1, d.streams)
        attn_out = torch.empty(h.shape[0], d.q_width, dtype=h.dtype, device=h.device)

        routed, x = self._streams_step(streams, None, None, self._first_reads)
        for L in self._layers:
            if isinstance(L.mixer, _Gdp):
                update = self._gdp(x, L.mixer, step)
            elif isinstance(L.mixer, _Attention):
                # Appends the row's K/V to its request's pages, then attends over them, causal
                # within the row's tokens. NoPE; q_scaling 1 keeps the softmax scale at
                # head_dim**-0.5 = 1/16.
                attn_out = invariant_gen_attention(
                    q=cublas_mm(x, L.mixer.qkv),
                    output=attn_out,
                    local_layer_idx=step["kv_layer"][L.mixer.layer],
                    tokens_per_seq=tokens_per_seq,
                    num_heads=d.heads_q,
                    num_kv_heads=d.heads_kv,
                    head_size=d.head_dim,
                    q_scaling=1.0,
                    **step["attention"],
                )
                update = invariant_mm(attn_out, L.mixer.o)
            else:
                update = invariant_mm(
                    softclamp_relu2(invariant_mm(x, L.mixer.up), d.clamp), L.mixer.down
                )
            if L.moe is None:
                routed, x = self._streams_step(streams, update, L.write, L.next_reads)
                continue
            _, shared = self._streams_step(streams, update, L.write, (L.shared_read,))
            update = self._moe(routed, shared, L.moe, step["moe_sizes"])
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
        """`Nemotron4GDPMixer` for the rows' tokens, each row continuing from its slot
        (`_gdp_generation`); placeholder rows (`PAD_SLOT_ID`) read zeros and write nothing."""
        d = self._d
        proj = cublas_mm(x, g.in_proj)  # [B * T, 18528] = [z | V0..V2 K0..K2 Q | b0..b2 | a]
        num_tokens = proj.shape[0]
        beta, decay = gdp_gates(proj[:, d.beta], proj[:, d.decay], g.neg_a, g.dt_bias)
        out = self._gdp_generation(proj, beta, decay, g, step["state_slots"], step)
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
        return cublas_mm(y, g.out_proj)

    def _gdp_generation(
        self,
        proj: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        g: _Gdp,
        slots: torch.Tensor,
        step: dict,
    ) -> torch.Tensor:
        """The GDP recurrence of the rows, request-major, each row from its slot; placeholder
        rows (`PAD_SLOT_ID`) read zeros and write nothing.

        Every row carries the same number of tokens, read off host shapes: the engine pads or
        truncates each row's drafts to the step's draft length (`resolve_draft_len`), the
        attention-DP dummy's included, and a step with none skips the worker's commit. One
        token goes through the decode entries, which update the slot in place. The 1 + D
        tokens of an MTP verify go through the verify entries: they run the row's tokens from
        its slot without writing it, and store the state after each token in the manager's
        per-position buffers, row b for row b. The worker's commit (`update_mamba_states`)
        then promotes the last accepted token's state into the slot, so a rejected draft
        never reaches it.
        """
        d = self._d
        num_rows, num_tokens = slots.shape[0], proj.shape[0]
        tokens = num_tokens // num_rows
        ssm_states, conv_states = step["ssm_states"][g.layer], step["conv_states"][g.layer]
        # In place: the conv columns of the in_proj output become the conv output.
        vkq = proj[:, d.conv]
        if tokens == 1:
            causal_conv1d_update(
                vkq.unsqueeze(-1), conv_states, g.conv, None, True, None, slots, PAD_SLOT_ID
            )
            return gdp_recurrent(
                vkq[:, d.conv_q].view(num_rows, d.gdp_heads, d.gdp_key),
                vkq[:, d.conv_k].view(num_rows, d.gdp_householder, d.gdp_heads, d.gdp_key),
                vkq[:, d.conv_v].view(num_rows, d.gdp_householder, d.gdp_heads, d.gdp_value),
                decay,
                beta,
                ssm_states,
                slots,
                d.gdp_scale,
            )
        causal_conv1d_verify(
            vkq.view(num_rows, tokens, -1).transpose(1, 2),
            conv_states,
            g.conv,
            slots,
            step["intermediate_conv"][g.layer],
            PAD_SLOT_ID,
        )
        out = gdp_verify(
            vkq[:, d.conv_q].view(num_rows, tokens, d.gdp_heads, d.gdp_key),
            vkq[:, d.conv_k].view(num_rows, tokens, d.gdp_householder, d.gdp_heads, d.gdp_key),
            vkq[:, d.conv_v].view(num_rows, tokens, d.gdp_householder, d.gdp_heads, d.gdp_value),
            decay.view(num_rows, tokens, d.gdp_heads),
            beta.view(num_rows, tokens, d.gdp_householder, d.gdp_heads),
            ssm_states,
            slots,
            step["intermediate_ssm"][g.layer],
            d.gdp_scale,
        )
        return out.view(num_tokens, d.gdp_heads, d.gdp_value)

    def _moe(
        self, routed: torch.Tensor, shared: torch.Tensor, m: _Moe, sizes: list[int] | None
    ) -> torch.Tensor:
        """`Nemotron4MoE` with its routed experts split over the ranks.

        This rank routes its own tokens at model width (fp32 logits) and projects them
        into the latent space; every rank's tokens are gathered, each rank runs its window
        of the experts over all of them, and the windows' fp32 partial sums come back to
        the tokens' own ranks summed, then rounded once to bf16 -- HF's single fp32 sum
        over a token's ten experts. `fc2_norm`, the projection back up and the shared
        expert run on this rank's tokens.

        A token's bits do not depend on the other rows of the step, on this rank or any
        other: its routing and projections run through `invariant_mm` and row-local
        entries; the window call gives each token the same bits whatever rows share it;
        and the reduce-scatter adds a token's four partials in the same order whatever the
        ranks' counts, padded or ragged.
        """
        d, core = self._d, self.core
        weights, ids = noaux_tc_op(
            invariant_mm(routed, m.router, torch.float32),
            m.router_bias,
            1,
            1,
            d.top_k,
            d.routed_scale,
        )
        latent = invariant_mm(routed, m.fc1_latent)
        # One byte row per token -- latent | ids | weights -- so one all-gather moves all three.
        packed = torch.cat(
            [latent.view(torch.uint8), ids.view(torch.uint8), weights.view(torch.uint8)], dim=1
        )
        gathered = allgather(packed, sizes, core.dp_group)
        latent_dtype, ids_dtype, weights_dtype = _DISPATCH_DTYPES
        partial = latent_moe_experts(
            gathered[:, d.dispatch_latent].view(latent_dtype),
            gathered[:, d.dispatch_ids].view(ids_dtype),
            gathered[:, d.dispatch_weights].view(weights_dtype),
            m.experts_up,
            m.experts_down,
            d.clamp,
            core.expert_offset,
            core.local_experts,
            torch.float32,
        )
        # The four windows' partial sums added in fp32, then the one rounding HF makes.
        experts = reducescatter(partial, sizes, core.dp_group).to(latent.dtype)
        experts = cublas_mm(flashinfer_rmsnorm(experts, m.fc2_norm, d.eps), m.fc2_latent)
        shared = invariant_mm(
            softclamp_relu2(cublas_mm(shared, m.shared_up), d.clamp), m.shared_down
        )
        return torch.add(experts, shared)

    def _mtp_moe(
        self, routed: torch.Tensor, shared: torch.Tensor, m: _Moe, sizes: list[int] | None
    ) -> torch.Tensor:
        """The MTP layer's `Nemotron4MoE`, on cuBLAS throughout, as PrefillTarget runs it, with
        the same collectives: a draft only proposes tokens the trunk then verifies, so its rows
        need not be batch-invariant, and its operands are `_mtp_view`'s `.t()` views, which
        `invariant_mm` would misread."""
        d, core = self._d, self.core
        weights, ids = noaux_tc_op(
            cublas_mm(routed, m.router, None, torch.float32),
            m.router_bias,
            1,
            1,
            d.top_k,
            d.routed_scale,
        )
        latent = cublas_mm(routed, m.fc1_latent)
        # One byte row per token -- latent | ids | weights -- so one all-gather moves all three.
        packed = torch.cat(
            [latent.view(torch.uint8), ids.view(torch.uint8), weights.view(torch.uint8)], dim=1
        )
        gathered = allgather(packed, sizes, core.dp_group)
        latent_dtype, ids_dtype, weights_dtype = _DISPATCH_DTYPES
        partial = latent_moe_experts(
            gathered[:, d.dispatch_latent].view(latent_dtype),
            gathered[:, d.dispatch_ids].view(ids_dtype),
            gathered[:, d.dispatch_weights].view(weights_dtype),
            m.experts_up,
            m.experts_down,
            d.clamp,
            core.expert_offset,
            core.local_experts,
            torch.float32,
        )
        # The four windows' partial sums added in fp32, then the one rounding HF makes.
        experts = reducescatter(partial, sizes, core.dp_group).to(latent.dtype)
        experts = cublas_mm(flashinfer_rmsnorm(experts, m.fc2_norm, d.eps), m.fc2_latent)
        shared = cublas_mm(softclamp_relu2(cublas_mm(shared, m.shared_up), d.clamp), m.shared_down)
        return torch.add(experts, shared)

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """`Nemotron4MultiTokenPredictionLayer` over one draft step's rows, all generation
        rows: the first step of a decode step (1 + D tokens per request) or any later step
        (one token per request).

        Returns the layer's `final_layernorm`-ed output, the hidden state HF hands the next
        depth and the head: the worker feeds each request's row back as the next step's
        `hidden_states`, and the shared head projects it with no norm of its own. Its MoE
        crosses the ranks, sized by `all_rank_num_tokens`, the counts the worker hands this
        step.

        The kernels are PrefillTarget's -- cuBLAS, thop_attention -- not the trunk's
        batch-invariant ones: a draft's bits move with the batch, which can change how many
        drafts the trunk accepts but never which tokens it emits.
        """
        d, m = self._d, self._mtp
        self._check_mtp_contract(attn_metadata, all_rank_num_tokens)
        step = self._mtp_step_args(attn_metadata, all_rank_num_tokens)

        # The next token's embedding first, then the previous step's hidden state.
        e = flashinfer_rmsnorm(nn.functional.embedding(input_ids, embed_tokens), m.enorm, d.eps)
        h = flashinfer_rmsnorm(hidden_states, m.hnorm, d.eps)
        x = cublas_mm(torch.cat((e, h), dim=-1), m.eh)
        # The shortcut block on an ordinary residual: routing reads the block input.
        routed = flashinfer_rmsnorm(x, m.shortcut_norm, d.eps)
        attn_out = thop_attention(
            q=cublas_mm(flashinfer_rmsnorm(x, m.input_norm, d.eps), m.attention.qkv),
            output=torch.empty(x.shape[0], d.q_width, dtype=x.dtype, device=x.device),
            local_layer_idx=step["kv_layer"],
            num_heads=d.heads_q,
            num_kv_heads=d.heads_kv,
            head_size=d.head_dim,
            **step["attention"],
            **_CALL_CONSTANTS,
        )
        # In place: the attention output becomes the shared expert's input, `x` the residual.
        shared, x = flashinfer_fused_add_rmsnorm(
            cublas_mm(attn_out, m.attention.o), x, m.pre_moe_norm, d.eps
        )
        out, _ = flashinfer_fused_add_rmsnorm(
            self._mtp_moe(routed, shared, m.moe, step["moe_sizes"]), x, m.final_norm, d.eps
        )
        return out


class MTPLayer:
    """The checkpoint's MTP layer, global layer `num_layers`, as the one-model MTP-Eagle worker
    runs it once per draft depth (`mtp_use_repeated_layer`: one layer serves every depth), on
    every rank for that rank's requests.

    Its output is already normed by the layer's `final_layernorm`, which is what HF hands the
    next depth and its head (`Nemotron4MultiTokenPredictionLayer.forward`), so `shared_head`
    is the trunk's fp32 lm_head alone, whole on every rank.
    """

    def __init__(self, core: Nemotron4NanoPreviewDep4ModelingV2) -> None:
        self.core = core

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """fp32 draft logits of every row, as the trunk's: HF's fp32 GEMM over the bf16
        hidden state and lm_head weight."""
        return cublas_mm(hidden_states, lm_head.weight.t(), None, torch.float32)

    def forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Route the draft step by the phase this call's metadata shows.

        Per call, not per engine step: after the first draft step the worker rewrites
        `attn_metadata` in place to generation rows only, so one mixed step's draft calls
        cross from prefill to decode. `all_rank_num_tokens` is the worker's own count for this
        draft step, every rank's in rank order, which sizes the layer's MoE collectives.
        """
        return self.core._select_target(attn_metadata).mtp_forward(
            embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata
        )


class DraftModel:
    """The container the spec worker reaches the drafter through.

    `embed_tokens` must resolve through the core on every read: this container is built while
    every parameter is still a meta tensor, and the engine materializes the registry by
    replacing the tensor objects, so a reference snapshotted here stays on meta.
    """

    def __init__(self, core: Nemotron4NanoPreviewDep4ModelingV2, lm_head) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class ModelingV2Nemotron4NanoPreviewSm103Dep4(
    DecoderModelForCausalLM[Nemotron4NanoPreviewDep4ModelingV2, PretrainedConfig]
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
        # Under attention DP the base class builds lm_head whole, [vocab, hidden] on every
        # rank: each rank computes the logits of its own requests.
        super().__init__(
            Nemotron4NanoPreviewDep4ModelingV2(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if spec_config is not None:
            self.mamba_metadata_cls = Nemotron4StateMetadata.for_draft_len(
                spec_config.max_draft_len
            )
            self.draft_model = DraftModel(self.model, self.lm_head)
            # Under attention DP the engine keeps the MTP layer's KV in the rank's one cache
            # manager beside the trunk's and builds no separate draft manager
            # (`_should_create_separate_draft_kv_cache`), so the worker swaps none in.
            self.spec_worker = get_spec_worker(
                spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=False,
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
    ) -> torch.Tensor | dict:
        """fp32 logits of this rank's requests, one row per request (every token with
        `return_context_logits`); under speculative decoding, the worker's step: acceptance,
        the commit of the accepted states, and the next drafts.

        HF's lm_head is an fp32 GEMM over the bf16 hidden state and weight
        (`Nemotron4ForCausalLM.forward`); `cublas_mm` with an fp32 output computes the same
        products, exact in fp32, and keeps the fp32 sum, where the default logits processor
        would round the logits to bf16 first. The worker gets the same logits for each
        context's last token and every token a generation row verifies (`gather_ids`), and
        `position_ids` in the engine's [1, T] shape, which it squeezes itself.
        """
        hidden = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )
        if self.spec_worker is None:
            if not return_context_logits:
                last = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                hidden = hidden[last]
            return cublas_mm(hidden, self.lm_head.weight.t(), None, torch.float32)
        logits = cublas_mm(
            hidden[spec_metadata.gather_ids], self.lm_head.weight.t(), None, torch.float32
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )
