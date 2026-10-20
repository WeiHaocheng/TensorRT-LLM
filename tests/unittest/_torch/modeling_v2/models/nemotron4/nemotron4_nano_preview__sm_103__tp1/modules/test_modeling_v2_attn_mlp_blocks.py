# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the nemotron4 tp1 target's attention and dense-MLP blocks against the reference.

What runs is the target itself: the whole text model on one GPU, built as the engine builds it
(``ModelConfig`` off the checkpoint, the target class, ``load_weights`` over every checkpoint tensor,
``post_load_weights``), the engine's own hybrid cache manager (``_create_kv_cache_manager``) and the
engine's TRTLLM attention metadata (``build_attention_metadata``) filled per step as the executor's
runner fills it. Its forward is never edited: the attention and dense-MLP blocks are observed through
the module globals the flat forward calls (``wide_residual``, ``cublas_mm`` and ``thop_attention``,
and in decode steps ``invariant_mm`` and ``invariant_gen_attention``), each call attributed to its
layer by the weight operand it was handed.

The blocks under test, with the stream reads and writes around them:

* attention (``shortcut_moe_attn`` layers 5, 9, 14, 19, 24, 29, 34, 38): the mixer read and its
  ``input_layernorm`` -> fused qkv GEMM -> NoPE causal attention over the paged KV cache (16 query
  heads, 1 KV head of 256, scale 1/16) -> ``o_proj`` -> the mixer write, and the shared expert's read
  that follows it;
* dense MLP (``mlp`` layers 1, 3, 41, 43): the read and its norm -> up GEMM -> soft-clamped relu^2 ->
  down GEMM -> the write, and the next block's reads that follow it.

Teacher forcing is per block: the reference block runs on exactly the streams the target's block
read (``S_in``, real activations of the fixed prompt set), so every number below is that block's own
error. Two kinds of comparison localize it:

* ``op``: each op of the block fed the target's own input to that op (the read on ``S_in``, the qkv
  GEMM on the target's normed read, the attention on the target's own q and cached K/V, ...);
* ``module``: the reference block end to end on ``S_in``, its attention over the reference's own
  K/V history.

Beside them, ``hf``: the same block in HuggingFace's own precision (bf16 cuBLAS GEMMs and bf16
``scaled_dot_product_attention``, as ``modeling_nemotron4.py`` runs under ``attn_implementation
"sdpa"``) -- an implementation as accurate as the trained model's, which a band must admit -- and
``rival_*``: one planted mistake each (a wrong softmax scale, RoPE on a NoPE model, attention without
its cached history, K and V swapped, a query that sees the next token, relu^2 without the soft
clamp, a write without the learned retention), which the bands must reject.

Phases, each the shapes the engine runs:

* ``context``: fresh prompts, chunked prefill off (non-paged context FMHA); a varlen batch of the six
  short prompts, two ~8K prompts alone, and varlen batches of prompt prefixes;
* ``chunked``: chunked prefill on (paged context FMHA), a ~8K prompt in 2048-token chunks, each over
  the KV the earlier chunks wrote;
* ``mixed``: chunked prefill on, a prompt's chunks beside generation rows in one step;
* ``decode_b1`` / ``decode_b3`` / ``decode_b64``: generation-only steps (the decode target), one
  token per request over histories of 31 to 8K tokens.

The reference is ``_nemotron4_reference.py`` (fp32 attention and IEEE fp32 GEMMs rounded once to
bf16, every rounding where HF rounds) on the checkpoint's tensors as stored, read independently of
the target's own weight relayout (fused qkv, bf16 residual coefficients). The bands, and what each
rests on, are ``OP_BANDS`` and ``MODULE_BANDS``.

A stream write is judged against the magnitude of the streams it reads and writes, not of its
result (``stream_scale``): at some tokens the last dense MLP (layer 43) writes an update that cancels
most of a stream, and an error that is small next to the operands is then large next to the result.
That cancellation is also why free-running drift between two implementations shows up at layer 43.

Needs the checkpoint under ``$LLM_MODELS_ROOT/nemotron4-nano-preview`` and ~220 GiB of free memory on
one GPU (the whole target, as served); ~2-3 minutes to load and ~4 minutes to run.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib
import json
import os
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F

__extra_import_path__ = ["../..", "../../../.."]
import _nemotron4_prompts as P  # noqa: E402 -- needs the path declared above
import _nemotron4_reference as R  # noqa: E402
from _validating import ULP  # noqa: E402

_MODELING = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4."
    "nemotron4_nano_preview__sm_103__tp1.modeling"
)

# The engine sizes this run uses everywhere (task.yaml, the boot gate, check_goldens).
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
TOKENS_PER_BLOCK = 32
CHUNK = 2048
#: The mixed steps' chunk: a whole number of KV blocks, as the scheduler cuts chunks.
MIXED_CHUNK = 3008
#: The largest decode batch below; also the state slots and attention rows the metadata holds.
MAX_BATCH = 64
#: Generated tokens a request may take here (KV capacity is reserved for them up front).
MAX_NEW = 8
#: KV tokens of the paged pool: every request of the campaign stays resident (~175K tokens). Kept
#: well below the free memory, which the reference blocks, their K/V histories and each step's
#: captured activations use.
KV_TOKENS = 262_144
KV_FRACTION = 0.1

BF16_ULP = ULP[torch.bfloat16]
ATTENTION_KINDS = ("shortcut_moe_attn",)
MLP_KINDS = ("mlp",)
LAYER = "model.language_model.layers.{}."
TEXT = R.TEXT_PREFIX


# ------------------------------------------------------------------------------------------------
# The target, its cache manager and its attention metadata, built as the engine builds them
# ------------------------------------------------------------------------------------------------


def _modeling():
    return importlib.import_module(_MODELING)


def _mapping():
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)


def _checkpoint_tensors(path: str) -> dict:
    """Every checkpoint tensor as a lazy safetensors slice (``[:]`` reads it), as the engine's
    lazy loader hands them to ``load_weights``."""
    from safetensors import safe_open

    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {
        shard: safe_open(os.path.join(path, shard), framework="pt", device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    return {key: handles[shard].get_slice(key) for key, shard in weight_map.items()}


def build_target(path: str):
    """``(ModelConfig, the loaded target)``: the target class off the checkpoint's config, every
    weight loaded through its own ``load_weights``, on the current CUDA device."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype

    config = ModelConfig.from_pretrained(path, trust_remote_code=True)
    validate_and_set_mamba_ssm_cache_dtype(config, "auto")
    with torch.device("cuda"):
        model = _modeling().ModelingV2Nemotron4NanoPreviewSm103Tp1(config)
    model.load_weights(_checkpoint_tensors(path))
    model.post_load_weights()
    torch.cuda.synchronize()
    return config, model.eval()


def build_manager(config, v2: bool = True):
    """The engine's hybrid cache manager for this config: paged KV for the attention layers,
    recurrent and conv state slots for the GDP layers. ``v2`` picks the default
    ``MambaHybridCacheManagerV2``; otherwise the C++ ``CppMambaHybridCacheManager``."""
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    kv_cache_config = KvCacheConfig(
        max_tokens=KV_TOKENS,
        enable_block_reuse=False,
        free_gpu_memory_fraction=KV_FRACTION,
        use_kv_cache_manager_v2=v2,
    )
    cls = get_kv_cache_manager_cls(config, kv_cache_config)
    return _create_kv_cache_manager(
        None,
        cls,
        _mapping(),
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=config,
        dtype=torch.bfloat16,
        max_cuda_graph_batch_size=MAX_BATCH,
    )


def build_metadata(config, manager, chunked_prefill: bool):
    """The engine's TRTLLM attention metadata; ``chunked_prefill`` is the engine's
    ``enable_chunked_prefill``, which turns on the paged context FMHA."""
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata

    md = build_attention_metadata(
        config,
        max_batch_size=MAX_BATCH,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        attention_backend=get_attention_backend("TRTLLM"),
        attention_runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=chunked_prefill,
            cache_reuse=False,
            has_speculative_draft_tokens=False,
            chunk_size=MAX_NUM_TOKENS,
        ),
        mapping=_mapping(),
        cache_indirection=None,
        kv_cache_manager=manager,
    )
    md.mamba_metadata_cls = _modeling().ModelingV2Nemotron4NanoPreviewSm103Tp1.mamba_metadata_cls
    return md


@dataclasses.dataclass
class Request:
    """One sequence: its prompt, the tokens whose KV and state are written, its greedy output."""

    rid: int
    name: str
    prompt: list[int]
    done: int = 0
    generated: list[int] = dataclasses.field(default_factory=list)

    @property
    def in_context(self) -> bool:
        return self.done < len(self.prompt)


@dataclasses.dataclass
class Row:
    """One row of a step: a request and the tokens it feeds (a prompt chunk, or one token)."""

    request: Request
    tokens: list[int]
    start: int = 0  # first token of the row in the step
    cached: int = 0  # tokens of the request already in KV and state


def run_step(model, md, manager, rows: list[Row]) -> torch.Tensor:
    """One forward over ``rows`` (context rows first), with the per-step fields set as the
    executor's runner sets them; returns each row's last-token fp32 logits."""
    from tensorrt_llm._torch.metadata import KVCacheParams

    num_contexts = sum(r.request.in_context for r in rows)
    assert all(r.request.in_context for r in rows[:num_contexts]), "context rows go first"
    start = 0
    for r in rows:
        r.start, r.cached = start, r.request.done
        start += len(r.tokens)
    md.seq_lens = torch.tensor([len(r.tokens) for r in rows], dtype=torch.int)
    md.beam_width = 1
    md.request_ids = [r.request.rid for r in rows]
    # As the runner: a context row's "prompt length" is its chunk's length (the attention op
    # takes it as that row's query length), a generation row's is the request's prompt length.
    md.prompt_lens = [
        len(r.tokens) if r.request.in_context else len(r.request.prompt) for r in rows
    ]
    md.num_contexts = num_contexts
    md.num_chunked_ctx_requests = 0
    md.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=[r.request.done for r in rows],
        num_extra_kv_tokens=0,
    )
    md.kv_cache_manager = manager
    md.padded_num_tokens = None
    md.prepare()
    ids = torch.tensor([t for r in rows for t in r.tokens], dtype=torch.int, device="cuda")
    logits = model.forward(attn_metadata=md, input_ids=ids)
    for r, best in zip(rows, logits.argmax(-1).tolist()):
        r.request.done += len(r.tokens)
        if not r.request.in_context:
            r.request.generated.append(best)
    return logits


# ------------------------------------------------------------------------------------------------
# Observing the blocks inside the target's flat forward
# ------------------------------------------------------------------------------------------------


class BlockObserver:
    """Wraps the module globals the target's forward calls and keeps, per layer under test, what
    its block read and wrote in the last forward.

    Calls are attributed by their weight operand: a GEMM by the target parameter it multiplies
    (``cublas_mm``'s ``mat_b``, a ``.t()`` view of it; ``invariant_mm``'s ``weight``, the parameter
    as stored), a ``wide_residual`` call by the read factors and the retention it was handed, an
    attention call (``thop_attention``, or ``invariant_gen_attention`` in decode) by its
    ``local_layer_idx``. Every recorded tensor is a copy taken at the call (``wide_residual`` writes
    the streams in place). A target that runs these blocks through other entries (another GEMM or
    attention entry) needs those names wrapped here too; until then a block's record misses that
    output and the campaign says which.
    """

    WRAPPED = (
        "wide_residual",
        "cublas_mm",
        "invariant_mm",
        "thop_attention",
        "invariant_gen_attention",
    )

    def __init__(self, module, core, manager):
        self.module = module
        self.orig = {name: getattr(module, name) for name in self.WRAPPED}
        w = core.w
        kinds = core.layer_types
        self.attention_layers = [i for i, t in enumerate(kinds) if t in ATTENTION_KINDS]
        self.mlp_layers = [i for i, t in enumerate(kinds) if t in MLP_KINDS]
        self.gemm, self.writes, self.reads = {}, {}, {}
        for i in self.attention_layers:
            self.gemm[w[f"l{i}_qkv"].data_ptr()] = (i, "qkv")
            self.gemm[w[f"l{i}_o"].data_ptr()] = (i, "update")
            self.writes[w[f"l{i}_mix_retain"].data_ptr()] = i
            self.reads[w[f"l{i}_mix_read"].data_ptr()] = (i, "x")
        for i in self.mlp_layers:
            self.gemm[w[f"l{i}_mlp_up"].data_ptr()] = (i, "up")
            self.gemm[w[f"l{i}_mlp_down"].data_ptr()] = (i, "update")
            self.writes[w[f"l{i}_res_retain"].data_ptr()] = i
            self.reads[w[f"l{i}_res_read"].data_ptr()] = (i, "x")
        self.layer_of_kv = {manager.layer_offsets[i]: i for i in self.attention_layers}
        self.record: dict[int, dict] = defaultdict(dict)

    def install(self, monkeypatch) -> None:
        for name in self.WRAPPED:
            monkeypatch.setattr(self.module, name, getattr(self, name))

    def take(self) -> dict[int, dict]:
        record, self.record = self.record, defaultdict(dict)
        return record

    def wide_residual(self, streams, update, retention, write, f1, g1, eps, f2=None, g2=None):
        layer = None if retention is None else self.writes.get(retention.data_ptr())
        if layer is not None:
            self.record[layer]["s_in"] = streams.clone()
        out = self.orig["wide_residual"](streams, update, retention, write, f1, g1, eps, f2, g2)
        if layer is not None:
            rec = self.record[layer]
            rec["s_out"] = streams.clone()
            rec["next_reads"] = [out[1].clone()] + ([] if f2 is None else [out[2].clone()])
        for factors, normed in ((f1, out[1]), (f2, out[2])):
            if factors is not None and factors.data_ptr() in self.reads:
                i, role = self.reads[factors.data_ptr()]
                self.record[i][role] = normed.clone()
        return out

    def _gemm(self, x, weight, out) -> None:
        tag = self.gemm.get(weight.data_ptr())
        if tag is not None:
            i, role = tag
            if role == "update":
                self.record[i]["update_in"] = x.clone()
            self.record[i][role] = out.clone()

    def cublas_mm(self, mat_a, mat_b, *args, **kwargs):
        out = self.orig["cublas_mm"](mat_a, mat_b, *args, **kwargs)
        self._gemm(mat_a, mat_b, out)
        return out

    def invariant_mm(self, x, weight, *args, **kwargs):
        out = self.orig["invariant_mm"](x, weight, *args, **kwargs)
        self._gemm(x, weight, out)
        return out

    def _attention(self, name, kwargs):
        out = self.orig[name](**kwargs)
        i = self.layer_of_kv[kwargs["local_layer_idx"]]
        self.record[i]["attention"] = out.clone()
        return out

    def thop_attention(self, **kwargs):
        return self._attention("thop_attention", kwargs)

    def invariant_gen_attention(self, **kwargs):
        return self._attention("invariant_gen_attention", kwargs)


# ------------------------------------------------------------------------------------------------
# The reference blocks
# ------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class AttentionWeights:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    qkv: torch.Tensor
    input_norm: torch.Tensor
    read_logit: torch.Tensor
    write_logit: torch.Tensor
    retention_logit: torch.Tensor
    shared_read_logit: torch.Tensor
    pre_moe_norm: torch.Tensor


@dataclasses.dataclass
class MlpWeights:
    up: torch.Tensor
    down: torch.Tensor
    input_norm: torch.Tensor
    read_logit: torch.Tensor
    write_logit: torch.Tensor
    retention_logit: torch.Tensor
    next_reads: list[tuple[torch.Tensor, torch.Tensor]]  # (read logit, norm weight) per read


def load_attention_weights(ckpt: R.Checkpoint, i: int) -> AttentionWeights:
    t = lambda name: ckpt.tensor(LAYER.format(i) + name)  # noqa: E731
    q, k, v = (t(f"self_attn.{p}_proj.weight") for p in "qkv")
    return AttentionWeights(
        q=q,
        k=k,
        v=v,
        o=t("self_attn.o_proj.weight"),
        qkv=torch.cat([q, k, v]),
        input_norm=t("input_layernorm.weight"),
        read_logit=t("mixer_residual_connection.read_logit"),
        write_logit=t("mixer_residual_connection.write_logit"),
        retention_logit=t("mixer_residual_connection.retention_logit"),
        shared_read_logit=t("moe_residual_connection.read_logit"),
        pre_moe_norm=t("pre_moe_layernorm.weight"),
    )


def load_mlp_weights(ckpt: R.Checkpoint, i: int) -> MlpWeights:
    t = lambda name: ckpt.tensor(LAYER.format(i) + name)  # noqa: E731
    kinds = ckpt.config.layer_types
    j = i + 1
    if j == len(kinds):
        next_reads = [
            (ckpt.tensor(TEXT + "residual_readout.logit"), ckpt.tensor(TEXT + "norm.weight"))
        ]
    elif kinds[j] in ("shortcut_moe_attn", "shortcut_moe_gdp"):
        n = lambda name: ckpt.tensor(LAYER.format(j) + name)  # noqa: E731
        next_reads = [
            (n("shortcut_read.logit"), n("shortcut_input_layernorm.weight")),
            (n("mixer_residual_connection.read_logit"), n("input_layernorm.weight")),
        ]
    else:
        n = lambda name: ckpt.tensor(LAYER.format(j) + name)  # noqa: E731
        next_reads = [(n("residual_connection.read_logit"), n("input_layernorm.weight"))]
    return MlpWeights(
        up=t("mlp.up_proj.weight"),
        down=t("mlp.down_proj.weight"),
        input_norm=t("input_layernorm.weight"),
        read_logit=t("residual_connection.read_logit"),
        write_logit=t("residual_connection.write_logit"),
        retention_logit=t("residual_connection.retention_logit"),
        next_reads=next_reads,
    )


def normed_read(streams, logit, norm, config: R.TextConfig) -> torch.Tensor:
    """A connection's read and the RMSNorm after it, in HF's order (``Nemotron4Layer._read``)."""
    x = R.stream_read(streams, logit, config.wide_residual_num_streams)
    return R.rms_norm(x, norm, config.rms_norm_eps)


def stream_write(streams, update, write_logit, retention_logit, config: R.TextConfig):
    return R.stream_write(
        streams,
        update,
        write_logit,
        retention_logit,
        config.wide_residual_retention_max_forget,
        config.wide_residual_num_streams,
    )


def split_qkv(qkv: torch.Tensor, config: R.TextConfig):
    """``[T, q | k | v]`` as ``q [T, heads, d]``, ``k`` and ``v`` ``[T, kv_heads, d]``."""
    heads, kv_heads, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q, k, v = qkv.split([heads * d, kv_heads * d, kv_heads * d], dim=-1)
    n = qkv.shape[0]
    return q.reshape(n, heads, d), k.reshape(n, kv_heads, d), v.reshape(n, kv_heads, d)


def sdpa_bf16(q, k, v, scale):
    """HF's ``sdpa_attention_forward`` on bf16 operands: K/V repeated over the query heads
    (``repeat_kv``), causal over the S cached keys plus the T new ones."""
    n, heads, d = q.shape
    total = k.shape[0]
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).expand(heads, total, d).unsqueeze(0).contiguous()
    vh = v.transpose(0, 1).expand(heads, total, d).unsqueeze(0).contiguous()
    if total == n:
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale)
    else:
        rows = torch.arange(n, device=q.device)[:, None] + (total - n)
        mask = torch.arange(total, device=q.device)[None, :] <= rows
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    return o[0].transpose(0, 1).contiguous()


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Rotary embedding over the whole head (half-split pairing): the positional encoding this
    NoPE model must not have."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = positions.float()[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    a, b = x.float()[..., : d // 2], x.float()[..., d // 2 :]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1).to(x.dtype)


# ------------------------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------------------------


def errors(out: torch.Tensor, ref: torch.Tensor, scale: torch.Tensor | None = None) -> dict:
    """``out`` against ``ref``, rows = last dim.

    ``worst_row`` is the largest element error over its row's scale -- by default the row's
    largest ``|ref|``, so that in units of ``BF16_ULP`` it is ``assert_within_ulp``'s element gate --
    and ``rel_rms`` its RMS gate. ``scale`` replaces the row's scale where the result is not the
    right yardstick (a stream write that cancels its operands, see ``stream_scale``);
    ``worst_row_ref`` then still reports the error over ``|ref|``. ``max_rel`` is the largest
    element error over the element's own ``|ref|``, among elements at least 1/16 of their row's
    largest (smaller ones are cancellation, where a relative error is meaningless). ``worst`` is the
    element behind ``worst_row``.
    """
    assert out.shape == ref.shape, (out.shape, ref.shape)
    tiny = torch.finfo(torch.float32).tiny
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    d = (o - r).abs()
    ref_scale = r.abs().amax(-1).clamp_min(tiny)
    row_scale = ref_scale if scale is None else scale.float().reshape(-1).clamp_min(tiny)
    rows = d.amax(-1) / row_scale
    row = int(rows.argmax())
    col = int(d[row].argmax())
    big = r.abs() >= ref_scale[:, None] / 16
    rel = torch.where(big, d / r.abs().clamp_min(tiny), 0.0)
    return dict(
        rows=int(o.shape[0]),
        worst_row=float(rows.max()),
        worst_row_ref=float((d.amax(-1) / ref_scale).max()),
        median_row=float(rows.median()),
        rel_rms=float((o - r).norm() / r.norm().clamp_min(tiny)),
        identical=float((out == ref).float().mean()),
        max_abs=float(d.max()),
        max_rel=float(rel.max()),
        worst=dict(row=row, col=col, out=float(o[row, col]), ref=float(r[row, col])),
    )


def stream_scale(s_in: torch.Tensor, s_out: torch.Tensor) -> torch.Tensor:
    """Per token, the largest stream magnitude a write reads or writes.

    A write computes ``r * S_in + w * U``; where the update cancels most of a stream (layer 43 does
    at some tokens: streams of max ~1600 leave it at max ~300), any error of ``U`` is large next
    to the result but not next to the operands it came from. A write is judged against the
    operands' magnitude, which bounds its error whatever the cancellation.
    """
    return torch.maximum(s_in.float().abs().amax(-1), s_out.float().abs().amax(-1))


def merge(a: dict, b: dict) -> dict:
    """The worst of two error records (``worst`` follows ``worst_row``; ``identical`` is the
    smaller fraction; ``rows`` add up)."""
    out = dict(a)
    for name in ("rel_rms", "median_row", "max_abs", "max_rel", "worst_row_ref"):
        out[name] = max(a[name], b[name])
    out["identical"] = min(a["identical"], b["identical"])
    out["rows"] = a["rows"] + b["rows"]
    if b["worst_row"] > a["worst_row"]:
        out.update(worst_row=b["worst_row"], worst=b["worst"], step=b["step"])
        if "layer" in b:
            out["layer"] = b["layer"]
    return out


def summarize(results: dict, by_layer: bool = False) -> dict:
    """``results`` merged over layers (or kept per layer), keyed (phase, kind, comparison,
    output[, layer])."""
    out: dict = {}
    for (phase, kind, comparison, output, layer), e in results.items():
        key = (phase, kind, comparison, output) + ((layer,) if by_layer else ())
        e = dict(e, layer=layer)
        out[key] = e if key not in out else merge(out[key], e)
    return out


# ------------------------------------------------------------------------------------------------
# The campaign: steps, and per step the comparisons of every block under test
# ------------------------------------------------------------------------------------------------

ATTENTION_RIVALS = ("scale", "rope", "no_history", "kv_swapped", "future_token", "no_retention")
MLP_RIVALS = ("no_clamp", "no_retention")


class Campaign:
    """Drives the target through the phases and checks every block under test after each step.

    ``results`` holds one entry per (phase, block kind, comparison, output): the per-step, per-layer
    errors, merged by taking the worst of each.
    """

    def __init__(self, checkpoint: str, model, manager, observer: BlockObserver, log=print):
        self.ckpt = R.Checkpoint(checkpoint)
        self.cfg = self.ckpt.config
        self.model = model
        self.manager = manager
        self.observer = observer
        self.log = log
        self.attention_weights = {
            i: load_attention_weights(self.ckpt, i) for i in observer.attention_layers
        }
        self.mlp_weights = {i: load_mlp_weights(self.ckpt, i) for i in observer.mlp_layers}
        self.scale = self.cfg.head_dim**-0.5
        # Per (layer, request id): K and V of every token so far, as each side computed them.
        self.history = {side: {} for side in ("target", "ref", "hf")}
        self.results: dict[tuple, dict] = {}
        self.steps: list[dict] = []
        self.next_rid = 1

    # --- requests and steps ---------------------------------------------------------------------

    def new_request(self, name: str, prompt: list[int]) -> Request:
        req = Request(self.next_rid, name, list(prompt))
        self.next_rid += 1
        added = self.manager.add_dummy_requests([req.rid], token_nums=[len(prompt) + MAX_NEW])
        assert added, f"the cache manager has no room for {name} ({len(prompt)} tokens)"
        return req

    def step(self, md, rows: list[Row], phase: str, rivals: bool = False) -> None:
        self.observer.take()
        run_step(self.model, md, self.manager, rows)
        torch.cuda.synchronize()
        record = self.observer.take()
        info = dict(
            phase=phase,
            rows=len(rows),
            contexts=sum(r.cached < len(r.request.prompt) for r in rows),
            tokens=sum(len(r.tokens) for r in rows),
            max_history=max(r.cached for r in rows),
        )
        self.steps.append(info)
        for i in self.observer.attention_layers:
            self._attention(i, record[i], rows, phase, rivals)
        for i in self.observer.mlp_layers:
            self._mlp(i, record[i], rows, phase, rivals)
        del record
        torch.cuda.empty_cache()
        self.log(f"step {len(self.steps)} {info}; free {_free_gib():.1f} GiB")

    def close(self) -> None:
        """Release the target, its cache and the histories (the GPU's ~200 GiB)."""
        shutdown = getattr(self.manager, "shutdown", None)
        if shutdown is not None:
            shutdown()
        self.model = self.manager = None
        self.history.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def _note(self, phase, kind, comparison, output, layer, err) -> None:
        """Merge one step's errors into ``results[(phase, kind, comparison, output, layer)]``."""
        key = (phase, kind, comparison, output, layer)
        err = dict(err, step=len(self.steps))
        merged = self.results.get(key)
        self.results[key] = err if merged is None else merge(merged, err)

    def _segments(self, rows):
        for r in rows:
            yield r, slice(r.start, r.start + len(r.tokens))

    # --- the attention block --------------------------------------------------------------------

    def _attention(self, i, rec, rows, phase, rivals) -> None:
        cfg, w = self.cfg, self.attention_weights[i]
        missing = {"s_in", "x", "qkv", "attention", "update_in", "update", "s_out", "next_reads"}
        missing -= rec.keys()
        assert not missing, f"layer {i}: the observer saw no {sorted(missing)}"
        s_in = rec["s_in"]
        n = s_in.shape[0]
        note = lambda c, o, out, ref, scale=None: self._note(  # noqa: E731
            phase, "attention", c, o, i, errors(out, ref, scale)
        )

        # The reads, both from S_in.
        x_ref = normed_read(s_in, w.read_logit, w.input_norm, cfg)
        note("op", "read", rec["x"], x_ref)

        # Each op on the target's own input to it.
        note("op", "qkv", rec["qkv"], R.linear(rec["x"], w.qkv))
        q_t, k_t, v_t = split_qkv(rec["qkv"], cfg)
        o_iso = torch.empty_like(rec["attention"]).view(n, cfg.num_attention_heads, cfg.head_dim)

        # The reference block on S_in, and HF's precision on S_in.
        q_r, k_r, v_r = split_qkv(R.linear(x_ref, w.qkv), cfg)
        q_h, k_h, v_h = split_qkv(F.linear(x_ref, w.qkv), cfg)
        o_ref = torch.empty_like(o_iso)
        o_hf = torch.empty_like(o_iso)
        for r, seg in self._segments(rows):
            key = (i, r.request.rid)
            for side, k_new, v_new in (("target", k_t, v_t), ("ref", k_r, v_r), ("hf", k_h, v_h)):
                past = self.history[side].get(key)
                k_all = k_new[seg] if past is None else torch.cat([past[0], k_new[seg]])
                v_all = v_new[seg] if past is None else torch.cat([past[1], v_new[seg]])
                assert k_all.shape[0] == r.cached + len(r.tokens), (key, k_all.shape, r.cached)
                self.history[side][key] = (k_all, v_all)
            k_all, v_all = self.history["target"][key]
            o_iso[seg] = R.causal_attention(q_t[seg], k_all, v_all, self.scale)
            k_all, v_all = self.history["ref"][key]
            o_ref[seg] = R.causal_attention(q_r[seg], k_all, v_all, self.scale)
            k_all, v_all = self.history["hf"][key]
            o_hf[seg] = sdpa_bf16(q_h[seg], k_all, v_all, self.scale)
        o_iso, o_ref, o_hf = (t.reshape(n, -1) for t in (o_iso, o_ref, o_hf))
        note("op", "attention", rec["attention"], o_iso)
        note("op", "o_proj", rec["update"], R.linear(rec["update_in"], w.o))
        s_op = stream_write(s_in, rec["update"], w.write_logit, w.retention_logit, cfg)
        note("op", "write", rec["s_out"], s_op, stream_scale(s_in, s_op))
        note(
            "op",
            "next_read",
            rec["next_reads"][0],
            normed_read(rec["s_out"], w.shared_read_logit, w.pre_moe_norm, cfg),
        )

        u_ref = R.linear(o_ref, w.o)
        s_ref = stream_write(s_in, u_ref, w.write_logit, w.retention_logit, cfg)
        read_ref = normed_read(s_ref, w.shared_read_logit, w.pre_moe_norm, cfg)
        scale = stream_scale(s_in, s_ref)
        note("module", "read", rec["x"], x_ref)
        note("module", "attention", rec["attention"], o_ref)
        note("module", "update", rec["update"], u_ref)
        note("module", "write", rec["s_out"], s_ref, scale)
        note("module", "next_read", rec["next_reads"][0], read_ref)

        u_hf = F.linear(o_hf, w.o)
        s_hf = stream_write(s_in, u_hf, w.write_logit, w.retention_logit, cfg)
        note("hf", "attention", o_hf, o_ref)
        note("hf", "update", u_hf, u_ref)
        note("hf", "write", s_hf, s_ref, scale)
        note(
            "hf",
            "next_read",
            normed_read(s_hf, w.shared_read_logit, w.pre_moe_norm, cfg),
            read_ref,
        )

        if rivals:
            for name in ATTENTION_RIVALS:
                u_rival = self._attention_rival(name, i, rows, q_r, u_ref, w)
                if u_rival is None:
                    continue
                retention = None if name == "no_retention" else w.retention_logit
                s_rival = stream_write(s_in, u_rival, w.write_logit, retention, cfg)
                note(f"rival_{name}", "update", u_rival, u_ref)
                note(f"rival_{name}", "write", s_rival, s_ref, scale)

    def _attention_rival(self, name, i, rows, q_r, u_ref, w):
        """The reference block's update with one planted mistake; None where the mistake changes
        nothing in this step (a history mistake on fresh prompts, a future token in decode)."""
        n = q_r.shape[0]
        if name == "no_retention":
            return u_ref
        if name == "no_history" and all(r.cached == 0 for r in rows):
            return None
        if name == "future_token" and all(len(r.tokens) == 1 for r in rows):
            return None
        out = torch.empty_like(q_r)
        for r, seg in self._segments(rows):
            k_all, v_all = self.history["ref"][(i, r.request.rid)]
            q = q_r[seg]
            scale = self.scale
            if name == "scale":
                scale = (self.cfg.head_dim // 2) ** -0.5
            elif name == "rope":
                pos = torch.arange(k_all.shape[0], device=q.device)
                q = rope(q, pos[r.cached :])
                k_all = rope(k_all, pos)
            elif name == "no_history":
                k_all, v_all = k_all[r.cached :], v_all[r.cached :]
            elif name == "kv_swapped":
                k_all, v_all = v_all, k_all
            elif name == "future_token":
                # Each query also sees the token after it (the last one sees itself twice).
                k_all = torch.cat([k_all, k_all[-1:]])
                v_all = torch.cat([v_all, v_all[-1:]])
            out[seg] = R.causal_attention(q, k_all, v_all, scale)
        return R.linear(out.reshape(n, -1), w.o)

    # --- the dense-MLP block ---------------------------------------------------------------------

    def _mlp(self, i, rec, rows, phase, rivals) -> None:
        cfg, w = self.cfg, self.mlp_weights[i]
        missing = {"s_in", "x", "up", "update_in", "update", "s_out", "next_reads"} - rec.keys()
        assert not missing, f"layer {i}: the observer saw no {sorted(missing)}"
        clamp = cfg.activation_clamp_value
        s_in = rec["s_in"]
        note = lambda c, o, out, ref, scale=None: self._note(  # noqa: E731
            phase, "mlp", c, o, i, errors(out, ref, scale)
        )

        x_ref = normed_read(s_in, w.read_logit, w.input_norm, cfg)
        note("op", "read", rec["x"], x_ref)
        note("op", "up", rec["up"], R.linear(rec["x"], w.up))
        note("op", "act", rec["update_in"], R.clamped_relu2(rec["up"], clamp))
        note("op", "down", rec["update"], R.linear(rec["update_in"], w.down))
        s_op = stream_write(s_in, rec["update"], w.write_logit, w.retention_logit, cfg)
        note("op", "write", rec["s_out"], s_op, stream_scale(s_in, s_op))
        for k, (logit, norm) in enumerate(w.next_reads):
            note(
                "op",
                f"next_read{k}",
                rec["next_reads"][k],
                normed_read(rec["s_out"], logit, norm, cfg),
            )

        h_ref = R.linear(x_ref, w.up)
        a_ref = R.clamped_relu2(h_ref, clamp)
        u_ref = R.linear(a_ref, w.down)
        s_ref = stream_write(s_in, u_ref, w.write_logit, w.retention_logit, cfg)
        reads_ref = [normed_read(s_ref, logit, norm, cfg) for logit, norm in w.next_reads]
        scale = stream_scale(s_in, s_ref)
        note("module", "read", rec["x"], x_ref)
        note("module", "up", rec["up"], h_ref)
        note("module", "act", rec["update_in"], a_ref)
        note("module", "update", rec["update"], u_ref)
        note("module", "write", rec["s_out"], s_ref, scale)
        for k, read_ref in enumerate(reads_ref):
            note("module", f"next_read{k}", rec["next_reads"][k], read_ref)

        u_hf = F.linear(R.clamped_relu2(F.linear(x_ref, w.up), clamp), w.down)
        s_hf = stream_write(s_in, u_hf, w.write_logit, w.retention_logit, cfg)
        note("hf", "update", u_hf, u_ref)
        note("hf", "write", s_hf, s_ref, scale)
        for k, ((logit, norm), read_ref) in enumerate(zip(w.next_reads, reads_ref)):
            note("hf", f"next_read{k}", normed_read(s_hf, logit, norm, cfg), read_ref)

        if rivals:
            for name in MLP_RIVALS:
                if name == "no_clamp":
                    u_rival = R.linear(R.clamped_relu2(h_ref, None), w.down)
                    retention = w.retention_logit
                else:
                    u_rival, retention = u_ref, None
                s_rival = stream_write(s_in, u_rival, w.write_logit, retention, cfg)
                note(f"rival_{name}", "update", u_rival, u_ref)
                note(f"rival_{name}", "write", s_rival, s_ref, scale)

    # --- the phases -------------------------------------------------------------------------------

    def run(self, prompts: list, md_unchunked, md_chunked, extra_lengths: list[int]) -> None:
        by_name = {p.name: p for p in prompts}
        short = [self.new_request(p.name, p.token_ids) for p in prompts if not p.long]
        req = {r.name: r for r in short}

        # context: the six short prompts in one varlen batch; two long prompts alone.
        self.step(md_unchunked, [Row(r, r.prompt) for r in short], "context", rivals=True)
        for name in ("gsm8k_fewshot", "news_digest"):
            r = req[name] = self.new_request(name, by_name[name].token_ids)
            self.step(md_unchunked, [Row(r, r.prompt)], "context", rivals=name == "news_digest")

        # chunked: a long prompt in 2048-token chunks over its own cached KV.
        r = req["gpqa_explain"] = self.new_request(
            "gpqa_explain", by_name["gpqa_explain"].token_ids
        )
        while r.in_context:
            chunk = r.prompt[r.done : r.done + CHUNK]
            self.step(md_chunked, [Row(r, chunk)], "chunked", rivals=r.done >= 2 * CHUNK)

        # mixed: the next long prompt's chunks beside the short prompts' generation rows.
        r = req["gsm8k_multiturn"] = self.new_request(
            "gsm8k_multiturn", by_name["gsm8k_multiturn"].token_ids
        )
        while r.in_context:
            gen = [Row(s, [s.generated[-1]]) for s in short]
            chunk = r.prompt[r.done : r.done + MIXED_CHUNK]
            self.step(md_chunked, [Row(r, chunk)] + gen, "mixed", rivals=r.done > 0)

        # More histories for the batch-64 decode: prompt prefixes of assorted lengths.
        sources = [by_name[n].token_ids for n in ("gsm8k_fewshot", "news_digest", "gpqa_explain")]
        extra = [
            self.new_request(f"prefix{k}", sources[k % len(sources)][:length])
            for k, length in enumerate(extra_lengths)
        ]
        batch: list[Row] = []
        for e in extra:
            if batch and sum(len(b.tokens) for b in batch) + len(e.prompt) > MAX_NUM_TOKENS:
                self.step(md_unchunked, batch, "context")
                batch = []
            batch.append(Row(e, e.prompt))
        if batch:
            self.step(md_unchunked, batch, "context")

        # decode: generation-only steps at batch 1, 3 and 64.
        for phase, names in (
            ("decode_b1", ["gsm8k_fewshot"]),
            ("decode_b3", ["news_digest", "gpqa_explain", "gsm8k_multiturn"]),
        ):
            for _ in range(2):
                self.step(
                    md_unchunked,
                    [Row(req[n], [req[n].generated[-1]]) for n in names],
                    phase,
                    rivals=phase == "decode_b3",
                )
        everyone = list(req.values()) + extra
        assert len(everyone) == MAX_BATCH, len(everyone)
        for _ in range(2):
            self.step(md_unchunked, [Row(e, [e.generated[-1]]) for e in everyone], "decode_b64")


#: Prompt prefixes for the batch-64 decode's extra histories (54 of them, with the ten prompts):
#: around the 32-token KV block (31, 32, 33), across 1-8K, odd lengths.
EXTRA_LENGTHS = [
    31,
    32,
    33,
    64,
    65,
    100,
    127,
    129,
    255,
    257,
    300,
    511,
    513,
    700,
    1000,
    1023,
    1025,
] + [1200 + 131 * k for k in range(37)]


def run_campaign(checkpoint: str, monkeypatch, log=print, kv_cache_manager_v2: bool = True):
    """Build everything and run every phase; ``Campaign.results`` holds the merged errors."""
    tokenizer = P.load_tokenizer(checkpoint)
    prompts = P.prompt_set(tokenizer)
    config, model = build_target(checkpoint)
    log(f"target loaded; free {_free_gib():.1f} GiB")
    manager = build_manager(config, kv_cache_manager_v2)
    log(f"{type(manager).__name__}; free {_free_gib():.1f} GiB")
    observer = BlockObserver(_modeling(), model.model, manager)
    observer.install(monkeypatch)
    md_unchunked = build_metadata(config, manager, chunked_prefill=False)
    md_chunked = build_metadata(config, manager, chunked_prefill=True)
    campaign = Campaign(checkpoint, model, manager, observer, log=log)
    with torch.inference_mode():
        campaign.run(prompts, md_unchunked, md_chunked, EXTRA_LENGTHS)
    return campaign


def report(results: dict, by_layer: bool = False) -> str:
    lines = []
    for key, e in sorted(summarize(results, by_layer).items()):
        phase, kind, comparison, output = key[:4]
        lines.append(
            f"{phase:10s} {kind:9s} {comparison:20s} {output:10s} "
            + (f"L{key[4]:<2d} " if by_layer else "")
            + f"rows {e['rows']:6d} worst_row {e['worst_row']:.3e} "
            f"({e['worst_row'] / BF16_ULP:6.2f} ulp, layer {e['layer']}, step {e['step']}) "
            f"rel_rms {e['rel_rms']:.3e} ({e['rel_rms'] / BF16_ULP:5.2f} ulp) median_row "
            f"{e['median_row']:.2e} identical {e['identical']:.4f} max_abs {e['max_abs']:.3e} "
            f"max_rel {e['max_rel']:.3e} worst {e['worst']}"
        )
    return "\n".join(lines)


def _free_gib() -> float:
    return torch.cuda.mem_get_info()[0] / 2**30


@pytest.fixture(scope="module")
def campaign():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if not os.environ.get("LLM_MODELS_ROOT"):
        pytest.skip(
            "set LLM_MODELS_ROOT (the checkpoint is $LLM_MODELS_ROOT/nemotron4-nano-preview)"
        )
    checkpoint = R.default_checkpoint()
    if not os.path.isdir(checkpoint):
        pytest.skip(f"{checkpoint} is not on disk")
    if _free_gib() < 220:
        pytest.skip("the whole tp1 target needs ~200 GiB of free GPU memory")
    mp = pytest.MonkeyPatch()
    result = None
    try:
        result = run_campaign(checkpoint, mp)
        yield result
    finally:
        mp.undo()
        if result is not None:
            result.close()


PHASES = ("context", "chunked", "mixed", "decode_b1", "decode_b3", "decode_b64")

# --- The bands ------------------------------------------------------------------------------------
#
# In BF16_ULP = 2^-8 of a row's scale (its largest |reference| value; for a stream write, the
# largest stream magnitude the write reads or writes, `stream_scale`): (worst element, relative RMS).
# One rounding flip of a row's largest element is at most 2 BF16_ULP. The reference rounds an
# IEEE-fp32 computation once to bf16 where HF rounds; every number cited is this campaign's own
# measurement, worst over every layer and phase, and none was widened to pass.

#: Each op fed the target's own input to it, against the reference op on that input.
OP_BANDS = {
    # wide_residual's read + RMSNorm against HF's order (two roundings): within one rounding flip
    # of the normalized value (op-wide-residual). Measured <= 2.56 / 0.02.
    ("attention", "read"): (4, 0.25),
    ("attention", "next_read"): (4, 0.25),
    ("mlp", "read"): (4, 0.25),
    ("mlp", "next_read"): (4, 0.25),
    # cublas_mm accumulates in fp32 in its own order and rounds once: one rounding flip where the
    # two fp32 sums straddle a bf16 boundary (the catalog cublas_mm band is 4 / 2). Measured
    # <= 1.98 / 0.17.
    ("attention", "qkv"): (4, 0.5),
    ("attention", "o_proj"): (4, 0.5),
    ("mlp", "up"): (4, 0.5),
    ("mlp", "down"): (4, 0.5),
    # thop_attention on the target's own q and cached K/V against fp32 softmax attention: the
    # kernels keep scores and softmax in fp32 and round P to bf16 for P.V (the catalog band is
    # 8 / 4). Measured <= 2.00 / 0.74; decode's split-KV combine puts its RMS above context's 0.45.
    ("attention", "attention"): (4, 1.5),
    # Bit-exact: softclamp_relu2 matches HF's torch ops on every bf16 pattern, and the stream
    # write is HF's fp32 r * S + w * U rounded once (op-softclamp-relu2, op-wide-residual).
    ("mlp", "act"): (0, 0),
    ("attention", "write"): (0, 0),
    ("mlp", "write"): (0, 0),
}

#: The whole block teacher-forced at its input streams S_in, against the reference block on S_in
#: (its attention over the reference's own K/V history). The same bands hold the block computed
#: in HF's own precision (bf16 cuBLAS GEMMs, bf16 SDPA): an implementation as accurate as the one
#: the model was trained and evaluated with must pass them.
MODULE_BANDS = {
    ("attention", "read"): OP_BANDS[("attention", "read")],
    ("mlp", "read"): OP_BANDS[("mlp", "read")],
    # The attention output on q/k/v that already carry the read's and the qkv GEMM's flips.
    # HF precision measured <= 2.81 / 0.53, the target <= 2.55 / 0.75.
    ("attention", "attention"): (4, 1.5),
    # o_proj's output, the block's update. HF precision <= 2.63 / 0.78, the target <= 2.00 / 1.07.
    ("attention", "update"): (4, 2),
    # One GEMM over a read within one flip. Measured <= 1.95 / 0.03.
    ("mlp", "up"): (4, 0.5),
    # relu^2 doubles its input's relative error: one flip of the up projection (<= 2) becomes <= 4
    # at the activation's largest element, plus its own rounding. Measured <= 4.68 / 0.08.
    ("mlp", "act"): (8, 0.5),
    # The MLP's update. HF precision and the target both <= 2.00 / 0.18 (the same cuBLAS GEMMs).
    ("mlp", "update"): (4, 1),
    # The write of an update within its band, against the operands' magnitude (`stream_scale`).
    ("attention", "write"): (4, 1),
    ("mlp", "write"): (4, 1),
    # The next block's read of the written streams: the RMSNorm divides by the read's RMS, which
    # lifts the streams' errors where the three streams partly cancel in the read. Attention:
    # measured <= 4.17 / 1.14; MLP <= 2.96 / 0.09.
    ("attention", "next_read"): (8, 2),
    ("mlp", "next_read"): (8, 1),
}


def _band(bands: dict, kind: str, output: str) -> tuple[float, float]:
    return bands[(kind, output.rstrip("0123456789"))]


def band_breaks(results: dict, comparison: str, bands: dict, phase: str, kind=None) -> list[str]:
    """Every (layer, output) of ``comparison`` in ``phase`` whose errors break its band."""
    out = []
    for (ph, k, comp, output, layer), e in sorted(results.items()):
        if comp != comparison or ph != phase or (kind is not None and k != kind):
            continue
        element, rms = _band(bands, k, output)
        if e["worst_row"] > element * BF16_ULP or e["rel_rms"] > rms * BF16_ULP:
            out.append(
                f"{k} layer {layer} {output}: worst element {e['worst_row'] / BF16_ULP:.2f} ulp "
                f"(band {element}), RMS {e['rel_rms'] / BF16_ULP:.3f} ulp (band {rms}); worst "
                f"{e['worst']} at step {e['step']}"
            )
    return out


EXPECTED = {
    ("attention", "op"): ("read", "qkv", "attention", "o_proj", "write", "next_read"),
    ("attention", "module"): ("read", "attention", "update", "write", "next_read"),
    ("attention", "hf"): ("attention", "update", "write", "next_read"),
    ("mlp", "op"): ("read", "up", "act", "down", "write", "next_read0"),
    ("mlp", "module"): ("read", "up", "act", "update", "write", "next_read0"),
    ("mlp", "hf"): ("update", "write", "next_read0"),
}


def test_campaign_covers_every_phase_layer_and_output(campaign):
    """Every block under test was checked in every phase, on every one of its outputs."""
    print(report(campaign.results))
    layers = dict(attention=campaign.observer.attention_layers, mlp=campaign.observer.mlp_layers)
    assert layers == dict(attention=[5, 9, 14, 19, 24, 29, 34, 38], mlp=[1, 3, 41, 43])
    missing = [
        (phase, kind, comparison, output, layer)
        for phase in PHASES
        for (kind, comparison), outputs in EXPECTED.items()
        for output in outputs
        for layer in layers[kind]
        if (phase, kind, comparison, output, layer) not in campaign.results
    ]
    assert not missing, f"never checked: {missing[:8]} ({len(missing)} in all)"
    steps = defaultdict(int)
    for step in campaign.steps:
        steps[step["phase"]] += 1
    assert set(steps) == set(PHASES), steps
    # The chunked and mixed phases attend over KV their earlier chunks wrote.
    assert max(s["max_history"] for s in campaign.steps if s["phase"] == "chunked") >= 3 * CHUNK
    assert max(s["max_history"] for s in campaign.steps if s["phase"] == "mixed") > 0
    assert max(s["rows"] for s in campaign.steps if s["phase"] == "decode_b64") == MAX_BATCH


@pytest.mark.parametrize("phase", PHASES)
def test_each_op_within_its_band(campaign, phase):
    """Each op of the attention and dense-MLP blocks, on the target's own input to it."""
    breaks = band_breaks(campaign.results, "op", OP_BANDS, phase)
    assert not breaks, "\n".join(breaks)


@pytest.mark.parametrize("phase", PHASES)
def test_each_block_within_its_band(campaign, phase):
    """Each attention and dense-MLP block, teacher-forced at its input streams."""
    breaks = band_breaks(campaign.results, "module", MODULE_BANDS, phase)
    assert not breaks, "\n".join(breaks)


@pytest.mark.parametrize("phase", PHASES)
def test_bands_admit_hf_precision(campaign, phase):
    """The block bands admit the same block computed in HF's own bf16 precision."""
    breaks = band_breaks(campaign.results, "hf", MODULE_BANDS, phase)
    assert not breaks, "\n".join(breaks)


@pytest.mark.parametrize(
    "kind,rival",
    [("attention", r) for r in ATTENTION_RIVALS] + [("mlp", r) for r in MLP_RIVALS],
)
def test_bands_reject_rival(campaign, kind, rival):
    """A planted mistake breaks the block bands, in every phase it was planted in."""
    comparison = f"rival_{rival}"
    phases = sorted({key[0] for key in campaign.results if key[1] == kind and key[2] == comparison})
    assert phases, f"{kind} rival {rival} was never planted"
    caught = {
        phase: band_breaks(campaign.results, comparison, MODULE_BANDS, phase, kind)
        for phase in phases
    }
    missed = [phase for phase, breaks in caught.items() if not breaks]
    assert not missed, f"{kind} rival {rival} passes every block band in {missed}"
