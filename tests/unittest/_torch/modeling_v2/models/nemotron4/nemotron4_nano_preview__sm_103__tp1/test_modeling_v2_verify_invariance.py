# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the nemotron4 tp1 MTP verify step against the decode steps of the same tokens.

Under one-model MTP-Eagle (``max_draft_len`` 2) a generation row carries its last accepted token
and two drafts, and the trunk verifies the three in one step: 3B rows through every GEMM, three
tokens per request through attention, the GDP mixers through the verify entries. Every entry of
that step but one gives token j the bits it gets as an MTP-off decode step of its own: the GEMMs
(``invariant_mm`` where cuBLAS is row-variant), the wide residual, the norms, the MoE and the GDP
and conv verify entries. Attention is the exception. A decode step attends through
``invariant_gen_attention``, a verify through ``thop_attention``'s generation kernels, which sum
the softmax in another order and pick their kernel by batch size and history length. So a verify
token is compared with its decode step three ways:

* **bitwise**, what it computes before attention's output first reaches it: the recurrent and
  conv state it leaves in the GDP layers before the first attention layer, and the K/V it appends
  in that layer. The verify writes no state slot, and the worker's commit
  (``update_mamba_states``) promotes exactly the state the verify stored for the last accepted
  token;
* **within a band**, everything downstream: its fp32 logits, and the state and K/V it leaves in
  every later layer. The bands are derived from measurement below (`LOGIT_BAND` and the rest);
* **reported**: whether a probe's verify rows move with the batch they run in -- thop's
  generation kernel is not batch-invariant, which is why ``invariant_gen_attention`` was written.

A last test follows MTP on along MTP off's own greedy tokens for as many tokens as the MTP
identity gate generates -- consecutive verifies that accept every draft, the state committed
after each -- and measures the drift of the logits at matched positions. The MTP identity gate's
near-tie band and the MTP acceptance gate's verify term rest on it (`DRIFT_GAP_BAND`,
`DRIFT_LOGIT_BAND`).

What runs is the target and the engine's own pieces around it, built as the engine builds them
under ``MTPDecodingConfig(max_draft_len=2)``: the spec config resolved against the checkpoint
(one-model MTP-Eagle over the one repeated MTP layer); ``ModelConfig`` off the checkpoint, the
target class with every weight loaded, the MTP layer's included; the trunk's hybrid cache
manager (``_create_kv_cache_manager``, the default ``MambaHybridCacheManagerV2``, with the
per-position buffers) and the MTP layer's separate draft KV manager; the engine's TRTLLM
attention metadata, filled per step as the runner fills it; and, for graphs,
``create_cuda_graph_metadata`` at draft length 2 and the runner's MTP padding dummy
(``CUDA_GRAPH_DUMMY_REQUEST_ID - 2``), whose rows pad every graph batch.

The requests: the ten prompts of the fixed set, two prefixes of the long ones and 244 more
prefixes as fillers, prefilled through the target's own prefill steps, the MTP layer's first
draft step over each prompt included, as the worker runs it. A probe's reference is three
MTP-off decode steps run alone, eagerly, from the state the prefill left: over t0, its greedy
next token, then t1 and t2, the greedy tokens after it -- the drafts a perfect drafter would
propose. Then the verify step over [t0, t1, t2] from the same state: alone, inside eager
batches, and inside padded CUDA graphs. Every run starts from the same state: each GDP slot is
restored from a snapshot, and a row's appends land where its own attention reads only after
writing them. One more request sits at the length limit the engine admits, its history zeroed
rather than prefilled: its verify reaches past ``max_seq_len`` by the drafts.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os

import pytest
import torch

__extra_import_path__ = ["..", "../../.."]
import _nemotron4_prompts as P  # noqa: E402 -- needs the path declared above
import _nemotron4_reference as R  # noqa: E402
from _validating import validating  # noqa: E402

_MODELING = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4."
    "nemotron4_nano_preview__sm_103__tp1.modeling"
)

# The engine sizes this run uses everywhere (task.yaml, the boot gate, check_goldens).
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
TOKENS_PER_BLOCK = 32
#: The checkpoint's trained draft depth; a verify row carries 1 + DRAFT tokens.
DRAFT = 2
TOKENS = 1 + DRAFT
#: Requests resident at once: the probes and fillers, and the padding dummy beside them.
NUM_REQUESTS = 256
#: Rows a step may hold.
MAX_BATCH = 264
#: KV tokens of the paged pools: every request stays resident, the probes with room for the drift.
KV_TOKENS = 192 * 1024
#: Verify batch sizes: 1 to 256, and two that leave a large graph mostly or slightly padded.
BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8, 16, 64, 100, 250, 256)
#: The CUDA-graph batch sizes captured; a batch runs in the smallest one that holds it, as the
#: runner pads.
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
#: Tokens the MTP identity gate generates per prompt: the first comes from the prefill, the rest
#: from verify steps. The drift follows each probe for the verifies of that many tokens.
SHORT_TOKENS, LONG_TOKENS = 256, 1024
#: The decode reference's top tokens whose gap to its top-1 a verify must keep (`Deviation.gap`).
TOP = 8

# The bands. A verify token differs from its decode step only through the attention kernel:
# thop_attention's generation kernels against invariant_gen_attention, both an fp32 softmax over
# the same bf16 K/V rounded once to bf16, summed in other orders. That leaves attention outputs a
# bf16 rounding apart, which every later layer carries into the token's state, its K/V and its
# logits; this model amplifies such differences (a decode step and a teacher-forced one, apart
# only in GEMM row counts, differ by up to 1.15 in their logits). How far is measured here, on
# this model's own state: every probe's three tokens alone (36), in eager batches of 2 to 256
# (396) and in padded graphs of 1 to 256 (432). The worst cases, the same in every run (the
# kernels are deterministic): |logit diff| 0.5371 alone and 1.1815 in batches and graphs, where
# thop picks other kernel variants; top-gap shift 0.3120 alone, 0.3991 in batches; the later
# layers' recurrent state, conv state and K/V 0.0462 / 0.087 / 0.0863 of max |decode|. Each band
# is its worst case plus a quarter, rounded up: room for a batch, history or graph size the runs
# did not take, where thop may pick yet another variant.
#: max |verify logit - decode logit| over the vocabulary, fp32.
LOGIT_BAND = 1.48
#: The largest change, verify against decode, of the gap between the decode step's top-1 and any
#: of its top `TOP` tokens: how close to a tie the verify can tip the decode step's choice.
GAP_BAND = 0.50
#: max |verify - decode| / max |decode|, per tensor, over the GDP layers after the first
#: attention layer (recurrent state fp32, conv state bf16) and the attention layers after it (K/V).
SSM_BAND = 0.058
CONV_BAND = 0.11
KV_BAND = 0.11
# The drift: the same two quantities over the 6132 matched positions of
# `test_the_drift_over_consecutive_verifies`, where the verify's state and K/V carry each earlier
# verify's deviation. Worst cases 5.0471 (|logit diff|, on the 2-shot chat `clips`; 0.68-1.5
# on every other probe) and 0.7819 (top-gap shift), plus a quarter, rounded up. The MTP identity
# gate's near-tie band is DRIFT_GAP_BAND; the MTP acceptance gate's verify term is the measured
# |logit diff|.
DRIFT_LOGIT_BAND = 6.31
DRIFT_GAP_BAND = 0.98


def _modeling():
    return importlib.import_module(_MODELING)


def _mapping():
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)


def _spec_config(path: str):
    """``MTPDecodingConfig(max_draft_len=2)`` resolved against the checkpoint as the model loader
    resolves it: one-model MTP-Eagle over the one repeated MTP layer."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    spec = MTPDecodingConfig(max_draft_len=DRAFT)
    update_spec_config_from_model_config(
        spec, ModelConfig.from_pretrained(path, trust_remote_code=True).pretrained_config
    )
    assert spec.spec_dec_mode.is_mtp_eagle_one_model(), spec.spec_dec_mode
    assert spec.max_draft_len == DRAFT and spec.tokens_per_gen_step == TOKENS
    return spec


def _cache_max_seq_len(spec) -> int:
    """The cache managers' bound on a sequence, as ``create_py_executor`` sets it under one-model
    speculative decoding with the overlap scheduler on: the drafts of the step in flight, the
    extra KV tokens of the one-engine drafter, and the drafts of the next step."""
    from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens

    return MAX_SEQ_LEN + 2 * (spec.tokens_per_gen_step - 1) + get_num_extra_kv_tokens(spec)


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


def _build_target(path: str, spec):
    """``(ModelConfig, the loaded target)``: the target class off the checkpoint's config with
    the spec config attached, every weight loaded through its own ``load_weights``."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype

    config = ModelConfig.from_pretrained(path, trust_remote_code=True, spec_config=spec)
    validate_and_set_mamba_ssm_cache_dtype(config, "auto")
    with torch.device("cuda"):
        model = _modeling().ModelingV2Nemotron4NanoPreviewSm103Tp1(config)
    model.load_weights(_checkpoint_tensors(path))
    model.post_load_weights()
    torch.cuda.synchronize()
    return config, model.eval()


def _build_managers(config, spec):
    """The engine's two cache managers under one-model MTP on tp1: the trunk's hybrid manager
    (paged KV for its attention layers, recurrent and conv state slots and the per-position
    buffers for its GDP layers) and the separate draft KV manager holding the MTP layer's KV."""
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.speculative.utils import get_num_spec_layers
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    num_layers = config.pretrained_config.num_hidden_layers
    num_draft_layers = get_num_spec_layers(spec)

    def manager(**draft):
        kv_cache_config = KvCacheConfig(
            max_tokens=KV_TOKENS, enable_block_reuse=False, free_gpu_memory_fraction=0.5
        )
        return _create_kv_cache_manager(
            None,
            get_kv_cache_manager_cls(config, kv_cache_config),
            _mapping(),
            kv_cache_config,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=_cache_max_seq_len(spec),
            max_batch_size=MAX_BATCH,
            spec_config=spec,
            sparse_attention_config=None,
            max_num_tokens=MAX_NUM_TOKENS,
            max_beam_width=1,
            kv_connector_manager=None,
            model_config=config,
            dtype=torch.bfloat16,
            max_cuda_graph_batch_size=max(GRAPH_SIZES),
            **draft,
        )

    trunk = manager(layer_mask=[True] * num_layers)
    draft = manager(
        is_draft=True,
        layer_mask=[False] * num_layers + [True] * num_draft_layers,
        num_layers=num_draft_layers,
    )
    return trunk, draft


def _build_metadata(config, manager, draft_manager):
    """The engine's TRTLLM attention metadata, chunked prefill off, with the draft manager the
    worker swaps in for the MTP layer's steps."""
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
            chunked_prefill=False,
            cache_reuse=False,
            has_speculative_draft_tokens=True,
            chunk_size=MAX_NUM_TOKENS,
        ),
        mapping=_mapping(),
        cache_indirection=None,
        kv_cache_manager=manager,
        draft_kv_cache_manager=draft_manager,
    )
    md.mamba_metadata_cls = _modeling().Nemotron4StateMetadata.for_draft_len(DRAFT)
    return md


@dataclasses.dataclass
class Request:
    """One sequence: its prompt, the three tokens its verify step feeds -- the prompt's greedy
    next token t0, then t1 and t2 -- and how many tokens the MTP identity gate generates for it."""

    rid: int
    name: str
    prompt: tuple[int, ...]
    tokens: tuple[int, ...] = ()
    generated: int = SHORT_TOKENS

    @property
    def drift_steps(self) -> int:
        """The verifies of the drift: every generated token but the prefill's, three a step."""
        return (self.generated - 1) // TOKENS


def _requests(prompts) -> tuple[list[Request], list[Request]]:
    """The probes -- the ten prompts, a history of 511 tokens, whose verify tokens straddle the
    first 512-key attention split and a page boundary, and one that ends mid-split -- and
    fillers enough for a batch of 256."""
    by_name = {p.name: p for p in prompts}
    probes = [
        Request(0, p.name, p.token_ids, generated=LONG_TOKENS if p.long else SHORT_TOKENS)
        for p in prompts
    ]
    probes.append(Request(0, "gpqa_explain[:511]", by_name["gpqa_explain"].token_ids[:511]))
    probes.append(Request(0, "gsm8k_multiturn[:4096]", by_name["gsm8k_multiturn"].token_ids[:4096]))
    sources = [
        by_name[n].token_ids
        for n in ("gsm8k_fewshot", "gsm8k_multiturn", "news_digest", "gpqa_explain")
    ]
    fillers = [
        Request(0, f"filler{k}", sources[k % len(sources)][: 24 + (37 * k) % 600])
        for k in range(NUM_REQUESTS - len(probes))
    ]
    for rid, r in enumerate(probes + fillers, start=1):
        r.rid = rid
    return probes, fillers


@dataclasses.dataclass
class Token:
    """What one token's step produced: its logits and what it wrote."""

    logits: torch.Tensor  # [vocab] fp32
    ssm: list[torch.Tensor]  # per GDP layer, the state after this token: [24, 64, 128] fp32
    conv: list[torch.Tensor]  # per GDP layer, the state after this token: [16896, 3] bf16
    kv: torch.Tensor  # [attention layers, K/V, head_dim] bf16, at this token's position


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view({4: torch.int32, 2: torch.int16}[x.element_size()])


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(_bits(a), _bits(b))


def _relative(got: torch.Tensor, ref: torch.Tensor) -> float:
    """max |got - ref| / max |ref|."""
    ref = ref.float()
    return float((got.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _logit_deviation(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """``(max |got - ref|, the largest change of a gap between ref's top-1 and one of its top
    TOP)``: the second bounds how far ``got`` can tip a choice ``ref`` made among its top tokens."""
    diff = got - ref
    top = ref.topk(TOP).indices
    shift = diff[top] - diff[top[0]]
    return float(diff.abs().max()), float(shift.abs().max())


def _margin(logits: torch.Tensor) -> float:
    """The top-1 minus top-2 logit: how close a greedy choice is to a tie."""
    two = logits.topk(2).values
    return float(two[0] - two[1])


@dataclasses.dataclass
class Deviation:
    """How a verify token compares with its decode step."""

    exact: list[str]  # bitwise quantities that differ: must be none
    logit: float
    gap: float
    ssm: float
    conv: float
    kv: float
    argmax: tuple[int, int]  # (verify, decode)
    margin: float  # the decode step's top-1 / top-2 margin

    def outside(self) -> list[str]:
        """What lies outside the bands."""
        out = list(self.exact)
        for name, value, band in (
            ("logits", self.logit, LOGIT_BAND),
            ("top-gap shift", self.gap, GAP_BAND),
            ("recurrent state", self.ssm, SSM_BAND),
            ("conv state", self.conv, CONV_BAND),
            ("K/V", self.kv, KV_BAND),
        ):
            if value > band:
                out.append(f"{name} {value:.4g} outside its band {band:.4g}")
        return out


def _deviation(got: Token, ref: Token, upstream: int, gdp_layers) -> Deviation:
    """Compare a verify token with its decode step: bitwise the GDP layers before the first
    attention layer (global index below ``upstream``) and the first attention layer's K/V,
    by deviation everything after."""
    exact = []
    for name, got_states, ref_states in (("ssm", got.ssm, ref.ssm), ("conv", got.conv, ref.conv)):
        bad = [
            layer
            for layer, a, b in zip(gdp_layers, got_states, ref_states)
            if layer < upstream and not _equal(a, b)
        ]
        if bad:
            exact.append(f"{name} state differs bitwise in GDP layers {bad}, before attention")
    if not _equal(got.kv[0], ref.kv[0]):
        exact.append("the K/V appended in the first attention layer differs bitwise")
    later = [k for k, layer in enumerate(gdp_layers) if layer > upstream]
    logit, gap = _logit_deviation(got.logits, ref.logits)
    return Deviation(
        exact=exact,
        logit=logit,
        gap=gap,
        ssm=max(_relative(got.ssm[k], ref.ssm[k]) for k in later),
        conv=max(_relative(got.conv[k], ref.conv[k]) for k in later),
        kv=max(_relative(a, b) for a, b in zip(got.kv[1:], ref.kv[1:])),
        argmax=(int(got.logits.argmax()), int(ref.logits.argmax())),
        margin=_margin(ref.logits),
    )


class Harness:
    """The loaded target, its cache managers and metadata, the prefilled requests, and the state
    every run starts from."""

    def __init__(self, checkpoint: str, log=print):
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
        from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens

        self.log = log
        tokenizer = P.load_tokenizer(checkpoint)
        self.spec = _spec_config(checkpoint)
        self.config, self.model = _build_target(checkpoint, self.spec)
        self.manager, self.draft_manager = _build_managers(self.config, self.spec)
        self.md = _build_metadata(self.config, self.manager, self.draft_manager)
        self.core = self.model.model
        self.lm_head = self.model.lm_head.weight
        self.gdp_layers = self.core.gdp_layers
        self.attention_layers = self.core.attention_layers
        #: Everything a token computes in layers below the first attention layer is upstream of
        #: attention's output.
        self.upstream = min(self.attention_layers)
        self.vocab = self.config.pretrained_config.text_config.vocab_size
        self.probes, self.fillers = _requests(P.prompt_set(tokenizer))
        # The longest history a verify starts from: the request's last accepted token is its
        # MAX_SEQ_LEN - 1-th, so its verify's lengths reach MAX_SEQ_LEN - 1 + DRAFT.
        self.limit = Request(
            len(self.probes) + len(self.fillers) + 1,
            "at the length limit",
            (0,) * (MAX_SEQ_LEN - 2),
        )
        requests = self.probes + self.fillers
        for r in requests + [self.limit]:
            # The prompt and the step's three tokens, a probe's drift besides, in both managers;
            # the manager adds the drafter's extra KV token itself.
            drift = TOKENS * r.drift_steps if r in self.probes else 0
            added = self.manager.add_dummy_requests(
                [r.rid],
                token_nums=[len(r.prompt) + drift + TOKENS],
                draft_kv_cache_manager=self.draft_manager,
            )
            assert added, f"the cache managers have no room for {r.name} ({len(r.prompt)} tokens)"
        self.extra_kv = get_num_extra_kv_tokens(self.spec)
        # The runner's MTP padding dummy: a generation request carrying two draft tokens.
        self.dummy = CUDA_GRAPH_DUMMY_REQUEST_ID - DRAFT
        self._prefill(requests)
        added = self.manager.add_dummy_requests(
            [self.dummy],
            is_gen=True,
            max_num_draft_tokens=DRAFT,
            draft_kv_cache_manager=self.draft_manager,
        )
        assert added, "the cache manager has no room for the CUDA-graph padding dummy"
        ids = [r.rid for r in requests + [self.limit]]
        self.slot = dict(zip(ids, self.manager.get_state_indices(ids)))
        self.dummy_slot = self.manager.get_state_indices([self.dummy], [True])[0]
        assert len(set(self.slot.values()) | {self.dummy_slot}) == len(ids) + 1
        self.kv_buffers = {i: self.manager.get_buffers(i) for i in self.attention_layers}
        self.pages = {
            r.rid: {
                i: self.manager.get_batch_cache_indices([r.rid], layer_idx=i)[0]
                for i in self.attention_layers
            }
            for r in requests + [self.limit]
        }
        # The request at the limit: zero history, a fresh slot, and t0 a token of the text.
        for i in self.attention_layers:
            pages = torch.tensor(self.pages[self.limit.rid][i], dtype=torch.long, device="cuda")
            self.kv_buffers[i][pages] = 0
        for i in self.gdp_layers:
            self.manager.get_ssm_states(i)[self.slot[self.limit.rid]] = 0
            self.manager.get_conv_states(i)[self.slot[self.limit.rid]] = 0
        self.limit.tokens = self.probes[0].tokens
        self.snapshot = [
            (self.manager.get_ssm_states(i).clone(), self.manager.get_conv_states(i).clone())
            for i in self.gdp_layers
        ]
        self.static_ids = torch.zeros(max(GRAPH_SIZES) * TOKENS, dtype=torch.int, device="cuda")
        self.graphs: dict[int, tuple] = {}
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.random_tokens = torch.Generator(device="cuda").manual_seed(9)
        self.alone: dict[int, torch.Tensor] = {}

    # --- steps -----------------------------------------------------------------------------------

    def fill(
        self,
        md,
        rows: list[Request],
        num_contexts: int,
        tokens: int = 1,
        cached: list[int] | None = None,
        padding: int = 0,
    ) -> None:
        """The step's per-request fields, set as the runner's ``_prepare_tp_inputs`` and the
        speculative step's ``update_spec_dec_param`` set them, then ``prepare()``. Context rows
        feed their whole prompt; generation rows feed ``tokens`` tokens over ``cached`` (default:
        the prompt); ``padding`` rows are the padding dummy's, a generation request of
        1 + D tokens of which D are history, as the runner fills it."""
        from tensorrt_llm._torch.metadata import KVCacheParams

        gen = rows[num_contexts:]
        cached = [len(r.prompt) for r in gen] if cached is None else cached
        if not md.is_cuda_graph:
            md.seq_lens = torch.tensor(
                [len(r.prompt) for r in rows[:num_contexts]] + [tokens] * len(gen),
                dtype=torch.int,
            )
        md.beam_width = 1
        md.request_ids = [r.rid for r in rows] + [self.dummy] * padding
        md.prompt_lens = [len(r.prompt) for r in rows] + [DRAFT] * padding
        md.num_contexts = num_contexts
        md.num_chunked_ctx_requests = 0
        md.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_contexts + list(cached) + [DRAFT] * padding,
            num_extra_kv_tokens=self.extra_kv,
        )
        md.kv_cache_manager = self.manager
        md.padded_num_tokens = None
        md.update_spec_dec_param(
            batch_size=len(rows) + padding,
            is_spec_decoding_enabled=True,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=DRAFT,
            max_total_draft_tokens=DRAFT,
            num_contexts=num_contexts,
        )
        md.prepare()

    def trunk(self, md, ids: torch.Tensor) -> torch.Tensor:
        """The trunk's fp32 logits for every token: the core's forward, then the lm_head the
        shell runs (``cublas_mm`` with an fp32 output) on every row, as it does for the tokens a
        decode step or a verify step gathers."""
        M = _modeling()
        hidden = self.core.forward(attn_metadata=md, input_ids=ids)
        return M.cublas_mm(hidden, self.lm_head.t(), None, torch.float32)

    def mtp(self, md, target, input_ids, hidden) -> torch.Tensor:
        """One MTP-layer draft step through ``target`` (an ``mtp_forward``) on the draft KV
        manager, swapped in as the worker swaps it."""
        from tensorrt_llm._torch.speculative.interface import (
            prepare_attn_metadata_for_draft_replay,
            restore_attn_metadata_after_draft_replay,
        )

        saved = prepare_attn_metadata_for_draft_replay(md, self.draft_manager)
        assert saved is not None, "the metadata carries no draft KV block offsets"
        try:
            return target(self.core.w["embed"], input_ids, hidden, md)
        finally:
            restore_attn_metadata_after_draft_replay(md, saved)

    def _prefill(self, requests: list[Request]) -> None:
        """Every prompt through the target's prefill steps, whole prompts packed up to
        ``MAX_NUM_TOKENS`` tokens a step, then the MTP layer's first draft step over the same
        rows, as the worker runs it (each token's hidden state with the next token's
        embedding), which fills the MTP layer's KV. A request's t0 is its greedy next token."""
        mtp_layer = self.model.draft_model.mtp_layers[0]
        batch: list[Request] = []
        for r in requests + [None]:
            if batch and (
                r is None or sum(len(b.prompt) for b in batch) + len(r.prompt) > MAX_NUM_TOKENS
            ):
                self.fill(self.md, batch, num_contexts=len(batch))
                ids = torch.tensor(
                    [t for b in batch for t in b.prompt], dtype=torch.int, device="cuda"
                )
                hidden = self.core.forward(attn_metadata=self.md, input_ids=ids)
                last = torch.cumsum(self.md.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                M = _modeling()
                logits = M.cublas_mm(hidden[last], self.lm_head.t(), None, torch.float32)
                for b, token in zip(batch, logits.argmax(-1).tolist()):
                    b.tokens = (token,)
                shifted = torch.tensor(
                    [t for b in batch for t in b.prompt[1:] + b.tokens],
                    dtype=torch.int,
                    device="cuda",
                )
                self.mtp(
                    self.md,
                    lambda e, i, h, md: mtp_layer(e, None, i, h, md),
                    shifted,
                    hidden,
                )
                batch = []
            if r is not None:
                batch.append(r)
        torch.cuda.synchronize()
        self.log(f"prefilled {len(requests)} requests, the MTP layer's KV included")

    def restore(self) -> None:
        """Every GDP slot back to its state after the prefill."""
        for i, (ssm, conv) in zip(self.gdp_layers, self.snapshot):
            self.manager.get_ssm_states(i).copy_(ssm)
            self.manager.get_conv_states(i).copy_(conv)

    def changed_slots(self) -> set[int]:
        """The slots whose recurrent or conv state differs from the snapshot, bitwise, in any
        GDP layer."""
        changed = set()
        for i, (ssm, conv) in zip(self.gdp_layers, self.snapshot):
            pools = (self.manager.get_ssm_states(i), ssm), (self.manager.get_conv_states(i), conv)
            for pool, saved in pools:
                diff = (_bits(pool) != _bits(saved)).flatten(1).any(dim=1)
                changed |= set(torch.nonzero(diff).flatten().tolist())
        return changed

    def _ids(self, rows: list[Request], padding: int) -> torch.Tensor:
        """A verify step's input ids, request-major: each row's three tokens, then arbitrary
        tokens on the padding rows, as the runner leaves whatever its static buffer held."""
        real = torch.tensor([t for r in rows for t in r.tokens], dtype=torch.int, device="cuda")
        pad = torch.randint(
            0,
            self.vocab,
            (padding * TOKENS,),
            generator=self.random_tokens,
            dtype=torch.int,
            device="cuda",
        )
        return torch.cat([real, pad])

    def decode(self, r: Request, j: int) -> torch.Tensor:
        """MTP-off decode of ``r``'s token j alone, eagerly, over the prompt and tokens 0..j-1
        already in the cache and in its slot; the token's logits row."""
        self.fill(self.md, [r], num_contexts=0, tokens=1, cached=[len(r.prompt) + j])
        ids = torch.tensor([r.tokens[j]], dtype=torch.int, device="cuda")
        return self.trunk(self.md, ids)[0]

    def eager(self, rows: list[Request]) -> torch.Tensor:
        """One verify step of exactly these rows, eagerly, from the snapshot; every token's
        logits, request-major."""
        self.restore()
        self.fill(self.md, rows, num_contexts=0, tokens=TOKENS)
        return self.trunk(self.md, self._ids(rows, 0))

    def graph_size(self, num_rows: int) -> int:
        return min(g for g in GRAPH_SIZES if g >= num_rows)

    def graph(
        self,
        rows: list[Request],
        ids: torch.Tensor | None = None,
        cached: list[int] | None = None,
        restore: bool = True,
    ) -> torch.Tensor:
        """One verify step of these rows replayed in the smallest captured graph that holds
        them, padded with the padding dummy's rows; the real rows' logits. It feeds each row's
        three tokens over its prompt from the snapshot, or ``ids`` over ``cached`` tokens from
        the slots as they are when ``restore`` is off."""
        size = self.graph_size(len(rows))
        padding = size - len(rows)
        gmd, graph, out = self.graphs[size]
        assert gmd.cuda_graph_workspace.data_ptr() == self.workspace_address, (
            "the attention workspace moved under the captured graphs"
        )
        if restore:
            self.restore()
        self.fill(gmd, rows, num_contexts=0, tokens=TOKENS, cached=cached, padding=padding)
        if ids is None:
            ids = self._ids(rows, padding)
        self.static_ids[: ids.numel()].copy_(ids)
        graph.replay()
        return out[: len(rows) * TOKENS]

    def capture_graphs(self) -> None:
        """Capture the verify step at every size of GRAPH_SIZES, as the runner does: first one
        eager forward on every size's own CUDA-graph metadata (draft length 2), which grows the
        attention workspace all of them share to what the largest call needs -- thop_attention
        resizes it in place when a call needs more -- then at each size two warmup forwards and
        the capture, which resize nothing. A resize after a capture would move the workspace
        under the graph, which keeps its old address."""
        from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

        rows = self.probes + self.fillers
        metadata = {}
        for size in sorted(GRAPH_SIZES, reverse=True):
            gmd = self.md.create_cuda_graph_metadata(size, False, DRAFT, get_memory_buffers())
            self.fill(gmd, rows[:size], num_contexts=0, tokens=TOKENS)
            self.static_ids[: size * TOKENS].copy_(self._ids(rows[:size], 0))
            self.trunk(gmd, self.static_ids[: size * TOKENS])
            metadata[size] = gmd
        torch.cuda.synchronize()
        self.workspace_address = metadata[max(GRAPH_SIZES)].cuda_graph_workspace.data_ptr()
        for size in GRAPH_SIZES:
            gmd = metadata[size]
            self.fill(gmd, rows[:size], num_contexts=0, tokens=TOKENS)
            self.static_ids[: size * TOKENS].copy_(self._ids(rows[:size], 0))
            ids = self.static_ids[: size * TOKENS]
            for _ in range(2):
                self.trunk(gmd, ids)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                out = self.trunk(gmd, ids)
            torch.cuda.synchronize()
            self.graphs[size] = (gmd, graph, out)
        assert gmd.cuda_graph_workspace.data_ptr() == self.workspace_address, (
            "a capture resized the attention workspace"
        )
        self.restore()

    def commit(self, md, accepted: list[int]) -> None:
        """The worker's commit after a verify step: ``update_mamba_states`` with the engine's own
        slots, padding rows included, each row's accepted count (its last token's, 1 to 3)."""
        counts = torch.tensor(accepted, dtype=torch.int32, device="cuda")
        self.manager.update_mamba_states(
            attn_metadata=md,
            num_accepted_tokens=counts,
            state_indices=md.mamba_metadata.state_indices,
        )

    # --- what a run left -------------------------------------------------------------------------

    def kv_at(self, r: Request, layer: int, j: int) -> torch.Tensor:
        """``[K/V, head_dim]``: what attention layer ``layer`` holds at ``r``'s token j, read
        through the manager's own page table and buffer view."""
        pos = len(r.prompt) + j
        page = self.pages[r.rid][layer][pos // TOKENS_PER_BLOCK]
        return self.kv_buffers[layer][page, :, pos % TOKENS_PER_BLOCK, 0].clone()

    def slot_token(self, r: Request, j: int, logits: torch.Tensor) -> Token:
        """Token j of ``r`` as a decode step left it: its logits, its slot's state, its K/V."""
        slot = self.slot[r.rid]
        return Token(
            logits=logits.clone(),
            ssm=[self.manager.get_ssm_states(i)[slot].clone() for i in self.gdp_layers],
            conv=[self.manager.get_conv_states(i)[slot].clone() for i in self.gdp_layers],
            kv=torch.stack([self.kv_at(r, i, j) for i in self.attention_layers]),
        )

    def verify_token(self, r: Request, row: int, j: int, logits: torch.Tensor) -> Token:
        """Token j of ``r`` as generation row ``row`` of a verify left it: its logits, the
        state the verify stored for it in the per-position buffers, its K/V."""
        return Token(
            logits=logits[row * TOKENS + j].clone(),
            ssm=[
                self.manager.get_intermediate_ssm_states(i)[row, j].clone() for i in self.gdp_layers
            ],
            conv=[
                self.manager.get_intermediate_conv_states(i)[row, j].clone()
                for i in self.gdp_layers
            ],
            kv=torch.stack([self.kv_at(r, i, j) for i in self.attention_layers]),
        )

    def run_reference(self, r: Request) -> tuple[list[Token], torch.Tensor]:
        """``r``'s three MTP-off decode steps alone, eagerly, from the snapshot: what each token
        left, and the trunk's hidden state of t0 (the MTP layer's input for the next token).
        Fixes ``r.tokens`` to t0 and the greedy tokens after it."""
        self.restore()
        tokens, hidden0 = [], None
        for j in range(TOKENS):
            if j == 0:
                self.fill(self.md, [r], num_contexts=0, tokens=1)
                ids = torch.tensor([r.tokens[0]], dtype=torch.int, device="cuda")
                hidden0 = self.core.forward(attn_metadata=self.md, input_ids=ids).clone()
                logits = _modeling().cublas_mm(hidden0, self.lm_head.t(), None, torch.float32)[0]
            else:
                logits = self.decode(r, j)
            tokens.append(self.slot_token(r, j, logits))
            if j + 1 < TOKENS:
                r.tokens = r.tokens[: j + 1] + (int(logits.argmax()),)
        self.restore()
        return tokens, hidden0

    def alone_logits(self, r: Request) -> torch.Tensor:
        """``r``'s verify logits alone, eagerly, from the snapshot: what its rows in a batch are
        held against bitwise, to report whether they move with the batch."""
        if r.rid not in self.alone:
            self.alone[r.rid] = self.eager([r])[:TOKENS].clone()
        return self.alone[r.rid]

    def compare(self, rows: list[Request], positions: list[int], logits) -> list[Deviation]:
        """Every token of the probes at ``positions`` against its decode step."""
        return [
            _deviation(
                self.verify_token(rows[pos], pos, j, logits),
                self.reference[rows[pos].rid][j],
                self.upstream,
                self.gdp_layers,
            )
            for pos in positions
            for j in range(TOKENS)
        ]


def _batches(h: Harness, size: int) -> list[tuple[list[Request], list[int]]]:
    """Batches of ``size`` verify rows that hold every probe at least once, each with the row
    positions of its probes. A batch at least as large as the probe set spreads them from the
    first row to the last; smaller batches take the probes in turn, reversed every other batch,
    and fill the rest with fillers. Each size takes its fillers from its own offset."""
    probes, fillers = h.probes, h.fillers
    offset = (size * 7) % len(fillers)
    rotated = fillers[offset:] + fillers[:offset]
    if size >= len(probes):
        positions = [round(k * (size - 1) / (len(probes) - 1)) for k in range(len(probes))]
        rows, spare = [], iter(rotated)
        for p in range(size):
            rows.append(probes[positions.index(p)] if p in positions else next(spare))
        return [(rows, positions)]
    out = []
    for start in range(0, len(probes), size):
        chunk = probes[start : start + size]
        if (start // size) % 2:
            chunk = chunk[::-1]
        rows = chunk + rotated[: size - len(chunk)]
        out.append((rows, list(range(len(chunk)))))
    return out


def _free_gib() -> float:
    return torch.cuda.mem_get_info()[0] / 2**30


@pytest.fixture(scope="module")
def harness():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    checkpoint = R.default_checkpoint()
    if not os.path.isdir(checkpoint):
        pytest.skip(f"{checkpoint} is not on disk")
    if _free_gib() < 245:
        pytest.skip("the whole tp1 target with its MTP layer and buffers needs ~230 GiB free")
    with torch.inference_mode():
        h = Harness(checkpoint)
        # Fillers verify their t0 and two arbitrary tokens.
        g = torch.Generator().manual_seed(17)
        for r in h.fillers:
            r.tokens = r.tokens + tuple(torch.randint(0, h.vocab, (DRAFT,), generator=g).tolist())
        # The reference: every probe's three decode steps alone, eagerly.
        h.reference, h.hidden0 = {}, {}
        for r in h.probes + [h.limit]:
            h.reference[r.rid], h.hidden0[r.rid] = h.run_reference(r)
        h.capture_graphs()
    yield h
    h.graphs.clear()
    h.manager.shutdown()
    h.draft_manager.shutdown()


def _summary(deviations: list[Deviation]) -> str:
    flips = [d for d in deviations if d.argmax[0] != d.argmax[1]]
    return (
        f"{len(deviations)} tokens: max |logit diff| {max(d.logit for d in deviations):.4f}, "
        f"max top-gap shift {max(d.gap for d in deviations):.4f}, recurrent / conv state / K/V "
        f"{max(d.ssm for d in deviations):.3g} / {max(d.conv for d in deviations):.3g} / "
        f"{max(d.kv for d in deviations):.3g} of max |decode|; {len(flips)} top-1 changes"
        + (f" at decode margins {[round(d.margin, 4) for d in flips]}" if flips else "")
    )


def _failures(what: str, rows, positions, deviations: list[Deviation]) -> list[str]:
    out = []
    for k, d in enumerate(deviations):
        pos, j = positions[k // TOKENS], k % TOKENS
        out += [f"{what}, {rows[pos].name} at row {pos}, token {j}: {f}" for f in d.outside()]
    return out


def _moved(h: Harness, rows, positions, logits) -> tuple[int, float]:
    """``(probes whose verify logits differ bitwise from their verify alone, max |diff|)``. Run
    after the batch's own comparisons: the verify alone overwrites what the batch left."""
    mine_all = [logits[pos * TOKENS : (pos + 1) * TOKENS].clone() for pos in positions]
    moved, largest = 0, 0.0
    for pos, mine in zip(positions, mine_all):
        alone = h.alone_logits(rows[pos])
        if not _equal(mine, alone):
            moved += 1
            largest = max(largest, float((mine - alone).abs().max()))
    return moved, largest


@torch.inference_mode()
def test_reference_rows_are_real(harness):
    """The reference is worth comparing against: every probe's logits are finite, the K/V read
    at its tokens' positions through the manager's page table are non-zero in every attention
    layer, and the slot's state after its first, second and third token differs pairwise in
    every GDP layer, recurrent and conv state alike -- so a state taken from the wrong position
    fails the comparison."""
    h = harness
    for r in h.probes:
        steps = h.reference[r.rid]
        for j, t in enumerate(steps):
            assert bool(torch.isfinite(t.logits).all()), (r.name, j)
            assert all(bool(kv.any()) for kv in t.kv), (r.name, j)
            assert all(bool(s.any()) for s in t.ssm) and all(bool(c.any()) for c in t.conv)
        for a in range(TOKENS):
            for b in range(a + 1, TOKENS):
                for layer, sa, sb, ca, cb in zip(
                    h.gdp_layers,
                    steps[a].ssm,
                    steps[b].ssm,
                    steps[a].conv,
                    steps[b].conv,
                ):
                    assert not _equal(sa, sb), (r.name, layer, a, b)
                    assert not _equal(ca, cb), (r.name, layer, a, b)
        h.log(f"{r.name}: verify tokens {r.tokens}")


@torch.inference_mode()
def test_decode_attends_through_the_invariant_entry_and_a_verify_through_thop(harness, monkeypatch):
    """The trunk's eight attention layers run ``invariant_gen_attention`` in a decode step (one
    token a row) and ``thop_attention`` in a verify step (three), and nothing else."""
    M = _modeling()
    h = harness
    calls = []
    for name in ("invariant_gen_attention", "thop_attention"):
        entry = getattr(M, name)

        def counted(*args, _entry=entry, _name=name, **kwargs):
            calls.append(_name)
            return _entry(*args, **kwargs)

        monkeypatch.setattr(M, name, counted)
    r = h.probes[0]
    h.restore()
    h.decode(r, 0)
    decode, calls[:] = list(calls), []
    h.eager([r])
    verify = list(calls)
    h.restore()
    layers = len(h.attention_layers)
    assert decode == ["invariant_gen_attention"] * layers, decode
    assert verify == ["thop_attention"] * layers, verify


@torch.inference_mode()
def test_verify_alone_is_within_the_band_of_decode(harness):
    """Every probe's verify step alone (B = 1), eagerly, with every entry's guard on: token j's
    state in the GDP layers before the first attention layer and its K/V in that layer are the
    bits of its decode step, everything later is within the bands, and the verify writes no
    slot."""
    M = _modeling()
    h = harness
    failures, deviations = [], []
    with validating(
        M.thop_attention,
        M.invariant_mm,
        M.cublas_mm,
        M.gdp_verify,
        M.causal_conv1d_verify,
    ):
        for r in h.probes:
            logits = h.eager([r])
            assert h.changed_slots() == set(), f"{r.name}: the verify wrote a state slot"
            h.alone[r.rid] = logits[:TOKENS].clone()
            found = h.compare([r], [0], logits)
            deviations += found
            failures += _failures("alone", [r], [0], found)
    h.log(f"alone: {_summary(deviations)}")
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_verify_rows_in_eager_batches(harness):
    """Every probe inside eager verify batches of every size: within the bands of its decode
    steps, bitwise before attention. Reports how many of its rows move with the batch -- differ
    from the same probe's verify alone -- and by how much."""
    h = harness
    failures, deviations, moved, compared, largest = [], [], 0, 0, 0.0
    for size in BATCH_SIZES:
        if size == 1:
            continue  # test_verify_alone_is_within_the_band_of_decode
        size_moved = 0
        for rows, positions in _batches(h, size):
            logits = h.eager(rows)
            found = h.compare(rows, positions, logits)
            deviations += found
            failures += _failures(f"eager B={size}", rows, positions, found)
            n, worst = _moved(h, rows, positions, logits)
            size_moved, compared, largest = (
                size_moved + n,
                compared + len(positions),
                max(largest, worst),
            )
        moved += size_moved
        h.log(f"eager B={size}: {size_moved} probe rows move from their verify alone")
    h.log(f"eager: {_summary(deviations)}")
    h.log(
        f"eager: {moved} of {compared} probe rows differ from their verify alone, max |logit diff| "
        f"{largest:.4f}"
    )
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_verify_rows_in_padded_graphs(harness):
    """Every probe inside padded CUDA-graph verify batches of every size, B = 1 included: within
    the bands of its decode steps, bitwise before attention, and the verify writes no slot, the
    padding rows' included. Reports how many rows differ from the same probe's eager verify
    alone."""
    h = harness
    failures, deviations, moved, compared, largest = [], [], 0, 0, 0.0
    for size in BATCH_SIZES:
        for rows, positions in _batches(h, size):
            logits = h.graph(rows)
            found = h.compare(rows, positions, logits)
            deviations += found
            failures += _failures(f"graph B={size} in {h.graph_size(size)}", rows, positions, found)
            written = h.changed_slots()
            assert written == set(), (
                f"graph B={size}: the verify wrote slots {sorted(written)} "
                f"(the padding dummy's slot is {h.dummy_slot})"
            )
            n, worst = _moved(h, rows, positions, logits)
            moved, compared, largest = moved + n, compared + len(positions), max(largest, worst)
    h.log(f"graph: sizes {sorted(h.graphs)}, {_summary(deviations)}")
    h.log(
        f"graph: {moved} of {compared} probe rows differ from their eager verify alone, max "
        f"|logit diff| {largest:.4f}"
    )
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_commit_promotes_the_state_the_verify_stored(harness):
    """After a padded graph verify of every probe (15 rows in the graph of 16), the worker's
    commit (``update_mamba_states`` with the engine's own slots, the padding dummy's included)
    leaves in each probe's slot, for one, two and three accepted, exactly the state the verify
    stored for its last accepted token, in every GDP layer: bitwise its decode step's before the
    first attention layer, within the state bands after it. It writes no other request's slot."""
    h = harness
    rows = h.probes + h.fillers[:3]
    accepted = [1 + k % TOKENS for k in range(len(rows))]
    size = h.graph_size(len(rows))
    h.graph(rows)
    gmd = h.graphs[size][0]
    stored = {
        r.rid: (
            [h.manager.get_intermediate_ssm_states(i)[row, n - 1].clone() for i in h.gdp_layers],
            [h.manager.get_intermediate_conv_states(i)[row, n - 1].clone() for i in h.gdp_layers],
        )
        for row, (r, n) in enumerate(zip(h.probes, accepted))
    }
    h.commit(gmd, accepted + [1] * (size - len(rows)))
    written = h.changed_slots()
    assert written <= {h.slot[r.rid] for r in rows} | {h.dummy_slot}, sorted(written)
    failures, ssm_dev, conv_dev = [], 0.0, 0.0
    for r, n in zip(h.probes, accepted):
        ref = h.reference[r.rid][n - 1]
        slot = h.slot[r.rid]
        for k, layer in enumerate(h.gdp_layers):
            ssm = h.manager.get_ssm_states(layer)[slot]
            conv = h.manager.get_conv_states(layer)[slot]
            if not (_equal(ssm, stored[r.rid][0][k]) and _equal(conv, stored[r.rid][1][k])):
                failures.append(
                    f"{r.name}, {n} accepted: the slot is not the stored state, {layer}"
                )
            if layer < h.upstream:
                if not (_equal(ssm, ref.ssm[k]) and _equal(conv, ref.conv[k])):
                    failures.append(f"{r.name}, {n} accepted: differs from decode bitwise, {layer}")
            else:
                ssm_dev = max(ssm_dev, _relative(ssm, ref.ssm[k]))
                conv_dev = max(conv_dev, _relative(conv, ref.conv[k]))
    h.restore()
    h.log(f"commit: recurrent / conv state {ssm_dev:.3g} / {conv_dev:.3g} of max |decode|")
    assert ssm_dev <= SSM_BAND and conv_dev <= CONV_BAND, (ssm_dev, conv_dev)
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_verify_at_the_length_limit(harness, monkeypatch):
    """A request whose last accepted token is its ``MAX_SEQ_LEN - 1``-th, the longest history a
    verify starts from, verifies two drafts: its lengths reach ``MAX_SEQ_LEN + 1``, past the
    limit itself. The cache manager's ``max_seq_len`` -- which thop_attention is handed as its
    bound and as its attention window -- counts the drafts, so the window holds every key; a
    window of the bare ``MAX_SEQ_LEN`` would drop the oldest. The verify, alone and in a padded
    graph, keeps each token within the bands of its decode step."""
    M = _modeling()
    h = harness
    r = h.limit
    calls = []
    original = M.thop_attention

    def recorded(**kwargs):
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(M, "thop_attention", recorded)
    with validating(original):
        logits = h.eager([r])
    monkeypatch.undo()
    longest = int(calls[0]["sequence_length"][0])
    bound, window = calls[0]["max_seq_len"], calls[0]["attention_window_size"]
    h.log(
        f"at the limit: lengths up to {longest}, max_seq_len {bound}, attention window {window} "
        f"(net {MAX_SEQ_LEN})"
    )
    assert longest == MAX_SEQ_LEN + DRAFT - 1 and min(bound, window) >= longest > MAX_SEQ_LEN
    assert all(bool(torch.isfinite(t.logits).all()) for t in h.reference[r.rid])
    failures = _failures("alone at the limit", [r], [0], h.compare([r], [0], logits))
    rows = [r, h.probes[0], h.probes[1]]
    found = h.compare(rows, [0, 1, 2], h.graph(rows))
    failures += _failures("graph B=3 at the limit", rows, [0, 1, 2], found)
    h.restore()
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_the_exact_comparison_moves_with_cublas(harness, monkeypatch):
    """The bitwise half has teeth on this model's own state: with the GEMMs that run through
    ``invariant_mm`` put back on ``cublas_mm`` -- what the verify ran before it was made
    invariant -- an eager verify batch of 64 rows (192 GEMM rows, where cuBLAS splits the dense
    MLPs of layers 1 and 3 otherwise than at one row) moves some probe's state before the first
    attention layer, or its K/V in that layer, off that rival's own decode steps alone."""
    M = _modeling()
    h = harness

    def replaced(x, weight, out_dtype=None):
        return M.cublas_mm(x, weight.t(), None, out_dtype)

    monkeypatch.setattr(M, "invariant_mm", replaced)
    ((rows, positions),) = _batches(h, 64)
    reference = {}
    for pos in positions:
        r = rows[pos]
        h.restore()
        reference[r.rid] = [h.slot_token(r, j, h.decode(r, j)) for j in range(TOKENS)]
    h.restore()
    logits = h.eager(rows)
    differ = []
    for pos in positions:
        r = rows[pos]
        for j in range(TOKENS):
            got = h.verify_token(r, pos, j, logits)
            exact = _deviation(got, reference[r.rid][j], h.upstream, h.gdp_layers).exact
            differ += [f"{r.name} token {j}: {e}" for e in exact]
    h.restore()
    h.log(f"rival cublas_mm: {len(differ)} bitwise differences before attention")
    for line in differ[:24]:
        h.log(f"  {line}")
    assert differ, (
        "with cublas_mm back in the verify nothing before attention moved, so the bitwise half "
        "cannot tell an invariant verify from a variant one here"
    )


@torch.inference_mode()
def test_the_band_tells_a_state_one_token_off(harness):
    """The bands have teeth: a verify that starts from the state one token too far -- every GDP
    slot holding the state after t0 already, as a rollback that kept a rejected draft would
    leave it -- moves every probe's logits by more than twice the logit band, and one of its top
    gaps by more than the gap band. (Measured: at least 4.529 and 0.767, the least on the long
    prompts, whose top gaps one token of GDP state moves least.)"""
    h = harness
    smallest, smallest_gap = float("inf"), float("inf")
    for r in h.probes:
        h.restore()
        after_t0 = h.reference[r.rid][0]
        slot = h.slot[r.rid]
        for k, layer in enumerate(h.gdp_layers):
            h.manager.get_ssm_states(layer)[slot] = after_t0.ssm[k]
            h.manager.get_conv_states(layer)[slot] = after_t0.conv[k]
        h.fill(h.md, [r], num_contexts=0, tokens=TOKENS)
        logits = h.trunk(h.md, h._ids([r], 0))
        moved = [_logit_deviation(logits[j], h.reference[r.rid][j].logits) for j in range(TOKENS)]
        worst, worst_gap = max(m[0] for m in moved), max(m[1] for m in moved)
        smallest, smallest_gap = min(smallest, worst), min(smallest_gap, worst_gap)
        h.log(
            f"{r.name}: one token off moves the logits by up to {worst:.3f}, a top gap by up to "
            f"{worst_gap:.3f}"
        )
    h.restore()
    h.log(
        f"one token off: at least {smallest:.3f} / {smallest_gap:.3f} on every probe, against "
        f"the bands {LOGIT_BAND} / {GAP_BAND}"
    )
    assert smallest > 2 * LOGIT_BAND and smallest_gap > GAP_BAND, (smallest, smallest_gap)


@torch.inference_mode()
def test_the_drift_over_consecutive_verifies(harness):
    """MTP on along MTP off's own greedy tokens, as far as the MTP identity gate generates: each
    probe's 255 or 1023 tokens after t0 decoded one a step (MTP off), then verified three a step
    from the same state, each verify alone in the padded graph of one (as the identity gate runs a
    prompt) with every draft accepted and its state committed (MTP on). At every matched
    position -- the same tokens before it -- the verify's logits stay within the drift bands of
    the decode step's, and every top-1 change sits at a decode margin within the gap band: a
    parting the identity gate would see there is a near-tie by its rule."""
    h = harness
    probes = h.probes
    longest = max(r.drift_steps for r in probes) * TOKENS
    # MTP off: every probe decoded together, one token a step (decode rows do not depend on the
    # batch), each until its own horizon. Position n feeds token n; its logits pick token n + 1.
    h.restore()
    tokens = {r.rid: [r.tokens[0]] for r in probes}
    decode = {r.rid: [] for r in probes}
    for n in range(longest):
        rows = [r for r in probes if n < r.drift_steps * TOKENS]
        h.fill(h.md, rows, 0, tokens=1, cached=[len(r.prompt) + n for r in rows])
        ids = torch.tensor([tokens[r.rid][n] for r in rows], dtype=torch.int, device="cuda")
        logits = h.trunk(h.md, ids)
        for r, row in zip(rows, logits):
            decode[r.rid].append(row.clone())
            tokens[r.rid].append(int(row.argmax()))
    for r in probes:
        assert tuple(tokens[r.rid][:TOKENS]) == r.tokens, r.name
        for j in range(TOKENS):
            assert _equal(decode[r.rid][j], h.reference[r.rid][j].logits), (r.name, j)
    # MTP on: each probe alone, from the same state; position 3s + j is row j of verify s.
    h.restore()
    worst_logit, worst_gap, flips, lines = 0.0, 0.0, [], []
    for r in probes:
        r_logit, r_gap = 0.0, 0.0
        for s in range(r.drift_steps):
            n = s * TOKENS
            ids = torch.tensor(tokens[r.rid][n : n + TOKENS], dtype=torch.int, device="cuda")
            logits = h.graph([r], ids=ids, cached=[len(r.prompt) + n], restore=False)
            h.commit(h.graphs[1][0], [TOKENS])
            for j in range(TOKENS):
                ref = decode[r.rid][n + j]
                logit, gap = _logit_deviation(logits[j], ref)
                r_logit, r_gap = max(r_logit, logit), max(r_gap, gap)
                if int(logits[j].argmax()) != int(ref.argmax()):
                    flips.append((r.name, n + j, _margin(ref)))
        worst_logit, worst_gap = max(worst_logit, r_logit), max(worst_gap, r_gap)
        lines.append(f"{r.name}: {r.drift_steps * TOKENS} tokens, {r_logit:.4f} / {r_gap:.4f}")
    h.restore()
    for line in lines:
        h.log(f"drift {line}")
    h.log(
        f"drift: max |logit diff| {worst_logit:.4f}, max top-gap shift {worst_gap:.4f} over "
        f"{sum(r.drift_steps for r in probes) * TOKENS} positions; top-1 changes (probe, "
        f"position, decode margin): {flips}"
    )
    assert worst_logit <= DRIFT_LOGIT_BAND and worst_gap <= DRIFT_GAP_BAND, (worst_logit, worst_gap)
    wide = [f for f in flips if f[2] > DRIFT_GAP_BAND]
    assert not wide, f"top-1 changes off a near-tie: {wide}"


@torch.inference_mode()
def test_the_mtp_layer_drafts_as_prefill_drafts(harness, monkeypatch):
    """The MTP layer's draft step runs the same kernels in the decode target as in the prefill
    target -- cuBLAS on `_mtp_view`'s `.t()` views, thop_attention -- so its drafts are the ones
    the target drafted before decode was made batch-invariant: on a draft step of every probe
    (t1 after the trunk's hidden state of t0, over the MTP layer's KV of the prompt), the decode
    target's output, through ``MTPLayer``'s routing, equals the prefill target's bit for bit.
    The decode target's row-invariant ``_moe`` reads the stored [out, in] weights, so handing it
    the MTP layer's views misreads them; its guard says so."""
    M = _modeling()
    h = harness
    rows = list(h.probes)
    ids = torch.tensor([r.tokens[1] for r in rows], dtype=torch.int, device="cuda")
    hidden = torch.cat([h.hidden0[r.rid] for r in rows])
    mtp_layer = h.model.draft_model.mtp_layers[0]

    def step(target):
        h.fill(h.md, rows, num_contexts=0, tokens=1)
        return h.mtp(h.md, target, ids, hidden)

    with validating(M.cublas_mm):
        routed = step(lambda e, i, x, md: mtp_layer(e, None, i, x, md))
        decode = step(h.core._decode.mtp_forward)
        prefill = step(h.core._prefill.mtp_forward)
    assert bool(torch.isfinite(prefill).all())
    assert _equal(decode, prefill), (
        "the decode target's MTP layer differs from the prefill target's"
    )
    assert _equal(routed, decode), "MTPLayer did not route a decode-only draft step to DecodeTarget"
    drafts = mtp_layer.shared_head(prefill, h.model.lm_head, h.md, False).argmax(-1).tolist()
    agree = sum(d == r.tokens[2] for d, r in zip(drafts, rows))
    h.log(f"draft step: {agree} of {len(rows)} drafts equal the trunk's greedy t2")

    monkeypatch.setattr(M.DecodeTarget, "_mtp_moe", M.DecodeTarget._moe)
    with (
        pytest.raises(AssertionError, match="x and weight must share K"),
        validating(M.invariant_mm),
    ):
        step(h.core._decode.mtp_forward)
