# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4-rank job: the nemotron4 tp4 MTP verify step against the decode steps of the same tokens, on
every rank, on real weights and real cache state.

Under one-model MTP-Eagle (``max_draft_len`` 2) a generation row carries its last accepted token
and two drafts, and the trunk verifies the three in one step: 3B rows through every GEMM, three
tokens per request through attention, the GDP mixers through the verify entries. On tp4 each
token's row-parallel partials also meet in an all-reduce, and every rank holds its own 6 of the
24 GDP heads of every state. Every entry of that step but one gives token j the bits it gets as an
MTP-off decode step of its own: the GEMMs (``invariant_mm`` where cuBLAS is row-variant at a
rank's shapes), the all-reduce, the wide residual, the norms, the MoE and the GDP and conv verify
entries. Attention is the exception: a decode step attends through ``invariant_gen_attention``, a
verify through ``thop_attention``'s generation kernels, which sum the softmax in another order and
pick their kernel by batch size and history length. So a verify token is compared with its decode
step as the tp1 test compares it (``test_modeling_v2_verify_invariance.py``): bitwise, the state it
leaves in the GDP layers before the first attention layer and the K/V it appends in that layer;
within a band derived from measurement, its logits and the state and K/V it leaves in every later
layer; reported, whether its rows move with the batch. A last check follows MTP on along MTP off's
own greedy tokens for as many tokens as the MTP identity gate generates and measures the drift of
the logits at matched positions, which tp4's MTP identity and acceptance gates rest on.

This script is its own launcher, as the tp4 decode-invariance job is: run it plainly and it starts
one MPI rank per device named in CUDA_VISIBLE_DEVICES under ``mpirun``, with a deadline it enforces
by killing the job.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _tp4_verify_invariance_ranks.py --report <json>

Each rank builds what the engine builds on it under ``MTPDecodingConfig(max_draft_len=2)``: the
spec config resolved against the checkpoint (one-model MTP-Eagle over the one repeated MTP layer);
``ModelConfig`` off the checkpoint with the rank's tp4 mapping (tp 4, MoE tp 4, no attention DP)
and the spec config, the target class with the rank's slices of every weight loaded, the MTP
layer's included, and ``post_load_weights`` (the TP group's all-reduce workspace); the trunk's
hybrid cache manager (``_create_kv_cache_manager``, the default ``MambaHybridCacheManagerV2``,
with the per-position buffers) and the MTP layer's separate draft KV manager; the engine's TRTLLM
attention metadata, filled per step as the runner fills it; and, for graphs,
``create_cuda_graph_metadata`` at draft length 2 and the runner's MTP padding dummy
(``CUDA_GRAPH_DUMMY_REQUEST_ID - 2``), whose rows pad every graph batch. Every rank runs the same
steps in the same order, as the engine's ranks do.

The requests: the ten prompts of the fixed set, two prefixes of the long ones and 1012 more
prefixes as fillers, prefilled through the target's own prefill steps, the MTP layer's first draft
step over each prompt included, as the worker runs it. A probe's reference is three MTP-off decode
steps run alone, eagerly, from the state the prefill left: over t0, its greedy next token, then t1
and t2, the greedy tokens after it -- the drafts a perfect drafter would propose. Then the verify
step over [t0, t1, t2] from the same state: alone, inside eager batches of up to 1024 requests
(3072 rows), and inside padded CUDA graphs. Every run starts from the same state: each GDP slot is
restored from a snapshot, and a row's appends land where its own attention reads only after
writing them. One more request sits at the length limit the engine admits, its history zeroed
rather than prefilled: its verify reaches past ``max_seq_len`` by the drafts.

Each rank checks its own rows. Rank 0 gathers every rank's findings, checks that every rank formed
the same logits and appended the same K/V, and writes the report the collected test reads
(``test_modeling_v2_tp4_verify_invariance.py``). The CUDA graphs are captured as the engine
captures them: one eager forward at every size first, which grows the attention workspace all of
them share to what the largest call needs, then the captures, which resize nothing. Not a pytest
module: the checks are one fixed sequence inside a single 4-rank job, each reading rank state only
``_run_one_rank`` binds.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "MODELING_V2_TP4_VERIFY_INVARIANCE_PID_DIR"
_REPORT_ENV = "MODELING_V2_TP4_VERIFY_INVARIANCE_REPORT"
#: The weight load takes ~2 min, the prefill ~1, the checks ~10.
DEADLINE_S = 3600
WORLD_SIZE = 4

_MODELING = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4."
    "nemotron4_nano_preview__sm_103__tp4.modeling"
)

# The engine sizes this run uses everywhere (task.yaml, the boot gate, check_goldens).
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
TOKENS_PER_BLOCK = 32
#: The checkpoint's trained draft depth; a verify row carries 1 + DRAFT tokens.
DRAFT = 2
TOKENS = 1 + DRAFT
#: Requests resident at once: the probes and fillers; the limit request and the padding dummy
#: beside them.
NUM_REQUESTS = 1024
#: Rows a step may hold.
MAX_BATCH = 1032
#: KV tokens of the paged pools: every request stays resident, the probes with room for the drift.
KV_TOKENS = 480 * 1024
#: Verify batch sizes: 1 to 1024 requests (3072 rows), and sizes that leave a graph mostly or
#: slightly padded.
BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8, 16, 64, 100, 250, 256, 500, 512, 1000, 1024)
#: The CUDA-graph batch sizes captured; a batch runs in the smallest one that holds it, as the
#: runner pads.
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
#: The eager verify batch size the cuBLAS rival runs at: on a tp4 rank cuBLAS changes kernels on
#: the GDP in_proj slice from 17 rows, which layer 0 runs before the first attention layer.
RIVAL_BATCH_SIZE = 64
#: Tokens the MTP identity gate generates per prompt: the first comes from the prefill, the rest
#: from verify steps. The drift follows each probe for the verifies of that many tokens.
SHORT_TOKENS = 256
LONG_TOKENS = 1024
#: The decode reference's top tokens whose gap to its top-1 a verify must keep.
TOP = 8

# The bands, derived from measurement as tp1's verify test derives its own (its comment says
# why a verify token differs from its decode step at all). A tp4 rank attends with 4 of the 16
# query heads, so thop_attention runs its 4-heads-per-KV-head kernels here and the bands are tp4's
# own. Measured on every rank: |logit diff| 0.5728 alone and 0.7400 in eager batches of 2 to 1024
# and padded graphs of 1 to 1024 (every rank alike: the logits are formed whole on every rank);
# top-gap shift 0.2001 alone, 0.2876 in batches; the later layers' recurrent state, conv state and
# K/V, over the ranks' own heads, 0.0545 / 0.069 / 0.0637 of max |decode|. Each band is its worst
# case plus a quarter, rounded up.
#: max |verify logit - decode logit| over the vocabulary, fp32.
LOGIT_BAND = 0.93
#: The largest change, verify against decode, of the gap between the decode step's top-1 and any
#: of its top `TOP` tokens.
GAP_BAND = 0.36
#: max |verify - decode| / max |decode| per tensor, over the GDP layers after the first attention
#: layer (the rank's heads of the recurrent state, fp32, and of the conv state, bf16) and the
#: attention layers after it (K/V).
SSM_BAND = 0.069
CONV_BAND = 0.087
KV_BAND = 0.08
# The drift: the same two quantities over the 6132 matched positions of the drift check. Worst
# cases 3.4226 (|logit diff|, on the 2-shot chat `clips`) and 1.2381 (top-gap shift, on
# `gsm8k_fewshot`), plus a quarter, rounded up. tp4's MTP identity gate's near-tie band is
# DRIFT_GAP_BAND; its MTP acceptance gate's verify term is the measured |logit diff|.
DRIFT_LOGIT_BAND = 4.28
DRIFT_GAP_BAND = 1.55

# Bound inside the rank body, never in the launcher: tensorrt_llm calls MPI_Init at import.
COMM = None
RANK = 0


def _modeling():
    return importlib.import_module(_MODELING)


def _mapping(rank: int):
    from tensorrt_llm.mapping import Mapping

    return Mapping(
        world_size=WORLD_SIZE,
        rank=rank,
        gpus_per_node=WORLD_SIZE,
        tp_size=WORLD_SIZE,
        moe_tp_size=WORLD_SIZE,
        moe_ep_size=1,
    )


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


def _build_target(path: str, mapping, spec):
    """``(ModelConfig, the loaded target)``: the target class off the checkpoint's config with
    this rank's mapping and the spec config, the rank's weights loaded through its own
    ``load_weights``."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype

    config = ModelConfig.from_pretrained(
        path,
        trust_remote_code=True,
        mapping=mapping,
        max_num_tokens=MAX_NUM_TOKENS,
        spec_config=spec,
    )
    validate_and_set_mamba_ssm_cache_dtype(config, "auto")
    with torch.device("cuda"):
        model = _modeling().ModelingV2Nemotron4NanoPreviewSm103Tp4(config)
    model.load_weights(_checkpoint_tensors(path))
    # Collective: builds the TP group's all-reduce workspace.
    model.post_load_weights()
    torch.cuda.synchronize()
    return config, model.eval()


def _build_managers(config, mapping, spec):
    """The engine's two cache managers under one-model MTP on tp4: the trunk's hybrid manager
    (paged KV for its attention layers, recurrent and conv state slots of the rank's heads and
    the per-position buffers for its GDP layers) and the separate draft KV manager holding the
    MTP layer's KV."""
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
            mapping,
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


def _build_metadata(config, manager, draft_manager, mapping):
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
        mapping=mapping,
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
    fillers enough for a batch of 1024."""
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
    """What one token's step produced on this rank: its logits and what it wrote."""

    logits: torch.Tensor  # [vocab] fp32
    ssm: list[torch.Tensor]  # per GDP layer, the state after this token: [6, 64, 128] fp32
    conv: list[torch.Tensor]  # per GDP layer, the state after this token: [4224, 3] bf16
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
    """How a verify token compares with its decode step on this rank."""

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
    """The rank's loaded target, its cache managers and metadata, the prefilled requests, and the
    state every run starts from."""

    def __init__(self, checkpoint: str, rank: int, log):
        import _nemotron4_prompts as P

        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
        from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens

        self.log = log
        self.mapping = _mapping(rank)
        tokenizer = P.load_tokenizer(checkpoint)
        self.spec = _spec_config(checkpoint)
        self.config, self.model = _build_target(checkpoint, self.mapping, self.spec)
        self.manager, self.draft_manager = _build_managers(self.config, self.mapping, self.spec)
        self.md = _build_metadata(self.config, self.manager, self.draft_manager, self.mapping)
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
        self.pages: dict[tuple[int, int], list[int]] = {}
        for i in self.attention_layers:
            pages = torch.tensor(self._pages(self.limit, i), dtype=torch.long, device="cuda")
            self.kv_buffers[i][pages] = 0
        # The request at the limit: zero history, a fresh slot, and t0 a token of the text.
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
        # The same seed on every rank: the ranks feed the padding rows the same tokens, as the
        # engine's ranks feed every row the same token.
        self.random_tokens = torch.Generator(device="cuda").manual_seed(9)
        self.alone: dict[int, torch.Tensor] = {}
        self.workspace_address = None

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
        shell runs (``cublas_mm`` with an fp32 output, the whole vocabulary on every rank) on
        every row, as it does for the tokens a decode step or a verify step gathers."""
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
        M = _modeling()
        mtp_layer = self.model.draft_model.mtp_layers[0]
        batch: list[Request] = []
        steps = 0
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
                steps += 1
            if r is not None:
                batch.append(r)
        torch.cuda.synchronize()
        self.log(
            f"prefilled {len(requests)} requests in {steps} steps, the MTP layer's KV included"
        )

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
        under the graph, which keeps its old address. Every rank runs the same steps."""
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

    def _pages(self, r: Request, layer: int) -> list[int]:
        if (r.rid, layer) not in self.pages:
            self.pages[r.rid, layer] = self.manager.get_batch_cache_indices(
                [r.rid], layer_idx=layer
            )[0]
        return self.pages[r.rid, layer]

    def kv_at(self, r: Request, layer: int, j: int) -> torch.Tensor:
        """``[K/V, head_dim]``: what attention layer ``layer`` holds at ``r``'s token j, read
        through the manager's own page table and buffer view."""
        pos = len(r.prompt) + j
        page = self._pages(r, layer)[pos // TOKENS_PER_BLOCK]
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
        M = _modeling()
        self.restore()
        tokens, hidden0 = [], None
        for j in range(TOKENS):
            if j == 0:
                self.fill(self.md, [r], num_contexts=0, tokens=1)
                ids = torch.tensor([r.tokens[0]], dtype=torch.int, device="cuda")
                hidden0 = self.core.forward(attn_metadata=self.md, input_ids=ids).clone()
                logits = M.cublas_mm(hidden0, self.lm_head.t(), None, torch.float32)[0]
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


def _summary(deviations: list[Deviation]) -> str:
    flips = [d for d in deviations if d.argmax[0] != d.argmax[1]]
    return (
        f"{len(deviations)} tokens: max |logit diff| {max(d.logit for d in deviations):.4f}, "
        f"max top-gap shift {max(d.gap for d in deviations):.4f}, recurrent / conv state / K/V "
        f"{max(d.ssm for d in deviations):.3g} / {max(d.conv for d in deviations):.3g} / "
        f"{max(d.kv for d in deviations):.3g} of max |decode|; {len(flips)} top-1 changes"
        + (f" at decode margins {[round(d.margin, 4) for d in flips]}" if flips else "")
    )


def _metrics(deviations: list[Deviation]) -> dict:
    """The worst case of each compared quantity, for the report."""
    return {
        "tokens": len(deviations),
        "logit": max(d.logit for d in deviations),
        "gap": max(d.gap for d in deviations),
        "ssm": max(d.ssm for d in deviations),
        "conv": max(d.conv for d in deviations),
        "kv": max(d.kv for d in deviations),
        "flip_margins": [d.margin for d in deviations if d.argmax[0] != d.argmax[1]],
    }


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


# --- The checks, in order. Each returns {"failures": [...], "log": [...], ...} for this rank. ----


def check_reference_rows_are_real(h: Harness) -> dict:
    """The reference is worth comparing against: every probe's logits are finite, the K/V read at
    its tokens' positions through the manager's page table are non-zero in every attention layer,
    and the slot's state after its first, second and third token differs pairwise in every GDP
    layer, recurrent and conv state alike -- so a state taken from the wrong position fails the
    comparison."""
    failures, log = [], []
    for r in h.probes:
        steps = h.reference[r.rid]
        for j, t in enumerate(steps):
            if not bool(torch.isfinite(t.logits).all()):
                failures.append(f"{r.name} token {j}: non-finite logits")
            if not all(bool(kv.any()) for kv in t.kv):
                failures.append(f"{r.name} token {j}: an all-zero appended K/V")
            if not (all(bool(s.any()) for s in t.ssm) and all(bool(c.any()) for c in t.conv)):
                failures.append(f"{r.name} token {j}: an all-zero recurrent or conv state")
        for a in range(TOKENS):
            for b in range(a + 1, TOKENS):
                for layer, sa, sb, ca, cb in zip(
                    h.gdp_layers, steps[a].ssm, steps[b].ssm, steps[a].conv, steps[b].conv
                ):
                    if _equal(sa, sb) or _equal(ca, cb):
                        failures.append(f"{r.name}: layer {layer} states after {a} and {b} equal")
        log.append(f"{r.name}: verify tokens {r.tokens}")
    return {"failures": failures, "log": log}


def check_every_rank_forms_the_same_logits_and_kv(h: Harness) -> dict:
    """Every rank forms each probe's decode logits and its verify logits from the same hidden
    state with the whole lm_head, and appends the K/V of the one KV head every rank keeps: the
    four ranks' bits must agree."""
    mine = {}
    for r in h.probes:
        logits = h.eager([r])
        mine[r.rid] = (
            b"".join(_bits(t.logits).cpu().numpy().tobytes() for t in h.reference[r.rid]),
            _bits(logits).cpu().numpy().tobytes(),
            b"".join(_bits(t.kv).cpu().numpy().tobytes() for t in h.reference[r.rid]),
        )
    h.restore()
    gathered = COMM.gather(mine, root=0)
    failures = []
    if RANK == 0:
        for r in h.probes:
            for other, theirs in enumerate(gathered):
                for what, a, b in zip(
                    ("decode logits", "verify logits", "appended K/V"), theirs[r.rid], mine[r.rid]
                ):
                    if a != b:
                        failures.append(f"{r.name}: rank {other}'s {what} differ from rank 0's")
    return {
        "failures": failures,
        "log": [f"{len(h.probes)} probes compared over {WORLD_SIZE} ranks"] if RANK == 0 else [],
    }


def check_decode_attends_through_the_invariant_entry_and_a_verify_through_thop(h: Harness) -> dict:
    """The trunk's eight attention layers run ``invariant_gen_attention`` in a decode step (one
    token a row) and ``thop_attention`` in a verify step (three), and nothing else."""
    M = _modeling()
    calls = []
    saved = {name: getattr(M, name) for name in ("invariant_gen_attention", "thop_attention")}

    def counted(name):
        def call(*args, **kwargs):
            calls.append(name)
            return saved[name](*args, **kwargs)

        return call

    try:
        for name in saved:
            setattr(M, name, counted(name))
        r = h.probes[0]
        h.restore()
        h.decode(r, 0)
        decode, calls[:] = list(calls), []
        h.eager([r])
        verify = list(calls)
    finally:
        for name, entry in saved.items():
            setattr(M, name, entry)
        h.restore()
    layers = len(h.attention_layers)
    failures = []
    if decode != ["invariant_gen_attention"] * layers:
        failures.append(f"a decode step called {decode}")
    if verify != ["thop_attention"] * layers:
        failures.append(f"a verify step called {verify}")
    return {
        "failures": failures,
        "log": [f"decode {decode[:1]} x{len(decode)}, verify {verify[:1]} x{len(verify)}"],
    }


def check_verify_alone_is_within_the_band_of_decode(h: Harness) -> dict:
    """Every probe's verify step alone (B = 1), eagerly, with every entry's guard on: token j's
    state in the GDP layers before the first attention layer and its K/V in that layer are the
    bits of its decode step, everything later is within the bands, and the verify writes no
    slot."""
    from _validating import validating

    M = _modeling()
    failures, deviations = [], []
    with validating(
        M.thop_attention,
        M.invariant_mm,
        M.cublas_mm,
        M.gdp_verify,
        M.causal_conv1d_verify,
        M.allreduce,
    ):
        for r in h.probes:
            logits = h.eager([r])
            if h.changed_slots():
                failures.append(f"{r.name}: the verify wrote a state slot")
            h.alone[r.rid] = logits[:TOKENS].clone()
            found = h.compare([r], [0], logits)
            deviations += found
            failures += _failures("alone", [r], [0], found)
    return {
        "failures": failures,
        "log": [f"alone: {_summary(deviations)}"],
        "metrics": _metrics(deviations),
    }


def check_verify_rows_in_eager_batches(h: Harness) -> dict:
    """Every probe inside eager verify batches of every size up to 1024 requests (3072 rows):
    within the bands of its decode steps, bitwise before attention. Reports how many of its rows
    move with the batch -- differ from the same probe's verify alone -- and by how much."""
    failures, deviations, moved, compared, largest, log = [], [], 0, 0, 0.0, []
    for size in BATCH_SIZES:
        if size == 1:
            continue  # check_verify_alone_is_within_the_band_of_decode
        size_moved = 0
        for rows, positions in _batches(h, size):
            logits = h.eager(rows)
            found = h.compare(rows, positions, logits)
            deviations += found
            failures += _failures(f"eager B={size}", rows, positions, found)
            n, worst = _moved(h, rows, positions, logits)
            size_moved += n
            compared += len(positions)
            largest = max(largest, worst)
        moved += size_moved
        log.append(f"eager B={size}: {size_moved} probe rows move from their verify alone")
    log.append(f"eager: {_summary(deviations)}")
    log.append(
        f"eager: {moved} of {compared} probe rows differ from their verify alone, max |logit "
        f"diff| {largest:.4f}"
    )
    metrics = dict(_metrics(deviations), moved=moved, compared=compared, moved_logit=largest)
    return {"failures": failures, "log": log, "metrics": metrics}


def check_verify_rows_in_padded_graphs(h: Harness) -> dict:
    """Every probe inside padded CUDA-graph verify batches of every size, B = 1 included: within
    the bands of its decode steps, bitwise before attention, and the verify writes no slot, the
    padding rows' included. Reports how many rows differ from the same probe's eager verify
    alone."""
    failures, deviations, moved, compared, largest = [], [], 0, 0, 0.0
    for size in BATCH_SIZES:
        for rows, positions in _batches(h, size):
            logits = h.graph(rows)
            found = h.compare(rows, positions, logits)
            deviations += found
            failures += _failures(f"graph B={size} in {h.graph_size(size)}", rows, positions, found)
            written = h.changed_slots()
            if written:
                failures.append(
                    f"graph B={size}: the verify wrote slots {sorted(written)} "
                    f"(the padding dummy's slot is {h.dummy_slot})"
                )
            n, worst = _moved(h, rows, positions, logits)
            moved, compared, largest = moved + n, compared + len(positions), max(largest, worst)
    log = [
        f"graph: sizes {sorted(h.graphs)}, {_summary(deviations)}",
        f"graph: {moved} of {compared} probe rows differ from their eager verify alone, max "
        f"|logit diff| {largest:.4f}",
    ]
    metrics = dict(_metrics(deviations), moved=moved, compared=compared, moved_logit=largest)
    return {"failures": failures, "log": log, "metrics": metrics}


def check_commit_promotes_the_state_the_verify_stored(h: Harness) -> dict:
    """After a padded graph verify of every probe and three fillers (15 rows in the graph of 16),
    the worker's commit (``update_mamba_states`` with the engine's own slots, the padding dummy's
    included) leaves in each probe's slot, for one, two and three accepted, exactly the state the
    verify stored for its last accepted token, in every GDP layer of the rank: bitwise its decode
    step's before the first attention layer, within the state bands after it. It writes no other
    request's slot."""
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
    torch.cuda.synchronize()
    failures = []
    written = h.changed_slots()
    allowed = {h.slot[r.rid] for r in rows} | {h.dummy_slot}
    if not written <= allowed:
        failures.append(f"the commit wrote slots {sorted(written - allowed)}")
    ssm_dev, conv_dev = 0.0, 0.0
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
    if ssm_dev > SSM_BAND or conv_dev > CONV_BAND:
        failures.append(f"committed state {ssm_dev:.4g} / {conv_dev:.4g} outside the state bands")
    h.restore()
    return {
        "failures": failures,
        "log": [f"commit: recurrent / conv state {ssm_dev:.3g} / {conv_dev:.3g} of max |decode|"],
        "metrics": {"ssm": ssm_dev, "conv": conv_dev},
    }


def check_verify_at_the_length_limit(h: Harness) -> dict:
    """A request whose last accepted token is its ``MAX_SEQ_LEN - 1``-th, the longest history a
    verify starts from, verifies two drafts: its lengths reach ``MAX_SEQ_LEN + 1``, past the
    limit itself. The cache manager's ``max_seq_len`` -- which thop_attention is handed as its
    bound and as its attention window -- counts the drafts, so the window holds every key; a
    window of the bare ``MAX_SEQ_LEN`` would drop the oldest. The verify, alone and in a padded
    graph, keeps each token within the bands of its decode step."""
    from _validating import validating

    M = _modeling()
    r = h.limit
    calls = []
    original = M.thop_attention

    def recorded(**kwargs):
        calls.append(dict(kwargs))
        return original(**kwargs)

    failures, log = [], []
    M.thop_attention = recorded
    try:
        with validating(original):
            logits = h.eager([r])
    finally:
        M.thop_attention = original
    longest = int(calls[0]["sequence_length"][0])
    bound, window = calls[0]["max_seq_len"], calls[0]["attention_window_size"]
    log.append(
        f"at the limit: lengths up to {longest}, max_seq_len {bound}, attention window {window} "
        f"(net {MAX_SEQ_LEN})"
    )
    if not (longest == MAX_SEQ_LEN + DRAFT - 1 and min(bound, window) >= longest > MAX_SEQ_LEN):
        failures.append(
            f"lengths up to {longest} against a bound of {bound} and a window of {window}"
        )
    if not all(bool(torch.isfinite(t.logits).all()) for t in h.reference[r.rid]):
        failures.append("non-finite reference logits at the limit")
    failures += _failures("alone at the limit", [r], [0], h.compare([r], [0], logits))
    rows = [r, h.probes[0], h.probes[1]]
    found = h.compare(rows, [0, 1, 2], h.graph(rows))
    failures += _failures("graph B=3 at the limit", rows, [0, 1, 2], found)
    h.restore()
    return {"failures": failures, "log": log}


def check_the_exact_comparison_moves_with_cublas(h: Harness) -> dict:
    """The bitwise half has teeth on this model's own state: with the GEMMs that run through
    ``invariant_mm`` put back on ``cublas_mm`` -- what the verify ran before it was made
    invariant -- an eager verify batch of 64 rows (192 GEMM rows; on a tp4 rank cuBLAS changes
    the GDP in_proj slice's kernel from 17 rows, and layer 0 runs it before the first attention
    layer) moves some probe's state before the first attention layer, or its K/V in that layer,
    off that rival's own decode steps alone, on every rank."""
    M = _modeling()
    original = M.invariant_mm

    def replaced(x, weight, out_dtype=None):
        return M.cublas_mm(x, weight.t(), None, out_dtype)

    M.invariant_mm = replaced
    differ, compared = [], 0
    try:
        ((rows, positions),) = _batches(h, RIVAL_BATCH_SIZE)
        reference = {}
        for pos in positions:
            r = rows[pos]
            h.restore()
            reference[r.rid] = [h.slot_token(r, j, h.decode(r, j)) for j in range(TOKENS)]
        logits = h.eager(rows)
        for pos in positions:
            r = rows[pos]
            for j in range(TOKENS):
                compared += 1
                got = h.verify_token(r, pos, j, logits)
                exact = _deviation(got, reference[r.rid][j], h.upstream, h.gdp_layers).exact
                differ += [f"{r.name} token {j}: {e}" for e in exact]
    finally:
        M.invariant_mm = original
        h.restore()
    return {
        "failures": [],
        "differ": len(differ),
        "compared": compared,
        "log": [f"rival cublas_mm: {len(differ)} bitwise differences before attention"]
        + [f"  {line}" for line in differ[:24]],
    }


def check_the_band_tells_a_state_one_token_off(h: Harness) -> dict:
    """The bands have teeth: a verify that starts from the state one token too far -- every GDP
    slot holding the state after t0 already, as a rollback that kept a rejected draft would
    leave it -- moves every probe's logits by more than twice the logit band, and one of its top
    gaps by more than the gap band. (Measured: at least 4.548 and 0.739, the least on the long
    prompts, whose top gaps one token of GDP state moves least.)"""
    smallest, smallest_gap, log = float("inf"), float("inf"), []
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
        log.append(
            f"{r.name}: one token off moves the logits by up to {worst:.3f}, a top gap by up to "
            f"{worst_gap:.3f}"
        )
    h.restore()
    log.append(
        f"one token off: at least {smallest:.3f} / {smallest_gap:.3f} on every probe, against "
        f"the bands {LOGIT_BAND} / {GAP_BAND}"
    )
    failures = []
    if not (smallest > 2 * LOGIT_BAND and smallest_gap > GAP_BAND):
        failures.append(f"a state one token off moves a probe only {smallest} / {smallest_gap}")
    return {
        "failures": failures,
        "log": log,
        "metrics": {"logit": smallest, "gap": smallest_gap},
    }


def check_the_drift_over_consecutive_verifies(h: Harness) -> dict:
    """MTP on along MTP off's own greedy tokens, as far as the MTP identity gate generates: each
    probe's 255 or 1023 tokens after t0 decoded one a step (MTP off), then verified three a step
    from the same state, each verify alone in the padded graph of one (as the identity gate runs a
    prompt) with every draft accepted and its state committed (MTP on). At every matched
    position -- the same tokens before it -- the verify's logits stay within the drift bands of
    the decode step's, and every top-1 change sits at a decode margin within the gap band."""
    probes = h.probes
    longest = max(r.drift_steps for r in probes) * TOKENS
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
    failures = []
    for r in probes:
        if tuple(tokens[r.rid][:TOKENS]) != r.tokens:
            failures.append(f"{r.name}: the decode leg's first tokens are not the reference's")
        for j in range(TOKENS):
            if not _equal(decode[r.rid][j], h.reference[r.rid][j].logits):
                failures.append(f"{r.name}: the decode leg's token {j} is not the reference's")
    h.restore()
    worst_logit, worst_gap, flips, log = 0.0, 0.0, [], []
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
        log.append(f"drift {r.name}: {r.drift_steps * TOKENS} tokens, {r_logit:.4f} / {r_gap:.4f}")
    h.restore()
    log.append(
        f"drift: max |logit diff| {worst_logit:.4f}, max top-gap shift {worst_gap:.4f} over "
        f"{sum(r.drift_steps for r in probes) * TOKENS} positions; top-1 changes (probe, "
        f"position, decode margin): {flips}"
    )
    if worst_logit > DRIFT_LOGIT_BAND or worst_gap > DRIFT_GAP_BAND:
        failures.append(f"drift {worst_logit:.4f} / {worst_gap:.4f} outside the drift bands")
    wide = [f for f in flips if f[2] > DRIFT_GAP_BAND]
    if wide:
        failures.append(f"top-1 changes off a near-tie: {wide}")
    return {
        "failures": failures,
        "log": log,
        "metrics": {"logit": worst_logit, "gap": worst_gap, "flips": flips},
    }


def check_the_mtp_layer_drafts_as_prefill_drafts(h: Harness) -> dict:
    """The MTP layer's draft step runs the same kernels in the decode target as in the prefill
    target -- cuBLAS on `_mtp_view`'s `.t()` views, thop_attention, the trunk's all-reduces -- so
    its drafts are the ones the target drafted before decode was made batch-invariant: on a
    draft step of every probe (t1 after the trunk's hidden state of t0, over the MTP layer's KV
    of the prompt), the decode target's output, through ``MTPLayer``'s routing, equals the
    prefill target's bit for bit, and every rank's equals rank 0's. The decode target's
    row-invariant ``_moe`` reads the stored [out, in] weights, so handing it the MTP layer's
    views misreads them; its guard says so."""
    from _validating import validating

    M = _modeling()
    rows = list(h.probes)
    ids = torch.tensor([r.tokens[1] for r in rows], dtype=torch.int, device="cuda")
    hidden = torch.cat([h.hidden0[r.rid] for r in rows])
    mtp_layer = h.model.draft_model.mtp_layers[0]

    def step(target):
        h.fill(h.md, rows, num_contexts=0, tokens=1)
        return h.mtp(h.md, target, ids, hidden)

    failures, log = [], []
    with validating(M.cublas_mm):
        routed = step(lambda e, i, x, md: mtp_layer(e, None, i, x, md))
        decode = step(h.core._decode.mtp_forward)
        prefill = step(h.core._prefill.mtp_forward)
    if not bool(torch.isfinite(prefill).all()):
        failures.append("non-finite MTP layer output")
    if not _equal(decode, prefill):
        failures.append("the decode target's MTP layer differs from the prefill target's")
    if not _equal(routed, decode):
        failures.append("MTPLayer did not route a decode-only draft step to DecodeTarget")
    drafts = mtp_layer.shared_head(prefill, h.model.lm_head, h.md, False).argmax(-1).tolist()
    agree = sum(d == r.tokens[2] for d, r in zip(drafts, rows))
    log.append(f"draft step: {agree} of {len(rows)} drafts equal the trunk's greedy t2")
    gathered = COMM.gather(_bits(prefill).cpu().numpy().tobytes(), root=0)
    if RANK == 0 and any(theirs != gathered[0] for theirs in gathered):
        failures.append("the ranks' MTP layer outputs differ")

    original = M.DecodeTarget._mtp_moe
    M.DecodeTarget._mtp_moe = M.DecodeTarget._moe
    try:
        with validating(M.invariant_mm):
            step(h.core._decode.mtp_forward)
        failures.append("the decode target's _moe took the MTP layer's .t() views unguarded")
    except AssertionError as exc:
        if "x and weight must share K" not in str(exc):
            failures.append(f"the hazard raised something else: {str(exc).splitlines()[0]}")
    finally:
        M.DecodeTarget._mtp_moe = original
    return {"failures": failures, "log": log}


CHECKS = (
    ("reference_rows_are_real", check_reference_rows_are_real),
    ("every_rank_forms_the_same_logits_and_kv", check_every_rank_forms_the_same_logits_and_kv),
    (
        "decode_attends_through_the_invariant_entry_and_a_verify_through_thop",
        check_decode_attends_through_the_invariant_entry_and_a_verify_through_thop,
    ),
    ("verify_alone_is_within_the_band_of_decode", check_verify_alone_is_within_the_band_of_decode),
    ("verify_rows_in_eager_batches", check_verify_rows_in_eager_batches),
    ("verify_rows_in_padded_graphs", check_verify_rows_in_padded_graphs),
    (
        "commit_promotes_the_state_the_verify_stored",
        check_commit_promotes_the_state_the_verify_stored,
    ),
    ("verify_at_the_length_limit", check_verify_at_the_length_limit),
    ("rival_cublas_mm", check_the_exact_comparison_moves_with_cublas),
    ("the_band_tells_a_state_one_token_off", check_the_band_tells_a_state_one_token_off),
    ("the_drift_over_consecutive_verifies", check_the_drift_over_consecutive_verifies),
    ("the_mtp_layer_drafts_as_prefill_drafts", check_the_mtp_layer_drafts_as_prefill_drafts),
)


def _run_one_rank() -> int:
    """Body of one MPI rank: build, run every check, and hand rank 0 the findings; a rank whose
    code raises exits nonzero, and mpirun ends the job."""
    global COMM, RANK
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    torch.cuda.set_device(local)

    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    assert COMM.Get_size() == WORLD_SIZE, f"tp4 runs {WORLD_SIZE} ranks, got {COMM.Get_size()}"
    pid_dir = os.environ.get(_PID_DIR_ENV)
    if pid_dir:
        Path(pid_dir, f"rank{RANK}.pid").write_text(str(os.getpid()))

    # Spawned rather than collected, so `__extra_import_path__` is not in play: the shared
    # helpers are reached the plain way, in these rank processes only.
    here = Path(__file__).resolve()
    sys.path[:0] = [str(here.parents[1]), str(here.parents[3])]
    import _nemotron4_reference as R

    def log(line: str) -> None:
        print(f"[rank {RANK}] {line}", flush=True)

    results = {}
    try:
        with torch.inference_mode():
            start = time.time()
            h = Harness(R.default_checkpoint(), RANK, log)
            log(f"built and prefilled in {time.time() - start:.0f}s")
            # Fillers verify their t0 and two arbitrary tokens, the same on every rank.
            g = torch.Generator().manual_seed(17)
            for r in h.fillers:
                r.tokens = r.tokens + tuple(
                    torch.randint(0, h.vocab, (DRAFT,), generator=g).tolist()
                )
            # The reference: every probe's three decode steps alone, eagerly.
            h.reference, h.hidden0 = {}, {}
            for r in h.probes + [h.limit]:
                h.reference[r.rid], h.hidden0[r.rid] = h.run_reference(r)
            start = time.time()
            h.capture_graphs()
            log(f"captured graphs {sorted(h.graphs)} in {time.time() - start:.0f}s")
            for name, check in CHECKS:
                start = time.time()
                result = check(h)
                result["seconds"] = round(time.time() - start, 1)
                results[name] = result
                for line in result["log"]:
                    log(line)
                log(f"{name}: {len(result['failures'])} failures ({result['seconds']}s)")
            log(
                f"torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
            )
    except BaseException:
        print(f"[rank {RANK}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Exit hard rather than return: a rank that leaves a collective early wedges every
        # sibling in it, and mpirun ends the job the moment it sees this exit.
        os._exit(13)
    gathered = COMM.gather(results, root=0)
    if RANK == 0:
        report = {
            "ranks": WORLD_SIZE,
            "probes": [r.name for r in h.probes],
            "batch_sizes": BATCH_SIZES,
            "graph_sizes": GRAPH_SIZES,
            "checks": {
                name: {"ranks": [rank_results[name] for rank_results in gathered]}
                for name, _ in CHECKS
            },
        }
        Path(os.environ[_REPORT_ENV]).write_text(json.dumps(report, indent=1))
    COMM.Barrier()
    torch.cuda.synchronize()
    log("done")
    sys.stdout.flush()
    # Finalize so mpirun counts this as a clean exit, then leave without running teardown: the
    # all-reduce workspace, the cache pools and the captured graphs go with the process.
    MPI.Finalize()
    os._exit(0)


def _standalone_env() -> dict:
    """This process's environment, minus any enclosing launcher's identity.

    A launching srun step or MPI job leaves its rank identity in the environment (SLURM_*, PMIX_*,
    PMI_*, OMPI_*); a nested mpirun that inherited it would join that job instead of starting its
    own world.
    """
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    return env


def _spawn_ranks(report: Path) -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 4 devices this run owns, e.g. 0,1,2,3"
    devices = [d for d in visible.split(",") if d.strip()]
    assert len(devices) == WORLD_SIZE, (
        f"CUDA_VISIBLE_DEVICES names {len(devices)} device(s); tp4 runs {WORLD_SIZE}"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; this job runs its ranks as an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="tp4_verify_invariance_")
    env = dict(_standalone_env(), **{_PID_DIR_ENV: pid_dir, _REPORT_ENV: str(report)})
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(WORLD_SIZE)]
    if os.geteuid() == 0:
        command.insert(1, "--allow-run-as-root")
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    print(f"[launcher] {' '.join(command)}", flush=True)
    job = subprocess.Popen(command, env=env, start_new_session=True)

    def _kill_all() -> None:
        for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 0)):
            try:
                os.killpg(job.pid, sig)
            except ProcessLookupError:
                pass
            try:
                job.wait(timeout=grace or None)
                break
            except subprocess.TimeoutExpired:
                continue
        # Ranks forked by mpirun's daemon may sit in their own process groups.
        for pid_file in Path(pid_dir).glob("rank*.pid"):
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass

    deadline = time.monotonic() + DEADLINE_S
    while job.poll() is None:
        if time.monotonic() > deadline:
            _kill_all()
            raise AssertionError(f"the {WORLD_SIZE}-rank run did not finish in {DEADLINE_S}s")
        time.sleep(0.5)
    shutil.rmtree(pid_dir, ignore_errors=True)
    if job.returncode != 0:
        raise AssertionError(f"the {WORLD_SIZE}-rank run failed: mpirun exited {job.returncode}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    assert "--report" in sys.argv, "usage: _tp4_verify_invariance_ranks.py --report <json>"
    _spawn_ranks(Path(sys.argv[sys.argv.index("--report") + 1]).resolve())
    print("OK", flush=True)
