# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the nemotron4 tp1 MTP verify step gives every token the bits of its decode step.

Under one-model MTP-Eagle (``max_draft_len`` 2) a generation row carries its last accepted token
and two drafts, and the trunk verifies the three in one step: 3B rows through every GEMM, three
tokens per request through attention, the GDP mixers through the verify entries. Greedy output
with MTP on is token-identical to MTP off only if token j of that step gets the bits it gets as
an MTP-off decode step of its own: its fp32 logits, the K/V it appends in every attention layer,
and the GDP recurrent and conv state after it in every GDP layer -- the state the verify stores
in the hybrid manager's per-position buffers and the worker's commit (``update_mamba_states``)
promotes into the slot.

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
#: KV tokens of the paged pools: every request stays resident.
KV_TOKENS = 192 * 1024
#: Verify batch sizes: 1 to 256, and two that leave a large graph mostly or slightly padded.
BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8, 16, 64, 100, 250, 256)
#: The CUDA-graph batch sizes captured; a batch runs in the smallest one that holds it, as the
#: runner pads.
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


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
    """One sequence: its prompt and the three tokens its verify step feeds -- the prompt's greedy
    next token t0, then t1 and t2."""

    rid: int
    name: str
    prompt: tuple[int, ...]
    tokens: tuple[int, ...] = ()


def _requests(prompts) -> tuple[list[Request], list[Request]]:
    """The probes -- the ten prompts, a history of 511 tokens, whose verify tokens straddle the
    first 512-key attention split and a page boundary, and one that ends mid-split -- and
    fillers enough for a batch of 256."""
    by_name = {p.name: p for p in prompts}
    probes = [Request(0, p.name, p.token_ids) for p in prompts]
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
    """What one token's decode step produced: its logits and what it wrote."""

    logits: torch.Tensor  # [vocab] fp32
    ssm: list[torch.Tensor]  # per GDP layer, the slot after this token: [24, 64, 128] fp32
    conv: list[torch.Tensor]  # per GDP layer, the slot after this token: [16896, 3] bf16
    kv: torch.Tensor  # [attention layers, K/V, head_dim] bf16, at this token's position


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view({4: torch.int32, 2: torch.int16}[x.element_size()])


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(_bits(a), _bits(b))


def _differences(got: Token, ref: Token, gdp_layers, attention_layers) -> list[str]:
    """What differs, bitwise, between two runs of one token."""
    out = []
    if not _equal(got.logits, ref.logits):
        diff = (got.logits - ref.logits).abs()
        out.append(
            f"logits: {int((diff > 0).sum())} of {diff.numel()} differ, max |diff| "
            f"{float(diff.max()):.3e}, argmax {int(got.logits.argmax())} vs {int(ref.logits.argmax())}"
        )
    for name, got_states, ref_states in (("ssm", got.ssm, ref.ssm), ("conv", got.conv, ref.conv)):
        bad = [layer for layer, a, b in zip(gdp_layers, got_states, ref_states) if not _equal(a, b)]
        if bad:
            out.append(f"{name} state differs in GDP layers {bad}")
    bad = [layer for layer, a, b in zip(attention_layers, got.kv, ref.kv) if not _equal(a, b)]
    if bad:
        out.append(f"appended K/V differs in attention layers {bad}")
    return out


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
            # The prompt and the step's three tokens, in both managers; the manager adds the
            # drafter's extra KV token itself.
            added = self.manager.add_dummy_requests(
                [r.rid],
                token_nums=[len(r.prompt) + TOKENS],
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

    def graph(self, rows: list[Request]) -> torch.Tensor:
        """One verify step of these rows replayed in the smallest captured graph that holds
        them, padded with the padding dummy's rows, from the snapshot; the real rows' logits."""
        size = self.graph_size(len(rows))
        padding = size - len(rows)
        if size not in self.graphs:
            self._capture(size, rows, padding)
        gmd, graph, out = self.graphs[size]
        self.restore()
        self.fill(gmd, rows, num_contexts=0, tokens=TOKENS, padding=padding)
        self.static_ids[: size * TOKENS].copy_(self._ids(rows, padding))
        graph.replay()
        return out[: len(rows) * TOKENS]

    def _capture(self, size: int, rows: list[Request], padding: int) -> None:
        """Capture the verify step at ``size`` rows of three tokens, as the runner does: its own
        CUDA-graph metadata at draft length 2, two eager warmup forwards, then the capture."""
        from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

        gmd = self.md.create_cuda_graph_metadata(size, False, DRAFT, get_memory_buffers())
        self.fill(gmd, rows, num_contexts=0, tokens=TOKENS, padding=padding)
        self.static_ids[: size * TOKENS].copy_(self._ids(rows, padding))
        ids = self.static_ids[: size * TOKENS]
        for _ in range(2):
            self.trunk(gmd, ids)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            out = self.trunk(gmd, ids)
        torch.cuda.synchronize()
        self.graphs[size] = (gmd, graph, out)

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
    yield h
    h.graphs.clear()
    h.manager.shutdown()
    h.draft_manager.shutdown()


def _check(h: Harness, rows: list[Request], positions: list[int], logits, what: str) -> list[str]:
    failures = []
    for pos in positions:
        r = rows[pos]
        for j in range(TOKENS):
            for diff in _differences(
                h.verify_token(r, pos, j, logits),
                h.reference[r.rid][j],
                h.gdp_layers,
                h.attention_layers,
            ):
                failures.append(f"{what}, {r.name} at row {pos}, token {j}: {diff}")
    return failures


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
def test_verify_alone_equals_decode(harness):
    """Every probe's verify step alone (B = 1), eagerly, with every entry's guard on: token j's
    logits, the state the verify stored for it and the K/V it appended are the bits of its
    decode step; the verify writes no slot. The guards check each call's inputs, among them
    that the cache manager's ``max_seq_len`` bounds every length the attention sees, the
    drafts included."""
    M = _modeling()
    h = harness
    failures = []
    with validating(
        M.invariant_gen_attention,
        M.invariant_mm,
        M.cublas_mm,
        M.gdp_verify,
        M.causal_conv1d_verify,
    ):
        for r in h.probes:
            logits = h.eager([r])
            assert h.changed_slots() == set(), f"{r.name}: the verify wrote a state slot"
            failures += _check(h, [r], [0], logits, "alone")
    h.log(f"alone: {len(h.probes)} verifies, {len(failures)} differences")
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_verify_rows_do_not_depend_on_the_batch(harness):
    """Every probe inside eager verify batches of every size: the bits of its decode steps."""
    h = harness
    failures, runs = [], 0
    for size in BATCH_SIZES:
        if size == 1:
            continue  # test_verify_alone_equals_decode
        for rows, positions in _batches(h, size):
            logits = h.eager(rows)
            failures += _check(h, rows, positions, logits, f"eager B={size}")
            runs += 1
    h.log(f"eager: {runs} batches, {len(failures)} differences")
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_graph_rows_equal_the_decode_rows(harness):
    """Every probe inside padded CUDA-graph verify batches of every size, B = 1 included: the
    bits of its decode steps. The verify writes no slot, the padding rows' included."""
    h = harness
    failures, runs = [], 0
    for size in BATCH_SIZES:
        for rows, positions in _batches(h, size):
            logits = h.graph(rows)
            failures += _check(
                h, rows, positions, logits, f"graph B={size} in {h.graph_size(size)}"
            )
            written = h.changed_slots()
            assert written == set(), (
                f"graph B={size}: the verify wrote slots {sorted(written)} "
                f"(the padding dummy's slot is {h.dummy_slot})"
            )
            runs += 1
    h.log(f"graph: {runs} batches over sizes {sorted(h.graphs)}, {len(failures)} differences")
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_commit_leaves_the_decode_state_of_the_accepted_tokens(harness):
    """After a padded graph verify of every probe (15 rows in the graph of 16), the worker's
    commit (``update_mamba_states`` with the engine's own slots, the padding dummy's included)
    leaves in each probe's slot the state its decode steps reach over the accepted tokens, for
    one, two and three accepted, in every GDP layer; it writes no other request's slot."""
    h = harness
    rows = h.probes + h.fillers[:3]
    accepted = [1 + k % TOKENS for k in range(len(rows))]
    size = h.graph_size(len(rows))
    h.graph(rows)
    gmd = h.graphs[size][0]
    counts = torch.tensor(accepted + [1] * (size - len(rows)), dtype=torch.int32, device="cuda")
    h.manager.update_mamba_states(
        attn_metadata=gmd,
        num_accepted_tokens=counts,
        state_indices=gmd.mamba_metadata.state_indices,
    )
    written = h.changed_slots()
    assert written <= {h.slot[r.rid] for r in rows} | {h.dummy_slot}, sorted(written)
    failures = []
    for r, n in zip(h.probes, accepted):
        ref = h.reference[r.rid][n - 1]
        slot = h.slot[r.rid]
        for layer_index, layer in enumerate(h.gdp_layers):
            if not _equal(h.manager.get_ssm_states(layer)[slot], ref.ssm[layer_index]):
                failures.append(f"{r.name}, {n} accepted: ssm state differs in GDP layer {layer}")
            if not _equal(h.manager.get_conv_states(layer)[slot], ref.conv[layer_index]):
                failures.append(f"{r.name}, {n} accepted: conv state differs in GDP layer {layer}")
    h.restore()
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_verify_at_the_length_limit(harness, monkeypatch):
    """A request whose last accepted token is its ``MAX_SEQ_LEN - 1``-th, the longest history a
    verify starts from, verifies two drafts: its lengths reach ``MAX_SEQ_LEN + 1``, past the
    limit itself. The cache manager's ``max_seq_len`` -- which the attention entry is handed
    and sizes its grid from -- counts the drafts, so the entry's guard accepts the call; at the
    bare ``MAX_SEQ_LEN`` it refuses it. The verify, alone and in a padded graph, gives each
    token the bits of its decode step."""
    M = _modeling()
    h = harness
    r = h.limit
    calls = []
    original = M.invariant_gen_attention

    def recorded(**kwargs):
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(M, "invariant_gen_attention", recorded)
    with validating(original):
        logits = h.eager([r])
    monkeypatch.undo()
    longest = int(calls[0]["sequence_length"][0])
    bound = calls[0]["max_seq_len"]
    h.log(f"at the limit: lengths up to {longest}, max_seq_len {bound} (net {MAX_SEQ_LEN})")
    assert longest == MAX_SEQ_LEN + DRAFT - 1 and bound >= longest
    assert all(bool(torch.isfinite(t.logits).all()) for t in h.reference[r.rid])
    with pytest.raises(AssertionError, match="exceeds max_seq_len"):
        original.is_valid(**dict(calls[0], max_seq_len=MAX_SEQ_LEN))
    failures = _check(h, [r], [0], logits, "alone at the limit")
    rows = [r, h.probes[0], h.probes[1]]
    failures += _check(h, rows, [0, 1, 2], h.graph(rows), "graph B=3 at the limit")
    h.restore()
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_the_verify_rows_move_with_cublas(harness, monkeypatch):
    """The comparison has teeth on this model's own state: with the GEMMs that run through
    ``invariant_mm`` put back on ``cublas_mm`` -- what the verify ran before it was made
    invariant -- some probe's verify tokens differ from that rival's own decode steps."""
    M = _modeling()
    h = harness

    def replaced(x, weight, out_dtype=None):
        return M.cublas_mm(x, weight.t(), None, out_dtype)

    monkeypatch.setattr(M, "invariant_mm", replaced)
    differ = []
    for r in h.probes:
        h.restore()
        ref = []
        for j in range(TOKENS):
            ref.append(h.slot_token(r, j, h.decode(r, j)))
        logits = h.eager([r])
        for j in range(TOKENS):
            diffs = _differences(
                h.verify_token(r, 0, j, logits), ref[j], h.gdp_layers, h.attention_layers
            )
            if diffs:
                differ.append(f"{r.name} token {j}: {'; '.join(diffs)}")
    h.restore()
    h.log(f"rival cublas_mm: {len(differ)} of {len(h.probes) * TOKENS} verify tokens differ")
    for line in differ:
        h.log(f"  {line}")
    assert differ, (
        "with cublas_mm back in the verify no token moved, so the comparison cannot tell an "
        "invariant verify from a variant one here"
    )


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
