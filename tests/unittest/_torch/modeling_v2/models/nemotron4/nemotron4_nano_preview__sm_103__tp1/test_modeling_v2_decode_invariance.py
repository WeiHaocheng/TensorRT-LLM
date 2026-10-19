# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: how far the nemotron4 tp1 decode step moves with its batch, on real weights and real
cache state.

A decode step runs each bf16 GEMM through the low-M kernel certified for its row count or through
cuBLAS (``DecodeTarget._mm``), its router through ``dsv3_router_gemm`` up to 8 rows, and its
attention through ``thop_attention``'s generation kernels, and every one of them picks its kernel by
the batch: the GEMMs by the row count, thop by the batch size and history length. So a request's
decode step gives slightly different bits alone, inside a larger eager batch and inside a padded
CUDA-graph batch, whose padding rows can move a GEMM onto another kernel -- what CUDA graphs change between the two
legs of a CUDA-graph on / off identity run, and what batching changes between a prompt alone and in
a batch. This test measures how far, per request: its fp32 logits (the largest |difference| and the
largest shift of a gap between its top-1 and one of its top `TOP` tokens), and the GDP recurrent and
conv state it writes into its slot (32 layers) and the K/V it appends (8 attention layers), each as
max |difference| / max |alone|. Every quantity is held to a band derived from that measurement
(`LOGIT_BAND` and the rest); the MTP identity gate's near-tie band and the MTP acceptance gate's
batch term rest on it. The padding rows of a graph batch write no slot, exactly.

What runs is the target and the engine's own pieces around it, built as the engine builds them:
``ModelConfig`` off the checkpoint, the target class, ``load_weights`` over every checkpoint
tensor and ``post_load_weights``; the engine's hybrid cache manager (``_create_kv_cache_manager``,
the default ``MambaHybridCacheManagerV2``); the engine's TRTLLM attention metadata
(``build_attention_metadata``), filled per step as the executor's runner fills it; and, for
graphs, ``create_cuda_graph_metadata`` and the runner's padding dummy (``add_dummy_requests`` of
``CUDA_GRAPH_DUMMY_REQUEST_ID`` as a generation request), whose rows pad every graph batch.

The requests: the ten prompts of the fixed set, two prefixes of the long ones and 244 more
prefixes as fillers, prefilled through the target's own prefill steps, so that every request
decodes from KV, recurrent and conv state the model wrote. Each then decodes its greedy next
token. The twelve probes run alone first, eagerly: that is the reference. Then they run inside
eager batches of every size below, at positions spread over the batch, and inside CUDA graphs
captured at the runner's sizes. Every run starts from the same state. Each GDP slot is restored
from a snapshot. A row's KV append lands at the one position its own attention reads only after
the append, so an earlier run's append there is overwritten before it can be read.
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

_MODELING = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4."
    "nemotron4_nano_preview__sm_103__tp1.modeling"
)

# The engine sizes this run uses everywhere (task.yaml, the boot gate, check_goldens).
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
TOKENS_PER_BLOCK = 32
#: Requests resident at once: the probes and fillers, and the padding dummy beside them.
NUM_REQUESTS = 256
#: Rows a step may hold.
MAX_BATCH = 264
#: KV tokens of the paged pool: every request stays resident.
KV_TOKENS = 192 * 1024
#: Decode batch sizes: 1 to 256, and two that leave a large graph mostly or slightly padded.
BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8, 16, 64, 100, 250, 256)
#: The CUDA-graph batch sizes captured; a batch runs in the smallest one that holds it, as the
#: runner pads.
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
#: The alone step's top tokens whose gap to its top-1 a batched step must keep (`Drift.gap`).
TOP = 8

# The bands: how far a probe's decode step in a batch may sit from the same step alone. Measured on
# this model's own state over every probe in the eager batches (2 to 256 rows, 132 rows) and the
# padded graphs (1 to 256, 144 rows) below; the kernels are deterministic, so the worst cases repeat
# run to run. The worst cases, eager and graph alike: |logit diff| 0.9276, top-gap shift 0.4056,
# recurrent state 0.0390, conv state 0.0500 and K/V 0.0633 of max |alone|; no top-1 changed. A graph
# batch against the same rows run eagerly, what the CUDA-graph identity gate's legs differ by in one
# step, measured 0.9966 / 0.3790: a padded graph batch can run a GEMM on another low-M kernel than
# the eager batch of its real rows. Each band is its worst case plus a quarter, rounded up: room
# for a batch, history or graph size the runs did not take, where a GEMM or thop may pick yet
# another kernel.
#: max |batched logit - alone logit| over the vocabulary, fp32.
LOGIT_BAND = 1.16
#: The largest change, batched against alone, of the gap between the alone step's top-1 and any of
#: its top `TOP` tokens: how close to a tie the batch can tip the alone step's choice.
GAP_BAND = 0.51
#: max |batched - alone| / max |alone|, per tensor, over every GDP layer (recurrent state fp32, conv
#: state bf16) and every attention layer (the appended K/V).
SSM_BAND = 0.049
CONV_BAND = 0.063
KV_BAND = 0.080


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


def _build_target(path: str):
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


def _build_manager(config):
    """The engine's hybrid cache manager for this config (the default V2 manager): paged KV for
    the attention layers, recurrent and conv state slots for the GDP layers."""
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    kv_cache_config = KvCacheConfig(
        max_tokens=KV_TOKENS, enable_block_reuse=False, free_gpu_memory_fraction=0.5
    )
    return _create_kv_cache_manager(
        None,
        get_kv_cache_manager_cls(config, kv_cache_config),
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
        max_cuda_graph_batch_size=max(GRAPH_SIZES),
    )


def _build_metadata(config, manager):
    """The engine's TRTLLM attention metadata, chunked prefill off."""
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
    """One sequence: its prompt, and the token its decode step feeds (the prompt's greedy next
    token)."""

    rid: int
    name: str
    prompt: tuple[int, ...]
    token: int = -1


def _requests(prompts) -> tuple[list[Request], list[Request]]:
    """The probes -- the ten prompts, plus a history that fills one 512-key attention split
    exactly and one that ends mid-split -- and fillers enough for a batch of 256."""
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


def _fill(md, manager, rows: list[Request], num_contexts: int, padding: int = 0) -> None:
    """The step's per-request fields, set as the runner's ``_prepare_tp_inputs`` sets them, then
    ``prepare()``. Context rows feed their whole prompt; generation rows feed one token over the
    prompt already cached; ``padding`` rows are the padding dummy's (prompt length 0, nothing
    cached, so one key: the dummy's own page)."""
    # The cache manager stages a step's block-table gather index in one pinned host buffer
    # that the device reads only when the step's copy runs (`IndexMapper::getCopyIndex`): a
    # step filled while an earlier one is still queued would hand the earlier one its rows,
    # and that step's K/V appends would land in another request's pages. So the device
    # drains first.
    torch.cuda.synchronize()
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

    if not md.is_cuda_graph:
        md.seq_lens = torch.tensor(
            [len(r.prompt) for r in rows[:num_contexts]] + [1] * (len(rows) - num_contexts),
            dtype=torch.int,
        )
    md.beam_width = 1
    md.request_ids = [r.rid for r in rows] + [CUDA_GRAPH_DUMMY_REQUEST_ID] * padding
    md.prompt_lens = [len(r.prompt) for r in rows] + [0] * padding
    md.num_contexts = num_contexts
    md.num_chunked_ctx_requests = 0
    md.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=[0] * num_contexts
        + [len(r.prompt) for r in rows[num_contexts:]]
        + [0] * padding,
        num_extra_kv_tokens=0,
    )
    md.kv_cache_manager = manager
    md.padded_num_tokens = None
    md.prepare()


@dataclasses.dataclass
class Row:
    """What one request's decode step produced: its logits and what it wrote."""

    logits: torch.Tensor  # [vocab] fp32
    ssm: list[torch.Tensor]  # per GDP layer, its slot: [24, 64, 128] fp32
    conv: list[torch.Tensor]  # per GDP layer, its slot: [16896, 3] bf16
    kv: torch.Tensor  # [attention layers, K/V, head_dim] bf16, at the decoded token's position


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view({4: torch.int32, 2: torch.int16}[x.element_size()])


def _relative(got: torch.Tensor, ref: torch.Tensor) -> float:
    """max |got - ref| / max |ref|."""
    ref = ref.float()
    return float((got.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _logit_drift(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
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
class Drift:
    """How a request's decode step in a batch compares with the same step alone."""

    logit: float
    gap: float
    ssm: float
    conv: float
    kv: float
    argmax: tuple[int, int]  # (batched, alone)
    margin: float  # the alone step's top-1 / top-2 margin

    def outside(self) -> list[str]:
        """What lies outside the bands."""
        out = []
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


def _drift(got: Row, ref: Row) -> Drift:
    """Compare a request's decode step with the same step alone."""
    logit, gap = _logit_drift(got.logits, ref.logits)
    return Drift(
        logit=logit,
        gap=gap,
        ssm=max(_relative(a, b) for a, b in zip(got.ssm, ref.ssm)),
        conv=max(_relative(a, b) for a, b in zip(got.conv, ref.conv)),
        kv=max(_relative(a, b) for a, b in zip(got.kv, ref.kv)),
        argmax=(int(got.logits.argmax()), int(ref.logits.argmax())),
        margin=_margin(ref.logits),
    )


def _summary(drifts: list[Drift]) -> str:
    flips = [d for d in drifts if d.argmax[0] != d.argmax[1]]
    return (
        f"{len(drifts)} rows: max |logit diff| {max(d.logit for d in drifts):.4f}, max top-gap "
        f"shift {max(d.gap for d in drifts):.4f}, recurrent / conv state / K/V "
        f"{max(d.ssm for d in drifts):.3g} / {max(d.conv for d in drifts):.3g} / "
        f"{max(d.kv for d in drifts):.3g} of max |alone|; {len(flips)} top-1 changes"
        + (f" at alone margins {[round(d.margin, 4) for d in flips]}" if flips else "")
    )


class Harness:
    """The loaded target, its cache manager and metadata, the prefilled requests, and the state
    every decode run starts from."""

    def __init__(self, checkpoint: str, log=print):
        self.log = log
        tokenizer = P.load_tokenizer(checkpoint)
        self.config, self.model = _build_target(checkpoint)
        self.manager = _build_manager(self.config)
        self.md = _build_metadata(self.config, self.manager)
        core = self.model.model
        self.gdp_layers = core.gdp_layers
        self.attention_layers = core.attention_layers
        self.probes, self.fillers = _requests(P.prompt_set(tokenizer))
        requests = self.probes + self.fillers
        for r in requests:
            added = self.manager.add_dummy_requests([r.rid], token_nums=[len(r.prompt) + 1])
            assert added, f"the cache manager has no room for {r.name} ({len(r.prompt)} tokens)"
        self._prefill(requests)
        # The runner's padding dummy: a generation request whose one token is its own.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        added = self.manager.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID], is_gen=True)
        assert added, "the cache manager has no room for the CUDA-graph padding dummy"
        ids = [r.rid for r in requests]
        self.slot = dict(zip(ids, self.manager.get_state_indices(ids)))
        self.dummy_slot = self.manager.get_state_indices([CUDA_GRAPH_DUMMY_REQUEST_ID], [True])[0]
        assert len(set(self.slot.values()) | {self.dummy_slot}) == len(requests) + 1
        self.kv_buffers = {i: self.manager.get_buffers(i) for i in self.attention_layers}
        self.pages = {
            r.rid: {
                i: self.manager.get_batch_cache_indices([r.rid], layer_idx=i)[0]
                for i in self.attention_layers
            }
            for r in requests
        }
        self.snapshot = [
            (self.manager.get_ssm_states(i).clone(), self.manager.get_conv_states(i).clone())
            for i in self.gdp_layers
        ]
        self.static_ids = torch.zeros(max(GRAPH_SIZES), dtype=torch.int, device="cuda")
        self.graphs: dict[int, tuple] = {}
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.padding_tokens = torch.Generator(device="cuda").manual_seed(9)

    def _prefill(self, requests: list[Request]) -> None:
        """Every prompt through the target's prefill steps, whole prompts packed up to
        ``MAX_NUM_TOKENS`` tokens a step; each request's decode token is its greedy next token."""
        batch: list[Request] = []
        for r in requests + [None]:
            if batch and (
                r is None or sum(len(b.prompt) for b in batch) + len(r.prompt) > MAX_NUM_TOKENS
            ):
                _fill(self.md, self.manager, batch, num_contexts=len(batch))
                ids = torch.tensor(
                    [t for b in batch for t in b.prompt], dtype=torch.int, device="cuda"
                )
                logits = self.model.forward(attn_metadata=self.md, input_ids=ids)
                for b, token in zip(batch, logits.argmax(-1).tolist()):
                    b.token = token
                batch = []
            if r is not None:
                batch.append(r)
        torch.cuda.synchronize()
        self.log(f"prefilled {len(requests)} requests")

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
        """The step's input ids: each row's decode token, then arbitrary tokens on the padding
        rows, as the runner leaves whatever its static buffer held there."""
        real = torch.tensor([r.token for r in rows], dtype=torch.int, device="cuda")
        pad = torch.randint(
            0,
            self.config.pretrained_config.text_config.vocab_size,
            (padding,),
            generator=self.padding_tokens,
            dtype=torch.int,
            device="cuda",
        )
        return torch.cat([real, pad])

    def eager(self, rows: list[Request]) -> torch.Tensor:
        """One decode step of exactly these rows, eagerly, from the snapshot; the rows' logits."""
        self.restore()
        _fill(self.md, self.manager, rows, num_contexts=0)
        return self.model.forward(attn_metadata=self.md, input_ids=self._ids(rows, 0))

    def graph_size(self, num_rows: int) -> int:
        return min(g for g in GRAPH_SIZES if g >= num_rows)

    def graph(self, rows: list[Request]) -> torch.Tensor:
        """One decode step of these rows replayed in the smallest captured graph that holds
        them, padded with the padding dummy's rows, from the snapshot; the real rows' logits."""
        size = self.graph_size(len(rows))
        padding = size - len(rows)
        gmd, graph, out = self.graphs[size]
        assert gmd.cuda_graph_workspace.data_ptr() == self.workspace_address, (
            "the attention workspace moved under the captured graphs"
        )
        self.restore()
        _fill(gmd, self.manager, rows, num_contexts=0, padding=padding)
        self.static_ids[:size].copy_(self._ids(rows, padding))
        graph.replay()
        return out[: len(rows)]

    def capture_graphs(self) -> None:
        """Capture the decode step at every size of GRAPH_SIZES, as the runner does: first one
        eager forward on every size's own CUDA-graph metadata, which grows the attention
        workspace all of them share to what the largest call needs -- thop_attention resizes it
        in place when a call needs more -- then at each size two warmup forwards and the
        capture, which resize nothing. A resize after a capture would move the workspace under
        the graph, which keeps its old address."""
        from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

        rows = self.probes + self.fillers
        metadata = {}
        for size in sorted(GRAPH_SIZES, reverse=True):
            gmd = self.md.create_cuda_graph_metadata(size, False, 0, get_memory_buffers())
            _fill(gmd, self.manager, rows[:size], num_contexts=0)
            self.static_ids[:size].copy_(self._ids(rows[:size], 0))
            self.model.forward(attn_metadata=gmd, input_ids=self.static_ids[:size])
            metadata[size] = gmd
        torch.cuda.synchronize()
        self.workspace_address = metadata[max(GRAPH_SIZES)].cuda_graph_workspace.data_ptr()
        for size in GRAPH_SIZES:
            gmd = metadata[size]
            _fill(gmd, self.manager, rows[:size], num_contexts=0)
            self.static_ids[:size].copy_(self._ids(rows[:size], 0))
            ids = self.static_ids[:size]
            for _ in range(2):
                self.model.forward(attn_metadata=gmd, input_ids=ids)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                out = self.model.forward(attn_metadata=gmd, input_ids=ids)
            torch.cuda.synchronize()
            self.graphs[size] = (gmd, graph, out)
        assert gmd.cuda_graph_workspace.data_ptr() == self.workspace_address, (
            "a capture resized the attention workspace"
        )
        self.restore()

    def kv_at(self, r: Request, layer: int) -> torch.Tensor:
        """``[K/V, head_dim]``: what attention layer ``layer`` holds at ``r``'s decoded position,
        read through the manager's own page table and buffer view."""
        pos = len(r.prompt)
        page = self.pages[r.rid][layer][pos // TOKENS_PER_BLOCK]
        return self.kv_buffers[layer][page, :, pos % TOKENS_PER_BLOCK, 0].clone()

    def row(self, r: Request, logits: torch.Tensor) -> Row:
        """What ``r``'s step left: its logits row, its slot's state in every GDP layer, and the
        K/V appended at its decoded position in every attention layer."""
        slot = self.slot[r.rid]
        return Row(
            logits=logits.clone(),
            ssm=[self.manager.get_ssm_states(i)[slot].clone() for i in self.gdp_layers],
            conv=[self.manager.get_conv_states(i)[slot].clone() for i in self.gdp_layers],
            kv=torch.stack([self.kv_at(r, i) for i in self.attention_layers]),
        )


def _batches(h: Harness, size: int) -> list[tuple[list[Request], list[int]]]:
    """Batches of ``size`` decode rows that hold every probe at least once, each with the row
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
    if _free_gib() < 220:
        pytest.skip("the whole tp1 target needs ~200 GiB of free GPU memory")
    with torch.inference_mode():
        h = Harness(checkpoint)
        # The reference: every probe decoding alone, eagerly.
        h.reference = {}
        for r in h.probes:
            logits = h.eager([r])
            h.reference[r.rid] = h.row(r, logits[0])
        h.capture_graphs()
    yield h
    h.graphs.clear()
    h.manager.shutdown()


def _compare(h: Harness, rows: list[Request], positions: list[int], logits) -> list[Drift]:
    """Every probe at ``positions`` against its step alone."""
    return [_drift(h.row(rows[pos], logits[pos]), h.reference[rows[pos].rid]) for pos in positions]


def _failures(what: str, rows, positions, drifts: list[Drift]) -> list[str]:
    return [
        f"{what}, {rows[pos].name} at row {pos}: {f}"
        for pos, d in zip(positions, drifts)
        for f in d.outside()
    ]


@torch.inference_mode()
def test_reference_rows_are_real(harness):
    """The reference is worth comparing against: every probe's logits are finite, its slot holds
    non-zero state in every GDP layer, and the K/V read at its decoded position through the
    manager's page table is non-zero in every attention layer."""
    h = harness
    for r in h.probes:
        ref = h.reference[r.rid]
        assert bool(torch.isfinite(ref.logits).all()), r.name
        assert all(bool(s.any()) for s in ref.ssm) and all(bool(c.any()) for c in ref.conv), r.name
        assert all(bool(kv.any()) for kv in ref.kv), r.name


@torch.inference_mode()
def test_decode_attends_through_thop_and_appends_each_rows_kv(harness, monkeypatch):
    """A decode step's eight attention layers run ``thop_attention`` and nothing else, and on a
    64-row eager step over the engine's own pool (histories of 5 to 7984 tokens) each row's K and
    V land where the manager's page table puts its token: the K and V columns of its qkv row, bit
    for bit."""
    h = harness
    M = _modeling()
    d = h.model.model._decode._d
    k_cols = slice(d.q_width, d.q_width + d.head_dim)
    v_cols = slice(d.q_width + d.head_dim, d.q_width + 2 * d.head_dim)
    rows = _batches(h, 64)[0][0]
    calls = []
    original = M.thop_attention

    def recorded(**kwargs):
        out = original(**kwargs)
        num_rows = len(rows)
        layer = next(
            i for i in h.attention_layers if h.manager.layer_offsets[i] == kwargs["local_layer_idx"]
        )
        q = kwargs["q"][:num_rows]
        expected = torch.stack([q[:, k_cols], q[:, v_cols]], dim=1)
        appended = torch.stack([h.kv_at(r, layer) for r in rows])
        calls.append((layer, expected.clone(), appended))
        return out

    monkeypatch.setattr(M, "thop_attention", recorded)
    h.eager(rows)
    assert len(calls) == len(h.attention_layers), [c[0] for c in calls]
    for layer, expected, appended in calls:
        assert torch.equal(_bits(appended), _bits(expected)), (
            f"layer {layer}: the appended K/V are not the qkv GEMM's K/V columns"
        )


@torch.inference_mode()
def test_eager_rows_stay_within_the_band(harness):
    """Every probe inside eager decode batches of every size, against its step alone: within the
    bands. Reports the drift and every top-1 change with the alone step's margin there."""
    h = harness
    failures, drifts = [], []
    for size in BATCH_SIZES:
        if size == 1:
            continue  # the reference itself
        for rows, positions in _batches(h, size):
            found = _compare(h, rows, positions, h.eager(rows))
            drifts += found
            failures += _failures(f"eager B={size}", rows, positions, found)
    h.log(f"eager: {_summary(drifts)}")
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_graph_rows_stay_within_the_band(harness):
    """Every probe inside padded CUDA-graph decode batches of every size, B = 1 included, against
    its eager step alone: within the bands. The padding rows write no slot: after every replay
    only the real rows' slots differ from the snapshot. Reports, besides, each graph batch against
    the same rows run eagerly -- what the CUDA-graph identity gate's two legs differ by in one
    step."""
    h = harness
    failures, drifts, legs = [], [], []
    for size in BATCH_SIZES:
        for rows, positions in _batches(h, size):
            logits = h.graph(rows)
            found = _compare(h, rows, positions, logits)
            drifts += found
            failures += _failures(f"graph B={size} in {h.graph_size(size)}", rows, positions, found)
            written = h.changed_slots()
            real = {h.slot[r.rid] for r in rows}
            assert written <= real, (
                f"graph B={size}: a padding row wrote slots {sorted(written - real)} "
                f"(the padding dummy's slot is {h.dummy_slot})"
            )
            graph_logits = logits[: len(rows)].clone()
            eager_logits = h.eager(rows)
            legs += [_logit_drift(graph_logits[pos], eager_logits[pos]) for pos in positions]
    h.log(f"graph: sizes {sorted(h.graphs)}, {_summary(drifts)}")
    h.log(
        f"graph against eager, the same rows: max |logit diff| {max(x for x, _ in legs):.4f}, "
        f"max top-gap shift {max(y for _, y in legs):.4f} over {len(legs)} rows"
    )
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_the_band_tells_a_state_one_token_off(harness):
    """The bands have teeth: a decode step from the state one token too far -- every GDP slot
    holding the state its own decode step left, as a rollback that kept a rejected draft would
    leave it -- moves every probe's logits by more than twice the logit band. (Measured: at least
    4.551, on the 4096-token prefix of `gsm8k_multiturn`, whose top gaps it moves by only 0.268,
    inside the gap band: there the logit band is what tells the two states apart.)"""
    h = harness
    smallest, smallest_gap = float("inf"), float("inf")
    for r in h.probes:
        h.restore()
        after = h.reference[r.rid]
        slot = h.slot[r.rid]
        for k, layer in enumerate(h.gdp_layers):
            h.manager.get_ssm_states(layer)[slot] = after.ssm[k]
            h.manager.get_conv_states(layer)[slot] = after.conv[k]
        _fill(h.md, h.manager, [r], num_contexts=0)
        logits = h.model.forward(attn_metadata=h.md, input_ids=h._ids([r], 0))
        worst, worst_gap = _logit_drift(logits[0], after.logits)
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
    assert smallest > 2 * LOGIT_BAND, (smallest, smallest_gap)
