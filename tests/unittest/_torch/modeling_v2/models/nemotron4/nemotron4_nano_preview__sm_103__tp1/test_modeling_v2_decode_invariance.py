# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the nemotron4 tp1 decode step is batch-invariant, on real weights and real cache state.

A request's decode step must give the same bits whatever batch it runs in -- alone, inside a
larger eager batch, or inside a padded CUDA-graph batch: its fp32 logits, the GDP recurrent and
conv state it writes into its slot (32 layers) and the K/V it appends (8 attention layers). That
is what CUDA graphs change between the two legs of a CUDA-graph on/off identity run: a decode step
of B requests runs as B rows eagerly, and as the next captured batch size, padding rows included,
under a graph.

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
from _validating import ULP, assert_within_ulp  # noqa: E402

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


def _differences(got: Row, ref: Row, gdp_layers, attention_layers) -> list[str]:
    """What differs, bitwise, between two of one request's decode steps."""
    out = []
    if not torch.equal(_bits(got.logits), _bits(ref.logits)):
        diff = (got.logits - ref.logits).abs()
        out.append(
            f"logits: {int((diff > 0).sum())} of {diff.numel()} differ, max |diff| "
            f"{float(diff.max()):.3e}, argmax {int(got.logits.argmax())} vs {int(ref.logits.argmax())}"
        )
    for name, got_states, ref_states in (("ssm", got.ssm, ref.ssm), ("conv", got.conv, ref.conv)):
        bad = [
            layer
            for layer, a, b in zip(gdp_layers, got_states, ref_states)
            if not torch.equal(_bits(a), _bits(b))
        ]
        if bad:
            out.append(f"{name} state differs in GDP layers {bad}")
    bad = [
        layer
        for layer, a, b in zip(attention_layers, got.kv, ref.kv)
        if not torch.equal(_bits(a), _bits(b))
    ]
    if bad:
        out.append(f"appended K/V differs in attention layers {bad}")
    return out


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
        if size not in self.graphs:
            self._capture(size, rows, padding)
        gmd, graph, out = self.graphs[size]
        self.restore()
        _fill(gmd, self.manager, rows, num_contexts=0, padding=padding)
        self.static_ids[:size].copy_(self._ids(rows, padding))
        graph.replay()
        return out[: len(rows)]

    def _capture(self, size: int, rows: list[Request], padding: int) -> None:
        """Capture the decode step at ``size`` rows, as the runner does: its own CUDA-graph
        metadata, two eager warmup forwards, then the capture."""
        from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

        gmd = self.md.create_cuda_graph_metadata(size, False, 0, get_memory_buffers())
        _fill(gmd, self.manager, rows, num_contexts=0, padding=padding)
        self.static_ids[:size].copy_(self._ids(rows, padding))
        ids = self.static_ids[:size]
        for _ in range(2):
            self.model.forward(attn_metadata=gmd, input_ids=ids)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            out = self.model.forward(attn_metadata=gmd, input_ids=ids)
        torch.cuda.synchronize()
        self.graphs[size] = (gmd, graph, out)

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
    yield h
    h.graphs.clear()
    h.manager.shutdown()


def _check(h: Harness, rows: list[Request], positions: list[int], logits, what: str) -> list[str]:
    failures = []
    for pos in positions:
        r = rows[pos]
        for diff in _differences(
            h.row(r, logits[pos]), h.reference[r.rid], h.gdp_layers, h.attention_layers
        ):
            failures.append(f"{what}, {r.name} at row {pos}: {diff}")
    return failures


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
def test_eager_rows_do_not_depend_on_the_batch(harness):
    """Every probe inside eager decode batches of every size: the same bits as alone."""
    h = harness
    failures, runs = [], 0
    for size in BATCH_SIZES:
        if size == 1:
            continue  # the reference itself
        for rows, positions in _batches(h, size):
            logits = h.eager(rows)
            failures += _check(h, rows, positions, logits, f"eager B={size}")
            runs += 1
    h.log(f"eager: {runs} batches, {len(failures)} differences")
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_graph_rows_equal_the_eager_row(harness):
    """Every probe inside padded CUDA-graph decode batches of every size: the same bits as
    alone and eager. The padding rows write no slot: after every replay only the real rows'
    slots differ from the snapshot."""
    h = harness
    failures, runs = [], 0
    for size in BATCH_SIZES:
        for rows, positions in _batches(h, size):
            logits = h.graph(rows)
            failures += _check(
                h, rows, positions, logits, f"graph B={size} in {h.graph_size(size)}"
            )
            written = h.changed_slots()
            real = {h.slot[r.rid] for r in rows}
            assert written <= real, (
                f"graph B={size}: a padding row wrote slots {sorted(written - real)} "
                f"(the padding dummy's slot is {h.dummy_slot})"
            )
            runs += 1
    h.log(f"graph: {runs} batches over sizes {sorted(h.graphs)}, {len(failures)} differences")
    assert not failures, "\n".join(failures[:40])


@torch.inference_mode()
def test_attention_is_within_band_of_thop_attention(harness, monkeypatch):
    """On a 64-row eager decode step over the engine's own pool (histories of 5 to 7984 tokens),
    in every attention layer:

    - ``invariant_gen_attention`` appends each row's K and V -- the K and V columns of the qkv
      GEMM's row, bit for bit -- where the manager's own page table puts the row's token;
    - ``thop_attention``'s append of the same tokens leaves those bytes as they were;
    - the two outputs are within the band of each other.

    Band: each kernel is within 2 element ulp of an fp32 reference (measured: 2.00 for this
    entry, 1.98 for thop's decode kernels), so 4 element ulp between them; relative RMS 0.64 and
    0.77 ulp, so at most 1.41 between them. Measured on this step: 1.87 to 1.98 element ulp and
    0.68 to 0.71 rms ulp over the eight layers.
    """
    h = harness
    M = _modeling()
    d = h.model.model._decode._d
    k_cols = slice(d.q_width, d.q_width + d.head_dim)
    v_cols = slice(d.q_width + d.head_dim, d.q_width + 2 * d.head_dim)
    rows = _batches(h, 64)[0][0]
    calls = []
    original = M.invariant_gen_attention

    def both(**kwargs):
        out = original(**kwargs)
        q, num_rows = kwargs["q"], kwargs["q"].shape[0]
        layer = next(
            i for i in h.attention_layers if h.manager.layer_offsets[i] == kwargs["local_layer_idx"]
        )
        ours = out[:num_rows].clone()
        appended = torch.stack([h.kv_at(r, layer) for r in rows])
        theirs = M.thop_attention(
            q=q,
            output=torch.empty_like(out),
            local_layer_idx=kwargs["local_layer_idx"],
            num_heads=kwargs["num_heads"],
            num_kv_heads=kwargs["num_kv_heads"],
            head_size=kwargs["head_size"],
            **h.model.model._prefill.step_args(h.md)["attention"],
            **M._CALL_CONSTANTS,
        )
        after_thop = torch.stack([h.kv_at(r, layer) for r in rows])
        expected = torch.stack([q[:num_rows, k_cols], q[:num_rows, v_cols]], dim=1)
        calls.append(
            (layer, ours, theirs[:num_rows].clone(), expected.clone(), appended, after_thop)
        )
        return out

    monkeypatch.setattr(M, "invariant_gen_attention", both)
    h.eager(rows)
    assert len(calls) == len(h.attention_layers)
    for layer, ours, theirs, expected, appended, after_thop in calls:
        assert torch.equal(_bits(appended), _bits(expected)), (
            f"layer {layer}: the appended K/V are not the qkv GEMM's K/V columns"
        )
        assert torch.equal(_bits(after_thop), _bits(appended)), (
            f"layer {layer}: thop_attention appended other bytes"
        )
        diff = (ours.float() - theirs.float()).abs()
        scale = theirs.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
        element = float((diff / scale).max()) / ULP[torch.bfloat16]
        rms = float(diff.pow(2).mean().sqrt() / theirs.float().pow(2).mean().sqrt())
        h.log(
            f"attention layer {layer}: {element:.2f} element ulp, "
            f"{rms / ULP[torch.bfloat16]:.3f} rms ulp from thop_attention"
        )
        assert_within_ulp(ours, theirs, element_ulp=4.0, rms_ulp=1.41, ulp=ULP[torch.bfloat16])


#: The eager batch sizes the rivals run at: cuBLAS changes kernels on its row-variant shapes
#: from two rows up, and thop_attention's KV split follows the batch.
RIVAL_BATCH_SIZES = (2, 3, 8, 64, 256)


@pytest.mark.parametrize("rival", ["cublas_mm", "thop_attention"])
@torch.inference_mode()
def test_the_replaced_paths_depend_on_the_batch(harness, monkeypatch, rival):
    """The comparison has teeth on this model's own state: with the decode step's GEMMs that
    run through ``invariant_mm`` put back on ``cublas_mm``, or its generation attention back on
    ``thop_attention`` -- what the decode step ran before it was made invariant -- some probe's
    step differs between its batch of one and the eager batches above. Each rival's reference
    is that rival's own batch of one. Measured: the ``cublas_mm`` rival moves 57 of the 60
    probe rows, from two rows up; the ``thop_attention`` rival moves 26, all twelve at 64 rows."""
    h = harness
    M = _modeling()
    if rival == "cublas_mm":

        def replaced(x, weight, out_dtype=None):
            return M.cublas_mm(x, weight.t(), None, out_dtype)

        monkeypatch.setattr(M, "invariant_mm", replaced)
    else:

        def replaced(q, output, local_layer_idx, num_heads, num_kv_heads, head_size, **unused):
            return M.thop_attention(
                q=q,
                output=output,
                local_layer_idx=local_layer_idx,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                **h.model.model._prefill.step_args(h.md)["attention"],
                **M._CALL_CONSTANTS,
            )

        monkeypatch.setattr(M, "invariant_gen_attention", replaced)
    reference = {r.rid: h.row(r, h.eager([r])[0]) for r in h.probes}
    differ, compared = [], 0
    for size in RIVAL_BATCH_SIZES:
        for rows, positions in _batches(h, size):
            logits = h.eager(rows)
            for pos in positions:
                r = rows[pos]
                compared += 1
                diffs = _differences(
                    h.row(r, logits[pos]), reference[r.rid], h.gdp_layers, h.attention_layers
                )
                if diffs:
                    differ.append(f"B={size} {r.name} at row {pos}: {'; '.join(diffs)}")
    h.log(f"rival {rival}: {len(differ)} of {compared} probe rows differ from their batch of one")
    for line in differ:
        h.log(f"  {line}")
    assert differ, (
        f"with {rival} back in the decode step no probe's row moved with the batch, so the "
        "comparison cannot tell an invariant step from a variant one here"
    )
