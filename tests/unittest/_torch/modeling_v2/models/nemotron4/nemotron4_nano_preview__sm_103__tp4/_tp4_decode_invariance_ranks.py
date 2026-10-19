# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4-rank job: the nemotron4 tp4 decode step is batch-invariant on every rank, on real weights and
real cache state.

A request's decode step must give the same bits whatever batch it runs in -- alone, inside a larger
eager batch, or inside a padded CUDA-graph batch: its fp32 logits, the GDP recurrent and conv state
it writes into its slot (32 layers, the rank's 6 of the 24 heads) and the K/V it appends (8
attention layers, the one KV head every rank keeps whole). That is what CUDA graphs change between
the two legs of a CUDA-graph on/off identity run: a decode step of B requests runs as B rows
eagerly, and as the next captured batch size, padding rows included, under a graph. On tp4 each
step's row-parallel partials meet in an all-reduce, so a row's bits also depend on the other
ranks' partials of that row, which every rank computes from the same rows.

This script is its own launcher, as the collective entries' matrices are: run it plainly and it
starts one MPI rank per device named in CUDA_VISIBLE_DEVICES under ``mpirun``, with a deadline it
enforces by killing the job.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _tp4_decode_invariance_ranks.py --report <json>

The ranks are an MPI world, as a tensor-parallel engine's workers are: the target's all-reduce
workspace is built over that world's communicator. Each rank builds what the engine builds on it:
``ModelConfig`` off the checkpoint with the rank's tp4 mapping (tp 4, MoE tp 4, no attention DP),
the target class, ``load_weights`` over every checkpoint tensor (the rank's slices) and
``post_load_weights``; the engine's hybrid cache manager over the rank's part of the state
(``_create_kv_cache_manager``, the default ``MambaHybridCacheManagerV2``); the engine's TRTLLM
attention metadata (``build_attention_metadata``), filled per step as the executor's runner fills
it; and, for graphs, ``create_cuda_graph_metadata`` and the runner's padding dummy
(``add_dummy_requests`` of ``CUDA_GRAPH_DUMMY_REQUEST_ID`` as a generation request), whose rows pad
every graph batch. Every rank runs the same steps in the same order, as the engine's ranks do.

The requests: the ten prompts of the fixed set, two prefixes of the long ones and 1012 more
prefixes as fillers, prefilled through the target's own prefill steps, so that every request
decodes from KV, recurrent and conv state the model wrote. Each then decodes its greedy next token.
The twelve probes run alone first, eagerly: that is the reference. Then they run inside eager
batches of every size below, at positions spread over the batch, and inside CUDA graphs captured at
the runner's sizes. Every run starts from the same state. Each GDP slot is restored from a
snapshot. A row's KV append lands at the one position its own attention reads only after the
append, so an earlier run's append there is overwritten before it can be read.

Each rank checks its own rows: its logits, its state, its appended K/V. Rank 0 gathers every rank's
findings and its reference rows, checks that every rank formed the same logits and appended the
same K/V, and writes the report the collected test reads
(``test_modeling_v2_tp4_decode_invariance.py``). Not a pytest module: the checks are one fixed
sequence inside a single 4-rank job, each reading rank state only ``_run_one_rank`` binds.
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
_PID_DIR_ENV = "MODELING_V2_TP4_INVARIANCE_PID_DIR"
_REPORT_ENV = "MODELING_V2_TP4_INVARIANCE_REPORT"
#: The weight load takes ~2 min, the checks ~10.
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
#: Requests resident at once: the probes and fillers, and the padding dummy beside them.
NUM_REQUESTS = 1024
#: Rows a step may hold.
MAX_BATCH = 1032
#: KV tokens of the paged pool: every request stays resident.
KV_TOKENS = 448 * 1024
#: Decode batch sizes: 1 to 1024, and sizes that leave a graph mostly or slightly padded.
BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8, 16, 64, 100, 250, 256, 500, 512, 1000, 1024)
#: The CUDA-graph batch sizes captured; a batch runs in the smallest one that holds it, as the
#: runner pads.
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
#: The eager batch sizes the rivals run at: cuBLAS changes kernels on the replaced shapes from 17
#: and 72 rows up, thop_attention's KV split follows the batch, and NCCL's sum the message.
RIVAL_BATCH_SIZES = (2, 3, 8, 64, 256, 1024)

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


def _build_target(path: str, mapping):
    """``(ModelConfig, the loaded target)``: the target class off the checkpoint's config with
    this rank's mapping, the rank's weights loaded through its own ``load_weights``."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype

    config = ModelConfig.from_pretrained(
        path, trust_remote_code=True, mapping=mapping, max_num_tokens=MAX_NUM_TOKENS
    )
    validate_and_set_mamba_ssm_cache_dtype(config, "auto")
    with torch.device("cuda"):
        model = _modeling().ModelingV2Nemotron4NanoPreviewSm103Tp4(config)
    model.load_weights(_checkpoint_tensors(path))
    # Collective: builds the TP group's all-reduce workspace.
    model.post_load_weights()
    torch.cuda.synchronize()
    return config, model.eval()


def _build_manager(config, mapping):
    """The engine's hybrid cache manager for this rank (the default V2 manager): paged KV for the
    attention layers, recurrent and conv state slots of the rank's heads for the GDP layers."""
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
        mapping,
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


def _build_metadata(config, manager, mapping):
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
        mapping=mapping,
        cache_indirection=None,
        kv_cache_manager=manager,
    )
    md.mamba_metadata_cls = _modeling().ModelingV2Nemotron4NanoPreviewSm103Tp4.mamba_metadata_cls
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
    exactly and one that ends mid-split -- and fillers enough for a batch of 1024."""
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
    """What one request's decode step produced on this rank: its logits and what it wrote."""

    logits: torch.Tensor  # [vocab] fp32
    ssm: list[torch.Tensor]  # per GDP layer, its slot: [6, 64, 128] fp32
    conv: list[torch.Tensor]  # per GDP layer, its slot: [4224, 3] bf16
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
    """The rank's loaded target, its cache manager and metadata, the prefilled requests, and the
    state every decode run starts from."""

    def __init__(self, checkpoint: str, rank: int, log):
        import _nemotron4_prompts as P

        self.log = log
        self.mapping = _mapping(rank)
        tokenizer = P.load_tokenizer(checkpoint)
        self.config, self.model = _build_target(checkpoint, self.mapping)
        self.manager = _build_manager(self.config, self.mapping)
        self.md = _build_metadata(self.config, self.manager, self.mapping)
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
        self._pages: dict[tuple[int, int], list[int]] = {}
        self.snapshot = [
            (self.manager.get_ssm_states(i).clone(), self.manager.get_conv_states(i).clone())
            for i in self.gdp_layers
        ]
        self.static_ids = torch.zeros(max(GRAPH_SIZES), dtype=torch.int, device="cuda")
        self.graphs: dict[int, tuple] = {}
        self.graph_pool = torch.cuda.graph_pool_handle()
        # The same seed on every rank: the ranks feed the padding rows the same tokens, as the
        # engine's ranks feed every row the same token.
        self.padding_tokens = torch.Generator(device="cuda").manual_seed(9)

    def _prefill(self, requests: list[Request]) -> None:
        """Every prompt through the target's prefill steps, whole prompts packed up to
        ``MAX_NUM_TOKENS`` tokens a step; each request's decode token is its greedy next token."""
        batch: list[Request] = []
        steps = 0
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
                steps += 1
            if r is not None:
                batch.append(r)
        torch.cuda.synchronize()
        self.log(f"prefilled {len(requests)} requests in {steps} steps")

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
        if (r.rid, layer) not in self._pages:
            self._pages[r.rid, layer] = self.manager.get_batch_cache_indices(
                [r.rid], layer_idx=layer
            )[0]
        page = self._pages[r.rid, layer][pos // TOKENS_PER_BLOCK]
        return self.kv_buffers[layer][page, :, pos % TOKENS_PER_BLOCK, 0].clone()

    def row(self, r: Request, logits: torch.Tensor) -> Row:
        """What ``r``'s step left on this rank: its logits row, its slot's state in every GDP
        layer, and the K/V appended at its decoded position in every attention layer."""
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


def _check_rows(h: Harness, rows, positions, logits, what: str, reference) -> list[str]:
    failures = []
    for pos in positions:
        r = rows[pos]
        for diff in _differences(
            h.row(r, logits[pos]), reference[r.rid], h.gdp_layers, h.attention_layers
        ):
            failures.append(f"{what}, {r.name} at row {pos}: {diff}")
    return failures


# --- The checks, in order. Each returns {"failures": [...], "log": [...], ...} for this rank. ----


def check_reference_rows_are_real(h: Harness) -> dict:
    """The reference is worth comparing against: every probe's logits are finite, its slot holds
    non-zero state in every GDP layer, and the K/V read at its decoded position through the
    manager's page table is non-zero in every attention layer."""
    failures = []
    for r in h.probes:
        ref = h.reference[r.rid]
        if not bool(torch.isfinite(ref.logits).all()):
            failures.append(f"{r.name}: non-finite logits")
        if not (all(bool(s.any()) for s in ref.ssm) and all(bool(c.any()) for c in ref.conv)):
            failures.append(f"{r.name}: an all-zero recurrent or conv state")
        if not all(bool(kv.any()) for kv in ref.kv):
            failures.append(f"{r.name}: an all-zero appended K/V")
    return {"failures": failures, "log": [f"{len(h.probes)} probes"]}


def check_every_rank_forms_the_same_logits_and_kv(h: Harness) -> dict:
    """Every rank forms each probe's logits over the whole vocabulary from the same hidden state
    (vocab-parallel and gathered for a few rows, the whole lm_head above), and appends the K/V of
    the one KV head every rank keeps: the four ranks' bits must agree."""
    mine = {
        r.rid: (
            _bits(h.reference[r.rid].logits).cpu().numpy().tobytes(),
            _bits(h.reference[r.rid].kv).cpu().numpy().tobytes(),
        )
        for r in h.probes
    }
    gathered = COMM.gather(mine, root=0)
    failures = []
    if RANK == 0:
        for r in h.probes:
            for other, theirs in enumerate(gathered):
                if theirs[r.rid][0] != mine[r.rid][0]:
                    failures.append(f"{r.name}: rank {other}'s logits differ from rank 0's")
                if theirs[r.rid][1] != mine[r.rid][1]:
                    failures.append(f"{r.name}: rank {other}'s appended K/V differ from rank 0's")
    return {
        "failures": failures,
        "log": [f"{len(h.probes)} probes compared over {WORLD_SIZE} ranks"] if RANK == 0 else [],
    }


def check_eager_rows_do_not_depend_on_the_batch(h: Harness) -> dict:
    """Every probe inside eager decode batches of every size: the same bits as alone."""
    failures, runs = [], 0
    for size in BATCH_SIZES:
        if size == 1:
            continue  # the reference itself
        for rows, positions in _batches(h, size):
            logits = h.eager(rows)
            failures += _check_rows(h, rows, positions, logits, f"eager B={size}", h.reference)
            runs += 1
    return {"failures": failures, "log": [f"eager: {runs} batches, {len(failures)} differences"]}


def check_graph_rows_equal_the_eager_row(h: Harness) -> dict:
    """Every probe inside padded CUDA-graph decode batches of every size: the same bits as alone
    and eager. The padding rows write no slot: after every replay only the real rows' slots differ
    from the snapshot."""
    failures, runs = [], 0
    for size in BATCH_SIZES:
        for rows, positions in _batches(h, size):
            logits = h.graph(rows)
            failures += _check_rows(
                h, rows, positions, logits, f"graph B={size} in {h.graph_size(size)}", h.reference
            )
            written = h.changed_slots()
            real = {h.slot[r.rid] for r in rows}
            if not written <= real:
                failures.append(
                    f"graph B={size}: a padding row wrote slots {sorted(written - real)} "
                    f"(the padding dummy's slot is {h.dummy_slot})"
                )
            runs += 1
    return {
        "failures": failures,
        "log": [
            f"graph: {runs} batches over sizes {sorted(h.graphs)}, {len(failures)} differences"
        ],
    }


def check_attention_is_within_band_of_thop_attention(h: Harness) -> dict:
    """On a 64-row eager decode step over the engine's own pool, in every attention layer:

    - ``invariant_gen_attention`` appends each row's K and V -- the K and V columns of the qkv
      GEMM's row, bit for bit -- where the manager's own page table puts the row's token;
    - ``thop_attention``'s append of the same tokens leaves those bytes as they were;
    - the two outputs, over the rank's 4 query heads, are within the band of each other.

    Band: each kernel is within 2 element ulp of an fp32 reference (measured: 2.00 for this
    entry, 1.98 for thop's decode kernels), so 4 element ulp between them; relative RMS 0.64 and
    0.77 ulp, so at most 1.41 between them.
    """
    from _validating import ULP, assert_within_ulp

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

    M.invariant_gen_attention = both
    try:
        h.eager(rows)
    finally:
        M.invariant_gen_attention = original
    failures, log = [], []
    if len(calls) != len(h.attention_layers):
        failures.append(f"{len(calls)} attention calls, expected {len(h.attention_layers)}")
    for layer, ours, theirs, expected, appended, after_thop in calls:
        if not torch.equal(_bits(appended), _bits(expected)):
            failures.append(f"layer {layer}: the appended K/V are not the qkv GEMM's K/V columns")
        if not torch.equal(_bits(after_thop), _bits(appended)):
            failures.append(f"layer {layer}: thop_attention appended other bytes")
        diff = (ours.float() - theirs.float()).abs()
        scale = theirs.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
        element = float((diff / scale).max()) / ULP[torch.bfloat16]
        rms = float(diff.pow(2).mean().sqrt() / theirs.float().pow(2).mean().sqrt())
        log.append(
            f"attention layer {layer}: {element:.2f} element ulp, "
            f"{rms / ULP[torch.bfloat16]:.3f} rms ulp from thop_attention"
        )
        try:
            assert_within_ulp(ours, theirs, element_ulp=4.0, rms_ulp=1.41, ulp=ULP[torch.bfloat16])
        except AssertionError as exc:
            failures.append(f"layer {layer}: outside the band: {str(exc).splitlines()[0]}")
    return {"failures": failures, "log": log}


def _rival(h: Harness, name: str):
    """What the decode step ran before it was made invariant, at one of its paths: ``invariant_mm``
    back on ``cublas_mm``, ``invariant_gen_attention`` back on ``thop_attention``, or -- the path
    tp4 adds -- the all-reduce on NCCL, whose sum is formed by the message rather than the
    element (op-allreduce). Returns ``(module attribute, replacement)``."""
    M = _modeling()
    if name == "cublas_mm":

        def replaced(x, weight, out_dtype=None):
            return M.cublas_mm(x, weight.t(), None, out_dtype)

        return "invariant_mm", replaced
    if name == "thop_attention":

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

        return "invariant_gen_attention", replaced
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy

    nccl = AllReduce(mapping=h.mapping, strategy=AllReduceStrategy.NCCL)

    def replaced(input, comm_buffer, buffer_flags):
        return nccl(input)

    return "allreduce", replaced


def check_the_replaced_paths_depend_on_the_batch(h: Harness, name: str) -> dict:
    """The comparison has teeth on this model's own state: with one of the decode step's paths put
    back on what it ran before it was made invariant, some probe's step differs between its batch
    of one and the eager batches above. Each rival's reference is that rival's own batch of one."""
    M = _modeling()
    attribute, replaced = _rival(h, name)
    original = getattr(M, attribute)
    setattr(M, attribute, replaced)
    try:
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
    finally:
        setattr(M, attribute, original)
    return {
        "failures": [],
        "differ": len(differ),
        "compared": compared,
        "log": [
            f"rival {name}: {len(differ)} of {compared} probe rows differ from their batch of one"
        ]
        + [f"  {line}" for line in differ],
    }


CHECKS = (
    ("reference_rows_are_real", check_reference_rows_are_real),
    ("every_rank_forms_the_same_logits_and_kv", check_every_rank_forms_the_same_logits_and_kv),
    ("eager_rows_do_not_depend_on_the_batch", check_eager_rows_do_not_depend_on_the_batch),
    ("graph_rows_equal_the_eager_row", check_graph_rows_equal_the_eager_row),
    (
        "attention_is_within_band_of_thop_attention",
        check_attention_is_within_band_of_thop_attention,
    ),
    ("rival_cublas_mm", lambda h: check_the_replaced_paths_depend_on_the_batch(h, "cublas_mm")),
    (
        "rival_thop_attention",
        lambda h: check_the_replaced_paths_depend_on_the_batch(h, "thop_attention"),
    ),
    (
        "rival_nccl_allreduce",
        lambda h: check_the_replaced_paths_depend_on_the_batch(h, "nccl_allreduce"),
    ),
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
            # The reference: every probe decoding alone, eagerly.
            h.reference = {}
            for r in h.probes:
                logits = h.eager([r])
                h.reference[r.rid] = h.row(r, logits[0])
            for name, check in CHECKS:
                start = time.time()
                result = check(h)
                result["seconds"] = round(time.time() - start, 1)
                results[name] = result
                for line in result["log"]:
                    log(line)
                log(f"{name}: {len(result['failures'])} failures ({result['seconds']}s)")
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
    pid_dir = tempfile.mkdtemp(prefix="tp4_decode_invariance_")
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
    assert "--report" in sys.argv, "usage: _tp4_decode_invariance_ranks.py --report <json>"
    _spawn_ranks(Path(sys.argv[sys.argv.index("--report") + 1]).resolve())
    print("OK", flush=True)
