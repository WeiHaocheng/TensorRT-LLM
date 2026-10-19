# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-identity gates for the nemotron4 modeling_v2 targets.

Greedy generation must not depend on how the engine runs a step: with CUDA
graphs on, a decode step replays a graph captured for a padded batch size;
with them off, the same step runs eagerly at its own size. Each target computes
a decode row from that row alone -- its decode GEMMs and generation attention
are batch-invariant, under tensor parallelism so are its all-reduces, and under
attention DP its routed experts give a token the same bits whatever the other
ranks hold -- so the two legs must emit the same tokens.

Prefill and mixed steps are not batch-invariant, and CUDA graphs never run
them, so they are identical in both legs only while both legs schedule the
same batches. This file therefore controls what reaches the scheduler: each
leg runs the same sequence of ``generate`` calls, and the batches every rank
ran -- how many context and generation requests each iteration ran there, and
their context tokens -- are compared between the legs before any token is.

Every test here passes ``modeling_v2="require"`` to ``LLM``: TensorRT-LLM has
no built-in implementation of this architecture, and ``"require"`` makes a
routing decline raise with its reason instead of finding no model class.
"""

import datetime
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SchedulingParams

from ..conftest import llm_models_root, unittest_path
from .accuracy_core import LlmapiAccuracyTestHarness

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# Setting this to a file path writes every leg's tokens, logprobs and iteration
# compositions there as JSON, for comparing runs offline.
REPORT_ENV = "NEMOTRON4_IDENTITY_REPORT"

# How the executor stamps the time an iteration began (`IterationStats.timestamp`).
_STATS_TIME_FORMAT = "%m-%d-%Y %H:%M:%S.%f"


def _prompt_module():
    """The nemotron4 targets' fixed prompt set, ``_nemotron4_prompts.py`` of the unit tests.

    It imports only ``tokenizers`` and ``pyarrow``, and pins every prompt's token ids
    (``PROMPT_SHA256``), so this file and the unit tests compare the same inputs.
    """
    name = "_nemotron4_prompts"
    if name not in sys.modules:
        path = unittest_path() / "_torch" / "modeling_v2" / "models" / "nemotron4" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves the module's string annotations through sys.modules.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class _Request:
    """One request of a ``generate`` call.

    A name, its prompt ids, its output length and, under attention DP, the rank
    it is pinned to (None lets the engine place it).
    """

    name: str
    token_ids: tuple[int, ...]
    max_tokens: int
    rank: int | None = None


@dataclass
class _Output:
    """What a request emitted: its tokens, and per token the top-2 raw logprobs."""

    token_ids: list[int]
    top2: list[list[tuple[int, float]]]


# One iteration as every rank ran it, in rank order: per rank (context requests,
# context tokens, generation requests), or None when the rank published no row.
_Step = tuple[tuple[int, int, int] | None, ...]


@dataclass
class _Call:
    """One ``generate`` call of a leg."""

    requests: list[_Request]
    outputs: dict[str, _Output]
    # When the call began to submit, on the clock the executor stamps its iterations with.
    start: datetime.datetime
    submit_seconds: float
    # The iteration statistics read once the call had ended; some may be earlier calls'.
    stats: list[dict]
    # The iterations that ran this call's requests (`_attribute_steps`).
    steps: list[_Step] = field(default_factory=list)


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    """The first position where two token lists differ, or None if they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _margin(top2: list[tuple[int, float]]) -> float:
    """The top-1 minus top-2 raw logprob: how close a greedy step was to a tie."""
    return top2[0][1] - top2[1][1] if len(top2) > 1 else float("inf")


def _promotable(step: _Step) -> bool:
    """Whether the runner may turn the iteration's context requests into graph decode rows.

    With CUDA graphs on, it does so when every context request of a rank's step
    is a final one-token chunk -- under attention DP, only when every rank's
    batch can run as a graph. Such a step runs other kernels in the two legs: a
    context row with CUDA graphs off, a generation row with them on.
    """
    contexts = [(c, t) for c, t, _ in filter(None, step) if c]
    return bool(contexts) and all(t == c for c, t in contexts)


def _attribute_steps(calls: dict[str, _Call]) -> None:
    """Give every call of a leg the iterations that ran its requests.

    An iteration's statistics can reach the client during a later call: under
    attention DP, rank 0 publishes the rows of every rank for an iteration only
    at a later rank-state all-gather, one iteration per executor loop, so the
    last iterations of a call wait for the next call's loops. Every row carries
    the time rank 0 began the iteration, on the client's clock (one node). The
    calls are separated by idle pauses, so a call's iterations all begin after
    it began to submit and before the next call did; rows from before the first
    call are the engine's start-up.
    """
    rows: dict[int, dict[int, tuple[int, int, int]]] = {}
    began: dict[int, datetime.datetime] = {}
    for call in calls.values():
        for stat in call.stats:
            batching = stat["inflightBatchingStats"]
            rows.setdefault(stat["iter"], {})[stat.get("attentionDpRank", 0)] = (
                batching["numContextRequests"],
                batching["numCtxTokens"],
                batching["numGenRequests"],
            )
            began[stat["iter"]] = datetime.datetime.strptime(stat["timestamp"], _STATS_TIME_FORMAT)
    num_ranks = 1 + max((rank for ranks in rows.values() for rank in ranks), default=0)
    by_start = sorted(calls.values(), key=lambda call: call.start)
    for it in sorted(rows):
        owner = None
        for call in by_start:
            if call.start <= began[it]:
                owner = call
        if owner is not None:
            owner.steps.append(tuple(rows[it].get(rank) for rank in range(num_ranks)))


def _unaccounted(call: _Call) -> list[str]:
    """Where a call's iterations miss work its requests did, per rank.

    Every prompt token runs in some context chunk of its request's rank, and
    every output token after the first in one generation row there, so the
    iterations attributed to a call must sum to exactly that on every rank.
    """
    expected = Counter()
    for r in call.requests:
        rank = r.rank or 0
        expected[rank, "context tokens"] += len(r.token_ids)
        expected[rank, "generation rows"] += len(call.outputs[r.name].token_ids) - 1
    seen = Counter()
    missing_rows = 0
    for step in call.steps:
        for rank, row in enumerate(step):
            if row is None:
                missing_rows += 1
                continue
            seen[rank, "context tokens"] += row[1]
            seen[rank, "generation rows"] += row[2]
    gaps = [
        f"rank {rank}: {what} {seen[rank, what]}, its requests did {count}"
        for (rank, what), count in sorted(expected.items())
        if seen[rank, what] != count
    ]
    gaps += [
        f"rank {rank}: {what} {count}, but none of its requests ran there"
        for (rank, what), count in sorted(seen.items())
        if count and (rank, what) not in expected
    ]
    if missing_rows:
        gaps.append(f"{missing_rows} rank rows missing from {len(call.steps)} iterations")
    return gaps


def _padded_sizes(calls: dict[str, _Call], graph_sizes: list[int]) -> Counter:
    """How often the decode-only iterations of the calls ran at each padded graph size.

    A decode-only iteration (no rank runs a context) replays the smallest
    captured graph that holds its busiest rank's rows; every rank pads to it.
    No rank holds more rows than the largest graph (MAX_BATCH_SIZE).
    """
    sizes = Counter()
    for call in calls.values():
        for step in call.steps:
            ranks = list(filter(None, step))
            if any(c for c, _, _ in ranks) or not any(g for _, _, g in ranks):
                continue
            rows = max(g for _, _, g in ranks)
            sizes[next(s for s in graph_sizes if s >= rows)] += 1
    return sizes


class _CudaGraphIdentity(LlmapiAccuracyTestHarness):
    """The legs, calls and checks every nemotron4 target's CUDA-graph identity gate shares.

    A target's class states its topology and serve config, and which requests
    each call submits: where each one runs, and in what order they reach the
    scheduler.
    """

    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The target's topology (LLM arguments) and its serve config.
    PARALLEL: dict = {}
    MAX_SEQ_LEN: int
    MAX_BATCH_SIZE: int
    MAX_NUM_TOKENS: int
    CHUNKED_PREFILL: bool
    KV_FRACTION: float
    AVG_SEQ_LEN: int

    # Output tokens per fixed prompt (criterion A), past any end id.
    MAX_TOKENS = 256

    # Calls whose outputs and batches are not compared between the legs: the
    # one-token prompt, which the two legs run through different kernels (a
    # promoted decode row against a context row), and the last call, which only
    # makes the executor publish the statistics of the calls before it.
    UNCOMPARED = ("one_token_fresh", "one_token_recycled", "flush")

    # The executor drains its request queue once per iteration, and when idle it
    # wakes on the first request alone. A pause around every call lets the
    # previous call's last iteration end, so the call's first request runs its
    # prefill alone in both legs, and the rest arrive while it runs (see each
    # target's `_batch_order`).
    SETTLE_SECONDS = 1.0

    def _kv_cache_config(self) -> KvCacheConfig:
        return KvCacheConfig(
            free_gpu_memory_fraction=self.KV_FRACTION, avg_seq_len=self.AVG_SEQ_LEN
        )

    def _cuda_graph_config(self) -> CudaGraphConfig:
        return CudaGraphConfig(enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE)

    def _llm(self, cuda_graphs: bool) -> LLM:
        # Built fresh for each leg; the legs differ in cuda_graph_config only.
        #
        # env_overrides is applied on every rank before the model is built:
        # - the target checks its step contract on its first forward;
        # - Triton autotune results go through the on-disk cache. The prefill's
        #   GDP chunk kernel (the vendored FLA `chunk_fwd_kernel_o`) is
        #   autotuned by timing in each process, and its configs sum the key
        #   dimension in different orders, so two engines could pick configs
        #   whose prefill outputs differ in the last bit. With the cache, the
        #   second engine reads the first one's choice. Printing shows which
        #   leg benchmarked and which read the cache.
        return LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={
                MODELING_V2_VALIDATE_ENV: "1",
                "TRITON_CACHE_AUTOTUNING": "1",
                "TRITON_PRINT_AUTOTUNING": "1",
            },
            trust_remote_code=True,
            max_seq_len=self.MAX_SEQ_LEN,
            max_batch_size=self.MAX_BATCH_SIZE,
            max_num_tokens=self.MAX_NUM_TOKENS,
            enable_chunked_prefill=self.CHUNKED_PREFILL,
            disable_overlap_scheduler=False,
            kv_cache_config=self._kv_cache_config(),
            cuda_graph_config=self._cuda_graph_config() if cuda_graphs else None,
            # Per-iteration batch compositions, compared between the legs.
            enable_iter_perf_stats=True,
            max_stats_len=-1,
            **self.PARALLEL,
        )

    def _generate(self, llm: LLM, requests: list[_Request]) -> _Call:
        """Submit the requests in order, as one call, and collect what each emitted.

        Greedy, past any end id, with the top-2 raw logprobs of every emitted
        token; a request with a rank is pinned there. The statistics available
        once the call has ended are read with it; `_attribute_steps` sorts them
        into calls after the leg.
        """
        time.sleep(self.SETTLE_SECONDS)
        start = datetime.datetime.now()
        began = time.monotonic()
        futures = [
            llm.generate_async(
                {"prompt_token_ids": list(r.token_ids)},
                SamplingParams(
                    max_tokens=r.max_tokens, temperature=0.0, ignore_eos=True, logprobs=2
                ),
                scheduling_params=None
                if r.rank is None
                else SchedulingParams(attention_dp_rank=r.rank, attention_dp_relax=False),
            )
            for r in requests
        ]
        submit_seconds = time.monotonic() - began
        outputs = {}
        for r, future in zip(requests, futures):
            out = future.result().outputs[0]
            top2 = [
                sorted(((t, lp.logprob) for t, lp in pos.items()), key=lambda x: -x[1])
                for pos in out.logprobs
            ]
            outputs[r.name] = _Output(list(out.token_ids), top2)
        time.sleep(self.SETTLE_SECONDS)
        return _Call(requests, outputs, start, submit_seconds, llm.get_stats(timeout=2))

    def _leg(self, cuda_graphs: bool, prompts) -> dict[str, _Call]:
        """The same sequence of calls in each leg.

        1. The one-token prompt, in the fresh engine.
        2. The fixed prompts as one batch.
        3. Each fixed prompt alone.
        4. The fixed prompts with the fillers.
        5. The saturation calls, after which no state slot is clean.
        6. The one-token prompt again, on a slot a finished request left. With
           CUDA graphs on, the runner promotes a final one-token context into a
           graph decode row, whose recurrent and conv state the decode entries
           read, so the target must zero that slot first.
        7. A short call that makes the executor publish the last statistics.
        """
        batch = self._batch_order(prompts)
        one_token = self._one_token(prompts)
        calls = {}
        with self._llm(cuda_graphs) as llm:
            calls["one_token_fresh"] = self._generate(llm, one_token)
            calls["batch"] = self._generate(llm, batch)
            for r in batch:
                calls[f"alone/{r.name}"] = self._generate(llm, [r])
            calls["batch_fillers"] = self._generate(llm, batch + self._fillers(prompts))
            for name, requests in self._saturations(prompts).items():
                calls[name] = self._generate(llm, requests)
            calls["one_token_recycled"] = self._generate(llm, one_token)
            calls["flush"] = self._generate(llm, self._flush(prompts))
        _attribute_steps(calls)
        return calls

    # What each target's class says: the requests of its calls, and why none of
    # its compared prompts ends in a one-token context chunk.

    def _one_token(self, prompts) -> list[_Request]:
        raise NotImplementedError

    def _batch_order(self, prompts) -> list[_Request]:
        raise NotImplementedError

    def _fillers(self, prompts) -> list[_Request]:
        raise NotImplementedError

    def _saturations(self, prompts) -> dict[str, list[_Request]]:
        raise NotImplementedError

    def _flush(self, prompts) -> list[_Request]:
        raise NotImplementedError

    def _check_chunks(self, prompts) -> None:
        raise NotImplementedError

    def _check_identity(self, monkeypatch) -> None:
        """Run both legs and gate: G7, equal batches, then equal tokens."""
        monkeypatch.setenv("LLM_MODELS_ROOT", llm_models_root())
        module = _prompt_module()
        prompts = module.prompt_set(module.load_tokenizer(self.MODEL_PATH))
        self._check_chunks(prompts)

        legs = {
            "cuda_graph": self._leg(True, prompts),
            "eager": self._leg(False, prompts),
        }
        on, off = legs["cuda_graph"], legs["eager"]
        self._report(legs)

        # The statistics must account for every call's work, rank by rank, or
        # the batches compared below would not be the ones that ran.
        for leg_name, leg in legs.items():
            for name, call in leg.items():
                if name == "flush":
                    continue
                gaps = _unaccounted(call)
                assert not gaps, (
                    f"{leg_name} {name}: the iteration statistics do not account for the "
                    f"call's work: {gaps}"
                )

        # G7: the recycled slot must hold nothing of its previous owner.
        for leg_name, leg in legs.items():
            for req, fresh in leg["one_token_fresh"].outputs.items():
                recycled = leg["one_token_recycled"].outputs[req]
                assert recycled.token_ids == fresh.token_ids and recycled.top2 == fresh.top2, (
                    f"{leg_name} {req}: the one-token prompt on a recycled slot emitted "
                    f"{recycled.token_ids[:16]}..., in the fresh engine "
                    f"{fresh.token_ids[:16]}...; first difference at "
                    f"{_first_divergence(fresh.token_ids, recycled.token_ids)}"
                )

        # The legs are comparable only where they scheduled the same batches,
        # and where no context ran through other kernels in the two legs.
        for name, call in on.items():
            if name in self.UNCOMPARED:
                continue
            assert call.steps == off[name].steps, (
                f"{name}: the legs ran different batches, so their tokens are not comparable "
                f"(iterations as (context requests, context tokens, generation requests) per "
                f"rank: CUDA graphs {call.steps[:8]}..., eager {off[name].steps[:8]}...). A "
                f"request arrived after the first iteration took the others; the call took "
                f"{call.submit_seconds:.3f} s / {off[name].submit_seconds:.3f} s to submit."
            )
            promoted = [i for i, step in enumerate(call.steps) if _promotable(step)]
            assert not promoted, (
                f"{name}: iterations {promoted} ran only final one-token context chunks, which "
                f"the CUDA-graph leg promotes into decode rows: "
                f"{[call.steps[i] for i in promoted]}"
            )

        partings = []
        for name, call in on.items():
            if name in self.UNCOMPARED:
                continue
            for req, out in call.outputs.items():
                other = off[name].outputs[req]
                pos = _first_divergence(out.token_ids, other.token_ids)
                if pos is None:
                    continue
                at = min(pos, len(out.top2) - 1, len(other.top2) - 1)
                partings.append(
                    f"{name}/{req} at token {pos}: CUDA graphs {out.top2[at]}, eager "
                    f"{other.top2[at]} (eager top-1 margin {_margin(other.top2[at]):.4f})"
                )
        assert not partings, "greedy tokens differ with CUDA graphs on and off:\n" + "\n".join(
            partings
        )

    def _report(self, legs: dict[str, dict[str, _Call]]) -> None:
        """Print what the gate does not assert, and dump the legs when asked.

        - Each fixed prompt in the batch against alone, in each leg: identical,
          or the first parting with the alone run's top-2 there. A parting where
          the batch's token is the alone run's runner-up, at a small margin, is
          a near-tie that the batch's prefill / mixed-step numerics tipped.
        - The batch-shape noise: the largest change of the emitted token's
          logprob, batch against alone, over the positions before any parting.
        - The one-token prompt, CUDA graphs against eager: a promoted decode
          row against a context row, which run other kernels.
        - Per call: whether the legs ran the same batches, the most rows a rank
          ran in a decode-only iteration, how many iterations ran only final
          one-token context chunks, and how many requests' logprobs are bitwise
          equal across the legs.
        - The padded graph sizes the compared calls' decode-only iterations ran.
        """
        for leg_name, leg in legs.items():
            batch = leg["batch"]
            noise = 0.0
            for req, out in batch.outputs.items():
                alone = leg[f"alone/{req}"].outputs[req]
                pos = _first_divergence(alone.token_ids, out.token_ids)
                common = len(alone.token_ids) if pos is None else pos
                for i in range(common):
                    noise = max(noise, abs(out.top2[i][0][1] - alone.top2[i][0][1]))
                if pos is None:
                    same_lp = all(out.top2[i] == alone.top2[i] for i in range(common))
                    print(
                        f"[modeling_v2] identity {leg_name} batch vs alone {req}: identical "
                        f"tokens, logprobs {'bitwise equal' if same_lp else 'differ'}"
                    )
                    continue
                runner_up = len(alone.top2[pos]) > 1 and out.token_ids[pos] == alone.top2[pos][1][0]
                print(
                    f"[modeling_v2] identity {leg_name} batch vs alone {req}: parts at token "
                    f"{pos}; alone top-2 {alone.top2[pos]} (margin {_margin(alone.top2[pos]):.4f}), "
                    f"batch emitted {out.token_ids[pos]} "
                    f"({'the alone runner-up' if runner_up else 'NOT the alone runner-up'})"
                )
            print(
                f"[modeling_v2] identity {leg_name} batch-shape noise: max |d logprob| of the "
                f"emitted token before any parting = {noise:.6f}"
            )
        on, off = legs["cuda_graph"], legs["eager"]
        for req, one_on in on["one_token_fresh"].outputs.items():
            one_off = off["one_token_fresh"].outputs[req]
            pos = _first_divergence(one_on.token_ids, one_off.token_ids)
            print(
                f"[modeling_v2] identity {req}, promoted decode row vs context row: "
                + (
                    "identical tokens"
                    if pos is None
                    else f"parts at token {pos} (eager top-2 {one_off.top2[pos]}, CUDA graphs "
                    f"{one_on.top2[pos]})"
                )
                + f"; first-token top-2 {one_on.top2[0]} vs {one_off.top2[0]}"
            )
        for name, call in on.items():
            same = sum(out.top2 == off[name].outputs[req].top2 for req, out in call.outputs.items())
            decode_rows = [
                max((row[2] for row in rows if row), default=0)
                for rows in zip(
                    *(step for step in call.steps if not any(row and row[0] for row in step))
                )
            ]
            promoted = sum(_promotable(step) for step in call.steps)
            print(
                f"[modeling_v2] identity {name}: {len(call.steps)} iterations (batches "
                f"{'equal' if call.steps == off[name].steps else 'differ'} across the legs, at "
                f"most {decode_rows} rows per rank in a decode-only one, {promoted} with only "
                f"final one-token contexts), submitted in {call.submit_seconds:.3f} s / "
                f"{off[name].submit_seconds:.3f} s; logprobs bitwise equal across the legs for "
                f"{same}/{len(call.outputs)} requests"
            )
        graph_sizes = self._cuda_graph_config().batch_sizes
        compared = {name: call for name, call in on.items() if name not in self.UNCOMPARED}
        replayed = _padded_sizes(compared, graph_sizes)
        print(
            f"[modeling_v2] identity graph sizes run by the compared calls' decode-only "
            f"iterations (iterations per size): {dict(sorted(replayed.items()))}; "
            f"captured but not run: {[s for s in graph_sizes if s not in replayed]}"
        )
        path = os.environ.get(REPORT_ENV)
        if path:
            with open(path, "w") as f:
                json.dump(
                    {
                        leg_name: {
                            name: dict(
                                steps=call.steps,
                                start=call.start.isoformat(),
                                submit_seconds=call.submit_seconds,
                                ranks={r.name: r.rank for r in call.requests},
                                outputs={
                                    req: dict(token_ids=out.token_ids, top2=out.top2)
                                    for req, out in call.outputs.items()
                                },
                            )
                            for name, call in leg.items()
                        }
                        for leg_name, leg in legs.items()
                    },
                    f,
                )
            print(f"[modeling_v2] identity report written to {path}")


class TestModelingV2Nemotron4NanoPreviewSm103Tp1Identity(_CudaGraphIdentity):
    """nemotron4-nano-preview (text model) / sm_103 / tp1: CUDA graphs on against off."""

    # The serve config the tp1 Pareto sweep measured (WS/bench/tp1_mtp0): an
    # explicit max_seq_len (the config's max_position_embeddings is 1,048,576),
    # 896 requests with a CUDA graph for every padded size up to 896, unchunked
    # prefill in 9088-token steps, and the state pool sized for 9.2K-token
    # requests (avg_seq_len). The CUDA-graph leg therefore carries every graph
    # the benchmark replays.
    MAX_SEQ_LEN = 9344
    MAX_BATCH_SIZE = 896
    MAX_NUM_TOKENS = 9088
    CHUNKED_PREFILL = False
    KV_FRACTION = 0.9
    AVG_SEQ_LEN = 9216

    # Short requests beside the fixed prompts in a second batch, finishing at
    # staggered lengths, so that one call's decode batch shrinks from 64 rows
    # through most padded graph sizes below it.
    NUM_FILLERS = 54

    def _one_token(self, prompts) -> list[_Request]:
        return [_Request("one_token", prompts[0].token_ids[:1], self.MAX_TOKENS)]

    def _batch_order(self, prompts) -> list[_Request]:
        """The fixed prompts as one batch, the first-submitted being a long one.

        The executor wakes on the first request alone, so that request's prefill
        runs alone in both legs. An ~8K-token prefill holds the executor for
        ~150 ms (the GDP chunk entry syncs with the host in every GDP layer),
        long enough for every other request of the call to arrive before the
        next iteration takes them all.
        """
        first = next(p for p in prompts if p.long)
        ordered = [first] + [p for p in prompts if p is not first]
        return [_Request(p.name, p.token_ids, self.MAX_TOKENS) for p in ordered]

    def _fillers(self, prompts) -> list[_Request]:
        """Short requests cut from a long prompt's ids: 8-64 tokens, 8-256 output tokens."""
        source = next(p for p in prompts if p.name == "news_digest").token_ids
        fillers = []
        for i in range(self.NUM_FILLERS):
            length = 8 + (5 * i) % 57
            start = 97 * i
            fillers.append(
                _Request(f"filler{i}", source[start : start + length], 8 + (37 * i) % 249)
            )
        return fillers

    def _saturations(self, prompts) -> dict[str, list[_Request]]:
        """A long prompt, then MAX_BATCH_SIZE - 1 short requests: one call holding every slot.

        All of them run at once, so afterwards each request slot of the state
        pool holds the recurrent and conv state of a finished request. As in
        `_batch_order`, the long prompt's prefill holds the executor while the
        rest arrive, and their prompts (4-16 tokens) fit the next iteration
        together. Their output lengths are spread over 16-143 tokens, then ten
        one apart, so the decode batch shrinks from MAX_BATCH_SIZE rows through
        every padded graph size down to one row.
        """
        long_prompt = next(p for p in prompts if p.name == "news_digest")
        source = next(p for p in prompts if p.name == "gpqa_explain").token_ids
        num_short = self.MAX_BATCH_SIZE - 1
        lengths = [16 + i % 128 for i in range(num_short - 10)] + [150 + i for i in range(10)]
        return {
            "saturation": [_Request("saturation_long", long_prompt.token_ids, 16)]
            + [
                _Request(f"saturation{i}", source[7 * i : 7 * i + 4 + i % 13], lengths[i])
                for i in range(num_short)
            ]
        }

    def _flush(self, prompts) -> list[_Request]:
        return [_Request("flush", prompts[0].token_ids, 8)]

    def _check_chunks(self, prompts) -> None:
        # Unchunked, a prompt is one context chunk of its full length. A final
        # one-token chunk would run as a context row with CUDA graphs off and as
        # a promoted graph decode row with them on -- other kernels -- so none of
        # the fixed prompts may end in one.
        for p in prompts:
            print(f"[modeling_v2] identity prompt {p.name}: {len(p.token_ids)} tokens, one chunk")
            assert 1 < len(p.token_ids) <= self.MAX_NUM_TOKENS, (
                f"{p.name} ({len(p.token_ids)} tokens) does not run as one context chunk "
                f"of more than one token at max_num_tokens {self.MAX_NUM_TOKENS}"
            )

    @skip_not_sm103
    def test_cuda_graph_identity(self, monkeypatch):
        """Greedy tokens with CUDA graphs on equal those with CUDA graphs off.

        On the fixed prompt set (6 short prompts and 4 of ~8K tokens, 256
        tokens each) as one batch, each alone, and beside 54 shorter requests;
        on MAX_BATCH_SIZE requests at once, whose decode batch passes through
        every padded graph size; and a fresh one-token prompt on a state slot a
        finished request left equals the same prompt in the fresh engine.

        Each prompt's output in the batch is also compared with its output alone
        and reported, not gated: prefill and mixed steps are not batch-invariant.
        """
        self._check_identity(monkeypatch)


class TestModelingV2Nemotron4NanoPreviewSm103Dep4Identity(_CudaGraphIdentity):
    """nemotron4-nano-preview (text model) / sm_103 / dep4: CUDA graphs on against off.

    Attention DP over the 4 GPUs of one node: every rank runs its own requests,
    and the ranks meet in every routed MoE block. Under CUDA graphs a decode
    step runs as a graph only when every rank's batch can, and every rank pads
    to the busiest rank's graph size; an idle rank runs the attention-DP dummy.
    Every request is pinned to a rank, so each leg places the same requests on
    the same ranks.
    """

    # The parallel topology the dep4 target implements: attention DP over the
    # 4 GPUs of one node, the routed experts split four ways.
    PARALLEL = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)
    RANKS = 4

    # dep4's GSM8K gate's serve config: an explicit max_seq_len, 512 requests per
    # rank with a CUDA graph for every padded size up to 512, and every step at
    # most 2048 tokens a rank, inside the routed MoE's certified all-gather and
    # reduce-scatter cells -- so the ~8K prompts prefill in 2048-token chunks.
    # The state pool is sized for 9.2K-token requests (avg_seq_len), as the 8K/1K
    # benchmark's is.
    MAX_SEQ_LEN = 9344
    MAX_BATCH_SIZE = 512
    MAX_NUM_TOKENS = 2048
    CHUNKED_PREFILL = True
    KV_FRACTION = 0.9
    AVG_SEQ_LEN = 9216

    # Requests per rank when the fixed prompts run with the fillers: every rank's
    # decode batch then shrinks from 64 rows through the padded graph sizes below.
    BATCH_ROWS_PER_RANK = 64

    def _ranks(self, prompts) -> dict[str, int]:
        """Where each fixed prompt runs: one long prompt per rank, the short ones dealt round-robin."""
        longs = [p.name for p in prompts if p.long]
        shorts = [p.name for p in prompts if not p.long]
        assert len(longs) == self.RANKS, f"{len(longs)} long prompts for {self.RANKS} ranks"
        ranks = {name: rank for rank, name in enumerate(longs)}
        ranks.update({name: i % self.RANKS for i, name in enumerate(shorts)})
        return ranks

    def _one_token(self, prompts) -> list[_Request]:
        token = prompts[0].token_ids[:1]
        return [
            _Request(f"one_token/rank{rank}", token, self.MAX_TOKENS, rank)
            for rank in range(self.RANKS)
        ]

    def _batch_order(self, prompts) -> list[_Request]:
        """The fixed prompts as one batch: the long ones first, one per rank, then the short ones.

        Each rank's batches are then the same whatever the arrival times. The
        idle executor wakes on the first request alone, and the others arrive
        while its first chunk runs. A rank's long prompt (more than three times
        MAX_NUM_TOKENS) then takes the rank's whole token budget for its first
        three chunks, and the scheduler, first come first served, holds the
        rank's other requests behind it until its last chunk -- by when every
        request of the call has arrived.
        """
        ranks = self._ranks(prompts)
        ordered = [p for p in prompts if p.long] + [p for p in prompts if not p.long]
        return [_Request(p.name, p.token_ids, self.MAX_TOKENS, ranks[p.name]) for p in ordered]

    def _fillers(self, prompts) -> list[_Request]:
        """Short requests dealt round-robin until every rank holds BATCH_ROWS_PER_RANK requests.

        Their prompts (8-32 tokens, cut from a long prompt's ids) fit one KV
        block, and the scheduler starts a context only with a block of token
        budget left, so they are never split into chunks; their output lengths
        (8-256 tokens) are staggered.
        """
        held = Counter(self._ranks(prompts).values())
        source = next(p for p in prompts if p.name == "news_digest").token_ids
        fillers = []
        while any(held[rank] < self.BATCH_ROWS_PER_RANK for rank in range(self.RANKS)):
            for rank in range(self.RANKS):
                if held[rank] == self.BATCH_ROWS_PER_RANK:
                    continue
                i = len(fillers)
                length = 8 + (7 * i) % 25
                start = (97 * i) % (len(source) - length)
                fillers.append(
                    _Request(f"filler{i}", source[start : start + length], 8 + (37 * i) % 249, rank)
                )
                held[rank] += 1
        return fillers

    def _saturations(self, prompts) -> dict[str, list[_Request]]:
        """For each rank in turn, MAX_BATCH_SIZE requests there at once.

        A call holds every request slot of its rank and leaves the recurrent and
        conv state of finished requests in the state slots it used, which the
        cache manager hands to later requests on the rank. As in
        `_batch_order`, the rank's long prompt goes first and holds the rank's
        token budget while the rest arrive; their prompts (4-16 tokens) are
        never chunked, and their output lengths (16-143 tokens, then ten one
        apart) shrink the rank's decode batch from MAX_BATCH_SIZE rows through
        every padded graph size down to one row. The other ranks run the
        attention-DP dummy, padded to the same graph.
        """
        ranks = self._ranks(prompts)
        source = next(p for p in prompts if p.name == "gpqa_explain").token_ids
        num_short = self.MAX_BATCH_SIZE - 1
        lengths = [16 + i % 128 for i in range(num_short - 10)] + [150 + i for i in range(10)]
        return {
            f"saturation/rank{ranks[p.name]}": [
                _Request("saturation_long", p.token_ids, 16, ranks[p.name])
            ]
            + [
                _Request(
                    f"saturation{i}",
                    source[7 * i : 7 * i + 4 + i % 13],
                    lengths[i],
                    ranks[p.name],
                )
                for i in range(num_short)
            ]
            for p in prompts
            if p.long
        }

    def _flush(self, prompts) -> list[_Request]:
        return [_Request("flush", prompts[0].token_ids, 8, 0)]

    def _check_chunks(self, prompts) -> None:
        # Chunked prefill cuts a prompt into chunks of whole KV blocks but the
        # last, so a final one-token chunk needs a length of one more than a
        # whole number of blocks. Such a chunk would run as a context row with
        # CUDA graphs off and as a promoted graph decode row with them on --
        # other kernels -- so none of the fixed prompts may have that length. The
        # gate also checks every compared iteration (`_promotable`).
        block = self._kv_cache_config().tokens_per_block
        for p in prompts:
            print(
                f"[modeling_v2] identity prompt {p.name}: {len(p.token_ids)} tokens on rank "
                f"{self._ranks(prompts)[p.name]}, last chunk {len(p.token_ids) % block} (mod "
                f"{block}) tokens"
            )
            assert len(p.token_ids) > 1 and len(p.token_ids) % block != 1, (
                f"{p.name} ({len(p.token_ids)} tokens) can end in a one-token context chunk "
                f"with {block}-token KV blocks"
            )

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_cuda_graph_identity(self, monkeypatch):
        """Greedy tokens with CUDA graphs on equal those with CUDA graphs off, on 4 ranks.

        On the fixed prompt set (6 short prompts and 4 of ~8K tokens, 256
        tokens each) spread over the ranks as one batch, each alone on its rank
        while the others idle, and beside short requests up to 64 a rank; on
        MAX_BATCH_SIZE requests on one rank at a time, whose decode batch passes
        through every padded graph size; and a fresh one-token prompt on every
        rank, on a state slot a finished request left, equals the same prompt in
        the fresh engine.

        Each prompt's output in the batch is also compared with its output alone
        and reported, not gated: prefill and mixed steps are not batch-invariant.
        """
        self._check_identity(monkeypatch)


class TestModelingV2Nemotron4NanoPreviewSm103Tp4Identity(
    TestModelingV2Nemotron4NanoPreviewSm103Tp1Identity
):
    """nemotron4-nano-preview (text model) / sm_103 / tp4: CUDA graphs on against off.

    Tensor parallel over the 4 GPUs of one node, without attention DP: every
    rank runs every request, in the batches rank 0's executor schedules, and
    the ranks meet in every block's all-reduce. A rank's decode step computes
    each row from that row alone, all-reduces included, so tp1's calls, batch
    check and reports carry over; only the saturation call changes.
    """

    PARALLEL = dict(tensor_parallel_size=4, moe_tensor_parallel_size=4)

    # The 8K/1K serve config, by the rule tp1's follows: an explicit max_seq_len,
    # 1024 requests with a CUDA graph for every padded size up to 1024 (the
    # largest batch tp4's decode-invariance test runs), unchunked prefill in
    # steps of 8192 tokens plus one row per request, and the state pool sized
    # for 9.2K-token requests (avg_seq_len).
    MAX_SEQ_LEN = 9344
    MAX_BATCH_SIZE = 1024
    MAX_NUM_TOKENS = 8192 + 1024
    KV_FRACTION = 0.9
    AVG_SEQ_LEN = 9216

    def _saturations(self, prompts) -> dict[str, list[_Request]]:
        """Two long prompts, then MAX_BATCH_SIZE - 2 short requests: one call holding every slot.

        tp1's saturation call with a second long prompt. A tp4 rank's ~8K-token
        prefill holds the executor for less time than tp1's, little more than a
        thousand requests take to arrive behind the first. The second long prompt is
        MAX_NUM_TOKENS - 3 tokens: beside the first one's decode row it leaves no
        room for a short request (4-12 tokens), and the scheduler, first come
        first served, stops at the first context that does not fit. So the short
        requests all wait for the call's third iteration, two prefills after it
        began, and run in it together. Their output lengths are tp1's: the
        decode batch shrinks from MAX_BATCH_SIZE rows through every padded graph
        size down to one row.
        """
        first = next(p for p in prompts if p.name == "news_digest").token_ids
        source = next(p for p in prompts if p.name == "gpqa_explain").token_ids
        second = (source + first)[: self.MAX_NUM_TOKENS - 3]
        num_short = self.MAX_BATCH_SIZE - 2
        lengths = [16 + i % 128 for i in range(num_short - 10)] + [150 + i for i in range(10)]
        shorts = [
            _Request(f"saturation{i}", source[7 * i : 7 * i + 4 + i % 9], lengths[i])
            for i in range(num_short)
        ]
        assert len(second) == self.MAX_NUM_TOKENS - 3 <= self.MAX_SEQ_LEN - 16, (
            "the second long prompt does not fill one iteration's token budget"
        )
        assert sum(len(r.token_ids) for r in shorts) + 2 <= self.MAX_NUM_TOKENS, (
            "the short requests do not fit one iteration beside the two decode rows"
        )
        return {
            "saturation": [
                _Request("saturation_long", first, 16),
                _Request("saturation_long2", second, 16),
            ]
            + shorts
        }

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_cuda_graph_identity(self, monkeypatch):
        """Greedy tokens with CUDA graphs on equal those with CUDA graphs off, on 4 ranks.

        tp1's calls (see its test), with MAX_BATCH_SIZE requests at once
        passing through all 33 padded graph sizes up to 1024.
        """
        super().test_cuda_graph_identity(monkeypatch)
