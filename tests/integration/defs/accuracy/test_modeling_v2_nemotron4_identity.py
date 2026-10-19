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
with them off, the same step runs eagerly at its own size. The tp1 and tp4
targets compute a decode row from that row alone (their decode GEMMs and
generation attention are batch-invariant, and so are tp4's all-reduces), so
the two legs must emit the same tokens.

Prefill and mixed steps are not batch-invariant, and CUDA graphs never run
them, so they are identical in both legs only while both legs schedule the
same batches. This file therefore controls what reaches the scheduler: each
leg runs the same sequence of ``generate`` calls, and a call's composition --
how many context and generation requests each iteration ran, and their context
tokens -- is compared between the legs before any token is.

Every test here passes ``modeling_v2="require"`` to ``LLM``: TensorRT-LLM has
no built-in implementation of this architecture, and ``"require"`` makes a
routing decline raise with its reason instead of finding no model class.
"""

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

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
    """One request of a ``generate`` call: a name, its prompt ids and its output length."""

    name: str
    token_ids: tuple[int, ...]
    max_tokens: int


@dataclass
class _Output:
    """What a request emitted: its tokens, and per token the top-2 raw logprobs."""

    token_ids: list[int]
    top2: list[list[tuple[int, float]]]


@dataclass
class _Call:
    """One ``generate`` call of a leg."""

    outputs: dict[str, _Output]
    # Per iteration: (context requests, context tokens, generation requests).
    steps: list[tuple[int, int, int]]
    submit_seconds: float


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    """The first position where two token lists differ, or None if they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _margin(top2: list[tuple[int, float]]) -> float:
    """The top-1 minus top-2 raw logprob: how close a greedy step was to a tie."""
    return top2[0][1] - top2[1][1] if len(top2) > 1 else float("inf")


class TestModelingV2Nemotron4NanoPreviewSm103Tp1Identity(LlmapiAccuracyTestHarness):
    """nemotron4-nano-preview (text model) / sm_103 / tp1: CUDA graphs on against off."""

    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The target's parallel topology, as LLM arguments: one GPU.
    PARALLEL: dict = {}

    # The serve config the tp1 Pareto sweep measured (WS/bench/tp1_mtp0): an
    # explicit max_seq_len (the config's max_position_embeddings is 1,048,576),
    # 896 requests with a CUDA graph for every padded size up to 896, unchunked
    # prefill in 9088-token steps, and the state pool sized for 9.2K-token
    # requests (avg_seq_len). The CUDA-graph leg therefore carries every graph
    # the benchmark replays.
    MAX_SEQ_LEN = 9344
    MAX_BATCH_SIZE = 896
    MAX_NUM_TOKENS = 9088
    KV_FRACTION = 0.9
    AVG_SEQ_LEN = 9216

    # Output tokens per fixed prompt (criterion A), past any end id.
    MAX_TOKENS = 256

    # Short requests beside the fixed prompts in a second batch, finishing at
    # staggered lengths, so that one call's decode batch shrinks from 64 rows
    # through most padded graph sizes below it.
    NUM_FILLERS = 54

    # Calls whose outputs are not compared between the legs: the one-token
    # prompt, which the two legs run through different kernels (a promoted
    # decode row against a context row).
    UNCOMPARED = ("one_token_fresh", "one_token_recycled")

    # The executor drains its request queue once per iteration, and when idle it
    # wakes on the first request alone. A pause around every call lets the
    # previous call's last iteration end, so the call's first request runs its
    # prefill alone in both legs, and the rest arrive while it runs (see
    # `_batch_order`).
    SETTLE_SECONDS = 1.0

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
            enable_chunked_prefill=False,
            disable_overlap_scheduler=False,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=self.KV_FRACTION, avg_seq_len=self.AVG_SEQ_LEN
            ),
            cuda_graph_config=(
                CudaGraphConfig(enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE)
                if cuda_graphs
                else None
            ),
            # Per-iteration batch compositions, compared between the legs.
            enable_iter_perf_stats=True,
            max_stats_len=-1,
            **self.PARALLEL,
        )

    def _generate(self, llm: LLM, requests: list[_Request]) -> _Call:
        """Submit the requests in order, as one call, and collect what each emitted.

        Greedy, past any end id, with the top-2 raw logprobs of every emitted
        token. The call's iteration statistics are read once its last iteration
        has ended: a statistic can reach the queue after the last response.
        """
        time.sleep(self.SETTLE_SECONDS)
        start = time.monotonic()
        futures = [
            llm.generate_async(
                {"prompt_token_ids": list(r.token_ids)},
                SamplingParams(
                    max_tokens=r.max_tokens, temperature=0.0, ignore_eos=True, logprobs=2
                ),
            )
            for r in requests
        ]
        submit_seconds = time.monotonic() - start
        outputs = {}
        for r, future in zip(requests, futures):
            out = future.result().outputs[0]
            top2 = [
                sorted(((t, lp.logprob) for t, lp in pos.items()), key=lambda x: -x[1])
                for pos in out.logprobs
            ]
            outputs[r.name] = _Output(list(out.token_ids), top2)
        time.sleep(self.SETTLE_SECONDS)
        steps = []
        for stat in llm.get_stats(timeout=2):
            batching = stat["inflightBatchingStats"]
            steps.append(
                (
                    batching["numContextRequests"],
                    batching["numCtxTokens"],
                    batching["numGenRequests"],
                )
            )
        return _Call(outputs, steps, submit_seconds)

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

    def _saturation(self, prompts) -> list[_Request]:
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
        return [_Request("saturation_long", long_prompt.token_ids, 16)] + [
            _Request(f"saturation{i}", source[7 * i : 7 * i + 4 + i % 13], lengths[i])
            for i in range(num_short)
        ]

    def _leg(self, cuda_graphs: bool, prompts, one_token: _Request) -> dict[str, _Call]:
        """The same sequence of calls in each leg.

        1. The one-token prompt alone, in the fresh engine.
        2. The fixed prompts as one batch.
        3. Each fixed prompt alone.
        4. The fixed prompts with the fillers.
        5. The saturation call, after which no state slot is clean.
        6. The one-token prompt alone again, on a slot a finished request left.
           With CUDA graphs on, the runner promotes a final one-token context
           into a graph decode row, whose recurrent and conv state the decode
           entries read, so the target must zero that slot first.
        """
        batch = self._batch_order(prompts)
        calls = {}
        with self._llm(cuda_graphs) as llm:
            calls["one_token_fresh"] = self._generate(llm, [one_token])
            calls["batch"] = self._generate(llm, batch)
            for r in batch:
                calls[f"alone/{r.name}"] = self._generate(llm, [r])
            calls["batch_fillers"] = self._generate(llm, batch + self._fillers(prompts))
            calls["saturation"] = self._generate(llm, self._saturation(prompts))
            calls["one_token_recycled"] = self._generate(llm, [one_token])
        return calls

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
        monkeypatch.setenv("LLM_MODELS_ROOT", llm_models_root())
        module = _prompt_module()
        prompts = module.prompt_set(module.load_tokenizer(self.MODEL_PATH))
        one_token = _Request("one_token", prompts[0].token_ids[:1], self.MAX_TOKENS)

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

        legs = {
            "cuda_graph": self._leg(True, prompts, one_token),
            "eager": self._leg(False, prompts, one_token),
        }
        on, off = legs["cuda_graph"], legs["eager"]
        self._report(legs)

        # G7: the recycled slot must hold nothing of its previous owner.
        for leg_name, leg in legs.items():
            fresh = leg["one_token_fresh"].outputs["one_token"]
            recycled = leg["one_token_recycled"].outputs["one_token"]
            assert recycled.token_ids == fresh.token_ids and recycled.top2 == fresh.top2, (
                f"{leg_name}: the one-token prompt on a recycled slot emitted "
                f"{recycled.token_ids[:16]}..., in the fresh engine {fresh.token_ids[:16]}...; "
                f"first difference at {_first_divergence(fresh.token_ids, recycled.token_ids)}"
            )

        # The legs are comparable only where they scheduled the same batches.
        for name, call in on.items():
            if name in self.UNCOMPARED:
                continue
            assert call.steps == off[name].steps, (
                f"{name}: the legs ran different batches, so their tokens are not comparable "
                f"(iterations as (context requests, context tokens, generation requests): CUDA "
                f"graphs {call.steps[:8]}..., eager {off[name].steps[:8]}...). A request arrived "
                f"after the first iteration took the others; the call took "
                f"{call.submit_seconds:.3f} s / {off[name].submit_seconds:.3f} s to submit."
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
        - Per call: whether the legs ran the same batches, the largest
          decode-only batch, and how many requests' logprobs are bitwise equal
          across the legs.
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
        one_on = on["one_token_fresh"].outputs["one_token"]
        one_off = off["one_token_fresh"].outputs["one_token"]
        pos = _first_divergence(one_on.token_ids, one_off.token_ids)
        print(
            "[modeling_v2] identity one-token prompt, promoted decode row vs context row: "
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
            decode_rows = max((g for c, _, g in call.steps if c == 0), default=0)
            print(
                f"[modeling_v2] identity {name}: {len(call.steps)} iterations (batches "
                f"{'equal' if call.steps == off[name].steps else 'differ'} across the legs, at "
                f"most {decode_rows} rows in a decode-only one), submitted in "
                f"{call.submit_seconds:.3f} s / {off[name].submit_seconds:.3f} s; logprobs "
                f"bitwise equal across the legs for {same}/{len(call.outputs)} requests"
            )
        path = os.environ.get(REPORT_ENV)
        if path:
            with open(path, "w") as f:
                json.dump(
                    {
                        leg_name: {
                            name: dict(
                                steps=call.steps,
                                submit_seconds=call.submit_seconds,
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

    def _saturation(self, prompts) -> list[_Request]:
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
        return [
            _Request("saturation_long", first, 16),
            _Request("saturation_long2", second, 16),
        ] + shorts

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_cuda_graph_identity(self, monkeypatch):
        """Greedy tokens with CUDA graphs on equal those with CUDA graphs off, on 4 ranks.

        tp1's calls (see its test), with MAX_BATCH_SIZE requests at once
        passing through all 33 padded graph sizes up to 1024.
        """
        super().test_cuda_graph_identity(monkeypatch)
