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
"""The MTP token-identity harness of the nemotron4 modeling_v2 targets.

Each target's gate is a subclass of ``_MtpIdentity`` in
test_modeling_v2_nemotron4.py, beside the target's other gates; this module
holds the legs, the gate and the batched report they share.

Greedy one-model MTP-Eagle accepts a draft token only where it equals the
token the verify step itself picks, so with MTP on the engine emits the greedy
text of its verify steps, and the recurrent and conv state a verify leaves
behind is the state after its last accepted token: the engine's commit promotes
the per-position GDP and conv state of that token, which rolls back the
rejected drafts. A verify runs the decode step's row-invariant GEMMs and GDP and
conv entries over 1 + D tokens per request, but attends through
``thop_attention``'s generation kernels where a decode step attends through
``invariant_gen_attention``. The two sum the softmax in other orders, so a
verify token's logits sit near the MTP-off decode step's rather than on them,
and MTP on can part from MTP off where MTP off's choice is nearly tied.

So the gate is the near-tie rule. Each prompt is compared up to its first
parting. A parting passes when the token MTP on emitted there is within the
target's band (``NEAR_TIE_BAND``) of MTP off's top-1 -- the runner-up at a
top-1 / top-2 margin within the band, or a lower rank where several
candidates are that close. The band is the verify's measured deviation from
decode, stated in each target's class with its derivation. A parting farther
off fails: a rollback that keeps a rejected draft's state, or a verify that
reads the wrong keys, moves a token wherever it lands. Every parting is
reported with its margin and the tokens on both sides.

Each prompt of the fixed set runs alone (``max_batch_size`` 1). Its prefill is
then the same steps in both legs, and only the speculative path differs: one
verify of 1 + D tokens against up to D + 1 decode steps. The ~8K-token prompts
generate 1K tokens each, a few hundred verify steps that reject drafts and roll
the state back. Under attention DP a prompt is pinned to one rank, the same in
both legs, and it is alone on the engine: the other ranks run the attention-DP
dummy beside it.

A batched run of the same prompts is reported, not gated. In a batch, a
prefill step carries the generation rows of the requests already running, and
with MTP each of them is 1 + D tokens: the step's GEMMs run at another row
count, and a verify row inside such a mixed step runs the prefill path's
kernels. Prefill and mixed steps are not batch-invariant, and MTP can also
change which requests share a step (a verify row takes 1 + D tokens of the
step's budget), so the two legs can part where the MTP-off logits are nearly
tied. Each parting is classified: it is a near-tie when the token MTP on
emitted is within the batch-shape noise measured in the same run (each prompt
alone against the batch), or within the target's verify band, of the MTP-off
top-1 there -- the runner-up, or a lower rank where several candidates are that
close -- or when MTP off itself emits that token over the same tokens with
another prefill/decode split. The MTP-off run then restarts from the MTP output
up to the parting, so every later parting is classified the same way and the
whole output is covered.

Every test here passes ``modeling_v2="require"`` to ``LLM``: TensorRT-LLM has no
built-in implementation of this architecture, and ``"require"`` makes a
routing decline raise with its reason instead of finding no model class.
"""

import contextlib
import datetime
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig, SchedulingParams

from ..conftest import llm_models_root
from ._nemotron4_cuda_graph_identity import _prompt_module
from .accuracy_core import LlmapiAccuracyTestHarness

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# Setting this to a directory writes each test's legs there as JSON (tokens,
# MTP-off top logprobs, iteration compositions, acceptance lengths and the
# classified partings), for comparing runs offline.
REPORT_DIR_ENV = "NEMOTRON4_MTP_IDENTITY_REPORT_DIR"

# How the executor stamps the time an iteration began (`IterationStats.timestamp`).
_STATS_TIME_FORMAT = "%m-%d-%Y %H:%M:%S.%f"


@dataclass(frozen=True)
class _Request:
    """One request: a name, its prompt ids, its output length and its rank.

    Under attention DP the request is pinned to ``rank``; None lets the engine
    place it.
    """

    name: str
    token_ids: tuple[int, ...]
    max_tokens: int
    rank: int | None = None


@dataclass
class _Output:
    """What a request emitted, and the request's own counters.

    ``top`` holds, with MTP off, per emitted token the (token, raw logprob)
    pairs of the highest logprobs, best first; greedy, the first is the emitted
    token. ``decoding_iter`` counts the iterations that gave the request tokens:
    its last context chunk, then each generation step. ``verify_steps`` counts
    the generation steps that verified drafts and ``accepted`` the drafts they
    accepted, both 0 with MTP off.
    """

    token_ids: list[int]
    top: list[list[tuple[int, float]]] | None
    decoding_iter: int = 0
    verify_steps: int = 0
    accepted: int = 0


# One iteration as every rank ran it, in rank order: per rank (context requests,
# context tokens, generation requests), or None when the rank published no row.
_Step = tuple[tuple[int, int, int] | None, ...]


@dataclass
class _Call:
    """One ``generate`` call: its requests, what each emitted, and the iterations that ran them.

    ``start`` is when the call began to submit, on the clock the executor
    stamps its iterations with, and ``stats`` the iteration statistics read
    once the call had ended, some of them earlier calls'. ``steps`` holds the
    iterations `_attribute_steps` gives the call.
    """

    requests: list[_Request]
    outputs: dict[str, _Output]
    start: datetime.datetime
    stats: list[dict]
    steps: list[_Step] = field(default_factory=list)

    @property
    def acceptance_length(self) -> float | None:
        """The engine's estimator over the call's requests, None when no step verified drafts.

        Tokens per verify step, (verify steps + accepted drafts) / verify steps,
        from the requests' own counters. The iteration statistics would count
        only rank 0's requests under attention DP: every rank's row of an
        iteration carries rank 0's speculative-decoding counts.
        """
        steps = sum(o.verify_steps for o in self.outputs.values())
        accepted = sum(o.accepted for o in self.outputs.values())
        return (steps + accepted) / steps if steps else None


@dataclass
class _Engine:
    """An LLM and every ``generate`` call made on it, in order."""

    llm: LLM
    calls: list[_Call] = field(default_factory=list)


@dataclass
class _Parting:
    """Where an MTP-on output parts from the MTP-off greedy run it is compared with.

    ``token`` indexes the MTP-on output. ``off_top`` is the MTP-off run's top
    (token, raw logprob) pairs there, and ``margin`` its top-1 minus top-2
    gap; a raw logprob gap is the logit gap. ``rank`` is where the token MTP on
    emitted ranks among them (2 for the runner-up, None when it is not among
    them) and ``gap`` its logprob below the top-1. ``near_tie`` and
    ``witness`` are set by the batched report's classifier: ``witness`` is a
    number of trailing tokens j such that MTP off, prefilling the prompt and
    the MTP-on output up to j tokens before the parting and decoding the rest,
    itself emits the token MTP on emitted there.
    """

    token: int
    mtp_token: int
    off_top: list[tuple[int, float]]
    margin: float
    rank: int | None
    gap: float | None
    near_tie: bool | None = None
    witness: int | None = None
    reanchor: int = 0
    # The two tokens as text, MTP off's top-1 and MTP on's, when a tokenizer gave them.
    off_text: str | None = None
    mtp_text: str | None = None

    def describe(self) -> str:
        where = (
            "not among them"
            if self.rank is None
            else f"rank {self.rank}, {self.gap:.4f} below the top-1"
        )
        texts = (
            ""
            if self.off_text is None
            else f"; MTP off {self.off_text!r}, MTP on {self.mtp_text!r}"
        )
        return (
            f"token {self.token}: MTP on emitted {self.mtp_token} ({where}); MTP off's top "
            f"{self.off_top[:4]}, top-2 gap {self.margin:.4f}{texts}"
        )


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    """The first position where two token lists differ, or None if they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _show(acceptance_length: float | None) -> str:
    return (
        "none (no step carried drafts)" if acceptance_length is None else f"{acceptance_length:.4f}"
    )


def _margin(top: list[tuple[int, float]]) -> float:
    """The top-1 minus top-2 raw logprob: how close a greedy step was to a tie."""
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def _parting(on: list[int], off: _Output, offset: int = 0) -> _Parting | None:
    """The first parting of ``on[offset:]`` from the MTP-off run ``off``, or None.

    ``off`` is a greedy run of the same prompt followed by ``on[:offset]``, so
    its position i is the MTP-on output's position ``offset + i``.
    """
    rest = on[offset:]
    pos = _first_divergence(off.token_ids, rest)
    if pos is None:
        return None
    assert pos < min(len(rest), len(off.top)), (
        f"one leg emitted fewer tokens: MTP on {len(rest)} and MTP off {len(off.token_ids)} "
        f"from token {offset}, equal up to {pos}"
    )
    top = off.top[pos]
    ranks = {token: i + 1 for i, (token, _) in enumerate(top)}
    rank = ranks.get(rest[pos])
    return _Parting(
        token=offset + pos,
        mtp_token=rest[pos],
        off_top=top,
        margin=_margin(top),
        rank=rank,
        gap=None if rank is None else top[0][1] - top[rank - 1][1],
    )


def _attribute_steps(calls: list[_Call]) -> None:
    """Give every call made on an engine the iterations that ran its requests.

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
    for call in calls:
        for stat in call.stats:
            batching = stat["inflightBatchingStats"]
            rows.setdefault(stat["iter"], {})[stat.get("attentionDpRank", 0)] = (
                batching["numContextRequests"],
                batching["numCtxTokens"],
                batching["numGenRequests"],
            )
            began[stat["iter"]] = datetime.datetime.strptime(stat["timestamp"], _STATS_TIME_FORMAT)
    num_ranks = 1 + max((rank for ranks in rows.values() for rank in ranks), default=0)
    by_start = sorted(calls, key=lambda call: call.start)
    for it in sorted(rows):
        owner = None
        for call in by_start:
            if call.start <= began[it]:
                owner = call
        if owner is not None:
            owner.steps.append(tuple(rows[it].get(rank) for rank in range(num_ranks)))


def _generation_iterations(call: _Call) -> int:
    """How many of the call's iterations ran a generation row on some rank."""
    return sum(1 for step in call.steps if any(row and row[2] for row in step))


def _alone_gaps(call: _Call) -> list[str]:
    """Where the iterations of a one-request call show other work than that request alone.

    The request's rank must run every one of its prompt tokens as context and
    never more than the one request in an iteration, and no other rank may run
    a request. The statistics leave dummies out, so an idle rank under
    attention DP reads as idle while it runs the attention-DP dummy.
    """
    (request,) = call.requests
    own = request.rank or 0
    gaps = []
    context_tokens = 0
    for i, step in enumerate(call.steps):
        for rank, row in enumerate(step):
            if row is None:
                gaps.append(f"iteration {i}: rank {rank} published no row")
            elif rank == own:
                context_tokens += row[1]
                if row[0] + row[2] > 1:
                    gaps.append(f"iteration {i}: rank {rank} ran {row}")
            elif any(row):
                gaps.append(f"iteration {i}: rank {rank} ran {row}, the request is on rank {own}")
    if context_tokens != len(request.token_ids):
        gaps.append(
            f"rank {own} ran {context_tokens} context tokens of a {len(request.token_ids)}-token "
            "prompt"
        )
    return gaps


class _MtpIdentity(LlmapiAccuracyTestHarness):
    """The two legs and their comparisons; a subclass per target sets the topology."""

    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The target's parallel topology as LLM arguments; tp1 passes none.
    PARALLEL: dict = {}

    # The config's max_position_embeddings is 1,048,576, which the engine would
    # otherwise take as max_seq_len and size the KV-cache estimate from. This
    # is the context the target serves: 8192 + 1024 tokens plus draft slack.
    MAX_SEQ_LEN = 9344

    # The checkpoint's trained draft depth: its one MTP layer serves both
    # depths. MTPDecodingConfig defaults to a depth of 1, so it is stated.
    MTP_DRAFT_LEN = 2

    # The near-tie band of the gate, in logits: how far MTP on's token may sit
    # below MTP off's top-1 at a parting. Each target's class states its own,
    # the verify's measured deviation from decode on that topology.
    NEAR_TIE_BAND: float

    # Output tokens per fixed prompt (criterion B): the ~8K-token prompts
    # generate 1K tokens, past any end id.
    SHORT_TOKENS = 256
    LONG_TOKENS = 1024

    # Each prompt alone: prefill is unchunked, so every fixed prompt is one
    # context step of its full length, the same step in both legs.
    ALONE_MAX_NUM_TOKENS = 8192
    CHUNKED_PREFILL = False

    # The batched report: every fixed prompt at once, CUDA graphs padded up to
    # this batch. Prefill is unchunked as the benchmark serves it, so a step
    # holds the longest prompt the classifier restarts from (the prompt and
    # all but one of its output tokens, under MAX_SEQ_LEN) beside a full batch
    # of verify rows.
    BATCH_MAX_BATCH_SIZE = 16
    BATCH_MAX_NUM_TOKENS = MAX_SEQ_LEN + BATCH_MAX_BATCH_SIZE * (1 + MTP_DRAFT_LEN)

    # How many times the classifier restarts the MTP-off run per prompt.
    MAX_REANCHORS = 32

    # Raw logprobs the MTP-off leg returns per emitted token, best first: enough
    # to find the token MTP on emitted at a parting where three or more
    # candidates are nearly tied, not only the runner-up.
    TOP_LOGPROBS = 8

    # Trailing-token counts the classifier tries, in order, when a parting is
    # farther from the MTP-off top-1 than the batch-shape noise: MTP off alone
    # prefills the prompt and the MTP-on output up to that many tokens before
    # the parting, then decodes the rest. A chunk-built state rounds apart
    # from a per-token one, so each count is another rounding of the same
    # tokens; one that emits MTP on's token shows the parting is a near-tie
    # wider than the noise measured from one batch.
    WITNESS_SPLITS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)

    # The executor drains its request queue once per iteration and reports an
    # iteration's statistics after its responses: a pause after every call
    # lets the call's last iteration end before its statistics are read, and
    # before the next call begins.
    SETTLE_SECONDS = 1.0

    # Output tokens of the last call on every engine, compared with nothing:
    # its iterations make the executor publish the statistics of the calls
    # before it, and with MTP these are still more iterations than the
    # executor holds back.
    FLUSH_TOKENS = 32

    def _llm(self, *, max_batch_size: int, max_num_tokens: int, mtp: bool) -> LLM:
        # Built fresh for each leg; the legs of a test differ in
        # speculative_config only.
        #
        # env_overrides is applied on every rank before the model is built:
        # - the target checks its step contract on its first forward;
        # - Triton autotune results go through the on-disk cache. The prefill's
        #   GDP chunk kernel (the vendored FLA `chunk_fwd_kernel_o`) is
        #   autotuned by timing in each process, and its configs sum the key
        #   dimension in different orders, so two engines in two worker
        #   processes could pick configs whose prefill outputs differ in the
        #   last bit. With the cache, the second engine reads the first one's
        #   choice. (When the suite's session reuse hands both legs the same
        #   worker processes, they share the in-memory choice as well.)
        #   Printing shows which leg benchmarked and which read the cache.
        #
        # trust_remote_code: the checkpoint's config class is its own
        # configuration_nemotron4.py. The overlap scheduler is on by default;
        # it is stated because the gate is defined with it on. Every
        # iteration's stats are kept (``max_stats_len=-1``): the per-call
        # compositions come from them.
        variant = {}
        if mtp:
            variant["speculative_config"] = MTPDecodingConfig(max_draft_len=self.MTP_DRAFT_LEN)
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
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            enable_chunked_prefill=self.CHUNKED_PREFILL,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=max_batch_size),
            enable_iter_perf_stats=True,
            max_stats_len=-1,
            **variant,
            **self.PARALLEL,
        )

    @contextlib.contextmanager
    def _engine(self, flush: _Request, *, max_batch_size: int, max_num_tokens: int, mtp: bool):
        """An engine that records its calls; on a clean exit, a last call, then every call's steps.

        ``flush`` is the last call's request: its iterations push out the
        statistics of the calls before it (`FLUSH_TOKENS`).
        """
        with self._llm(
            max_batch_size=max_batch_size, max_num_tokens=max_num_tokens, mtp=mtp
        ) as llm:
            engine = _Engine(llm)
            yield engine
            self._generate(engine, [flush], logprobs=False)
        _attribute_steps(engine.calls)

    def _ranks(self, prompts) -> dict[str, int]:
        """Where each fixed prompt runs, by name; a prompt left out is placed by the engine."""
        return {}

    def _batch_order(self, requests: list[_Request]) -> list[_Request]:
        """The order the batched report submits the fixed prompts in."""
        return requests

    def _requests(self, monkeypatch) -> list[_Request]:
        """The fixed prompt set with its output lengths and ranks, checked to fit the context."""
        monkeypatch.setenv("LLM_MODELS_ROOT", llm_models_root())
        module = _prompt_module()
        self._tokenizer = module.load_tokenizer(self.MODEL_PATH)
        prompts = module.prompt_set(self._tokenizer)
        ranks = self._ranks(prompts)
        requests = [
            _Request(
                p.name,
                p.token_ids,
                self.LONG_TOKENS if p.long else self.SHORT_TOKENS,
                ranks.get(p.name),
            )
            for p in prompts
        ]
        block = KvCacheConfig().tokens_per_block
        for r in requests:
            total = len(r.token_ids) + r.max_tokens + self.MTP_DRAFT_LEN
            print(
                f"[modeling_v2] mtp identity prompt {r.name}: {len(r.token_ids)} tokens, "
                f"{r.max_tokens} generated{_on_rank(r)}"
            )
            if self.CHUNKED_PREFILL:
                # Chunked prefill cuts a prompt into chunks of whole KV blocks
                # but the last, so a final one-token chunk needs a length of one
                # more than a whole number of blocks. Such a chunk runs as a
                # promoted decode row in one leg's graphs, so none of the fixed
                # prompts may have that length.
                assert len(r.token_ids) > 1 and len(r.token_ids) % block != 1, (
                    f"{r.name} ({len(r.token_ids)} tokens) can end in a one-token context chunk "
                    f"with {block}-token KV blocks"
                )
            else:
                # Unchunked, a prompt is one context step of its full length. A
                # final one-token chunk would run as a promoted decode row in one
                # leg's graphs, so none of the fixed prompts may be one token long.
                assert 1 < len(r.token_ids) <= self.ALONE_MAX_NUM_TOKENS, (
                    f"{r.name} ({len(r.token_ids)} tokens) does not run as one context step of "
                    f"more than one token at max_num_tokens {self.ALONE_MAX_NUM_TOKENS}"
                )
            assert total <= self.MAX_SEQ_LEN, (
                f"{r.name}: {len(r.token_ids)} prompt + {r.max_tokens} output + "
                f"{self.MTP_DRAFT_LEN} draft tokens exceed max_seq_len {self.MAX_SEQ_LEN}"
            )
        return requests

    def _flush_request(self, requests: list[_Request]) -> _Request:
        """The last call's request: the shortest fixed prompt, on its rank."""
        shortest = min(requests, key=lambda r: len(r.token_ids))
        return replace(shortest, name="flush", max_tokens=self.FLUSH_TOKENS)

    def _generate(self, engine: _Engine, requests: list[_Request], *, logprobs: bool) -> _Call:
        """Submit the requests in order, as one call, and collect what each emitted.

        Greedy, past any end id; a request with a rank is pinned there. With
        ``logprobs``, the TOP_LOGPROBS highest raw logprobs of every emitted
        token come back too; one-model speculative decoding rejects logprobs,
        so only the MTP-off leg asks for them. The statistics available once
        the call has ended are read with it; `_attribute_steps` sorts them into
        calls when the engine closes.
        """
        params = [
            SamplingParams(
                max_tokens=r.max_tokens,
                temperature=0.0,
                ignore_eos=True,
                logprobs=self.TOP_LOGPROBS if logprobs else None,
            )
            for r in requests
        ]
        start = datetime.datetime.now()
        futures = [
            engine.llm.generate_async(
                {"prompt_token_ids": list(r.token_ids)},
                p,
                scheduling_params=None
                if r.rank is None
                else SchedulingParams(attention_dp_rank=r.rank, attention_dp_relax=False),
            )
            for r, p in zip(requests, params)
        ]
        outputs = {}
        for r, future in zip(requests, futures):
            result = future.result()
            out = result.outputs[0]
            top = None
            if logprobs:
                top = [
                    sorted(((t, lp.logprob) for t, lp in pos.items()), key=lambda x: -x[1])[
                        : self.TOP_LOGPROBS
                    ]
                    for pos in out.logprobs
                ]
            # Per draft position i, the verify steps that drafted it and those
            # that accepted it: every verify step drafts position 0.
            drafted = result.per_pos_drafted or [0]
            outputs[r.name] = _Output(
                list(out.token_ids),
                top,
                decoding_iter=result.decoding_iter,
                verify_steps=drafted[0],
                accepted=sum(result.per_pos_accepted or ()),
            )
        time.sleep(self.SETTLE_SECONDS)
        call = _Call(requests, outputs, start, engine.llm.get_stats(timeout=2))
        engine.calls.append(call)
        return call

    def _alone_leg(self, requests: list[_Request], mtp: bool) -> dict[str, _Call]:
        """Each request alone, one call each, in an engine that holds one request a rank."""
        calls = {}
        with self._engine(
            self._flush_request(requests),
            max_batch_size=1,
            max_num_tokens=self.ALONE_MAX_NUM_TOKENS,
            mtp=mtp,
        ) as engine:
            for r in requests:
                calls[r.name] = self._generate(engine, [r], logprobs=not mtp)
        return calls

    def _write_report(self, name: str, report: dict) -> None:
        report_dir = os.environ.get(REPORT_DIR_ENV)
        if not report_dir:
            return
        os.makedirs(report_dir, exist_ok=True)
        path = os.path.join(report_dir, f"{type(self).__name__}.{name}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[modeling_v2] mtp identity report: {path}")

    def _texts(self, parting: _Parting) -> _Parting:
        """The parting with MTP off's top-1 and MTP on's token as text."""
        parting.off_text = self._tokenizer.decode([parting.off_top[0][0]])
        parting.mtp_text = self._tokenizer.decode([parting.mtp_token])
        return parting

    @skip_not_sm103
    def test_mtp_identity(self, monkeypatch):
        """Greedy tokens with MTP on (max_draft_len 2) follow those with MTP off.

        Each prompt of the fixed set alone: 6 short prompts with 256 tokens and
        4 of ~8K tokens with 1K tokens. Both legs run under modeling_v2
        "require" with the overlap scheduler and CUDA graphs on.

        Each prompt is compared up to its first parting. A parting passes when
        MTP on's token there is within NEAR_TIE_BAND of MTP off's top-1 (its
        gap below the top-1, which is the top-1 / top-2 margin when it is the
        runner-up): the verify's attention rounds otherwise than decode's, by
        up to that much. A parting farther off, or at a token outside MTP
        off's top logprobs, fails: a rollback that keeps a rejected draft's
        state, or a verify that reads the wrong keys, moves a token wherever
        it lands. Every parting is printed with its margin and the tokens on
        both sides.
        """
        requests = self._requests(monkeypatch)
        off = self._alone_leg(requests, mtp=False)
        on = self._alone_leg(requests, mtp=True)

        partings = {}
        for r in requests:
            call_off, call_on = off[r.name], on[r.name]
            # One request in the engine is what makes the prefill the same
            # steps in both legs.
            for leg, call in (("MTP off", call_off), ("MTP on", call_on)):
                gaps = _alone_gaps(call)
                assert not gaps, (
                    f"{r.name}, {leg}: the iterations show other work than the request alone on "
                    f"its rank: {gaps[:4]}"
                )
            out_off, out_on = call_off.outputs[r.name], call_on.outputs[r.name]
            for leg, out in (("MTP off", out_off), ("MTP on", out_on)):
                assert len(out.token_ids) == r.max_tokens, (
                    f"{r.name}, {leg}: {len(out.token_ids)} tokens, {r.max_tokens} asked for"
                )
            parting = _parting(out_on.token_ids, out_off)
            if parting is not None:
                parting.near_tie = parting.gap is not None and parting.gap <= self.NEAR_TIE_BAND
                partings[r.name] = self._texts(parting)
            print(
                f"[modeling_v2] mtp identity {r.name}{_on_rank(r)}: {r.max_tokens} tokens, "
                f"{'identical' if parting is None else f'parts at token {parting.token}'}; "
                f"acceptance length {_show(call_on.acceptance_length)} over "
                f"{out_on.verify_steps} verify steps; iterations with generation rows "
                f"{_generation_iterations(call_on)} (MTP off: {_generation_iterations(call_off)})"
            )
            if parting is not None:
                verdict = "a near-tie" if parting.near_tie else "NOT a near-tie"
                print(
                    f"[modeling_v2]   {r.name} at {parting.describe()}: {verdict} "
                    f"(band {self.NEAR_TIE_BAND})"
                )

        self._write_report(
            "test_mtp_identity",
            {
                "legs": {
                    leg: {
                        name: {
                            "rank": call.requests[0].rank,
                            "token_ids": call.outputs[name].token_ids,
                            "top": call.outputs[name].top,
                            "decoding_iter": call.outputs[name].decoding_iter,
                            "verify_steps": call.outputs[name].verify_steps,
                            "accepted": call.outputs[name].accepted,
                            "steps": call.steps,
                            "acceptance_length": call.acceptance_length,
                        }
                        for name, call in calls.items()
                    }
                    for leg, calls in (("off", off), ("mtp", on))
                },
                "partings": {name: asdict(p) for name, p in partings.items()},
                "near_tie_band": self.NEAR_TIE_BAND,
            },
        )
        print(
            f"[modeling_v2] mtp identity: {len(requests) - len(partings)} of {len(requests)} "
            f"prompts identical, {sum(p.near_tie for p in partings.values())} part at a near-tie "
            f"within {self.NEAR_TIE_BAND}"
        )
        wide = {name: p for name, p in partings.items() if not p.near_tie}
        assert not wide, (
            f"MTP on parts from MTP off off a near-tie (band {self.NEAR_TIE_BAND}):\n"
            + "\n".join(f"{name} at {p.describe()}" for name, p in wide.items())
        )

    @skip_not_sm103
    def test_mtp_batched_report(self, monkeypatch):
        """MTP on against off with the fixed prompts in one batch: reported, not gated.

        The batch-shape noise is the largest change of an emitted token's
        logprob, MTP off, between each prompt alone and in the batch, over the
        tokens before they part. Each parting of MTP on from MTP off is a
        near-tie when the token MTP on emitted is within that noise, or within
        the verify's band (NEAR_TIE_BAND), of the MTP-off top-1 there: the
        runner-up, or a lower rank where three or more candidates are that
        close. A parting farther off is a near-tie still
        when MTP off itself emits that token over the same tokens with another
        prefill/decode split (a witness): where several candidates sit within a
        few tenths, one batch's noise does not bound how far rounding moves
        them. The MTP-off run then restarts from the prompt and the MTP-on
        output up to and including the parting token, so each later parting is
        classified against a run that shares its prefix. The test asserts only
        that every output is whole and every parting was reached; the table and
        the classes are what it reports.
        """
        requests = self._batch_order(self._requests(monkeypatch))
        assert len(requests) <= self.BATCH_MAX_BATCH_SIZE
        engine = dict(
            max_batch_size=self.BATCH_MAX_BATCH_SIZE, max_num_tokens=self.BATCH_MAX_NUM_TOKENS
        )
        flush = self._flush_request(requests)
        with self._engine(flush, mtp=True, **engine) as mtp_engine:
            on = self._generate(mtp_engine, requests, logprobs=False)
        with self._engine(flush, mtp=False, **engine) as off_engine:
            off = self._generate(off_engine, requests, logprobs=True)
            alone = {r.name: self._generate(off_engine, [r], logprobs=True) for r in requests}
            noise, noise_at = self._batch_shape_noise(requests, off, alone)
            partings, uncovered = self._classify(off_engine, requests, on, off, noise)

        for r in requests:
            for leg, out in (("MTP on", on.outputs[r.name]), ("MTP off", off.outputs[r.name])):
                assert len(out.token_ids) == r.max_tokens, (
                    f"{r.name}, {leg}: {len(out.token_ids)} tokens, {r.max_tokens} asked for"
                )
        # Which requests shared a prefill step: a verify row takes 1 + D tokens
        # of a mixed step, so the same composition still runs other GEMM shapes.
        prefills = {
            leg: [(i, s) for i, s in enumerate(call.steps) if any(row and row[0] for row in s)]
            for leg, call in (("MTP on", on), ("MTP off", off))
        }
        print(
            f"[modeling_v2] mtp batched: acceptance length {_show(on.acceptance_length)}; "
            f"batch-shape noise {noise:.6f} ({noise_at}); iterations MTP on {len(on.steps)}, "
            f"MTP off {len(off.steps)}; prefill iterations as (index, per rank (context requests, "
            f"context tokens, generation requests)): {prefills}"
        )
        for r in requests:
            alone_parts = _parting(alone[r.name].outputs[r.name].token_ids, off.outputs[r.name])
            within = sum(bool(p.near_tie) for p in partings[r.name])
            witnessed = sum(p.witness is not None for p in partings[r.name])
            print(
                f"[modeling_v2] mtp batched {r.name}{_on_rank(r)}: {len(partings[r.name])} "
                f"partings, {within} within the noise or the band, {witnessed} witnessed; MTP off "
                f"alone vs batch: "
                f"{'identical' if alone_parts is None else f'parts at {alone_parts.token}'}"
            )
            for p in partings[r.name]:
                if p.near_tie:
                    verdict = "near-tie"
                elif p.witness is not None:
                    verdict = (
                        f"near-tie wider than the noise: MTP off emits {p.mtp_token} there itself "
                        f"when it decodes the last {p.witness} tokens"
                    )
                else:
                    verdict = f"NOT a near-tie: no split of {self.WITNESS_SPLITS} emits it"
                print(f"[modeling_v2]   restart {p.reanchor}, {p.describe()}: {verdict}")
        self._write_report(
            "test_mtp_batched_report",
            {
                "noise": noise,
                "noise_at": noise_at,
                "acceptance_length": on.acceptance_length,
                "steps": {"mtp": on.steps, "off": off.steps},
                "outputs": {
                    r.name: {
                        "rank": r.rank,
                        "mtp": on.outputs[r.name].token_ids,
                        "mtp_verify_steps": on.outputs[r.name].verify_steps,
                        "mtp_accepted": on.outputs[r.name].accepted,
                        "off": off.outputs[r.name].token_ids,
                        "off_top": off.outputs[r.name].top,
                        "alone": alone[r.name].outputs[r.name].token_ids,
                        "alone_top": alone[r.name].outputs[r.name].top,
                    }
                    for r in requests
                },
                "partings": {name: [asdict(p) for p in ps] for name, ps in partings.items()},
                "uncovered": uncovered,
            },
        )
        assert not uncovered, (
            f"the classifier stopped after {self.MAX_REANCHORS} restarts before the end of "
            f"{uncovered}"
        )

    @staticmethod
    def _batch_shape_noise(
        requests: list[_Request], batch: _Call, alone: dict[str, _Call]
    ) -> tuple[float, str]:
        """The largest |logprob change| of an emitted token, alone against the batch.

        Over each prompt's tokens before its alone and batched outputs part
        (where both emitted the same token), MTP off.
        """
        noise, noise_at = 0.0, "no token moved"
        for r in requests:
            in_batch = batch.outputs[r.name]
            by_itself = alone[r.name].outputs[r.name]
            pos = _first_divergence(by_itself.token_ids, in_batch.token_ids)
            common = len(in_batch.token_ids) if pos is None else pos
            for i in range(common):
                delta = abs(in_batch.top[i][0][1] - by_itself.top[i][0][1])
                if delta > noise:
                    noise, noise_at = delta, f"{r.name} token {i}"
        return noise, noise_at

    def _witness(
        self, engine: _Engine, request: _Request, mtp_tokens: list[int], token: int
    ) -> int | None:
        """The first of WITNESS_SPLITS at which MTP off itself emits MTP on's token.

        MTP off alone prefills the prompt and ``mtp_tokens[: token - j]``, then
        decodes ``j + 1`` tokens. The split counts only while the decoded tokens
        follow ``mtp_tokens`` up to the parting, so the run reaches it over the
        same tokens. None when no split emits it.
        """
        for j in self.WITNESS_SPLITS:
            if j > token:
                break
            run = replace(
                request,
                token_ids=request.token_ids + tuple(mtp_tokens[: token - j]),
                max_tokens=j + 1,
            )
            emitted = self._generate(engine, [run], logprobs=False).outputs[request.name].token_ids
            if emitted == mtp_tokens[token - j : token + 1]:
                return j
        return None

    def _classify(
        self,
        engine: _Engine,
        requests: list[_Request],
        on: _Call,
        off: _Call,
        noise: float,
    ) -> tuple[dict[str, list[_Parting]], list[str]]:
        """Every parting of the MTP-on outputs from MTP off, each classified.

        A parting within the noise, or within the verify's band, of the MTP-off
        top-1 is a near-tie; one farther off gets a witness search
        (``_witness``). After a parting at token t, the MTP-off engine runs the
        prompt followed by MTP on's tokens up to and including t, greedy, for
        the rest of the output, and the next parting is looked for in that run.
        The restarts of all prompts run as one call per round, each on its
        prompt's rank.
        """
        partings = {r.name: [] for r in requests}
        # name -> (offset into the MTP-on output, the MTP-off run from there).
        pending = {r.name: (0, off.outputs[r.name]) for r in requests}
        uncovered = []
        for reanchor in range(self.MAX_REANCHORS + 1):
            restarts, offsets = [], {}
            for r in requests:
                if r.name not in pending:
                    continue
                offset, run = pending.pop(r.name)
                mtp_tokens = on.outputs[r.name].token_ids
                parting = _parting(mtp_tokens, run, offset)
                if parting is None:
                    continue
                parting.reanchor = reanchor
                parting.near_tie = parting.gap is not None and parting.gap <= max(
                    noise, self.NEAR_TIE_BAND
                )
                if not parting.near_tie:
                    parting.witness = self._witness(engine, r, mtp_tokens, parting.token)
                partings[r.name].append(parting)
                resume = parting.token + 1
                if resume == len(mtp_tokens):
                    continue
                if reanchor == self.MAX_REANCHORS:
                    uncovered.append(f"{r.name} (from token {resume})")
                    continue
                restarts.append(
                    replace(
                        r,
                        token_ids=r.token_ids + tuple(mtp_tokens[:resume]),
                        max_tokens=len(mtp_tokens) - resume,
                    )
                )
                offsets[r.name] = resume
            if not restarts:
                break
            call = self._generate(engine, restarts, logprobs=True)
            for req in restarts:
                pending[req.name] = (offsets[req.name], call.outputs[req.name])
        return partings, uncovered


def _on_rank(request: _Request) -> str:
    return "" if request.rank is None else f" on rank {request.rank}"
