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
"""The acceptance-length harness of the nemotron4 modeling_v2 targets' MTP draft path.

Each target's gate is a subclass of ``_MtpAcceptance`` in
test_modeling_v2_nemotron4.py, beside the target's other gates; this module
holds the legs, the reference walk and the checks they share.

A draft path that computes the wrong thing leaves greedy output correct: the
verify step rejects the bad drafts and emits the target model's own tokens. It
only makes decoding slower, so the acceptance length is what sees it.

TensorRT-LLM has no built-in implementation of this architecture, so there is
no ``modeling_v2="off"`` leg to anchor the acceptance length. The anchor is the
checkpoint's own MTP layer instead: the plain-torch reference of the unit tests
(``_nemotron4_reference.py``, transcribed from the checkpoint's
``modeling_nemotron4.py`` and cross-checked against it at 8K tokens) runs
teacher-forced over the target's own greedy output. Its drafts at every
position, walked the way greedy MTP decoding walks them, give the acceptance
length the target should measure.

The two sides draft from the same tokens, so their drafts can disagree only
where the reference's draft is a near-tie: where some token's logit could gain
on the reference's top-1 by as much as the two implementations' logits differ.
The gate measures where the reference's drafts are near-ties and derives its
tolerance from them.

Every engine here passes ``modeling_v2="require"``: there is nothing to fall
back to, and ``"require"`` makes a routing decline raise with its reason.
"""

import gc
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass

import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm.llmapi import CudaGraphConfig, MTPDecodingConfig, SchedulingParams

from ..conftest import llm_models_root, unittest_path
from .accuracy_core import LlmapiAccuracyTestHarness, acceptance_length_from_iteration_stats

# Setting this to a file path writes the per-prompt and pooled figures there as
# JSON, for comparing runs offline.
REPORT_ENV = "NEMOTRON4_ACCEPTANCE_REPORT"


def _unittest_module(name: str):
    """A module of the nemotron4 unit tests, loaded by path.

    ``_nemotron4_prompts`` is the fixed prompt set, its token ids pinned;
    ``_nemotron4_reference`` the plain-torch reference. Both import neither
    transformers' modeling code nor TensorRT-LLM.
    """
    if name not in sys.modules:
        path = unittest_path() / "_torch" / "modeling_v2" / "models" / "nemotron4" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves the module's string annotations through sys.modules.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass
class _EngineRun:
    """What the engine did for one request.

    ``chunks`` holds the number of tokens of each streamed response: the
    executor answers a streaming request once per iteration, so the first holds
    the prefill's token and every later one a verify step's 1 + accepted tokens.
    The other fields are the request's own speculative-decoding counters.
    """

    output: list[int]
    chunks: list[int]
    accepted: int
    drafted: int
    per_pos_accepted: list[int]
    per_pos_drafted: list[int]
    decoding_iter: int


def _engine_walk(
    name: str, prompt_len: int, run: _EngineRun, draft_len: int
) -> list[tuple[int, int]]:
    """The engine's verify steps as ``(q, accepted)``, ``q`` the drafting position.

    The prefill drafts at the prompt's last position, and a step that accepts
    ``a`` drafts makes the next one draft ``1 + a`` positions further on (the
    reference's ``greedy_walk``). The last step is cut at ``max_tokens``, so its
    accepted count is the request's total less the others'.

    Before a step is used, the streamed sizes are held against the request's
    counters; they count the same steps, so they must agree exactly.
    """
    assert sum(run.chunks) == len(run.output), (name, run.chunks, len(run.output))
    assert run.chunks[0] == 1, f"{name}: the prefill's response carried {run.chunks[0]} tokens"
    sizes = run.chunks[1:]
    assert sizes, f"{name}: no verify step"
    assert all(1 <= size <= 1 + draft_len for size in sizes), (name, sizes)
    accepted = [size - 1 for size in sizes[:-1]]
    last = run.accepted - sum(accepted)
    assert sizes[-1] - 1 <= last <= draft_len, (
        f"{name}: the streamed steps account for {sum(accepted)} of the {run.accepted} "
        f"accepted drafts, leaving {last} for a last step that emitted {sizes[-1]} tokens"
    )
    accepted.append(last)
    steps = len(accepted)
    assert run.drafted == steps * draft_len, (name, run.drafted, steps)
    assert run.per_pos_drafted[:draft_len] == [steps] * draft_len, (name, run.per_pos_drafted)
    assert run.per_pos_accepted[:draft_len] == [
        sum(a > d for a in accepted) for d in range(draft_len)
    ], (name, run.per_pos_accepted, accepted)
    walk = []
    q = prompt_len - 1
    for a in accepted:
        walk.append((q, a))
        q += 1 + a
    return walk


def _reference_accepted(q: int, seq: list[int], drafts: list[list[int]]) -> int | None:
    """The reference's accepted count for a verify step drafting at ``q``.

    None if the step's window runs past the known tokens.
    """
    if q + 1 + len(drafts) >= len(seq):
        return None
    accepted = 0
    while accepted < len(drafts) and drafts[accepted][q] == seq[q + 2 + accepted]:
        accepted += 1
    return accepted


def _al(accepted: list[int]) -> float:
    """The engine's estimator: (accepted drafts + verify steps) / verify steps."""
    return (sum(accepted) + len(accepted)) / len(accepted)


class _MtpAcceptance(LlmapiAccuracyTestHarness):
    """The legs and checks every nemotron4 target's MTP acceptance gate shares.

    A target's class states its topology and the serve settings it needs, where
    each prompt runs, how far its MTP logits can be from the reference's, and
    how it reads its acceptance length off the engine.
    """

    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The config's max_position_embeddings is 1,048,576, which the engine would
    # otherwise take as max_seq_len. 8192 + 1024 tokens plus draft slack.
    MAX_SEQ_LEN = 9344

    # The checkpoint's trained draft depth: its one MTP layer serves both
    # depths. MTPDecodingConfig defaults to a depth of 1, so it is stated.
    MTP_DRAFT_LEN = 2

    # Every prompt of the set runs in one batch: verify, draft steps and the
    # state commit run on several requests at once and beside the long prompts'
    # prefills, and the batch then drains through the smaller padded graphs.
    MAX_BATCH_SIZE = 16

    # Output tokens, past any end id: the short prompts get criterion A's 256,
    # the ~8K prompts the benchmark's 1024.
    SHORT_MAX_TOKENS = 256
    LONG_MAX_TOKENS = 1024

    # A collapsed draft path accepts nothing (1.0). The reference's own drafts
    # give about 2.8 over the target's outputs of the fixed prompts, 2.27-2.97
    # per prompt.
    MIN_ACCEPTANCE_LENGTH = 2.0

    # The target's topology and the serve settings it needs, as LLM arguments.
    ENGINE_ARGS: dict = {}

    # How far the target's MTP logits can move a margin of the reference's
    # drafts, per draft depth, and how far the target trunk's logits can be
    # from the reference's: each target states its own, and where they were
    # measured.
    MTP_LOGIT_ERROR: tuple[float, float]
    TRUNK_LOGIT_BAND: float

    # How long the GPU may take to come back once the engine has exited.
    RELEASE_SECONDS = 60

    def _epsilon(self) -> tuple[float, ...]:
        return tuple(error + self.TRUNK_LOGIT_BAND for error in self.MTP_LOGIT_ERROR)

    def _max_tokens(self, prompt) -> int:
        return self.LONG_MAX_TOKENS if prompt.long else self.SHORT_MAX_TOKENS

    def _ranks(self, prompts) -> dict[str, int]:
        """The attention-DP rank each prompt is pinned to; none without attention DP."""
        return {}

    def _engine_al(
        self, stats: list[dict], walks: dict[str, list[tuple[int, int]]], ranks: dict[str, int]
    ) -> tuple[float, int, dict]:
        """AL_trt, the verify steps it counts beyond the requests' own, and what to report of it.

        AL_trt comes from the iteration statistics, as every acceptance gate of
        the suite reads it. They count steps the requests do not: when a
        request's last step starts with two or more tokens left, the overlap
        scheduler has already scheduled it for one more iteration, and that
        iteration's statistics count the request again with its last accepted
        count (at most one per request). The request's own counters do not, so
        the extra steps are the iteration statistics' step count less the
        requests'.
        """
        spec_stats = [s["specDecodingStats"] for s in stats if s.get("specDecodingStats")]
        stats_steps = sum(s["numRequestsWithDraftTokens"] for s in spec_stats)
        request_steps = sum(len(walk) for walk in walks.values())
        return (
            acceptance_length_from_iteration_stats(stats),
            abs(stats_steps - request_steps),
            dict(
                iteration_stats_steps=stats_steps,
                iteration_stats_accepted=sum(s["numAcceptedTokens"] for s in spec_stats),
            ),
        )

    def _engine_leg(
        self, prompts, ranks: dict[str, int]
    ) -> tuple[dict[str, _EngineRun], list[dict]]:
        """All prompts in one batch through the target, streamed; and the run's iteration stats.

        The executor answers a streaming request once per iteration, so the
        sizes of the responses are the request's verify steps. A prompt with a
        rank runs there and nowhere else.
        """
        # The target checks its step contract on the first forward and only when
        # asked; it travels as ``env_overrides`` because the check runs on the
        # rank. trust_remote_code: the checkpoint's config class is its own
        # configuration_nemotron4.py. The overlap scheduler is on by default; it
        # is stated because the gate is defined with it on. Every iteration's
        # statistics are kept (``max_stats_len=-1``), not the last 1000.
        with LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            trust_remote_code=True,
            max_seq_len=self.MAX_SEQ_LEN,
            max_batch_size=self.MAX_BATCH_SIZE,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE
            ),
            speculative_config=MTPDecodingConfig(max_draft_len=self.MTP_DRAFT_LEN),
            enable_iter_perf_stats=True,
            max_stats_len=-1,
            **self.ENGINE_ARGS,
        ) as llm:
            futures = [
                llm.generate_async(
                    {"prompt_token_ids": list(p.token_ids)},
                    SamplingParams(
                        max_tokens=self._max_tokens(p), temperature=0.0, ignore_eos=True
                    ),
                    streaming=True,
                    scheduling_params=SchedulingParams(
                        attention_dp_rank=ranks[p.name], attention_dp_relax=False
                    )
                    if p.name in ranks
                    else None,
                )
                for p in prompts
            ]
            runs = {}
            # Each request's responses wait in its own queue, so reading the
            # requests one after the other still sees every response apart.
            for p, future in zip(prompts, futures):
                chunks = [len(step.outputs[0].token_ids_diff) for step in future]
                runs[p.name] = _EngineRun(
                    output=list(future.outputs[0].token_ids),
                    chunks=chunks,
                    accepted=future.spec_dec_totals[0],
                    drafted=future.spec_dec_totals[1],
                    per_pos_accepted=list(future.per_pos_accepted),
                    per_pos_drafted=list(future.per_pos_drafted),
                    decoding_iter=future.decoding_iter,
                )
            # A statistic can reach the queue after the last response.
            time.sleep(1.0)
            stats = llm.get_stats(timeout=2)
        return runs, stats

    def _reference_leg(self, reference, prompts, runs: dict[str, _EngineRun]) -> dict:
        """The reference teacher-forced over each prompt and the target's output.

        Both MTP-Eagle (the semantics the engine runs) and HF's per-depth MTP.
        """
        # The engine's processes give their memory back as they exit.
        deadline = time.monotonic() + self.RELEASE_SECONDS
        free, _ = torch.cuda.mem_get_info()
        while free < 200 * 2**30 and time.monotonic() < deadline:
            time.sleep(2.0)
            free, _ = torch.cuda.mem_get_info()
        assert free >= 200 * 2**30, (
            f"the reference needs ~190 GiB of the GPU after the engine exits; "
            f"{free / 2**30:.1f} GiB is free"
        )
        model = reference.Nemotron4Reference(self.MODEL_PATH)
        out = {}
        for p in prompts:
            seq = list(p.token_ids) + runs[p.name].output
            ids = torch.tensor(seq, dtype=torch.long, device="cuda")
            with torch.inference_mode():
                hidden = model.forward(ids, logits=None).hidden
                # The reference's own greedy token at every output position, for the report.
                greedy = model.greedy_ids(hidden[len(p.token_ids) - 1 : -1]).tolist()
            out[p.name] = dict(
                greedy_agreement=sum(g == t for g, t in zip(greedy, runs[p.name].output))
                / len(greedy),
                **{
                    mode: model.teacher_forced_mtp_acceptance(
                        seq,
                        len(p.token_ids),
                        hidden=hidden,
                        mode=mode,
                        num_depths=self.MTP_DRAFT_LEN,
                    )
                    for mode in ("eagle", "hf")
                },
            )
            del ids, hidden
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return out

    def _check_acceptance(self, monkeypatch) -> None:
        """The acceptance length the target measures matches the reference's.

        Leg 1 runs the fixed prompt set (6 short prompts x 256 tokens, 4 of ~8K
        tokens x 1024) through the target in one batch with MTP-Eagle at the
        class's draft length D (``MTP_DRAFT_LEN``), the overlap scheduler and
        CUDA graphs on: AL_trt, read as the target's class reads it
        (``_engine_al``). Leg 2 runs the reference teacher-forced over each
        prompt and the target's own output, D depths deep: AL_ref, its greedy
        walk pooled with the same estimator.

        **Tolerance.** At a verify step the two sides' accepted counts differ
        only if the deciding draft's outcome flips, which needs some token to
        gain at least the reference's margin there (``draft_scores``) against
        the logit error ``epsilon`` between the implementations. A flip at depth
        d of D moves the step's count by at most D - d + 1: at D = 2, by 2 at
        depth 1 and 1 at depth 2. So over the reference's n steps:

            tol = (sum over d of (D - d + 1) * k_d + D * E) / n

        which at D = 2 is (2 * k1 + k2 + 2 * E) / n. k_d counts the steps whose
        first d - 1 drafts matched, whose depth-d margin is below epsilon and
        whose earlier margins are not (k1: the depth-1 margin is below it). E
        counts the steps AL_trt holds that the reference walk cannot pair, each
        moving a mean of n steps by at most D / n:

        - the engine's last step of an output whose window passes the last
          token, which the reference cannot score (at most one per request);
        - the steps AL_trt counts beyond the requests' own (``_engine_al``).

        **At the engine's own positions.** The streamed responses give the
        engine's walk step by step, so the reference scores each of the
        engine's steps where the engine took it. A step whose accepted count
        differs must sit at a near-tie of the draft that decides it; anywhere
        else the target's draft path computed something other than the
        checkpoint's MTP layer.

        AL_trt >= 2.0 is a tripwire for a collapsed draft path. The reference's
        AL with HF's per-depth MTP (``mode="hf"``) is reported beside it.
        """
        draft_len = self.MTP_DRAFT_LEN
        monkeypatch.setenv("LLM_MODELS_ROOT", llm_models_root())
        prompts_module = _unittest_module("_nemotron4_prompts")
        reference = _unittest_module("_nemotron4_reference")
        prompts = prompts_module.prompt_set(prompts_module.load_tokenizer(self.MODEL_PATH))
        ranks = self._ranks(prompts)

        runs, stats = self._engine_leg(prompts, ranks)
        walks = {
            p.name: _engine_walk(p.name, len(p.token_ids), runs[p.name], draft_len) for p in prompts
        }
        al_trt, extra_steps, engine_report = self._engine_al(stats, walks, ranks)
        # Iterations of a rank that ran context rows beside generation rows.
        batches = [
            (s.get("attentionDpRank", 0), s.get("inflightBatchingStats") or {}) for s in stats
        ]
        mixed_steps = Counter(
            rank
            for rank, batch in batches
            if batch.get("numContextRequests", 0) > 0 and batch.get("numGenRequests", 0) > 0
        )
        gc.collect()
        torch.cuda.empty_cache()

        refs = self._reference_leg(reference, prompts, runs)

        epsilon = self._epsilon()
        assert len(epsilon) == draft_len, f"epsilon {epsilon} for {draft_len} draft depths"
        rows, flips_all, offenders = [], [], []
        pooled = dict(ref_accepted=[], ref_hf_accepted=[], trt_accepted=[], k=[0] * draft_len)
        boundary_total = 0
        for p in prompts:
            run, ref = runs[p.name], refs[p.name]
            eagle, hf = ref["eagle"], ref["hf"]
            seq = list(p.token_ids) + run.output
            walk = walks[p.name]

            # The reference's own walk and its near-ties (the tolerance): a step's count can
            # flip at depth d + 1 only when its first d drafts matched (a >= d).
            k = [0] * draft_len
            for q, a in zip(eagle["positions"], eagle["accepted"]):
                for depth in range(min(a + 1, draft_len)):
                    if eagle["margins"][depth][q] < epsilon[depth]:
                        k[depth] += 1
                        break

            # The engine's walk, each step scored by the reference where the engine took it.
            flips, boundary, matched_trt, matched_ref = [], 0, [], []
            for step, (q, a) in enumerate(walk):
                a_ref = _reference_accepted(q, seq, eagle["drafts"])
                if a_ref is None:
                    boundary += 1
                    continue
                matched_trt.append(a)
                matched_ref.append(a_ref)
                if a == a_ref:
                    continue
                depth = min(a, a_ref)
                margin = eagle["margins"][depth][q]
                flip = dict(
                    prompt=p.name,
                    step=step,
                    q=q,
                    accepted_trt=a,
                    accepted_ref=a_ref,
                    depth=depth + 1,
                    margin=round(margin, 4),
                    epsilon=round(epsilon[depth], 4),
                    ref_draft=eagle["drafts"][depth][q],
                    true_token=seq[q + 2 + depth],
                )
                flips.append(flip)
                if margin >= epsilon[depth]:
                    offenders.append(flip)

            al_trt_request = _al([a for _, a in walk])
            row = dict(
                prompt=p.name,
                **({"rank": ranks[p.name]} if p.name in ranks else {}),
                prompt_tokens=len(p.token_ids),
                output_tokens=len(run.output),
                trt_steps=len(walk),
                al_trt=round(al_trt_request, 4),
                ref_steps=eagle["verify_steps"],
                al_ref=round(eagle["acceptance_length"], 4),
                al_ref_hf=round(hf["acceptance_length"], 4),
                **{f"k{depth + 1}": k[depth] for depth in range(draft_len)},
                al_ref_at_trt_steps=round(_al(matched_ref), 4),
                al_trt_at_trt_steps=round(_al(matched_trt), 4),
                flips=len(flips),
                flip_margins=[f["margin"] for f in flips],
                boundary_steps=boundary,
                depth_match=[round(x, 4) for x in eagle["depth_match"]],
                ref_greedy_agreement=round(ref["greedy_agreement"], 4),
                decoding_iter=run.decoding_iter,
            )
            rows.append(row)
            print(f"[modeling_v2] mtp acceptance {json.dumps(row)}")
            pooled["ref_accepted"] += eagle["accepted"]
            pooled["ref_hf_accepted"] += hf["accepted"]
            pooled["trt_accepted"] += [a for _, a in walk]
            pooled["k"] = [total + mine for total, mine in zip(pooled["k"], k)]
            flips_all += flips
            boundary_total += boundary

        n = len(pooled["ref_accepted"])
        al_ref = _al(pooled["ref_accepted"])
        al_ref_hf = _al(pooled["ref_hf_accepted"])
        al_trt_requests = _al(pooled["trt_accepted"])
        request_steps = len(pooled["trt_accepted"])
        unpaired = boundary_total + extra_steps
        weighted = " + ".join(
            f"{draft_len - depth} * {pooled['k'][depth]}" for depth in range(draft_len)
        )
        tol = (
            sum((draft_len - depth) * pooled["k"][depth] for depth in range(draft_len))
            + draft_len * unpaired
        ) / n
        summary = dict(
            al_trt=round(al_trt, 4),
            al_trt_from_requests=round(al_trt_requests, 4),
            **engine_report,
            request_steps=request_steps,
            request_accepted=sum(pooled["trt_accepted"]),
            al_ref=round(al_ref, 4),
            al_ref_hf=round(al_ref_hf, 4),
            ref_steps=n,
            **{f"k{depth + 1}": pooled["k"][depth] for depth in range(draft_len)},
            requests=len(prompts),
            epsilon=[round(e, 4) for e in epsilon],
            tol=round(tol, 4),
            delta=round(al_trt - al_ref, 4),
            flips=len(flips_all),
            max_flip_margin=[
                max((f["margin"] for f in flips_all if f["depth"] == depth), default=None)
                for depth in range(1, draft_len + 1)
            ],
            boundary_steps=boundary_total,
            unpaired_steps=unpaired,
            off_near_tie=len(offenders),
            mixed_steps_by_rank=dict(sorted(mixed_steps.items())),
        )
        print(f"[modeling_v2] mtp acceptance pooled {json.dumps(summary)}")
        report_path = os.environ.get(REPORT_ENV)
        if report_path:
            engine = {name: asdict(run) for name, run in runs.items()}
            with open(report_path, "w") as f:
                json.dump(dict(summary=summary, prompts=rows, engine=engine), f, indent=1)

        assert not offenders, (
            f"{len(offenders)} verify steps accept a different number of drafts than the "
            f"reference where the reference's deciding draft is no near-tie (margin >= epsilon "
            f"{[round(e, 4) for e in epsilon]}): the target's draft path computes something other "
            f"than the checkpoint's MTP layer there. First ones: {offenders[:5]}"
        )
        assert al_trt >= self.MIN_ACCEPTANCE_LENGTH, (
            f"acceptance length {al_trt:.4f} < {self.MIN_ACCEPTANCE_LENGTH}: the draft path "
            f"barely drafts anything the target accepts"
        )
        assert abs(al_trt - al_ref) <= tol, (
            f"AL_trt {al_trt:.4f} vs AL_ref {al_ref:.4f}: |delta| {abs(al_trt - al_ref):.4f} > "
            f"tol {tol:.4f} = ({weighted} + {draft_len} * {unpaired}) / {n}"
        )
