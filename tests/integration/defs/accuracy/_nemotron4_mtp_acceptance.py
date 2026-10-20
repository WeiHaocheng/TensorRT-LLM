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
tolerance from them. A deep draft can also part where the trunk's ordinary
drift meets an ill-conditioned MTP chain; on one GPU the harness records the
trunk hidden states the engine computed (``_TrunkCapture``) and re-checks such
a step from them (``_MtpAcceptance._check_acceptance``).

Every engine here passes ``modeling_v2="require"``: there is nothing to fall
back to, and ``"require"`` makes a routing decline raise with its reason.
"""

import gc
import importlib.util
import inspect
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


class _TrunkCapture:
    """The target trunk's final normed hidden state at every position of every request, as the
    engine computed it.

    The trunk hands that state, one row per token of the step, to the MTP-Eagle worker
    (``Eagle3OneModelWorker._forward_impl``), whose first draft step reads it: the draft at a
    verify step's last accepted position, and the MTP layer's depth-1 keys and values at every
    position. Three wrappers record it without changing what the engine computes:

    - ``_forward_impl`` copies the step's hidden states and positions into fixed device buffers.
      The copies are captured into the CUDA graphs with the rest of the step, so every replay
      refreshes them;
    - ``TrtllmAttentionMetadata.prepare`` notes the step's row layout: the request of each row
      group and its token count, padding rows included;
    - ``PyTorchModelEngine.forward`` clones the buffers' used rows after each step, on the stream
      the next step runs on, so no later step overwrites them first, and notes which client
      request (``GenerationResult.request_id``) each engine request serves.

    The executor must run in this process for the wrappers to reach it
    (``TLLM_WORKER_USE_SINGLE_PROCESS``), which TensorRT-LLM offers for one GPU only.
    """

    SINGLE_PROCESS_ENV = "TLLM_WORKER_USE_SINGLE_PROCESS"

    def __init__(self, hidden_size: int, max_rows: int):
        self.hidden_size = hidden_size
        self.max_rows = max_rows
        # Per step: (request id per row group, tokens per group, positions, hidden rows).
        self.steps: list[tuple[list[int], list[int], torch.Tensor, torch.Tensor]] = []
        # Steps recorded, and steps with more rows than the buffers hold, whose rows are not kept.
        self.captured_steps = 0
        self.overflow_steps = 0
        # Engine request id -> client request id.
        self.client_ids: dict[int, int] = {}
        self._layout: tuple[list[int], list[int]] | None = None

    def install(self, monkeypatch) -> None:
        from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
        from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

        monkeypatch.setenv(self.SINGLE_PROCESS_ENV, "1")
        # Allocated before the engine captures its graphs, which then copy into them. The engine
        # runs under inference mode, so they are inference tensors.
        with torch.inference_mode():
            hidden = torch.zeros(
                self.max_rows, self.hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            positions = torch.zeros(self.max_rows, dtype=torch.int64, device="cuda")
        capture = self

        forward_impl = Eagle3OneModelWorker._forward_impl
        signature = inspect.signature(forward_impl)

        def record_trunk(worker, *args, **kwargs):
            bound = signature.bind(worker, *args, **kwargs).arguments
            rows = min(bound["hidden_states"].shape[0], capture.max_rows)
            hidden[:rows].copy_(bound["hidden_states"][:rows])
            positions[:rows].copy_(bound["position_ids"].reshape(-1)[:rows])
            return forward_impl(worker, *args, **kwargs)

        prepare = TrtllmAttentionMetadata.prepare

        def record_layout(metadata, *args, **kwargs):
            result = prepare(metadata, *args, **kwargs)
            capture._layout = (
                list(metadata.request_ids),
                metadata.seq_lens[: metadata.num_seqs].tolist(),
            )
            return result

        engine_forward = PyTorchModelEngine.forward

        def record_step(engine, batch, *args, **kwargs):
            capture._layout = None
            outputs = engine_forward(engine, batch, *args, **kwargs)
            if not engine.is_warmup and capture._layout is not None:
                for request in batch.all_requests():
                    capture.client_ids[request.py_request_id] = request.py_client_id
                request_ids, lengths = capture._layout
                rows = sum(lengths)
                if rows > capture.max_rows:
                    capture.overflow_steps += 1
                else:
                    capture.captured_steps += 1
                    with torch.inference_mode():
                        capture.steps.append(
                            (request_ids, lengths, positions[:rows].clone(), hidden[:rows].clone())
                        )
            return outputs

        monkeypatch.setattr(Eagle3OneModelWorker, "_forward_impl", record_trunk)
        monkeypatch.setattr(TrtllmAttentionMetadata, "prepare", record_layout)
        monkeypatch.setattr(PyTorchModelEngine, "forward", record_step)

    def traces(self, lengths: dict[int, int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per client request id: its ``[length, hidden]`` trace on the CPU and which positions it
        holds.

        A position keeps the last step that ran it. A verify step runs the token after the last
        accepted one and the drafts behind it; the positions of the drafts it rejects run again,
        from the token it chose, in the steps after it, and no step runs a position before its
        own first one. So every position up to a verify step's last accepted one holds the hidden
        state of the step that accepted it, which is the one the MTP layer read.
        """
        out = {}
        for request_id, length in lengths.items():
            out[request_id] = (
                torch.zeros(length, self.hidden_size, dtype=torch.bfloat16, device="cuda"),
                torch.zeros(length, dtype=torch.bool, device="cuda"),
            )
        with torch.inference_mode():
            for request_ids, group_lengths, positions, hidden in self.steps:
                start = 0
                for request_id, count in zip(request_ids, group_lengths):
                    client_id = self.client_ids.get(request_id)
                    if client_id in out:
                        trace, held = out[client_id]
                        where = positions[start : start + count]
                        keep = where < trace.shape[0]
                        where = where[keep]
                        trace.index_copy_(0, where, hidden[start : start + count][keep])
                        held.index_fill_(0, where, True)
                    start += count
        return {request_id: (trace.cpu(), held.cpu()) for request_id, (trace, held) in out.items()}


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

    def _trunk_capture(self) -> _TrunkCapture | None:
        """The recorder of the engine's trunk hidden states (``_TrunkCapture``), where the
        executor can run in this process: on one GPU. None on more."""
        if self.ENGINE_ARGS.get("tensor_parallel_size", 1) != 1:
            return None
        config = _unittest_module("_nemotron4_reference").TextConfig.from_checkpoint(
            self.MODEL_PATH
        )
        # A step carries at most max_num_tokens tokens: the engine's 8192 unless stated.
        return _TrunkCapture(config.hidden_size, self.ENGINE_ARGS.get("max_num_tokens", 8192))

    def _engine_leg(
        self, prompts, ranks: dict[str, int], capture: _TrunkCapture | None = None
    ) -> tuple[dict[str, _EngineRun], list[dict], dict | None]:
        """All prompts in one batch through the target, streamed; the run's iteration stats; and,
        with a ``capture`` installed, each prompt's trunk trace (``_TrunkCapture.traces``).

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
            runs, request_ids = {}, {}
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
                request_ids[p.name] = future.request_id
            # A statistic can reach the queue after the last response.
            time.sleep(1.0)
            if capture is None:
                stats = llm.get_stats(timeout=2)
            else:
                # An executor in this process answers get_stats from a queue nothing feeds, so
                # the statistics are read as the out-of-process proxy reads them for its RPC.
                executor = llm._executor
                stats = [json.loads(executor._stats_serializer(s)) for s in executor.fetch_stats()]
        traces = None
        if capture is not None:
            by_request = capture.traces(
                {request_ids[p.name]: len(p.token_ids) + len(runs[p.name].output) for p in prompts}
            )
            traces = {p.name: by_request[request_ids[p.name]] for p in prompts}
            capture.steps.clear()
        return runs, stats, traces

    def _load_reference(self, reference):
        """The plain-torch reference on the GPU, once the engine has given it back."""
        # An engine in other processes gives its memory back as they exit; one in
        # this process as soon as it is collected.
        deadline = time.monotonic() + self.RELEASE_SECONDS
        free, _ = torch.cuda.mem_get_info()
        while free < 200 * 2**30 and time.monotonic() < deadline:
            time.sleep(2.0)
            free, _ = torch.cuda.mem_get_info()
        assert free >= 200 * 2**30, (
            f"the reference needs ~190 GiB of the GPU after the engine exits; "
            f"{free / 2**30:.1f} GiB is free"
        )
        return reference.Nemotron4Reference(self.MODEL_PATH)

    def _reference_leg(self, model, prompts, runs: dict[str, _EngineRun], traces) -> dict:
        """The reference teacher-forced over each prompt and the target's output.

        Both MTP-Eagle (the semantics the engine runs) and HF's per-depth MTP. With the engine's
        trunk ``traces``, MTP-Eagle once more over the engine's own trunk hidden states
        (``engine_trunk``; the reference's own stand in at positions the capture missed), and the
        trunk's drift at every position: the relative RMS of the engine's hidden state against
        the reference's (NaN where missed).
        """
        out = {}
        for p in prompts:
            seq = list(p.token_ids) + runs[p.name].output
            ids = torch.tensor(seq, dtype=torch.long, device="cuda")
            with torch.inference_mode():
                hidden = model.forward(ids, logits=None).hidden
                # The reference's own greedy token at every output position, for the report.
                greedy = model.greedy_ids(hidden[len(p.token_ids) - 1 : -1]).tolist()
            entry = dict(
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
            if traces is not None:
                trace, held = (t.cuda() for t in traces[p.name])
                with torch.inference_mode():
                    exact = hidden.float()
                    drift = (trace.float() - exact).norm(dim=-1) / exact.norm(dim=-1)
                    engine_trunk = torch.where(held[:, None], trace, hidden)
                    entry.update(
                        engine_trunk=model.teacher_forced_mtp_acceptance(
                            seq,
                            len(p.token_ids),
                            hidden=engine_trunk,
                            mode="eagle",
                            num_depths=self.MTP_DRAFT_LEN,
                        ),
                        drift=torch.where(held, drift, torch.nan).cpu(),
                        held=held.cpu(),
                        hidden=hidden.cpu(),
                    )
                del trace, held, exact, drift, engine_trunk
            out[p.name] = entry
            del ids, hidden
        return out

    # Random perturbations per flagged step, and how many flagged steps get them
    # (``_perturbation_flips``).
    PERTURBATION_SEEDS = 16
    PERTURBATION_STEPS = 32

    def _drafts_at(self, model, reference, ids, hidden_before, hidden_at, kv1, q) -> list[int]:
        """The reference's MTP-Eagle drafts at q from the given trunk hidden states at q - 1 and q.

        Depth 1 runs over positions q - 1 and q only, attending over the depth-1 keys and values
        ``kv1`` of the positions before them; depths 2 .. D run at q. Teacher-forced as
        ``mtp_draft_hidden`` is, so it reproduces that function's drafts at q from the same states.
        """
        config, weights, embed = model.config, model.mtp, model.embed
        out, base_kv = reference.mtp_layer(
            torch.stack([hidden_before, hidden_at]),
            embed[ids[q : q + 2]],
            weights,
            config,
            (kv1[0][: q - 1], kv1[1][: q - 1]),
        )
        hidden, drafts, extra_kv = out[1:], [], []
        for depth in range(self.MTP_DRAFT_LEN):
            if depth:
                hidden, kv = reference._mtp_eagle_depth(
                    hidden,
                    embed[ids[q + 1 + depth : q + 2 + depth]],
                    weights,
                    config,
                    base_kv,
                    extra_kv,
                )
                extra_kv.append(kv)
            drafts.append(int(model.logits(hidden).argmax()))
        return drafts

    def _perturbation_flips(
        self, model, reference, seq: list[int], ref: dict, q: int, depth: int, r: float
    ):
        """How many of ``PERTURBATION_SEEDS`` random perturbations of the reference's trunk hidden
        state at q - 1 and q, each of relative RMS ``r``, change its MTP-Eagle draft at depth
        index ``depth`` at q; and whether the windowed chain (``_drafts_at``) reproduces the
        reference's own draft there. Information on how well conditioned that decision is.
        """
        ids = torch.tensor(seq, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hidden = ref["hidden"].cuda()
            if "kv1" not in ref:
                # Depth 1 at every position reads the next token's embedding.
                shifted = torch.cat([ids[1:], ids.new_zeros(1)])
                ref["kv1"] = reference.mtp_layer(
                    hidden, model.embed[shifted], model.mtp, model.config
                )[1]
            kv1 = ref["kv1"]
            unperturbed = self._drafts_at(model, reference, ids, hidden[q - 1], hidden[q], kv1, q)
            generator = torch.Generator(device="cuda").manual_seed(q)
            flips = 0
            for _ in range(self.PERTURBATION_SEEDS):
                perturbed = []
                for position in (q - 1, q):
                    row = hidden[position].float()
                    noise = torch.randn(row.shape, generator=generator, device="cuda")
                    noise *= r * row.norm() / noise.norm()
                    perturbed.append((row + noise).to(hidden.dtype))
                drafts = self._drafts_at(model, reference, ids, *perturbed, kv1, q)
                flips += drafts[depth] != unperturbed[depth]
        return flips, unperturbed[depth] == ref["eagle"]["drafts"][depth][q]

    def _recheck(
        self, model, reference, flip: dict, seq: list[int], ref: dict, epsilon, measure_perturbation
    ) -> None:
        """Re-check a flagged step from the engine's own trunk hidden states and record the
        verdict in ``flip`` (``_check_acceptance``, *Flagged steps*); with
        ``measure_perturbation``, also the step's perturbation count."""
        q, accepted, depth = flip["q"], flip["accepted_trt"], flip["depth"] - 1
        if "engine_trunk" not in ref:
            flip.update(
                explained=False,
                reason="not re-checked: the executor ran in other processes, so no trunk "
                "hidden state was captured",
            )
            return
        missing = (~ref["held"][: q + 1]).nonzero().flatten().tolist()
        if missing:
            flip.update(
                explained=False,
                reason=f"not re-checked: the capture missed positions {missing[:4]} "
                f"({len(missing)} up to q)",
            )
            return
        sub = ref["engine_trunk"]
        accepted_sub = _reference_accepted(q, seq, sub["drafts"])
        drift = float(ref["drift"][q])
        verdict = dict(
            accepted_sub=accepted_sub,
            margin_sub_at_depth=round(sub["margins"][depth][q], 4),
            drift=round(drift, 4),
            drift_before=round(float(ref["drift"][q - 1]), 4),
        )
        if measure_perturbation:
            flips, reproduced = self._perturbation_flips(
                model, reference, seq, ref, q, depth, drift
            )
            verdict.update(
                perturbation_flips=f"{flips}/{self.PERTURBATION_SEEDS}",
                perturbation_reproduces_draft=reproduced,
            )
        if accepted_sub == accepted:
            verdict.update(
                explained=True,
                reason=f"explained: on the engine's trunk hidden states the reference accepts "
                f"{accepted_sub}, as the engine does",
            )
        else:
            # The depth where the re-run reference and the engine part decides the step now.
            depth_sub = min(accepted, accepted_sub)
            margin_sub = sub["margins"][depth_sub][q]
            explained = margin_sub < epsilon[depth_sub]
            verdict.update(
                depth_sub=depth_sub + 1,
                margin_sub=round(margin_sub, 4),
                explained=explained,
                reason=f"{'explained' if explained else 'unexplained'}: on the engine's trunk "
                f"hidden states the reference accepts {accepted_sub} against the engine's "
                f"{accepted}, decided at depth {depth_sub + 1} by {margin_sub:.4f} "
                f"{'<' if explained else '>='} epsilon {epsilon[depth_sub]:.4f}",
            )
        flip.update(verdict)

    def _check_acceptance(self, monkeypatch) -> None:
        """The acceptance length the target measures matches the reference's.

        Leg 1 runs the fixed prompt set (6 short prompts x 256 tokens, 4 of ~8K
        tokens x 1024) through the target in one batch with MTP-Eagle at the
        class's draft length D (``MTP_DRAFT_LEN``), the overlap scheduler and
        CUDA graphs on: AL_trt, read as the target's class reads it
        (``_engine_al``). Leg 2 runs the reference teacher-forced over each
        prompt and the target's own output, D depths deep: AL_ref, its greedy
        walk pooled with the same estimator.

        **At the engine's own positions.** The streamed responses give the
        engine's walk step by step, so the reference scores each of the
        engine's steps where the engine took it. The two sides' accepted counts
        differ at a step only if the deciding draft's outcome flips, which needs
        some token to gain at least the reference's margin there
        (``draft_scores``). ``epsilon`` bounds that gain per depth: the target
        MTP layer's logit error against the reference's on the same trunk
        hidden states (``MTP_LOGIT_ERROR``, the MTP module gate's measurement)
        plus the band on the target trunk's logits against the reference's
        (``TRUNK_LOGIT_BAND``). A step whose count differs where the reference's
        margin is at least epsilon is **flagged**.

        **Flagged steps.** epsilon covers the trunk's drift in logit space, but
        a draft at depth d reads the trunk's hidden state through d applications
        of the MTP layer, and where the reference's chain is ill-conditioned
        (an MoE routing switch at a deep depth, say) an ordinary drift of the
        trunk's hidden state moves the deep draft by more than its logits move.
        So every flagged step is re-checked from the engine's own inputs: the
        reference's MTP-Eagle chain runs again, teacher-forced, over the trunk
        hidden states the engine itself computed (``_TrunkCapture``) at every
        position up to the step's drafting position q. Those are what the
        engine's MTP layer read: at q itself, and through the depth-1 keys and
        values it wrote at every earlier position. The two sides then differ
        only in the MTP layer's own arithmetic, which ``MTP_LOGIT_ERROR``
        bounds on identical trunk inputs, and epsilon is that bound plus the
        trunk's band:

        - the step is **explained** by the trunk's drift if the reference, so
          re-run, accepts what the engine accepted, or decides the step by less
          than epsilon;
        - otherwise it **fails**: from the same inputs, the target's draft path
          computes something other than the checkpoint's MTP layer.

        The re-check takes the engine's trunk as it is: the trunk's own
        accuracy is the other gates' (check_goldens, CUDA-graph identity,
        GSM8K). A fault anywhere between the trunk's hidden state and the
        drafts still fails it (``...MtpAcceptanceFault`` plants one).

        Each flagged step is reported with the trunk's drift at q and q - 1
        (the relative RMS of the engine's hidden state against the
        reference's, and its rank among the run's positions) and its margin
        before and after the re-run; the first ``PERTURBATION_STEPS`` also with
        how many of 16 random perturbations of the reference's trunk hidden
        state at q - 1 and q, as large as the drift at q, change the deciding
        draft (``_perturbation_flips``; information on how ill-conditioned the
        decision is). The capture needs the executor in this process, which
        TensorRT-LLM runs for one GPU only; on more GPUs a flagged step is not
        re-checked and fails.

        **Tolerance.** A flip at depth d of D moves the step's count by at most
        D - d + 1: at D = 2, by 2 at depth 1 and 1 at depth 2. So over the
        reference's n steps:

            tol = (sum over d of (D - d + 1) * (k_d + x_d) + D * E) / n

        which at D = 2 is (2 * (k1 + x1) + (k2 + x2) + 2 * E) / n. k_d counts
        the reference walk's steps whose first d - 1 drafts matched, whose
        depth-d margin is below epsilon and whose earlier margins are not (k1:
        the depth-1 margin is below it). x_d counts the flagged steps the
        trunk's drift explains, by their deciding depth. E counts the steps
        AL_trt holds that the reference walk cannot pair, each moving a mean of
        n steps by at most D / n:

        - the engine's last step of an output whose window passes the last
          token, which the reference cannot score (at most one per request);
        - the steps AL_trt counts beyond the requests' own (``_engine_al``).

        AL_trt >= 2.0 is a tripwire for a collapsed draft path. The reference's
        AL with HF's per-depth MTP (``mode="hf"``) is reported beside it.
        """
        draft_len = self.MTP_DRAFT_LEN
        monkeypatch.setenv("LLM_MODELS_ROOT", llm_models_root())
        prompts_module = _unittest_module("_nemotron4_prompts")
        reference = _unittest_module("_nemotron4_reference")
        prompts = prompts_module.prompt_set(prompts_module.load_tokenizer(self.MODEL_PATH))
        ranks = self._ranks(prompts)

        capture = self._trunk_capture()
        if capture is not None:
            capture.install(monkeypatch)
        runs, stats, traces = self._engine_leg(prompts, ranks, capture)
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

        epsilon = self._epsilon()
        assert len(epsilon) == draft_len, f"epsilon {epsilon} for {draft_len} draft depths"
        rows, flips_all, flagged = [], [], []
        pooled = dict(
            ref_accepted=[],
            ref_hf_accepted=[],
            trt_accepted=[],
            k=[0] * draft_len,
            x=[0] * draft_len,
            agree_ref=0,
            agree_engine_trunk=0,
            paired=0,
        )
        boundary_total = 0
        drifts = []
        model = self._load_reference(reference)
        try:
            refs = self._reference_leg(model, prompts, runs, traces)
            for p in prompts:
                run, ref = runs[p.name], refs[p.name]
                eagle, hf = ref["eagle"], ref["hf"]
                seq = list(p.token_ids) + run.output
                walk = walks[p.name]
                engine_trunk = ref.get("engine_trunk")

                # The reference's own walk and its near-ties (the tolerance): a step's count can
                # flip at depth d + 1 only when its first d drafts matched (a >= d).
                k = [0] * draft_len
                for q, a in zip(eagle["positions"], eagle["accepted"]):
                    for depth in range(min(a + 1, draft_len)):
                        if eagle["margins"][depth][q] < epsilon[depth]:
                            k[depth] += 1
                            break

                # The engine's walk, each step scored by the reference where the engine took it.
                flips, boundary, matched_trt, matched_ref, agree_engine_trunk = [], 0, [], [], 0
                for step, (q, a) in enumerate(walk):
                    a_ref = _reference_accepted(q, seq, eagle["drafts"])
                    if a_ref is None:
                        boundary += 1
                        continue
                    matched_trt.append(a)
                    matched_ref.append(a_ref)
                    if engine_trunk is not None:
                        agree_engine_trunk += (
                            _reference_accepted(q, seq, engine_trunk["drafts"]) == a
                        )
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
                        self._recheck(
                            model,
                            reference,
                            flip,
                            seq,
                            ref,
                            epsilon,
                            measure_perturbation=len(flagged) < self.PERTURBATION_STEPS,
                        )
                        flagged.append(flip)
                        if flip["explained"]:
                            pooled["x"][depth] += 1
                if "drift" in ref:
                    # From the prompt's last position to the last one the engine's last verify
                    # step accepted: every position whose captured state is final. The drafts
                    # that step rejected never run again.
                    last_q, last_accepted = walk[-1]
                    drifts.append(ref["drift"][len(p.token_ids) - 1 : last_q + last_accepted + 2])

                al_trt_request = _al([a for _, a in walk])
                agree_ref = sum(t == r for t, r in zip(matched_trt, matched_ref))
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
                    agree_ref=f"{agree_ref}/{len(matched_trt)}",
                    **(
                        {"agree_engine_trunk": f"{agree_engine_trunk}/{len(matched_trt)}"}
                        if engine_trunk is not None
                        else {}
                    ),
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
                pooled["agree_ref"] += agree_ref
                pooled["agree_engine_trunk"] += agree_engine_trunk
                pooled["paired"] += len(matched_trt)
                flips_all += flips
                boundary_total += boundary
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # The trunk's drift over the positions the engine drafted from and accepted, and each
        # flagged step's rank among them.
        drift_summary = None
        if drifts:
            drift = torch.cat(drifts)
            drift = drift[~drift.isnan()]
            quantiles = drift.quantile(torch.tensor([0.5, 0.9, 0.99, 1.0])).tolist()
            drift_summary = dict(
                positions=drift.numel(), **dict(zip(("median", "p90", "p99", "max"), quantiles))
            )
            drift_summary = {key: round(v, 4) for key, v in drift_summary.items()}
            for flip in flagged:
                if "drift" in flip:
                    flip["drift_rank"] = f"{int((drift > flip['drift']).sum()) + 1}/{drift.numel()}"

        n = len(pooled["ref_accepted"])
        al_ref = _al(pooled["ref_accepted"])
        al_ref_hf = _al(pooled["ref_hf_accepted"])
        al_trt_requests = _al(pooled["trt_accepted"])
        request_steps = len(pooled["trt_accepted"])
        unpaired = boundary_total + extra_steps
        weighted = " + ".join(
            f"{draft_len - depth} * ({pooled['k'][depth]} + {pooled['x'][depth]})"
            for depth in range(draft_len)
        )
        tol = (
            sum(
                (draft_len - depth) * (pooled["k"][depth] + pooled["x"][depth])
                for depth in range(draft_len)
            )
            + draft_len * unpaired
        ) / n
        unexplained = [f for f in flagged if not f["explained"]]
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
            **{f"x{depth + 1}": pooled["x"][depth] for depth in range(draft_len)},
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
            flagged=len(flagged),
            explained=len(flagged) - len(unexplained),
            unexplained=len(unexplained),
            agree_ref=f"{pooled['agree_ref']}/{pooled['paired']}",
            **(
                dict(
                    agree_engine_trunk=f"{pooled['agree_engine_trunk']}/{pooled['paired']}",
                    trunk_drift=drift_summary,
                    captured_steps=capture.captured_steps,
                    overflow_steps=capture.overflow_steps,
                )
                if capture is not None
                else {}
            ),
            mixed_steps_by_rank=dict(sorted(mixed_steps.items())),
        )
        print(f"[modeling_v2] mtp acceptance pooled {json.dumps(summary)}")
        for flip in flagged[: self.PERTURBATION_STEPS]:
            print(f"[modeling_v2] mtp acceptance flagged {json.dumps(flip)}")
        self.report = dict(summary=summary, prompts=rows, flagged=flagged)
        report_path = os.environ.get(REPORT_ENV)
        if report_path:
            engine = {name: asdict(run) for name, run in runs.items()}
            with open(report_path, "w") as f:
                json.dump(dict(**self.report, engine=engine), f, indent=1)

        assert not unexplained, (
            f"{len(unexplained)} of {len(flagged)} flagged verify steps (the engine accepts a "
            f"different number of drafts than the reference, whose deciding draft is no near-tie: "
            f"margin >= epsilon {[round(e, 4) for e in epsilon]}) are not explained by the "
            f"trunk's drift: the target's draft path computes something other than the "
            f"checkpoint's MTP layer there. First ones: {unexplained[:5]}"
        )
        assert al_trt >= self.MIN_ACCEPTANCE_LENGTH, (
            f"acceptance length {al_trt:.4f} < {self.MIN_ACCEPTANCE_LENGTH}: the draft path "
            f"barely drafts anything the target accepts"
        )
        assert abs(al_trt - al_ref) <= tol, (
            f"AL_trt {al_trt:.4f} vs AL_ref {al_ref:.4f}: |delta| {abs(al_trt - al_ref):.4f} > "
            f"tol {tol:.4f} = ({weighted} + {draft_len} * {unpaired}) / {n}"
        )
