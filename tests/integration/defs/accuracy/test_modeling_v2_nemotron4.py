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
"""Whole-model gates for the nemotron4 modeling_v2 targets.

One file per model family, beside the other accuracy suites rather than inside
test_llm_api_pytorch.py: these gate a parallel implementation, and reading them
next to the built-in model's tests would invite treating one as a variant of
the other.

Every gate of every nemotron4 target is a class here, its topology and serve
constants stated in it:

* routing on the checkpoint's own config (``...Routing``);
* GSM8K, the accuracy anchor, with MTP off and on (``...Tp1``, ``...Dep4``,
  ``...Tp4``);
* token identity with CUDA graphs on and off (``...Identity``, criterion A);
* MTP on against MTP off, parting only at a near-tie (``...MtpIdentity``,
  criterion B(1));
* the MTP acceptance length against the checkpoint's own MTP layer
  (``...MtpAcceptance``, criterion B(2)).

The legs and checks the identity and acceptance gates share are the harnesses
beside this file: ``_nemotron4_cuda_graph_identity.py``,
``_nemotron4_mtp_identity.py`` and ``_nemotron4_mtp_acceptance.py``. The MTP
layer's module accuracy (criterion B(3)), the decode invariance the CUDA-graph
identity gates rest on, and the verify's measured deviation from decode, which
the MTP gates' bands rest on, are unit tests, under
tests/unittest/_torch/modeling_v2/models/nemotron4/.

Every test here passes ``modeling_v2="require"`` to ``LLM``. TensorRT-LLM has no
built-in implementation of this architecture, so there is nothing to fall back
to: under ``"auto"`` a deployment routing declines would find no model class at
all, while ``"require"`` makes the decline raise with routing's reason. The
same absence means there is no ``modeling_v2="off"`` oracle either; the anchor
in references/gsm8k.yaml is the checkpoint's own HuggingFace implementation,
measured under the protocol these tests run.
"""

import dataclasses
from collections import Counter, defaultdict

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import (
    MODELING_V2_VALIDATE_ENV,
    modeling_v2_resolve,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.speculative.utils import (
    resolve_mtp_checkpoint_source,
    update_spec_config_from_model_config,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping

from ..conftest import llm_models_root
from ._nemotron4_cuda_graph_identity import _CudaGraphIdentity, _Request
from ._nemotron4_mtp_acceptance import _al, _MtpAcceptance
from ._nemotron4_mtp_identity import _MtpIdentity
from ._nemotron4_mtp_identity import _Request as _MtpRequest
from .accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    acceptance_length_from_iteration_stats,
    compute_acceptance_length,
)

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# The filter the anchor was measured on. Unset, the evaluator averages every
# metric GSM8K reports, which means the mean of strict-match and
# flexible-extract -- two numbers measuring different things.
_SCORES_FILTER = {"scores_filter": "exact_match,flexible-extract"}


# Each target's layout as the engine's mapping states it, and the class routing returns.
_TOPOLOGIES = {
    "tp1": ({}, "ModelingV2Nemotron4NanoPreviewSm103Tp1"),
    "dep4": (
        dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True),
        "ModelingV2Nemotron4NanoPreviewSm103Dep4",
    ),
    "tp4": (
        dict(world_size=4, tp_size=4, moe_ep_size=1, moe_tp_size=4),
        "ModelingV2Nemotron4NanoPreviewSm103Tp4",
    ),
    "tp2": (dict(world_size=2, tp_size=2), None),
}


class TestModelingV2Nemotron4NanoPreviewRouting:
    """Routing on the checkpoint's own config, resolved as the model loader resolves it.

    The routing unit test (tests/unittest/_torch/modeling_v2/test_modeling_v2_routing.py)
    drives every decision with a synthetic config of this checkpoint's shape. This reads the
    shipped config.json through the checkpoint's own configuration_nemotron4.py and resolves
    the speculative config against it, as the model loader does before it routes, so a
    config that drifted from the shape the targets were written for fails here. No weights
    are read and no engine is built.
    """

    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    def _resolve(self, topology: str, spec_config) -> type:
        config = ModelConfig.from_pretrained(self.MODEL_PATH, trust_remote_code=True)
        if spec_config is not None:
            # What the model loader does to the speculative config before it routes.
            resolve_mtp_checkpoint_source(spec_config, self.MODEL_PATH)
            update_spec_config_from_model_config(spec_config, config.pretrained_config)
        mapping = Mapping(**_TOPOLOGIES[topology][0])
        config = dataclasses.replace(config, mapping=mapping, spec_config=spec_config)
        # Routing reads only the mode off the arguments.
        llm_args = TorchLlmArgs(model=self.MODEL_PATH, modeling_v2="require")
        return modeling_v2_resolve(config, llm_args)

    @skip_not_sm103
    @pytest.mark.parametrize("mtp", [False, True], ids=["mtp_off", "mtp2"])
    @pytest.mark.parametrize("topology", ["tp1", "dep4", "tp4"])
    def test_each_target_claims_its_deployment(self, topology, mtp):
        spec_config = MTPDecodingConfig(max_draft_len=2) if mtp else None
        assert self._resolve(topology, spec_config).__name__ == _TOPOLOGIES[topology][1]

    @skip_not_sm103
    @pytest.mark.parametrize(
        "topology, make_spec_config, reason",
        [
            ("tp2", lambda: None, "tp2 is not implemented"),
            (
                "tp4",
                lambda: MTPDecodingConfig(max_draft_len=2, use_mtp_vanilla=True),
                "use_mtp_vanilla selects vanilla MTP",
            ),
            (
                "tp1",
                lambda: MTPDecodingConfig(max_draft_len=3),
                "max_draft_len=3: the target runs 2 drafts",
            ),
            (
                "dep4",
                lambda: MTPDecodingConfig(max_draft_len=2, draft_len_schedule={4: 2, 64: 1}),
                "per-step all-gather and reduce-scatter would not meet",
            ),
        ],
        ids=["tp2", "vanilla_mtp", "draft_len_3", "draft_len_schedule_under_attention_dp"],
    )
    def test_other_deployments_are_declined_with_the_reason(
        self, topology, make_spec_config, reason
    ):
        with pytest.raises(
            ValueError, match="no target fits 'Nemotron4ForConditionalGeneration'"
        ) as e:
            self._resolve(topology, make_spec_config())
        assert reason in str(e.value)


class TestModelingV2Nemotron4NanoPreviewSm103Tp1(LlmapiAccuracyTestHarness):
    """nemotron4-nano-preview (text model) / sm_103 / tp1, identity and the mtp2 variant."""

    MODEL_NAME = "Nemotron4/nemotron4-nano-preview"
    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The config's max_position_embeddings is 1,048,576, which the engine would
    # otherwise take as max_seq_len and size the KV-cache estimate from. This
    # is the context the target serves: 8192 + 1024 tokens plus draft slack.
    MAX_SEQ_LEN = 9344

    # The decode batch each leg serves, with CUDA graphs captured with padding for
    # every batch size up to it: the serve configs of tp1's 8K/1K Pareto sweeps,
    # 896 with MTP off and 448 with MTP on, where the verify's per-position
    # states (85 MB a row) leave the KV cache room for about half the batch. The
    # 1319 questions are submitted at once, so each leg fills its batch and then
    # drains it through the smaller graphs.
    MAX_BATCH_SIZE = 896
    MTP_MAX_BATCH_SIZE = 448

    # The MTP leg's kv-cache fraction. The verify's per-position GDP and conv
    # states (~36 GiB at max_batch_size 448) are allocated before the engine's
    # profiling forward, and the cache pools take this share of the memory
    # left after it. At the default 0.9 (44.96 GiB of pools) the first
    # generation steps found too little left for one of their buffers, a
    # recovered allocator OOM; 0.8 leaves about 5 GiB more.
    MTP_KV = KvCacheConfig(free_gpu_memory_fraction=0.8)

    # The checkpoint's trained draft depth: its one MTP layer serves both
    # depths. MTPDecodingConfig defaults to a depth of 1, so it is stated.
    MTP_DRAFT_LEN = 2

    # There is deliberately no standalone identity gsm8k case. The paired test
    # below evaluates the identity config as its first leg, and
    # ``task.evaluate`` asserts accuracy against the reference on the way past,
    # so a separate one would gate nothing new for one more engine boot.

    def _llm(self, max_batch_size: int, **variant) -> LLM:
        # Built fresh for each leg, so nothing the engine resolves into one
        # leg's config objects reaches the other.
        #
        # The target checks its step contract on the first forward and only
        # when asked. The contract is about metadata this engine prepared, so
        # it is checked in a run that prepared it. It travels as
        # ``env_overrides`` because the check runs on the rank.
        #
        # trust_remote_code: the checkpoint's config class is its own
        # configuration_nemotron4.py. The overlap scheduler is on by default;
        # it is stated because the gate is defined with it on.
        return LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            trust_remote_code=True,
            max_seq_len=self.MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=max_batch_size),
            **variant,
        )

    @skip_not_sm103
    def test_gsm8k_identity_vs_mtp2(self, mocker):
        """The identity accuracy gate, and the gate on MTP not moving it.

        Sampling is greedy, so a draft is accepted only where it equals the
        target's own next token: MTP on emits the target model's greedy text.
        The two scores can differ only where a step of another shape tips a
        near-tie the other way. The verify runs 3 tokens per request through
        the trunk, and the two legs batch their requests differently.

        Both legs assert accuracy against the registered reference as they
        run, and the MTP row there carries the same anchor, so MTP on is held
        to the bar MTP off is held to. This test is therefore the identity
        gate as well as the comparison.

        Both measurements are taken in this one test on purpose: paired, the
        variance of the session is common to both and cancels.

        The acceptance length is reported, not gated. GSM8K has no reference
        acceptance for this checkpoint, and an anchor taken from the target's
        own number would make the gate self-referential. Computing it does
        assert that the MTP leg drafted at all.
        """
        # The stock protocol, as the anchor was measured: 5-shot, no chat
        # template, the default 256 output tokens. The checkpoint's chat
        # template is not usable for this: in chat form the model never emits
        # <|im_end|>, so an answer ends only at a stop word or the token cap,
        # and its thinking mode answers without reasoning. Only the filter is
        # patched.
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        task = GSM8K(self.MODEL_NAME)

        with self._llm(self.MAX_BATCH_SIZE) as llm:
            identity = task.evaluate(llm)

        # Every iteration's stats are kept (``max_stats_len=-1``), so the
        # acceptance length covers the whole run rather than its last 1000
        # iterations.
        with self._llm(
            self.MTP_MAX_BATCH_SIZE,
            speculative_config=MTPDecodingConfig(max_draft_len=self.MTP_DRAFT_LEN),
            kv_cache_config=self.MTP_KV,
            enable_iter_perf_stats=True,
            max_stats_len=-1,
        ) as llm:
            mtp2 = task.evaluate(llm)
            acceptance_length = compute_acceptance_length(llm)

        delta = mtp2 - identity
        print(
            f"[modeling_v2] gsm8k identity={identity:.4f} mtp2={mtp2:.4f} "
            f"delta={delta:+.4f} acceptance_length={acceptance_length:.4f}"
        )
        # 2 sigma at the 1.36 stderr this benchmark reports at n=1319 for a
        # score near 57%.
        assert abs(delta) < 2.7, (
            f"MTP moved gsm8k by {delta:+.4f} (identity={identity:.4f}, "
            f"mtp2={mtp2:.4f}); greedy MTP emits the target's own tokens, so "
            f"this is not near-tie noise"
        )


def _record_requests(llm: LLM, mocker) -> list:
    """Every request ``llm`` is given from here on, as the output its caller waits on."""
    requests = []
    generate_async = llm.generate_async

    def generate_and_record(*args, **kwargs):
        requests.append(generate_async(*args, **kwargs))
        return requests[-1]

    mocker.patch.object(llm, "generate_async", generate_and_record)
    return requests


def _acceptance_length(requests: list) -> float:
    """Tokens per verify step over ``requests``: (verify steps + accepted drafts) / verify steps.

    It is ``compute_acceptance_length``'s estimator, summed over each request's
    own counters rather than over the iteration statistics: under attention DP
    every rank's row of an iteration carries rank 0's speculative-decoding
    counts, so the iteration statistics measure rank 0's requests alone.
    Computing it asserts that the MTP leg drafted at all.
    """
    # Every verify step drafts position 0.
    steps = sum((r.per_pos_drafted or [0])[0] for r in requests)
    accepted = sum(sum(r.per_pos_accepted or ()) for r in requests)
    assert steps, f"None of the {len(requests)} requests verified a draft"
    return (steps + accepted) / steps


class TestModelingV2Nemotron4NanoPreviewSm103Dep4(LlmapiAccuracyTestHarness):
    """nemotron4-nano-preview (text model) / sm_103 / dep4, identity and the mtp2 variant."""

    MODEL_NAME = "Nemotron4/nemotron4-nano-preview"
    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The parallel topology the dep4 target implements: attention DP over the
    # 4 GPUs of one node, the routed experts split four ways. Routing derives
    # the target from these, and the target asserts them against the mapping
    # the engine built.
    DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)

    # The context the target serves, as on tp1: 8192 + 1024 tokens plus draft
    # slack, stated because the config's max_position_embeddings is 1,048,576.
    MAX_SEQ_LEN = 9344

    # The serve configs of dep4's 8K/1K Pareto sweeps, per rank: under
    # attention DP every rank schedules its own requests, and the CUDA graphs
    # are captured per rank, with padding, for every batch size up to the
    # leg's batch, 1024 with MTP off and 512 with MTP on. A step holds at most
    # 8192 tokens a rank plus a row per request, of 1 + D tokens with MTP:
    # 9216 and 9728 tokens, up to 38912 gathered rows into each expert window,
    # inside the routed MoE's certified collectives and window calls. A 5-shot
    # GSM8K prompt (at most ~1.5K tokens) fits one step, so prompts are not
    # chunked. The 1319 questions, spread over the four ranks, are all in
    # flight at once and drain through the smaller graphs.
    MAX_BATCH_SIZE = 1024
    MAX_NUM_TOKENS = 9216
    MTP_MAX_BATCH_SIZE = 512
    MTP_MAX_NUM_TOKENS = 9728

    # The checkpoint's trained draft depth: its one MTP layer serves both
    # depths. MTPDecodingConfig defaults to a depth of 1, so it is stated.
    MTP_DRAFT_LEN = 2

    # There is deliberately no standalone identity gsm8k case. The paired test
    # below evaluates the identity config as its first leg, and
    # ``task.evaluate`` asserts accuracy against the reference on the way past,
    # so a separate one would gate nothing new for one more 4-rank engine boot.

    def _llm(self, max_batch_size: int, max_num_tokens: int, **variant) -> LLM:
        # Built fresh for each leg, so nothing the engine resolves into one
        # leg's config objects reaches the other.
        #
        # The step contract is checked on every rank's first forward, as on
        # tp1; the switch travels as ``env_overrides`` because the check runs
        # on the ranks.
        return LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            trust_remote_code=True,
            max_seq_len=self.MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=max_batch_size),
            **self.DEP4,
            **variant,
        )

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_identity_vs_mtp2(self, mocker):
        """The identity accuracy gate on 4 ranks, and the gate on MTP not moving it.

        It holds dep4 to the bar tp1 is held to: the anchor is a property of
        the checkpoint, and the topology only changes where each request runs
        and in which order the routed experts' partial sums meet. The target
        adds the four ranks' fp32 partial sums before its one rounding to
        bf16, which is the rounding HF makes.

        Sampling is greedy, so a draft is accepted only where it equals the
        target's own next token: MTP on emits the target model's greedy text.
        A verify attends through thop_attention, so its tokens sit near the
        bits a decode step gives them rather than on them, and the two scores
        can differ where a verify, or a prefill or mixed step, tips a near-tie
        the other way. Under attention DP those steps change between any two
        engines, MTP or not: which rank serves a request, and beside which
        others, depends on when it arrives.

        Both legs assert accuracy against the registered reference as they
        run, and the MTP row there carries the same anchor, so MTP on is held
        to the bar MTP off is held to. Both are measured in this one test, so
        the variance of the session is common to both and cancels.

        The acceptance length is reported, not gated, as on tp1: GSM8K has no
        reference acceptance for this checkpoint. It is taken from the
        requests' own counters (``_acceptance_length``).
        """
        # The stock protocol, as the anchor was measured: 5-shot, no chat
        # template, the default 256 output tokens. Only the filter is patched.
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        task = GSM8K(self.MODEL_NAME)

        with self._llm(self.MAX_BATCH_SIZE, self.MAX_NUM_TOKENS) as llm:
            identity = task.evaluate(llm)

        # The default kv-cache fraction is kept, as dep4's MTP serve config
        # keeps it. A rank runs every recurrent layer, so its verify states are
        # tp1's (~41 GiB at 512 requests), but it holds a quarter of the routed
        # experts: the quota left with MTP, ~153 GiB a rank, is far more than
        # the state slots and a full batch of GSM8K requests take, and the
        # tenth of the free memory the fraction leaves, ~17 GiB, is over three
        # times what tp1's MTP leg would keep at the default (``MTP_KV``).
        with self._llm(
            self.MTP_MAX_BATCH_SIZE,
            self.MTP_MAX_NUM_TOKENS,
            speculative_config=MTPDecodingConfig(max_draft_len=self.MTP_DRAFT_LEN),
        ) as llm:
            requests = _record_requests(llm, mocker)
            mtp2 = task.evaluate(llm)
            acceptance_length = _acceptance_length(requests)

        delta = mtp2 - identity
        print(
            f"[modeling_v2] gsm8k dep4 identity={identity:.4f} mtp2={mtp2:.4f} "
            f"delta={delta:+.4f} acceptance_length={acceptance_length:.4f} "
            f"over {len(requests)} requests"
        )
        # 2 sigma at the 1.36 stderr this benchmark reports at n=1319 for a
        # score near 57%, as on tp1.
        assert abs(delta) < 2.7, (
            f"MTP moved gsm8k by {delta:+.4f} on dep4 (identity={identity:.4f}, "
            f"mtp2={mtp2:.4f}); greedy MTP emits the target's own tokens, so "
            f"this is not near-tie noise"
        )


class TestModelingV2Nemotron4NanoPreviewSm103Tp4(LlmapiAccuracyTestHarness):
    """nemotron4-nano-preview (text model) / sm_103 / tp4, identity and the mtp2 variant."""

    MODEL_NAME = "Nemotron4/nemotron4-nano-preview"
    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The parallel topology the tp4 target implements: tensor parallel over the
    # 4 GPUs of one node without attention DP, the mixers' heads and every
    # MLP's and routed expert's intermediate width split four ways. Routing
    # derives the target from these, and the target asserts them against the
    # mapping the engine built.
    TP4 = dict(tensor_parallel_size=4, moe_tensor_parallel_size=4)

    # The context the target serves, as on tp1: 8192 + 1024 tokens plus draft
    # slack, stated because the config's max_position_embeddings is 1,048,576.
    MAX_SEQ_LEN = 9344

    # The batch of tp4's serve config (its MTP 8K/1K Pareto sweep), for both
    # legs. Every rank runs every request, so this is the engine's batch, as on
    # tp1, and the CUDA graphs are captured with padding for every batch size
    # up to it on each rank. The 1319 questions are submitted at once, so the
    # run fills the batch and then drains it through the smaller graphs.
    MAX_BATCH_SIZE = 512

    # Every step's tokens, for which the all-reduce workspace is sized: 8192
    # with MTP off, and with MTP the serve config's 8192 plus 1 + D tokens a
    # request. A 5-shot GSM8K prompt (at most ~1.5K tokens) fits in one step,
    # so prompts are not chunked.
    MAX_NUM_TOKENS = 8192
    MTP_MAX_NUM_TOKENS = 9728

    # The checkpoint's trained draft depth: its one MTP layer serves both
    # depths. It is also a depth at which a tp4 rank's GDP verify gives every
    # token the bits decode gives it (gdp_verify's note). MTPDecodingConfig
    # defaults to a depth of 1, so it is stated.
    MTP_DRAFT_LEN = 2

    # There is deliberately no standalone identity gsm8k case. The paired test
    # below evaluates the identity config as its first leg, and
    # ``task.evaluate`` asserts accuracy against the reference on the way past,
    # so a separate one would gate nothing new for one more 4-rank engine boot.

    def _llm(self, max_num_tokens: int, **variant) -> LLM:
        # Built fresh for each leg, so nothing the engine resolves into one
        # leg's config objects reaches the other.
        #
        # The step contract is checked on every rank's first forward, as on
        # tp1; the switch travels as ``env_overrides`` because the check runs
        # on the ranks.
        return LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            trust_remote_code=True,
            max_seq_len=self.MAX_SEQ_LEN,
            max_batch_size=self.MAX_BATCH_SIZE,
            max_num_tokens=max_num_tokens,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE
            ),
            **self.TP4,
            **variant,
        )

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_identity_vs_mtp2(self, mocker):
        """The identity accuracy gate on 4 ranks, and the gate on MTP not moving it.

        It holds tp4 to the bar tp1 is held to: the anchor is a property of
        the checkpoint, and the topology only changes where the row-parallel
        projections' and the routed experts' sums are rounded. Each rank
        rounds its partial to bf16, and the all-reduce adds the four in fp32,
        in rank order, and rounds once, so every rank holds the same bits and
        forms the same logits.

        Sampling is greedy, so a draft is accepted only where it equals the
        target's own next token: MTP on emits the target model's greedy text.
        A verify attends through thop_attention, so its tokens sit near the
        bits a decode step gives them rather than on them, and the two scores
        can differ where a verify, or a prefill or mixed step, which the two
        legs batch differently, tips a near-tie the other way.

        Both legs assert accuracy against the registered reference as they
        run, and the MTP row there carries the same anchor, so MTP on is held
        to the bar MTP off is held to. Both are measured in this one test, so
        the variance of the session is common to both and cancels.

        The acceptance length is reported, not gated, as on tp1: GSM8K has no
        reference acceptance for this checkpoint. Computing it does assert
        that the MTP leg drafted at all.
        """
        # The stock protocol, as the anchor was measured: 5-shot, no chat
        # template, the default 256 output tokens. Only the filter is patched.
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        task = GSM8K(self.MODEL_NAME)

        with self._llm(self.MAX_NUM_TOKENS) as llm:
            identity = task.evaluate(llm)

        # The default kv-cache fraction is kept. A rank's verify states are a
        # quarter of tp1's (~10 GiB at max_batch_size 512), so the quota left
        # with MTP, ~190 GiB a rank against ~203 GiB without, is far more than
        # a full batch of GSM8K requests takes.
        #
        # Every iteration's stats are kept (``max_stats_len=-1``), so the
        # acceptance length covers the whole run rather than its last 1000
        # iterations.
        with self._llm(
            self.MTP_MAX_NUM_TOKENS,
            speculative_config=MTPDecodingConfig(max_draft_len=self.MTP_DRAFT_LEN),
            enable_iter_perf_stats=True,
            max_stats_len=-1,
        ) as llm:
            mtp2 = task.evaluate(llm)
            acceptance_length = compute_acceptance_length(llm)

        delta = mtp2 - identity
        print(
            f"[modeling_v2] gsm8k tp4 identity={identity:.4f} mtp2={mtp2:.4f} "
            f"delta={delta:+.4f} acceptance_length={acceptance_length:.4f}"
        )
        # 2 sigma at the 1.36 stderr this benchmark reports at n=1319 for a
        # score near 57%, as on tp1.
        assert abs(delta) < 2.7, (
            f"MTP moved gsm8k by {delta:+.4f} on tp4 (identity={identity:.4f}, "
            f"mtp2={mtp2:.4f}); greedy MTP emits the target's own tokens, so "
            f"this is not near-tie noise"
        )


class TestModelingV2Nemotron4NanoPreviewSm103Tp1Identity(_CudaGraphIdentity):
    """nemotron4-nano-preview (text model) / sm_103 / tp1: CUDA graphs on against off."""

    # The serve config tp1's MTP-off 8K/1K Pareto sweep measured: an
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

    # The batch of dep4's MTP-off 8K/1K serve config: an explicit max_seq_len and
    # 1024 requests per rank with a CUDA graph for every padded size up to 1024,
    # so the CUDA-graph leg carries every graph the benchmark replays. Every step
    # holds at most 2048 tokens a rank, so the ~8K prompts prefill in 2048-token
    # chunks, which is what makes each rank's batches the same in both legs
    # (`_batch_order`). The state pool is sized for 9.2K-token requests
    # (avg_seq_len), as the 8K/1K benchmark's is.
    MAX_SEQ_LEN = 9344
    MAX_BATCH_SIZE = 1024
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


class TestModelingV2Nemotron4NanoPreviewSm103Tp1MtpIdentity(_MtpIdentity):
    """nemotron4-nano-preview (text model) / sm_103 / tp1: MTP-Eagle on against off."""

    # How far tp1's verify can move a gap between MTP off's top-1 and another of
    # its top tokens: the drift band of tp1's verify test
    # (test_modeling_v2_verify_invariance.py, DRIFT_GAP_BAND), which follows each
    # fixed prompt alone for the tokens this gate generates, MTP on along MTP
    # off's own tokens, and measured at most 0.7819 over 6132 positions; plus a
    # quarter.
    NEAR_TIE_BAND = 0.98


class TestModelingV2Nemotron4NanoPreviewSm103Dep4MtpIdentity(_MtpIdentity):
    """nemotron4-nano-preview (text model) / sm_103 / dep4: MTP-Eagle on against off, on 4 ranks.

    Attention DP over the 4 GPUs of one node: every rank runs its own
    requests, and the ranks meet in every routed MoE block, the MTP layer's
    included. Each prompt is pinned to a rank, the same one in both legs: the
    MoE's reduce-scatter adds a token's four partial sums in an order that ends
    at the token's own rank, so the rank is part of what a token computes.

    Alone, the prompt's rank runs it and the other three run the attention-DP
    dummy, a generation row of 1 + D tokens with MTP on and of one token with
    it off. That changes only how many rows every routed MoE block gathers
    from the other ranks, and the target gives a token the same bits whatever
    those rows are: the expert window computes each row by itself, and the
    reduce-scatter's order does not depend on the ranks' counts. What parts
    the legs is the verify's attention, as on tp1, by up to NEAR_TIE_BAND.
    """

    pytestmark = pytest.mark.skip_less_device(4)

    # The parallel topology the dep4 target implements: attention DP over the
    # 4 GPUs of one node, the routed experts split four ways.
    PARALLEL = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)
    RANKS = 4

    # Every step at most 2048 tokens a rank, as dep4's CUDA-graph identity gate
    # runs, inside the routed MoE's certified all-gather and reduce-scatter
    # cells: the ~8K prompts prefill in 2048-token chunks, the same chunks in
    # both legs, and the batched report's requests are chunked the same way.
    ALONE_MAX_NUM_TOKENS = 2048
    CHUNKED_PREFILL = True
    BATCH_MAX_NUM_TOKENS = 2048

    # How far dep4's verify can move a gap between MTP off's top-1 and another
    # of its top tokens: the drift band of dep4's verify test
    # (test_modeling_v2_dep4_verify_invariance.py, DRIFT_GAP_BAND in its job),
    # which follows each fixed prompt alone on its rank for the tokens this gate
    # generates, MTP on along MTP off's own tokens, and measured at most 0.7819
    # over 6132 positions; plus a quarter.
    NEAR_TIE_BAND = 0.98

    def _ranks(self, prompts) -> dict[str, int]:
        """One long prompt per rank, the short ones dealt round-robin."""
        longs = [p.name for p in prompts if p.long]
        shorts = [p.name for p in prompts if not p.long]
        assert len(longs) == self.RANKS, f"{len(longs)} long prompts for {self.RANKS} ranks"
        ranks = {name: rank for rank, name in enumerate(longs)}
        ranks.update({name: i % self.RANKS for i, name in enumerate(shorts)})
        return ranks

    def _batch_order(self, requests: list[_MtpRequest]) -> list[_MtpRequest]:
        """The long prompts first, one per rank, then the short ones.

        The idle executor wakes on the first request alone, and the others
        arrive while its first chunk runs. A rank's long prompt then takes the
        rank's token budget for its first chunks, and the scheduler, first come
        first served, holds the rank's short prompts behind it whatever the
        arrival times.
        """
        return sorted(requests, key=lambda r: r.max_tokens != self.LONG_TOKENS)


class TestModelingV2Nemotron4NanoPreviewSm103Tp4MtpIdentity(_MtpIdentity):
    """nemotron4-nano-preview (text model) / sm_103 / tp4: MTP-Eagle on against off, on 4 ranks.

    Tensor parallel over the 4 GPUs of one node, without attention DP: every
    rank runs every request, in the steps rank 0's executor schedules, on its
    quarter of the mixers' heads and of every MLP's and routed expert's
    intermediate width, the MTP layer's included, and the ranks meet in every
    block's all-reduce. Every rank forms the same logits: the all-reduce sums a
    token's four partials by one formula whatever the step's row count, and the
    GDP verify reads a request's tokens 3 x 4632 elements apart at the draft
    length here, which like the decode step's 4632 is not a multiple of 16, so
    its kernel rounds as the decode kernel does. What parts the legs is the
    verify's attention, as on tp1, by up to NEAR_TIE_BAND. So tp1's legs, checks
    and reports carry over unchanged.
    """

    pytestmark = pytest.mark.skip_less_device(4)

    # The parallel topology the tp4 target implements: tensor parallel over the
    # 4 GPUs of one node, the mixers' heads and every MLP's and routed expert's
    # intermediate width split four ways.
    PARALLEL = dict(tensor_parallel_size=4, moe_tensor_parallel_size=4)

    # How far tp4's verify can move a gap between MTP off's top-1 and another of
    # its top tokens: the drift band of tp4's verify test
    # (test_modeling_v2_tp4_verify_invariance.py, DRIFT_GAP_BAND in its job),
    # which follows each fixed prompt alone for the tokens this gate generates,
    # MTP on along MTP off's own tokens, and measured at most 1.2381 over 6132
    # positions; plus a quarter. A rank attends with 4 of the 16 query heads, so
    # thop_attention runs other kernels than on tp1, and the band is tp4's own.
    NEAR_TIE_BAND = 1.55


class TestModelingV2Nemotron4NanoPreviewSm103Tp1MtpAcceptance(_MtpAcceptance):
    """nemotron4-nano-preview (text model) / sm_103 / tp1, MTP-Eagle draft 2.

    The acceptance length against the checkpoint's own MTP layer.
    """

    # How far the target's MTP logits can be from the reference's, per draft
    # depth: the MTP module gate's measured worst |logit difference| against the
    # reference's own drafting chain on the same trunk hidden states
    # (test_modeling_v2_mtp_layer.py, 49,877 depth-1 and 4,098 depth-2 rows; the
    # largest gain of any token on the reference's top-1 there is 2.619 / 2.430),
    # plus the band on the target trunk's logits against the reference's
    # (check_goldens.py: free-running logits over the whole vocabulary, 2.0, the
    # larger of its two logit bands; the decode-step one is 1.25), because here
    # the target's MTP layer reads the target trunk's hidden states and the
    # reference's reads its own, at prefill rows and decode rows alike. The trunk
    # band gains the verify's measured term, 5.0471: with MTP on the trunk
    # verifies through thop_attention, whose logits drift from the decode steps'
    # by up to that much along the same tokens (test_modeling_v2_verify_invariance.py,
    # the drift over 6132 positions of the fixed prompts).
    MTP_LOGIT_ERROR = (2.6806, 2.4847)
    TRUNK_LOGIT_BAND = 2.0 + 5.0471

    @skip_not_sm103
    def test_mtp2_acceptance_matches_reference(self, monkeypatch):
        """The acceptance length the target measures matches the reference's (``_check_acceptance``).

        The ~8K prompts' prefills run beside the short prompts' verify steps,
        and AL_trt comes from the iteration statistics.
        """
        self._check_acceptance(monkeypatch)


class TestModelingV2Nemotron4NanoPreviewSm103Dep4MtpAcceptance(_MtpAcceptance):
    """nemotron4-nano-preview (text model) / sm_103 / dep4, MTP-Eagle draft 2.

    The acceptance length against the checkpoint's own MTP layer, on 4 ranks
    with attention DP. Each rank drafts and verifies its own requests, and the
    ranks meet in every routed MoE block, the MTP layer's included; a rank with
    no request left runs the attention-DP dummy. Every prompt is pinned to a
    rank, and MAX_BATCH_SIZE and the CUDA graphs are per rank.
    """

    # Attention DP over the 4 GPUs of one node with the routed experts split
    # four ways: the topology the dep4 target implements. Every rank-step
    # carries at most 2048 tokens, inside the routed MoE's certified all-gather
    # and reduce-scatter cells, so the ~8K prompts prefill in 2048-token chunks.
    ENGINE_ARGS = dict(
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        enable_attention_dp=True,
        max_num_tokens=2048,
        enable_chunked_prefill=True,
    )
    RANKS = 4

    # How far the target's MTP logits can move a margin of the reference's
    # drafts, per draft depth: the larger of the MTP module gate's worst |logit
    # difference| against the reference's own drafting chain on the same trunk
    # hidden states and its largest gain of any token on that chain's top-1,
    # over the 4 ranks, each running the prompt set in its own rotation
    # (test_modeling_v2_dep4_mtp_layer.py; depth 1: 2.6806 and 2.619, depth 2:
    # 2.8998 and 4.5655, the gain rank 3's), plus the band on the target trunk's
    # logits against the reference's: dep4's trunk passes check_goldens_dep4.py
    # with tp1's bands, whose free-running logits band is 2.0. The trunk band
    # gains the verify's measured term, 4.5999: with MTP on the trunk verifies
    # through thop_attention, whose logits drift from the decode steps' by up to
    # that much along the same tokens (test_modeling_v2_dep4_verify_invariance.py,
    # the drift over 6132 positions of the fixed prompts).
    MTP_LOGIT_ERROR = (2.6806, 4.5655)
    TRUNK_LOGIT_BAND = 2.0 + 4.5999

    def _ranks(self, prompts) -> dict[str, int]:
        """One long prompt per rank, the short ones dealt round-robin.

        Submitted in the prompt set's order, the short prompts first, so a
        rank's long prompt prefills in chunks while the rank's short requests
        verify.
        """
        longs = [p.name for p in prompts if p.long]
        shorts = [p.name for p in prompts if not p.long]
        assert len(longs) == self.RANKS, f"{len(longs)} long prompts for {self.RANKS} ranks"
        ranks = {name: rank for rank, name in enumerate(longs)}
        ranks.update({name: i % self.RANKS for i, name in enumerate(shorts)})
        return ranks

    def _engine_al(
        self, stats: list[dict], walks: dict[str, list[tuple[int, int]]], ranks: dict[str, int]
    ) -> tuple[float, int, dict]:
        """AL_trt from the requests' own walks, which are exact on every rank.

        Under attention DP the iteration statistics hold one row per rank and
        iteration, and every row carries rank 0's speculative-decoding counts
        (``ADPIterStatsBuffer._make_rank_iter_stats``): they count rank 0's
        requests once per rank and the other ranks' requests never. They are
        reported, not used: the suite's estimator over them, how many
        iterations' rows agree, and rank 0's rows against rank 0's requests.
        """
        rows = [s for s in stats if (s.get("specDecodingStats") or {}).get("numDraftTokens", 0) > 0]
        by_iteration = defaultdict(list)
        for s in rows:
            by_iteration[s["iter"]].append(s["specDecodingStats"])
        rank0_rows = [s["specDecodingStats"] for s in rows if s["attentionDpRank"] == 0]
        rank0_walks = [walks[name] for name, rank in ranks.items() if rank == 0]
        return (
            _al([a for walk in walks.values() for _, a in walk]),
            0,
            dict(
                iteration_stats_al=round(acceptance_length_from_iteration_stats(stats), 4),
                iteration_stats_rows_by_rank=dict(
                    sorted(Counter(s["attentionDpRank"] for s in rows).items())
                ),
                iterations_with_drafts=len(by_iteration),
                iterations_whose_rows_agree=sum(
                    all(row == group[0] for row in group) for group in by_iteration.values()
                ),
                iteration_stats_rank0_steps=sum(
                    s["numRequestsWithDraftTokens"] for s in rank0_rows
                ),
                iteration_stats_rank0_accepted=sum(s["numAcceptedTokens"] for s in rank0_rows),
                rank0_request_steps=sum(len(walk) for walk in rank0_walks),
                rank0_request_accepted=sum(a for walk in rank0_walks for _, a in walk),
            ),
        )

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_mtp2_acceptance_matches_reference(self, monkeypatch):
        """The acceptance length the target measures matches the reference's (``_check_acceptance``).

        Each rank verifies its requests beside its long prompt's prefill chunks
        and beside ranks that verify, prefill or idle, and AL_trt comes from the
        requests' own walks.
        """
        self._check_acceptance(monkeypatch)


class TestModelingV2Nemotron4NanoPreviewSm103Tp4MtpAcceptance(_MtpAcceptance):
    """nemotron4-nano-preview (text model) / sm_103 / tp4, MTP-Eagle draft 2.

    The acceptance length against the checkpoint's own MTP layer, on 4 ranks
    tensor parallel without attention DP. Every rank runs every request on its
    quarter of the mixers' heads and of every MLP's and routed expert's
    intermediate width, the MTP layer's included, and the all-reduces give
    every rank the same bits; the lm_head is whole on every rank, so every rank
    drafts the same tokens. MAX_BATCH_SIZE and the CUDA graphs are the
    engine's, as on tp1.
    """

    # Tensor parallel over the 4 GPUs of one node without attention DP: the
    # topology the tp4 target implements. Every step carries at most 8192
    # tokens, for which the all-reduce workspace is sized, so a ~8K prompt
    # prefills in one step beside the other requests' verify rows, as on tp1.
    ENGINE_ARGS = dict(tensor_parallel_size=4, moe_tensor_parallel_size=4, max_num_tokens=8192)

    # How far the target's MTP logits can move a margin of the reference's
    # drafts, per draft depth: the larger of the MTP module gate's worst |logit
    # difference| against the reference's own drafting chain on the same trunk
    # hidden states and its largest gain of any token on that chain's top-1,
    # every rank alike (test_modeling_v2_tp4_mtp_layer.py; depth 1: 2.7225 and
    # 2.6390, depth 2: 2.47151 and 2.47161, the gain, rounded up), plus the band
    # on the target trunk's logits against the reference's: tp4's trunk passes
    # check_goldens_tp4.py at tp1's free-running logits band, 2.0, plus its
    # measured tensor-parallel term, 1.1922 (tp4 against tp1 on the same
    # prompts, largest on an ~8K prompt), plus the verify's measured term,
    # 3.4226: with MTP on the trunk verifies through thop_attention, whose logits
    # drift from the decode steps' by up to that much along the same tokens
    # (test_modeling_v2_tp4_verify_invariance.py, the drift over 6132 positions
    # of the fixed prompts).
    MTP_LOGIT_ERROR = (2.7225, 2.4717)
    TRUNK_LOGIT_BAND = 3.1922 + 3.4226

    def _engine_al(
        self, stats: list[dict], walks: dict[str, list[tuple[int, int]]], ranks: dict[str, int]
    ) -> tuple[float, int, dict]:
        """AL_trt from the requests' own walks, which count each verify step once.

        Every rank runs every request, so rank 0's iteration statistics see
        them all, but they count a request once more after its last step when
        the overlap scheduler had already scheduled it again
        (``_MtpAcceptance._engine_al``). They are reported, not used: the
        suite's estimator over them, and the steps they count beyond the
        requests'.
        """
        spec_stats = [s["specDecodingStats"] for s in stats if s.get("specDecodingStats")]
        stats_steps = sum(s["numRequestsWithDraftTokens"] for s in spec_stats)
        return (
            _al([a for walk in walks.values() for _, a in walk]),
            0,
            dict(
                iteration_stats_al=round(acceptance_length_from_iteration_stats(stats), 4),
                iteration_stats_steps=stats_steps,
                iteration_stats_accepted=sum(s["numAcceptedTokens"] for s in spec_stats),
                iteration_stats_extra_steps=stats_steps - sum(len(w) for w in walks.values()),
            ),
        )

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_mtp2_acceptance_matches_reference(self, monkeypatch):
        """The acceptance length the target measures matches the reference's (``_check_acceptance``).

        Every rank verifies every request beside the ~8K prompts' prefills, and
        AL_trt comes from the requests' own walks.
        """
        self._check_acceptance(monkeypatch)
