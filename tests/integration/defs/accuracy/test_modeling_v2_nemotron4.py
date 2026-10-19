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

Every test here passes ``modeling_v2="require"`` to ``LLM``. TensorRT-LLM has no
built-in implementation of this architecture, so there is nothing to fall back
to: under ``"auto"`` a deployment routing declines would find no model class at
all, while ``"require"`` makes the decline raise with routing's reason. The
same absence means there is no ``modeling_v2="off"`` oracle either; the anchor
in references/gsm8k.yaml is the checkpoint's own HuggingFace implementation,
measured under the protocol these tests run.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, MTPDecodingConfig

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness, compute_acceptance_length

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# The filter the anchor was measured on. Unset, the evaluator averages every
# metric GSM8K reports, which means the mean of strict-match and
# flexible-extract -- two numbers measuring different things.
_SCORES_FILTER = {"scores_filter": "exact_match,flexible-extract"}


class TestModelingV2Nemotron4NanoPreviewSm103Tp1(LlmapiAccuracyTestHarness):
    """nemotron4-nano-preview (text model) / sm_103 / tp1, identity and the mtp2 variant."""

    MODEL_NAME = "Nemotron4/nemotron4-nano-preview"
    MODEL_PATH = f"{llm_models_root()}/nemotron4-nano-preview"

    # The config's max_position_embeddings is 1,048,576, which the engine would
    # otherwise take as max_seq_len and size the KV-cache estimate from. This
    # is the context the target serves: 8192 + 1024 tokens plus draft slack.
    MAX_SEQ_LEN = 9344

    # The decode batch the target serves, and CUDA graphs captured with padding
    # for every batch size up to it. The 1319 questions are submitted at once,
    # so the run fills the batch and then drains it through the smaller graphs.
    MAX_BATCH_SIZE = 512

    # The checkpoint's trained draft depth: its one MTP layer serves both
    # depths. MTPDecodingConfig defaults to a depth of 1, so it is stated.
    MTP_DRAFT_LEN = 2

    # There is deliberately no standalone identity gsm8k case. The paired test
    # below evaluates the identity config as its first leg, and
    # ``task.evaluate`` asserts accuracy against the reference on the way past,
    # so a separate one would gate nothing new for one more engine boot.

    def _llm(self, **variant) -> LLM:
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
            max_batch_size=self.MAX_BATCH_SIZE,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE
            ),
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

        with self._llm() as llm:
            identity = task.evaluate(llm)

        # The default kv-cache fraction is kept. The verify's per-position GDP
        # and conv states (~41 GiB at max_batch_size 512) are allocated before
        # the engine's profiling forward, so the cache pools are sized from the
        # memory left after them, ~40 GiB: about twice what the state slots and
        # a full batch of GSM8K requests take.
        #
        # Every iteration's stats are kept (``max_stats_len=-1``), so the
        # acceptance length covers the whole run rather than its last 1000
        # iterations.
        with self._llm(
            speculative_config=MTPDecodingConfig(max_draft_len=self.MTP_DRAFT_LEN),
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


class TestModelingV2Nemotron4NanoPreviewSm103Dep4(LlmapiAccuracyTestHarness):
    """nemotron4-nano-preview (text model) / sm_103 / dep4, identity."""

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

    # Per rank: under attention DP every rank schedules its own requests, and
    # the CUDA graphs are captured per rank, with padding, for every batch size
    # up to it. The 1319 questions, spread over the four ranks, are all in
    # flight at once and drain through the smaller graphs.
    MAX_BATCH_SIZE = 512

    # Every step's rows per rank. The routed MoE's all-gather and
    # reduce-scatter are certified up to 2048 rows a rank, so the gate keeps
    # each step inside them. A 5-shot GSM8K prompt (at most ~1.5K tokens)
    # still fits in one step, so prompts are not chunked.
    MAX_NUM_TOKENS = 2048

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self, mocker):
        """The identity accuracy gate on 4 ranks.

        It holds dep4 to the bar tp1 is held to: the anchor is a property of
        the checkpoint, and the topology only changes where each request runs
        and in which order the routed experts' partial sums meet. The target
        adds the four ranks' fp32 partial sums before its one rounding to
        bf16, which is the rounding HF makes.

        The step contract is checked on every rank's first forward, as on tp1;
        the switch travels as ``env_overrides`` because the check runs on the
        ranks.
        """
        # The stock protocol, as the anchor was measured: 5-shot, no chat
        # template, the default 256 output tokens. Only the filter is patched.
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        task = GSM8K(self.MODEL_NAME)

        with LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            trust_remote_code=True,
            max_seq_len=self.MAX_SEQ_LEN,
            max_batch_size=self.MAX_BATCH_SIZE,
            max_num_tokens=self.MAX_NUM_TOKENS,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE
            ),
            **self.DEP4,
        ) as llm:
            task.evaluate(llm)
