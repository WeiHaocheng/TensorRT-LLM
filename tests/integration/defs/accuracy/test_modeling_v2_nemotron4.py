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
from tensorrt_llm.llmapi import CudaGraphConfig

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness

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
    """nemotron4-nano-preview (text model) / sm_103 / tp1."""

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

    @skip_not_sm103
    def test_gsm8k(self, mocker):
        # The stock protocol, as the anchor was measured: 5-shot, no chat
        # template, the default 256 output tokens. The checkpoint's chat
        # template is not usable for this: in chat form the model never emits
        # <|im_end|>, so an answer ends only at a stop word or the token cap,
        # and its thinking mode answers without reasoning. Only the filter is
        # patched.
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)

        # The target checks its step contract on the first forward and only
        # when asked. The contract is about metadata this engine prepared, so
        # it is checked in a run that prepared it. It travels as
        # ``env_overrides`` because the check runs on the rank.
        #
        # trust_remote_code: the checkpoint's config class is its own
        # configuration_nemotron4.py. The overlap scheduler is on by default;
        # it is stated because the gate is defined with it on.
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
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
