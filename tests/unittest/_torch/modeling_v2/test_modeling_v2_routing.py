# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``modeling_v2_resolve`` actually does, driven by synthetic configs.

No checkpoint and no weights: routing reads config *shape*, the mapping, the
SM version and the LLM API arguments the deployment was configured with, all
of which can be stated directly. The SM version is monkeypatched so these run
on any device -- the point here is the decision logic, not the kernels.

The things worth proving:

* ``off`` changes nothing. This is the whole safety argument for making the
  decision in the model loader at all.
* a matching configuration resolves to the target class itself, imported
  from its own directory; no registry lookup stands between the decision and
  the class.
* ``require`` raises on a near-miss and says, in words, which criterion
  missed; ``auto`` falls back to the built-in implementation instead.
* speculative decoding is an identity criterion: a target that does not
  implement the requested mode does not claim the deployment.
* ``AutoModelForCausalLM._resolve_class`` reads the decision off the
  ``ModelConfig`` rather than deciding again.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, NGramDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_BUILTIN_GPT_OSS = "tensorrt_llm._torch.models.modeling_gpt_oss"
_GPT_OSS_TARGET = "models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling.ModelingV2GptOss120bSm103Tp1"
_R1_TARGET = (
    "models.deepseek_v3.r1_0528_nvfp4__sm_103__dep4.modeling.ModelingV2DeepseekR10528Nvfp4Sm103Dep4"
)
_NEMOTRON4_TARGET = (
    "models.nemotron4.nemotron4_nano_preview__sm_103__tp1.modeling."
    "ModelingV2Nemotron4NanoPreviewSm103Tp1"
)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _gpt_oss_config(**overrides):
    """The shape gpt-oss-120b's own config.json declares."""
    fields = dict(
        architectures=["GptOssForCausalLM"],
        model_type="gpt_oss",
        num_hidden_layers=36,
        hidden_size=2880,
        num_local_experts=128,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _r1_config(**overrides):
    """The shape DeepSeek-R1-0528-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["DeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        num_hidden_layers=61,
        hidden_size=7168,
        n_routed_experts=256,
        q_lora_rank=1536,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


#: nemotron4-nano-preview's text_config.layer_types, one letter per block.
_NEMOTRON4_BLOCKS = dict(L="linear_attention", M="mlp", A="shortcut_moe_attn", G="shortcut_moe_gdp")
_NEMOTRON4_LAYOUT = "LMLMLAGGLAGGGLAGGGLAGGGLAGGGLAGGGLAGGLAGLMLM"


def _nemotron4_config(layout=_NEMOTRON4_LAYOUT, **text_overrides):
    """The shape nemotron4-nano-preview's own config.json declares.

    A vision-language composite: the fingerprint lives in its text_config, the
    text model being what the target serves.
    """
    text = dict(
        model_type="nemotron4_text",
        hidden_size=3072,
        layer_types=[_NEMOTRON4_BLOCKS[block] for block in layout],
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=256,
        gdp_num_heads=24,
        gdp_num_groups=24,
        gdp_key_head_dim=128,
        gdp_head_dim=64,
        gdp_num_householder=3,
        gdp_conv_kernel=4,
        intermediate_size=12288,
        mlp_hidden_act="relu2",
        n_routed_experts=512,
        num_experts_per_tok=10,
        moe_latent_size=768,
        moe_intermediate_size=4096,
        moe_shared_expert_intermediate_size=4096,
        moe_use_norm_before_up_proj=True,
        moe_shortcut_post_norm=False,
        wide_residual_num_streams=3,
        wide_residual_learned_retention=True,
        residual_in_fp32=False,
        tie_word_embeddings=False,
        num_nextn_predict_layers=2,
        mtp_use_repeated_layer=True,
    )
    text.update(text_overrides)
    return PretrainedConfig(
        architectures=["Nemotron4ForConditionalGeneration"],
        model_type="nemotron4",
        text_config=PretrainedConfig(**text),
    )


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """The deployment, as the LLM API would have been given it.

    The switch is an LLM API argument, so the tests set one too -- that is the
    surface under test.
    """
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _model_config(pretrained_config, spec_config=None, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config, mapping=mapping, spec_config=spec_config
    )


# A speculative mode no target implements.
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True)


@pytest.mark.parametrize("mode", [None, "off"], ids=["default", "off"])
def test_off_resolves_nothing(mode):
    """The default must be indistinguishable from modeling_v2 not existing.

    Leaving the argument alone and an explicit "off" have to behave
    identically: the common case is that nobody has heard of this package.
    """
    args = TorchLlmArgs(model=_DUMMY_MODEL) if mode is None else _llm_args(mode)
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, args) is None


def test_off_still_reaches_the_builtin_implementation():
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, _llm_args("off")) is None
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved is not None
    assert resolved.__module__ == _BUILTIN_GPT_OSS


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_gpt_oss_tp1_matches(mode):
    config = _model_config(_gpt_oss_config())
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _GPT_OSS_TARGET


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_r1_dep4_matches(mode):
    config = _model_config(_r1_config(), **_DEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _R1_TARGET


def test_resolve_class_reads_the_decision_off_the_config():
    """The loader decides and writes the class on the ModelConfig; the class
    lookup only has to honour it."""
    config = _model_config(_gpt_oss_config())
    target = modeling_v2_resolve(config, _llm_args())
    decided = replace(config, modeling_v2_target=target)
    assert AutoModelForCausalLM._resolve_class(decided) is target


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a GptOss checkpoint of another size
        (dict(num_hidden_layers=24, num_local_experts=32), {}, r"shape \(24, 2880, 32\)"),
        # the right checkpoint, a topology no target implements
        (dict(), dict(world_size=2, tp_size=2), "2 ranks"),
    ],
)
def test_gpt_oss_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_gpt_oss_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_r1_without_attention_dp_does_not_match():
    """dep4 and tep4 differ in where the attention weights are split, so
    attention DP is an identity criterion rather than a knob."""
    mapping_kwargs = dict(_DEP4, enable_attention_dp=False)
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args()) is None


def test_require_names_the_criterion_that_missed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "no target fits" in message and "GptOssForCausalLM" in message
    assert "compute capability (10, 0)" in message


def test_an_unrouted_architecture_is_not_an_error_under_auto():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_an_unrouted_architecture_raises_under_require():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_an_unknown_mode_is_rejected_when_the_arguments_are_built():
    """A typo must not read as "off".

    That would hand back the built-in implementation while the caller believed
    they had asked for a target -- the exact mis-attribution the require mode
    exists to prevent. Pydantic refuses it before anything is built.
    """
    # "yes" rather than a misspelling: it is what someone reaching for a
    # boolean would write, and it is the reading that must not be invented.
    with pytest.raises(ValidationError, match="modeling_v2"):
        _llm_args("yes")


# --- speculative decoding is an identity criterion ------------------------------


def test_gpt_oss_with_speculative_decoding_falls_back_under_auto():
    """The tp1 target implements no speculative path, so claiming the
    deployment would trade a fallback for a crash inside the target."""
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_gpt_oss_with_speculative_decoding_raises_under_require():
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "NGramDecodingConfig" in message


def test_r1_dep4_with_mtp_matches():
    config = _model_config(_r1_config(), spec_config=MTPDecodingConfig(max_draft_len=3), **_DEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _R1_TARGET


def test_r1_dep4_with_another_speculative_mode_does_not_match():
    """The dep4 target carries the checkpoint's MTP module and nothing else."""
    config = _model_config(_r1_config(), spec_config=_NGRAM, **_DEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_r1_dep4_with_vanilla_mtp_does_not_match():
    """``use_mtp_vanilla`` resolves to the multi-model MTP loop; the target
    asserts the one-model MTP-Eagle loop at construction, so route must say
    no before that assert can."""
    vanilla = MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)
    config = _model_config(_r1_config(), spec_config=vanilla, **_DEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="MTP"):
        modeling_v2_resolve(config, _llm_args("require"))
    with pytest.raises(ValueError, match="MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- nemotron4: a composite checkpoint routed on its text model ------------------


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_nemotron4_tp1_matches(mode):
    config = _model_config(_nemotron4_config())
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _NEMOTRON4_TARGET


def test_nemotron4_has_no_builtin_to_fall_back_to():
    """There is no built-in Nemotron4 implementation, so ``auto`` falling back
    finds no model class at all: ``require`` is the only mode this checkpoint
    runs under."""
    config = _model_config(_nemotron4_config())
    assert modeling_v2_resolve(config, _llm_args("off")) is None
    assert AutoModelForCausalLM._resolve_class(config) is None


@pytest.mark.parametrize(
    "config, mapping_kwargs, missed",
    [
        # another width
        (_nemotron4_config(hidden_size=4096), {}, r"hidden_size=4096 \(target: 3072\)"),
        # the same blocks in another order: block 5 a GDP mixer instead of attention
        (
            _nemotron4_config(layout=_NEMOTRON4_LAYOUT[:5] + "G" + _NEMOTRON4_LAYOUT[6:]),
            {},
            "layer_pattern='LMLMLGGG",
        ),
        # another attention head geometry
        (_nemotron4_config(num_key_value_heads=2), {}, r"num_key_value_heads=2 \(target: 1\)"),
        # another latent MoE
        (_nemotron4_config(moe_latent_size=1024), {}, r"moe_latent_size=1024 \(target: 768\)"),
        # another GDP geometry
        (_nemotron4_config(gdp_num_householder=1), {}, r"gdp_num_householder=1 \(target: 3\)"),
        # the right checkpoint, tensor parallel over two ranks
        (_nemotron4_config(), dict(world_size=2, tp_size=2), "world_size=2"),
        # the right checkpoint, attention DP with expert-parallel MoE over two ranks
        (
            _nemotron4_config(),
            dict(world_size=2, tp_size=2, moe_ep_size=2, moe_tp_size=1, enable_attention_dp=True),
            "attention_dp=True",
        ),
    ],
    ids=["hidden", "layout", "kv_heads", "latent", "householder", "tp2", "dep2"],
)
def test_nemotron4_near_misses_do_not_match(config, mapping_kwargs, missed):
    model_config = _model_config(config, **mapping_kwargs)
    assert modeling_v2_resolve(model_config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(model_config, _llm_args("require"))


def test_nemotron4_on_another_sm_names_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_nemotron4_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "Nemotron4ForConditionalGeneration" in message
    assert "compute capability (10, 0)" in message
    assert "text config matches nemotron4-nano-preview" in message


def test_nemotron4_without_a_text_config_does_not_match():
    config = _model_config(PretrainedConfig(architectures=["Nemotron4ForConditionalGeneration"]))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="text_config"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "spec_config",
    [MTPDecodingConfig(max_draft_len=2), _NGRAM],
    ids=["mtp", "ngram"],
)
def test_nemotron4_tp1_declines_speculative_decoding(spec_config):
    """The tp1 target carries no draft path, so every speculative mode is
    declined, the checkpoint's own MTP module included, and the message names
    the mode."""
    config = _model_config(_nemotron4_config(), spec_config=spec_config)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=type(spec_config).__name__):
        modeling_v2_resolve(config, _llm_args("require"))
