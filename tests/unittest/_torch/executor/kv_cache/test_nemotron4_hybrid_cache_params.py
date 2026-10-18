# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron 4 hybrid GDP + attention cache layout, without checkpoint files."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    get_nemotron4_layer_masks,
    is_hybrid_linear,
    is_nemotron4,
    resolve_auto_ssm_cache_dtype,
)

pytestmark = pytest.mark.cpu_only

# Released checkpoint geometry: 24 heads, value head 64, key head 128, three
# Householder steps per token and a width-4 short convolution over
# [V_0..V_2 | K_0..K_2 | Q] = 3 * 24 * 64 + 4 * 24 * 128 = 16896 channels.
_GDP = dict(
    gdp_num_heads=24,
    gdp_head_dim=64,
    gdp_key_head_dim=128,
    gdp_num_groups=24,
    gdp_num_householder=3,
    gdp_conv_kernel=4,
)
_GDP_CONV_WIDTH = 16896
_LAYER_TYPES = ["linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp"]


def _text_config(**overrides) -> PretrainedConfig:
    values = dict(model_type="nemotron4_text", layer_types=list(_LAYER_TYPES), **_GDP)
    values.update(overrides)
    return PretrainedConfig(**values)


def _config(**text_overrides) -> PretrainedConfig:
    return PretrainedConfig(model_type="nemotron4", text_config=_text_config(**text_overrides))


def test_composite_and_text_configs_are_hybrid():
    for config in (_config(), _text_config()):
        assert is_nemotron4(config)
        assert is_hybrid_linear(config)
    assert not is_nemotron4(PretrainedConfig(model_type="nemotron_h"))


def test_layer_masks_follow_layer_types():
    expected = ([False, False, True, False], [True, False, False, True])
    assert get_nemotron4_layer_masks(_config()) == expected
    assert get_nemotron4_layer_masks(_text_config()) == expected

    with pytest.raises(ValueError, match="Unsupported Nemotron 4 layer types"):
        get_nemotron4_layer_masks(_config(layer_types=["linear_attention", "full_attention"]))
    with pytest.raises(ValueError, match="must define layer_types"):
        get_nemotron4_layer_masks(_config(layer_types=[]))


def test_cache_params_size_the_gdp_conv_and_recurrent_state():
    # An explicit bf16 request (quickstart_advanced's default) still gets the
    # fp32 recurrent state the gated delta product runs in.
    quant_config = PretrainedConfig(mamba_ssm_cache_dtype=torch.bfloat16)
    params = extract_mamba_kv_cache_params(_config(), quant_config=quant_config)

    assert (params.state_size, params.conv_kernel) == (128, 4)
    assert (params.num_heads, params.head_dim) == (24, 64)
    assert params.head_dim * params.num_heads + 2 * params.n_groups * params.state_size == (
        _GDP_CONV_WIDTH
    )
    assert params.mamba_layer_mask == [True, False, False, True]
    assert params.target_full_attention_layer_mask == [False, False, True, False]
    assert params.num_mamba_layers == 2
    assert params.num_draft_layers == 0
    assert params.mamba_ssm_cache_dtype is torch.float32
    assert resolve_auto_ssm_cache_dtype(_config(), torch.bfloat16) is torch.float32


def test_conv_width_the_managers_cannot_express_is_rejected():
    # 2 * 3 * 4 + 3 * 1 * 6 = 42 channels; 42 - 3 * 4 = 30 is not a multiple
    # of 2 * 6, so no n_groups reproduces the width.
    config = _config(
        gdp_num_heads=3,
        gdp_head_dim=4,
        gdp_key_head_dim=6,
        gdp_num_groups=1,
        gdp_num_householder=2,
    )
    with pytest.raises(ValueError, match="not expressible"):
        extract_mamba_kv_cache_params(config)


def test_model_config_counts_attention_and_gdp_layers():
    model_config = ModelConfig(pretrained_config=_config())
    assert model_config.get_num_attention_layers() == 1
    assert model_config.get_num_mamba_layers() == 2
