# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Nemotron4ForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. The engine has no built-in
Nemotron4 implementation, so in ``auto`` a declined deployment finds no model
class at all; in ``require`` it raises this error instead.

The checkpoint is a vision-language composite and every target here serves its
text model alone, so the fingerprint is read off ``text_config``; the vision
tower takes no part in the decision.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..._router_index import NoTarget

#: One letter per block type of ``text_config.layer_types``, so the block layout
#: reads as one string here and in a ``NoTarget`` message.
_BLOCK_LETTERS = {
    "linear_attention": "L",  # GDP mixer
    "mlp": "M",  # dense MLP
    "shortcut_moe_attn": "A",  # attention mixer beside a shortcut MoE
    "shortcut_moe_gdp": "G",  # GDP mixer beside a shortcut MoE
}

#: nemotron4-nano-preview's text model, as its config.json declares it. Each
#: field fixes the forward's structure or a weight geometry the target declares:
#: the block layout, the attention and GDP head geometry, the latent MoE, the
#: optional weights that exist (learned retention, the latent norm, no post-MoE
#: norm, an untied lm_head), the residual dtype, and the MTP module shipped
#: beside the trunk.
_NANO_PREVIEW = dict(
    hidden_size=3072,
    layer_pattern="LMLMLAGGLAGGGLAGGGLAGGGLAGGGLAGGGLAGGLAGLMLM",
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


def _text_shape(text_config: Any) -> dict:
    """The fingerprint's fields as this text config states them; a field it
    does not carry reads as None and so cannot match."""
    shape = {name: getattr(text_config, name, None) for name in _NANO_PREVIEW}
    layer_types = getattr(text_config, "layer_types", None) or ()
    shape["layer_pattern"] = "".join(_BLOCK_LETTERS.get(t, "?") for t in layer_types)
    return shape


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is None:
        raise NoTarget("the config carries no text_config; every nemotron4 target serves one")

    # The checkpoint is recognized by its text config's shape, the upstream
    # idiom (``is_mla``, ``is_nemotron_hybrid``). A fine-tune with the same
    # shape routes here too; the accuracy gate is what pins the identity.
    shape = _text_shape(text_config)
    differs = [
        f"{name}={shape[name]!r} (target: {want!r})"
        for name, want in _NANO_PREVIEW.items()
        if shape[name] != want
    ]

    # tp1: one rank holds the whole model. Attention DP over one rank changes
    # how the shell builds lm_head and which metadata a step carries, so it is
    # a different deployment rather than a knob.
    tp1 = mapping.world_size == 1 and not mapping.enable_attention_dp

    # nemotron4-nano-preview's text model on one GB300 (sm103), no speculative
    # decoding: the target implements no draft path, so claiming such a
    # deployment would crash in the model instead of declining here.
    if compute_capability == (10, 3) and not differs and tp1 and spec_config is None:
        # One line on purpose: test_modeling_v2_claims finds a family's targets
        # by reading this import, and the formatter and isort would wrap it.
        # fmt: off
        # isort: off
        from .nemotron4_nano_preview__sm_103__tp1.modeling import ModelingV2Nemotron4NanoPreviewSm103Tp1
        # isort: on
        # fmt: on

        return ModelingV2Nemotron4NanoPreviewSm103Tp1

    speculation = (
        "off"
        if spec_config is None
        else f"{type(spec_config).__name__} ({spec_config.spec_dec_mode.name})"
    )
    raise NoTarget(
        f"compute capability {compute_capability}, text config "
        + (f"differs: {', '.join(differs)}" if differs else "matches nemotron4-nano-preview")
        + f", world_size={mapping.world_size} attention_dp={mapping.enable_attention_dp}, "
        f"speculative decoding {speculation}; the one nemotron4 target is "
        "nemotron4-nano-preview's text model on sm103, tp1 (one rank, no attention DP), "
        "without speculative decoding"
    )
