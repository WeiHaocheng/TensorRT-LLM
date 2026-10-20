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

Three topologies have a target: tp1, dep4 and tp4. The other layouts of one
4-GPU node that could serve the 8K/1K workload are named in the ``NoTarget``
message with the reason they have none.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

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


#: The other layouts of one 4-GPU node weighed for the 8K/1K workload, and why each has
#: no target: what the performance model of this checkpoint on GB300 says of it.
_NOT_IMPLEMENTED = {
    "tp2": "on the 8K/1K performance model it never reaches the frontier the other topologies form",
    "dep2": "on the 8K/1K performance model it leads only below 25 tok/s/user, by 5-8% over dep4",
    "tep4": "on the 8K/1K performance model, from measured op times, it is below tp4 at "
    "every concurrency measured (C = 1-16), at 0.89-0.94x of tp4's tok/s/user with MTP",
}


def _topology(mapping: Any) -> Optional[str]:
    """The node layout this mapping is, named as target directories are: ``tp<n>`` (heads and
    experts' widths split), ``tep<n>`` (heads split, whole experts per rank) or ``dep<n>``
    (attention DP, whole experts per rank). None for any other layout: pipeline or context
    parallelism, a MoE split mixing tensor, expert or cluster parallelism, a vocab-parallel
    lm_head under attention DP, attention DP over one rank or with tensor-parallel experts,
    and, without attention DP, attention split other than by heads over every rank
    (``attn_tp_size`` short of the TP size, the rest attention context parallelism)."""
    n = mapping.tp_size
    if mapping.pp_size != 1 or mapping.cp_size != 1 or mapping.world_size != n:
        return None
    whole_experts = mapping.moe_ep_size == n and mapping.moe_tp_size == 1
    if mapping.enable_attention_dp:
        if n > 1 and whole_experts and not mapping.enable_lm_head_tp_in_adp:
            return f"dep{n}"
        return None
    if mapping.attn_tp_size != n:
        return None
    if mapping.moe_tp_size == n:
        return f"tp{n}"
    return f"tep{n}" if whole_experts else None


def _parallel_layout(mapping: Any) -> str:
    """The mapping's facts a target is chosen on, as the ``NoTarget`` message states them."""
    layout = (
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} attention_dp={mapping.enable_attention_dp} "
        f"moe_ep={mapping.moe_ep_size} moe_tp={mapping.moe_tp_size}"
    )
    if mapping.enable_lm_head_tp_in_adp:
        layout += " lm_head_tp_in_adp=True"
    if mapping.attn_tp_size != mapping.tp_size:
        layout += f" attn_tp={mapping.attn_tp_size} attn_cp={mapping.attn_cp_size}"
    return layout


def _text_shape(text_config: Any) -> dict:
    """The fingerprint's fields as this text config states them; a field it
    does not carry reads as None and so cannot match."""
    shape = {name: getattr(text_config, name, None) for name in _NANO_PREVIEW}
    layer_types = getattr(text_config, "layer_types", None) or ()
    shape["layer_pattern"] = "".join(_BLOCK_LETTERS.get(t, "?") for t in layer_types)
    return shape


#: The one draft length the targets run: the checkpoint's trained depth, two
#: drafts over its one repeated MTP layer, so a verify step carries 3 tokens per
#: request. It is the length the MTP identity, acceptance and GSM8K gates run.
_DRAFT_LEN = 2


def _speculation_declined(spec_config: Optional[Any], topology: Optional[str]) -> Optional[str]:
    """Why the target does not run this speculative decoding; None when it does.

    The draft path is the checkpoint's own MTP layer (``mtp.layers.0``, which
    ``mtp_use_repeated_layer`` repeats for every draft depth) driven by the
    one-model MTP-Eagle loop: one linear draft chain per request, the drafter's
    weights loaded from the target checkpoint, ``_DRAFT_LEN`` drafts on every
    step. The model loader resolves the mode against the checkpoint config
    before routing, so ``spec_dec_mode`` is the mode the engine will run.
    """
    if spec_config is None:
        return None
    if not isinstance(spec_config, MTPDecodingConfig):
        return "the target drafts only with the checkpoint's own MTP layer"
    if not spec_config.spec_dec_mode.is_mtp_eagle_one_model():
        cause = (
            "use_mtp_vanilla selects"
            if spec_config.use_mtp_vanilla
            else f"num_nextn_predict_layers={spec_config.num_nextn_predict_layers} resolved it to"
        )
        return (
            f"{cause} vanilla MTP (one MTP module and KV layer per draft depth), and the "
            "target replays its one repeated MTP layer through the one-model MTP-Eagle loop"
        )
    if spec_config.use_dynamic_tree:
        return (
            "use_dynamic_tree drafts a tree, and the target verifies one linear draft chain "
            "per request, stepping each request's recurrent state through it"
        )
    if spec_config.needs_separate_draft_weights:
        return (
            f"speculative_model={str(spec_config.speculative_model)!r} supplies separate "
            "draft weights, and the target drafts with the checkpoint's own MTP layer"
        )
    if spec_config.draft_len_schedule is not None:
        # max_concurrency reaches the engine as a one-entry schedule: no drafts above it.
        schedule = f"draft_len_schedule={spec_config.draft_len_schedule}"
        if spec_config.max_concurrency is not None:
            schedule = f"max_concurrency={spec_config.max_concurrency} ({schedule})"
        if topology == "dep4":
            why = (
                "under attention DP each rank picks its step's draft length from its own batch "
                "size, so the ranks could run different numbers of draft steps, and the MTP "
                "layer's per-step all-gather and reduce-scatter would not meet"
            )
        else:
            why = (
                f"no gate covers a draft length that changes from step to step: the gates run "
                f"{_DRAFT_LEN} drafts on every step"
            )
        return f"{schedule} changes the draft length per step, and {why}"
    if spec_config.max_draft_len != _DRAFT_LEN:
        why = (
            f"the target runs {_DRAFT_LEN} drafts, the checkpoint's trained depth, which is the "
            "length its gates certify"
        )
        if topology == "tp4":
            why += (
                "; on a tp4 rank it is also the one certified length whose verify gives "
                "decode's bits (gdp_verify: a request's stride, (1 + max_draft_len) x 4632, "
                "must not be a multiple of 16, as decode's 4632 is not)"
            )
        return f"max_draft_len={spec_config.max_draft_len}: {why}"
    return None


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

    # The node layout, read off the mapping. Each target is one layout because the layout
    # fixes which weights a rank holds and in what shape, and which collectives a step makes.
    #   tp1: one rank holds the whole model. Attention DP over one rank changes how the
    #     shell builds lm_head and which metadata a step carries, so it is not tp1.
    #   dep4: attention DP over 4 ranks with the routed experts split four ways and nothing
    #     else divided. The engine states attention DP as tensor_parallel_size=4 plus
    #     enable_attention_dp, so "nothing else" means whole experts per rank (moe_tp=1) and
    #     lm_head whole (no enable_lm_head_tp_in_adp); tep4 and tp4 split the mixers' heads.
    #   tp4: tensor parallel over 4 ranks without attention DP. Every rank runs every
    #     request on its quarter of the mixers' heads and of every MLP's and routed
    #     expert's intermediate width (moe_tp=4); lm_head is whole on every rank.
    topology = _topology(mapping)

    # Speculative decoding changes the step's structure (a verify step, a draft loop, the
    # MTP layer's weights and KV), so every mode the targets have no path for is declined
    # here rather than left to crash in the model. The draft length is part of that
    # structure: it sets the tokens per request of every verify step and the steps of the
    # draft loop, and the targets run only the trained depth. So is a schedule that changes
    # it from step to step. The acceptance rule and SA drafting are knobs: they change
    # which drafts are kept, not which code runs.
    declined = _speculation_declined(spec_config, topology)

    # nemotron4-nano-preview's text model on GB300s (sm103), with one-model MTP-Eagle on the
    # checkpoint's MTP layer at its trained draft length or no speculative decoding.
    if compute_capability == (10, 3) and not differs and declined is None:
        # One line per target on purpose: test_modeling_v2_claims finds a family's targets
        # by reading these imports, and the formatter and isort would wrap them.
        # fmt: off
        # isort: off
        if topology == "tp1":
            from .nemotron4_nano_preview__sm_103__tp1.modeling import ModelingV2Nemotron4NanoPreviewSm103Tp1

            return ModelingV2Nemotron4NanoPreviewSm103Tp1
        if topology == "dep4":
            from .nemotron4_nano_preview__sm_103__dep4.modeling import ModelingV2Nemotron4NanoPreviewSm103Dep4

            return ModelingV2Nemotron4NanoPreviewSm103Dep4
        if topology == "tp4":
            from .nemotron4_nano_preview__sm_103__tp4.modeling import ModelingV2Nemotron4NanoPreviewSm103Tp4

            return ModelingV2Nemotron4NanoPreviewSm103Tp4
        # isort: on
        # fmt: on

    speculation = (
        "off"
        if spec_config is None
        else f"{type(spec_config).__name__} ({spec_config.spec_dec_mode.name})"
    )
    if declined is not None:
        speculation += f", which the target does not run: {declined}"
    not_implemented = (
        f"; {topology} is not implemented: {_NOT_IMPLEMENTED[topology]}"
        if topology in _NOT_IMPLEMENTED
        else ""
    )
    raise NoTarget(
        f"compute capability {compute_capability}, text config "
        + (f"differs: {', '.join(differs)}" if differs else "matches nemotron4-nano-preview")
        + f", {topology or 'parallel layout'} ({_parallel_layout(mapping)}), "
        f"speculative decoding {speculation}{not_implemented}; the nemotron4 targets are "
        "nemotron4-nano-preview's text model on sm103 at tp1 (one rank, no attention DP), "
        "dep4 (4 ranks, attention DP, the routed experts split four ways and nothing else "
        "divided) and tp4 (4 ranks, no attention DP, the mixers' heads and every MLP's and "
        "routed expert's intermediate width split four ways), each with one-model MTP-Eagle "
        f"on the checkpoint's MTP layer at max_draft_len={_DRAFT_LEN} without a "
        "draft_len_schedule, or without speculative decoding"
    )
