# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: nemotron4-nano-preview (text) / sm_103 / tp4 — self-contained modeling code.

The text model of the Nemotron4ForConditionalGeneration checkpoint: 44 blocks
of hidden 3072 on three wide-residual streams -- 12 GDP mixers alone, 4 dense
relu2 MLPs, 8 attention and 20 GDP mixers each beside a latent MoE (512 routed
experts at top-10 in a 768-wide latent space, plus a shared expert) -- and the
checkpoint's MTP layer, live only under a `speculative_config`. Every weight is
the checkpoint's bf16, declared in the sibling weights.py.

`tp4` is tensor parallel over the 4 GPUs of one node, without attention DP:
every rank runs every request. A rank holds 6 of the 24 GDP heads, 4 of the 16
query heads over the one KV head, which every rank keeps whole, a quarter of
every MLP's intermediate width -- the dense MLPs', the shared experts' and each
of the 512 routed experts' -- and a quarter of the vocabulary's lm_head rows.
The wide residual streams, the norms, the router and the latent projections
are replicated, so every branch ends in a partial sum over the ranks.

The target routes and loads its weights; its forward is not implemented, so the
first step raises `NotImplementedError`.
"""

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM

from . import weights as _weights

_FORWARD_MISSING = (
    "nemotron4-nano-preview / sm_103 / tp4 has no forward yet: the target routes "
    "and loads its weights only"
)


class Nemotron4NanoPreviewTp4ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None

    def post_load_weights(self) -> None:
        """Construct the two targets, now that the weights are real.

        Must only be called after meta init is over, never from `__init__`, where the
        shell's containers are still meta.

        Named for the shell hook that drives it. Not an override: that hook lives on
        DecoderModelForCausalLM, and a core derives from DecoderModel, which has no
        such method -- so there is no super() to call here.
        """
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment, and a predicate that reached for a second field
        would be reading one this call is not guaranteed to carry. A mixed batch -- context
        and generation rows in one step -- goes to prefill, so prefill is the general case
        and decode the specialization that may assume it has no context rows.
        """
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Route the step, and nothing else."""
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: Nemotron4NanoPreviewTp4ModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(_FORWARD_MISSING)


class DecodeTarget:
    """The specialization: routed to only when there are no context rows."""

    def __init__(self, core: Nemotron4NanoPreviewTp4ModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(_FORWARD_MISSING)


class ModelingV2Nemotron4NanoPreviewSm103Tp4(
    DecoderModelForCausalLM[Nemotron4NanoPreviewTp4ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        # The composite config keeps the text model's widths under text_config.
        cfg = model_config.pretrained_config
        text = cfg.text_config
        # The engine reads the depth off the composite config when it sizes its caches, but
        # text_config derives it (a property over layer_types) and ModelConfig's mirror of
        # text_config onto the composite copies stored fields only.
        cfg.num_hidden_layers = text.num_hidden_layers
        # Without attention DP the base class builds lm_head column-parallel: each rank
        # holds its quarter of the vocabulary's rows.
        super().__init__(
            Nemotron4NanoPreviewTp4ModelingV2(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
