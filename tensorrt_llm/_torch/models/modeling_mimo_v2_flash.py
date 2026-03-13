# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""TensorRT-LLM modeling for MiMo-V2-Flash (Xiaomi).

MiMo-V2-Flash is a hybrid SWA/Full attention + MoE/Dense MLP model with:
- Asymmetric Q/K head_dim (192) vs V head_dim (128)
- Partial rotary embedding (partial_rotary_factor=0.334)
- Per-layer sliding window vs full attention with different num_kv_heads
  and rope_theta
- Attention sink bias for SWA layers
- Dense MLP (layer 0) + MoE with DeepSeekV3-style routing (layers 1-47)
- FP8 block-wise quantization with o_proj excluded

Note on asymmetric V head dimension:
  The TRT-LLM attention backends assume K and V have the same per-head
  dimension (head_dim). MiMo-V2-Flash has K head_dim=192 but V head_dim=128.
  To work around this, the QKV projection outputs V with v_head_dim=128 per
  head. Before the attention kernel, V is zero-padded per head to head_dim=192
  at runtime. After attention, the output is truncated from head_dim back to
  v_head_dim per head before the output projection. This wastes some KV cache
  memory but produces correct results with no FP8 scale approximation.
"""

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import DeepSeekV3MoeRoutingMethod, create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                               WeightsLoadingConfig)
from ..modules.rms_norm import RMSNorm
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import AuxStreamType
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class MiMoV2FlashAttention(Attention):
    """Custom attention for MiMo-V2-Flash.

    Handles asymmetric Q/K head_dim (192) vs V head_dim (128) by:
    1. Overriding qkv_proj to output Q(head_dim) + K(head_dim) + V(v_head_dim)
    2. Splitting QKV and padding V to head_dim at runtime before the kernel
    3. Truncating the attention output from head_dim to v_head_dim after the
       kernel
    4. Using a correctly-sized o_proj (input = num_heads * v_head_dim)

    Also handles: partial rotary embedding, per-layer sliding window with
    different rope_theta, and attention sink bias for SWA layers.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        is_swa = config.hybrid_layer_pattern[layer_idx] == 1

        if is_swa:
            num_attention_heads = config.swa_num_attention_heads
            num_key_value_heads = config.swa_num_key_value_heads
            qk_head_dim = config.swa_head_dim
            v_head_dim = config.swa_v_head_dim
            rope_theta = config.swa_rope_theta
        else:
            num_attention_heads = config.num_attention_heads
            num_key_value_heads = config.num_key_value_heads
            qk_head_dim = config.head_dim
            v_head_dim = config.v_head_dim
            rope_theta = config.rope_theta

        self.is_swa = is_swa
        self._v_head_dim = v_head_dim
        self._qk_head_dim = qk_head_dim

        # Partial rotary: only a fraction of head_dim gets RoPE
        rope_dim = int(qk_head_dim * config.partial_rotary_factor)

        rope_params = RopeParams(
            theta=rope_theta,
            dim=rope_dim,
            max_positions=config.max_position_embeddings,
        )

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )

        # SWA layers use sliding window attention (not chunked attention).
        # Store window size to pass at forward time, following Gemma3 pattern.
        self._attention_window_size = None
        if is_swa:
            self._attention_window_size = getattr(
                config, 'sliding_window_size',
                getattr(config, 'sliding_window', None))

        # Call parent __init__ with head_dim=qk_head_dim.
        # The parent creates qkv_proj and o_proj with head_dim for BOTH K and V.
        # We override both below: qkv_proj to use v_head_dim for V, and o_proj
        # to use v_head_dim for input. The attention backend still uses
        # head_dim=192 for V in the KV cache (we pad at runtime).
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        # After parent init: self.head_dim=192, self.num_heads and
        # self.num_key_value_heads are TP-divided, self.q_size=num_heads*192,
        # self.kv_size=num_kv_heads*192.
        # We need V at v_head_dim=128 in the projection, padded to 192 at runtime.

        self.v_head_dim = v_head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim
        # Keep kv_size for K (num_kv_heads * head_dim=192)
        # self.kv_size is already set by parent for K

        # Save mapping from parent-created qkv_proj
        mapping = self.qkv_proj.mapping

        # Override qkv_proj with correct asymmetric output size:
        # Q(num_heads * head_dim) + K(num_kv_heads * head_dim) + V(num_kv_heads * v_head_dim)
        total_qkv = (self.tp_size * self.q_size +
                      self.tp_size * self.kv_size +
                      self.tp_size * self.v_size)

        qkv_shard_indices_mapping = {
            "q": (0, self.q_size),
            "k": (self.q_size, self.kv_size),
            "v": (self.q_size + self.kv_size, self.v_size),
        }

        self.qkv_proj = Linear(
            config.hidden_size,
            total_qkv,
            bias=config.attention_bias,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            fused_weight_shard_indices_mapping=qkv_shard_indices_mapping,
        )

        # Override o_proj: input = num_heads * v_head_dim, not num_heads * head_dim
        o_proj_input_size = self.tp_size * self.num_heads * self.v_head_dim
        self.o_proj = Linear(
            o_proj_input_size,
            config.hidden_size,
            bias=self.dense_bias,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            lora=self.o_lora,
            reduce_output=True,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
        )

        # Update LoRA layer sizes for the corrected split
        self.splitted_qkv_lora = LoraLayer([
            LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V
        ], [self.q_size, self.kv_size, self.v_size])
        self.fused_qkv_lora = LoraLayer(
            [LoraModuleType.ATTENTION_QKV],
            [self.q_size + self.kv_size + self.v_size])

        # Attention sink bias for SWA layers
        add_swa_sink = getattr(config, 'add_swa_attention_sink_bias', False)
        add_full_sink = getattr(config, 'add_full_attention_sink_bias', False)
        if (is_swa and add_swa_sink) or (not is_swa and add_full_sink):
            self.attention_sink_bias = nn.Parameter(
                torch.empty(self.num_heads, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.attention_sink_bias = None

    def split_qkv(self, q, k=None, v=None):
        """Override to handle asymmetric K/V sizes.

        K uses kv_size = num_kv_heads * head_dim (192).
        V uses v_size = num_kv_heads * v_head_dim (128).
        """
        if k is None and v is None:
            q, k, v = q.split(
                [self.q_size, self.kv_size, self.v_size], dim=-1)
        return q, k, v

    def _pad_v_to_head_dim(self, v: torch.Tensor) -> torch.Tensor:
        """Pad V from v_head_dim to head_dim per KV head.

        V has shape [num_tokens, num_kv_heads * v_head_dim].
        Returns shape [num_tokens, num_kv_heads * head_dim].
        """
        if self.v_head_dim == self.head_dim:
            return v
        num_tokens = v.shape[0]
        # Reshape to [num_tokens, num_kv_heads, v_head_dim]
        v = v.view(num_tokens, self.num_key_value_heads, self.v_head_dim)
        # Pad last dim from v_head_dim to head_dim with zeros
        pad_size = self.head_dim - self.v_head_dim
        v_padded = F.pad(v, (0, pad_size), value=0.0)
        # Reshape back to [num_tokens, num_kv_heads * head_dim]
        return v_padded.view(num_tokens, -1)

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, "Fp4QuantizedTensor"],
        attn_metadata: AttentionMetadata,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with runtime V padding and output truncation.

        Steps:
        1. QKV projection outputs Q(head_dim) + K(head_dim) + V(v_head_dim)
        2. Split QKV, apply RoPE to Q and K
        3. Pad V from v_head_dim to head_dim (required by attention kernel)
        4. Run attention with padded V (head_dim per head in KV cache)
        5. Truncate output from head_dim to v_head_dim per head
        6. Pass through o_proj (sized for v_head_dim)
        """
        # QKV projection
        qkv = self.qkv_proj(hidden_states)

        if bool(lora_params):
            qkv_lora = self.splitted_qkv_lora(hidden_states, lora_params,
                                               self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora
            qkv_lora = self.fused_qkv_lora(hidden_states, lora_params,
                                            self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

        # Split into Q, K, V (V has v_head_dim per head)
        q, k, v = self.split_qkv(qkv)

        # Apply partial RoPE to Q and K
        if not self.rope_fusion and position_ids is not None:
            q, k = self.rotary_emb(position_ids, [q, k])

        # Pad V from v_head_dim to head_dim for the attention kernel
        v = self._pad_v_to_head_dim(v)

        # Convert to the format expected by the backend (fused or separate)
        q, k, v = self.convert_qkv(q, k, v)

        # Run attention (V now has head_dim per head in the kernel)
        attn_output = self.forward_impl(
            q,
            k,
            v,
            attn_metadata,
            attention_mask,
            attention_window_size=self._attention_window_size,
            attention_mask_data=None,
            mrope_config=None,
            attention_sinks=self.attention_sink_bias,
        )

        # Truncate output from head_dim to v_head_dim per head
        num_tokens = attn_output.shape[0]
        attn_output = attn_output.view(num_tokens, self.num_heads,
                                       self.head_dim)
        attn_output = attn_output[:, :, :self.v_head_dim].contiguous()
        attn_output = attn_output.view(num_tokens, -1)

        # Output projection
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params,
                                  lora_params=lora_params,
                                  layer_idx=self.layer_idx)
        return attn_output


class MiMoV2FlashMoE(nn.Module):
    """MoE module for MiMo-V2-Flash layers 1-47.

    Uses DeepSeekV3-style sigmoid + noaux_tc routing with
    e_score_correction_bias.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        # Router gate operates in FP32
        self.gate = Linear(
            self.hidden_dim,
            self.num_experts,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
        )

        # e_score_correction_bias for noaux_tc routing
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        routed_scaling_factor = getattr(config, 'routed_scaling_factor', None)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=self.top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=routed_scaling_factor,
            callable_e_score_correction_bias=lambda: self.
            e_score_correction_bias,
        )

        self.experts = create_moe(
            routing_method=routing_method,
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=config.moe_intermediate_size,
            dtype=config.torch_dtype,
            reduce_results=True,
            model_config=model_config,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            layer_idx=layer_idx,
        )

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        """Load e_score_correction_bias from the weight dict."""
        assert len(weights) == 1
        w = weights[0]
        if "e_score_correction_bias" in w:
            self.e_score_correction_bias.copy_(
                w["e_score_correction_bias"][:].to(
                    self.e_score_correction_bias.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        # Router expects FP32 input
        hidden_states_f32 = hidden_states.to(torch.float32)
        router_logits = self.gate(hidden_states_f32)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False,
        )
        return final_hidden_states


class MiMoV2FlashDecoderLayer(DecoderLayer):
    """Decoder layer for MiMo-V2-Flash.

    Handles hybrid attention (full/SWA) and hybrid MLP (dense/MoE).
    Uses the fused residual + norm pattern for performance.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config

        self.self_attn = MiMoV2FlashAttention(
            model_config=model_config,
            layer_idx=layer_idx,
        )

        # Layer 0 uses dense MLP, layers 1-47 use MoE (from moe_layer_freq)
        is_moe = (hasattr(config, 'moe_layer_freq')
                   and config.moe_layer_freq[layer_idx] == 1)
        if is_moe:
            self.mlp = MiMoV2FlashMoE(
                model_config=model_config,
                aux_stream=aux_stream,
                layer_idx=layer_idx,
            )
        else:
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                activation=F.silu,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
            )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.torch_dtype,
        )

        self.is_moe = is_moe

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        # Fused residual + norm pattern
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Pre-MLP norm with fused residual
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        # MLP or MoE
        if self.is_moe:
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class MiMoV2FlashModel(DecoderModel):
    """Base model for MiMo-V2-Flash."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config

        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList([
            MiMoV2FlashDecoderLayer(model_config, layer_idx, self.aux_stream)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # After layer creation, override num_key_value_heads to a per-layer
        # list for correct KV cache allocation. Full-attention layers use
        # config.num_key_value_heads, SWA layers use
        # config.swa_num_key_value_heads.
        # The get_bindings_model_config checks if num_key_value_heads is
        # a list and sets num_kv_heads_per_layer accordingly.
        full_kv_heads = config.num_key_value_heads
        swa_kv_heads = getattr(config, 'swa_num_key_value_heads',
                               full_kv_heads)
        if full_kv_heads != swa_kv_heads:
            config.num_key_value_heads = [
                swa_kv_heads
                if config.hybrid_layer_pattern[i] == 1 else full_kv_heads
                for i in range(config.num_hidden_layers)
            ]

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds "
                "at the same time, and must specify either one")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None

        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("MiMoV2FlashForCausalLM")
class MiMoV2FlashForCausalLM(
        DecoderModelForCausalLM[MiMoV2FlashModel, PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(
            MiMoV2FlashModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def __post_init__(self):
        # Ensure o_proj is excluded from FP8 quantization (it is bfloat16)
        quant_config = self.model_config.quant_config
        if quant_config is not None:
            if quant_config.exclude_modules is None:
                quant_config.exclude_modules = []
            for i in range(self.config.num_hidden_layers):
                exclude_name = f"model.layers.{i}.self_attn.o_proj"
                if exclude_name not in quant_config.exclude_modules:
                    quant_config.exclude_modules.append(exclude_name)
        super().__post_init__()

    @staticmethod
    def _trim_fp8_block_scale(weight_shape_0, scale, block_size=128):
        """Trim FP8 block scale rows to match the actual weight size.

        Some quantization tools pad scales to per-head boundaries, giving
        more scale rows than ceil(weight_rows / block_size). This trims
        the scale to the exact number of blocks needed by the weight.
        """
        expected_rows = math.ceil(weight_shape_0 / block_size)
        if scale.shape[0] > expected_rows:
            return scale[:expected_rows]
        return scale

    def load_weights(self, weights: Dict, **kwargs):
        """Custom weight loading for MiMo-V2-Flash.

        Handles:
        - FP8 block scale trimming for Q/K/V projections (checkpoint may
          have per-head-aligned scales with more rows than expected).
        - MoE expert weight renaming (gate_proj->w1, up_proj->w3, down_proj->w2).
        - MoE e_score_correction_bias renaming from gate submodule.
        - MTP weight filtering.

        The default _load_weights_impl handles:
        - QKV fusion via params_map (qkv_proj -> [q_proj, k_proj, v_proj]).
        - Dense MLP gate/up fusion via params_map (gate_up_proj -> [gate_proj, up_proj]).
        - MoE expert weight stacking via the MoE backend's load_weights.
        - All other modules: loaded directly by matching name prefix.
        """
        transformed = {}

        for key, value in weights.items():
            # Skip MTP weights (multi-token prediction, not needed)
            if '.mtp.' in key:
                continue

            new_key = key

            # Rename e_score_correction_bias from gate submodule to mlp level.
            # HF: model.layers.X.mlp.gate.e_score_correction_bias
            # TRT-LLM module: model.layers.X.mlp.e_score_correction_bias
            if '.mlp.gate.e_score_correction_bias' in new_key:
                new_key = new_key.replace('.mlp.gate.e_score_correction_bias',
                                         '.mlp.e_score_correction_bias')

            # Rename MoE expert projections from HF naming to TRT-LLM MoE
            # naming convention used by the fused MoE backends.
            # HF: model.layers.X.mlp.experts.{e}.gate_proj -> w1
            # HF: model.layers.X.mlp.experts.{e}.up_proj   -> w3
            # HF: model.layers.X.mlp.experts.{e}.down_proj -> w2
            if '.mlp.experts.' in new_key:
                new_key = new_key.replace('.gate_proj.', '.w1.')
                new_key = new_key.replace('.up_proj.', '.w3.')
                new_key = new_key.replace('.down_proj.', '.w2.')

            transformed[new_key] = value

        # Trim FP8 block scales for Q/K/V projections.
        # The checkpoint quantization tool may produce per-head-aligned
        # block scales with more rows than ceil(weight_rows / block_size).
        # The standard fused QKV loader concatenates individual scales,
        # and the sum of over-sized scales won't fit the fused module's
        # weight_scale tensor. Trimming ensures correct alignment.
        scale_suffix = '.weight_scale_inv'
        for key in list(transformed.keys()):
            if key.endswith(scale_suffix):
                # Find the corresponding weight to get its actual row count
                weight_key = key[:-len(scale_suffix)] + '.weight'
                if weight_key in transformed:
                    w = transformed[weight_key]
                    w_rows = w.shape[0] if isinstance(
                        w, torch.Tensor) else w.get_shape()[0]
                    s = transformed[key]
                    s_val = s if isinstance(s, torch.Tensor) else s[:]
                    trimmed = self._trim_fp8_block_scale(w_rows, s_val)
                    if trimmed is not s_val:
                        transformed[key] = trimmed

        super().load_weights(transformed, **kwargs)
