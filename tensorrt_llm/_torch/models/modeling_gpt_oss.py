from typing import Dict, List, Optional, Type

import torch
from torch import nn

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..distributed import AllReduce, AllReduceParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (CutlassFusedMoE, RenormalizeMoeRoutingMethod,
                                 TRTLLMGenFusedMoE, create_moe, get_moe_cls)
from ..modules.fused_moe.interface import MoE, MoEWeightLoadingMode
from ..modules.fused_moe.routing import BaseMoeRoutingMethod
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)

try:
    from transformers import GptOssConfig
except ImportError:
    from transformers import AutoConfig, PretrainedConfig

    class GptOssConfig(PretrainedConfig):
        model_type = "gpt_oss"

    AutoConfig.register(GptOssConfig.model_type, GptOssConfig)


class GptOssGate(nn.Module):
    """Router gate with bias for GptOss MoE."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: Optional[torch.dtype] = None,
        moe_backend_cls: Type[MoE] = CutlassFusedMoE,
    ):
        super().__init__()
        self.top_k = top_k
        self.moe_backend_cls = moe_backend_cls
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=dtype),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=dtype),
            requires_grad=False,
        )
        self.out_dtype = dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.ops.trtllm.cublas_mm(hidden_states,
                                             self.weight.t(),
                                             bias=self.bias,
                                             out_dtype=self.out_dtype)
        return logits

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        assert len(weights) == 1
        w = weights[0].get("weight")
        b = weights[0].get("bias")
        if not allow_partial_loading:
            assert w is not None
        if w is not None:
            self.weight.copy_(w[:])
        if b is not None:
            self.bias.copy_(b[:])

    @property
    def routing_method(self) -> BaseMoeRoutingMethod:
        output_dtype = (torch.bfloat16 if self.moe_backend_cls
                        == TRTLLMGenFusedMoE else torch.float32)
        return RenormalizeMoeRoutingMethod(top_k=self.top_k,
                                           output_dtype=output_dtype)


class GptOssMoE(nn.Module):
    """MoE module with custom SwiGLU activation and bias."""

    def __init__(
        self,
        model_config: ModelConfig[GptOssConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.mapping = model_config.mapping

        self.allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy)

        self.gate = GptOssGate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=config.torch_dtype,
            moe_backend_cls=get_moe_cls(model_config),
        )

        swiglu_limit = getattr(config, 'swiglu_limit', 7.0)
        swiglu_alpha = torch.full((self.num_experts,),
                                  1.702,
                                  dtype=torch.float32,
                                  device='cuda')
        swiglu_beta = torch.full((self.num_experts,),
                                 1.0,
                                 dtype=torch.float32,
                                 device='cuda')
        swiglu_limit_t = torch.full((self.num_experts,),
                                    swiglu_limit,
                                    dtype=torch.float32,
                                    device='cuda')
        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=config.intermediate_size,
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            layer_idx=layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            bias=True,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit_t,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
        )

        if self.allreduce is not None:
            final_hidden_states = self.allreduce(
                final_hidden_states, all_reduce_params=all_reduce_params)

        return final_hidden_states.view(orig_shape)


class GptOssAttention(Attention):
    """Attention with attention sinks and per-layer sliding window."""

    def __init__(
        self,
        model_config: ModelConfig[GptOssConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        self._attention_window_size = (config.sliding_window
                                       if self.is_sliding else None)

        rope_params = RopeParams.from_config(config)
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        # Attention sinks: one value per local attention head (TP-adjusted).
        # self.num_heads is already divided by tp_size in Attention.__init__.
        self.sinks = nn.Parameter(
            torch.empty(self.num_heads, dtype=config.torch_dtype),
            requires_grad=False,
        )

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: Optional[PredefinedAttentionMask] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (
                PredefinedAttentionMask.SLIDING_WINDOW_CAUSAL
                if self.is_sliding else PredefinedAttentionMask.CAUSAL)
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            attention_window_size=self._attention_window_size,
            attention_sinks=self.sinks,
            **kwargs,
        )


class GptOssDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[GptOssConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping

        self.self_attn = GptOssAttention(model_config, layer_idx=layer_idx)
        self.mlp = GptOssMoE(model_config,
                              aux_stream=aux_stream,
                              layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, attn_metadata)

        return hidden_states, residual


class GptOssModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[GptOssConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.aux_stream = torch.cuda.Stream()

        if model_config.mapping.enable_attention_dp:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                enable_torch_compile_for_embedding=model_config.
                enable_torch_compile_for_embedding,
            )
        else:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                enable_torch_compile_for_embedding=model_config.
                enable_torch_compile_for_embedding,
            )

        self.layers = nn.ModuleList([
            GptOssDecoderLayer(model_config, layer_idx, self.aux_stream)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
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
                "You cannot specify both input_ids and inputs_embeds at "
                "the same time, and must specify either one.")

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
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("GptOssForCausalLM")
class GptOssForCausalLM(DecoderModelForCausalLM[GptOssModel, GptOssConfig]):

    def __init__(self, model_config: ModelConfig[GptOssConfig]):
        config = model_config.pretrained_config
        super().__init__(
            GptOssModel(model_config),
            config=model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )

    def load_weights(self, weights: Dict, **kwargs):
        transformed = self._transform_weights(weights)
        super().load_weights(transformed, **kwargs)

    def _transform_weights(self, weights: Dict) -> Dict:
        """Transform HF checkpoint weights to TRT-LLM FUSED_GATE_UP_PROJ format.

        Key transformations:
        1. Rename mlp.router -> mlp.gate
        2. De-interleave gate_up_proj (interleaved gate/up rows), re-concat
           as [w1=up, w3=gate], reshape MXFP4 blocks, transpose for the
           FUSED_GATE_UP_PROJ protocol
        3. Reshape + transpose down_proj for FUSED_GATE_UP_PROJ protocol
        4. TP-slice attention sinks to local heads
        """
        mapping = self.model_config.mapping
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        tp_rank = mapping.tp_rank

        transformed = {}
        for key, value in weights.items():
            # Rename router -> gate
            if '.mlp.router.' in key:
                new_key = key.replace('.mlp.router.', '.mlp.gate.')
                transformed[new_key] = value
            # De-interleave + fuse gate_up expert weights
            elif '.mlp.experts.gate_up_proj_blocks' in key:
                self._fuse_gate_up(key, value, transformed, 'gate_up_proj')
            elif '.mlp.experts.gate_up_proj_scales' in key:
                self._fuse_gate_up(key, value, transformed,
                                   'gate_up_proj_weight_scale')
            elif '.mlp.experts.gate_up_proj_bias' in key:
                self._fuse_gate_up(key, value, transformed,
                                   'gate_up_proj.bias')
            # Transpose down_proj expert weights
            elif '.mlp.experts.down_proj_blocks' in key:
                self._fuse_down(key, value, transformed, 'down_proj')
            elif '.mlp.experts.down_proj_scales' in key:
                self._fuse_down(key, value, transformed,
                                'down_proj_weight_scale')
            elif '.mlp.experts.down_proj_bias' in key:
                self._fuse_down(key, value, transformed, 'down_proj.bias')
            # TP-slice attention sinks to local heads
            elif '.self_attn.sinks' in key:
                if tp_size > 1:
                    num_heads = value.shape[0]
                    heads_per_tp = num_heads // tp_size
                    start = tp_rank * heads_per_tp
                    value = value[start:start + heads_per_tp]
                transformed[key] = value
            else:
                transformed[key] = value
        return transformed

    def _fuse_gate_up(self, key: str, value: torch.Tensor,
                      transformed: Dict, target_suffix: str):
        """De-interleave gate/up, re-concat as [w1, w3], prepare for
        FUSED_GATE_UP_PROJ protocol.

        The HF checkpoint stores interleaved gate/up rows:
          row 0 = gate, row 1 = up, row 2 = gate, row 3 = up, ...
        De-interleave: even rows = gate (w3), odd rows = up (w1).
        Re-concat as [w1=up, w3=gate] (FUSED_GATE_UP_PROJ chunk order).

        For blocks/scales the loader does .transpose(0,1).chunk(2,dim=0),
        so we store as [E, in_dim, 2*inter] (transposed) to match.
        For bias the loader only does .chunk(2,dim=0), no transpose.
        """
        prefix = key.rsplit('.mlp.experts.', 1)[0] + '.mlp.experts'
        new_key = f'{prefix}.{target_suffix}'

        if target_suffix == 'gate_up_proj':
            # blocks: [E, 2*inter, num_blocks, block_size]
            gate = value[:, 0::2, :, :]
            up = value[:, 1::2, :, :]
            # reshape to [E, inter, packed_hidden]
            gate = gate.reshape(gate.shape[0], gate.shape[1], -1)
            up = up.reshape(up.shape[0], up.shape[1], -1)
            # concat [w1=up, w3=gate] → [E, 2*inter, packed_hidden]
            fused = torch.cat([up, gate], dim=1)
            # transpose for FUSED_GATE_UP_PROJ protocol → [E, packed_hidden, 2*inter]
            transformed[new_key] = fused.transpose(1, 2).contiguous()
        elif target_suffix == 'gate_up_proj_weight_scale':
            # scales: [E, 2*inter, num_blocks]
            gate = value[:, 0::2, :]
            up = value[:, 1::2, :]
            # concat [w1=up, w3=gate] → [E, 2*inter, num_blocks]
            fused = torch.cat([up, gate], dim=1)
            # transpose for protocol → [E, num_blocks, 2*inter]
            transformed[new_key] = fused.transpose(1, 2).contiguous()
        else:
            # bias: [E, 2*inter] — no transpose, only chunk in loader
            gate = value[:, 0::2]
            up = value[:, 1::2]
            # concat [w1=up, w3=gate] → [E, 2*inter]
            transformed[new_key] = torch.cat([up, gate], dim=1)

    def _fuse_down(self, key: str, value: torch.Tensor, transformed: Dict,
                   target_suffix: str):
        """Prepare down_proj for FUSED_GATE_UP_PROJ protocol.

        For blocks/scales the loader does .transpose(0,1), so we store
        transposed. For bias, no transformation needed.
        """
        prefix = key.rsplit('.mlp.experts.', 1)[0] + '.mlp.experts'
        new_key = f'{prefix}.{target_suffix}'

        if target_suffix == 'down_proj':
            # blocks: [E, hidden, num_blocks, block_size]
            # reshape → [E, hidden, packed_inter]
            value = value.reshape(value.shape[0], value.shape[1], -1)
            # transpose for protocol → [E, packed_inter, hidden]
            transformed[new_key] = value.transpose(1, 2).contiguous()
        elif target_suffix == 'down_proj_weight_scale':
            # scales: [E, hidden, num_blocks]
            # transpose for protocol → [E, num_blocks, hidden]
            transformed[new_key] = value.transpose(1, 2).contiguous()
        else:
            # bias: [E, hidden] — no transformation
            transformed[new_key] = value
