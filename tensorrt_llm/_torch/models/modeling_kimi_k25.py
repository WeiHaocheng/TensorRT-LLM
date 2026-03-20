"""
TensorRT-LLM modeling implementation for Kimi-K2.5.

Kimi-K2.5 is a multimodal model composed of:
1. Vision Tower (MoonViT3dPretrainedModel): 3D ViT with spatial-temporal attention
2. Multi-Modal Projector (KimiK25MultiModalProjector): 2-layer MLP with pre-norm
3. Language Model (DeepseekV3ForCausalLM): DeepseekV3-architecture LM (reused)
"""

import copy
import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)

from ..._utils import nvtx_range
from ...inputs import (BaseMultimodalDummyInputsBuilder,
                       BaseMultimodalInputProcessor, ExtraProcessedInputs,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import filter_weights, register_auto_model

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


# ---------------------------------------------------------------------------
# Vision tower helper functions
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int,
                                        pos: np.ndarray) -> np.ndarray:
    """Generate 1D sincos positional embedding from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    """Generate 1D sincos positional embedding."""
    grid_t = np.arange(t_size, dtype=np.float32)
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)


def _get_rope_shape_impl(org: torch.Tensor, interpolation_mode: str,
                          shape: Tuple[int, int]) -> torch.Tensor:
    return (F.interpolate(
        org.permute((2, 0, 1)).unsqueeze(0),
        size=shape,
        mode=interpolation_mode,
    ).squeeze(0).permute((1, 2, 0)).flatten(end_dim=1))


def _apply_rope(xq: torch.Tensor, xk: torch.Tensor,
                freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary position embedding to query and key tensors."""
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Vision tower modules
# ---------------------------------------------------------------------------

class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """2D learnable position embedding with fixed temporal sincos extension."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_get_1d_sincos_pos_embed(
                self.dim, self.num_frames)).float().unsqueeze(1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames
            if (h, w) == (self.height, self.width):
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = _get_rope_shape_impl(self.weight,
                                                   self.interpolation_mode,
                                                   (h, w))
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) +
                    self.time_weight[0:t])

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        return x + torch.cat(pos_embs)


class MoonVision3dPatchEmbed(nn.Module):
    """3D patch embedding for vision tower: Conv2d + learnable 2D pos emb."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int = 14,
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim,
                              out_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
            )
        else:
            raise NotImplementedError(
                f"Unsupported pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding for vision encoder."""

    def __init__(self, dim: int, max_height: int, max_width: int,
                 theta_base: float = 10000.0):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self,
                               device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[:(self.dim // 4)].float().to(
            device)
        freqs = 1.0 / (self.theta_base**(dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1),
             y_cis.unsqueeze(dim=-1)], dim=-1)
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer("freqs_cis",
                                 self._precompute_freqs_cis(device),
                                 persistent=False)

        shapes = grid_thws.tolist()
        freqs_cis = torch.cat([
            self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
            for t, h, w in shapes
        ],
                              dim=0)
        return freqs_cis


class MoonViTMLP(nn.Module):
    """Two-layer MLP for vision encoder blocks."""

    def __init__(self, hidden_dim: int, mlp_dim: int, activation, bias: bool = True):
        super().__init__()
        self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=bias)
        self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        x = self.fc1(x)
        return x


class MoonViTEncoderLayer(nn.Module):
    """Single encoder layer for MoonViT with 2D RoPE attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        activation=None,
        attn_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MoonViTMLP(hidden_dim, mlp_dim,
                              activation if activation is not None else F.gelu,
                              bias=True)
        self.wqkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def _attention(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = x.size(0)
        xqkv = self.wqkv(x)
        xqkv = xqkv.view(seq_length, 3, self.num_heads, self.head_dim)
        xq, xk, xv = torch.unbind(xqkv, dim=1)

        # Apply 2D RoPE
        xq, xk = _apply_rope(xq, xk, rope_freqs_cis)

        # Use flash attention if available, otherwise fall back to SDPA
        try:
            from flash_attn import flash_attn_varlen_func
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_out = flash_attn_varlen_func(
                xq,
                xk,
                xv,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )
        except ImportError:
            # Fallback: simple batched SDPA using cu_seqlens
            attn_out = self._sdpa_varlen(xq, xk, xv, cu_seqlens)

        attn_out = attn_out.reshape(seq_length, self.num_heads * self.head_dim)
        return self.wo(attn_out)

    def _sdpa_varlen(self, xq: torch.Tensor, xk: torch.Tensor,
                     xv: torch.Tensor,
                     cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Fallback SDPA-based variable-length attention."""
        outputs = []
        num_seqs = cu_seqlens.shape[0] - 1
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            q = xq[start:end].transpose(0, 1).unsqueeze(0)
            k = xk[start:end].transpose(0, 1).unsqueeze(0)
            v = xv[start:end].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self._attention(hidden_states, cu_seqlens,
                                        rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):
    """Full encoder stack for MoonViT 3D."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
    ) -> None:
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.ModuleList(
            [MoonViTEncoderLayer(**block_cfg) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device)

        lengths = torch.cat((
            torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        ))
        cu_seqlens = lengths.to(hidden_states.device).cumsum(
            dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens,
                                  rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def _tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: Tuple[int, int] = (2, 2),
) -> List[torch.Tensor]:
    """Temporal pooling patch merger."""
    kh, kw = merge_kernel_size
    lengths = (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).tolist()
    seqs = x.split(lengths, dim=0)

    outputs = []
    for seq, (t, h, w) in zip(seqs, grid_thws.tolist()):
        nh, nw = h // kh, w // kw
        v = seq.view(t, nh, kh, nw, kw, -1)
        v = v.mean(dim=0)  # temporal pooling
        out = v.permute(0, 2, 1, 3, 4).reshape(nh * nw, kh * kw, -1)
        outputs.append(out)

    return outputs


class MoonViT3dPretrainedModel(nn.Module):
    """Main vision tower model: 3D ViT with temporal pooling."""

    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        self.config = config
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        # Map KimiK25VisionConfig attributes to standard names
        hidden_size = getattr(config, 'vt_hidden_size',
                              getattr(config, 'hidden_size', 1152))
        num_attention_heads = getattr(config, 'vt_num_attention_heads',
                                      getattr(config, 'num_attention_heads', 16))
        intermediate_size = getattr(config, 'vt_intermediate_size',
                                    getattr(config, 'intermediate_size', 4304))
        num_hidden_layers = getattr(config, 'vt_num_hidden_layers',
                                    getattr(config, 'num_hidden_layers', 27))

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=hidden_size,
            num_layers=num_hidden_layers,
            block_cfg={
                "num_heads": num_attention_heads,
                "hidden_dim": hidden_size,
                "mlp_dim": intermediate_size,
                "activation": lambda x: F.gelu(x, approximate="tanh"),
                "attn_bias": True,
            },
        )

    def forward(self, pixel_values: torch.Tensor,
                grid_thws: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == "sd2_tpool":
            hidden_states = _tpool_patch_merger(
                hidden_states,
                grid_thws,
                merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f"Unsupported merge_type: {self.merge_type}")
        return hidden_states


# ---------------------------------------------------------------------------
# Multi-modal projector
# ---------------------------------------------------------------------------

class KimiK25MultiModalProjector(nn.Module):
    """Multi-modal projector: pre-norm + patch flattening + 2-layer MLP."""

    def __init__(self, config):
        super().__init__()
        # Hidden size after patch merging
        vt_hidden_size = getattr(config, 'vt_hidden_size',
                                 getattr(config, 'hidden_size', 1152))
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = vt_hidden_size * merge_h * merge_w

        self.pre_norm = nn.LayerNorm(
            vt_hidden_size,
            eps=getattr(config, 'projector_ln_eps', 1e-5))

        # Output dimension: text_hidden_size (7168)
        text_hidden_size = getattr(config, 'text_hidden_size', 7168)

        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(self.hidden_size, text_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(
            -1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Vision tower + projector forward helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _mm_projector_forward(mm_projector: nn.Module,
                          vt_output: List[torch.Tensor]) -> List[torch.Tensor]:
    """Apply MM projector to vision tower outputs."""
    num_embedding_list = [x.shape[0] for x in vt_output]
    batched = torch.cat(vt_output, dim=0)
    proj_out = mm_projector(batched)
    proj_out = proj_out.reshape(-1, proj_out.shape[-1])
    proj_out = torch.split(proj_out, num_embedding_list)
    return list(proj_out)


@torch.inference_mode()
def _vision_tower_forward(
    vision_tower: nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    mm_projector: nn.Module,
) -> List[torch.Tensor]:
    """Run vision tower forward + projector."""
    vt_outputs = vision_tower(pixel_values, grid_thw)
    return _mm_projector_forward(mm_projector, vt_outputs)


# ---------------------------------------------------------------------------
# Input processor for multimodal data
# ---------------------------------------------------------------------------

class KimiK25InputProcessor(BaseMultimodalInputProcessor,
                            BaseMultimodalDummyInputsBuilder):

    def __init__(self,
                 model_path: str,
                 config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True,
                 **kwargs):
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         **kwargs)
        self._config = config
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self.use_fast,
            trust_remote_code=trust_remote_code)

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {})

        images = mm_data.get("image")

        # Convert to Kimi K2.5 processor format (medias + text)
        medias = []
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                medias.append({"type": "image", "image": img})

        processor_output = self.processor(
            medias=medias,
            text=text_prompt,
            return_tensors="pt",
        ).to(dtype=self._dtype)

        input_ids = processor_output["input_ids"]
        pixel_values = processor_output.get("pixel_values")
        grid_thws = processor_output.get("grid_thws")

        # The framework inserts 1 <|media_pad|> token per image, but the
        # vision tower produces grid_thws.prod(-1) / merge_factor embeddings
        # per image. Expand each single placeholder to the correct count.
        if grid_thws is not None and len(grid_thws) > 0:
            media_token_id = getattr(self._config,
                                     "media_placeholder_token_id", 163605)
            # merge_kernel_size=[2,2] reduces spatial dims by 2x2=4
            vt_cfg = getattr(self._config, "vision_tower_config", None)
            if vt_cfg is not None:
                mk = vt_cfg.get("merge_kernel_size", [2, 2])
                if isinstance(mk, list):
                    merge_factor = mk[0] * mk[1]
                else:
                    merge_factor = mk * mk
            else:
                merge_factor = 4
            num_tokens_per_image = (grid_thws.prod(dim=-1) //
                                    merge_factor).tolist()
            expanded_ids = []
            img_idx = 0
            for tid in input_ids[0]:
                if tid.item(
                ) == media_token_id and img_idx < len(num_tokens_per_image):
                    expanded_ids.extend([media_token_id] *
                                        num_tokens_per_image[img_idx])
                    img_idx += 1
                else:
                    expanded_ids.append(tid.item())
            input_ids = torch.tensor([expanded_ids], dtype=input_ids.dtype)

        return input_ids, pixel_values, grid_thws

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        input_ids, pixel_values, grid_thws = self._preprocess(inputs)
        multimodal_data = None
        if pixel_values is not None:
            multimodal_data = {
                "multimodal_data": {
                    "image": {
                        "pixel_values": pixel_values,
                        "grid_thws": grid_thws,
                    }
                },
            }
        return input_ids[0].to(torch.int32).tolist(), multimodal_data


# ---------------------------------------------------------------------------
# Top-level multimodal model
# ---------------------------------------------------------------------------

@register_auto_model("KimiK25ForConditionalGeneration")
@register_input_processor(
    KimiK25InputProcessor,
    model_type="kimi_k25",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<|media_pad|>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class KimiK25ForConditionalGeneration(PreTrainedModel):
    """Kimi-K2.5 multimodal model wrapping a DeepseekV3 language model,
    a 3D ViT vision tower, and a multi-modal projector."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        if _is_disagg():
            raise NotImplementedError(
                "KimiK25ForConditionalGeneration does not support "
                "disaggregated inference yet. Please unset the "
                f"{_MULTIMODAL_ENV_NAME} environment variable, or set it to '0'."
            )

        config = model_config.pretrained_config
        super().__init__(config)

        self._device = "cuda"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        # Media placeholder token ID
        self.media_token_id = getattr(config, "media_placeholder_token_id",
                                      163605)
        self._media_token_ids = torch.tensor([self.media_token_id],
                                             dtype=torch.int32,
                                             device=self._device)

        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp

        # Build language model with text_config
        llm_model_config = self._get_sub_model_config(model_config_cp,
                                                      "text_config")
        self.language_model = DeepseekV3ForCausalLM(llm_model_config)

        # Propagate extra_attrs (e.g. mla_layers registered by DeepseekV3)
        # from the sub-config back to the parent config, since
        # dataclasses.replace() creates a fresh extra_attrs dict.
        model_config_cp.extra_attrs.update(llm_model_config.extra_attrs)

        # Build vision tower and projector (no quantization)
        vision_config = config.vision_config
        if not _is_disagg():
            self.vision_tower = MoonViT3dPretrainedModel(vision_config)
            self.vision_tower = self.vision_tower.to(device=self._device,
                                                     dtype=self.model_dtype)
            self.mm_projector = KimiK25MultiModalProjector(vision_config)
            self.mm_projector = self.mm_projector.to(device=self._device,
                                                     dtype=self.model_dtype)
        else:
            self.vision_tower = None
            self.mm_projector = None

        self._post_config()
        self.is_loaded = True

    @staticmethod
    def _get_sub_model_config(
        model_config: ModelConfig,
        name: str,
    ) -> ModelConfig:
        """Extract sub-model config for text or vision components."""
        pretrained_config = getattr(model_config.pretrained_config, name)
        # Vision tower and projector are not quantized
        quant_config = model_config.quant_config if name == "text_config" else None

        # Remap exclude_modules for the language model sub-config.
        # Checkpoint uses "language_model.layers.X.self_attn*" but
        # DeepseekV3ForCausalLM internally uses "model.layers.X.self_attn*".
        if name == "text_config" and quant_config is not None:
            remapped_excludes = []
            for pattern in (quant_config.exclude_modules or []):
                if pattern.startswith("language_model."):
                    new_pattern = pattern[len("language_model."):]
                    if new_pattern.startswith("layers."):
                        new_pattern = "model." + new_pattern
                    remapped_excludes.append(new_pattern)
                else:
                    remapped_excludes.append(pattern)
            quant_config = copy.deepcopy(quant_config)
            quant_config.exclude_modules = remapped_excludes

        sub_model_config = dataclasses.replace(
            model_config,
            pretrained_config=pretrained_config,
            quant_config=quant_config,
        )
        if name == "text_config":
            sub_model_config._frozen = False
            sub_model_config.skip_create_weights_in_init = True
            # Ensure torch_dtype is propagated
            if (hasattr(sub_model_config.pretrained_config, "torch_dtype")
                    and sub_model_config.pretrained_config.torch_dtype is None):
                sub_model_config.pretrained_config.torch_dtype = (
                    model_config.pretrained_config.torch_dtype)
            sub_model_config._frozen = True

        return sub_model_config

    def _post_config(self):
        """Update config to point to the language model config for
        downstream consumers that check model.config.vocab_size etc."""
        self.config = self.language_model.config
        self.model_config.pretrained_config = self.language_model.config

    def load_weights(self, weights: Dict, weight_mapper=None, *args, **kwargs):
        """Load weights with appropriate prefix remapping.

        Checkpoint weight prefix structure:
        - language_model.model.* -> model.* (for DeepseekV3ForCausalLM)
        - language_model.lm_head.* -> lm_head.* (for DeepseekV3ForCausalLM)
        - vision_tower.* -> vision_tower (MoonViT3dPretrainedModel)
        - mm_projector.* -> mm_projector (KimiK25MultiModalProjector)

        The checkpoint may also have legacy prefix "language_model.layers."
        which should be remapped to "language_model.model.layers.".
        """
        # Handle legacy prefix remapping
        remapped_weights = {}
        for key, value in weights.items():
            new_key = key
            # Legacy NVFP4 checkpoint compatibility
            if new_key.startswith("language_model.layers."):
                new_key = new_key.replace("language_model.layers.",
                                          "language_model.model.layers.", 1)
            remapped_weights[new_key] = value

        # Load language model weights
        llm_weights = filter_weights("language_model", remapped_weights)
        logger.info("Loading language model weights...")
        self.language_model.load_weights(llm_weights)
        logger.info("Successfully loaded language model weights.")

        if not _is_disagg():
            # Load vision tower weights (direct load, no transformation needed)
            vt_weights = filter_weights("vision_tower", remapped_weights)
            logger.info("Loading vision tower weights...")
            self._load_vision_tower_weights(vt_weights)
            logger.info("Successfully loaded vision tower weights.")

            # Load MM projector weights with rename: proj.0 -> linear_1, proj.2 -> linear_2
            mm_weights = filter_weights("mm_projector", remapped_weights)
            logger.info("Loading MM projector weights...")
            self._load_mm_projector_weights(mm_weights)
            logger.info("Successfully loaded MM projector weights.")

    def _load_vision_tower_weights(self, weights: Dict):
        """Load vision tower weights directly using load_state_dict.
        All vision tower weights are bfloat16 with no transformation needed."""
        # Materialize lazy safetensor slices
        materialized = {k: v[:] if hasattr(v, '__getitem__') else v
                        for k, v in weights.items()}
        self.vision_tower.load_state_dict(materialized, strict=True)

    def _load_mm_projector_weights(self, weights: Dict):
        """Load MM projector weights with checkpoint name remapping.

        Checkpoint names:
          proj.0.weight -> linear_1.weight
          proj.0.bias   -> linear_1.bias
          proj.2.weight -> linear_2.weight
          proj.2.bias   -> linear_2.bias
          pre_norm.weight -> pre_norm.weight (no change)
          pre_norm.bias   -> pre_norm.bias (no change)
        """
        renamed_weights = {}
        for key, value in weights.items():
            new_key = key
            new_key = new_key.replace("proj.0.", "linear_1.")
            new_key = new_key.replace("proj.2.", "linear_2.")
            # Materialize lazy safetensor slices
            renamed_weights[new_key] = value[:] if hasattr(value, '__getitem__') else value
        self.mm_projector.load_state_dict(renamed_weights, strict=True)

    def infer_max_seq_len(self) -> int:
        return self.language_model.infer_max_seq_len()

    @property
    def mm_token_ids(self):
        return self._media_token_ids

    @nvtx_range("[Vision] process")
    def _get_vision_features(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run pixel values through vision tower and projector."""
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])

        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            return _vision_tower_forward(
                self.vision_tower,
                pixel_values,
                grid_thws,
                mm_projector=self.mm_projector,
            )

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests = attn_metadata.num_contexts
        num_generation_requests = attn_metadata.num_generations
        logger.debug(
            f"[KimiK25::forward] {num_context_requests=}, {num_generation_requests=}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])

        mm_embeds = []
        if len(multimodal_params) > 0 and not _is_disagg():
            pixel_values_list = []
            grid_thws_list = []
            for mp in multimodal_params:
                data = mp.multimodal_data.get("image", {})
                if "pixel_values" in data:
                    pv = data["pixel_values"]
                    if isinstance(pv, list):
                        pv = torch.cat(pv, dim=0)
                    pixel_values_list.append(pv)
                if "grid_thws" in data:
                    gt = data["grid_thws"]
                    grid_thws_list.append(gt)

            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0)
                grid_thws = torch.cat(grid_thws_list, dim=0)
                vision_features = self._get_vision_features(
                    pixel_values, grid_thws)
                mm_embeds = [
                    feat.contiguous() for feat in vision_features
                ]

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.language_model.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=self._media_token_ids,
            **kwargs,
        )

        logits = self.language_model.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
        )
        return logits
