# TensorRT-LLM Porting Plan: KimiK25ForConditionalGeneration

Generated: 2026-03-18
Model: Kimi-K2.5-NVFP4
Source: vLLM reference implementation (`kimi_k25.py`, `kimi_k25_vit.py`)
Checkpoint: `/home/scratch.trt_llm_data_ci/llm-models/Kimi-K2.5-NVFP4/`

---

## Part 1: Module Mapping

### 1.1 Model Overview

| Property | Reference (vLLM) | TensorRT-LLM |
|----------|-------------------|--------------|
| Top-Level Model Class | `KimiK25ForConditionalGeneration` | `KimiK25ForConditionalGeneration` (new multimodal wrapper) |
| Language Model Class | `DeepseekV3ForCausalLM` (via `init_vllm_registered_model`) | `DeepseekV3ForCausalLM` (existing in `modeling_deepseekv3.py`) |
| Language Model Base Class | `DeepseekV2ForCausalLM` | `SpecDecOneEngineForCausalLM[DeepseekV3Model]` |
| Vision Tower Class | `MoonViT3dPretrainedModel` | New implementation required (3D ViT encoder) |
| MM Projector Class | `KimiK25MultiModalProjector` | New implementation required (2-layer MLP with pre-norm + patch merging) |
| Config Class | `KimiK25Config` (contains `text_config: DeepseekV3Config` + `vision_config: KimiK25VisionConfig`) | Custom config wrapping existing DeepseekV3 config |

### 1.2 Architecture Summary

Kimi-K2.5 is a multimodal model composed of three main components:

1. **Vision Tower** (`MoonViT3dPretrainedModel`): A 3D Vision Transformer with spatial-temporal attention, 2D RoPE, learnable 2D position embeddings with fixed temporal sincos embeddings, and a temporal-pooling patch merger.
2. **Multi-Modal Projector** (`KimiK25MultiModalProjector`): A 2-layer MLP with LayerNorm pre-normalization that projects merged vision features from vision hidden size to language model hidden size.
3. **Language Model** (`DeepseekV3ForCausalLM`): A DeepseekV3-architecture language model with MLA (Multi-head Latent Attention) and MoE (Mixture of Experts).

The language model is identical to DeepseekV3 in architecture. TensorRT-LLM already has a full implementation in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`. The porting effort primarily involves building the multimodal wrapper, vision tower, and projector.

### 1.3 Module Mapping Table

#### 1.3.1 Top-Level Multimodal Model

| Reference Module Path | Reference Class | TensorRT-LLM Class | Notes |
|----------------------|----------------|-------------------|-------|
| (root) | `KimiK25ForConditionalGeneration` | `KimiK25ForConditionalGeneration` (new) | Top-level multimodal wrapper |
| `vision_tower` | `MoonViT3dPretrainedModel` | `MoonViT3dPretrainedModel` (new) | 3D ViT with temporal pooling |
| `mm_projector` | `KimiK25MultiModalProjector` | `KimiK25MultiModalProjector` (new) | Patch merging + 2-layer MLP |
| `language_model` | `DeepseekV3ForCausalLM` | `DeepseekV3ForCausalLM` (existing) | Reuse existing TRT-LLM implementation |

#### 1.3.2 Vision Tower: MoonViT3dPretrainedModel

| Reference Module Path | Reference Class | TensorRT-LLM Class | Notes |
|----------------------|----------------|-------------------|-------|
| `vision_tower.patch_embed` | `MoonVision3dPatchEmbed` | Custom module (new) | Conv2d + learnable 2D position embedding with temporal sincos |
| `vision_tower.patch_embed.proj` | `nn.Conv2d` | `nn.Conv2d` (PyTorch native) | kernel_size=14, stride=14, in_channels=3, out_channels=1152 |
| `vision_tower.patch_embed.pos_emb` | `Learnable2DInterpPosEmbDivided_fixed` | Custom module (new) | Learnable spatial + fixed temporal sincos position embedding |
| `vision_tower.encoder` | `MoonViT3dEncoder` | Custom module (new) | Encoder stack with 2D RoPE |
| `vision_tower.encoder.rope_2d` | `Rope2DPosEmbRepeated` | Custom module (new) | 2D rotary position embedding for vision, NOT the language model RoPE |
| `vision_tower.encoder.blocks[k]` | `MoonViTEncoderLayer` | Custom encoder layer (new) | Pre-norm transformer block with 2D RoPE |
| `vision_tower.encoder.blocks[k].norm0` | `nn.LayerNorm` | `LayerNorm` | Pre-attention normalization, hidden_size=1152 |
| `vision_tower.encoder.blocks[k].wqkv` | `QKVParallelLinear` | `Linear` (or QKVParallelLinear equivalent) | Fused QKV, in=1152, out=3456 (3*1152), bias=True |
| `vision_tower.encoder.blocks[k].wo` | `RowParallelLinear` | `Linear` | Output projection, in=1152, out=1152, bias=True |
| `vision_tower.encoder.blocks[k].attn` | `MMEncoderAttention` | Vision encoder attention (FlashAttention) | 16 heads, head_dim=72 |
| `vision_tower.encoder.blocks[k].norm1` | `nn.LayerNorm` | `LayerNorm` | Pre-MLP normalization, hidden_size=1152 |
| `vision_tower.encoder.blocks[k].mlp.fc0` | `ColumnParallelLinear` | `Linear` | Up projection, in=1152, out=4304, bias=True |
| `vision_tower.encoder.blocks[k].mlp.fc1` | `RowParallelLinear` | `Linear` | Down projection, in=4304, out=1152, bias=True |
| `vision_tower.encoder.final_layernorm` | `nn.LayerNorm` | `LayerNorm` | Final normalization, hidden_size=1152 |

#### 1.3.3 Multi-Modal Projector: KimiK25MultiModalProjector

| Reference Module Path | Reference Class | TensorRT-LLM Class | Notes |
|----------------------|----------------|-------------------|-------|
| `mm_projector.pre_norm` | `nn.LayerNorm` | `LayerNorm` | Pre-normalization, hidden_size=1152, eps=1e-5 |
| `mm_projector.linear_1` (ckpt: `mm_projector.proj.0`) | `ReplicatedLinear` | `Linear` | in=4608 (1152 x 2 x 2), out=4608, bias=True |
| `mm_projector.act` | `GELUActivation` | GELU activation | Applied between linear_1 and linear_2 |
| `mm_projector.linear_2` (ckpt: `mm_projector.proj.2`) | `ReplicatedLinear` | `Linear` | in=4608, out=7168 (text_hidden_size), bias=True |

Note on patch merging: The projector's `pre_norm` operates on the vision encoder's output (dim=1152). Then the hidden states are reshaped from `[nh*nw, kh*kw, 1152]` (where kh=kw=2, the merge kernel) to `[nh*nw, 4608]` before being fed to `linear_1`. This flattening of the spatial merge kernel is a critical step in the forward pass.

#### 1.3.4 Language Model: DeepseekV3ForCausalLM (Existing TRT-LLM Implementation)

| Reference Module Path | Reference Class | TensorRT-LLM Class | Notes |
|----------------------|----------------|-------------------|-------|
| `language_model.model.embed_tokens` | `nn.Embedding` | `Embedding` | vocab_size=163840, hidden_size=7168, dtype=bfloat16 |
| `language_model.model.layers[i]` | Decoder layer | `DeepseekV3DecoderLayer` | 61 layers total |
| `language_model.model.layers[i].input_layernorm` | `RMSNorm` | `RMSNorm` | hidden_size=7168, eps=1e-5 |
| `language_model.model.layers[i].self_attn` | MLA attention | `DeepseekV3Attention` (extends `MLA`) | Multi-head Latent Attention |
| `language_model.model.layers[i].self_attn.q_a_proj` | `nn.Linear` | Fused into `kv_a_proj_with_mqa` | in=7168, out=1536 |
| `language_model.model.layers[i].self_attn.q_a_layernorm` | `RMSNorm` | `RMSNorm` | hidden_size=1536 |
| `language_model.model.layers[i].self_attn.q_b_proj` | `nn.Linear` | `Linear` | in=1536, out=12288 (64 heads x 192 dim) |
| `language_model.model.layers[i].self_attn.kv_a_proj_with_mqa` | `nn.Linear` | `DeepseekV3Linear` | in=7168, out=576 (512+64) |
| `language_model.model.layers[i].self_attn.kv_a_layernorm` | `RMSNorm` | `RMSNorm` | hidden_size=512 |
| `language_model.model.layers[i].self_attn.kv_b_proj` | `nn.Linear` | `Linear` | in=512, out=16384 (64 heads x (128+128)) |
| `language_model.model.layers[i].self_attn.o_proj` | `nn.Linear` | `Linear` | in=8192 (64 x 128), out=7168 |
| `language_model.model.layers[i].post_attention_layernorm` | `RMSNorm` | `RMSNorm` | hidden_size=7168, eps=1e-5 |
| `language_model.model.layers[0].mlp` (dense, layer 0 only) | `GatedMLP` | `GatedMLP` | Dense FFN for first_k_dense_replace=1 layers |
| `language_model.model.layers[0].mlp.gate_proj` | `nn.Linear` | Fused into `gate_up_proj` | in=7168, out=18432 |
| `language_model.model.layers[0].mlp.up_proj` | `nn.Linear` | Fused into `gate_up_proj` | in=7168, out=18432 |
| `language_model.model.layers[0].mlp.down_proj` | `nn.Linear` | `Linear` | in=18432, out=7168 |
| `language_model.model.layers[i>=1].mlp` (MoE, layers 1-60) | `Deepseekv3MoE` | `Deepseekv3MoE` | MoE with gate, routed experts, shared expert |
| `language_model.model.layers[i].mlp.gate` | Router gate | `DeepseekV3Gate` | 384 experts, hidden_size=7168 |
| `language_model.model.layers[i].mlp.experts[j]` | Expert MLP | `MoE` (via `create_moe`) | 384 routed experts, intermediate_size=2048 |
| `language_model.model.layers[i].mlp.shared_experts` | Shared expert MLP | `GatedMLP` | 1 shared expert, intermediate_size=2048 |
| `language_model.model.norm` | `RMSNorm` | `RMSNorm` | Final normalization, hidden_size=7168 |
| `language_model.lm_head` | `nn.Linear` | `Linear` (LMHead) | in=7168, out=163840, no bias |

### 1.4 TensorRT-LLM Module Initialization Parameters

#### MLA (Multi-head Latent Attention)

The language model uses TRT-LLM's existing `MLA` module (in `tensorrt_llm/_torch/modules/attention.py`). Key initialization parameters derived from `text_config`:

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_size` | 7168 | `text_config.hidden_size` |
| `num_attention_heads` | 64 | `text_config.num_attention_heads` |
| `num_key_value_heads` | 64 | `text_config.num_key_value_heads` |
| `qk_nope_head_dim` | 128 | `text_config.qk_nope_head_dim` |
| `qk_rope_head_dim` | 64 | `text_config.qk_rope_head_dim` |
| `v_head_dim` | 128 | `text_config.v_head_dim` |
| `q_lora_rank` | 1536 | `text_config.q_lora_rank` |
| `kv_lora_rank` | 512 | `text_config.kv_lora_rank` |
| `max_position_embeddings` | 262144 | `text_config.max_position_embeddings` |
| `bias` | False | `text_config.attention_bias` |
| `pos_embd_params.type` | `PositionEmbeddingType.yarn` | `text_config.rope_scaling.type == "yarn"` |
| `pos_embd_params.rope.theta` | 50000.0 | `text_config.rope_theta` |
| `pos_embd_params.rope.scaling_factor` | 64.0 | `text_config.rope_scaling.factor` |
| `pos_embd_params.rope.original_max_position_embeddings` | 4096 | `text_config.rope_scaling.original_max_position_embeddings` |
| `pos_embd_params.rope.beta_fast` | 32.0 | `text_config.rope_scaling.beta_fast` |
| `pos_embd_params.rope.beta_slow` | 1.0 | `text_config.rope_scaling.beta_slow` |
| `pos_embd_params.rope.mscale` | 1.0 | `text_config.rope_scaling.mscale` |
| `pos_embd_params.rope.mscale_all_dim` | 1.0 | `text_config.rope_scaling.mscale_all_dim` |
| `pos_embd_params.is_neox` | False | DeepseekV3 uses non-neox interleaved RoPE |

The existing `DeepseekV3Attention` class (extending `MLA`) creates a fused `kv_a_proj_with_mqa` that combines `q_a_proj` and `kv_a_proj_with_mqa`:
- Output dimension = `kv_lora_rank(512) + qk_rope_head_dim(64) + q_lora_rank(1536) = 2112`

#### GatedMLP (Dense layers: layer 0)

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_size` | 7168 | `text_config.hidden_size` |
| `intermediate_size` | 18432 | `text_config.intermediate_size` |
| `bias` | False | DeepseekV3 convention |
| `dtype` | bfloat16 | `text_config.dtype` |

#### Deepseekv3MoE (MoE layers: layers 1-60)

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_experts` | 384 | `text_config.n_routed_experts` |
| `top_k` | 8 | `text_config.num_experts_per_tok` |
| `hidden_size` | 7168 | `text_config.hidden_size` |
| `intermediate_size` | 2048 | `text_config.moe_intermediate_size` |
| `shared_expert_intermediate_size` | 2048 | `moe_intermediate_size * n_shared_experts` |
| `n_group` | 1 | `text_config.n_group` |
| `topk_group` | 1 | `text_config.topk_group` |
| `routed_scaling_factor` | 2.827 | `text_config.routed_scaling_factor` |
| `scoring_func` | sigmoid | `text_config.scoring_func` |
| `topk_method` | noaux_tc | `text_config.topk_method` |
| `norm_topk_prob` | True | `text_config.norm_topk_prob` |

#### RMSNorm

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_size` | 7168 | `text_config.hidden_size` |
| `eps` | 1e-5 | `text_config.rms_norm_eps` |
| `dtype` | bfloat16 | `text_config.dtype` |

#### Embedding

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_embeddings` | 163840 | `text_config.vocab_size` |
| `embedding_dim` | 7168 | `text_config.hidden_size` |
| `dtype` | bfloat16 | `text_config.dtype` |

#### Vision Tower Modules

**MoonVision3dPatchEmbed**:

| Parameter | Value | Source |
|-----------|-------|--------|
| `out_dim` | 1152 | `vision_config.vt_hidden_size` |
| `in_dim` | 3 | RGB channels |
| `patch_size` | 14 | `vision_config.patch_size` |
| `pos_emb_height` | 64 | `vision_config.init_pos_emb_height` |
| `pos_emb_width` | 64 | `vision_config.init_pos_emb_width` |
| `pos_emb_time` | 4 | `vision_config.init_pos_emb_time` |
| `pos_emb_type` | "divided_fixed" | `vision_config.pos_emb_type` |

**MoonViTEncoderLayer** (27 layers):

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_heads` | 16 | `vision_config.vt_num_attention_heads` |
| `hidden_dim` | 1152 | `vision_config.vt_hidden_size` |
| `mlp_dim` | 4304 | `vision_config.vt_intermediate_size` |
| `activation` | gelu_pytorch_tanh | Hardcoded in vLLM reference |
| `attn_bias` | True | Hardcoded in reference |

**Rope2DPosEmbRepeated**:

| Parameter | Value | Source |
|-----------|-------|--------|
| `dim` | 72 | `hidden_dim // num_heads = 1152 // 16` |
| `max_height` | 512 | Hardcoded in encoder |
| `max_width` | 512 | Hardcoded in encoder |
| `theta_base` | 10000 | Default |

**KimiK25MultiModalProjector**:

| Parameter | Value | Source |
|-----------|-------|--------|
| `pre_norm` hidden_size | 1152 | `vision_config.vt_hidden_size` |
| `pre_norm` eps | 1e-5 | `vision_config.projector_ln_eps` |
| `linear_1` in_features | 4608 | `vt_hidden_size * merge_h * merge_w = 1152 * 2 * 2` |
| `linear_1` out_features | 4608 | Same as in_features |
| `linear_2` in_features | 4608 | Same |
| `linear_2` out_features | 7168 | Checkpoint-verified (proj.2.weight shape [7168, 4608]) |
| `activation` | GELU | `GELUActivation` |

**IMPORTANT**: There is a discrepancy in the config. The `vision_config.mm_hidden_size` is listed as 1152 in `config.json`, but the actual checkpoint weight `mm_projector.proj.2.weight` has shape `[7168, 4608]`, meaning the projector output is 7168 (= `text_hidden_size`). The checkpoint is authoritative. When implementing, set the projector's output dimension to `text_hidden_size` (7168), not `mm_hidden_size` (1152). In the vLLM code, `KimiK25MultiModalProjector.__init__` uses `config.mm_hidden_size` for the output of `linear_2`, but at the config level `mm_hidden_size` is resolved to the correct value. The checkpoint shape confirms the correct output dimension is 7168.

---

## Part 2: Weight Loading

### 2.1 Weight Loading Overview

| Checkpoint Format | File Pattern | Loading Method |
|------------------|--------------|----------------|
| SafeTensors (sharded) | `model-{00001..00119}-of-00119.safetensors` | `safetensors.torch.load_file()` with index from `model.safetensors.index.json` |

Total weight count: 278,341 tensors across 119 shards.

### 2.2 Weight Name Table

#### 2.2.1 Checkpoint Weight Name Patterns (with shapes and dtypes)

**Vision Tower weights** (all bfloat16, no quantization):

| Checkpoint Weight Pattern | Shape | Dtype |
|--------------------------|-------|-------|
| `vision_tower.patch_embed.proj.weight` | [1152, 3, 14, 14] | bfloat16 |
| `vision_tower.patch_embed.proj.bias` | [1152] | bfloat16 |
| `vision_tower.patch_embed.pos_emb.weight` | [64, 64, 1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.wqkv.weight` | [3456, 1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.wqkv.bias` | [3456] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.wo.weight` | [1152, 1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.wo.bias` | [1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.norm0.weight` | [1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.norm0.bias` | [1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.norm1.weight` | [1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.norm1.bias` | [1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.mlp.fc0.weight` | [4304, 1152] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.mlp.fc0.bias` | [4304] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.mlp.fc1.weight` | [1152, 4304] | bfloat16 |
| `vision_tower.encoder.blocks.{k}.mlp.fc1.bias` | [1152] | bfloat16 |
| `vision_tower.encoder.final_layernorm.weight` | [1152] | bfloat16 |
| `vision_tower.encoder.final_layernorm.bias` | [1152] | bfloat16 |

**MM Projector weights** (all bfloat16, no quantization):

| Checkpoint Weight Pattern | Shape | Dtype |
|--------------------------|-------|-------|
| `mm_projector.pre_norm.weight` | [1152] | bfloat16 |
| `mm_projector.pre_norm.bias` | [1152] | bfloat16 |
| `mm_projector.proj.0.weight` | [4608, 4608] | bfloat16 |
| `mm_projector.proj.0.bias` | [4608] | bfloat16 |
| `mm_projector.proj.2.weight` | [7168, 4608] | bfloat16 |
| `mm_projector.proj.2.bias` | [7168] | bfloat16 |

**Language Model - Attention weights** (bfloat16, excluded from NVFP4 quantization):

| Checkpoint Weight Pattern | Shape | Dtype | Notes |
|--------------------------|-------|-------|-------|
| `language_model.model.layers.{i}.self_attn.q_a_proj.weight` | [1536, 7168] | bfloat16 | Q low-rank down projection |
| `language_model.model.layers.{i}.self_attn.q_a_layernorm.weight` | [1536] | bfloat16 | Q compression layernorm |
| `language_model.model.layers.{i}.self_attn.q_b_proj.weight` | [12288, 1536] | bfloat16 | Q low-rank up projection (64 heads x 192 dim) |
| `language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight` | [576, 7168] | bfloat16 | KV compression (512 kv_lora_rank + 64 rope_dim) |
| `language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight` | [512] | bfloat16 | KV compression layernorm |
| `language_model.model.layers.{i}.self_attn.kv_b_proj.weight` | [16384, 512] | bfloat16 | KV up projection (64 heads x (128 nope + 128 v)) |
| `language_model.model.layers.{i}.self_attn.o_proj.weight` | [7168, 8192] | bfloat16 | Output projection (64 heads x 128 v_head_dim) |
| `language_model.model.layers.{i}.self_attn.k_proj.k_scale` | [] (scalar) | float32 | FP8 KV cache K scale |
| `language_model.model.layers.{i}.self_attn.v_proj.v_scale` | [] (scalar) | float32 | FP8 KV cache V scale |

**Language Model - Dense MLP weights (layer 0 only, NVFP4 quantized)**:

| Checkpoint Weight Pattern | Shape | Dtype | Notes |
|--------------------------|-------|-------|-------|
| `language_model.model.layers.0.mlp.gate_proj.weight` | [18432, 3584] | uint8 (FP4 packed) | Gate projection, packed 2 FP4 values per byte |
| `language_model.model.layers.0.mlp.gate_proj.weight_scale` | [18432, 448] | float8_e4m3fn | Per-block FP8 scales (group_size=16, so 7168/16=448) |
| `language_model.model.layers.0.mlp.gate_proj.weight_scale_2` | [] (scalar) | float32 | Global scale factor |
| `language_model.model.layers.0.mlp.gate_proj.input_scale` | [] (scalar) | float32 | Activation quantization scale |
| `language_model.model.layers.0.mlp.up_proj.*` | Same structure as gate_proj | | |
| `language_model.model.layers.0.mlp.down_proj.weight` | [7168, 9216] | uint8 | Down projection |
| `language_model.model.layers.0.mlp.down_proj.weight_scale` | [7168, 1152] | float8_e4m3fn | |
| `language_model.model.layers.0.mlp.down_proj.weight_scale_2` | [] (scalar) | float32 | |
| `language_model.model.layers.0.mlp.down_proj.input_scale` | [] (scalar) | float32 | |

**Language Model - MoE weights (layers 1-60, NVFP4 quantized)**:

| Checkpoint Weight Pattern | Shape | Dtype | Notes |
|--------------------------|-------|-------|-------|
| `language_model.model.layers.{i}.mlp.gate.weight` | [384, 7168] | bfloat16 | Router gate weight (unquantized) |
| `language_model.model.layers.{i}.mlp.gate.e_score_correction_bias` | [384] | bfloat16 | Correction bias for noaux_tc routing |
| `language_model.model.layers.{i}.mlp.experts.{j}.gate_proj.weight` | [2048, 3584] | uint8 | Expert gate projection (NVFP4) |
| `language_model.model.layers.{i}.mlp.experts.{j}.gate_proj.weight_scale` | [2048, 448] | float8_e4m3fn | |
| `language_model.model.layers.{i}.mlp.experts.{j}.gate_proj.weight_scale_2` | [] | float32 | |
| `language_model.model.layers.{i}.mlp.experts.{j}.gate_proj.input_scale` | [] | float32 | |
| `language_model.model.layers.{i}.mlp.experts.{j}.up_proj.*` | Same as gate_proj | | |
| `language_model.model.layers.{i}.mlp.experts.{j}.down_proj.weight` | [7168, 1024] | uint8 | Expert down projection (NVFP4) |
| `language_model.model.layers.{i}.mlp.experts.{j}.down_proj.weight_scale` | [7168, 128] | float8_e4m3fn | |
| `language_model.model.layers.{i}.mlp.experts.{j}.down_proj.weight_scale_2` | [] | float32 | |
| `language_model.model.layers.{i}.mlp.experts.{j}.down_proj.input_scale` | [] | float32 | |
| `language_model.model.layers.{i}.mlp.shared_experts.gate_proj.*` | Same as expert gate_proj | | |
| `language_model.model.layers.{i}.mlp.shared_experts.up_proj.*` | Same as expert up_proj | | |
| `language_model.model.layers.{i}.mlp.shared_experts.down_proj.*` | Same as expert down_proj | | |

**Language Model - Normalization and Embedding weights (bfloat16, unquantized)**:

| Checkpoint Weight Pattern | Shape | Dtype |
|--------------------------|-------|-------|
| `language_model.model.embed_tokens.weight` | [163840, 7168] | bfloat16 |
| `language_model.model.layers.{i}.input_layernorm.weight` | [7168] | bfloat16 |
| `language_model.model.layers.{i}.post_attention_layernorm.weight` | [7168] | bfloat16 |
| `language_model.model.norm.weight` | [7168] | bfloat16 |
| `language_model.lm_head.weight` | [163840, 7168] | bfloat16 |

### 2.3 Weight Loading Details

This section documents potential modeling errors and issues that may occur during TensorRT-LLM implementation.

#### Checkpoint Prefix Remapping

The checkpoint uses a specific prefix structure that needs remapping. The vLLM code shows these rename rules:

```
"language_model.layers." -> "language_model.model.layers."   (legacy NVFP4 checkpoint compatibility)
"mm_projector.proj.0"    -> "mm_projector.linear_1"
"mm_projector.proj.2"    -> "mm_projector.linear_2"
```

In TensorRT-LLM, the language model weights should be loaded with the prefix `language_model.model.` stripped, mapping directly to the existing `DeepseekV3ForCausalLM` weight loader which expects prefixes like `model.layers.{i}.*`.

**Critical Prefix Mapping for TRT-LLM**:
- Checkpoint `language_model.model.layers.{i}.*` maps to TRT-LLM `model.layers.{i}.*`
- Checkpoint `language_model.model.embed_tokens.*` maps to TRT-LLM `model.embed_tokens.*`
- Checkpoint `language_model.model.norm.*` maps to TRT-LLM `model.norm.*`
- Checkpoint `language_model.lm_head.*` maps to TRT-LLM `lm_head.*`

#### MLA Weight Fusion: q_a_proj + kv_a_proj_with_mqa

The existing TRT-LLM `DeepseekV3Attention` fuses `q_a_proj` and `kv_a_proj_with_mqa` into a single linear layer called `kv_a_proj_with_mqa`.

**Error-prone detail**: The fused weight must be concatenated in this specific order: `[q_a_proj, kv_a_proj_with_mqa]` along dim=0.
- `q_a_proj.weight`: shape [1536, 7168]
- `kv_a_proj_with_mqa.weight`: shape [576, 7168]
- Fused `kv_a_proj_with_mqa.weight`: shape [2112, 7168]

This fusion is already handled by the existing `DeepseekV3WeightLoader.load_weights()` method, which checks for the `kv_a_proj_with_mqa` module name and performs the fusion. No new logic is needed for the language model weights.

#### MLA kv_b_proj Weight Splitting

The `kv_b_proj` weight has shape `[16384, 512]`, which encodes both K nope and V projections for all heads:
- Reshaped to `[64, 256, 512]` (64 heads, 128 nope + 128 v, 512 kv_lora_rank)
- Split into `k_nope_weight [64, 128, 512]` and `v_weight [64, 128, 512]`
- `k_nope_weight` is transposed to get `k_b_proj_trans [64, 512, 128]` for absorbed attention
- Both are stored separately in the TRT-LLM MLA module

This splitting and transposition logic is already implemented in `DeepseekV3WeightLoader.load_kv_b_proj_and_k_b_proj_trans()`.

#### MoE Expert Weight Renaming

The existing TRT-LLM weight loader renames MoE expert weights:
- `down_proj` -> `w2`
- `up_proj` -> `w3`
- `gate_proj` -> `w1`

This is handled by the `rename_moe_weight` function in `DeepseekV3WeightLoader`.

#### Dense MLP Gate/Up Fusion

For the dense layer (layer 0), TRT-LLM's `GatedMLP` expects a fused `gate_up_proj` weight. The checkpoint has separate `gate_proj` and `up_proj` weights:
- `gate_proj.weight`: [18432, 3584] (NVFP4 packed, logical [18432, 7168])
- `up_proj.weight`: [18432, 3584] (NVFP4 packed, logical [18432, 7168])
- Fused `gate_up_proj.weight`: concatenate along dim=0 -> [36864, 3584]

The existing weight loader handles this through the `params_map = {'gate_up_proj': ['gate_proj', 'up_proj']}` mapping.

#### FP8 KV Cache Scale Weights

The checkpoint contains per-layer FP8 KV cache scales:
- `self_attn.k_proj.k_scale`: scalar float32 -- K cache quantization scale
- `self_attn.v_proj.v_scale`: scalar float32 -- V cache quantization scale

These are loaded by the existing TRT-LLM MLA module for FP8 KV cache support.

#### MM Projector Weight Remapping

The checkpoint stores the projector weights with sequential layer naming (`proj.0`, `proj.2`), while the vLLM model uses renamed attributes (`linear_1`, `linear_2`). The weight remapping is:
- `mm_projector.proj.0.weight` -> `mm_projector.linear_1.weight`
- `mm_projector.proj.0.bias` -> `mm_projector.linear_1.bias`
- `mm_projector.proj.2.weight` -> `mm_projector.linear_2.weight`
- `mm_projector.proj.2.bias` -> `mm_projector.linear_2.bias`

For TRT-LLM, either naming convention can be used as long as the weight loader handles the mapping consistently.

#### Vision Tower Weights

All vision tower weights are in bfloat16 and can be loaded directly without any transformation. The weight names in the checkpoint match the module structure directly:
- `vision_tower.patch_embed.*` maps directly to the patch embedding module
- `vision_tower.encoder.blocks.{k}.*` maps directly to each encoder layer
- `vision_tower.encoder.final_layernorm.*` maps to the final layer norm

No fusion or renaming is needed for vision tower weights.

#### NVFP4 Weight Packing Format

For NVFP4-quantized weights, the data is stored in a packed format:
- `weight`: uint8 tensor, packing 2 FP4 (e2m1) values per byte. Shape is `[out_features, in_features/2]`
- `weight_scale`: float8_e4m3fn tensor, per-block scaling factors. With `group_size=16`, shape is `[out_features, in_features/2/8]` (since 16 logical values / 2 packed = 8 bytes per group, and each group gets one FP8 scale)
- `weight_scale_2`: scalar float32, global secondary scale factor
- `input_scale`: scalar float32, activation quantization scale

The dequantization formula is:
```
dequantized_value = fp4_to_float(packed_weight) * weight_scale * weight_scale_2
```

#### NVFP4 Fused A-proj Weight Loading

For the fused `kv_a_proj_with_mqa` in NVFP4 mode, the weight loader performs special handling:
1. Checks if both `q_a_proj` and `kv_a_proj_with_mqa` are NVFP4 quantized
2. If so, verifies their `input_scale` values match (they must be identical for fusion)
3. Reconciles `weight_scale_2` values -- if they differ, the weight with the smaller scale is re-quantized using the larger scale
4. Concatenates the FP4 weights and their corresponding FP8 block scales
5. Computes `alpha = input_scale * weight_scale_2` for the fused module

In this specific checkpoint, since all `self_attn` layers are excluded from NVFP4 quantization, the attention weights remain in bfloat16 and this NVFP4 fusion path is not exercised for attention. The NVFP4 fusion path only applies to MLP weights (gate/up fusion).

---

## Part 3: Quantization

### 3.1 Quantization Strategy

The checkpoint uses **NVFP4** quantization (4-bit floating point, e2m1 format) produced by NVIDIA ModelOpt v0.41.0, with **FP8 KV cache** quantization.

**Quantization Configuration (from `config.json` and `hf_quant_config.json`)**:

```yaml
quantization_type: NVFP4
quant_method: modelopt
producer: modelopt v0.41.0
source_config: config.json + hf_quant_config.json
config_fields:
  quant_algo: NVFP4
  weight_bits: 4
  weight_type: float (e2m1)
  activation_bits: 4
  activation_type: float (e2m1)
  group_size: 16
  kv_cache_quant_algo: FP8
  kv_cache_bits: 8
  kv_cache_type: float (e4m3fn)

trtllm_config:
  quant_algo: QuantAlgo.NVFP4
  kv_cache_quant_algo: QuantAlgo.FP8
  group_size: 16
```

**NVFP4 Weight Structure**:
Each quantized linear layer has 4 tensors:
1. `weight` (uint8): Packed FP4 weights (2 values per byte)
2. `weight_scale` (float8_e4m3fn): Per-block FP8 scaling factors (1 scale per 16 values)
3. `weight_scale_2` (float32 scalar): Global secondary scale
4. `input_scale` (float32 scalar): Activation quantization scale

The dequantization formula is:
```
dequantized_value = fp4_to_float(packed_weight) * weight_scale * weight_scale_2
```

And during inference, the activation is also quantized:
```
quantized_input = input * (1 / input_scale)  # in FP4
output = fp4_gemm(quantized_input, weight) * (input_scale * weight_scale_2)
```

### 3.2 Excluded Modules

The following modules are excluded from NVFP4 quantization and remain in **bfloat16**.

**Using TensorRT-LLM naming conventions**:

```yaml
exclude_modules:
  # Language model attention (ALL 61 layers excluded from NVFP4)
  - model.layers.*.self_attn.*     # All self_attn submodules for all layers 0-60
  # Language model head
  - lm_head
  # Normalization layers (inherently unquantized)
  - model.layers.*.input_layernorm
  - model.layers.*.post_attention_layernorm
  - model.norm
  # Embedding (inherently unquantized)
  - model.embed_tokens
  # Vision tower (entirely unquantized)
  - vision_tower.*
  # Multi-modal projector (entirely unquantized)
  - mm_projector.*
  # MoE gate/router (unquantized)
  - model.layers.*.mlp.gate.*
```

**Detailed breakdown from `hf_quant_config.json`**:

The `exclude_modules` list explicitly excludes:
1. `language_model.lm_head` -- LM head remains BF16
2. `language_model.layers.{0-60}.self_attn*` -- All 61 attention layers remain BF16
3. `mm_projector*` -- Entire MM projector remains BF16
4. `vision_tower*` -- Entire vision tower remains BF16

**Modules that ARE quantized with NVFP4**:
- Dense MLP (layer 0): `gate_proj`, `up_proj`, `down_proj`
- MoE routed experts (layers 1-60): Each of the 384 experts' `gate_proj`, `up_proj`, `down_proj`
- MoE shared experts (layers 1-60): `gate_proj`, `up_proj`, `down_proj`

**NOTE on KV cache quantization**: The `kv_cache_scheme` specifies FP8 (8-bit float) quantization for the KV cache. The per-layer K and V scales are stored as:
- `self_attn.k_proj.k_scale`: Per-layer K cache FP8 scale
- `self_attn.v_proj.v_scale`: Per-layer V cache FP8 scale

### 3.3 Quantization Weight Loading

#### NVFP4 Weight Loading Procedure

For each NVFP4-quantized module, the following weights must be loaded:

```yaml
quantization_weights:
  # Dense MLP (layer 0) - gate_up_proj fusion
  - name: model.layers.0.mlp.gate_up_proj.weight
    source: Fuse language_model.model.layers.0.mlp.gate_proj.weight + up_proj.weight
    dtype: uint8 (FP4 packed)
    description: Concatenate gate_proj and up_proj FP4 weights along dim=0

  - name: model.layers.0.mlp.gate_up_proj.weight_scale
    source: Fuse gate_proj.weight_scale + up_proj.weight_scale
    dtype: float8_e4m3fn
    description: Concatenate per-block scales along dim=0

  - name: model.layers.0.mlp.gate_up_proj.weight_scale_2
    source: Reconcile gate_proj.weight_scale_2 and up_proj.weight_scale_2
    dtype: float32 (scalar)
    description: Must use the larger of the two; re-quantize the other if different

  - name: model.layers.0.mlp.gate_up_proj.input_scale
    source: gate_proj.input_scale (must equal up_proj.input_scale)
    dtype: float32 (scalar)
    description: Activation scale for fused gate_up projection

  - name: model.layers.0.mlp.down_proj.*
    source: Direct load from language_model.model.layers.0.mlp.down_proj.*
    description: No fusion needed, direct weight/scale loading

  # MoE experts (layers 1-60) - renamed
  - name: model.layers.{i}.mlp.experts.{j}.w1.weight  (gate_proj)
    source: language_model.model.layers.{i}.mlp.experts.{j}.gate_proj.weight
    dtype: uint8
    description: Renamed from gate_proj to w1

  - name: model.layers.{i}.mlp.experts.{j}.w3.weight  (up_proj)
    source: language_model.model.layers.{i}.mlp.experts.{j}.up_proj.weight
    dtype: uint8
    description: Renamed from up_proj to w3

  - name: model.layers.{i}.mlp.experts.{j}.w2.weight  (down_proj)
    source: language_model.model.layers.{i}.mlp.experts.{j}.down_proj.weight
    dtype: uint8
    description: Renamed from down_proj to w2

  # Each expert also has corresponding weight_scale, weight_scale_2, input_scale
  # for each of w1, w2, w3

  # Shared experts (layers 1-60) - gate_up_proj fusion
  - name: model.layers.{i}.mlp.shared_experts.gate_up_proj.*
    source: Fuse shared_experts.gate_proj + shared_experts.up_proj
    description: Same fusion logic as dense MLP gate_up_proj

  # FP8 KV cache scales
  - name: model.layers.{i}.self_attn.k_proj.k_scale
    source: language_model.model.layers.{i}.self_attn.k_proj.k_scale
    dtype: float32 (scalar)
    description: Per-layer K cache quantization scale

  - name: model.layers.{i}.self_attn.v_proj.v_scale
    source: language_model.model.layers.{i}.self_attn.v_proj.v_scale
    dtype: float32 (scalar)
    description: Per-layer V cache quantization scale
```

#### NVFP4 Block Scale Interleaving

The existing TRT-LLM weight loader applies block scale interleaving for NVFP4 weights via `torch.ops.trtllm.block_scale_interleave()`. This rearranges the FP8 per-block scales into a layout optimized for the NVFP4 GEMM kernel. This transformation is applied during weight loading, not during inference.

#### Key Quantization Loading Pitfalls

1. **Gate/Up Weight Scale Reconciliation**: When fusing `gate_proj` and `up_proj` into `gate_up_proj`, their `weight_scale_2` values may differ. The loader must use the larger value and re-quantize the weights of the module with the smaller scale. This involves:
   - Dequantizing: FP4 -> BF16 using the original scale
   - Re-quantizing: BF16 -> FP4 using the reconciled scale
   - The `requantize_weight_with_new_scale` function in the existing weight loader handles this.

2. **Attention Weights Are BF16**: Despite the model being NVFP4 quantized, ALL attention weights (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj, layernorms) are stored in bfloat16. The weight loader must detect this and skip quantization-related loading for these modules. The `is_module_excluded_from_quantization()` method on the `QuantConfig` is used for this check.

3. **MoE Expert Weight Stacking**: The MoE `create_moe()` implementation expects expert weights to be loaded and stacked into a single tensor per parameter type (e.g., all 384 experts' w1 weights stacked into `[384, 2048, 3584]`). The `load_weights` method of the MoE module handles this stacking.

4. **Vision Tower and Projector**: These are entirely unquantized (bfloat16). They should be loaded with no quantization config applied. The vLLM code explicitly passes `quant_config=None` when `CompressedTensorsConfig` is detected (see `_maybe_ignore_quant_config`). In TRT-LLM, the multimodal wrapper should ensure the vision tower and projector are excluded from the quantization config.
