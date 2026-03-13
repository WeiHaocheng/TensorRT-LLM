# TensorRT-LLM Porting Plan: MiMoV2FlashForCausalLM

Generated: 2026-03-12
Model: MiMo-V2-Flash (Xiaomi)
Source: HuggingFace custom model (`modeling_mimo_v2_flash.py`)

---

## Part 1: Module Mapping

### 1.1 Model Overview

| Property | HuggingFace | TensorRT-LLM |
|----------|-------------|--------------|
| Model Class | `MiMoV2FlashForCausalLM` | `MiMoV2FlashForCausalLM` |
| Base Class | `PreTrainedModel` | `DecoderModelForCausalLM` |
| Config Class | `MiMoV2FlashConfig` | `PretrainedConfig` (reuse HF config) |
| Architecture | Hybrid SWA/Full Attention + MoE/Dense MLP | Same |
| Num Layers | 48 | 48 |
| Hidden Size | 4096 | 4096 |
| Vocab Size | 152576 | 152576 |
| Torch Dtype | bfloat16 | bfloat16 |

### 1.2 Key Architectural Features

MiMo-V2-Flash has several distinctive features that affect the module mapping:

1. **Hybrid Attention Pattern** (`hybrid_layer_pattern`): A 48-element list where `0` = full attention (9 layers: 0, 5, 11, 17, 23, 29, 35, 41, 47) and `1` = sliding window attention (39 layers). The two types have **different** head configurations:
   - **Full Attention (pattern=0)**: `num_attention_heads=64`, `num_key_value_heads=4`, `head_dim=192`, `v_head_dim=128`
   - **SWA (pattern=1)**: `swa_num_attention_heads=64`, `swa_num_key_value_heads=8`, `swa_head_dim=192`, `swa_v_head_dim=128`

2. **Asymmetric Q/K vs V Head Dimensions**: Q and K projections use `head_dim=192` while V projection uses `v_head_dim=128`. This means the attention output has shape `[batch, seq, num_heads * v_head_dim]` = `[batch, seq, 64*128]` = `[batch, seq, 8192]`, NOT `[batch, seq, num_heads * head_dim]` = `[batch, seq, 64*192]` = `[batch, seq, 12288]`.

3. **Partial Rotary Embedding**: Only `partial_rotary_factor=0.334` of the head dimension receives RoPE. This means `rope_dim = int(192 * 0.334) = 64`. The remaining `192 - 64 = 128` dimensions of Q and K are NOT rotated (NoPE).

4. **Different RoPE Theta for SWA vs Full**:
   - Full attention: `rope_theta=5000000`
   - SWA: `swa_rope_theta=10000`

5. **Hybrid Dense/MoE MLP**: Layer 0 uses dense MLP (`moe_layer_freq[0]=0`), layers 1-47 use MoE (`moe_layer_freq[i]=1`).

6. **MoE Configuration**: 256 routed experts, 8 experts per token, sigmoid scoring + `noaux_tc` routing with `e_score_correction_bias`, `n_group=1`, `topk_group=1`.

7. **Attention Sink Bias**: SWA layers have a learnable per-head `attention_sink_bias` parameter (`add_swa_attention_sink_bias=true`). Full attention layers do NOT have this (`add_full_attention_sink_bias=false`).

8. **No Attention Bias**: All attention projections have `bias=False`.

### 1.3 Module Mapping Table

#### Top-Level Model

| HuggingFace Module Path | HuggingFace Class | TensorRT-LLM Class | Notes |
|------------------------|-------------------|-------------------|-------|
| `MiMoV2FlashForCausalLM` | `PreTrainedModel` | `DecoderModelForCausalLM` | Top-level CausalLM wrapper |
| `model` | `MiMoV2Model` | `DecoderModel` (custom subclass) | Base model containing layers |
| `model.embed_tokens` | `nn.Embedding` | `Embedding` | Token embedding |
| `model.layers[i]` | `MiMoV2DecoderLayer` | `DecoderLayer` (custom subclass) | Decoder layer |
| `model.norm` | `MiMoV2RMSNorm` | `RMSNorm` | Final normalization |
| `lm_head` | `nn.Linear` | `LMHead` (or `Linear`) | Language model head |

#### Attention (per layer `i`)

| HuggingFace Module Path | HuggingFace Class | TensorRT-LLM Class | Notes |
|------------------------|-------------------|-------------------|-------|
| `model.layers[i].self_attn` | `MiMoV2Attention` | `Attention` (custom subclass needed) | See Section 1.4 for critical issues |
| `model.layers[i].self_attn.q_proj` | `nn.Linear(4096, 12288)` | Part of fused `qkv_proj` | Q: `[hidden_size, num_heads * head_dim]` = `[4096, 64*192]` |
| `model.layers[i].self_attn.k_proj` | `nn.Linear(4096, 768)` (full) / `nn.Linear(4096, 1536)` (SWA) | Part of fused `qkv_proj` | K: `[hidden_size, num_kv_heads * head_dim]` = `[4096, 4*192]` (full) or `[4096, 8*192]` (SWA) |
| `model.layers[i].self_attn.v_proj` | `nn.Linear(4096, 512)` (full) / `nn.Linear(4096, 1024)` (SWA) | Part of fused `qkv_proj` | V: `[hidden_size, num_kv_heads * v_head_dim]` = `[4096, 4*128]` (full) or `[4096, 8*128]` (SWA) |
| `model.layers[i].self_attn.o_proj` | `nn.Linear(8192, 4096)` | `Linear` (dense/o_proj) | O: `[num_heads * v_head_dim, hidden_size]` = `[64*128, 4096]` |
| `model.layers[i].self_attn.attention_sink_bias` | `nn.Parameter(64)` | Attention sinks parameter | Only for SWA layers; per-head learnable bias |

#### Normalization (per layer `i`)

| HuggingFace Module Path | HuggingFace Class | TensorRT-LLM Class | Notes |
|------------------------|-------------------|-------------------|-------|
| `model.layers[i].input_layernorm` | `MiMoV2RMSNorm` | `RMSNorm` | Pre-attention normalization |
| `model.layers[i].post_attention_layernorm` | `MiMoV2RMSNorm` | `RMSNorm` | Pre-MLP normalization |

#### Dense MLP (Layer 0 only)

| HuggingFace Module Path | HuggingFace Class | TensorRT-LLM Class | Notes |
|------------------------|-------------------|-------------------|-------|
| `model.layers[0].mlp` | `MiMoV2MLP` | `GatedMLP` | Dense MLP with SwiGLU |
| `model.layers[0].mlp.gate_proj` | `nn.Linear(4096, 16384)` | Part of fused `gate_up_proj` | Fuse with up_proj |
| `model.layers[0].mlp.up_proj` | `nn.Linear(4096, 16384)` | Part of fused `gate_up_proj` | Fuse with gate_proj |
| `model.layers[0].mlp.down_proj` | `nn.Linear(16384, 4096)` | `Linear` (down_proj) | Down projection |

#### MoE (Layers 1-47)

| HuggingFace Module Path | HuggingFace Class | TensorRT-LLM Class | Notes |
|------------------------|-------------------|-------------------|-------|
| `model.layers[i].mlp` | `MiMoV2MoE` | Custom MoE wrapper (similar to `MiniMaxM2MoE`) | MoE module with routing |
| `model.layers[i].mlp.gate` | `MiMoV2MoEGate` | `Linear` (router) + routing method | Router linear + `DeepSeekV3MoeRoutingMethod` |
| `model.layers[i].mlp.gate.weight` | `nn.Parameter(256, 4096)` | `Linear(4096, 256)` | Router weight (gating projection) |
| `model.layers[i].mlp.gate.e_score_correction_bias` | `nn.Parameter(256)` | `nn.Parameter` in routing method | Bias for noaux_tc routing |
| `model.layers[i].mlp.experts[e]` | `MiMoV2MLP` | Part of fused MoE | 256 experts, each with gate/up/down projections |
| `model.layers[i].mlp.experts[e].gate_proj` | `nn.Linear(4096, 2048)` | Fused into MoE `fc1` | Expert gate projection |
| `model.layers[i].mlp.experts[e].up_proj` | `nn.Linear(4096, 2048)` | Fused into MoE `fc1` | Expert up projection |
| `model.layers[i].mlp.experts[e].down_proj` | `nn.Linear(2048, 4096)` | Fused into MoE `fc2` | Expert down projection |

### 1.4 Critical Architectural Issue: Asymmetric Q/K vs V Head Dimensions

The standard TensorRT-LLM `Attention` class assumes Q, K, and V all share the same `head_dim`. Specifically, in `Attention.__init__`:

```
self.kv_size = self.num_key_value_heads * self.head_dim
```

This means V is assumed to have the same head dimension as K. However, MiMo-V2-Flash has:

- **Q/K head_dim** = 192 (for computing attention scores with partial RoPE)
- **V head_dim** = 128 (for computing attention output)

Consequently, the QKV fusion logic and `o_proj` input size in the standard `Attention` class are incorrect for this model:

- Standard TRT-LLM: `qkv_proj` output = `num_heads * head_dim + 2 * num_kv_heads * head_dim`
- MiMo-V2-Flash needs: `qkv_proj` output = `num_heads * head_dim + num_kv_heads * head_dim + num_kv_heads * v_head_dim`

And for the output projection:
- Standard TRT-LLM: `o_proj` input = `num_heads * head_dim`
- MiMo-V2-Flash needs: `o_proj` input = `num_heads * v_head_dim` = `64 * 128 = 8192`

**Resolution**: A custom `Attention` subclass is required that:
1. Overrides the QKV projection to handle different K and V sizes
2. Overrides `split_qkv` to split with the correct sizes: `[q_size, k_size, v_size]` where `k_size != v_size`
3. Sets `o_proj` input dimension to `num_heads * v_head_dim` instead of `num_heads * head_dim`
4. Implements partial RoPE by splitting Q and K into rope/nope parts before applying rotary embedding

### 1.5 TensorRT-LLM Module Initialization Parameters

#### Embedding

```python
Embedding(
    num_embeddings=152576,       # config.vocab_size
    embedding_dim=4096,          # config.hidden_size
    dtype=torch.bfloat16,        # config.torch_dtype
    mapping=...,                 # Mapping object from ModelConfig
)
```

#### RMSNorm

```python
RMSNorm(
    hidden_size=4096,            # config.hidden_size
    eps=1e-05,                   # config.layernorm_epsilon
    dtype=torch.bfloat16,        # config.torch_dtype
)
```

#### Attention (Custom Subclass Needed)

For full attention layers (pattern=0):
```python
Attention(
    hidden_size=4096,                    # config.hidden_size
    num_attention_heads=64,              # config.num_attention_heads
    num_key_value_heads=4,               # config.num_key_value_heads
    max_position_embeddings=262144,      # config.max_position_embeddings
    bias=False,                          # config.attention_bias
    pos_embd_params=PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(
            theta=5000000,               # config.rope_theta
            dim=64,                      # int(head_dim * partial_rotary_factor) = int(192 * 0.334)
            max_positions=262144,
        ),
    ),
    layer_idx=...,
    dtype=torch.bfloat16,
    config=model_config,
)
# NOTE: head_dim will be read from config.head_dim = 192
# Custom subclass must override o_proj to use v_head_dim=128
```

For SWA layers (pattern=1):
```python
Attention(
    hidden_size=4096,                    # config.hidden_size
    num_attention_heads=64,              # config.swa_num_attention_heads
    num_key_value_heads=8,               # config.swa_num_key_value_heads
    max_position_embeddings=262144,      # config.max_position_embeddings
    bias=False,                          # config.attention_bias
    pos_embd_params=PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(
            theta=10000,                 # config.swa_rope_theta
            dim=64,                      # int(swa_head_dim * partial_rotary_factor) = int(192 * 0.334)
            max_positions=262144,
        ),
    ),
    layer_idx=...,
    dtype=torch.bfloat16,
    config=model_config,
    attention_chunk_size=128,            # config.attention_chunk_size (for SWA layers)
)
# NOTE: Attention sinks must be handled for SWA layers
# head_dim will be read from config as 192 (swa_head_dim same as head_dim)
# Custom subclass must override o_proj to use swa_v_head_dim=128
```

#### GatedMLP (Dense, Layer 0)

```python
GatedMLP(
    hidden_size=4096,                    # config.hidden_size
    intermediate_size=16384,             # config.intermediate_size
    bias=False,
    activation=F.silu,                   # config.hidden_act = "silu"
    dtype=torch.bfloat16,
    config=model_config,
    layer_idx=0,
)
```

#### MoE (Layers 1-47)

The MoE module should be constructed using `create_moe()` with the `DeepSeekV3MoeRoutingMethod` routing method (since MiMo-V2-Flash uses the same sigmoid + noaux_tc routing as DeepSeek V3):

```python
# Router gate (separate Linear)
gate = Linear(
    hidden_size=4096,                    # config.hidden_size
    num_experts=256,                     # config.n_routed_experts
    bias=False,
    dtype=torch.float32,                 # Router operates in FP32
)

# Routing method
routing_method = DeepSeekV3MoeRoutingMethod(
    top_k=8,                             # config.num_experts_per_tok
    n_group=1,                           # config.n_group
    topk_group=1,                        # config.topk_group
    routed_scaling_factor=1.0,           # config.routed_scaling_factor (None -> 1.0)
    callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
)

# MoE experts
experts = create_moe(
    routing_method=routing_method,
    num_experts=256,                     # config.n_routed_experts
    hidden_size=4096,                    # config.hidden_size
    intermediate_size=2048,              # config.moe_intermediate_size
    dtype=torch.bfloat16,
    reduce_results=True,
    model_config=model_config,
    layer_idx=...,
    activation_type=ActivationType.Swiglu,
)
```

#### LMHead

```python
LMHead(
    num_embeddings=152576,               # config.vocab_size
    embedding_dim=4096,                  # config.hidden_size
    dtype=torch.bfloat16,
    mapping=...,
)
```

### 1.6 Positional Embedding Details

The model uses **partial rotary embedding** applied in a split-and-concat manner:

1. Q and K are projected to `head_dim=192`
2. Split into `rope_part` (first 64 dims) and `nope_part` (remaining 128 dims)
3. RoPE is applied only to `rope_part`
4. Concatenated back: `[rope_part_rotated, nope_part]`

TRT-LLM's `RopeParams.from_config()` already reads `partial_rotary_factor` from the config and computes `dim = int(head_dim * partial_rotary_factor)`. However, the split-and-concat logic must be handled correctly in the custom attention subclass or through the attention backend's RoPE fusion.

Two separate rotary embeddings are needed:
- **Full attention layers**: `rope_theta=5000000`, `dim=64`
- **SWA layers**: `rope_theta=10000` (`swa_rope_theta`), `dim=64`

---

## Part 2: Weight Loading

### 2.1 Weight Loading Overview

| Checkpoint Format | File Pattern | Loading Method |
|------------------|--------------|----------------|
| SafeTensors (Sharded) | `model_*.safetensors` | `safetensors.torch.load_file()` |
| Index File | `model.safetensors.index.json` | Read to determine shard mapping |

**Shard Organization**:
- `model_0.safetensors` through `model_47.safetensors`: Per-layer non-expert weights (attention projections, norms, gate weights)
- `model_{i}_linear_fc1.safetensors`: Per-layer expert gate_proj + up_proj weights (for MoE layers)
- `model_{i}_linear_fc2.safetensors`: Per-layer expert down_proj weights (for MoE layers)
- `model_embedding.safetensors`: Embedding weights
- `model_final.safetensors`: Final norm + lm_head weights
- `model_mtp.safetensors`: Multi-token prediction weights (not needed for base model inference)

### 2.2 Weight Name Table

#### HuggingFace Module Weights

**Per-layer non-expert weights** (all 48 layers):
- `model.layers.{i}.self_attn.q_proj.weight` -- shape `[12288, 4096]`, dtype float8_e4m3fn
- `model.layers.{i}.self_attn.q_proj.weight_scale_inv` -- FP8 block scale
- `model.layers.{i}.self_attn.k_proj.weight` -- shape `[768, 4096]` (full) or `[1536, 4096]` (SWA)
- `model.layers.{i}.self_attn.k_proj.weight_scale_inv`
- `model.layers.{i}.self_attn.v_proj.weight` -- shape `[512, 4096]` (full) or `[1024, 4096]` (SWA)
- `model.layers.{i}.self_attn.v_proj.weight_scale_inv`
- `model.layers.{i}.self_attn.o_proj.weight` -- shape `[4096, 8192]`, dtype bfloat16 (NOT FP8, in `ignored_layers`)
- `model.layers.{i}.input_layernorm.weight` -- shape `[4096]`
- `model.layers.{i}.post_attention_layernorm.weight` -- shape `[4096]`

**SWA-layer-only weights** (pattern=1, 39 layers):
- `model.layers.{i}.self_attn.attention_sink_bias` -- shape `[64]`

**Dense MLP weights** (Layer 0 only):
- `model.layers.0.mlp.gate_proj.weight` -- shape `[16384, 4096]`, FP8
- `model.layers.0.mlp.gate_proj.weight_scale_inv`
- `model.layers.0.mlp.up_proj.weight` -- shape `[16384, 4096]`, FP8
- `model.layers.0.mlp.up_proj.weight_scale_inv`
- `model.layers.0.mlp.down_proj.weight` -- shape `[4096, 16384]`, FP8
- `model.layers.0.mlp.down_proj.weight_scale_inv`

**MoE gate weights** (Layers 1-47):
- `model.layers.{i}.mlp.gate.weight` -- shape `[256, 4096]`
- `model.layers.{i}.mlp.gate.e_score_correction_bias` -- shape `[256]`

**MoE expert weights** (Layers 1-47, 256 experts each):
- `model.layers.{i}.mlp.experts.{e}.gate_proj.weight` -- shape `[2048, 4096]`, FP8
- `model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale_inv`
- `model.layers.{i}.mlp.experts.{e}.up_proj.weight` -- shape `[2048, 4096]`, FP8
- `model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale_inv`
- `model.layers.{i}.mlp.experts.{e}.down_proj.weight` -- shape `[4096, 2048]`, FP8
- `model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale_inv`

**Global weights**:
- `model.embed_tokens.weight` -- shape `[152576, 4096]`
- `model.norm.weight` -- shape `[4096]`
- `lm_head.weight` -- shape `[152576, 4096]`

**MTP weights** (multi-token prediction, optional):
- `model.mtp.layers.{i}.*` -- Not needed for standard inference

#### TensorRT-LLM Module Weights

- `vocab_embedding.weight`
- `layers.{i}.attention.qkv_proj.weight` (fused from q_proj, k_proj, v_proj)
- `layers.{i}.attention.qkv_proj.weight_scale_inv` (fused FP8 block scales)
- `layers.{i}.attention.dense.weight` (from o_proj, bfloat16)
- `layers.{i}.attention.attention_sink_bias` (SWA layers only)
- `layers.{i}.input_layernorm.weight`
- `layers.{i}.post_layernorm.weight`
- **Dense MLP (layer 0)**:
  - `layers.0.mlp.gate_up_proj.weight` (fused gate + up)
  - `layers.0.mlp.gate_up_proj.weight_scale_inv`
  - `layers.0.mlp.down_proj.weight`
  - `layers.0.mlp.down_proj.weight_scale_inv`
- **MoE (layers 1-47)**:
  - `layers.{i}.mlp.gate.weight` (router)
  - `layers.{i}.mlp.e_score_correction_bias`
  - `layers.{i}.mlp.experts.fc1.weight` (fused gate+up for all experts, stacked)
  - `layers.{i}.mlp.experts.fc1.weight_scale_inv`
  - `layers.{i}.mlp.experts.fc2.weight` (down for all experts, stacked)
  - `layers.{i}.mlp.experts.fc2.weight_scale_inv`
- `norm.weight`
- `lm_head.weight`

### 2.3 Weight Loading Details -- Potential Modeling Errors and Pitfalls

#### 2.3.1 Attention QKV Fusion with Asymmetric V Dimension

**Error**: TensorRT-LLM's standard `Attention` fuses Q, K, V into a single `qkv_proj` with the assumption that K and V have the same per-head dimension (`head_dim`). In MiMo-V2-Flash, K uses `head_dim=192` but V uses `v_head_dim=128`.

**Issue**: The standard QKV weight concatenation `[Q, K, V]` along dim 0 produces a tensor of shape `[q_size + 2 * kv_size, hidden_size]` where `kv_size = num_kv_heads * head_dim`. But here, `k_size = num_kv_heads * head_dim` and `v_size = num_kv_heads * v_head_dim` are different.

**Solution**: Fuse QKV weights with correct sizes:
```
# Full attention (pattern=0):
Q: [12288, 4096]  (64 * 192)
K: [768, 4096]    (4 * 192)
V: [512, 4096]    (4 * 128)
QKV: [13568, 4096]

# SWA (pattern=1):
Q: [12288, 4096]  (64 * 192)
K: [1536, 4096]   (8 * 192)
V: [1024, 4096]   (8 * 128)
QKV: [14848, 4096]
```

The `split_qkv` method in the custom Attention subclass must split as `[q_size, k_size, v_size]` NOT `[q_size, kv_size, kv_size]`.

#### 2.3.2 FP8 Block Scaling with weight_scale_inv

**Error**: The checkpoint uses FP8 (E4M3) quantization with `weight_block_size=[128, 128]` and stores `weight_scale_inv` tensors (inverse scales). TRT-LLM expects block-wise FP8 scales in a specific format.

**Issue**: The `weight_scale_inv` tensors are per-block inverse scaling factors. The block size is `[128, 128]`, meaning for a weight of shape `[out, in]`, the scale tensor has shape `[ceil(out/128), ceil(in/128)]`.

**Solution**: Load `weight_scale_inv` and map it to TRT-LLM's `weight_scale_inv` (or convert to the expected scale format). TRT-LLM's FP8 block scaling support expects the same block-wise inverse scale format, so these should map directly.

#### 2.3.3 o_proj is NOT FP8 Quantized

**Error**: The `o_proj` weights for all layers are in the `ignored_layers` list of the quantization config. They are stored in bfloat16, not FP8.

**Issue**: When loading `o_proj` weights, do NOT apply FP8 dequantization. There is no `weight_scale_inv` for `o_proj`.

**Solution**: Check the `ignored_layers` list and load `o_proj` weights directly as bfloat16 without quantization handling.

#### 2.3.4 Heterogeneous Layer Configurations

**Error**: Full attention and SWA layers have different `num_key_value_heads` (4 vs 8) and therefore different weight shapes for K and V projections.

**Issue**: If the model implementation assumes all layers have the same attention configuration, weight loading will fail or produce incorrect results for some layers.

**Solution**: The decoder layer constructor must read `hybrid_layer_pattern[layer_idx]` and set attention parameters accordingly. Weight shapes vary per layer:

| Weight | Full Attention (pattern=0) | SWA (pattern=1) |
|--------|---------------------------|-----------------|
| `q_proj.weight` | `[12288, 4096]` | `[12288, 4096]` |
| `k_proj.weight` | `[768, 4096]` | `[1536, 4096]` |
| `v_proj.weight` | `[512, 4096]` | `[1024, 4096]` |
| `o_proj.weight` | `[4096, 8192]` | `[4096, 8192]` |

#### 2.3.5 MoE Gate Weight Transpose

**Error**: The HuggingFace `MiMoV2MoEGate` stores the gate weight as `nn.Parameter(torch.empty(n_routed_experts, gating_dim))` and computes `F.linear(hidden_states, self.weight)`. In TRT-LLM, this is typically a `Linear(hidden_size, n_routed_experts)` with weight shape `[n_routed_experts, hidden_size]`.

**Issue**: The weight shapes match (`[256, 4096]` in both cases), so no transpose is needed. However, the gate forward pass must cast inputs to FP32 before computing logits (as done in HuggingFace).

**Solution**: Map `model.layers.{i}.mlp.gate.weight` directly to the TRT-LLM router `gate.weight`. Ensure the router linear layer operates in FP32.

#### 2.3.6 MoE Expert Weight Stacking

**Error**: TRT-LLM's fused MoE expects expert weights stacked into a single tensor rather than stored as individual expert modules.

**Issue**: HuggingFace stores experts as individual `nn.Linear` modules: `experts.{e}.gate_proj.weight`, `experts.{e}.up_proj.weight`, `experts.{e}.down_proj.weight`. TRT-LLM expects:
- `fc1.weight`: shape `[num_experts, 2 * moe_intermediate_size, hidden_size]` = `[256, 4096, 4096]` (fused gate+up)
- `fc2.weight`: shape `[num_experts, hidden_size, moe_intermediate_size]` = `[256, 4096, 2048]`

**Solution**: Stack all 256 experts' gate_proj and up_proj weights into a single fc1 tensor, and all down_proj weights into a single fc2 tensor. For each expert `e`:
```
fc1[e] = concat([gate_proj.weight[e], up_proj.weight[e]], dim=0)  # [4096, 4096]
fc2[e] = down_proj.weight[e]  # [4096, 2048]
```

The corresponding `weight_scale_inv` tensors must also be stacked similarly.

#### 2.3.7 Dense MLP Gate/Up Fusion

**Error**: TRT-LLM's `GatedMLP` expects a fused `gate_up_proj` weight tensor.

**Issue**: Layer 0 has separate `gate_proj` and `up_proj` weights.

**Solution**: Concatenate along dim 0:
```
gate_up_proj.weight = concat([gate_proj.weight, up_proj.weight], dim=0)
# [32768, 4096] = concat([16384, 4096], [16384, 4096])
```

Also concatenate the corresponding `weight_scale_inv` tensors.

#### 2.3.8 Tied Embeddings

**Error**: The config has `tie_word_embeddings=false`, so `lm_head.weight` and `model.embed_tokens.weight` are separate.

**Issue**: No issue -- both weights exist independently in the checkpoint.

**Solution**: Load them separately. No weight sharing needed.

#### 2.3.9 Normalization Layer Name Mismatch

**Error**: HuggingFace uses `post_attention_layernorm` while TRT-LLM uses `post_layernorm`.

**Issue**: Weight name mapping must account for this difference.

**Solution**: Map `model.layers.{i}.post_attention_layernorm.weight` to `layers.{i}.post_layernorm.weight`.

#### 2.3.10 Attention Sink Bias Loading

**Error**: The `attention_sink_bias` is a non-gradient parameter that is only present in SWA layers.

**Issue**: Must only load this parameter for layers where `hybrid_layer_pattern[i] == 1`.

**Solution**: Check the layer pattern before loading. Map `model.layers.{i}.self_attn.attention_sink_bias` to the attention module's sink parameter. Shape is `[num_attention_heads]` = `[64]`.

#### 2.3.11 MoE Expert Weight Naming Convention (Discovered During Testing)

**Error**: TRT-LLM's fused MoE backend expects expert weights named `{expert_id}.w1.weight` (gate), `{expert_id}.w3.weight` (up), `{expert_id}.w2.weight` (down), but HuggingFace uses `gate_proj`, `up_proj`, `down_proj`.

**Issue**: Without renaming, the MoE backend's `load_expert_w3_w1_weight` fails with `w1_weight is None`.

**Solution**: Rename expert projections in `load_weights()`:
```
gate_proj -> w1
up_proj -> w3
down_proj -> w2
```

#### 2.3.12 FP8 Block Scale Per-Head Padding (Discovered During Testing)

**Error**: The checkpoint's FP8 block scales for K projections have more rows than `ceil(K_weight_rows / block_size)` due to per-head-aligned quantization padding. E.g., K weight `[768, 4096]` has scale shape `[8, 32]` instead of `[6, 32]` because the quantization tool aligns to head boundaries (`4 kv_heads * ceil(192/128) = 8`).

**Issue**: When the standard fused QKV loader concatenates individual Q/K/V scales, the sum of rows exceeds the fused module's expected scale size.

**Solution**: Trim scale rows to `ceil(weight_rows / block_size)` before fused QKV loading.

#### 2.3.13 Per-Layer KV Cache Allocation (Discovered During Testing)

**Error**: Full-attention layers use 4 KV heads while SWA layers use 8 KV heads. The global `config.num_key_value_heads=4` causes the KV cache to be allocated with only 4 KV heads for ALL layers.

**Issue**: SWA layers with 8 KV heads access out-of-bounds KV cache memory, causing CUDA illegal memory access.

**Solution**: After layer construction, override `config.num_key_value_heads` to a per-layer list so `get_bindings_model_config` allocates the correct KV cache per layer.

---

## Part 3: Quantization

### 3.1 Quantization Strategy

The model checkpoint uses **FP8 (E4M3) block-wise quantization** with dynamic activation scheme.

**Quantization Configuration** (from `config.json`):

```yaml
quantization_type: fp8
source_config: config.json (quantization_config section)
config_fields:
  quant_method: "fp8"
  activation_scheme: "dynamic"
  fmt: "e4m3"
  weight_block_size: [128, 128]
  packed_modules_mapping: {}
  ignored_layers:
    - model.layers.*.self_attn.o_proj  (all 48 layers)
    - model.decoder.self_attn.o_proj

trtllm_config:
  quant_algo: QuantAlgo.FP8  (specifically FP8 block scaling)
  weight_block_size: [128, 128]
  activation_scheme: dynamic
```

**Key Details**:

1. **FP8 E4M3 Format**: Weights are stored in `float8_e4m3fn` dtype with per-block inverse scaling factors (`weight_scale_inv`).

2. **Block Size [128, 128]**: Each 128x128 block of the weight matrix shares a single scaling factor. For a weight of shape `[M, N]`, the scale tensor has shape `[ceil(M/128), ceil(N/128)]`.

3. **Dynamic Activation Quantization**: Activations are quantized dynamically at runtime (not stored in the checkpoint). This means `activation_scheme: "dynamic"` -- no pre-computed activation scales are needed.

4. **Ignored Layers**: All `o_proj` layers across all 48 layers are excluded from FP8 quantization. These weights remain in bfloat16.

5. **TRT-LLM Mapping**: This corresponds to TRT-LLM's `QuantAlgo.FP8` with block scaling mode. TRT-LLM supports FP8 block-wise scaling through its `fp8_block_scales` quantization mode.

### 3.2 Excluded Modules

The following modules should be excluded from quantization (using TRT-LLM naming conventions):

```yaml
exclude_modules:
  # Embedding layers
  - vocab_embedding
  # Normalization layers
  - layers.*.input_layernorm
  - layers.*.post_layernorm
  - norm
  # LM head
  - lm_head
  # Output projection (explicitly in ignored_layers)
  - layers.*.attention.dense  # o_proj is NOT FP8 quantized
  # MoE router
  - layers.*.mlp.gate  # Router operates in FP32
  # Attention sink bias
  - layers.*.attention.attention_sink_bias
  # MoE e_score_correction_bias
  - layers.*.mlp.e_score_correction_bias
```

**Module Name Mapping from HuggingFace to TRT-LLM**:

| HuggingFace Name | TRT-LLM Name | Reason for Exclusion |
|------------------|--------------|---------------------|
| `model.embed_tokens` | `vocab_embedding` | Embedding, always full precision |
| `model.layers.{i}.input_layernorm` | `layers.{i}.input_layernorm` | Normalization, always full precision |
| `model.layers.{i}.post_attention_layernorm` | `layers.{i}.post_layernorm` | Normalization, always full precision |
| `model.norm` | `norm` | Normalization, always full precision |
| `lm_head` | `lm_head` | LM head, always full precision |
| `model.layers.{i}.self_attn.o_proj` | `layers.{i}.attention.dense` | In `ignored_layers`, bfloat16 |
| `model.layers.{i}.mlp.gate` | `layers.{i}.mlp.gate` | Router, operates in FP32 |

### 3.3 Quantization Weight Loading

#### FP8 Block-Scaled Weight Loading

For each FP8-quantized linear layer, two tensors must be loaded:

1. **`weight`** (dtype: `float8_e4m3fn`): The quantized weight tensor
2. **`weight_scale_inv`** (dtype: `float32` or `bfloat16`): Per-block inverse scaling factor with shape `[ceil(out_features/128), ceil(in_features/128)]`

**Loading Procedure**:

```yaml
quantization_weights:
  # Attention Q projection (FP8)
  - name: layers.{i}.attention.qkv_proj.weight
    source: Fused from q_proj.weight + k_proj.weight + v_proj.weight
    dtype: float8_e4m3fn
    description: Fused QKV weight, requires special handling for asymmetric V dim

  - name: layers.{i}.attention.qkv_proj.weight_scale_inv
    source: Fused from q_proj.weight_scale_inv + k_proj.weight_scale_inv + v_proj.weight_scale_inv
    dtype: float32
    description: Fused block scaling factors, must align block boundaries

  # Attention O projection (NOT FP8 -- bfloat16)
  - name: layers.{i}.attention.dense.weight
    source: model.layers.{i}.self_attn.o_proj.weight
    dtype: bfloat16
    description: Output projection, excluded from quantization

  # Dense MLP gate_up (FP8, Layer 0 only)
  - name: layers.0.mlp.gate_up_proj.weight
    source: Fused from gate_proj.weight + up_proj.weight
    dtype: float8_e4m3fn

  - name: layers.0.mlp.gate_up_proj.weight_scale_inv
    source: Fused from gate_proj.weight_scale_inv + up_proj.weight_scale_inv
    dtype: float32

  # Dense MLP down (FP8, Layer 0 only)
  - name: layers.0.mlp.down_proj.weight
    source: model.layers.0.mlp.down_proj.weight
    dtype: float8_e4m3fn

  - name: layers.0.mlp.down_proj.weight_scale_inv
    source: model.layers.0.mlp.down_proj.weight_scale_inv
    dtype: float32

  # MoE expert fc1 (FP8, Layers 1-47)
  - name: layers.{i}.mlp.experts.fc1.weight
    source: Stacked from experts.{e}.gate_proj.weight + experts.{e}.up_proj.weight for all e
    dtype: float8_e4m3fn
    shape: "[256, 4096, 4096]"
    description: Fused gate+up for all 256 experts

  - name: layers.{i}.mlp.experts.fc1.weight_scale_inv
    source: Stacked from experts.{e}.gate_proj.weight_scale_inv + experts.{e}.up_proj.weight_scale_inv
    dtype: float32
    description: Block scales for fused fc1

  # MoE expert fc2 (FP8, Layers 1-47)
  - name: layers.{i}.mlp.experts.fc2.weight
    source: Stacked from experts.{e}.down_proj.weight for all e
    dtype: float8_e4m3fn
    shape: "[256, 4096, 2048]"

  - name: layers.{i}.mlp.experts.fc2.weight_scale_inv
    source: Stacked from experts.{e}.down_proj.weight_scale_inv
    dtype: float32
```

#### Key Pitfalls in FP8 Quantization Weight Loading

1. **QKV Fusion with FP8 Block Scales**: When fusing Q, K, V weights and their block scales, the block boundaries must be carefully maintained. Since each component has its own block scale grid, concatenating the weights along dim 0 requires concatenating the corresponding scale rows as well.

2. **MoE Expert Stacking with FP8**: When stacking 256 experts into a single tensor, each expert's weight and scale tensors must be correctly indexed in the stacked dimension. The fused gate+up for each expert must have scales concatenated row-wise before stacking across experts.

3. **o_proj Exclusion**: The `o_proj` weight is bfloat16 with no `weight_scale_inv`. Do not attempt to load a scale tensor for this weight. When the TRT-LLM model has global FP8 quantization enabled, `o_proj` must be specifically excluded from quantization.

4. **Router in FP32**: The MoE gate weight (`mlp.gate.weight`) and `e_score_correction_bias` are in full precision. The router forward pass must cast hidden states to FP32 before computing logits, then the routing is performed in FP32.
