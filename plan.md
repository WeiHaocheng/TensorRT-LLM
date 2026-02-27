# TensorRT-LLM Porting Plan: GptOssForCausalLM

Generated: 2026-02-26
Model: gpt-oss-20b
Source: HuggingFace Transformers (`modeling_gpt_oss.py`)
Checkpoint: `/home/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b/`

---

## Part 1: Module Mapping

### 1.1 Model Overview

| Property | HuggingFace | TensorRT-LLM |
|----------|-------------|--------------|
| Model Class | `GptOssForCausalLM` | `GptOssForCausalLM` |
| Base Class | `PreTrainedModel` | `DecoderModelForCausalLM` |
| Config Class | `GptOssConfig` | Reuse HuggingFace `GptOssConfig` via `pretrained_config` |
| Architecture | MoE Transformer (Decoder-only) | MoE Transformer (Decoder-only) |
| Num Layers | 24 | 24 |
| Hidden Size | 2880 | 2880 |
| Num Attention Heads | 64 | 64 |
| Num KV Heads | 8 (GQA) | 8 (GQA) |
| Head Dim | 64 | 64 |
| Intermediate Size | 2880 | 2880 |
| Num Experts | 32 | 32 |
| Experts per Token | 4 (top-4) | 4 (top-4) |
| Vocab Size | 201088 | 201088 |
| RoPE | YaRN with theta=150000 | YaRN with theta=150000 |
| Attention Bias | True | True |
| Sliding Window | 128 (alternating layers) | 128 (alternating layers) |
| Normalization | RMSNorm (eps=1e-5) | RMSNorm (eps=1e-5) |
| Activation | Custom SwiGLU (alpha=1.702, limit=7.0) | SwiGLU with swiglu_alpha/swiglu_limit params |
| Quantization | MXFP4 (experts only) | MXFP4 (via `QuantAlgo.W4A8_MXFP4_MXFP8` or `W4A16_MXFP4`) |
| Tie Word Embeddings | False | False |

### 1.2 Module Mapping Table

| HuggingFace Module Path | HuggingFace Class | TensorRT-LLM Class | Notes |
|------------------------|-------------------|-------------------|-------|
| `model.embed_tokens` | `nn.Embedding` | `Embedding` | Token embedding, shape `[201088, 2880]` |
| `model.layers[i].input_layernorm` | `GptOssRMSNorm` | `RMSNorm` | Pre-attention normalization |
| `model.layers[i].self_attn` | `GptOssAttention` | `Attention` | GQA with 64 heads, 8 KV heads, bias=True |
| `model.layers[i].self_attn.q_proj` | `nn.Linear` | (fused into `Attention.qkv_proj`) | Fuse Q, K, V into single QKV projection |
| `model.layers[i].self_attn.k_proj` | `nn.Linear` | (fused into `Attention.qkv_proj`) | Fuse Q, K, V into single QKV projection |
| `model.layers[i].self_attn.v_proj` | `nn.Linear` | (fused into `Attention.qkv_proj`) | Fuse Q, K, V into single QKV projection |
| `model.layers[i].self_attn.o_proj` | `nn.Linear` | `Attention.o_proj` (Linear) | Output projection with bias=True |
| `model.layers[i].self_attn.sinks` | `nn.Parameter` | Passed as `attention_sinks` argument to `Attention.forward()` | Per-head attention sink logits, shape `[64]` |
| `model.layers[i].post_attention_layernorm` | `GptOssRMSNorm` | `RMSNorm` | Pre-MLP normalization |
| `model.layers[i].mlp` | `GptOssMLP` | MoE wrapper module | Contains router + experts |
| `model.layers[i].mlp.router` | `GptOssTopKRouter` | Gate module (custom) | Router with weight and bias, projects to `[num_experts]` |
| `model.layers[i].mlp.experts` | `GptOssExperts` | `create_moe(...)` | 32 experts, each with gate_up + down projections |
| `model.layers[i].mlp.experts.gate_up_proj` | `nn.Parameter` | MoE `w1` weight | Fused gate+up, shape `[32, 2880, 5760]` (transposed vs TRT-LLM) |
| `model.layers[i].mlp.experts.gate_up_proj_bias` | `nn.Parameter` | MoE `w1_bias` | Bias shape `[32, 5760]` |
| `model.layers[i].mlp.experts.down_proj` | `nn.Parameter` | MoE `w2` weight | Down projection, shape `[32, 2880, 2880]` (transposed vs TRT-LLM) |
| `model.layers[i].mlp.experts.down_proj_bias` | `nn.Parameter` | MoE `w2_bias` | Bias shape `[32, 2880]` |
| `model.norm` | `GptOssRMSNorm` | `RMSNorm` | Final normalization |
| `lm_head` | `nn.Linear` | `LMHead` | Language model head, no bias |
| `model.rotary_emb` | `GptOssRotaryEmbedding` | `RotaryEmbedding` (built into `Attention`) | YaRN RoPE, theta=150000 |

### 1.3 Module Compatibility Verification Summary

All modules have been verified against TensorRT-LLM source code:

| Module / Feature | Status | Verification Notes |
|-----------------|--------|-------------------|
| Embedding | ✅ Supported | Standard `Embedding` module |
| RMSNorm | ✅ Supported | Standard `RMSNorm` module |
| Attention (GQA, 64 heads / 8 KV heads) | ✅ Supported | Standard `Attention` module with explicit `head_dim=64` |
| Attention Sinks | ✅ Supported | Native `attention_sinks` arg in `Attention.forward()`, TRTLLM backend only. Threaded end-to-end to C++ kernel. |
| Sliding Window (alternating layers) | ✅ Supported | Per-layer `PredefinedAttentionMask.SLIDING_WINDOW_CAUSAL` vs `CAUSAL` |
| YaRN RoPE | ✅ Supported | Built into `Attention` via `RopeParams(scaling_type="yarn")` |
| MoE (32 experts, top-4) | ✅ Supported | `create_moe()` with `RenormalizeMoeRoutingMethod` |
| MoE Expert Biases | ✅ Supported | `create_moe(bias=True)` — supported in CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE |
| MoE Router with Bias | ✅ Supported | Custom `GptOssGate` class (~30 lines) passing `self.bias` to `cublas_mm` op (which already accepts bias). Follows `Qwen3Gate`/`DeepseekV3Gate` pattern. |
| Custom SwiGLU (alpha=1.702, limit=7.0) | ✅ Supported | `create_moe(swiglu_alpha=..., swiglu_beta=..., swiglu_limit=...)`. TRT-LLM `swiglu_torch` semantically identical to HF `_apply_gate`. Supported in CutlassFusedMoE, TritonFusedMoE, TRTLLMGenFusedMoE (MXFP4/NVFP4 required for TRTLLMGen). |
| MXFP4 Quantization (experts only) | ✅ Supported | `QuantAlgo.W4A8_MXFP4_MXFP8` (SM100) or `W4A16_MXFP4` (older) |
| LMHead | ✅ Supported | Standard `LMHead` module |

### 1.4 TensorRT-LLM Module Initialization Parameters

#### Embedding

```
Embedding(
    num_embeddings=201088,           # config.vocab_size
    embedding_dim=2880,              # config.hidden_size
    dtype=torch.bfloat16,            # model dtype
    mapping=mapping,                 # parallel mapping
    tensor_parallel_mode=TensorParallelMode.ROW,
    gather_output=True,
)
```

#### RMSNorm (input_layernorm, post_attention_layernorm, norm)

```
RMSNorm(
    hidden_size=2880,                # config.hidden_size
    eps=1e-5,                        # config.rms_norm_eps
    dtype=torch.bfloat16,            # NOTE: _keep_in_fp32_modules suggests FP32 for norms
)
```

**Important**: The HuggingFace model lists `post_attention_layernorm`, `input_layernorm`, and `norm` in `_keep_in_fp32_modules`, meaning these layers should preserve FP32 precision. The checkpoint stores these as BF16 weights (`shape=[2880], dtype=torch.bfloat16`), but during inference they should operate in FP32 for numerical stability.

#### Attention (self_attn)

```
Attention(
    hidden_size=2880,                # config.hidden_size
    num_attention_heads=64,          # config.num_attention_heads
    num_key_value_heads=8,           # config.num_key_value_heads
    max_position_embeddings=131072,  # config.max_position_embeddings
    bias=True,                       # config.attention_bias (both QKV and O projections)
    pos_embd_params=PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(
            theta=150000.0,          # config.rope_theta
            max_positions=131072,    # config.max_position_embeddings
            dim=64,                  # config.head_dim
            scaling_type="yarn",     # config.rope_scaling.rope_type
            scaling_factor=32.0,     # config.rope_scaling.factor
            original_max_position_embeddings=4096,  # config.rope_scaling.original_max_position_embeddings
            beta_fast=32.0,          # config.rope_scaling.beta_fast
            beta_slow=1.0,           # config.rope_scaling.beta_slow
        ),
    ),
    layer_idx=i,                     # layer index
    dtype=torch.bfloat16,
    config=model_config,
)
```

**Key Notes on Attention**:
- `head_dim` is explicitly 64 (not derived from `hidden_size // num_attention_heads` which would be 45). This means `q_size = 64 * 64 = 4096` and `kv_size = 8 * 64 = 512`. The `o_proj` shape confirms this: `[2880, 4096]`.
- `bias=True` applies to both QKV and O projections. The `dense_bias` parameter does not need separate override.
- **Attention sinks**: The model has a per-head `sinks` parameter (`shape=[num_attention_heads]`) that is passed as `attention_sinks` to `Attention.forward()`. This is a novel feature that adds a learned sink logit per head concatenated to attention logits before softmax, then dropped after softmax. TRT-LLM `Attention.forward()` supports this via the `attention_sinks` argument (TRTLLM backend only).
- **Sliding window**: Alternating layers use `sliding_attention` (window=128) and `full_attention`. This is controlled per-layer via `attention_mask` (using `PredefinedAttentionMask.CAUSAL` for full attention and `PredefinedAttentionMask.SLIDING_WINDOW_CAUSAL` or `attention_window_size=128` for sliding window layers).
- **RoPE style**: The HuggingFace model uses chunk-based RoPE (`first_half, second_half = torch.chunk(x, 2, dim=-1)`) which corresponds to `rope_gpt_neox` (non-interleaved) style in TRT-LLM, with `is_neox=True`.

#### MoE (mlp) -- Gate Module

The router should be implemented as a custom Gate module:

```
GptOssGate(
    hidden_size=2880,                # config.hidden_size
    num_experts=32,                  # config.num_local_experts
    dtype=torch.bfloat16,
    bias=True,                       # GptOssTopKRouter has both weight and bias
    routing_method=RenormalizeMoeRoutingMethod(
        top_k=4,                     # config.num_experts_per_tok
        output_dtype=torch.float32,
    ),
)
```

**Key Note on Routing**: The HuggingFace `GptOssTopKRouter` computes:
1. `router_logits = F.linear(hidden_states, weight, bias)` -- raw logits
2. `router_top_value, router_indices = torch.topk(router_logits, top_k)` -- top-k on raw logits
3. `router_scores = softmax(router_top_value)` -- softmax on top-k values only

This matches `RenormalizeMoeRoutingMethod` in TRT-LLM, which does top-k first then softmax (renormalize) on the selected values. The router has **bias**, which is unusual and must be accounted for.

#### MoE (mlp) -- Expert Module

```
create_moe(
    routing_method=RenormalizeMoeRoutingMethod(top_k=4),
    num_experts=32,                  # config.num_local_experts
    hidden_size=2880,                # config.hidden_size
    intermediate_size=2880,          # config.intermediate_size
    dtype=torch.bfloat16,
    model_config=model_config,
    weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
    bias=True,                       # experts have gate_up_proj_bias and down_proj_bias
    layer_idx=i,
    activation_type=ActivationType.Swiglu,
    swiglu_alpha=torch.tensor([1.702] * 32, dtype=torch.float32).cuda(),
    swiglu_beta=torch.tensor([1.0] * 32, dtype=torch.float32).cuda(),
    swiglu_limit=torch.tensor([7.0] * 32, dtype=torch.float32).cuda(),
)
```

**Key Notes on MoE**:
- **Fused gate+up**: HuggingFace stores `gate_up_proj` as a single parameter of shape `[num_experts, hidden_size, 2 * intermediate_size]`, which is already fused. Use `MoEWeightLoadingMode.FUSED_GATE_UP_PROJ`.
- **Bias**: Both gate_up and down projections have biases. Set `bias=True` in `create_moe`.
- **Custom SwiGLU activation**: The model uses a parameterized SwiGLU: `gate = gate.clamp(max=limit)`, `up = up.clamp(-limit, limit)`, `glu = gate * sigmoid(gate * alpha)`, `output = (up + 1) * glu`. The `swiglu_alpha`, `swiglu_beta`, and `swiglu_limit` parameters in `create_moe` support this. Note that the HuggingFace implementation uses `alpha=1.702`, `limit=7.0`, and the `(up + 1)` term corresponds to `swiglu_beta=1.0`.
- **MXFP4 quantization**: Expert weights are quantized to MXFP4 format (see Part 3).
- **Interleaved gate/up**: The HuggingFace model uses an interleaved layout for gate and up: `gate = gate_up[..., ::2]`, `up = gate_up[..., 1::2]`. This means the gate_up_proj weight has gate and up interleaved along the last dimension, rather than concatenated `[gate, up]`. This requires a de-interleaving transformation when loading weights.

#### LMHead

```
LMHead(
    num_embeddings=201088,           # config.vocab_size
    embedding_dim=2880,              # config.hidden_size
    bias=False,                      # no bias on lm_head
    dtype=torch.bfloat16,
    mapping=mapping,
    tensor_parallel_mode=TensorParallelMode.COLUMN,
    gather_output=True,
)
```

---

## Part 2: Weight Loading

### 2.1 Weight Loading Overview

| Checkpoint Format | File Pattern | Loading Method |
|------------------|--------------|----------------|
| SafeTensors (Sharded) | `model-*-of-00002.safetensors` | `safetensors.torch.load_file()`, merge shards via `model.safetensors.index.json` |

Total size: ~13.76 GB across 3 shard files.

### 2.2 Weight Name Table

#### HuggingFace Module Weights (per layer i, for i in 0..23)

**Attention Weights** (BF16, not quantized per `modules_to_not_convert`):
- `model.layers.{i}.self_attn.q_proj.weight` -- shape: `[4096, 2880]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.q_proj.bias` -- shape: `[4096]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.k_proj.weight` -- shape: `[512, 2880]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.k_proj.bias` -- shape: `[512]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.v_proj.weight` -- shape: `[512, 2880]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.v_proj.bias` -- shape: `[512]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.o_proj.weight` -- shape: `[2880, 4096]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.o_proj.bias` -- shape: `[2880]`, dtype: `bfloat16`
- `model.layers.{i}.self_attn.sinks` -- shape: `[64]`, dtype: `bfloat16`

**MoE Router Weights** (BF16, not quantized per `modules_to_not_convert`):
- `model.layers.{i}.mlp.router.weight` -- shape: `[32, 2880]`, dtype: `bfloat16`
- `model.layers.{i}.mlp.router.bias` -- shape: `[32]`, dtype: `bfloat16`

**MoE Expert Weights** (MXFP4 quantized -- blocks + scales):
- `model.layers.{i}.mlp.experts.gate_up_proj_blocks` -- shape: `[32, 5760, 90, 16]`, dtype: `uint8`
- `model.layers.{i}.mlp.experts.gate_up_proj_scales` -- shape: `[32, 5760, 90]`, dtype: `uint8`
- `model.layers.{i}.mlp.experts.gate_up_proj_bias` -- shape: `[32, 5760]`, dtype: `bfloat16`
- `model.layers.{i}.mlp.experts.down_proj_blocks` -- shape: `[32, 2880, 90, 16]`, dtype: `uint8`
- `model.layers.{i}.mlp.experts.down_proj_scales` -- shape: `[32, 2880, 90]`, dtype: `uint8`
- `model.layers.{i}.mlp.experts.down_proj_bias` -- shape: `[32, 2880]`, dtype: `bfloat16`

**Normalization Weights** (BF16):
- `model.layers.{i}.input_layernorm.weight` -- shape: `[2880]`, dtype: `bfloat16`
- `model.layers.{i}.post_attention_layernorm.weight` -- shape: `[2880]`, dtype: `bfloat16`

**Non-layer Weights**:
- `model.embed_tokens.weight` -- shape: `[201088, 2880]`, dtype: `bfloat16`
- `model.norm.weight` -- shape: `[2880]`, dtype: `bfloat16`
- `lm_head.weight` -- shape: `[201088, 2880]`, dtype: `bfloat16`

#### TensorRT-LLM Module Weights

- `vocab_embedding.weight`
- `layers.{i}.attention.qkv_proj.weight` (fused Q+K+V)
- `layers.{i}.attention.qkv_proj.bias` (fused Q+K+V)
- `layers.{i}.attention.o_proj.weight`
- `layers.{i}.attention.o_proj.bias`
- `layers.{i}.attention.sinks` (custom parameter, shape `[num_heads_per_tp]`)
- `layers.{i}.mlp.gate.weight` (router weight)
- `layers.{i}.mlp.gate.bias` (router bias)
- `layers.{i}.mlp.experts.w1` (fused gate+up, MXFP4)
- `layers.{i}.mlp.experts.w1_bias` (fused gate+up bias)
- `layers.{i}.mlp.experts.w1_weight_scale` (MXFP4 scales)
- `layers.{i}.mlp.experts.w2` (down projection, MXFP4)
- `layers.{i}.mlp.experts.w2_bias` (down projection bias)
- `layers.{i}.mlp.experts.w2_weight_scale` (MXFP4 scales)
- `layers.{i}.input_layernorm.weight`
- `layers.{i}.post_layernorm.weight`
- `norm.weight`
- `lm_head.weight`

### 2.3 Weight Loading Details

This section documents potential modeling errors and issues that may occur during TensorRT-LLM implementation.

#### Attention Module QKV Fusion

- **Error**: TensorRT-LLM's `Attention` module requires QKV projection weights to be fused into a single `qkv_proj` weight tensor.
- **Issue**: HuggingFace stores separate `q_proj`, `k_proj`, and `v_proj` linear layers.
- **Solution**: Concatenate Q, K, V weights along dimension 0 before loading into `attention.qkv_proj.weight`.
- **Specifics for this model**: Q has shape `[4096, 2880]`, K has shape `[512, 2880]`, V has shape `[512, 2880]`. The fused QKV weight should have shape `[4096 + 512 + 512, 2880] = [5120, 2880]`. Similarly, the fused bias should be `[5120]`.

```python
# Pseudo-code for QKV fusion
qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
```

#### Head Dim vs Hidden Size Mismatch

- **Error**: The model has `hidden_size=2880` and `num_attention_heads=64`, but `head_dim=64` is explicitly specified in config rather than derived from `hidden_size // num_attention_heads` (which would be 45).
- **Issue**: This means `q_size = num_attention_heads * head_dim = 64 * 64 = 4096`, which is larger than `hidden_size = 2880`. The q_proj projects from 2880 to 4096.
- **Solution**: Ensure that `head_dim` is read from the config directly (`config.head_dim = 64`) rather than computed. TRT-LLM's `Attention.__init__` reads `head_dim` from `config.pretrained_config.head_dim` if available, which handles this correctly.

#### Attention Sinks (Novel Feature)

- **Error**: The model uses a learned "attention sinks" parameter per attention head that modifies attention computation.
- **Issue**: This is a custom parameter `self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))` in the HuggingFace `GptOssAttention`. During attention, the sink logit is appended to the attention logits as an extra column, softmax is computed over the extended logits, and then the sink column is dropped. This effectively acts as a learnable "softmax denominator adjustment" per head.
- **Solution**: TRT-LLM's `Attention.forward()` already supports the `attention_sinks` argument. The sinks parameter must be stored per-layer and passed during forward. Under tensor parallelism, the sinks tensor should be sliced to `[num_heads_per_tp]` for each TP rank.

#### MoE Expert Weight Transposition

- **Error**: HuggingFace and TensorRT-LLM use different weight layouts for MoE experts.
- **Issue**: HuggingFace `GptOssExperts` stores `gate_up_proj` with shape `[num_experts, hidden_size, 2 * intermediate_size]` (i.e., `[32, 2880, 5760]`), where the computation is `gate_up = current_state @ gate_up_proj[expert_idx]`. TensorRT-LLM expects `w1` with shape `[num_experts, 2 * intermediate_size, hidden_size]` (i.e., `[32, 5760, 2880]`).
- **Solution**: Transpose the weight from `[E, hidden, 2*inter]` to `[E, 2*inter, hidden]` during loading. However, since the checkpoint stores MXFP4 quantized blocks (not dense weights), the transposition must be handled at the block level (see Part 3 for MXFP4 details).

#### MoE Expert Gate/Up Interleaved Layout

- **Error**: The HuggingFace model uses an interleaved layout for gate and up projections within the fused weight.
- **Issue**: In `GptOssExperts._apply_gate()`, the gate and up values are extracted via `gate = gate_up[..., ::2]` and `up = gate_up[..., 1::2]` (stride-2 interleaving). This means the fused weight interleaves gate and up columns: `[g0, u0, g1, u1, ...]`. TensorRT-LLM's `FUSED_GATE_UP_PROJ` mode expects the gate and up weights to be concatenated: `[g0, g1, ..., u0, u1, ...]`.
- **Solution**: During weight loading, de-interleave the fused gate_up weight and bias:

```python
# Pseudo-code for de-interleaving
# HF layout: [E, out_dim_interleaved, hidden] where out_dim = 2 * intermediate_size
# gate columns at even indices, up columns at odd indices
gate_up_weight = hf_gate_up_proj  # [E, 2*inter, hidden] (after transposing)
gate = gate_up_weight[:, 0::2, :]  # [E, inter, hidden]
up = gate_up_weight[:, 1::2, :]    # [E, inter, hidden]
trtllm_w1 = torch.cat([gate, up], dim=1)  # [E, 2*inter, hidden]

# Same for bias
gate_up_bias = hf_gate_up_proj_bias  # [E, 2*inter]
gate_bias = gate_up_bias[:, 0::2]    # [E, inter]
up_bias = gate_up_bias[:, 1::2]      # [E, inter]
trtllm_w1_bias = torch.cat([gate_bias, up_bias], dim=1)  # [E, 2*inter]
```

#### MoE Router Bias

- **Error**: The MoE router has both weight and bias, which is uncommon.
- **Issue**: Most MoE implementations use a bias-free router (`nn.Linear(..., bias=False)`). The GptOss `GptOssTopKRouter` has `self.bias = nn.Parameter(torch.zeros(self.num_experts))`, and the forward uses `F.linear(hidden_states, self.weight, self.bias)`.
- **Solution**: The Gate module in TRT-LLM must store and apply the router bias. If using a custom Gate class, ensure `F.linear` with bias is used. The weight has shape `[32, 2880]` and bias has shape `[32]`.

#### Normalization Layer Name Differences

- **Error**: Post-attention normalization layer name differs between HuggingFace and TensorRT-LLM.
- **Issue**: HuggingFace uses `post_attention_layernorm` while TensorRT-LLM uses `post_layernorm`.
- **Solution**: Map `model.layers.{i}.post_attention_layernorm.weight` to `layers.{i}.post_layernorm.weight`.

#### Sliding Window Attention (Per-Layer)

- **Error**: The model uses alternating sliding window and full attention layers.
- **Issue**: `layer_types` in the config specifies: even-indexed layers (0, 2, 4, ...) use `sliding_attention` with window=128, odd-indexed layers (1, 3, 5, ...) use `full_attention`. This must be handled per-layer in the model's forward pass.
- **Solution**: In the DecoderLayer forward, select the appropriate `attention_mask` and `attention_window_size`:
  - For `sliding_attention` layers: use `PredefinedAttentionMask.SLIDING_WINDOW_CAUSAL` and `attention_window_size=128`
  - For `full_attention` layers: use `PredefinedAttentionMask.CAUSAL`

#### Custom SwiGLU Activation

- **Error**: The model uses a non-standard SwiGLU variant with clamping and parameterized alpha.
- **Issue**: The standard SwiGLU is `silu(gate) * up`. This model uses: `gate = clamp(gate, max=limit)`, `up = clamp(up, -limit, limit)`, `glu = gate * sigmoid(gate * alpha)`, `output = (up + 1) * glu`, where `alpha=1.702` and `limit=7.0`.
- **Solution**: Use the `swiglu_alpha`, `swiglu_beta`, and `swiglu_limit` parameters in `create_moe()`. These parameters are supported in NVFP4/MXFP4 quantization modes with the TRTLLMGen backend.

#### Tied Embeddings

- **Error**: `tie_word_embeddings` is `false` in this model, so `lm_head.weight` is a separate parameter.
- **Issue**: No issue -- both `model.embed_tokens.weight` and `lm_head.weight` exist as separate tensors in the checkpoint.
- **Solution**: Load them independently. No weight sharing needed.

---

## Part 3: Quantization

### 3.1 Quantization Strategy

**Source Configuration**: `config.json` -> `quantization_config`

```json
{
    "quantization_config": {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head"
        ],
        "quant_method": "mxfp4"
    }
}
```

**Analysis**:

| Property | Value |
|----------|-------|
| Quantization Type | MXFP4 (Microscaling FP4) |
| Quant Method | `mxfp4` |
| Group Size | 32 (MXFP4 block size, inferred from weight shapes: `90 * 16 = 1440` per row block when `hidden_size=2880`, meaning `2880 / 32 = 90` blocks of 32 elements) |
| Modules NOT Quantized | `self_attn` (all attention projections), `mlp.router`, `embed_tokens`, `lm_head` |
| Modules Quantized | `mlp.experts.gate_up_proj`, `mlp.experts.down_proj` (MoE expert weights only) |

**TensorRT-LLM Configuration**:

```yaml
quantization_type: mxfp4
trtllm_config:
  quant_algo: W4A8_MXFP4_MXFP8  # On SM100 (Blackwell); W4A16_MXFP4 on older GPUs
  group_size: 32
```

The `QuantAlgo` selection depends on GPU architecture:
- **SM100+ (Blackwell)**: `QuantAlgo.W4A8_MXFP4_MXFP8` (4-bit weights with 8-bit activations using MXFP8 format)
- **SM90 or older**: `QuantAlgo.W4A16_MXFP4` (4-bit weights with 16-bit activations)

The quant config is loaded via `ModelConfig.load_hf_quant_config()` which detects `quant_method == "mxfp4"` and automatically sets the appropriate `QuantAlgo`.

### 3.2 Excluded Modules

Modules excluded from quantization (using TensorRT-LLM naming conventions):

From the HuggingFace `modules_to_not_convert`:

```yaml
exclude_modules:
  # From HF config modules_to_not_convert:
  - "model.layers.*.self_attn"     # All attention projections remain BF16
  - "model.layers.*.mlp.router"    # Router weights remain BF16
  - "model.embed_tokens"           # Embedding remains BF16
  - "lm_head"                      # LM head remains BF16

  # TRT-LLM internal default exclude_modules for MXFP4:
  - "block.*.attn.out"
  - "block.*.mlp.gate"
  - "block.*.attn.qkv"
  - "embedding"
  - "unembedding"
```

**TRT-LLM naming convention mapping** (for the model's actual modules):

| HuggingFace Module | TRT-LLM Module | Quantized? |
|--------------------|----------------|------------|
| `model.embed_tokens` | `vocab_embedding` | No (BF16) |
| `model.layers.*.self_attn.q_proj` | `layers.*.attention.qkv_proj` (Q part) | No (BF16) |
| `model.layers.*.self_attn.k_proj` | `layers.*.attention.qkv_proj` (K part) | No (BF16) |
| `model.layers.*.self_attn.v_proj` | `layers.*.attention.qkv_proj` (V part) | No (BF16) |
| `model.layers.*.self_attn.o_proj` | `layers.*.attention.o_proj` | No (BF16) |
| `model.layers.*.mlp.router` | `layers.*.mlp.gate` | No (BF16) |
| `model.layers.*.mlp.experts.gate_up_proj` | `layers.*.mlp.experts.w1` | Yes (MXFP4) |
| `model.layers.*.mlp.experts.down_proj` | `layers.*.mlp.experts.w2` | Yes (MXFP4) |
| `model.layers.*.input_layernorm` | `layers.*.input_layernorm` | No (BF16) |
| `model.layers.*.post_attention_layernorm` | `layers.*.post_layernorm` | No (BF16) |
| `model.norm` | `norm` | No (BF16) |
| `lm_head` | `lm_head` | No (BF16) |

### 3.3 Quantization Weight Loading

#### MXFP4 Weight Format

The checkpoint stores MXFP4 quantized expert weights in a **blocks + scales** format:

**For gate_up_proj** (per layer):
- `gate_up_proj_blocks`: shape `[32, 5760, 90, 16]`, dtype `uint8`
  - `32` = number of experts
  - `5760` = 2 * intermediate_size (fused gate+up output dimension)
  - `90` = number of blocks per row (hidden_size / group_size = 2880 / 32 = 90)
  - `16` = packed block data (32 FP4 values packed into 16 bytes, 2 values per byte)
- `gate_up_proj_scales`: shape `[32, 5760, 90]`, dtype `uint8`
  - One scale per block (E8M0 format, 8-bit unsigned exponent)

**For down_proj** (per layer):
- `down_proj_blocks`: shape `[32, 2880, 90, 16]`, dtype `uint8`
  - `32` = number of experts
  - `2880` = hidden_size (down projection output dimension)
  - `90` = number of blocks per row (intermediate_size / group_size = 2880 / 32 = 90)
  - `16` = packed block data
- `down_proj_scales`: shape `[32, 2880, 90]`, dtype `uint8`
  - One scale per block

#### Weight Loading Considerations

**MXFP4 Block Layout**:
- Each block of 16 bytes contains 32 FP4 values (2 values packed per byte)
- The scale is an E8M0 value (8-bit unsigned exponent only, no mantissa) shared across the 32 values in each block
- Block dimension corresponds to the input (hidden_size) dimension, not the output dimension

**Transposition Issue for MXFP4 Weights**:
- HuggingFace stores expert weights in `[E, out_dim, num_blocks, block_size]` format, where `out_dim` is the output dimension of the linear layer
- The weight matrix semantically represents `output = input @ weight`, where `weight` has logical shape `[E, hidden_size, out_dim]` (HuggingFace convention: `[E, in_dim, out_dim]`)
- TensorRT-LLM MoE expects `w1` with shape `[E, out_dim, in_dim]` (transposed convention)
- Since the blocks encode the input dimension, and blocks are already organized with the output dimension as the second axis, the MXFP4 blocks may already be in the correct layout for TRT-LLM (depending on the MoE backend's expected format). Verify against the TRT-LLM MXFP4 loading code.

**De-interleaving for Gate/Up in MXFP4**:
- The gate_up_proj has the interleaved layout issue (gate at even indices, up at odd indices along the output dimension)
- De-interleaving must be done on the blocks and scales tensors along the second dimension (out_dim=5760):

```python
# Pseudo-code for MXFP4 de-interleaving
blocks = hf_gate_up_proj_blocks  # [E, 5760, 90, 16]
scales = hf_gate_up_proj_scales  # [E, 5760, 90]

gate_blocks = blocks[:, 0::2, :, :]  # [E, 2880, 90, 16]
up_blocks = blocks[:, 1::2, :, :]    # [E, 2880, 90, 16]
trtllm_w1_blocks = torch.cat([gate_blocks, up_blocks], dim=1)  # [E, 5760, 90, 16]

gate_scales = scales[:, 0::2, :]  # [E, 2880, 90]
up_scales = scales[:, 1::2, :]    # [E, 2880, 90]
trtllm_w1_scales = torch.cat([gate_scales, up_scales], dim=1)  # [E, 5760, 90]

# Bias de-interleaving (BF16, not MXFP4)
bias = hf_gate_up_proj_bias  # [E, 5760]
gate_bias = bias[:, 0::2]    # [E, 2880]
up_bias = bias[:, 1::2]      # [E, 2880]
trtllm_w1_bias = torch.cat([gate_bias, up_bias], dim=1)  # [E, 5760]
```

#### Summary of Quantization Weights per Layer

```yaml
quantization_weights:
  # Gate+Up projection (MXFP4 quantized)
  - name: layers.{i}.mlp.experts.w1
    source: model.layers.{i}.mlp.experts.gate_up_proj_blocks (de-interleaved)
    dtype: uint8 (packed FP4)
    shape: "[32, 5760, 90, 16]"
    description: MXFP4 packed blocks for fused gate+up projection

  - name: layers.{i}.mlp.experts.w1_weight_scale
    source: model.layers.{i}.mlp.experts.gate_up_proj_scales (de-interleaved)
    dtype: uint8 (E8M0)
    shape: "[32, 5760, 90]"
    description: MXFP4 per-block scales for gate+up projection

  - name: layers.{i}.mlp.experts.w1_bias
    source: model.layers.{i}.mlp.experts.gate_up_proj_bias (de-interleaved)
    dtype: bfloat16
    shape: "[32, 5760]"
    description: Bias for fused gate+up projection

  # Down projection (MXFP4 quantized)
  - name: layers.{i}.mlp.experts.w2
    source: model.layers.{i}.mlp.experts.down_proj_blocks
    dtype: uint8 (packed FP4)
    shape: "[32, 2880, 90, 16]"
    description: MXFP4 packed blocks for down projection

  - name: layers.{i}.mlp.experts.w2_weight_scale
    source: model.layers.{i}.mlp.experts.down_proj_scales
    dtype: uint8 (E8M0)
    shape: "[32, 2880, 90]"
    description: MXFP4 per-block scales for down projection

  - name: layers.{i}.mlp.experts.w2_bias
    source: model.layers.{i}.mlp.experts.down_proj_bias
    dtype: bfloat16
    shape: "[32, 2880]"
    description: Bias for down projection

  # Non-quantized weights (BF16)
  - name: layers.{i}.attention.qkv_proj.weight
    source: Fused from q_proj.weight + k_proj.weight + v_proj.weight
    dtype: bfloat16
    shape: "[5120, 2880]"

  - name: layers.{i}.attention.qkv_proj.bias
    source: Fused from q_proj.bias + k_proj.bias + v_proj.bias
    dtype: bfloat16
    shape: "[5120]"

  - name: layers.{i}.attention.o_proj.weight
    source: model.layers.{i}.self_attn.o_proj.weight
    dtype: bfloat16
    shape: "[2880, 4096]"

  - name: layers.{i}.attention.o_proj.bias
    source: model.layers.{i}.self_attn.o_proj.bias
    dtype: bfloat16
    shape: "[2880]"
```
