# TensorRT-LLM PyTorch Modules List

This document lists all user-facing modules under the TensorRT-LLM `_torch/modules` path, including a brief description and code path for each module.

---

## Table of Contents

1. [Core Modules](#core-modules)
   - [Attention](#attention)
   - [MLA (Multi-head Latent Attention)](#mla-multi-head-latent-attention)
   - [Linear](#linear)
   - [Embedding](#embedding)
   - [MLP (Multi-Layer Perceptron)](#mlp-multi-layer-perceptron)
   - [GatedMLP (Gated Multi-Layer Perceptron)](#gatedmlp-gated-multi-layer-perceptron)
2. [Normalization Modules](#normalization-modules)
   - [RMSNorm](#rmsnorm)
   - [LayerNorm](#layernorm)
3. [Positional Encoding Modules](#positional-encoding-modules)
   - [RotaryEmbedding](#rotaryembedding)
   - [MRotaryEmbedding (Multi-dimensional Rotary Embedding)](#mrotaryembedding-multi-dimensional-rotary-embedding)
4. [Decoder Layer](#decoder-layer)
5. [Special Attention Modules](#special-attention-modules)
   - [QKNormRoPEAttention](#qknormropeattention)
6. [MoE (Mixture of Experts) Module](#moe-mixture-of-experts-module)
7. [Mamba Modules](#mamba-modules)
8. [Other Modules](#other-modules)
   - [LogitsProcessor](#logitsprocessor)
   - [LMHead](#lmhead)
   - [SwiGLU](#swiglu)
9. [FLA (Flash Linear Attention) Module](#fla-flash-linear-attention-module)

---

## Core Modules

### Attention

**Description**: Standard multi-head attention module, containing QKV projection, attention computation, and output projection. Supports RoPE positional encoding fusion, quantization, LoRA fine-tuning, and other features.

**Main Features**:
- Multi-head self-attention computation
- Grouped Query Attention (GQA) support
- RoPE positional encoding (optional fusion)
- Sliding Window Attention support
- Chunked Attention support
- Attention Sinks support
- Tensor Parallelism support
- Multiple quantization methods support
- LoRA fine-tuning support

**Code Path**: `tensorrt_llm/_torch/modules/attention.py`

---

### MLA (Multi-head Latent Attention)

**Description**: Multi-head Latent Attention module, primarily used for DeepSeek series models. Reduces KV cache size through low-rank compression and supports weight absorption optimization to improve inference performance.

**Main Features**:
- Low-rank KV compression
- Helix parallelism support
- Different optimization paths for context and generation phases
- DSA (Dynamic Sparse Attention) support

**Code Path**: `tensorrt_llm/_torch/modules/attention.py`

---

### Linear

**Description**: High-performance linear layer module supporting multiple quantization methods and tensor parallelism strategies. Serves as the fundamental component for Attention and MLP modules.

**Main Features**:
- Tensor Parallelism support (Column and Row modes)
- Multiple quantization methods support
- LoRA fine-tuning support
- Fused QKV and Gate-Up projection support

**Code Path**: `tensorrt_llm/_torch/modules/linear.py`

---

### Embedding

**Description**: Word embedding layer module that maps input token IDs to dense vector representations. Supports tensor parallelism and multiple parallelism strategies.

**Main Features**:
- Word embedding lookup
- Tensor Parallelism support (Column and Row modes)
- torch.compile optimization support

**Code Path**: `tensorrt_llm/_torch/modules/embedding.py`

---

### MLP (Multi-Layer Perceptron)

**Description**: Standard multi-layer perceptron module containing up projection and down projection linear layers connected through an activation function.

**Main Features**:
- Up projection and down projection
- Custom activation function support
- Tensor Parallelism support
- LoRA fine-tuning support

**Code Path**: `tensorrt_llm/_torch/modules/mlp.py`

---

### GatedMLP (Gated Multi-Layer Perceptron)

**Description**: Gated multi-layer perceptron module, an MLP variant using gating mechanisms (such as SwiGLU). Gate and Up projections are fused for improved performance.

**Main Features**:
- Fused Gate-Up projection
- SwiGLU activation support
- Quantized output support
- Tensor Parallelism support
- LoRA fine-tuning support

**Code Path**: `tensorrt_llm/_torch/modules/gated_mlp.py`

---

## Normalization Modules

### RMSNorm

**Description**: Root Mean Square Layer Normalization, more computationally efficient than LayerNorm by removing mean centering.

**Main Features**:
- RMS normalization
- Gemma-style normalization support
- Residual connection fusion support
- Quantized output support

**Utility Function**: `group_rms_norm()` can simultaneously perform RMS normalization on multiple inputs, offering better performance than normalizing them separately.

**Code Path**: `tensorrt_llm/_torch/modules/rms_norm.py`

---

### LayerNorm

**Description**: Standard Layer Normalization module with learnable weight and bias parameters.

**Main Features**:
- Standard layer normalization
- Residual connection support
- torch.compile optimization support

**Code Path**: `tensorrt_llm/_torch/modules/layer_norm.py`

---

## Positional Encoding Modules

### RotaryEmbedding

**Description**: Rotary Position Embedding (RoPE) implementation that injects positional information into the attention mechanism through rotary transformations.

**Main Features**:
- Standard RoPE positional encoding
- NeoX and non-NeoX interleaved modes support
- Batch application to Q and K

**Code Path**: `tensorrt_llm/_torch/modules/rotary_embedding.py`

---

### MRotaryEmbedding (Multi-dimensional Rotary Embedding)

**Description**: Multi-dimensional Rotary Embedding, used for multimodal models like Qwen-VL, supporting different positional encodings for different dimensions.

**Main Features**:
- Multi-dimensional positional encoding (temporal, height, width)
- Interleaved RoPE mode support
- Inherits from RotaryEmbedding

**Code Path**: `tensorrt_llm/_torch/modules/rotary_embedding.py`

---

## Decoder Layer

### DecoderLayer

**Description**: Abstract base class for Transformer decoder layers, defining the interface specification for decoder layers.

**Main Features**:
- Forward interface definition
- Residual connection support
- skip_forward support for skipping computation

**Code Path**: `tensorrt_llm/_torch/modules/decoder_layer.py`

---

## Special Attention Modules

### QKNormRoPEAttention

**Description**: RoPE attention module with QK Norm, used for models like Gemma3, Qwen3, and ExaOne4. Supports fused QK Norm and RoPE operations.

**Main Features**:
- Q/K normalization before RoPE
- Fused QK Norm + RoPE kernel support
- YARN positional encoding extension support

**Code Path**: `tensorrt_llm/_torch/modules/qk_norm_attention.py`

---

## MoE (Mixture of Experts) Module

**Description**: Mixture of Experts (MoE) module implementing sparse activation expert mixing mechanism, selectively activating a subset of experts for computation through a routing network.

**Constructor Function**: Use the `create_moe()` function to create MoE module instances, which automatically selects the optimal implementation based on configuration.

**Main Features**:
- Sparse expert activation
- Multiple routing strategies support
- Expert Parallelism support
- Quantization support

### Routing Methods

| Class Name | Description |
|------------|-------------|
| `DefaultMoeRoutingMethod` | Standard Top-K routing |
| `RenormalizeMoeRoutingMethod` | Routing with renormalization |
| `DeepSeekV3MoeRoutingMethod` | DeepSeek V3 specific routing |
| `LoadBalancedMoeRoutingMethod` | Load-balanced routing |
| `Llama4RenormalizeMoeRoutingMethod` | Llama 4 specific routing |

**Code Path**: `tensorrt_llm/_torch/modules/fused_moe/`

---

## Mamba Modules

**Description**: Collection of Mamba state space model related modules for Mamba and Mamba2 architectures. Mamba is a sequence model based on selective state spaces with linear time complexity.

### Mamba2Mixer

**Description**: Mamba2 core mixer module implementing Mamba2's state space model computation.

**Main Features**:
- Selective state space model computation
- Causal 1D convolution
- Tensor Parallelism support
- Paged state management support

**Code Path**: `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`

### Auxiliary Modules

| Module | Description | Code Path |
|--------|-------------|-----------|
| `CausalConv1d` | Causal 1D convolution | `mamba/causal_conv1d.py` |
| `LayerNormGated` | Gated layer normalization | `mamba/layernorm_gated.py` |
| `SelectiveStateUpdate` | Selective state update | `mamba/selective_state_update.py` |
| `SSDCombined` | Combined SSD scan operations | `mamba/ssd_combined.py` |

**Code Path**: `tensorrt_llm/_torch/modules/mamba/`

---

## Other Modules

### LogitsProcessor

**Description**: Logits processor for computing output logits from hidden states. Supports returning only the last token's logits or all context logits.

**Code Path**: `tensorrt_llm/_torch/modules/logits_processor.py`

---

### LMHead

**Description**: Language model head that maps hidden states to vocabulary-sized logits. Inherits from Linear and supports tensor parallelism.

**Code Path**: `tensorrt_llm/_torch/modules/embedding.py`

---

### SwiGLU

**Description**: SwiGLU activation function implementation, combining Swish activation and Gated Linear Units.

**Main Features**:
- Swish (SiLU) activation
- Gated Linear Unit
- Quantized output support

**Code Path**: `tensorrt_llm/_torch/modules/swiglu.py`

---

## FLA (Flash Linear Attention) Module

**Description**: Flash Linear Attention related modules for efficient linear attention computation, primarily used for linear attention models such as Delta Rule.

**Code Path**: `tensorrt_llm/_torch/modules/fla/`

---

## Module Dependencies

```
DecoderLayer
    ├── Attention / MLA / QKNormRoPEAttention
    │   ├── Linear (qkv_proj, o_proj)
    │   ├── RotaryEmbedding / MRotaryEmbedding
    │   └── RMSNorm (for MLA)
    ├── MLP / GatedMLP / MoE
    │   ├── Linear (up_proj, down_proj, gate_proj)
    │   └── SwiGLU (activation function)
    └── RMSNorm / LayerNorm (input/output normalization)
```

---

## Usage Examples

### Creating an Attention Module

```python
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.model_config import ModelConfig

attention = Attention(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=4096,
    bias=False,
    config=ModelConfig(),
)
```

### Creating a GatedMLP Module

```python
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.model_config import ModelConfig

mlp = GatedMLP(
    hidden_size=4096,
    intermediate_size=11008,
    bias=False,
    config=ModelConfig(),
)
```

### Creating an MoE Module

```python
from tensorrt_llm._torch.modules.fused_moe import create_moe
from tensorrt_llm._torch.model_config import ModelConfig

moe = create_moe(
    hidden_size=4096,
    intermediate_size=1024,
    num_experts=8,
    top_k=2,
    activation=ActivationType.Swiglu,
    config=ModelConfig(),
)
```
