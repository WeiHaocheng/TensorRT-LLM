---
name: trtllm-modeling-analysis
description: Guide for analyzing HuggingFace models to determine TensorRT-LLM compatibility and generate a structured porting plan. Use this skill to produce a plan.md file (written to the current working directory) containing module mapping, weight loading details, and quantization analysis.
---

# TensorRT-LLM Model Analysis Skill

This skill provides guidance for analyzing a HuggingFace model and generating a structured porting plan for TensorRT-LLM implementation. The final output **must be written to a file named `plan.md` in the current working directory**.

## Overview

The Model Analysis skill takes a HuggingFace model and produces:

1. **Porting Plan** (`./plan.md`, written to the current working directory): A comprehensive document containing:
   - Module mapping between HuggingFace and TensorRT-LLM
   - Weight loading specifications
   - Quantization analysis
2. **Compatibility Conclusion**: Whether the model can be built using TensorRT-LLM

### ⚠️ Output Constraints — What NOT to Include

The generated `plan.md` must **strictly focus on analysis and mapping**. The following content is **explicitly prohibited** in the output:

1. **No work schedule or timeline plan.** Do NOT include any estimated timelines, milestones, Gantt charts, task assignments, person-day estimates, sprint plans, or any form of project scheduling. The plan is a *technical analysis document*, not a project management artifact.
2. **No TensorRT-LLM modeling implementation code.** Do NOT generate or include any TensorRT-LLM Python modeling code (e.g., model class definitions, `forward()` implementations, or full weight-loading scripts). The plan should only describe *what* to map and *what pitfalls to watch for*, not provide a ready-to-run implementation. Short illustrative pseudo-code snippets for weight transformations (e.g., showing how to concatenate QKV tensors) are acceptable, but complete modeling files are not.

## Inputs

| Input | Description | Example |
|-------|-------------|---------|
| `checkpoint_path` | Path to model checkpoint directory containing weights and config | `/models/llama-7b/` |
| `hf_modeling_path` | Path to HuggingFace transformers modeling source code | `/transformers/src/transformers/models/llama/modeling_llama.py` |

## Outputs

### Output 1: Porting Plan (`plan.md`)

**IMPORTANT**: This output must be written to a file named `plan.md` in the current working directory. Do NOT just print the content — use a file-write tool to persist it to disk so the user can access it after the agent finishes.

**NOTE**: When analyzing `hf_modeling_path`, do NOT only look at the modeling file itself. You should also examine **other code files in the same directory** (e.g., `configuration_*.py`, `convert_*.py`, etc.), as they may contain important config classes, model constants, or helper utilities that the modeling code depends on. Make sure to list and review all relevant files under that directory to get a complete picture of the model implementation.

A markdown document with three major sections:

#### Part 1: Module Mapping

Describes which HuggingFace modules should be represented by which TensorRT-LLM modules, including initialization parameters for each TensorRT-LLM module.

```markdown
# TensorRT-LLM Porting Plan: LlamaForCausalLM

## Part 1: Module Mapping

### 1.1 Model Overview

| Property | HuggingFace | TensorRT-LLM |
|----------|-------------|--------------|
| Model Class | `LlamaForCausalLM` | `LlamaForCausalLM` |
| Base Class | `PreTrainedModel` | `DecoderModelForCausalLM` |
| Config Class | `LlamaConfig` | `LlamaConfig` |

### 1.2 Module Mapping Table

| HuggingFace Module Path | HuggingFace Class | TensorRT-LLM Class | Notes |
|------------------------|-------------------|-------------------|-------|
| `model.embed_tokens` | `nn.Embedding` | `Embedding` | Token embedding |
| `model.layers[i].self_attn` | `LlamaAttention` | `Attention` | Multi-head attention with GQA support |
| `model.layers[i].self_attn.q_proj` | `nn.Linear` | `Linear` | Fuse with k_proj, v_proj into single QKV projection |
| `model.layers[i].self_attn.k_proj` | `nn.Linear` | `Linear` | Fuse with q_proj, v_proj into single QKV projection |
| `model.layers[i].self_attn.v_proj` | `nn.Linear` | `Linear` | Fuse with q_proj, k_proj into single QKV projection |
| `model.layers[i].self_attn.o_proj` | `nn.Linear` | `Linear` | Output projection |
| `model.layers[i].mlp.gate_proj` | `nn.Linear` | `Linear` | Fuse with up_proj into single projection |
| `model.layers[i].mlp.up_proj` | `nn.Linear` | `Linear` | Fuse with gate_proj into single projection |
| `model.layers[i].mlp.down_proj` | `nn.Linear` | `Linear` | Down projection |
| `model.layers[i].input_layernorm` | `LlamaRMSNorm` | `RMSNorm` | Pre-attention normalization |
| `model.layers[i].post_attention_layernorm` | `LlamaRMSNorm` | `RMSNorm` | Pre-MLP normalization |
| `model.norm` | `LlamaRMSNorm` | `RMSNorm` | Final normalization |
| `lm_head` | `nn.Linear` | `Linear` | Language model head |

### 1.3 TensorRT-LLM Module Initialization Parameters

**Critical**: Finding the correct initialization parameters for each TensorRT-LLM module is essential to ensure that the TensorRT-LLM modules have the same semantics as the corresponding HuggingFace model. Incorrect parameters can lead to semantic mismatches, causing the ported model to produce different outputs than the original HuggingFace model. Carefully map each parameter from the HuggingFace model configuration to the corresponding TensorRT-LLM module initialization argument, verifying that dimensions, data types, and behavioral flags (such as bias, normalization settings, etc.) match exactly.

For each TensorRT-LLM module listed in the mapping table above (Embedding, Attention, Linear, RMSNorm, etc.), look up its `__init__` signature directly in the TensorRT-LLM source code under `tensorrt_llm/_torch/modules/`. Document every initialization parameter, its type, its purpose, and how it should be derived from the HuggingFace model's configuration.
```

#### Part 2: Weight Loading

Lists weight names and documents potential modeling errors and issues that may occur during TensorRT-LLM implementation. Focuses on common pitfalls rather than comprehensive loading procedures.

```markdown
## Part 2: Weight Loading

### 2.1 Weight Loading Overview

| Checkpoint Format | File Pattern | Loading Method |
|------------------|--------------|----------------|
| SafeTensors | `*.safetensors` | `safetensors.torch.load_file()` |
| PyTorch | `pytorch_model*.bin` | `torch.load()` |
| Sharded | `model-*-of-*.safetensors` | Load and merge shards |

### 2.2 Weight Name Table

#### HuggingFace Module Weights

- `model.embed_tokens.weight`
- `model.layers.{i}.self_attn.q_proj.weight`
- `model.layers.{i}.self_attn.k_proj.weight`
- `model.layers.{i}.self_attn.v_proj.weight`
- `model.layers.{i}.self_attn.o_proj.weight`
- `model.layers.{i}.mlp.gate_proj.weight`
- `model.layers.{i}.mlp.up_proj.weight`
- `model.layers.{i}.mlp.down_proj.weight`
- `model.layers.{i}.input_layernorm.weight`
- `model.layers.{i}.post_attention_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight`

#### TensorRT-LLM Module Weights

- `vocab_embedding.weight`
- `layers.{i}.attention.qkv_proj.weight`
- `layers.{i}.attention.dense.weight`
- `layers.{i}.mlp.gate_up_proj.weight`
- `layers.{i}.mlp.down_proj.weight`
- `layers.{i}.input_layernorm.weight`
- `layers.{i}.post_layernorm.weight`
- `norm.weight`
- `lm_head.weight`

### 2.3 Weight Loading Details

This section documents potential modeling errors and issues that may occur during TensorRT-LLM implementation. Focus on common pitfalls and required transformations.

#### Common Modeling Errors

**Attention Module QKV Fusion**
- **Error**: TensorRT-LLM's `Attention` module requires QKV projection weights to be fused into a single `qkv_proj` weight tensor.
- **Issue**: HuggingFace models typically have separate `q_proj`, `k_proj`, and `v_proj` linear layers.
- **Solution**: Concatenate Q, K, V weights along dimension 0 before loading into `attention.qkv_proj.weight`.
- **Example**: For GQA models, concatenate as `[Q, K, V]` where Q shape is `[num_heads * head_dim, hidden_size]` and K/V shapes are `[num_kv_heads * head_dim, hidden_size]`.

**MLP Gate/Up Fusion**
- **Error**: TensorRT-LLM's MLP modules (e.g., SwiGLU) require gate and up projections to be fused.
- **Issue**: HuggingFace models have separate `gate_proj` and `up_proj` linear layers.
- **Solution**: Concatenate gate and up weights along dimension 0 before loading into `mlp.gate_up_proj.weight`.

**Weight Name Mismatches**
- **Error**: TensorRT-LLM uses different weight naming conventions than HuggingFace.
- **Issue**: Module paths differ (e.g., `model.layers.{i}.self_attn.o_proj` → `layers.{i}.attention.dense`).
- **Solution**: Map HuggingFace weight names to TensorRT-LLM weight names correctly.

**Tied Embeddings**
- **Error**: When `tie_word_embeddings=True`, `lm_head.weight` may not exist in checkpoint.
- **Issue**: Need to use `model.embed_tokens.weight` for both embedding and LM head.
- **Solution**: Check `config.tie_word_embeddings` and handle tied weights appropriately.

**Normalization Layer Names**
- **Error**: Post-attention normalization layer name differs between HuggingFace and TensorRT-LLM.
- **Issue**: HuggingFace uses `post_attention_layernorm` while TensorRT-LLM uses `post_layernorm`.
- **Solution**: Map weight names correctly when loading normalization weights.
```

#### Part 3: Quantization

Describes the quantization strategy, excluded modules, and how to load quantization-related weights.

```markdown
## Part 3: Quantization

### 3.1 Quantization Strategy

Read quantization configuration from checkpoint directory (e.g., `config.json` or `quantize_config.json`) and describe the quantization strategy used in the model.

**Key information to extract:**
1. **Quantization type**: Determine from `quantization_config.quant_method` or weight dtypes (none, gptq, awq, int8_weight_only, fp8, etc.)
2. **Configuration parameters**: Extract relevant parameters from config files:
   - `bits`: Quantization bit width (e.g., 4, 8)
   - `group_size`: Group size for quantization (if applicable)
   - Other quantization-specific parameters (e.g., `desc_act`, `sym`, `zero_point`, `fp8_format`)
3. **TensorRT-LLM configuration**: Map quantization type to TensorRT-LLM `QuantAlgo` and create `QuantConfig`:
   - `none` → `quant_config: null`
   - `gptq` → `QuantAlgo.W4A16_GPTQ` (or `W8A16_GPTQ` for 8-bit)
   - `awq` → `QuantAlgo.W4A16_AWQ`
   - `int8_weight_only` → `QuantAlgo.W8A16`
   - `fp8` → `QuantAlgo.FP8`

**Example structure:**

```yaml
quantization_type: <detected_type>  # none, gptq, awq, int8_weight_only, fp8, etc.
source_config: config.json  # or quantize_config.json
config_fields:
  # Extract relevant fields from quantization_config
  bits: <value>
  group_size: <value>  # if applicable
  # ... other quantization-specific parameters

trtllm_config:
  quant_algo: <QuantAlgo enum>
  group_size: <value>  # if applicable
  # ... other TensorRT-LLM quantization parameters
```

### 3.2 Excluded Modules

List modules that should be excluded from quantization. Module names must use TensorRT-LLM naming conventions (not HuggingFace names).

**Common excluded modules:**
- Embedding layers: `vocab_embedding`
- LayerNorm/RMSNorm layers: `layers.{i}.input_layernorm`, `layers.{i}.post_layernorm`, `norm`
- LM head: `lm_head`
- Other sensitive layers that should remain in full precision

**Example:**

```yaml
exclude_modules:
  - vocab_embedding
  - norm
  - lm_head
  - layers.*.input_layernorm
  - layers.*.post_layernorm
```

**Note**: When mapping from HuggingFace module names to TensorRT-LLM names:
- `model.embed_tokens` → `vocab_embedding`
- `model.norm` → `norm`
- `model.layers.{i}.input_layernorm` → `layers.{i}.input_layernorm`
- `model.layers.{i}.post_attention_layernorm` → `layers.{i}.post_layernorm`
- `lm_head` → `lm_head`

### 3.3 Quantization Weight Loading

Describe how to load quantization-related weights from the checkpoint. This includes additional weights beyond standard model weights.

**Common quantization weight types:**
- **GPTQ/AWQ**: `qweight` (packed quantized weights), `scales`, `qzeros`, `g_idx` (optional)
- **INT8 weight-only**: `weight` (int8 dtype), `weight_scale`
- **FP8**: `weight` (fp8 dtype), `weight_scale`

**Key considerations:**
1. **Weight name mapping**: Map HuggingFace weight names to TensorRT-LLM module names (use TensorRT-LLM naming conventions)
2. **Fusion requirements**: For fused modules (e.g., QKV projection, gate/up projection), load and concatenate quantization weights from separate HuggingFace layers
3. **Weight loading order**: Load quantization weights in the same order as standard weights, ensuring proper fusion

**Example structure:**

```yaml
quantization_weights:
  - name: layers.{i}.attention.qkv_proj.qweight
    source: model.layers.{i}.self_attn.q_proj.qweight (fused with k_proj, v_proj)
    dtype: int32
    description: Packed quantized weights for QKV projection
  - name: layers.{i}.attention.qkv_proj.scales
    source: model.layers.{i}.self_attn.q_proj.scales (fused)
    dtype: float16
    description: Quantization scales
  # ... additional quantization weights

loading_code: |
  # Load quantization weights for each module
  # For fused modules (QKV, gate/up), concatenate weights from separate HF layers
  # Map HF weight names to TensorRT-LLM module names
  # Use appropriate loading method based on quantization type
```
```

### Output 2: Compatibility Conclusion

A clear conclusion about whether the model can be built using TensorRT-LLM:

**Supported Model Example:**

```
╔══════════════════════════════════════════════════════════════════╗
║                    TensorRT-LLM COMPATIBILITY                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: LlamaForCausalLM                                         ║
║  Status: ✅ SUPPORTED                                            ║
╠══════════════════════════════════════════════════════════════════╣
║  All modules have TensorRT-LLM equivalents.                      ║
║  Quantization: FP16 (native support)                             ║
║                                                                  ║
║  Output: plan.md written to ./plan.md                            ║
╚══════════════════════════════════════════════════════════════════╝
```

**Unsupported Model Example:**

```
╔══════════════════════════════════════════════════════════════════╗
║                    TensorRT-LLM COMPATIBILITY                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: CustomMoEForCausalLM                                     ║
║  Status: ❌ NOT SUPPORTED                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Unsupported Modules:                                            ║
║    - MoELayer: Mixture of Experts routing not supported          ║
║    - CustomSparseAttention: Sparse attention not available       ║
║                                                                  ║
║  Unsupported Quantization:                                       ║
║    - GGML format not supported in TensorRT-LLM                   ║
║                                                                  ║
║  Output: plan.md written to ./plan.md (with limitations noted)   ║
╚══════════════════════════════════════════════════════════════════╝
```

## Analysis Workflow

### Step 1: Read Model Information

Obtain input parameters from Task context:

- `checkpoint_path`: Path to the model checkpoint directory
- `hf_modeling_path`: Path to HuggingFace transformers modeling source code

### Step 2: Analyze Checkpoint

Use tools to read files in the checkpoint directory:

1. **Read `config.json`**: Extract model configuration
   - `model_type`, `architectures`
   - `hidden_size`, `num_hidden_layers`, `num_attention_heads`
   - `num_key_value_heads` (GQA support)
   - `intermediate_size`, `hidden_act`
   - `rms_norm_eps`, `rope_theta`
   - `tie_word_embeddings`
   - `quantization_config` (if present)

2. **List weight files**: Get list of weight file names
   - `*.safetensors` or `pytorch_model*.bin`

3. **Read weight metadata**: Get tensor names, shapes, dtypes
   - Use `safetensors` metadata or load partial weights

**Example config.json key fields:**

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 11008,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "quant_method": "gptq"
  }
}
```

### Step 3: Analyze HuggingFace Modeling Code

Read and analyze the HuggingFace modeling source code file:

1. **Module structure**: Identify class hierarchy and child modules
   - `*ForCausalLM` → `*Model` → `*DecoderLayer` → `*Attention`, `*MLP`
   
2. **Weight definitions**: Identify all `nn.Linear`, `nn.Embedding`, `*Norm` layers
   - Extract module definitions from `__init__`
   
3. **Forward flow**: Analyze data flow
   - Pre-norm vs Post-norm
   - Attention pattern (MHA/GQA/MQA)
   - MLP pattern (SwiGLU/GeGLU/Standard)

### Step 4: Generate Module Mapping (Part 1)

Generate module mapping for the first part of plan.md:

1. **Read TensorRT-LLM modules reference**: Read the `modules_list.md` file from TensorRT-LLM to understand available modules, their initialization parameters, and capabilities

2. **Iterate through HuggingFace modules**: For each module in the HuggingFace model:
   - Identify the module type (Embedding, Linear, Attention, Norm, etc.)
   - Find the corresponding TensorRT-LLM class from `modules_list.md`
   - Record the initialization parameters required by the TensorRT-LLM class
   - Mark modules that need fusion (QKV, gate/up)

**Output format example:**

| HuggingFace Module | HuggingFace Class | TensorRT-LLM Class | Notes |
|-------------------|-------------------|-------------------|-------|
| `model.embed_tokens` | `nn.Embedding` | `Embedding` | Token embedding |
| `model.layers[i].self_attn` | `LlamaAttention` | `Attention` | Fuse QKV |

#### MoE-Specific Analysis

When the model uses Mixture of Experts, document the following in `plan.md`:

1. **MoE architecture**: `num_experts`, `experts_per_token`, routing method (top-k renormalize, etc.), whether the gate/router has bias.
2. **weight_loading_mode**: Determine whether to use `FUSED_GATE_UP_PROJ` or `VANILLA`. **Default to `FUSED_GATE_UP_PROJ`** for performance. Only use `VANILLA` if weights cannot be stacked into a single `[num_experts, ...]` tensor.
3. **Weight layout**: Document whether the checkpoint stores gate/up weights in concatenated (`[gate_rows; up_rows]`) or interleaved (`[gate_row0, up_row0, gate_row1, up_row1, ...]`) format. If interleaved, note that `_transform_weights` must de-interleave and re-concatenate before loading.
4. **Custom activation parameters**: Document any SwiGLU `alpha`/`beta`/`limit` values and where they come from (config field or hardcoded in HuggingFace source).
5. **MoE bias**: Whether the expert MLPs use bias terms (`bias=True` in `create_moe`).
6. **Gate/Router mapping**: How router weight/bias names map to the TRT-LLM Gate module (e.g., `mlp.router` → `mlp.gate`).

### Step 5: Generate Weight Loading Plan (Part 2)

Generate weight loading plan for the second part of plan.md:

- List all weight names in the HuggingFace checkpoint (Section 2.2)
- Document potential modeling errors and issues (Section 2.3)
- Focus on common pitfalls such as:
  - QKV fusion requirements for attention modules
  - Gate/Up fusion requirements for MLP modules
  - Weight name mismatches between HuggingFace and TensorRT-LLM
  - Tied embedding handling
  - Normalization layer name differences

**Output format example:**

Section 2.3 should document modeling errors, not comprehensive loading details. For example:
- **Attention Module QKV Fusion**: TensorRT-LLM requires QKV weights to be fused, but HuggingFace has separate q_proj, k_proj, v_proj layers.
- **MLP Gate/Up Fusion**: TensorRT-LLM requires gate and up projections to be fused.

### Step 6: Analyze Quantization (Part 3)

Analyze quantization information for the third part of plan.md, which consists of three sections:

#### 6.1 Read Quantization Configuration

1. **Read checkpoint config files**: Check `config.json` or `quantize_config.json` in the checkpoint directory
2. **Extract quantization parameters**: 
   - `quantization_config.quant_method` (gptq, awq, etc.)
   - `quantization_config.bits`
   - `quantization_config.group_size`
   - Other quantization-specific parameters
3. **Document quantization strategy**: Describe the quantization type and TensorRT-LLM configuration in Section 3.1

**Quantization type detection:**

| Detection Condition | Quantization Type |
|--------------------|-------------------|
| `quantization_config.quant_method == "gptq"` | GPTQ |
| `quantization_config.quant_method == "awq"` | AWQ |
| Weights contain `qweight`, `scales`, `qzeros` | GPTQ/AWQ |
| Weight dtype is int8 | INT8 weight-only |
| Weight dtype is float8 | FP8 |
| None of the above | None (FP16/BF16) |

#### 6.2 Determine Excluded Modules

1. **Identify modules to exclude from quantization**: Typically include:
   - Embedding layers (vocab_embedding)
   - Normalization layers (input_layernorm, post_layernorm, norm)
   - LM head (lm_head)
   - Other sensitive layers that should remain in full precision
2. **Map to TensorRT-LLM module names**: Convert HuggingFace module names to TensorRT-LLM naming conventions
3. **Document in Section 3.2**: List excluded modules using TensorRT-LLM names

**Module name mapping:**
- `model.embed_tokens` → `vocab_embedding`
- `model.norm` → `norm`
- `model.layers.{i}.input_layernorm` → `layers.{i}.input_layernorm`
- `model.layers.{i}.post_attention_layernorm` → `layers.{i}.post_layernorm`
- `lm_head` → `lm_head`

#### 6.3 Document Quantization Weight Loading

1. **Identify quantization-related weights**: Look for additional weights beyond standard model weights:
   - GPTQ/AWQ: `qweight`, `scales`, `qzeros`, `g_idx`
   - INT8: `weight` (int8), `weight_scale`
   - FP8: `weight` (fp8), `weight_scale`
2. **Map weight names**: Map HuggingFace weight names to TensorRT-LLM module names
3. **Document loading procedure**: Describe how to load and fuse quantization weights (especially for QKV and gate/up projections) in Section 3.3

### Step 7: Evaluate Compatibility

Evaluate whether the model can be built with TensorRT-LLM:

1. **Check module compatibility**: Whether each HuggingFace module has a corresponding TensorRT-LLM class
2. **Check quantization compatibility**: Whether the quantization method is supported by TensorRT-LLM

**Supported quantization methods:**
- None (FP16/BF16), GPTQ, AWQ, INT8, FP8, SmoothQuant


### Step 8: Write plan.md to Disk

Compile all analysis results into a structured markdown document and **write it to a file named `plan.md` in the current working directory**. Use a file-write tool (e.g., `write_file`, `save_file`, or equivalent) to persist the content to disk. Do NOT just display the content in the chat — the file must exist on disk after this step completes.

**File path**: `./plan.md` (current working directory)

**File content structure**:

```markdown
# TensorRT-LLM Porting Plan: {ModelName}

## Part 1: Module Mapping
[Module mapping table and initialization parameters]

## Part 2: Weight Loading
[Weight mapping table and loading code]

## Part 3: Quantization
[Quantization analysis results and configuration]
```

**⚠️ Reminder**: The plan.md must contain ONLY the three parts above. Do NOT append a work schedule/timeline section or a TensorRT-LLM modeling code section. See the "Output Constraints" section above.

After writing the file, verify it was created successfully, then output the compatibility conclusion.

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Analysis Workflow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      INPUTS                               │   │
│  │  • checkpoint_path: /path/to/model/                       │   │
│  │  • hf_modeling_path: /path/to/modeling_xxx.py             │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Step 1: Parse Checkpoint & Modeling File         │   │
│  │  • Read config.json                                       │   │
│  │  • Analyze weight files (names, shapes, dtypes)           │   │
│  │  • Parse Python modeling source code                      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Step 2: Generate Module Mapping (Part 1)         │   │
│  │  • Map HF modules to TRT-LLM classes                      │   │
│  │  • Define initialization parameters                       │   │
│  │  • Document fusion opportunities (QKV, gate/up)           │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Step 3: Generate Weight Loading Plan (Part 2)    │   │
│  │  • List all HuggingFace weight names                      │   │
│  │  • Document potential modeling errors and issues          │   │
│  │  • Focus on common pitfalls (QKV fusion, etc.)             │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Step 4: Analyze Quantization (Part 3)            │   │
│  │  • Read quantization config from checkpoint               │   │
│  │  • Determine excluded modules (TRT-LLM names)            │   │
│  │  • Document quantization weight loading procedure        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Step 5: Evaluate Compatibility                   │   │
│  │  • Check all modules against TRT-LLM capabilities         │   │
│  │  • Verify quantization support                            │   │
│  │  • Generate compatibility conclusion                      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      OUTPUTS                              │   │
│  │  • ./plan.md (written to current working directory)       │   │
│  │    - Part 1: Module Mapping                               │   │
│  │    - Part 2: Weight Loading                               │   │
│  │    - Part 3: Quantization                                 │   │
│  │  • Compatibility Conclusion: Supported / Not Supported    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Example Usage

### Input

```yaml
checkpoint_path: /models/Llama-2-7b-hf/
hf_modeling_path: /transformers/src/transformers/models/llama/modeling_llama.py
```

### Command

```python
# Run model analysis
result = analyze_model(
    checkpoint_path="/models/Llama-2-7b-hf/",
    hf_modeling_path="/transformers/src/transformers/models/llama/modeling_llama.py"
)

# Outputs:
# - plan.md (porting plan document)
# - Compatibility conclusion
```

### Expected Output

**File written to disk: `./plan.md`** (in the current working directory)

```markdown
# TensorRT-LLM Porting Plan: LlamaForCausalLM

Generated: 2024-01-15
Model: Llama-2-7b-hf
Source: HuggingFace Transformers

## Part 1: Module Mapping
[... module mapping content ...]

## Part 2: Weight Loading
[... weight loading content ...]

## Part 3: Quantization
Quantization Type: None (FP16)
No quantization detected in checkpoint.
[... quantization details ...]
```

**Compatibility Conclusion:**

```
╔══════════════════════════════════════════════════════════════════╗
║                    TensorRT-LLM COMPATIBILITY                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: LlamaForCausalLM                                         ║
║  Status: ✅ SUPPORTED                                            ║
╠══════════════════════════════════════════════════════════════════╣
║  All modules have TensorRT-LLM equivalents.                      ║
║  Quantization: FP16 (native support)                             ║
║                                                                  ║
║  Output: plan.md written to ./plan.md                            ║
╚══════════════════════════════════════════════════════════════════╝
```

## Available Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `read_checkpoint_config` | Extract config.json from checkpoint | When analyzing checkpoint |
| `list_checkpoint_weights` | List weight tensor names and metadata | For weight mapping analysis |
| `read_hf_modeling_file` | Read modeling Python source | When analyzing module structure |
| `detect_quantization` | Detect quantization from checkpoint | For Part 3 quantization analysis |
