# Claude Code Configuration for TensorRT-LLM Modeling

This directory contains the Claude Code agents and skills used for converting HuggingFace models to TensorRT-LLM modeling code.

## Directory Structure

```
.claude/
├── README.md                          # This file
├── settings.local.json                # Claude Code permission settings
├── agents/                            # Subagent definitions
│   ├── trtllm-model-analysis.md       # Plan generation agent
│   ├── trtllm-model-coder.md          # Code generation agent
│   ├── trtllm-model-test.md           # Model-level test agent
│   └── trtllm-module-verify.md        # Module verification agent
└── skills/
    └── trtllm-modeling/               # Orchestrator skill
        ├── SKILL.md                   # Skill definition (workflow orchestration)
        ├── references/                # Reference docs used by agents
        │   ├── moe_module_notes.md    # MoE module implementation notes
        │   ├── trtllm_model_test_cases.md      # Test case templates
        │   └── trtllm_test_fix_recommendations.md  # Common test fix patterns
        └── scripts/
            └── generate_report.py     # Test report generation script
```

## Skill: trtllm-modeling (Orchestrator)

**Trigger**: `/trtllm-modeling` slash command or direct invocation.

The orchestrator skill coordinates the full end-to-end workflow for porting a HuggingFace model to TensorRT-LLM. It dispatches the four subagents below in sequence, with iterative loops for verification and debugging.

### Workflow

```
Step 1: trtllm-model-analysis  →  plan.md
Step 2: trtllm-module-verify   →  resolve uncertain modules in plan.md
Step 3: trtllm-model-coder     →  modeling_<model>.py
Step 4: trtllm-module-verify   →  consistency checks + module-level tests
Step 5: trtllm-model-test      →  end-to-end model tests
```

### Inputs

| Input | Required | Description |
|-------|----------|-------------|
| `checkpoint_path` | Yes | Path to HuggingFace checkpoint directory (with `config.json` and weight files) |
| `hf_modeling_path` | Yes | Path to HuggingFace `modeling_*.py` source file |
| `plan_path` | No | Path to existing `plan.md` to skip plan generation |

### Outputs

| Output | Path | Description |
|--------|------|-------------|
| `plan.md` | `./plan.md` | Porting plan with module mapping, weight loading, quantization |
| Modeling code | `tensorrt_llm/_torch/models/modeling_<model>.py` | TRT-LLM model implementation |
| Test report | `./<MODEL>-auto-test-report.md` | End-to-end test results |
| Module tests | `tests/<model>_auto_generated_tests/` | Auto-generated per-module test files |

---

## Agents

### trtllm-model-analysis

Analyzes a HuggingFace model and generates a structured porting plan (`plan.md`).

**Dispatched by**: Orchestrator (Step 1)

**Inputs**: `checkpoint_path`, `hf_modeling_path`
**Outputs**: `plan.md` in working directory

**Dependencies**:
- Reads HuggingFace model source code and `config.json`
- Reads TRT-LLM module source code under `tensorrt_llm/_torch/modules/`
- Reads `tensorrt_llm/_torch/modules/modules_list.md` for available TRT-LLM module classes
- Reads existing TRT-LLM model implementations under `tensorrt_llm/_torch/models/` as reference patterns

### trtllm-module-verify

Verifies whether a specific HuggingFace module can be implemented with TRT-LLM modules. Operates in two modes:

1. **Verification mode** (default): Determines if an uncertain (⚠️) module is implementable
2. **Consistency-check mode** (when `modeling_code_path` is provided): Compares generated code against `plan.md` and runs module-level tests

**Dispatched by**: Orchestrator (Steps 2 and 4)

**Inputs**: `hf_module_path`, `hf_class_name`, `proposed_trtllm_class`, `uncertainty_reason`, `hf_source_code`, `config_fields`, `checkpoint_path`, and optionally `modeling_code_path`
**Outputs**: Verdict (✅/❌), consistency report, module-level test results

**Dependencies**:
- Reads `tensorrt_llm/_torch/modules/modules_list.md` for module entry points
- Reads TRT-LLM module source code (e.g., `attention.py`, `rms_norm.py`, `fused_moe/`)
- Reads `plan.md` for expected implementation details
- Reads HuggingFace model source code
- Loads checkpoint weights for numeric comparison tests
- Creates test files under `tests/<model>_auto_generated_tests/`
- **Requires CUDA GPU** for running module-level comparison tests

### trtllm-model-coder

Generates the complete TRT-LLM modeling file based on `plan.md`.

**Dispatched by**: Orchestrator (Step 3)

**Inputs**: `plan_path`, `hf_modeling_path`, `checkpoint_path`
**Outputs**: `tensorrt_llm/_torch/models/modeling_<model>.py`, updates to `__init__.py`

**Dependencies**:
- Reads `plan.md` for module mapping, weight loading, and quantization details
- Reads existing TRT-LLM model implementations as reference (e.g., `modeling_llama.py`, `modeling_qwen.py`, `modeling_deepseek_v3.py`)
- Reads TRT-LLM module source code for constructor signatures
- Reads `tensorrt_llm/_torch/models/__init__.py` for registration patterns
- Reads `.claude/skills/trtllm-modeling/references/moe_module_notes.md` for MoE implementation patterns

### trtllm-model-test

Runs end-to-end model-level tests (instantiation, weight loading, inference).

**Dispatched by**: Orchestrator (Step 5)

**Inputs**: `model_definition_file`, `hf_modeling_path`, `plan_file`
**Outputs**: `<MODEL>-auto-test-report.md` in working directory

**Dependencies**:
- Reads generated TRT-LLM modeling code
- Reads `plan.md` and HuggingFace modeling code
- Reads `.claude/skills/trtllm-modeling/references/trtllm_model_test_cases.md` for test templates
- Reads `.claude/skills/trtllm-modeling/references/trtllm_test_fix_recommendations.md` for debugging patterns
- Runs `.claude/skills/trtllm-modeling/scripts/generate_report.py` for report generation
- Loads full model checkpoint for inference tests
- **Requires CUDA GPU** with sufficient memory to load the model
- **Requires TensorRT-LLM runtime** to be installed and functional

---

## Runtime Requirements

### Software

| Dependency | Required By | Notes |
|------------|------------|-------|
| TensorRT-LLM | All agents | Must be installed in the Python environment (`tensorrt_llm` package) |
| PyTorch | All agents | With CUDA support |
| HuggingFace Transformers | Analysis, Verify, Test agents | For loading HF model configs and weights |
| safetensors | Verify, Test agents | For loading checkpoint weight files |
| pytest | Verify, Test agents | For running auto-generated tests |

### Hardware

| Resource | Required By | Notes |
|----------|------------|-------|
| CUDA GPU | Verify, Test agents | Module-level and model-level tests require GPU execution |
| GPU Memory | Test agent | Must be sufficient to load the full model (e.g., ~14 GB for gpt-oss-20b with MXFP4) |

### Repository Files

The agents depend on these files existing in the TensorRT-LLM repository:

| File | Required By | Description |
|------|------------|-------------|
| `tensorrt_llm/_torch/modules/modules_list.md` | Analysis, Verify | Index of available TRT-LLM module classes |
| `tensorrt_llm/_torch/modules/*.py` | All agents | TRT-LLM module source code |
| `tensorrt_llm/_torch/models/__init__.py` | Coder | Model registration file |
| `tensorrt_llm/_torch/models/modeling_*.py` | Coder | Existing model implementations as reference |

---

## Usage

### Full workflow (from scratch)

```
/trtllm-modeling checkpoint_path="/path/to/checkpoint/" hf_modeling_path="/path/to/modeling_model.py"
```

### Resume from existing plan

```
/trtllm-modeling checkpoint_path="/path/to/checkpoint/" hf_modeling_path="/path/to/modeling_model.py" plan_path="./plan.md"
```

### Skip to debugging phase

```
/trtllm-modeling enter the debugging phase
```
