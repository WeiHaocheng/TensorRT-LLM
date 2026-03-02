---
name: trtllm-module-verify
description: Verify whether a specific HuggingFace module can be implemented using TensorRT-LLM's existing modules. Reads TRT-LLM module SKILL files and source code to determine compatibility, writes and runs simple verification scripts when needed, and returns a clear verdict with implementation details or blocking reasons.
---

# TensorRT-LLM Module Verification Agent

This agent verifies whether a specific HuggingFace module (marked as ⚠️ uncertain in `plan.md`) can be implemented using TensorRT-LLM's existing modules and features. It reads TRT-LLM source code, SKILL files, and existing model implementations. Module code entry points (class names, descriptions, and source file paths) can be found in `tensorrt_llm/_torch/modules/modules_list.md`. When static analysis alone is insufficient, it can write and run simple verification scripts to confirm compatibility (e.g., instantiating a TRT-LLM module with specific parameters to check if it accepts them, or testing weight shape transformations).

## Inputs

| Input | Description |
|-------|-------------|
| `hf_module_path` | HuggingFace module path (e.g., `model.layers[i].self_attn`) |
| `hf_class_name` | HuggingFace class name (e.g., `GptOssAttention`) |
| `proposed_trtllm_class` | Proposed TensorRT-LLM class (e.g., `Attention`) |
| `uncertainty_reason` | Why this module was marked ⚠️ (e.g., "attention sinks not verified") |
| `hf_source_code` | Relevant HuggingFace source code snippet (class definition, forward method) |
| `config_fields` | Relevant fields from `config.json` that affect this module |
| `checkpoint_path` | Path to the HuggingFace checkpoint directory (for inspecting weight names, shapes, and formats) |
| `modeling_code_path` | *(Optional)* Path to the TRT-LLM modeling code file (e.g., `tensorrt_llm/_torch/models/modeling_gpt_oss.py`). When provided, the agent switches to **consistency-check mode**: it compares the module's actual implementation in the modeling code against the description in `plan.md`, and reports any inconsistencies. |

### Behavior when `modeling_code_path` is provided

When `modeling_code_path` is given, the agent operates in **consistency-check mode** instead of the default verification mode:

1. Read the modeling code at the given path and locate the implementation of the specified module (identified by `hf_module_path` / `hf_class_name`).
2. Read the corresponding section in `plan.md` that describes how this module should be implemented (TRT-LLM class, constructor parameters, weight mapping, special handling, etc.).
3. Compare the actual implementation against the plan and identify any discrepancies, including but not limited to:
   - Different TRT-LLM class used than what the plan specifies
   - Missing or extra constructor parameters
   - Weight mapping mismatches (names, shapes, transformations)
   - Missing or incorrect special handling (e.g., normalization, activation functions)
   - Structural differences (e.g., layer ordering, skip connections)
4. Return the list of inconsistencies (see **Consistency-Check Output** below). If fully consistent, return a confirmation.

## Output

### Default mode (no `modeling_code_path`)

A verification result containing:

| Field | Description |
|-------|-------------|
| `verdict` | ✅ **implementable** or ❌ **not implementable** |
| `trtllm_class` | Confirmed TensorRT-LLM class to use |
| `approach` | How to implement (if ✅) — constructor parameters, weight mapping, special handling |
| `blocking_reason` | What is missing (if ❌) — specific feature or capability gap |
| `required_modifications` | Whether TRT-LLM source code changes are needed (none / minor / major) |
| `confidence` | High / Medium / Low — based on how thoroughly the verification was done |
| `evidence` | References to TRT-LLM code or documentation that support the verdict |

### Consistency-check mode (when `modeling_code_path` is provided)

A consistency report containing:

| Field | Description |
|-------|-------------|
| `consistent` | ✅ **yes** or ❌ **no** |
| `inconsistencies` | List of discrepancies found between the modeling code and `plan.md`. Each entry includes: the aspect (e.g., class choice, parameter, weight mapping), what the plan says, what the code actually does, and the severity (critical / minor). Empty if fully consistent. |
| `summary` | Brief natural-language summary of the comparison result |
