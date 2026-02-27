---
name: trtllm-modeling
description: Orchestrate the full workflow of converting a HuggingFace model to TensorRT-LLM modeling code. Takes a checkpoint path and HuggingFace modeling source as input, generates a porting plan, verifies uncertain modules, and produces a final compatibility assessment.
---

# TensorRT-LLM Modeling Conversion Skill

This skill orchestrates the end-to-end workflow for converting a HuggingFace model into TensorRT-LLM modeling code. It coordinates two subagents — **modeling_plan** (for initial analysis) and **module_verify** (for uncertain module verification) — in an iterative loop until every module is resolved as either **implementable** or **not implementable**.

---

## Inputs

| Input | Description | Example |
|-------|-------------|---------|
| `checkpoint_path` | Path to the HuggingFace model checkpoint directory (containing `config.json`, weight files, etc.) | `/models/llama-7b/` |
| `hf_modeling_path` | Path to the HuggingFace transformers modeling source code (`.py` file or directory) | `/transformers/src/transformers/models/llama/modeling_llama.py` |

---

## Outputs

1. **`plan.md`** — A comprehensive porting plan written to the current working directory, containing:
   - Module mapping (HuggingFace → TensorRT-LLM)
   - Weight loading specifications and pitfalls
   - Quantization analysis
   - **Final compatibility status for every module** (all resolved to ✅ or ❌)

2. **Compatibility Conclusion** — A clear summary indicating whether the full model can be ported to TensorRT-LLM, and if not, which specific modules are blocking.

---

## Workflow

1. **Generate Plan**: Call subagent `.claude/agents/trtllm-modeling-analysis.md` with `checkpoint_path` and `hf_modeling_path` → generates `plan.md` (with ✅/⚠️/❌ status per module).

2. **Verify Uncertain Modules**: For each ⚠️ module in `plan.md`, call subagent `.claude/agents/trtllm-module-verify.md` with the module info and HF source code → returns ✅ or ❌ verdict. Update `plan.md` accordingly.

3. **Repeat** until all modules are resolved to ✅ (implementable) or ❌ (not implementable). Output the final `plan.md` and compatibility conclusion.
