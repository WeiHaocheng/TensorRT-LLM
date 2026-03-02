---
name: trtllm-modeling
description: Orchestrate the full workflow of converting a HuggingFace model to TensorRT-LLM modeling code. Takes a checkpoint path and HuggingFace modeling source as input, generates a porting plan, verifies uncertain modules, and produces a final compatibility assessment.
---

# TensorRT-LLM Modeling Conversion Skill

This skill orchestrates the end-to-end workflow for converting a HuggingFace model into TensorRT-LLM modeling code. It coordinates three subagents — **modeling_analysis** (for initial plan generation), **module_verify** (for uncertain module verification and post-generation consistency checks), and **modeling_coder** (for code generation) — in an iterative loop until every module is resolved as either **implementable** or **not implementable**, and the generated modeling code is verified against the plan.

---

## Inputs

| Input | Description | Example |
|-------|-------------|---------|
| `checkpoint_path` | Path to the HuggingFace model checkpoint directory (containing `config.json`, weight files, etc.) | `/models/llama-7b/` |
| `hf_modeling_path` | Path to the HuggingFace transformers modeling source code (`.py` file or directory) | `/transformers/src/transformers/models/llama/modeling_llama.py` |
| `plan_path` | *(Optional)* Path to an existing `plan.md` from a previous analysis run. If provided, skip plan generation and route based on its content. | `./plan.md` |

---

## Outputs

1. **`plan.md`** — A comprehensive porting plan written to the current working directory, containing:
   - Module mapping (HuggingFace → TensorRT-LLM)
   - Weight loading specifications and pitfalls
   - Quantization analysis
   - **Final compatibility status for every module** (all resolved to ✅ or ❌)

2. **Compatibility Conclusion** — A clear summary indicating whether the full model can be ported to TensorRT-LLM, and if not, which specific modules are blocking.

3. **Modeling Code** *(conditional)* — If the model is fully implementable, a complete `modeling_<model_name>.py` file under `tensorrt_llm/_torch/models/`, along with registration updates to `__init__.py`.

---

## Workflow

0. **Entry Point Routing**: If the user provides a `plan_path` (an existing `plan.md`), read it and check the module compatibility statuses:
   - If **any module is ⚠️ (uncertain)** → go to **step 2** to verify those modules.
   - If **all modules are ✅ (implementable)** → skip directly to **step 4** to generate modeling code.
   - If **any module is ❌ (not implementable)** → report the blocking modules to the user and stop.
   - If `plan_path` is **not provided**, start from **step 1**.

1. **Generate Plan**: Use the **Task tool** to dispatch the `trtllm-modeling-analysis` subagent with `checkpoint_path` and `hf_modeling_path` as inputs → generates `plan.md` (with ✅/⚠️/❌ status per module).

2. **Verify Uncertain Modules**: For each ⚠️ module in `plan.md`, use the **Task tool** to dispatch the `trtllm-module-verify` subagent with the module info and HF source code → returns ✅ or ❌ verdict. Update `plan.md` accordingly.

3. **Repeat** until all modules are resolved to ✅ (implementable) or ❌ (not implementable). Output the final `plan.md` and compatibility conclusion.

4. **Generate Modeling Code**: If the conclusion from step 3 is that the model is **implementable** (all modules resolved to ✅), use the **Task tool** to dispatch the `trtllm-modeling-coder` subagent with the `plan.md` path (and optionally `hf_modeling_path` and `checkpoint_path`) to generate the complete TensorRT-LLM modeling code. If any module is ❌ (not implementable), skip this step and report the blocking modules to the user.

5. **Verify Modeling Code**: After step 4 generates the modeling code, for **each module** in `plan.md`, use the **Task tool** to dispatch the `trtllm-module-verify` subagent in **consistency-check mode** (by providing the `modeling_code_path` parameter pointing to the generated modeling file) to verify that the implementation matches the plan. Collect all inconsistency reports; if any module has critical inconsistencies, fix the modeling code and re-verify until all modules pass.
