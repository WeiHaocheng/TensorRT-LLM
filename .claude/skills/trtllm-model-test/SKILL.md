---
name: trtllm-model-test
description: >-
  Multi-step workflow for running model-level and module-level tests in TensorRT-LLM,
  given model/module definition files specified in the user's prompt. Triggers on test model or test modules.
---

# TRTLLM Model Test Skill

## Overview

Guides through a 6-step workflow: parse input → discover tests → run tests → parse results → generate recommendations → write report.

Supports two test scopes that can be combined:
- **Model-level** — exercise a full model end-to-end
- **Module-level** — exercise individual modules' implementations or debugging

The final output is a file named `<MODEL_NAME>-auto-test-report.md` written to the current working directory.

---

## Workflow

### Step 1: Parse User Input

Extract from the user's prompt:

1. **Test scope**: Look for keywords to determine which levels to run:
   - `"model"`, `"model-level"`, `"integration"` → model-level only
   - `"module"`, `"module-level"`, `"unit"` → module-level only
   - Both keywords present, or neither (ambiguous) → run both scopes
   - If only one definition file is provided, infer scope from its path (see Step 1.2)

2. **Definition file paths**: Extract any file paths mentioned and classify each:
   - A HuggingFace modeling file (`modeling_*.py`) or HF checkpoint directory → `model_definition_file`; scope includes model-level
   - A TRT-LLM module file (path under `tensorrt_llm/_torch/modules/`) → `module_definition_file`; scope includes module-level

   If the test scope is **Benchmark** or **Feature Support Matrix**, immediately stop and respond:
   > "This test type is not yet supported. Only **model-level** and **module-level** Functional tests are currently available."

3. **MODEL_NAME**: Derive from the provided file path(s):
   - `modeling_llama.py` → `llama`
   - `tensorrt_llm/_torch/modules/linear.py` → `linear`
   - Checkpoint directory `/models/Llama-2-7b-hf/` → `Llama-2-7b-hf`
   - When both files are provided, prefer the model name over the module name.

4. **Repo path**: Use the path explicitly stated in the prompt, or default to the current working directory.

---

### Step 2: Discover Relevant Tests

#### 2a. Model-Level Test Discovery

`pytest examples/llm-api/quickstart_advanced.py --model_dir <MODEL_PATH> --prompt 'Hello, how are you?' --tp_size 1 -v`. Replace `<MODEL_PATH>` and the model name in test command with the provided model path directly. **Don't refer references/trtllm_model_test_cases.md**

#### 2b. Module-Level Test Discovery

Given `module_definition_file` or 'model_definition_file' containing module imports, perform:

1. **Extract module name and class names** by reading the file (look for `class` definitions).

2. **Create Test File** under `tests/`:
   - Review given model or module definition file and HuggingFace model definition file to identify MoE modules, attention modules, feedforward modules, etc.
   - Based on the identified module types, create a new test file in the appropriate location under 'tests'.
   - In test file, write one test case for each identified module, by comparing the result from HuggingFace module and the corresponding TRT-LLM modules.

3. **Build test command list** from created test file.

---

### Step 3: Run Tests

Run model-level and module-level test sets **separately** so results appear in distinct report sections.

For each test set, execute:

```bash
  <command> <parameters> 2>&1 | tee /tmp/test_output_$(date +%s).txt
echo "EXIT_CODE:$?"
```

- Record the captured output file path and exit code for each run.
- If a test requires multiple GPUs (TP > 1) and only 1 GPU is available, it will be automatically skipped — treat as informational, not a failure.
- Cap per-file execution: if a single file contains more than 10 test functions, run only the first 10 unless the user explicitly asks for all.

---

### Step 4: Parse and Analyze Results

For each captured output file, call the report generation script to extract structured data:

```bash
python3 .claude/skills/trtllm-model-test/scripts/generate_report.py \
  --output-file <captured_output_file> \
  --exit-code <exit_code> \
  --format markdown
```

Then manually parse the output to gather the following for each failure:

1. **Summary counts**: passed, failed, error, skipped, total, duration
2. **Failed test IDs**: full pytest node IDs
3. **Error details** per failure:
   - Exception type and message
   - Traceback snippet (last 15 lines)
   - Test function docstring if present
4. **Root-cause classification** — classify each failure using the table in the Resources section below:
   - Classify as: `ImportError`, `AssertionError`, `shape mismatch`, `CUDA error`, `weight not found`, `timeout`, `OOM`, or `other`
   - Cross-reference the traceback against the definition file and related TRT-LLM source files to identify the likely cause

---

### Step 5: Generate Fix Recommendations

For each failed test, produce a recommendation block:

```markdown
#### Fix for `<test_id>`

**Error type**: <ImportError | AssertionError | shape mismatch | ...>

**Root cause**: <1-2 sentence explanation>

**Recommended fix**:
Reference `references/trtllm_test_failure_recommendations.md` for common error patterns and fixes, and customize the recommendation based on the specific error details and root cause identified.

- <Concrete action 1>
- <Concrete action 2 if needed>

**Files to modify**: `<file_path>:<line_number>`

---

### Step 6: Write Test Report

Write a file named `<MODEL_NAME>-auto-test-report.md` to the current working directory:

```markdown
# TensorRT-LLM Auto Test Report: <MODEL_NAME>

**Generated**: <timestamp>
**Repo**: <repo_path>
**Test Type**: <Model-Level | Module-Level | Both>

## Overall Status

| Category | Status | Passed | Failed | Skipped | Total | Duration |
|----------|--------|--------|--------|---------|-------|----------|
| Model-Level Tests | ✅ PASS / ❌ FAIL | N | N | N | N | Ns |
| Module-Level Tests | ✅ PASS / ❌ FAIL | N | N | N | N | Ns |

**Overall: ✅ ALL PASSED** or **❌ FAILURES DETECTED**

---

## Model-Level Test Results

### Test Files Run
- `<test_file_path>`

### Results Table

| Test ID | Status | Duration |
|---------|--------|----------|
| `tests/unittest/_torch/modeling/test_modeling_llama.py::TestLlama::test_forward` | ✅ PASSED | 1.2s |
| `tests/integration/defs/examples/test_llm_api_with_mpi.py::...` | ❌ FAILED | 3.4s |

### Failure Details

#### `<failed_test_id>`

**Error**:
```
<traceback snippet>
```

---

## Module-Level Test Results

### Test Files Run
- `<test_file_path>`

### Results Table

| Test ID | Status | Duration |
|---------|--------|----------|
| `tests/unittest/_torch/modules/test_linear.py::TestLinear::test_forward` | ✅ PASSED | 0.5s |

### Failure Details

*(none)*

---

## Recommendations

> The following fixes are recommended for the **N failed tests** found above.

<Per-failure recommendation blocks from Step 5, ordered by severity>

---

## Raw Output

<details>
<summary>Model-Level Test Raw Output</summary>

```
<captured stdout/stderr>
```

</details>

<details>
<summary>Module-Level Test Raw Output</summary>

```
<captured stdout/stderr>
```

</details>
```

After writing the file, confirm to the user:
```
Test report written to: ./<MODEL_NAME>-auto-test-report.md
```

---

## Resources

- **`references/trtllm_model_test_cases.md`** — Known integration test commands indexed by model name. Read before Step 2a to select integration tests.
- **`scripts/generate_report.py`** — Report generation script called in Step 4 to parse pytest output into structured data.

## Notes

- Tests requiring multiple GPUs (TP > 1) are skipped when only 1 GPU is available — this is expected and not a failure.
- If no test files are found for a given model/module, report `No tests found` and suggest checking naming conventions (e.g., `test_modeling_<model_type>.py` or `test_<module_name>.py`).
- The skill reads definition files only for root-cause analysis — it does not modify source files. All recommendations are advisory.
