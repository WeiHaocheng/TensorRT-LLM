---
name: trtllm-model-test
description: Perform model-level tests for TensorRT-LLM, given model definition files specified in the user's prompt. Generates a <MODEL_NAME>-auto-test-report.md file with test results and recommendations to fix errors found.
---

# TensorRT-LLM Auto Test Agent

This agent runs **model-level** tests for TensorRT-LLM. The user provides paths to model definition files in their prompt. The agent discovers and runs matching tests, then writes a report named `<MODEL_NAME>-auto-test-report.md` containing results and actionable fix recommendations for any failures.

The final output is a file named `<MODEL_NAME>-auto-test-report.md` written to the current working directory.

## Inputs

| Input | Description | Example |
|-------|-------------|---------|
| `model_definition_file` | Path to HuggingFace modeling source file (for model-level tests) | `/path/to/modeling_llama.py` |
| `repo_path` | Root of the TensorRT-LLM repo (default: current working directory) | `/home/user/tekit` |
| `plan_file` | Path to plan.md that documents the modeling approach/design decisions | `plan.md` |

`model_definition_file` must be provided.

## Output

Return **all** of the following to the calling agent, and **also** write them to a file named `<MODEL_NAME>-auto-test-report.md` in the current working directory:

- Overall pass/fail status
- Per-test results table
- Error details for each failure
- Actionable recommendations to fix each error found

---

## Workflow

### Step 1: Parse User Input

Extract from the user's prompt:

1. **Definition file path**: Extract the HuggingFace modeling file path (`modeling_*.py`) or HF checkpoint directory → `model_definition_file`.

   If the test scope is **Benchmark** or **Feature Support Matrix**, immediately stop and respond:
   > "This test type is not yet supported. Only **model-level** Functional tests are currently available."

2. **MODEL_NAME**: Derive from the provided file path:
   - `modeling_llama.py` → `llama`
   - Checkpoint directory `/models/Llama-2-7b-hf/` → `Llama-2-7b-hf`

3. **Repo path**: Use the path explicitly stated in the prompt, or default to the current working directory.

---

### Step 2: Discover Relevant Tests

#### Model-Level Test Discovery

`pytest examples/llm-api/quickstart_advanced.py --model_dir <MODEL_PATH> --prompt 'Hello, how are you?' --tp_size 1 -v`. Replace `<MODEL_PATH>` and the model name in test command with the provided model path directly. **Don't refer references/trtllm_model_test_cases.md**

---

### Step 3: Run Tests

Run model-level tests.

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
python3 .claude/skills/trtllm-modeling/scripts/generate_report.py \
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
Reference `trtllm-modeling/references/trtllm_test_fix_recommendations.md` for common error patterns and fixes, and customize the recommendation based on the specific error details and root cause identified.

- <Concrete action 1>
- <Concrete action 2 if needed>

**Files to modify**: `<file_path>:<line_number>`
```

---

### Step 6: Write Test Report

Write a file named `<MODEL_NAME>-auto-test-report.md` to the current working directory:

```markdown
# TensorRT-LLM Auto Test Report: <MODEL_NAME>

**Generated**: <timestamp>
**Repo**: <repo_path>
**Test Type**: Model-Level

## Overall Status

| Category | Status | Passed | Failed | Skipped | Total | Duration |
|----------|--------|--------|--------|---------|-------|----------|
| Model-Level Tests | ✅ PASS / ❌ FAIL | N | N | N | N | Ns |

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
```

After writing the file:

1. **Return the full report content** to the calling agent so it can act on the results without reading the file.
2. Confirm to the user:
```
Test report written to: ./<MODEL_NAME>-auto-test-report.md
```

---

## Resources

- **`.claude/skills/trtllm-modeling/references/trtllm_model_test_cases.md`** — Known integration test commands indexed by model name. Read before Step 2 to select integration tests.
- **`.claude/skills/trtllm-modeling/scripts/generate_report.py`** — Report generation script called in Step 4 to parse pytest output into structured data.

## Notes

- Tests requiring multiple GPUs (TP > 1) are skipped when only 1 GPU is available — this is expected and not a failure.
- If no test files are found for a given model, report `No tests found` and suggest checking naming conventions (e.g., `test_modeling_<model_type>.py`).
- The skill reads definition files only for root-cause analysis — it does not modify source files. All recommendations are advisory.
