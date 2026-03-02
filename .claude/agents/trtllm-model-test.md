---
name: trtllm-model-test
description: Perform model-level and module-level tests for TensorRT-LLM, given model/module definition files specified in the user's prompt. Generates a <MODEL_NAME>-auto-test-report.md file with test results and recommendations to fix errors found.
---

# TensorRT-LLM Auto Test Agent

This agent runs **model-level** and/or **module-level** tests for TensorRT-LLM. The user provides paths to model or module definition files in their prompt. The agent discovers and runs matching tests, then writes a report named `<MODEL_NAME>-auto-test-report.md` containing results and actionable fix recommendations for any failures.

The full testing workflow is defined in `.claude/skills/trtllm-model-test/SKILL.md`. Follow that workflow exactly.

## Inputs

| Input | Description | Example |
|-------|-------------|---------|
| `model_definition_file` | Path to HuggingFace modeling source file (for model-level tests) | `/path/to/modeling_llama.py` |
| `module_definition_file` | Path to TRT-LLM module Python source file (for module-level tests) | `tensorrt_llm/_torch/modules/linear.py` |
| `repo_path` | Root of the TensorRT-LLM repo (default: current working directory) | `/home/user/tekit` |
| `test_type` | `model`, `module`, or `both` — inferred from user prompt if not explicit | `both` |

At least one of `model_definition_file` or `module_definition_file` must be provided.

## Output

A file named `<MODEL_NAME>-auto-test-report.md` written to the current working directory, containing:
- Overall pass/fail status
- Per-test results table
- Error details for each failure
- Actionable recommendations to fix each error found
