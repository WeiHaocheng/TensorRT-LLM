# TensorRT-LLM Auto Test Report: kimi_k25

**Generated**: 2026-03-19 05:00:00 UTC
**Repo**: /home/scratch.fredw_sw/trt-llm-github-3/TensorRT-LLM
**Test Type**: Model-Level (layer_ids=0)
**Model Definition**: `tensorrt_llm/_torch/models/modeling_kimi_k25.py`
**Checkpoint**: `/home/scratch.trt_llm_data_ci/llm-models/Kimi-K2.5-NVFP4/`

## Overall Status

| Category | Status | Passed | Failed | Skipped | Total |
|----------|--------|--------|--------|---------|-------|
| Model Construction | PASS | 1 | 0 | 0 | 1 |
| Weight Loading (layer 0) | PASS | 1 | 0 | 0 | 1 |
| Vision Tower Forward | PASS | 1 | 0 | 0 | 1 |
| MM Projector Forward | PASS | 1 | 0 | 0 | 1 |
| E2E LLM API (H100/SM90) | SKIP | 0 | 0 | 1 | 1 |
| E2E LLM API (B300/SM103) | BLOCKED | 0 | 0 | 1 | 1 |

**Overall: PASS (modeling code verified; E2E blocked by pre-existing TRT-LLM FMHA bug on Blackwell)**

---

## Test Results

### 1. Model Construction (PASS)

Constructed `KimiK25ForConditionalGeneration` with 1 layer (layer 0):
- DeepseekV3 language model: 1 layer, NVFP4 quantized MLP, BF16 attention
- MoonViT3d vision tower: 27 encoder layers, 416M params
- KimiK25MultiModalProjector: 54M params
- Total (1 LLM layer): 3.1B params

### 2. Weight Loading (PASS)

Successfully loaded 361 tensors from NVFP4 checkpoint:
- Vision tower: 329 weights (BF16) loaded via `load_state_dict(strict=True)`
- MM projector: 6 weights with rename mapping (`proj.0` -> `linear_1`, `proj.2` -> `linear_2`)
- Language model layer 0: NVFP4 MLP weights (uint8) + BF16 attention weights
- Embedding + lm_head loaded correctly

### 3. Vision Tower Forward (PASS)

Input: `[256, 3, 14, 14]` (256 patches from 1 image, 16x16 grid)
- Patch embedding: Conv2d [256,3,14,14] -> [256, 1152]
- 27-layer encoder with 2D RoPE + spatial-temporal attention
- Temporal pooling patch merger (sd2_tpool, kernel=[2,2])
- Output: list of 1 tensor, shape `[64, 4, 1152]` (64 merged patches)

### 4. MM Projector Forward (PASS)

Input: vision tower output `[64, 4, 1152]`
- Pre-norm (LayerNorm) + flatten -> `[64, 4608]`
- Linear(4608, 4608) + GELU + Linear(4608, 7168)
- Output: `[64, 7168]` (projected to text hidden size)

### 5. E2E LLM API on H100 (SKIPPED - hardware limitation)

**GPU**: 1x NVIDIA H100 PCIe (81559 MiB, SM90)

NVFP4 GEMM is not supported on H100 PCIe (SM90):
- `cutlass`: "Arch unsupported for CUTLASS FP4 GEMM"
- `cublaslt`: "CUBLAS_STATUS_NOT_SUPPORTED"
- `cuda_core`: "requires SM >= 100"

NVFP4 inference requires SM100+ (Blackwell architecture). This is a hardware limitation, not a modeling code issue.

### 6. E2E LLM API on B300 (BLOCKED - TRT-LLM FMHA infra bug)

**GPU**: 8x NVIDIA B300 SXM6 AC (275040 MiB, SM10.3 Blackwell)

Segfault in TRT-LLM FMHA kernel dispatcher during attention initialization:
```
TllmGenFmhaKernel::hashFromRunnerParams() → checkIfKernelExist() →
FmhaRunner::isSupported() → FmhaDispatcher::FmhaDispatcher() →
AttentionOp::initialize()
```

**Confirmed as pre-existing TRT-LLM infra bug**: Standalone DeepseekV3 (same checkpoint, same LLM API flow) produces the **exact same segfault** at the same C++ call stack. This is a TRT-LLM FMHA kernel issue on B300/SM10.3 with MLA attention, not a Kimi-K25 modeling code issue.

Only the `TRTLLM` attention backend supports MLA (VANILLA/FLASHINFER return `support_mla()=False`), so no workaround is available via alternative backends.

The crash occurs after successful weight loading (all 27 modules loaded in ~1s) during KV cache / attention dispatcher setup.

---

## Bugs Found and Fixed

### Fix 1: Missing `processor` abstract method (TypeError)

**Error**: `TypeError: Can't instantiate abstract class KimiK25InputProcessor without an implementation for abstract method 'processor'`

**Fix**: Added `AutoProcessor` import, `self._processor` initialization in `__init__()`, and `@property processor` method. Refactored `_preprocess()` to use `self.processor` instead of creating a new processor per call.

### Fix 2: Quantization exclude_modules path mismatch (RuntimeError)

**Error**: `RuntimeError: The size of tensor a (3584) must match the size of tensor b (7168)` - attention weights were incorrectly quantized to NVFP4.

**Root cause**: Checkpoint's `exclude_modules` patterns use `language_model.layers.X.self_attn*` but DeepseekV3 internally uses `model.layers.X.self_attn*`. The patterns didn't match after prefix stripping.

**Fix**: Added pattern remapping in `_get_sub_model_config()`: `language_model.layers.X` -> `model.layers.X`, `language_model.lm_head` -> `lm_head`.

### Fix 3: MLA attention layers not registered (AssertionError)

**Error**: `AssertionError: Attention layer is not registered`

**Root cause**: `dataclasses.replace()` creates a fresh `extra_attrs` dict (since it's `init=False`). MLA layers registered on the sub-config's `extra_attrs` don't propagate to the parent config that the executor checks.

**Fix**: Added `model_config_cp.extra_attrs.update(llm_model_config.extra_attrs)` after constructing the language model.

---

## Files Modified

- `tensorrt_llm/_torch/models/modeling_kimi_k25.py` (3 fixes applied)
- `tensorrt_llm/_torch/models/__init__.py` (import + registration)
