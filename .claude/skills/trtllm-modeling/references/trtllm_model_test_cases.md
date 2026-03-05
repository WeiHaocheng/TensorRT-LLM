# Test Cases — Model-Parameterized Tests

**Path note:** After stripping the CI suite prefix (e.g. `A10-PyTorch-1/`, `B200_PCIe-PackageSanityCheck-PY312-DLFW/`), files were resolved to `tests/integration/defs/<path>` relative to project root.

## Test Cases by test type

### Functional Test

| Test Command | Model Parameter | GPU Count |
|---|---|---|
| `pytest tests/integration/defs/examples/test_llm_api_with_mpi.py::test_llm_api_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_conditional[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_cuda_graph[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_mixed[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_ngram[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_overlap[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun_trt_backend[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py::test_disaggregated_simple_llama[False-True-TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py::test_disaggregated_simple_llama[True-False-TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/disaggregated/test_workers.py::test_workers_kv_cache_events[TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
| `pytest tests/integration/defs/test_e2e.py::test_trtllm_bench_invalid_token_pytorch[TinyLlama-1.1B-Chat-v1.0-TinyLlama-1.1B-Chat-v1.0] -v` | TinyLlama-1.1B-Chat-v1.0 | 1 |
