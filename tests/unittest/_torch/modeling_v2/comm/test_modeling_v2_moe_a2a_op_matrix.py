# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the moe_a2a_dispatch / moe_a2a_combine certification matrix.

Distinct from ``tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py``: that file covers TensorRT-LLM's
``NVLinkOneSided`` strategy object; this covers the two op calls the catalog entries make, cell by cell, on
nemotron4's dep4 shapes and payloads, fp32 and bf16 partials.

The matrix is ``_moe_a2a_op_matrix.py`` beside this file, which is its own launcher: it runs its ranks as an
MPI world under ``mpirun``, the way a dep4 serving engine's workers run.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "moe_a2a requires CUDA devices"


# The case starts its own 4 rank processes over every visible device; under xdist several workers would fight
# for the same GPUs, so this must run alone, as the other collective entries' matrices do.
@pytest.mark.no_xdist
def test_moe_a2a_op_matrix() -> None:
    _rank_job.run("moe_a2a")
