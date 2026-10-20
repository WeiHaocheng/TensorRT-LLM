# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the nemotron4 tp4 target's allreduce_wide_residual certification matrix.

The matrix is ``_allreduce_wide_residual_op_matrix.py`` beside this file, which is its own
launcher: it runs its ranks as an MPI world under ``mpirun``, the way a tensor-parallel serving
engine's workers run, and says what it certifies. It is started by the shared
``comm/_rank_job.py``, which owns nothing but the devices and the exit code.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "allreduce_wide_residual requires CUDA devices"


# The case starts its own 4 rank processes over every visible device; under xdist several workers
# would fight for the same GPUs, so it runs alone, as the other collective matrices do. A one-GPU
# stage collects this tree too (l0_b300), and skips it there; l0_gb300_multi_gpus runs it.
@pytest.mark.no_xdist
def test_allreduce_wide_residual_op_matrix() -> None:
    visible = torch.cuda.device_count()
    if visible < _rank_job.WORLD_SIZE:
        pytest.skip(f"tp4 runs {_rank_job.WORLD_SIZE} ranks; {visible} device(s) visible")
    _rank_job.run(
        "allreduce_wide_residual",
        Path(__file__).resolve().with_name("_allreduce_wide_residual_op_matrix.py"),
    )
