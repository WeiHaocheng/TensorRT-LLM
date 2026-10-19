# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allreduce op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the *fusion patterns* through the
``AllReduce`` module, while this covers the one op call the catalog entry
makes, cell by cell -- the bits every rank receives, their independence from
the rows beside them, and how the sum is formed. The names are kept apart so
review does not read one as a copy of the other.

The matrix is ``_allreduce_op_matrix.py`` beside this file, which is its own
launcher: it runs its ranks as an MPI world under ``mpirun``, the way a
tensor-parallel serving engine's workers run.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "allreduce requires CUDA devices"


# The case starts its own 4 rank processes over every visible device. Under
# xdist several workers would fight for the same GPUs, so this must run
# alone -- the same reason the other collective entries' matrices carry the
# marker.
@pytest.mark.no_xdist
def test_allreduce_op_matrix() -> None:
    _rank_job.run("allreduce")
