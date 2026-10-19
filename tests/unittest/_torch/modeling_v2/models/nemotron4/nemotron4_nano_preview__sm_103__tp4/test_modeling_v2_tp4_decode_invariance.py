# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one 4-GPU node): the nemotron4 tp4 decode step is batch-invariant on every rank, on
real weights and real cache state.

Collected entry point of ``_tp4_decode_invariance_ranks.py`` beside it, which says what runs and
how. That job runs its four ranks as an MPI world under ``mpirun``, the way a tensor-parallel
engine's workers run, so it is started in a fresh interpreter: the launcher must not import
``tensorrt_llm``, which calls ``MPI_Init``, and an MPI-initialized process cannot start
``mpirun``. The job runs once for the module; each test below reads its own section of the job's
report, every rank's findings in it.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

__extra_import_path__ = [".."]
import _nemotron4_reference as R  # noqa: E402 -- needs the path declared above

_JOB = Path(__file__).resolve().with_name("_tp4_decode_invariance_ranks.py")
#: Headroom over the job's own deadline, so its message wins the race.
_GRACE_S = 300
#: Per device: ~47 GiB of the rank's weights, ~15 GiB of state pools and their snapshot.
_FREE_GIB = 100


def _job_constants() -> dict:
    """The job's module-level constants, read off its source. Executing it here would define its
    dataclasses in a module no import system knows, which `dataclasses` refuses."""
    tree = ast.parse(_JOB.read_text())
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


@pytest.fixture(scope="module")
def report():
    constants = _job_constants()
    world = constants["WORLD_SIZE"]
    if torch.cuda.device_count() < world:
        pytest.skip(f"tp4 runs {world} ranks; {torch.cuda.device_count()} device(s) visible")
    root = os.environ.get("LLM_MODELS_ROOT")
    checkpoint = R.default_checkpoint() if root else None
    if checkpoint is None or not os.path.isdir(checkpoint):
        pytest.skip("the checkpoint is not under $LLM_MODELS_ROOT/nemotron4-nano-preview")
    for device in range(world):
        free = torch.cuda.mem_get_info(device)[0] / 2**30
        if free < _FREE_GIB:
            pytest.skip(f"device {device} has {free:.0f} GiB free; a tp4 rank needs ~{_FREE_GIB}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [d for d in visible.split(",") if d.strip()] if visible else None
    devices = devices[:world] if devices else [str(i) for i in range(world)]
    # The ranks import tensorrt_llm absolutely, and a source checkout is not necessarily
    # installed: this file sits seven levels below the repo root.
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), f"no tensorrt_llm/ at {repo_root}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    with tempfile.TemporaryDirectory(prefix="tp4_decode_invariance_") as tmp:
        out = Path(tmp) / "report.json"
        completed = subprocess.run(
            [sys.executable, str(_JOB), "--report", str(out)],
            env=env,
            capture_output=True,
            text=True,
            timeout=constants["DEADLINE_S"] + _GRACE_S,
        )
        sys.stdout.write(completed.stdout)
        if completed.returncode != 0:
            raise AssertionError(
                f"{_JOB.name} exited {completed.returncode}\n"
                f"--- stdout ---\n{completed.stdout[-20000:]}\n"
                f"--- stderr ---\n{completed.stderr[-20000:]}"
            )
        yield json.loads(out.read_text())


def _failures(report: dict, check: str) -> list[str]:
    return [
        f"rank {rank}: {failure}"
        for rank, found in enumerate(report["checks"][check]["ranks"])
        for failure in found["failures"]
    ]


@pytest.mark.no_xdist
def test_reference_rows_are_real(report):
    """The reference is worth comparing against, on every rank: every probe's logits are finite,
    its slot holds non-zero state of the rank's heads in every GDP layer, and the K/V read at its
    decoded position through the manager's page table is non-zero in every attention layer."""
    failures = _failures(report, "reference_rows_are_real")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_every_rank_forms_the_same_logits_and_kv(report):
    """Every rank forms each probe's logits over the whole vocabulary from the same hidden state
    (vocab-parallel and gathered for a few rows, the whole lm_head above) and appends the K/V of
    the one KV head every rank keeps: the four ranks agree bit for bit. With the tests below,
    every rank's rows in every batch are then the same bits."""
    failures = _failures(report, "every_rank_forms_the_same_logits_and_kv")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_eager_rows_do_not_depend_on_the_batch(report):
    """Every probe inside eager decode batches of every size up to 1024, on every rank: the same
    logits, recurrent and conv state and appended K/V as alone."""
    failures = _failures(report, "eager_rows_do_not_depend_on_the_batch")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_graph_rows_equal_the_eager_row(report):
    """Every probe inside padded CUDA-graph decode batches of every size up to 1024, on every
    rank: the same bits as alone and eager, and the padding rows write no slot."""
    failures = _failures(report, "graph_rows_equal_the_eager_row")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_attention_is_within_band_of_thop_attention(report):
    """On a 64-row eager step in every attention layer of every rank: the invariant entry's append
    is the qkv GEMM's K/V, thop_attention's append writes the same bytes, and the two outputs
    over the rank's 4 query heads are within 4 element / 1.41 rms bf16 ulp of each other (each
    kernel within 2 element ulp of an fp32 reference; relative RMS 0.64 and 0.77 ulp)."""
    failures = _failures(report, "attention_is_within_band_of_thop_attention")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
@pytest.mark.parametrize("rival", ["cublas_mm", "thop_attention", "nccl_allreduce"])
def test_the_replaced_paths_depend_on_the_batch(report, rival):
    """The comparison has teeth on this model's own state: with the decode step's GEMMs that run
    through ``invariant_mm`` put back on ``cublas_mm``, its generation attention back on
    ``thop_attention``, or its all-reduce on NCCL instead of the MNNVL entry, some probe's step
    differs between its batch of one and the eager batches, on every rank. Each rival's
    reference is that rival's own batch of one."""
    ranks = report["checks"][f"rival_{rival}"]["ranks"]
    moved = [found["differ"] for found in ranks]
    assert all(moved), (
        f"with {rival} back in the decode step, no probe's row moved with the batch on some rank "
        f"({moved} of {ranks[0]['compared']} per rank), so the comparison cannot tell an "
        "invariant step from a variant one there"
    )
