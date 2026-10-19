# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one 4-GPU node): how far the nemotron4 tp4 decode step moves with its batch on every
rank, on real weights and real cache state.

A request's decode step runs cuBLAS and ``thop_attention``'s generation kernels, which pick their
kernels by the batch, so its logits and the state it writes move with the batch it runs in. Each
probe's step in eager and padded CUDA-graph batches is held to its step alone within bands derived
from measurement (the job's ``LOGIT_BAND`` and the rest, stated there); every rank holds the same
bits of it, and the padding rows write no slot, exactly.

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
    the one KV head every rank keeps: the four ranks agree bit for bit, alone here and in every
    batch of the tests below."""
    failures = _failures(report, "every_rank_forms_the_same_logits_and_kv")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_decode_attends_through_thop_and_appends_each_rows_kv(report):
    """A decode step's eight attention layers run ``thop_attention`` and nothing else, on every
    rank, and on a 64-row eager step each row's K and V land where the manager's page table puts
    its token: the K and V columns of its qkv row, bit for bit."""
    failures = _failures(report, "decode_attends_through_thop_and_appends_each_rows_kv")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_eager_rows_stay_within_the_band(report):
    """Every probe inside eager decode batches of every size up to 1024, on every rank, against its
    step alone: its logits, recurrent and conv state and appended K/V within the bands, and every
    rank holding the same bits of its logits and K/V. The job reports the drift and every top-1
    change with the alone step's margin there."""
    failures = _failures(report, "eager_rows_stay_within_the_band")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_graph_rows_stay_within_the_band(report):
    """Every probe inside padded CUDA-graph decode batches of every size up to 1024, B = 1
    included, on every rank, against its eager step alone: within the bands, every rank holding
    the same bits, and the padding rows write no slot. The job reports, besides, each graph batch
    against the same rows run eagerly."""
    failures = _failures(report, "graph_rows_stay_within_the_band")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_the_band_tells_a_state_one_token_off(report):
    """A decode step from the state one token too far moves every probe's logits by more than
    twice the logit band, on every rank (its top gaps it can move by less than the gap band)."""
    failures = _failures(report, "the_band_tells_a_state_one_token_off")
    assert not failures, "\n".join(failures)
