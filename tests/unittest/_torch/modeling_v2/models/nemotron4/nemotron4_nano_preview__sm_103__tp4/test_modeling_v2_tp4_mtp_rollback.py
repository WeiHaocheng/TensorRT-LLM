# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one 4-GPU node): an MTP verify step of the nemotron4 tp4 target's GDP mixer rolls
back to the state MTP-off decode reaches, on every rank's own hybrid cache manager.

Collected entry point of ``_tp4_mtp_rollback_ranks.py`` beside it, which says what runs and how,
and derives its bands (``state_band`` and ``OUT_BAND``): layer 0's real GDP mixer at each rank's 6
heads, through the targets' own ``_gdp`` with its real all-reduce, on both hybrid managers built
under ``MTPDecodingConfig(max_draft_len=2)``, committed by the managers' own
``update_mamba_states``. The job runs its four ranks as an MPI world under
``mpirun``, so it is started in a fresh interpreter: the launcher must not import
``tensorrt_llm``, which calls ``MPI_Init``, and an MPI-initialized process cannot start
``mpirun``. Each test reads its own case from the job's report, every rank's record of it.
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

_JOB = Path(__file__).resolve().with_name("_tp4_mtp_rollback_ranks.py")
#: Headroom over the job's own deadline, so its message wins the race.
_GRACE_S = 120
#: Per device: layer 0's mixer, the embedding, two small cache managers and the workspace.
_FREE_GIB = 24


def _job_constants() -> dict:
    """The job's module-level constants, read off its source rather than by importing it."""
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
            pytest.skip(f"device {device} has {free:.0f} GiB free; a rank needs ~{_FREE_GIB}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [d for d in visible.split(",") if d.strip()] if visible else None
    devices = devices[:world] if devices else [str(i) for i in range(world)]
    # The ranks import tensorrt_llm absolutely, and a source checkout is not necessarily
    # installed: this file sits seven levels below the repo root.
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), f"no tensorrt_llm/ at {repo_root}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    with tempfile.TemporaryDirectory(prefix="tp4_mtp_rollback_") as tmp:
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


def _records(report: dict, kind: str, case: str) -> list[dict]:
    found = [r for r in report["records"] if r["kind"] == kind and r["case"] == case]
    assert sorted(r["rank"] for r in found) == list(range(report["ranks"])), (
        f"{kind} {case}: records from ranks {sorted(r['rank'] for r in found)}"
    )
    return found


def _separated(records: list[dict]) -> None:
    assert all(r["distinct"] for r in records), (
        "the states after 1, 2 and 3 tokens lie within the state band of each other, so a commit "
        f"from the wrong position could pass: {[r['separation'] for r in records]} fp32 ulp apart"
    )


@pytest.mark.no_xdist
@pytest.mark.parametrize("kind", ["v2", "v1"])
def test_verify_then_commit_reaches_decode_over_the_accepted_tokens(report, kind):
    """A padded CUDA-graph-shaped verify step on every rank: 5 requests and 3 padding rows
    (graph batch 8), T = 3, two verify -> commit rounds covering accepted counts 1..3. Every real
    row's committed state of the rank's heads is its slot's after the accepted decode steps (one
    token each through the decode entries, on the verify's own in_proj output): the conv state
    bit for bit, the recurrent state within the job's ``state_band``; the states after one, two
    and three tokens lie farther apart than that band; the verify writes no slot and the commit
    only the step's slots; the GDP verify entry's output rows lie within the job's ``OUT_BAND`` of
    the decode entry's."""
    records = _records(report, kind, "padded")
    failures = [f"rank {r['rank']}: {f}" for r in records for f in r["failures"]]
    assert not failures, "\n".join(failures)
    _separated(records)


@pytest.mark.no_xdist
@pytest.mark.parametrize("kind", ["v2", "v1"])
def test_mixed_step_commits_only_its_generation_rows(report, kind):
    """A mixed step on every rank: a fresh 37-token prompt (context row 0) beside 3 verify rows.
    The context row's tokens go through the prefill entries into its own slot; the generation
    rows verify from theirs, and the commit, offset past the context row, leaves the state the
    decode entries reach over the verify rows' own in_proj output -- the conv state bit for bit,
    the recurrent state within ``state_band`` -- never touching the context row's slot; the GDP
    verify entry's output rows lie within ``OUT_BAND`` of the decode entry's."""
    records = _records(report, kind, "mixed")
    failures = [f"rank {r['rank']}: {f}" for r in records for f in r["failures"]]
    assert not failures, "\n".join(failures)
    _separated(records)
