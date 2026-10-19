# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one 4-GPU node): the nemotron4 tp4 MTP verify step against the decode steps of the
same tokens, on every rank, on real weights and real cache state.

A verify runs the decode step's entries at three tokens a row -- cuBLAS at three times the rows,
``thop_attention``'s generation kernels at three tokens a row, the GDP and conv verify entries --
so a verify token is held to its decode step as tp1's test holds it
(``test_modeling_v2_verify_invariance.py``): within bands derived from measurement (the job's
``LOGIT_BAND`` and the rest, stated there), everything it leaves; reported, whether its rows move
with the batch. The drift check measures the deviation tp4's MTP identity and acceptance gates
rest on.

Collected entry point of ``_tp4_verify_invariance_ranks.py`` beside it, which says what runs and
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

_JOB = Path(__file__).resolve().with_name("_tp4_verify_invariance_ranks.py")
#: Headroom over the job's own deadline, so its message wins the race.
_GRACE_S = 300
#: Per device: ~48 GiB of the rank's weights with the MTP layer, ~22 GiB of verify buffers at a
#: 1032-row batch, ~15 GiB of state pools and their snapshot.
_FREE_GIB = 120


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
    with tempfile.TemporaryDirectory(prefix="tp4_verify_invariance_") as tmp:
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
    the K/V read at its tokens' positions are non-zero in every attention layer, and the slot's
    state of the rank's heads after its first, second and third token differs pairwise in every
    GDP layer -- so a state taken from the wrong position fails the comparison."""
    failures = _failures(report, "reference_rows_are_real")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_every_rank_forms_the_same_logits_and_kv(report):
    """Every rank forms each probe's decode and verify logits from the same hidden state with the
    whole lm_head, and appends the K/V of the one KV head every rank keeps: the four ranks agree
    bit for bit."""
    failures = _failures(report, "every_rank_forms_the_same_logits_and_kv")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_decode_and_verify_attend_through_thop(report):
    """The trunk's eight attention layers run ``thop_attention`` in a decode step (one token a
    row) and in a verify step (three), and nothing else, on every rank."""
    failures = _failures(report, "decode_and_verify_attend_through_thop")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_verify_alone_is_within_the_band_of_decode(report):
    """Every probe's verify step alone (B = 1), eagerly, with every entry's guard on, on every
    rank: each token within the bands of its decode step, and the verify writes no slot."""
    failures = _failures(report, "verify_alone_is_within_the_band_of_decode")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_verify_rows_in_eager_batches(report):
    """Every probe inside eager verify batches of every size up to 1024 requests (3072 rows), on
    every rank: within the bands of its decode steps; the job reports how many rows move with the
    batch."""
    failures = _failures(report, "verify_rows_in_eager_batches")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_verify_rows_in_padded_graphs(report):
    """Every probe inside padded CUDA-graph verify batches of every size up to 1024, on every
    rank: within the bands of its decode steps, and the verify writes no slot, the padding rows'
    included."""
    failures = _failures(report, "verify_rows_in_padded_graphs")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_commit_promotes_the_state_the_verify_stored(report):
    """After a padded graph verify, the worker's commit leaves in each probe's slot, on every
    rank, exactly the state the verify stored for its last accepted token (1, 2 and 3), in every
    GDP layer, within the state bands of its decode step's, and writes no other request's slot."""
    failures = _failures(report, "commit_promotes_the_state_the_verify_stored")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.no_xdist
def test_verify_at_the_length_limit(report):
    """A verify reaching ``MAX_SEQ_LEN + 1`` by its drafts: thop_attention's bound and attention
    window, the cache manager's ``max_seq_len``, hold every key, and alone and in a padded graph
    each token stays within the bands of its decode step, on every rank."""
    failures = _failures(report, "verify_at_the_length_limit")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_the_band_tells_a_state_one_token_off(report):
    """A verify from the state one token too far moves every probe's logits by more than twice the
    logit band, and one of its top gaps past the gap band, on every rank."""
    failures = _failures(report, "the_band_tells_a_state_one_token_off")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_the_drift_over_consecutive_verifies(report):
    """MTP on along MTP off's own greedy tokens for as many tokens as the MTP identity gate
    generates, MTP off decoding each probe alone, each verify alone in the graph of one with every
    draft accepted and committed: at every matched position the logits stay within the drift
    bands of the decode step's, and every top-1 change sits at a decode margin within the gap
    band, on every rank."""
    failures = _failures(report, "the_drift_over_consecutive_verifies")
    assert not failures, "\n".join(failures)


@pytest.mark.no_xdist
def test_the_mtp_layer_drafts_as_prefill_drafts(report):
    """The MTP layer's draft step through ``MTPLayer``'s routing reaches the decode target, whose
    draft logits agree with the prefill target's within the logit band, on every rank; every
    rank's outputs of both targets are rank 0's bits."""
    failures = _failures(report, "the_mtp_layer_drafts_as_prefill_drafts")
    assert not failures, "\n".join(failures)
