# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: an MTP verify step of the nemotron4 dep4 target's GDP mixer rolls back to the accepted
tokens, on every rank of one node.

Under one-model MTP-Eagle a generation row carries its last token and D drafts. The target's GDP
mixer runs the row's 1 + D tokens from its state slot without writing it, and stores the state
after every token in the hybrid cache manager's per-position buffers; after acceptance the
engine's worker commits the last accepted token's state into the slot (`update_mamba_states`). A
rejected draft must leave no trace: the committed slot has to be the state MTP-off decode reaches
over the accepted tokens alone -- its conv state bit for bit, a window of inputs copied, and its
recurrent state within the band the two GDP entries' own bands put between them (the job's
`STATE_ULP`), which a state from the wrong position misses by orders. Under attention DP every rank
has its own manager -- one holding the trunk's state and KV and the MTP layer's KV -- and an idle
rank verifies the attention-DP dummy, whose drafts the commit promotes into the dummy's own slot.

The measurement runs once, as a 4-rank MPI job (`_dep4_mtp_rollback_job.py`, which says what each
rank builds and runs: layer 0's mixer on real weights through the targets' own `_gdp`, both
engine managers built for the rank's dep4 mapping); this module starts it and gates every rank's
records: where the commit writes, bit for bit; what the verify computes, within the job's bands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["."]
import _dep4_mtp_rollback_job as J  # noqa: E402 -- beside this file

pytestmark = pytest.mark.no_xdist

_JOB = Path(__file__).resolve().with_name("_dep4_mtp_rollback_job.py")


def _checkpoint_or_skip() -> str:
    root = os.environ.get("LLM_MODELS_ROOT")
    path = os.path.join(root, "nemotron4-nano-preview") if root else None
    if not path or not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
        pytest.skip("needs the checkpoint at $LLM_MODELS_ROOT/nemotron4-nano-preview")
    return path


def _devices_or_skip() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d.strip() for d in visible.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    if len(devices) < J.WORLD_SIZE:
        pytest.skip(f"dep4 runs on {J.WORLD_SIZE} GPUs of one node; {len(devices)} visible")
    return ",".join(devices[: J.WORLD_SIZE])


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> list:
    """Every rank's records, from one run of the 4-rank job."""
    checkpoint = _checkpoint_or_skip()
    devices = _devices_or_skip()
    out = tmp_path_factory.mktemp("dep4_mtp_rollback") / "records.json"
    completed = subprocess.run(
        [sys.executable, str(_JOB), "--out", str(out), "--checkpoint", checkpoint],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices),
        capture_output=True,
        text=True,
        timeout=J.DEADLINE_S + 120,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{_JOB.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-20000:]}\n"
            f"--- stderr ---\n{completed.stderr[-20000:]}"
        )
    return json.loads(out.read_text())


@pytest.mark.gpu4
@pytest.mark.parametrize("kind", J.MANAGERS)
@pytest.mark.parametrize("case", J.CASES)
def test_verify_then_commit_matches_decode_over_the_accepted_tokens(measured, kind, case) -> None:
    """On every rank, with the rank's own manager of `kind`: `case`'s verify writes no slot, the
    commit writes only the step's rows' engine slots (never a context row's), and each real row's
    committed state is its slot after the accepted decode steps -- the conv state bit for bit, the
    recurrent state within `J.STATE_ULP` -- with the states after one, two and three tokens
    differing pairwise by more than that; the verify's GDP entry outputs are within `J.OUT_ULP` of
    the decode steps'. Prints, per rank, the recurrent state's largest distance in units of its
    band, the outputs' in bf16 ulps and the least distance between positions in bands (`-s`)."""
    records = [r for r in measured if r["kind"] == kind and r["case"] == case]
    assert sorted(r["rank"] for r in records) == list(range(J.WORLD_SIZE)), records
    for r in records:
        measures = (
            f"; recurrent state {r['state']:.3g} of its band, GDP outputs {r['out']:.3g} bf16 ulp "
            f"(band {J.OUT_ULP}), positions {r['separation']:.3g} bands apart"
            if "state" in r
            else ""
        )
        print(f"rank {r['rank']} {r['manager']} {case}: {len(r['failures'])} failures{measures}")
        assert not r["failures"], "\n".join(r["failures"])
        assert r["distinct"], (
            f"rank {r['rank']}: the states after 1..3 tokens do not all differ beyond the band"
        )
    if case in ("padded", "mixed"):
        assert {n for r in records for n in r["accepted"]} == {1, 2, 3}
