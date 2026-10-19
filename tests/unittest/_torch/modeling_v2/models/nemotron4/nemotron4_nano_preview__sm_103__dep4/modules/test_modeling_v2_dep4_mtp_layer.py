# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of the dep4 target's MTP layer, on the checkpoint's own weights, over the four
GPUs of one node.

dep4's MTP layer is the checkpoint's `Nemotron4MultiTokenPredictionLayer` on every rank for that
rank's requests, its KV in the rank's one hybrid cache manager (global layer 44 at that manager's
index) and its MoE across the ranks: each rank routes its own tokens, the latents reach the
windows that hold their experts (the one-sided all-to-all on a balanced draft step, the all-gather
on a ragged one), each rank runs its window of 128 of the layer's 512 experts with fp32 partial
sums, and the partials come back summed and rounded once -- with collectives sized by the counts
the engine's worker hands each draft step. It is held to what the tp1 target's MTP layer is held
to (`_nemotron4_mtp_layer`: the same campaign of cells through the worker's own draft loop, the
same references -- the plain-torch transcription of `modeling_nemotron4.py` and HF's own numerics
-- the same bands, the same six planted mistakes), on every rank, plus the EP term: what the four
windows' fp32 partials, added in the combine's or the reduce-scatter's order, change against one
full-window call.

The measurement runs once, as a 4-rank MPI job (`_dep4_mtp_layer_job.py`, which says what each rank
builds and how the ranks keep their collectives in lockstep); this module starts it and gates every
rank's records. Set `NEMOTRON4_MTP_TRUNK_CACHE=<file>` to keep the trunk's hidden states (which
take the whole text model to compute, ~2-3 min) between runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

__extra_import_path__ = [".", "../.."]
import _dep4_mtp_layer_job as J  # noqa: E402 -- beside this file
import _nemotron4_mtp_layer as H  # noqa: E402

pytestmark = pytest.mark.no_xdist

_JOB = Path(__file__).resolve().with_name("_dep4_mtp_layer_job.py")

EP_ROWS_BAND = 1e-3
"""The EP term, as the share of MoE output rows whose bits differ from the single full-window call.
A row differs when any of its 768 latent sums meets a bf16 rounding boundary in one order and not
the other (op-moe-ep-cells: ~1 in 1e7 latent elements on random inputs), and the projection back up
then spreads the one-ulp difference over the row. Measured over the whole campaign on 4 ranks: 3 of
2,620,292 rows, 12 elements each, at most 0.15 ulp of the row's largest value. The band is a
structural check rather than a fit to that: a wrong window, a missing partial or partials rounded
to bf16 before the sum move most rows (bf16 partials alone move ~60% of the experts' outputs,
dep4-core)."""


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
def measured(tmp_path_factory) -> dict:
    """Every rank's records, from one run of the 4-rank job."""
    checkpoint = _checkpoint_or_skip()
    devices = _devices_or_skip()
    tmp = tmp_path_factory.mktemp("dep4_mtp_layer")
    trunk_cache = os.environ.get(H.TRUNK_CACHE_ENV) or str(tmp / "trunk_hidden.pt")
    completed = subprocess.run(
        [
            sys.executable,
            str(_JOB),
            "--out",
            str(tmp / "records.json"),
            "--checkpoint",
            checkpoint,
            "--trunk-cache",
            trunk_cache,
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices),
        capture_output=True,
        text=True,
        timeout=J.DEADLINE_S + 300,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{_JOB.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-20000:]}\n"
            f"--- stderr ---\n{completed.stderr[-20000:]}"
        )
    print(completed.stdout)
    records = json.loads((tmp / "records.json").read_text())
    for r in records["ranks"]:
        print(f"\n=== rank {r['rank']} (prompts in order {r['order']})\n{r['report']}")
        print(
            f"rank {r['rank']}: drained {r['drained_steps']} attention-DP dummy steps; torch peak "
            f"{r['peak_gib']:.1f} GiB, device in use {r['device_in_use_gib']:.1f} GiB"
        )
    return records


@pytest.mark.gpu4
def test_reference_pieces_are_mtp_draft_hidden(measured):
    """`_nemotron4_mtp_layer.sequence_reference`'s `R_own` -- the MTP layer split at its MoE so the
    routing can be swapped -- is `mtp_draft_hidden(mode="eagle")` bit for bit at both depths, on a
    short prompt of the set with the trunk's own hidden states, on every rank."""
    ranks = measured["ranks"]
    assert len(ranks) == J.WORLD_SIZE
    assert all(r["reference_pieces"] == [True, True] for r in ranks), [
        r["reference_pieces"] for r in ranks
    ]


@pytest.mark.gpu4
@pytest.mark.parametrize("depth", (1, 2))
@pytest.mark.parametrize("kind", H.CELL_KINDS)
def test_mtp_layer(measured, kind, depth):
    """Every step of the cell on every rank: the MTP layer's output against `R_t`, its logits, its
    router and each of its ops within the tp1 target's bands (`_nemotron4_mtp_layer.failures`).
    The MoE stage's reference adds the ten experts in one fp32 sum, so the EP term is inside it."""
    for r in measured["ranks"]:
        cell = r["cells"][f"{kind}|{depth}"]
        assert cell["rows"] > 0, (r["rank"], kind, depth)
        print(
            f"rank {r['rank']} {kind} depth {depth}: {cell['rows']} rows, {cell['steps']} steps, "
            f"T-R_t {cell['output']['T|R_t']:.2f} / {cell['output_rms']['T|R_t']:.3f} ulps, "
            f"logits T-R_t {cell['logits']['T|R_t']:.4f}"
        )
        assert not cell["failures"], (r["rank"], cell["failures"])


@pytest.mark.gpu4
@pytest.mark.parametrize("depth", (1, 2))
def test_drafts_against_the_reference_chain(measured, depth):
    """Every cell of one depth together, on every rank, each side drafting with its own routing:
    the target's MTP logits are no further from the reference's drafting chain (`R_own`, what
    `mtp-acceptance` compares the engine's drafts with) than `EPSILON_VS_HF` times HF's own
    implementation's. Prints the largest MTP-logit difference (the epsilon the acceptance gate
    needs), split by whether the target routed the row to the reference's experts, and the draft
    agreement."""
    for r in measured["ranks"]:
        totals = dict(r["by_depth"][str(depth)])
        margins = totals.pop("draft_flip_margins")
        print(f"rank {r['rank']} depth {depth}: {json.dumps(totals)}")
        for side, values in margins.items():
            if values:
                print(
                    f"  {side}: {len(values)} drafts differ from R_own's, at R_own margins up to "
                    f"{max(values):.4f} (median {sorted(values)[len(values) // 2]:.4f})"
                )
        eps_t, eps_h = totals["logits"]["T|R_own"], totals["logits"]["H_own|R_own"]
        assert eps_t <= H.EPSILON_VS_HF * eps_h, (r["rank"], eps_t, eps_h)


@pytest.mark.gpu4
@pytest.mark.parametrize("rival", sorted(H.RIVALS))
def test_the_gates_refuse_plausible_wrong_layers(measured, rival):
    """Each planted mistake breaks the output band and the gate of the op it sits in, on every
    rank."""
    for r in measured["ranks"]:
        missing = set(H.RIVALS_BREAK[rival]) - set(r["rivals"][rival])
        assert not missing, (r["rank"], rival, r["rivals"][rival], sorted(missing))


@pytest.mark.gpu4
def test_the_ep_term(measured):
    """The EP term on every MTP-layer call of every cell and rank: the MoE output of dep4's four
    fp32 window partials, combined (or reduce-scattered) and rounded once, against the same
    composition with one full-window call. It is reported per cell and depth, and the share of rows it moves is
    held to `EP_ROWS_BAND`."""
    rows = differing = 0
    for r in measured["ranks"]:
        for key, e in sorted(r["ep"].items()):
            rows += e["rows"]
            differing += e["rows_differing"]
            if e["rows_differing"]:
                print(
                    f"rank {r['rank']} {key}: {e['rows_differing']} of {e['rows']} rows, "
                    f"{e['elements_differing']} elements, max |d| {e['max_abs']:.3e} "
                    f"({e['max_ulp']:.2f} ulp of the row's largest)"
                )
    print(f"EP term: {differing} of {rows} MoE output rows differ from one full-window call")
    assert rows > 0
    assert differing <= EP_ROWS_BAND * rows, (differing, rows)
