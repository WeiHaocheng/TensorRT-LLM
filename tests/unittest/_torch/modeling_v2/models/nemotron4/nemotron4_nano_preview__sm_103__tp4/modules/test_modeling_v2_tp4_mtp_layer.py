# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of the tp4 target's MTP layer, on the checkpoint's own weights, over the four
GPUs of one node.

tp4's MTP layer is the checkpoint's `Nemotron4MultiTokenPredictionLayer` split over the ranks as a
trunk shortcut block is: its front end (`enorm`, `hnorm`, `eh_proj`), its norms and the router
replicated, each rank attending with 4 of the 16 query heads over the one KV head and running a
quarter of the shared expert's and of every routed expert's intermediate width, the partials
summed by the TP group's all-reduce; its KV lives in the separate draft manager. It is held to
what the tp1 target's MTP layer is held to (`_nemotron4_mtp_layer`: the same campaign of cells
through the worker's own draft loop, the same references -- the plain-torch transcription of
`modeling_nemotron4.py` and HF's own numerics -- the same bands, the same six planted mistakes),
on every rank, with each per-op reference cut to the rank's slice, plus the TP term: what the
ranks' bf16 partials, summed in fp32 and rounded once, change against the single full-K call.

The measurement runs once, as a 4-rank MPI job (`_tp4_mtp_layer_job.py`, which says what each rank
builds); this module starts it and gates every rank's records. Set `NEMOTRON4_MTP_TRUNK_CACHE=
<file>` to keep the trunk's hidden states (which take the whole text model to compute, ~2-3 min)
between runs.
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
import _nemotron4_mtp_layer as H  # noqa: E402
import _tp4_mtp_layer_job as J  # noqa: E402 -- beside this file

pytestmark = pytest.mark.no_xdist

_JOB = Path(__file__).resolve().with_name("_tp4_mtp_layer_job.py")

TP_TERM_BAND = (26.0, 5.4)
"""The TP term at the MTP layer's output, element / relative RMS in bf16 ulps: the target against
its twin with every all-reduce replaced by the single full-K call. Twice the worst measured over
every cell, depth and rank: 12.73 / 2.66 (the element in output column 2953, where the MoE's and
the shared expert's outputs nearly cancel the residual; see `OUTPUT_BAND`)."""

TP_TERM_MOE = (49.0, 5.0)
"""The TP term at the MTP layer's MoE output (`moe|W`), element / relative RMS in bf16 ulps: the
target's `_moe` against its twin's on the same call. Twice the worst measured over every cell, depth
and rank: 24.09 / 2.50 (the element in a prompt-pack cell, the RMS in a decode walk's second
depth). The latent and shared partials are each rounded to bf16 on a rank before the fp32 sum, and
the latent ranks' partials cancel, so this is the largest TP term of the layer; the MoE stage gate
in `test_mtp_layer` adds it to tp1's band."""

TP_TERM_LOGITS = 1.82
"""The TP term in the fp32 MTP logits (max |difference|, target against its twin). Twice the
worst measured: 0.908 (decode at batch 512, depth 1; 0.82-0.91 in three cells, at most 0.42 in the
others). Each rank's bf16 partials of the `o_proj` and the MoE are rounded before the fp32 sum, and
where the ranks' partials cancel -- the latent MoE's do -- that rounding is large against the sum:
the MoE output alone sits 2.6 ulps RMS from the twin's."""


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
        pytest.skip(f"tp4 runs on {J.WORLD_SIZE} GPUs of one node; {len(devices)} visible")
    return ",".join(devices[: J.WORLD_SIZE])


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dict:
    """Every rank's records, from one run of the 4-rank job."""
    checkpoint = _checkpoint_or_skip()
    devices = _devices_or_skip()
    tmp = tmp_path_factory.mktemp("tp4_mtp_layer")
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
    if os.environ.get("NEMOTRON4_TP4_MTP_LAYER_RECORDS"):
        Path(os.environ["NEMOTRON4_TP4_MTP_LAYER_RECORDS"]).write_text(json.dumps(records))
    for r in records["ranks"]:
        print(f"\n=== rank {r['rank']}\n{r['report']}")
        print(
            f"rank {r['rank']}: torch peak {r['peak_gib']:.1f} GiB, device in use "
            f"{r['device_in_use_gib']:.1f} GiB"
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
    """Every step of the cell on every rank: the MTP layer's output against `R_t`, its router and
    each of its ops (at the rank's slices) within the tp1 target's bands
    (`_nemotron4_mtp_layer.failures`); its fp32 logits against `R_t` within the tp1 target's
    `LOGIT_BAND` plus this cell's TP term (`test_the_tp_term`). The output's and the MoE stage's
    references run whole, so the TP term is inside them. The output's band has room for it; the
    MoE stage's, tp1's (25, 2.3), does not: the latent and shared partials, rounded to bf16 on
    each rank before the fp32 sum, put the MoE alone up to ~2.5 ulps RMS from its twin. So the MoE
    stage is held to tp1's band plus this cell's MoE TP term (`moe|W`), which `test_the_tp_term`
    bounds by `TP_TERM_MOE`, as the logits are."""
    key = f"{kind}|{depth}"
    for r in measured["ranks"]:
        cell = r["cells"][key]
        assert cell["rows"] > 0, (r["rank"], kind, depth)
        tp_term = r["tp"][key]["logits"]
        print(
            f"rank {r['rank']} {kind} depth {depth}: {cell['rows']} rows, {cell['steps']} steps, "
            f"T-R_t {cell['output']['T|R_t']:.2f} / {cell['output_rms']['T|R_t']:.3f} ulps, "
            f"logits T-R_t {cell['logits']['T|R_t']:.4f} (TP term {tp_term:.4f})"
        )
        failures = [f for f in cell["failures"] if not f.startswith(("logits: T|R_t", "moe:"))]
        assert not failures, (r["rank"], failures)
        moe, moe_tp, band = cell["stages"]["moe"], r["tp"][key]["moe|W"], H.STAGE_BANDS["moe"]
        assert moe[0] <= band[0] + moe_tp["element"] and moe[1] <= band[1] + moe_tp["rms"], (
            r["rank"],
            moe,
            moe_tp,
        )
        assert cell["logits"]["T|R_t"] <= H.LOGIT_BAND + tp_term, (
            r["rank"],
            cell["logits"]["T|R_t"],
            tp_term,
        )


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
def test_the_tp_term(measured):
    """The TP term on every MTP-layer call of every cell and rank: the layer's output and fp32
    logits against its twin with each all-reduce replaced by the single full-K call (tp1's
    composition on the call's own inputs), within `TP_TERM_BAND` / `TP_TERM_LOGITS`; the twin
    itself against `R_t` within the tp1 target's `OUTPUT_BAND` / `LOGIT_BAND`, which holds the
    composition the TP term is measured against to what the tp1 target is held to; and the MoE
    output against the twin's within `TP_TERM_MOE`. The `o_proj` sum against the twin's is printed
    beside them."""
    # The records are in bf16 ulps.
    worst = dict(element=0.0, rms=0.0, logits=0.0)
    for r in measured["ranks"]:
        for key, t in sorted(r["tp"].items()):
            out, twin = t["T|W"], t["W|R_t"]
            print(
                f"rank {r['rank']} {key}: T-W {out['element']:.2f} / {out['rms']:.3f} ulps "
                f"({out['identical']:.4f} identical), logits {t['logits']:.4f}; o-W "
                f"{t['o|W']['element']:.2f} / {t['o|W']['rms']:.3f}; moe-W "
                f"{t['moe|W']['element']:.2f} / {t['moe|W']['rms']:.3f}; W-R_t "
                f"{twin['element']:.2f} / {twin['rms']:.3f}, logits {t['twin_logits']:.4f}"
            )
            assert out["element"] <= TP_TERM_BAND[0], (r["rank"], key, out)
            assert out["rms"] <= TP_TERM_BAND[1], (r["rank"], key, out)
            assert t["logits"] <= TP_TERM_LOGITS, (r["rank"], key, t["logits"])
            assert t["moe|W"]["element"] <= TP_TERM_MOE[0], (r["rank"], key, t["moe|W"])
            assert t["moe|W"]["rms"] <= TP_TERM_MOE[1], (r["rank"], key, t["moe|W"])
            assert twin["element"] <= H.OUTPUT_BAND[0] and twin["rms"] <= H.OUTPUT_BAND[1], (
                r["rank"],
                key,
                twin,
            )
            assert t["twin_logits"] <= H.LOGIT_BAND, (r["rank"], key, t["twin_logits"])
            worst = dict(
                element=max(worst["element"], out["element"]),
                rms=max(worst["rms"], out["rms"]),
                logits=max(worst["logits"], t["logits"]),
            )
    print(f"TP term, worst over every cell, depth and rank: {json.dumps(worst)}")
