# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of the dep4 target's expert-parallel latent MoE block, on the checkpoint's own
weights, over the four GPUs of one node.

What runs is the target's own block on every rank: `PrefillTarget._moe` or `DecodeTarget._moe`
of a core built by its own constructor with that rank's dep4 mapping, over the target's weight
table -- the router GEMM and `noaux_tc_op` on the rank's own tokens, `fc1_latent_proj`, the
tokens' latent, ids and weights carried to the windows (the one-sided all-to-all on a balanced
step; one packed row per token all-gathered on the MPI-session symbol on a ragged one), the
rank's window of 128 experts in `latent_moe_experts_fused_act` with fp32 output, the fp32 combine
or reduce-scatter and its one rounding, `fc2_norm`, `fc2_latent_proj` and the shared expert; a
balanced decode step on the GEMM kernels a CUDA graph replays. The inputs are the routed and
shared inputs the model itself produces over the fixed prompt set, for layers 5 and 6.

The measurement runs once, as a 4-rank MPI job (`_dep4_moe_block_job.py`, which says what each
rank records); this module starts it, reads the records and gates them. On every rank, for the
rows whose output it owns:

* each step (`test_moe_block`): the family's MoE block check (`_nemotron4_moe_block.failures`) --
  routing against HF's router, the block against the fp32-GEMM and HF-order references, each op
  on the target's own input -- with its bands unchanged, plus the expert-parallel path's own
  gates (`failures`): the dispatch moves bytes, the window is the entry's arithmetic, and the
  expert sums differ from the full window's by the EP term alone;
* steps with an idle rank (`test_an_idle_ranks_dummy_keeps_the_real_rows_within_the_band`): the
  real rows stay within `FILLER_BAND` whatever the dummy and padding rows hold, and equal the
  single-GPU block's up to the EP term;
* `test_the_decode_block_replays_in_a_cuda_graph`: captured at each decode height on all four
  ranks, a replay is bitwise the eager call on the same kernels;
* `test_the_gates_refuse_plausible_wrong_blocks`: one planted mistake at a time breaks the gates
  named for it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../.."]
import _dep4_moe_block_job as J  # noqa: E402 — beside this file
import _nemotron4_moe_block as K  # noqa: E402 — needs the path declared above

# Each GPU test also carries `pytest.mark.gpu4`, which this tree (`utils/util.py`, imported by
# its conftest) turns into a decorator that skips below four GPUs, so it cannot sit in this list.
pytestmark = pytest.mark.no_xdist

_JOB = Path(__file__).resolve().with_name("_dep4_moe_block_job.py")


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


# ----------------------------------------------------------------------------------------------
# Bands of the expert-parallel path
# ----------------------------------------------------------------------------------------------
#
# The block bands are the family's (`_nemotron4_moe_block`), unchanged: everything on a rank's
# rows outside the expert sum is the single-GPU block's arithmetic on the same GEMM shapes, and
# the expert sum adds only the EP term below.

WINDOW_BAND = (4.0, 0.3)
"""A rank's window -- its 128 experts over the rows it was handed, fp32 out -- against the entry's
reference on the same arguments, in bf16 ulps of the row scale: the band `latent_moe_experts`
certifies its fp32-output cells with (entry and reference round at the same points and differ in
grouped-GEMM accumulation order alone); `latent_moe_experts_fused_act` gives its bits. Measured
on the model's own inputs, every step here: at most 3.71 element and 0.088 RMS; on random inputs
the entry measured up to 2.02 and 0.133."""

EP_ELEMENT_ULP = 1.0
EP_DIFFERING_FRACTION = 1e-5
"""The EP term: a rank's expert sums -- four windows' fp32 partials combined (or reduce-scattered)
in fp32, then rounded once -- against the full window's one fp32 sum of the same ten expert
outputs, rounded once (`latent_moe_experts` over every expert on the same rows, the order HF sums
them in).

Each window gives each of its slots the full call's bits, so the two fp32 sums add the same bf16
values grouped differently. fp32 holds most such sums exactly; where it does not, the two differ
by about one fp32 ulp, and after the one rounding they agree except where that difference
straddles a bf16 rounding boundary -- and there they are neighbours, at most one bf16 ulp of the
row scale apart. Measured: none of 87,465,984 elements differs over every step here (and none of
9,666,048 in the idle-rank steps); on random inputs, one in ten million. So: one ulp element-wise,
and at most one element in 10^5 (or one) differing, a hundred times the random-input rate.
Rounding each window to bf16 first, the alternative, moves 36.2-37.5% of them (`RIVALS`)."""

FILLER_BAND = (12.0, 0.8)
"""A real row's block output with the non-real rows holding one of `J.FILLERS`, against the same
row with them holding the first, in bf16 ulps of the row scale (element, relative RMS). Every rank
keeps its row count between the runs, and the gathered batch its size, so each cuBLAS GEMM runs
the same kernel, which computes a row from that row alone. What can move is the routed experts:
the fillers route to other experts, which changes the group sizes of `latent_moe_experts`' grouped
GEMMs, and the entry holds a row to its reference within its band (6 element / 0.4 RMS) rather
than to the bits it gets beside other rows. Each run sits within that band of the same reference,
so the two within twice it; the norm and the projection back up that follow carry a relative
error through unchanged. Measured over every idle step, layer, rank and filler: no element moved."""


def failures(metrics: dict) -> list[str]:
    """The gates one rank's measured step breaks, by name: the family's, then the
    expert-parallel path's."""
    broken = K.failures(metrics)
    s = metrics["stages"]
    dispatch = s["dispatch"]
    if not (
        dispatch["gathered_rows"] == dispatch["gathered_rows_expected"]
        and dispatch["own_rows_bitwise"]
        and dispatch["ids_bitwise"]
        and dispatch["weights_bitwise"]
        and dispatch["experts_are_the_rounded_sum"]
    ):
        broken.append("stage: dispatch")
    window = s["window"]
    if window["element_ulp"] > WINDOW_BAND[0] or window["rms_ulp"] > WINDOW_BAND[1]:
        broken.append("stage: window")
    ep = s["ep"]
    if ep["element_ulp"] > EP_ELEMENT_ULP or ep["differing"] > max(
        1, int(ep["numel"] * EP_DIFFERING_FRACTION)
    ):
        broken.append("stage: ep")
    if metrics.get("ep_extra_rows", 0):
        broken.append("block: T vs single-GPU twin")
    return broken


# ----------------------------------------------------------------------------------------------
# The job
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dict:
    """Every rank's records, from one run of the 4-rank job."""
    checkpoint = _checkpoint_or_skip()
    devices = _devices_or_skip()
    out = tmp_path_factory.mktemp("dep4_moe_block") / "records.json"
    completed = subprocess.run(
        [sys.executable, str(_JOB), "--out", str(out), "--checkpoint", checkpoint],
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
    return json.loads(out.read_text())


def _per_rank(records: list[dict], **match) -> list[dict]:
    """The four ranks' records of one measurement, in rank order."""
    found = sorted(
        (r for r in records if all(r[k] == v for k, v in match.items())), key=lambda r: r["rank"]
    )
    assert [r["rank"] for r in found] == list(range(J.WORLD_SIZE)), (match, len(found))
    return found


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------


@pytest.mark.gpu4
@pytest.mark.parametrize("step", J.CELLS, ids=[s.name for s in J.CELLS])
@pytest.mark.parametrize("layer", J.LAYERS, ids=[f"layer{i}" for i in J.LAYERS])
def test_moe_block(layer, step, measured) -> None:
    """One step of the attention-DP group: every rank's routing, block output and ops within their
    bands (`failures`). The measured figures are printed (`-rP`), one JSON record per rank."""
    broken = {}
    for record in _per_rank(measured["cells"], layer=layer, step=step.name):
        print(json.dumps(record))
        found = failures(record["metrics"])
        if found:
            broken[record["rank"]] = found
    assert not broken, f"layer {layer}, {step.name} {step.counts}: ranks break {broken}"


@pytest.mark.gpu4
@pytest.mark.parametrize("case", J.IDLE_STEPS, ids=[c.name for c in J.IDLE_STEPS])
@pytest.mark.parametrize("layer", J.LAYERS, ids=[f"layer{i}" for i in J.LAYERS])
def test_an_idle_ranks_dummy_keeps_the_real_rows_within_the_band(layer, case, measured) -> None:
    """A step in which a rank has no request -- it runs the attention-DP dummy -- and graph-padded
    ranks. Under each of `J.FILLERS` in every non-real row, the real rows' block outputs stay
    within `FILLER_BAND` of their outputs under the first, on every rank that has any; and under
    the first they equal the single-GPU block's on the same rows, which never sees the dummy or
    another rank, wherever the expert sums agree with the full window's (the EP term, gated as in
    `failures`). Prints the largest distance between fillers (`-s`).

    The fillers route to different experts, so the experts that real rows share with them change
    group sizes between runs: the check reaches the grouped GEMMs, not only a GEMM at fixed M."""
    records = _per_rank(measured["idle"], layer=layer, step=case.name)
    for record in records:
        print(json.dumps(record))
    assert any(any(r["non_real_experts_differ"]) for r in records), (
        "the fillers routed alike: the check would not move the experts' group sizes"
    )
    across = [d for r in records if r["real"] for d in r["real_rows_across_fillers"]]
    print(
        f"fillers, layer {layer}, {case.name}: real rows at most "
        f"{max(d['element_ulp'] for d in across):.3g} element / "
        f"{max(d['rms_ulp'] for d in across):.3g} RMS bf16 ulp from the first filler's, "
        f"{sum(d['differing'] for d in across)} of {sum(d['numel'] for d in across)} elements "
        "differing"
    )
    for record in records:
        if not record["real"]:
            continue
        where = f"layer {layer}, {case.name}, rank {record['rank']}"
        for filler, d in zip(J.FILLERS[1:], record["real_rows_across_fillers"]):
            assert d["element_ulp"] <= FILLER_BAND[0] and d["rms_ulp"] <= FILLER_BAND[1], (
                f"{where}: {filler} moved the real rows beyond the band: {d}"
            )
        ep = record["real_ep"]
        assert ep["element_ulp"] <= EP_ELEMENT_ULP and ep["differing"] <= max(
            1, int(ep["numel"] * EP_DIFFERING_FRACTION)
        ), f"{where}: expert sums beyond the EP term: {ep}"
        assert record["real_twin_extra_rows"] == 0, (
            f"{where}: real rows differ from the single-GPU block where the expert sums agree"
        )


@pytest.mark.gpu4
@pytest.mark.parametrize("height", J.GRAPH_HEIGHTS)
@pytest.mark.parametrize("layer", J.LAYERS, ids=[f"layer{i}" for i in J.LAYERS])
def test_the_decode_block_replays_in_a_cuda_graph(layer, height, measured) -> None:
    """`DecodeTarget._moe`, the all-to-all included, captured once on all four ranks at a decode
    height, then replayed on other requests' rows: bitwise the eager call on the same kernels on
    every rank. The decode target runs inside CUDA graphs, so a host sync, a buffer sized from
    device data or a collective that does not capture would show here; and the all-to-all puts
    a token in a receive row of its own choosing on every call, so a partial that depended on
    where its row lands would too."""
    for record in _per_rank(measured["graph"], layer=layer, height=height):
        assert record["replays_bitwise"] and all(record["replays_bitwise"]), record


#: rival -> the gates it must break on every rank. Each is one plausible way to get the block
#: wrong; the job plants it in the target's block on all four ranks at once.
RIVALS = {
    "routing weights not scaled by routed_scaling_factor": {
        "routing: weights",
        "routing: weight sums",
    },
    "routing weights rounded to bf16": {"routing: weights", "routing: weight sums"},
    "selection without the correction bias": {"routing: flips only at near-ties"},
    "the routing weight applied after down_proj": {
        "block: T vs R_t",
        "stage: experts",
        "stage: window",
        "stage: ep",
    },
    "fc2_norm left out": {"block: T vs R_t"},
    "fc2_norm rounded twice, as HF rounds it": {"block: T vs R_t", "stage: fc2_norm"},
    "the shared expert reading the routed input": {"block: T vs R_t", "stage: tail"},
    "bf16 window partial sums": {
        "block: T vs R_t",
        "stage: experts",
        "stage: window",
        "stage: ep",
    },
    "every rank running experts 0..127": {"block: T vs R_t", "stage: experts", "stage: ep"},
    "routing weights dispatched as bf16": {
        "block: T vs R_t",
        "stage: dispatch",
        "stage: experts",
        "stage: ep",
    },
    "the window partials kept, not combined": {
        "block: T vs R_t",
        "stage: experts",
        "stage: ep",
    },
    "routing weights all-gathered as bf16": {
        "block: T vs R_t",
        "stage: dispatch",
        "stage: experts",
        "stage: ep",
    },
    "the window partials kept, not reduce-scattered": {
        "block: T vs R_t",
        "stage: experts",
        "stage: ep",
    },
}


def test_every_rival_is_planted() -> None:
    assert set(RIVALS) == set(J.RIVALS)


@pytest.mark.gpu4
@pytest.mark.parametrize("rival", J.RIVALS)
def test_the_gates_refuse_plausible_wrong_blocks(rival, measured) -> None:
    """Each rival plants one mistake in the target's block on every rank, at every rank
    prefilling 2048 tokens of layer 5 (the all-to-all) or, for the all-gather path's own two, at a
    ragged prefill, and the gates named in `RIVALS` must break on every rank.

    The unscaled routing weights also show why the weights are gated directly: `fc2_norm`
    divides any per-token scale of the routed experts back out, so no block gate sees them.
    The two rounding mistakes an expert-parallel block invites are far outside the bands:
    measured on the all-gather path, bf16 partial sums reduce-scattered in bf16 put the block
    0.98-1.02 ulp of relative RMS from `R_t` (band 0.6) and 36.2-37.5% of the expert sums off the
    full window's; bf16 routing weights in the dispatch row, 0.93-0.98 and 37.0-38.9%. Running
    experts 0..127 on every rank cannot break the window gate on rank 0, whose window that is, so
    `RIVALS` does not name it."""
    for record in _per_rank(measured["rivals"], rival=rival):
        broken = set(failures(record["metrics"]))
        print(json.dumps(dict(rival=rival, rank=record["rank"], broken=sorted(broken))))
        assert RIVALS[rival] <= broken, (
            f"{rival}, rank {record['rank']}: {sorted(RIVALS[rival] - broken)} held"
        )
        if rival == "routing weights not scaled by routed_scaling_factor":
            assert not {b for b in broken if b.startswith("block")}, sorted(broken)
