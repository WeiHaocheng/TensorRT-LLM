# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: how far the nemotron4 dep4 decode step moves with its batch across the attention-DP
group, on real weights and real cache state, over the four GPUs of one node.

A decode step runs its GEMMs through cuBLAS and its attention through ``thop_attention``'s
generation kernels, and both pick their kernel by the rank's batch: cuBLAS by the row count, thop by
the batch size and history length. The routed MoE sees every rank's tokens besides: the other ranks'
load changes the batch the experts' windows run over and the split the reduce-scatter combines. So
a request's decode step gives slightly different bits alone, inside a larger eager batch or a padded
CUDA-graph batch on its rank, and beside other ranks that idle (each runs the attention-DP dummy),
decode or prefill -- what CUDA graphs change between the two legs of a CUDA-graph on / off identity
run, and what batching changes between a prompt alone and in a batch. This test measures how far,
per request: its fp32 logits (the largest |difference| and the largest shift of a gap between its
top-1 and one of its top `J.TOP` tokens), and the GDP recurrent and conv state it writes into its
slot (32 layers) and the K/V it appends (8 attention layers), each as max |difference| / max
|alone|. Every quantity is held to a band derived from that measurement (`LOGIT_BAND` and the
rest). Placeholder rows -- CUDA-graph padding and the attention-DP dummy -- write no slot, exactly.

The measurement runs once, as a 4-rank MPI job (`_dep4_decode_invariance_job.py`, which says what
each rank builds and runs); this module starts it, reads the records and gates them:

* `test_reference_rows_are_real`: what the runs are compared against is worth comparing;
* `test_decode_attends_through_thop_and_appends_each_rows_kv`: the decode step's attention on this
  target's own pools;
* `test_eager_rows_stay_within_the_band` and `test_graph_rows_stay_within_the_band`: every probe
  against its step alone, group by group of runs, and no placeholder row writes a state slot;
* `test_the_band_tells_a_state_one_token_off`: the bands have teeth.
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
import _dep4_decode_invariance_job as J  # noqa: E402 -- beside this file

# Each GPU test also carries `pytest.mark.gpu4`, which this tree (`utils/util.py`, imported by its
# conftest) turns into a decorator that skips below four GPUs, so it cannot sit in this list.
pytestmark = pytest.mark.no_xdist

_JOB = Path(__file__).resolve().with_name("_dep4_decode_invariance_job.py")

# The bands: how far a probe's decode step may sit from the same step alone on its rank, the other
# ranks idle. Measured on this model's own state over every probe in every run of the job: eager
# batches of 2 to 256 rows beside idle, decoding and prefilling ranks, and padded graphs of 1 to
# 256, every rank padded to the busiest rank's size and an idle rank on the attention-DP dummy. The
# kernels are deterministic, so the worst cases repeat run to run. The worst cases, over 246 eager
# and 241 graph rows: |logit diff| 0.8574, top-gap shift 0.2512, recurrent state 0.0275, conv state
# 0.0355 and K/V 0.0606 of max |alone|; no top-1 changed. A graph batch against the same rows run
# eagerly, what the CUDA-graph identity gate's legs differ by in one step, measured 0.7520 /
# 0.2760. Each band is its worst case plus a quarter, rounded up: room for a batch, history, graph
# size or load on the other ranks the runs did not take, where cuBLAS or thop may pick yet another
# kernel.
LOGIT_BAND = 1.08
"""max |batched logit - alone logit| over the vocabulary, fp32."""
GAP_BAND = 0.32
"""The largest change, batched against alone, of the gap between the alone step's top-1 and any of
its top `J.TOP` tokens: how close to a tie the batch can tip the alone step's choice."""
SSM_BAND = 0.035
CONV_BAND = 0.045
KV_BAND = 0.076
"""max |batched - alone| / max |alone|, per tensor, over every GDP layer (recurrent state fp32, conv
state bf16) and every attention layer (the appended K/V)."""


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
    out = tmp_path_factory.mktemp("dep4_decode_invariance") / "records.json"
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


def _outside(drift: dict) -> list[str]:
    """What of a probe row's drift lies outside the bands."""
    out = []
    for name, key, band in (
        ("logits", "logit", LOGIT_BAND),
        ("top-gap shift", "gap", GAP_BAND),
        ("recurrent state", "ssm", SSM_BAND),
        ("conv state", "conv", CONV_BAND),
        ("K/V", "kv", KV_BAND),
    ):
        if drift[key] > band:
            out.append(f"{name} {drift[key]:.4g} outside its band {band:.4g}")
    return out


def _summary(drifts: list[dict]) -> str:
    flips = [d for d in drifts if d["argmax"][0] != d["argmax"][1]]
    return (
        f"{len(drifts)} rows: max |logit diff| {max(d['logit'] for d in drifts):.4f}, max top-gap "
        f"shift {max(d['gap'] for d in drifts):.4f}, recurrent / conv state / K/V "
        f"{max(d['ssm'] for d in drifts):.3g} / {max(d['conv'] for d in drifts):.3g} / "
        f"{max(d['kv'] for d in drifts):.3g} of max |alone|; {len(flips)} top-1 changes"
        + (f" at alone margins {[round(d['margin'], 4) for d in flips]}" if flips else "")
    )


def _gate(measured: dict, graph: bool) -> tuple[list[dict], list[str], list[str]]:
    """The groups of runs that replay a CUDA graph (`graph`) or run eagerly: prints each group's
    drift, and returns every probe row's drift, what lies outside the bands, and the runs whose
    placeholder rows wrote a state slot."""
    drifts, failures, leaks = [], [], []
    for group in dict.fromkeys(s.group for s in J.STEPS if s.graph == graph):
        runs = [r for r in measured["runs"] if r["group"] == group]
        found = [d for r in runs for d in r["drifts"]]
        assert found, f"{group}: no probe row was compared"
        sizes = sorted({r["graph_size"] for r in runs if r["graph_size"]})
        print(f"{group}: {_summary(found)}" + (f", graph sizes {sizes}" if sizes else ""))
        drifts += found
        failures += [
            f"{r['step']}, rank {r['rank']}, {d['probe']} at row {d['row']}: {f}"
            for r in runs
            for d in r["drifts"]
            for f in _outside(d)
        ]
        leaks += [
            f"{r['step']} rank {r['rank']}: slots {r['placeholder_writes']}"
            for r in runs
            if r["placeholder_writes"]
        ]
    return drifts, failures, leaks


@pytest.mark.gpu4
def test_reference_rows_are_real(measured) -> None:
    """The reference -- each probe decoding alone on its rank, the other ranks idle -- is worth
    comparing against: finite logits, non-zero state in every GDP layer of its slot, non-zero K/V
    at its decoded position in every attention layer; three probes on every rank."""
    refs = measured["reference"]
    assert sorted(r["rank"] for r in refs) == sorted(list(range(J.WORLD_SIZE)) * 3)
    for r in refs:
        assert r["finite"] and r["ssm_nonzero"] and r["conv_nonzero"] and r["kv_nonzero"], r


@pytest.mark.gpu4
def test_decode_attends_through_thop_and_appends_each_rows_kv(measured) -> None:
    """A decode step's eight attention layers run `thop_attention` and nothing else, one token a
    row, and on a 64-row eager step on every rank each probe's K and V land where the manager's
    page table puts its token: the K and V columns of its qkv row, bit for bit."""
    records = measured["attention"]
    for rank in range(J.WORLD_SIZE):
        layers = sorted(r["layer"] for r in records if r["rank"] == rank)
        assert len(layers) == 8 and len(set(layers)) == 8, f"rank {rank}: thop calls {layers}"
    for r in records:
        assert r["q_rows"] == 64, f"rank {r['rank']} layer {r['layer']}: {r['q_rows']} q rows"
        assert r["appended_is_qkv"], (
            f"rank {r['rank']} layer {r['layer']}: the appended K/V are not the qkv GEMM's K/V "
            "columns"
        )


@pytest.mark.gpu4
def test_eager_rows_stay_within_the_band(measured) -> None:
    """Every probe in every eager run -- decode batches on its rank while the other ranks idle,
    decode or prefill -- against its step alone: within the bands, and no placeholder row (the
    attention-DP dummy) wrote a state slot. Reports each group's drift and every top-1 change
    with the alone step's margin there (`-s`), then every eager run's together."""
    drifts, failures, leaks = _gate(measured, graph=False)
    print(f"eager: {_summary(drifts)}")
    assert not failures, "\n".join(failures[:40])
    assert not leaks, "placeholder rows wrote state:\n" + "\n".join(leaks[:20])


@pytest.mark.gpu4
def test_graph_rows_stay_within_the_band(measured) -> None:
    """Every probe in every padded CUDA-graph run, B = 1 included, every rank padded to the busiest
    rank's graph and an idle rank on the attention-DP dummy, against its eager step alone: within
    the bands, and no placeholder row wrote a state slot. Reports each group's drift, every graph
    run's together, and each graph run against the same rows run eagerly beside the same ranks --
    what the CUDA-graph identity gate's two legs differ by in one step."""
    drifts, failures, leaks = _gate(measured, graph=True)
    runs = [r for r in measured["runs"] if r["graph_size"]]
    legs = [leg for r in runs for leg in r["legs"]]
    print(f"graph: sizes {sorted({r['graph_size'] for r in runs})}, {_summary(drifts)}")
    print(
        f"graph against eager, the same rows: max |logit diff| {max(x for x, _ in legs):.4f}, "
        f"max top-gap shift {max(y for _, y in legs):.4f} over {len(legs)} rows"
    )
    assert not failures, "\n".join(failures[:40])
    assert not leaks, "placeholder rows wrote state:\n" + "\n".join(leaks[:20])


@pytest.mark.gpu4
def test_the_band_tells_a_state_one_token_off(measured) -> None:
    """The bands have teeth: a decode step from the state one token too far -- every GDP slot
    holding the state its own decode step left, as a rollback that kept a rejected draft would
    leave it -- moves every probe's logits by more than twice the logit band, each probe alone on
    its rank, the other ranks idle. (Measured: at least 4.604, on the 4096-token prefix of
    `gsm8k_multiturn`, whose top gaps it moves by only 0.282, inside the gap band: there the logit
    band is what tells the two states apart.)"""
    records = measured["teeth"]
    assert len(records) == J.WORLD_SIZE
    for r in records:
        for line in r["lines"]:
            print(line)
    logit = min(r["logit"] for r in records)
    gap = min(r["gap"] for r in records)
    print(
        f"one token off: at least {logit:.3f} / {gap:.3f} on every probe, against the bands "
        f"{LOGIT_BAND} / {GAP_BAND}"
    )
    assert logit > 2 * LOGIT_BAND, (logit, gap)
