# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the nemotron4 dep4 decode step is batch-invariant across the attention-DP group, on
real weights and real cache state, over the four GPUs of one node.

A request's decode step must give the same bits whatever batch it runs in on its own rank -- alone,
inside a larger eager batch, or inside a padded CUDA-graph batch -- and whatever the other three
ranks run beside it: nothing (each runs the attention-DP dummy), decode batches of their own, or a
prefill chunk. Under attention DP all of that changes from step to step with the other ranks' load:
whether the step runs as a graph, the graph size every rank pads to, the routed MoE's gathered
token count, where this rank's tokens sit in it, and whether the combine runs the padded or the
ragged collective. Compared per request: its fp32 logits, the GDP recurrent and conv state it
writes into its slot (32 layers) and the K/V it appends (8 attention layers).

The measurement runs once, as a 4-rank MPI job (`_dep4_decode_invariance_job.py`, which says what
each rank builds and runs); this module starts it, reads the records and gates them:

* `test_the_combine_sums_each_tokens_partials_in_one_order`: the routed MoE's fp32 reduce-scatter
  returns, on every rank and every split the decode steps make (padded and ragged, eager and
  replayed), exactly the chain ((P[r+1] + P[r+2]) + P[r+3]) + P[r] of a token's four partials;
* `test_reference_rows_are_real`: what the runs are compared against is worth comparing;
* `test_rows_do_not_depend_on_the_batch`, per group of runs: every probe's bits equal its
  reference's, and no placeholder row (padding, attention-DP dummy) writes a state slot;
* `test_attention_is_within_band_of_thop_attention`: the decode step's attention entry on this
  target's own pools;
* `test_the_replaced_paths_depend_on_the_batch`: the comparison has teeth on this model's state.
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


@pytest.mark.gpu4
def test_the_combine_sums_each_tokens_partials_in_one_order(measured) -> None:
    """The routed MoE's combine -- the catalog `reducescatter` on fp32 partials of the MoE width,
    sizes None when the ranks' counts agree, else the counts -- returns each token's sum of its
    four partials in one order on every split: the ring chain that ends at the token's own rank,
    bit for bit on every element. Even splits run NCCL's reduce-scatter and ragged ones a group
    of reduces, so this is what keeps a token's bits from following the other ranks' loads. That
    order is NCCL's algorithm choice on this node rather than a promise of the op; if this fails,
    the order changed and dep4's decode is no longer invariant across the ranks' loads. The chain
    is told apart from another order: summed in ascending rank order, some elements differ on
    every split on every rank but the last, where the two orders coincide (measured: 55-87% of
    the elements agree)."""
    records = measured["combine"]
    assert len(records) == J.WORLD_SIZE * (len(J.COMBINE_LAYOUTS) + len(J.COMBINE_GRAPH_HEIGHTS))
    off = [r for r in records if r["ring_equal"] != 1.0]
    distinct = [r for r in records if r["rank"] < J.WORLD_SIZE - 1]
    print(
        f"combine: {len(records) - len(off)} of {len(records)} rank records equal the ring chain "
        f"bitwise; ascending order equals it on {min(r['ascending_equal'] for r in distinct):.3f}"
        f"-{max(r['ascending_equal'] for r in distinct):.3f} of elements on ranks 0-2"
    )
    assert not off, f"the combine left the ring chain on {off[:8]}"
    blind = [r for r in distinct if r["ascending_equal"] == 1.0]
    assert not blind, f"the payload cannot tell the ring chain from ascending order on {blind[:8]}"


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
@pytest.mark.parametrize("group", J.GROUPS)
def test_rows_do_not_depend_on_the_batch(measured, group) -> None:
    """Every probe in every run of `group` (eager or padded-graph batches on its rank, the other
    ranks idle, decoding or prefilling): the same bits as alone, and no placeholder row -- CUDA-graph
    padding or the attention-DP dummy -- wrote a state slot. Prints the counts (`-s`)."""
    runs = [r for r in measured["runs"] if r["group"] == group]
    compared = sum(r["compared"] for r in runs)
    diffs = [d for r in runs for d in r["diffs"]]
    leaks = [
        f"{r['step']} rank {r['rank']}: slots {r['placeholder_writes']}"
        for r in runs
        if r["placeholder_writes"]
    ]
    steps = {r["step"] for r in runs}
    sizes = sorted({r["graph_size"] for r in runs if r["graph_size"]})
    print(
        f"{group}: {len(steps)} steps, {len(runs) // J.WORLD_SIZE} lockstep runs, "
        f"{compared} probe rows compared, {len(diffs)} differences"
        + (f", graph sizes {sizes}" if sizes else "")
    )
    assert compared > 0, f"{group}: no probe row was compared"
    assert not diffs, "\n".join(diffs[:40])
    assert not leaks, "placeholder rows wrote state:\n" + "\n".join(leaks[:20])


@pytest.mark.gpu4
def test_attention_is_within_band_of_thop_attention(measured) -> None:
    """On a 64-row eager decode step on every rank, in every attention layer: the entry appends
    each probe's K and V -- the qkv GEMM's K and V columns, bit for bit -- where the manager's page
    table puts the token; `thop_attention`'s append of the same tokens leaves those bytes as they
    were; and the two outputs are within the band of each other.

    Band, as the tp1 decode test derives it: each kernel is within 2 element ulp of an fp32
    reference (2.00 for this entry, 1.98 for thop's decode kernels), so 4 element ulp between
    them; relative RMS 0.64 and 0.77 ulp, so at most 1.41 between them."""
    records = measured["attention"]
    assert len(records) == J.WORLD_SIZE * 8, len(records)
    for r in records:
        print(
            f"rank {r['rank']} attention layer {r['layer']}: {r['element_ulp']:.2f} element ulp, "
            f"{r['rms_ulp']:.3f} rms ulp from thop_attention"
        )
        assert r["appended_is_qkv"], f"rank {r['rank']} layer {r['layer']}: appended K/V"
        assert r["thop_reappend_same"], f"rank {r['rank']} layer {r['layer']}: thop re-append"
        assert r["element_ulp"] <= 4.0 and r["rms_ulp"] <= 1.41, r


@pytest.mark.gpu4
@pytest.mark.parametrize("rival", J.RIVALS)
def test_the_replaced_paths_depend_on_the_batch(measured, rival) -> None:
    """The comparison has teeth on this model's own state: with the decode step's GEMMs that run
    through `invariant_mm` put back on `cublas_mm`, or its generation attention back on
    `thop_attention`, some probe's step differs between its batch of one and the eager batches of
    `J.RIVAL_SIZES` on its rank (the other ranks idle). Each rival's reference is that rival's own
    batch of one."""
    records = [r for r in measured["rivals"] if r["rival"] == rival]
    assert len(records) == J.WORLD_SIZE
    compared = sum(r["compared"] for r in records)
    differing = sum(r["differing"] for r in records)
    print(f"rival {rival}: {differing} of {compared} probe rows differ from their batch of one")
    for line in (d for r in records for d in r["differ"]):
        print(f"  {line}")
    assert differing, (
        f"with {rival} back in the decode step no probe's row moved with the batch, so the "
        "comparison cannot tell an invariant step from a variant one here"
    )
