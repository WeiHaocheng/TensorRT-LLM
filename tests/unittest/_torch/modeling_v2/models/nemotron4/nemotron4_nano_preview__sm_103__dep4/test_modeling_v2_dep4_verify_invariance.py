# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the nemotron4 dep4 MTP verify step against the decode steps of the same tokens, across
the attention-DP group, on real weights and real cache state, over the four GPUs of one node.

Under one-model MTP-Eagle (``max_draft_len`` 2) a generation row carries its last accepted token
and two drafts, and the trunk verifies the three in one step: 3B rows through every GEMM, three
tokens per request through attention, the GDP mixers through the verify entries, and every rank's
tokens through the routed MoE's collectives. Every entry of that step but one gives token j the
bits it gets as an MTP-off decode step of its own, whatever batch it runs in on its rank and
whatever the other three ranks run beside it. Attention is the exception: a verify runs
``thop_attention``'s generation kernels where decode runs ``invariant_gen_attention``. So a verify
token is held to its decode step as tp1's test holds it (``test_modeling_v2_verify_invariance.py``):
bitwise, its state in the GDP layers before the first attention layer and its K/V in that layer;
within bands derived from measurement (the job's ``LOGIT_BAND`` and the rest, stated there), its
fp32 logits and its state and K/V in every later layer; reported, whether its logits move from its
verify alone.

The measurement runs once, as a 4-rank MPI job (`_dep4_verify_invariance_job.py`, which says what
each rank builds and runs); this module starts it, reads the records and gates them:

* `test_the_combine_sums_each_tokens_partials_in_one_order`: on every split the runs make, the
  routed MoE's fp32 reduce-scatter adds a token's four partials in one order;
* `test_reference_rows_are_real`: what the runs are compared against is worth comparing;
* `test_verify_rows_within_the_band_of_decode`, per group of runs: every probe's three verify
  tokens against its decode steps', and no run wrote a slot its rows do not own;
* `test_commit_promotes_the_state_the_verify_stored`: the worker's commit promotes exactly the
  verified state, on every rank and on an idle one;
* `test_verify_at_the_length_limit`: the longest history a verify starts from;
* `test_the_exact_comparison_moves_with_cublas`: the bitwise half has teeth on this model's state;
* `test_the_band_tells_a_state_one_token_off`: the bands have teeth;
* `test_the_drift_over_consecutive_verifies`: MTP on along MTP off's own tokens, the deviation
  dep4's MTP identity and acceptance gates rest on;
* `test_the_mtp_layer_drafts_as_prefill_drafts`: the MTP layer's draft step, its MoE across the
  ranks, is the same in both targets, and the decode target's row-invariant MoE is not handed its
  weights.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["."]
import _dep4_verify_invariance_job as J  # noqa: E402 -- beside this file

# Each GPU test also carries `pytest.mark.gpu4`, which this tree (`utils/util.py`, imported by its
# conftest) turns into a decorator that skips below four GPUs, so it cannot sit in this list.
pytestmark = pytest.mark.no_xdist

_JOB = Path(__file__).resolve().with_name("_dep4_verify_invariance_job.py")


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
    out = tmp_path_factory.mktemp("dep4_verify_invariance") / "records.json"
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
    four partials in one order on every split the verify runs make, padded and ragged, eager and
    replayed: the ring chain ((P[r+1] + P[r+2]) + P[r+3]) + P[r] that ends at the token's own rank,
    bit for bit. Verify steps send three rows per request through it, so their splits are not the
    decode steps'. That order is NCCL's algorithm choice on this node rather than a promise of the
    op; if this fails, the order changed and a verify token's bits follow the other ranks' loads.
    The chain is told apart from ascending rank order on every split on every rank but the last,
    where the two orders coincide."""
    records = measured["combine"]
    assert len(records) == J.WORLD_SIZE * len(J.combine_layouts())
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
    """The reference -- each probe's three MTP-off decode steps alone on its rank, the other ranks
    idle -- is worth comparing against: finite logits, non-zero state in every GDP layer of its
    slot, non-zero K/V at its tokens' positions in every attention layer, and the slot's state
    after its first, second and third token differs pairwise in every GDP layer, recurrent and
    conv state alike -- so a state taken from the wrong position fails the comparison. Three
    probes on every rank."""
    refs = measured["reference"]
    assert sorted(r["rank"] for r in refs) == sorted(list(range(J.WORLD_SIZE)) * 3)
    for r in refs:
        print(f"rank {r['rank']} {r['probe']}: verify tokens {r['tokens']}")
        assert r["finite"] and r["kv_nonzero"] and r["states_nonzero"], r
        assert r["positions_distinct"], r


@pytest.mark.gpu4
@pytest.mark.parametrize("group", J.GROUPS)
def test_verify_rows_within_the_band_of_decode(measured, group) -> None:
    """Every probe in every run of `group` (its verify alone with every entry's guard on, then
    eager or padded-graph verify batches on its rank, the other ranks idle, verifying, decoding or
    prefilling): each of its three verify tokens has the bits of its decode step in the GDP layers
    before the first attention layer and in that layer's K/V, and is within the bands after. A
    verify writes no slot, nor does a placeholder row (padding, the attention-DP dummy); a
    decoding rank writes only its rows' slots, a prefilling one only its loader's. Prints the worst
    case of each compared quantity and how many probe rows move from their verify alone (`-s`)."""
    runs = [r for r in measured["runs"] if r["group"] == group]
    compared = sum(r["compared"] for r in runs)
    failures = [f for r in runs for f in r["failures"]]
    leaks = [
        f"{r['step']} rank {r['rank']} ({r['kind']}): slots {r['illegal_writes']}"
        for r in runs
        if r["illegal_writes"]
    ]
    steps = {r["step"] for r in runs}
    sizes = sorted({r["graph_size"] for r in runs if r["graph_size"]})
    worst = [r["worst"] for r in runs if r["worst"]]
    summary = (
        ", ".join(f"{key} {max(w[key] for w in worst):.4g}" for key in J.WORST_KEYS)
        if worst
        else "none"
    )
    flips = [m for w in worst for m in w["flip_margins"]]
    print(
        f"{group}: {len(steps)} steps, {len(runs) // J.WORLD_SIZE} lockstep runs, "
        f"{compared} probe rows ({compared * J.TOKENS} tokens) compared; worst {summary}; "
        f"{len(flips)} top-1 changes at decode margins {[round(m, 4) for m in flips]}; "
        f"{sum(r['moved'] for r in runs)} probe rows move from their verify alone, max |logit "
        f"diff| {max(r['moved_logit'] for r in runs):.4f}"
        + (f", graph sizes {sizes}" if sizes else "")
    )
    assert compared > 0, f"{group}: no probe row was compared"
    assert not failures, "\n".join(failures[:40])
    assert not leaks, "runs wrote slots their rows do not own:\n" + "\n".join(leaks[:20])


@pytest.mark.gpu4
def test_commit_promotes_the_state_the_verify_stored(measured) -> None:
    """After a padded graph verify (15 rows in the graph of 16 on every rank, then with the last
    rank idle: the attention-DP dummy and padding), the worker's commit (`update_mamba_states` with
    the engine's own slots, the dummies' included) leaves in each probe's slot exactly the state
    the verify stored for its last accepted token -- one, two or three -- in every GDP layer: its
    decode steps' bits before the first attention layer, within the state bands after it. It
    writes no slot outside the step's rows'."""
    records = measured["commit"]
    assert len(records) == 2 * J.WORLD_SIZE, len(records)
    failures = [f for r in records for f in r["failures"]]
    outside = [r for r in records if r["outside"]]
    for r in records:
        print(
            f"commit rank {r['rank']}{' (idle)' if r['idle'] else ''}: {r['rows']} rows, probes "
            f"accepted {r['accepted']}, real slots written {len(r['real_written'])}, recurrent / "
            f"conv state {r['ssm']:.3g} / {r['conv']:.3g} of max |decode|"
        )
    assert {n for r in records for n in r["accepted"]} == {1, 2, 3}
    assert not failures, "\n".join(failures[:40])
    assert not outside, f"the commit wrote slots outside the step's rows: {outside[:4]}"
    busy = [r for r in records if not r["idle"]]
    assert all(len(r["real_written"]) == r["rows"] for r in busy), busy


@pytest.mark.gpu4
def test_verify_at_the_length_limit(measured) -> None:
    """A request whose last accepted token is its `MAX_SEQ_LEN - 1`-th, the longest history a
    verify starts from, verifies two drafts on every rank: its lengths reach `MAX_SEQ_LEN + 1`.
    The cache manager's `max_seq_len` -- which thop_attention is handed as its bound and as its
    attention window -- counts the drafts, so the window holds every key; a window of the bare
    `MAX_SEQ_LEN` would drop the oldest. The verify, alone and in a padded graph beside two
    probes, keeps each token within the bands of its decode step."""
    records = measured["limit"]
    assert len(records) == J.WORLD_SIZE
    for r in records:
        print(
            f"limit rank {r['rank']}: lengths up to {r['longest']}, max_seq_len {r['bound']}, "
            f"attention window {r['window']} (net {J.MAX_SEQ_LEN}), {r['compared']} rows compared"
        )
        assert r["longest"] == J.MAX_SEQ_LEN + J.DRAFT - 1, r
        assert min(r["bound"], r["window"]) >= r["longest"] > J.MAX_SEQ_LEN, r
        assert r["finite"], r
        assert r["compared"] == 4, r
        assert not r["failures"], "\n".join(r["failures"][:40])


@pytest.mark.gpu4
def test_the_exact_comparison_moves_with_cublas(measured) -> None:
    """The bitwise half has teeth on this model's own state: with the GEMMs that run through
    `invariant_mm` put back on `cublas_mm` -- what the verify ran before it was made invariant --
    an eager verify batch of 64 rows on every rank (192 GEMM rows, where cuBLAS splits the dense
    MLPs of layers 1 and 3 otherwise than at one row) moves some probe's state before the first
    attention layer, or its K/V in that layer, off that rival's own decode steps alone."""
    records = measured["rivals"]
    assert len(records) == J.WORLD_SIZE
    compared = sum(r["compared"] for r in records)
    differ = [d for r in records for d in r["differ"]]
    # Each line names one token ("... at row R: token J: what differs"), several per token.
    tokens = {re.match(r"^(.*?: token \d+): ", line).group(1) for line in differ}
    print(f"rival cublas_mm: {len(tokens)} of {compared} verify tokens differ before attention")
    for line in differ[:24]:
        print(f"  {line}")
    assert differ, (
        "with cublas_mm back in the verify nothing before attention moved, so the bitwise half "
        "cannot tell an invariant verify from a variant one here"
    )


@pytest.mark.gpu4
def test_the_band_tells_a_state_one_token_off(measured) -> None:
    """The bands have teeth: a verify that starts from the state one token too far -- every GDP
    slot holding the state after t0 already, as a rollback that kept a rejected draft would leave
    it -- moves every probe's logits by more than twice the logit band, and one of its top gaps by
    more than the gap band, each probe alone on its rank."""
    records = measured["teeth"]
    assert len(records) == J.WORLD_SIZE
    for r in records:
        for line in r["lines"]:
            print(line)
    logit = min(r["logit"] for r in records)
    gap = min(r["gap"] for r in records)
    print(f"one token off: at least {logit:.3f} / {gap:.3f}, bands {J.LOGIT_BAND} / {J.GAP_BAND}")
    assert logit > 2 * J.LOGIT_BAND and gap > J.GAP_BAND, (logit, gap)


@pytest.mark.gpu4
def test_the_drift_over_consecutive_verifies(measured) -> None:
    """MTP on along MTP off's own greedy tokens, as far as dep4's MTP identity gate generates:
    each probe's 255 or 1023 tokens after t0 decoded one a step (MTP off), then verified three a
    step from the same state, alone on its rank in the padded graph of one (as the identity gate
    runs a prompt) with every draft accepted and its state committed (MTP on). At every matched
    position the verify's logits stay within the drift bands of the decode step's, and every top-1
    change sits at a decode margin within the gap band."""
    records = measured["drift"]
    assert len(records) == J.WORLD_SIZE
    for r in records:
        for line in r["lines"]:
            print(f"drift {line}")
    flips = [f for r in records for f in r["flips"]]
    print(
        f"drift: max |logit diff| {max(r['logit'] for r in records):.4f}, max top-gap shift "
        f"{max(r['gap'] for r in records):.4f} over {sum(r['positions'] for r in records)} "
        f"positions; top-1 changes (probe, position, decode margin): {flips}"
    )
    failures = [f for r in records for f in r["failures"]]
    assert not failures, "\n".join(failures[:40])


@pytest.mark.gpu4
def test_the_mtp_layer_drafts_as_prefill_drafts(measured) -> None:
    """The MTP layer's draft step runs the same kernels in the decode target as in the prefill
    target -- cuBLAS on `_mtp_view`'s `.t()` views, thop_attention, its MoE across the ranks sized
    by the worker's counts -- so on a draft step of every rank's probes (t1 after the trunk's
    hidden state of t0, over the MTP layer's KV of the prompt) the decode target's output, through
    `MTPLayer`'s routing, equals the prefill target's bit for bit. The decode target's
    row-invariant `_moe` reads the stored [out, in] weights, so handing it the MTP layer's views
    misreads them; its guard says so, on every rank, before the step's first collective."""
    records = measured["draft"]
    assert len(records) == J.WORLD_SIZE
    agree = sum(r["drafts_equal_t2"] for r in records)
    rows = sum(r["rows"] for r in records)
    print(f"draft step: {agree} of {rows} drafts equal the trunk's greedy t2")
    for r in records:
        assert r["finite"], r
        assert r["decode_equals_prefill"], (
            f"rank {r['rank']}: the decode target's MTP layer differs"
        )
        assert r["routed_equals_decode"], (
            f"rank {r['rank']}: MTPLayer did not route to DecodeTarget"
        )
        assert "x and weight must share K" in r["hazard_caught"], r
