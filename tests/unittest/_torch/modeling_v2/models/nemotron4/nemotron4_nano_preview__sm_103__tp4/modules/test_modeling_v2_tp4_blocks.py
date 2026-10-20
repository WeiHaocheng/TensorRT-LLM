# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of the tp4 target's tensor-parallel blocks, on the checkpoint's own weights and
the model's own activations, over the four GPUs of one node.

What runs is the tp4 target itself on every rank -- its own forward, with the engine's hybrid
cache manager and attention metadata, over the fixed prompt set in the shapes a served engine
runs: varlen context batches, two ~8K prompts unchunked, a long prompt in 2048-token chunks,
chunks beside generation rows, and decode at batch 1, 4, 16 and 64 over histories up to ~8K.
The measurement runs once, as a 4-rank MPI job (`_tp4_blocks_job.py`, which says what each rank
records); this module starts it, reads the records and gates them. The blocks: the GDP mixer
(6 of 24 heads per rank, out_proj row-parallel), attention (4 of 16 query heads over the whole
KV head, o_proj row-parallel), the dense MLP and the shared expert (column / row parallel), the
routed experts (every expert's intermediate sliced, the latent partial all-reduced with the
shared partial before `fc2_norm`), and the lm_head, which the target keeps whole on every rank:
there is no vocab-parallel head or logits all-gather in it to measure.

The bands are the tp1 module tests' (the GDP, attention / dense-MLP and MoE block tests), plus
the TP rounding term. Every all-reduce here sums bf16 partials -- each rank's GEMM rounds its
K-slice's fp32 sum to bf16, and the all-reduce adds the four in fp32, in rank order, rounding
once -- where a single GPU rounds the full-K sum once. The term is measured as the target
against its twin: the same block with the all-reduce replaced by the one full-K (or
full-width) call on the ranks' gathered inputs. The gates hold the composition apart:

* the all-reduce is exact: its output is the fp32 rank-order sum of the four gathered partials,
  bit for bit, and every rank holds the same bits of every replicated output
  (`test_every_allreduce_is_its_certified_sum_on_every_rank`) -- which is also what catches a
  reduction that rounds as it goes, a mistake no module band can see;
* the TP term is rounding: at every row-parallel GEMM's all-reduce, each element of the target
  against its twin lies within the rounding of its four partials and the two sums
  (`test_the_tp_term_is_within_the_rounding_of_its_partials`), and its size is the band derived
  from that rounding (`TP_BANDS`, `test_the_tp_term_is_within_its_band`);
* the twin against the reference, at the tp1 bands: tp4's per-rank ops composed, without the TP
  rounding, are as accurate as tp1's block (`test_the_twin_is_within_the_tp1_bands`);
* the target against the reference, at the tp1 band plus the TP band where the all-reduce feeds
  the output (`test_each_block_is_within_the_tp1_band_plus_the_tp_term`,
  `test_the_moe_block_is_within_the_tp1_bands_plus_the_tp_term`).

Bands are in bf16 ulps (2^-8) of a row's scale -- its largest |reference|, or for a stream write
the largest stream magnitude the write reads or writes -- as (worst element, relative RMS); the
RMS of the TP term is tiered at `K.MANY_ROWS` rows, where a batch's relative RMS becomes a stable
statistic. Run with `-rP` to see the measured figures.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../.."]
import _nemotron4_moe_block as K  # noqa: E402 — needs the path declared above
import _tp4_blocks_job as J  # noqa: E402 — beside this file

# Each GPU test also carries `pytest.mark.gpu4`, which this tree (`utils/util.py`, imported by
# its conftest) turns into a decorator that skips below four GPUs, so it cannot sit in this list.
pytestmark = pytest.mark.no_xdist

_JOB = Path(__file__).resolve().with_name("_tp4_blocks_job.py")

KINDS = ("gdp", "attention", "mlp", "moe")


# ----------------------------------------------------------------------------------------------
# Bands
# ----------------------------------------------------------------------------------------------

OP_BANDS = {
    # wide_residual's normed reads against HF's two-rounding order: within one rounding flip of
    # the normalized value (the GDP test's 4 / 0.5; the attention / MLP test's 4 / 0.25).
    ("gdp", "read"): (4.0, 0.5),
    ("attention", "read"): (4.0, 0.25),
    ("mlp", "read"): (4.0, 0.25),
    ("moe", "routed_read"): (4.0, 0.25),
    ("moe", "shared_read"): (4.0, 0.25),
    ("gdp", "next_read"): (4.0, 0.25),
    ("attention", "next_read"): (4.0, 0.25),
    ("mlp", "next_read"): (4.0, 0.25),
    ("moe", "next_read"): (4.0, 0.25),
    # A rank's GEMM on its own input against the IEEE-fp32 GEMM on the rank's slice of the
    # checkpoint weight: both accumulate in fp32, in different orders, and round once, so they
    # differ by rounding flips (the attention / MLP test's 4 / 0.5; GDP's in_proj at cublas_mm's
    # 4 / 2, as the GDP test gated it).
    ("gdp", "in_proj"): (4.0, 2.0),
    ("gdp", "out_proj"): (4.0, 0.5),
    ("attention", "qkv"): (4.0, 0.5),
    ("attention", "o_proj"): (4.0, 0.5),
    ("mlp", "up"): (4.0, 0.5),
    ("mlp", "down"): (4.0, 0.5),
    ("moe", "shared_up"): (4.0, 0.5),
    ("moe", "shared_down"): (4.0, 0.5),
    # thop_attention on the rank's own q and the K / V it wrote, against fp32 softmax attention
    # (the attention test's op band; the entry's is 8 / 4).
    ("attention", "attention"): (4.0, 1.5),
    # latent_moe_experts on the rank's slice of every expert, bf16 out, against the entry's own
    # reference on the same arguments: the entry's certified band.
    ("moe", "experts_partial"): (6.0, 0.4),
    # Bit-exact: the soft-clamped relu2 matches HF's ops on every bf16 pattern, and a stream
    # write is HF's fp32 r * S + w * U rounded once.
    ("mlp", "act"): (0.0, 0.0),
    ("moe", "shared_act"): (0.0, 0.0),
    ("gdp", "write"): (0.0, 0.0),
    ("attention", "write"): (0.0, 0.0),
    ("mlp", "write"): (0.0, 0.0),
    ("moe", "write"): (0.0, 0.0),
}
"""Each op of a rank on the rank's own input to it (`op`), against the reference op on the rank's
slice of the checkpoint weights."""

TP1_BANDS = {
    # The attention and dense-MLP blocks' bands on tp1 (their test's MODULE_BANDS).
    ("attention", "read"): (4.0, 0.25),
    ("attention", "attention"): (4.0, 1.5),
    ("attention", "update"): (4.0, 2.0),
    ("attention", "write"): (4.0, 1.0),
    ("attention", "next_read"): (8.0, 2.0),
    ("mlp", "read"): (4.0, 0.25),
    ("mlp", "up"): (4.0, 0.5),
    ("mlp", "act"): (8.0, 0.5),
    ("mlp", "update"): (4.0, 1.0),
    ("mlp", "write"): (4.0, 1.0),
    ("mlp", "next_read"): (8.0, 1.0),
    # The GDP block's on tp1 (its test's `_BANDS`): the mixer output and its out_proj input at the
    # chunked prefill entry's output band, the loosest entry in the path.
    ("gdp", "h"): (32.0, 2.0),
    ("gdp", "update"): (32.0, 2.0),
    # The GDP and MoE writes and the reads after them, which their tp1 tests did not measure, take
    # the attention test's derivation: a write of an update within its band, judged against the
    # operands' magnitude; the next block's normed read of the written streams (the norm divides
    # by the read's RMS, which lifts the streams' errors where the three streams partly cancel).
    ("gdp", "write"): (4.0, 1.0),
    ("gdp", "next_read"): (8.0, 2.0),
    ("moe", "write"): (4.0, 1.0),
    ("moe", "next_read"): (8.0, 2.0),
}
"""tp1's band of each block output, for the twin and, with the TP term's, for the target. The MoE
block's own are the family helper's (`K.BLOCK_BANDS`, `K.HF_ROUNDING_BANDS`, `K.STAGE_BANDS`)."""

GDP_STATE_BANDS = {
    # The recurrent state of the rank's heads against the reference fed the target's in_proj
    # output: the chunked prefill entry's state band, element errors scored against each state
    # column's operand scale (a Householder step can leave a column far below the values its
    # rounding was made on). Against the reference on its own in_proj, the RMS alone: one bf16
    # step that in_proj's summation order flips in a recurrence input moves single state elements
    # by a large share of themselves. Measured: at most 7.31 / 0.87, at layer 42 in the short
    # prompts' fresh varlen prefill (5.52 at layer 6, 2.9 at layer 0, the layers the tp1 test
    # measured).
    ("module@in_proj", "state"): (8.0, 2.0),
    ("module", "state"): (math.inf, 2.0),
    # The conv state is a copy of the last three conv inputs: in_proj outputs. Bit-exact against
    # the target's own in_proj; cublas_mm's band against the reference's.
    ("module@in_proj", "conv_state"): (0.0, 0.0),
    ("module", "conv_state"): (4.0, 2.0),
}

COLPAR_BAND = (2.0, 0.25)
"""A column-parallel GEMM's output (a rank's rows of the weight) against the same columns of the
full GEMM on one GPU: the same dot products, accumulated in fp32 in the order each cuBLAS kernel
picks and rounded once, so at most one rounding flip of an element, 2 ulp of its row's largest.
Measured: bitwise from 64 rows; at most 1.46 / 0.115 below (op-tp4-cells-mm saw the same at one
row)."""

# The TP term: the target against its twin.
#
# Where it is made. Rank r rounds the fp32 sum P_r of its K-slice to the bf16 partial p_r; the
# all-reduce rounds p_0 + p_1 + p_2 + p_3 (fp32, rank order) once; one GPU rounds P_0 + ... + P_3
# once. The difference is six roundings, each at most half a bf16 spacing of its own value --
# which `test_the_tp_term_is_within_the_rounding_of_its_partials` checks element by element.
#
# Its RMS. A bf16 rounding of a value whose mantissa is spread log-uniformly has a relative RMS
# error of 0.425 ulp (2^-8): spacing / value is 2^-7 / m for mantissa m in [1, 2), and
# E[(2^-7 / m)^2] / 12 = 2^-14 * 0.541 / 12. With q = sqrt(sum ||p_r||^2) / ||sum|| -- how much
# the partials cancel, measured at every all-reduce (`partials_over_sum`) -- the six independent
# roundings give a relative RMS of 0.425 * sqrt(q^2 + 2) ulp. From 64 rows the measurement sits
# at 0.74-1.15x that (median 0.90-1.08) at every all-reduce, with q at most 1.47 (attention): at
# most 0.87 by the model, 0.83 measured. Below 64 rows the statistic belongs to a few elements:
# every block's output has a channel (47) of |u| ~ 30-220, and one rounding flip of a row's
# largest element is a spacing, up to 2^-7 of it, so in a row that element carries it moves the
# row's relative RMS by up to 2 ulp by itself -- with the rest of the row's rounding (<= 0.9),
# sqrt(2^2 + 0.9^2) = 2.2. Measured below 64 rows: at most 1.49 (GDP, one row).

_TP_ALLREDUCE = {True: (6.0, 1.2), False: (6.0, 2.2)}
"""At an all-reduce's output (GDP / attention / MLP `update`, MoE `latent_sum` / `shared_sum`),
keyed by rows >= `K.MANY_ROWS`. Element 6: the six roundings are at most one ulp of the row's
largest each where no partial exceeds it (measured: at most 3.05, MoE latent). RMS 1.2 from 64
rows (1.4x the model's 0.87 at the largest q; measured 0.83) and 2.2 below (measured 1.49)."""

_TP_WRITE = {True: (8.0, 1.2), False: (8.0, 2.2)}
"""At the write of an all-reduced update into the streams, judged against the operands'
magnitude: the update's TP term times the write factor (<= 2), against streams at least as large
as the update's contribution, plus one flip of the write's own rounding where the two updates
straddle a boundary (2 ulp). Measured: at most 2.68 / 0.66 from 64 rows, 2.01 / 1.45 below."""

_TP_READ = {True: (12.0, 1.8), False: (12.0, 2.2)}
"""At the next block's normed read of the written streams: the norm divides by the read's RMS,
which lifts the streams' errors where the three streams partly cancel in the read -- tp1's bands
double the element there, and the RMS measures up to 1.44x the update's. Measured: at most
5.86 / 0.98 from 64 rows (GDP), 2.98 / 1.54 below."""

_TP_MOE_BLOCK = {True: (12.0, 1.8), False: (12.0, 2.2)}
"""At the MoE block's output: the latent sum's TP term through `fc2_norm` (which keeps relative
error) and `fc2_latent_proj` (768 differences add up in each output), plus the shared sum's:
sqrt(0.77^2 + 0.74^2) = 1.07 RMS by the two measured halves; measured 1.24 from 64 rows, 1.42
below, element at most 6.02. Element 12 (2x), RMS 1.8 (1.45x) and 2.2 (1.55x)."""

TP_BANDS = {
    ("gdp", "update"): _TP_ALLREDUCE,
    ("attention", "update"): _TP_ALLREDUCE,
    ("mlp", "update"): _TP_ALLREDUCE,
    ("moe", "latent_sum"): _TP_ALLREDUCE,
    ("moe", "shared_sum"): _TP_ALLREDUCE,
    ("moe", "block"): _TP_MOE_BLOCK,
    **{(kind, "write"): _TP_WRITE for kind in KINDS},
    **{(kind, "next_read"): _TP_READ for kind in KINDS},
}
"""The TP term's band at every output an all-reduce feeds, keyed by rows >= `K.MANY_ROWS`."""


# ----------------------------------------------------------------------------------------------
# The job and its records
# ----------------------------------------------------------------------------------------------


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
    """Every rank's records, from one run of the 4-rank job (also written to the path
    `NEMOTRON4_TP4_BLOCKS_RECORDS` names, when set)."""
    checkpoint = _checkpoint_or_skip()
    devices = _devices_or_skip()
    out = tmp_path_factory.mktemp("tp4_blocks") / "records.json"
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
    print(completed.stdout[-6000:])
    text = out.read_text()
    keep = os.environ.get("NEMOTRON4_TP4_BLOCKS_RECORDS")
    if keep:
        Path(keep).write_text(text)
    records = json.loads(text)
    # The family helper's MoE records state no row count; their step's routing record does.
    rows = {
        (r["step"], r["rank"], r["layer"]): r["rows"]
        for r in records["records"]
        if r["comparison"] == "routing"
    }
    for r in records["records"]:
        if "rows" not in r:
            r["rows"] = rows.get((r["step"], r["rank"], r["layer"]), 0)
    return records


def _records(measured: dict, **match) -> list[dict]:
    """The records whose fields equal `match`."""
    return [r for r in measured["records"] if all(r[k] == v for k, v in match.items())]


def _base(output: str) -> str:
    """An output's name without its read index (`next_read1` -> `next_read`)."""
    return "next_read" if output.startswith("next_read") else output


def _sum(*bands) -> tuple[float, float]:
    """Bands composed by the triangle inequality: element and RMS each add."""
    return tuple(sum(b[k] for b in bands) for k in range(2))


def _many(r: dict) -> bool:
    return r["rows"] >= K.MANY_ROWS


def _breaks(records: list[dict], band_of) -> list[str]:
    """Every record outside its band (`band_of(record)` -> (element ulp, RMS ulp)), described with
    where its worst element sits."""
    out = []
    for r in records:
        element, rms = band_of(r)
        if r["element_ulp"] > element or r["rms_ulp"] > rms:
            out.append(
                f"{r['kind']} layer {r['layer']} {r['comparison']} {r['output']} (rank "
                f"{r['rank']}, step {r['step']} {r['phase']}, {r['rows']} rows): element "
                f"{r['element_ulp']:.3f} (band {element}), RMS {r['rms_ulp']:.3f} (band {rms}); "
                f"worst {r['worst']}"
            )
    return out


def _worst(records: list[dict]) -> str:
    """The largest element and RMS over `records`, for the printed figures."""
    if not records:
        return "-"
    return (
        f"{max(r['element_ulp'] for r in records):.3f} / "
        f"{max(r['rms_ulp'] for r in records):.3f} ({len(records)} records)"
    )


def _report(records: list[dict]) -> None:
    """Print the worst figures per (comparison, output) and tier."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["comparison"], _base(r["output"]), _many(r))].append(r)
    for (comparison, output, many), rs in sorted(groups.items()):
        tier = ">= 64 rows" if many else "< 64 rows"
        print(f"{comparison:24s} {output:28s} {tier:10s} {_worst(rs)}")


#: Outputs an all-reduce feeds: they carry the TP term. Every other output of a block is a rank's
#: own column-parallel or head-local work, whose band is tp1's alone.
_AFTER_ALLREDUCE = ("update", "write", "next_read")


def _module_band(kind: str, r: dict) -> tuple[float, float]:
    """The target's band at a block output: tp1's, plus the TP term's where an all-reduce feeds
    it."""
    output = _base(r["output"])
    base = TP1_BANDS[(kind, output)]
    if output in _AFTER_ALLREDUCE:
        return _sum(base, TP_BANDS[(kind, output)][_many(r)])
    return base


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------


@pytest.mark.gpu4
def test_the_job_covers_every_block_phase_and_comparison(measured) -> None:
    """Every block under test was measured on every rank, in every phase, in each comparison."""
    meta, steps = measured["meta"], measured["steps"]
    print(json.dumps(meta))
    for step in steps:
        print(json.dumps(step))
    assert not meta["quick"], "the records come from the quick shakeout, not the campaign"
    assert meta["layers"] == {k: list(v) for k, v in J.LAYERS.items()}
    assert {s["phase"] for s in steps} == set(J.PHASES)
    seen = defaultdict(set)
    for r in measured["records"]:
        seen[(r["kind"], r["layer"], r["comparison"])].add((r["phase"], r["rank"]))
    every = {(phase, rank) for phase in J.PHASES for rank in range(J.WORLD_SIZE)}
    expected = {
        "gdp": ("op", "colpar", "tp", "twin", "twin@in_proj", "module", "module@in_proj", "flag"),
        "attention": ("op", "colpar", "tp", "twin", "module", "flag"),
        "mlp": ("op", "colpar", "tp", "twin", "module", "flag"),
        "moe": ("op", "colpar", "tp", "twin", "module", "block", "stage", "routing", "flag"),
    }
    missing = [
        (kind, layer, comparison, sorted(every - seen[(kind, layer, comparison)])[:4])
        for kind, comparisons in expected.items()
        for layer in J.LAYERS[kind]
        for comparison in comparisons
        if every - seen[(kind, layer, comparison)]
    ]
    assert not missing, f"never measured: {missing[:6]} ({len(missing)} in all)"
    assert every <= seen[("lm_head", -1, "op")]
    # The chunked and mixed phases attend over KV their earlier chunks wrote; decode reaches 64.
    assert max(s["max_history"] for s in steps if s["phase"] == "chunked") >= 3 * J.CHUNK
    assert max(s["max_history"] for s in steps if s["phase"] == "mixed") > 0
    assert max(s["rows"] for s in steps if s["phase"] == "decode_b64") == J.MAX_BATCH
    assert max(s["tokens"] for s in steps if s["phase"] == "context") > 7500


@pytest.mark.gpu4
def test_every_allreduce_is_its_certified_sum_on_every_rank(measured) -> None:
    """Every all-reduce's output is `allreduce.reference` of the four gathered bf16 partials --
    their fp32 sum in rank order, rounded once -- bit for bit; every rank holds the same bits of
    every replicated output (the all-reduces', the reads after them, the MoE block's, the
    logits); the MoE's packed call carries the two partials end to end and `fc2_norm` and the
    add read its two halves; lm_head reads the final hidden and the forward returns its GEMM.
    Each rival reduction differs from the certified sum."""
    flags = [
        r
        for r in _records(measured, comparison="flag")
        if not r["output"].startswith("tp_within_rounding")
    ]
    counts = defaultdict(lambda: [0, 0])
    broken = []
    for r in flags:
        rival = r["output"].startswith("rival_")
        counts[(r["kind"], r["output"])][r["ok"]] += 1
        if r["ok"] == rival:
            broken.append(
                f"{r['kind']} L{r['layer']} {r['output']} rank {r['rank']} step {r['step']}"
            )
    for key, (bad, good) in sorted(counts.items()):
        print(f"{key[0]:9s} {key[1]:45s} ok {good} of {good + bad}")
    assert {kind for kind, _ in counts} == set(KINDS) | {"lm_head"}
    assert not broken, f"{len(broken)} flags broken: {broken[:8]}"


@pytest.mark.gpu4
def test_the_tp_term_is_within_the_rounding_of_its_partials(measured) -> None:
    """At every row-parallel GEMM's all-reduce (GDP out_proj, attention o_proj, MLP down, the
    shared expert's down), every element of the target against its single-GPU twin lies within
    the four partials' and the two sums' roundings (`_tp4_blocks_job._Campaign.within_rounding`):
    the TP term is rounding and nothing else."""
    records = [
        r
        for r in _records(measured, comparison="flag")
        if r["output"].startswith("tp_within_rounding")
    ]
    assert {r["kind"] for r in records} == set(KINDS)
    worst = defaultdict(float)
    elements = defaultdict(int)
    for r in records:
        key = (r["kind"], r["output"])
        worst[key] = max(worst[key], r["max_share_of_bound"])
        elements[key] += r["numel"]
    for key, share in sorted(worst.items()):
        print(
            f"{key[0]:9s} {key[1]:32s} {elements[key]} elements; largest share of the bound {share:.3f}"
        )
    beyond = [r for r in records if not r["ok"]]
    assert not beyond, [
        (r["kind"], r["layer"], r["output"], r["step"], r["rank"], r["beyond"]) for r in beyond[:8]
    ]


@pytest.mark.gpu4
@pytest.mark.parametrize("kind", KINDS)
def test_the_tp_term_is_within_its_band(measured, kind) -> None:
    """The target against its twin -- the same block with the all-reduce replaced by the single
    full-K or full-width call on the ranks' gathered inputs -- at the all-reduce's output and at
    every output after it (`TP_BANDS`). Also printed: the RMS against the rounding model at the
    all-reduce, with the step's measured cancellation q."""
    records = _records(measured, kind=kind, comparison="tp")
    assert records
    _report(records)
    q = {
        (r["layer"], r["step"], r["rank"]): r["partials_over_sum"]
        for r in _records(measured, kind=kind, comparison="flag")
        if "partials_over_sum" in r
    }
    site = "latent_sum" if kind == "moe" else "update"
    for many in (True, False):
        ratios = [
            r["rms_ulp"] / (0.425 * math.sqrt(q[(r["layer"], r["step"], r["rank"])] ** 2 + 2))
            for r in records
            if r["output"] == site and _many(r) == many
        ]
        if ratios:
            print(
                f"{site} {'>= 64' if many else '< 64'} rows: RMS / (0.425 sqrt(q^2 + 2)) "
                f"{min(ratios):.3f} .. {max(ratios):.3f}"
            )
    breaks = _breaks(records, lambda r: TP_BANDS[(kind, _base(r["output"]))][_many(r)])
    assert not breaks, "\n".join(breaks[:20])


@pytest.mark.gpu4
@pytest.mark.parametrize("kind", KINDS)
def test_each_rank_op_is_within_its_band(measured, kind) -> None:
    """Each op of every rank, on the rank's own input to it, against the reference op on the
    rank's slice of the checkpoint weights (`OP_BANDS`)."""
    records = _records(measured, kind=kind, comparison="op")
    assert records
    _report(records)
    unbanded = {r["output"] for r in records if (kind, _base(r["output"])) not in OP_BANDS}
    assert not unbanded, f"ops without a band: {sorted(unbanded)}"
    breaks = _breaks(records, lambda r: OP_BANDS[(kind, _base(r["output"]))])
    assert not breaks, "\n".join(breaks[:20])


@pytest.mark.gpu4
def test_column_parallel_gemms_give_the_full_gemms_columns(measured) -> None:
    """A rank's column-parallel GEMM (GDP in_proj, the fused qkv, MLP up, the shared expert's
    up) against the same columns of the full GEMM on one GPU (`COLPAR_BAND`)."""
    records = _records(measured, comparison="colpar")
    assert {r["kind"] for r in records} == set(KINDS)
    _report(records)
    breaks = _breaks(records, lambda r: COLPAR_BAND)
    assert not breaks, "\n".join(breaks[:20])


@pytest.mark.gpu4
@pytest.mark.parametrize("kind", KINDS)
def test_the_twin_is_within_the_tp1_bands(measured, kind) -> None:
    """tp4's per-rank work composed without the TP rounding -- the twin -- against the reference
    block, at tp1's bands: a rank's slices and heads compute what one GPU computes for them."""
    records = [
        r for r in _records(measured, kind=kind) if r["comparison"] in ("twin", "twin@in_proj")
    ]
    assert records
    _report(records)

    def band(r):
        if r["output"] == "block vs R_t":
            return K.BLOCK_BANDS[_many(r)]
        if r["output"] == "block vs R_hf":
            return K.HF_ROUNDING_BANDS[_many(r)]
        return TP1_BANDS[(kind, _base(r["output"]))]

    breaks = _breaks(records, band)
    assert not breaks, "\n".join(breaks[:20])


@pytest.mark.gpu4
@pytest.mark.parametrize("kind", ("gdp", "attention", "mlp"))
def test_each_block_is_within_the_tp1_band_plus_the_tp_term(measured, kind) -> None:
    """The target's block, teacher-forced at the streams it read, against the reference block:
    tp1's band, plus the TP term's at every output an all-reduce feeds (`_module_band`). GDP also
    against the reference fed the target's own in_proj output (`module@in_proj`)."""
    records = [
        r
        for r in _records(measured, kind=kind)
        if r["comparison"] in ("module", "module@in_proj")
        and r["output"] not in ("state", "conv_state")
    ]
    assert records
    _report(records)
    breaks = _breaks(records, lambda r: _module_band(kind, r))
    assert not breaks, "\n".join(breaks[:20])


@pytest.mark.gpu4
def test_the_gdp_states_are_within_their_bands(measured) -> None:
    """The recurrent and conv state each tracked request left in every rank's pools -- its heads
    and its channels -- against the references' carried states (`GDP_STATE_BANDS`)."""
    records = [r for r in _records(measured, kind="gdp") if r["output"] in ("state", "conv_state")]
    assert {(r["comparison"], r["output"]) for r in records} == set(GDP_STATE_BANDS)
    _report(records)
    breaks = _breaks(records, lambda r: GDP_STATE_BANDS[(r["comparison"], r["output"])])
    assert not breaks, "\n".join(breaks[:20])


#: Stage records that satisfy `K.failures`' stage gates, so that it can be asked about routing.
_PASSING_STAGES = {
    name: dict(element_ulp=0.0, rms_ulp=0.0, differing=0, numel=1)
    for name in ("fc1", "experts", "fc2_norm", "tail")
}


@pytest.mark.gpu4
def test_the_moe_routing_is_hfs_router(measured) -> None:
    """The replicated router on every rank against `Nemotron4TopkRouter` on IEEE fp32 logits:
    the family helper's routing gates, unchanged -- the router and the selection run before any
    tensor-parallel arithmetic, on every rank alike."""
    records = _records(measured, kind="moe", comparison="routing")
    assert records
    broken = []
    for r in records:
        found = K.failures(dict(routing=r, block={}, stages=_PASSING_STAGES))
        if found:
            broken.append((r["layer"], r["phase"], r["step"], r["rank"], found))
    flips = [
        (r["layer"], r["step"], m, b)
        for r in records
        if r["rank"] == 0
        for m, b in zip(r["flip_margins"], r["flip_bounds"])
    ]
    print(
        f"rows per rank {sum(r['rows'] for r in records if r['rank'] == 0)}; flips (layer, step, "
        f"margin, bound) {flips}; thinnest margin {min(r['thinnest_margin'] for r in records):.3g}"
    )
    assert not broken, broken[:8]


@pytest.mark.gpu4
def test_the_moe_block_is_within_the_tp1_bands_plus_the_tp_term(measured) -> None:
    """The MoE block on every rank against the family's references (`R_t` rounds where the
    target rounds; `R_hf`, `H` and `R_own` where HF rounds) and each stage on the target's own
    input: the family's bands, plus the TP term's where the packed all-reduce feeds the output --
    the block itself, its tail (the shared half), and the `experts` stage, which is the
    all-reduced latent sum against the full-width reference loop. The block's write and the reads
    after it at `_module_band`."""
    records = [
        r for r in _records(measured, kind="moe") if r["comparison"] in ("block", "stage", "module")
    ]
    _report(records)
    breaks = []
    for r in records:
        many = _many(r)
        if r["comparison"] == "block":
            if r["output"] == "R_t vs R_hf":  # fc2_norm's single vs double rounding: reported
                continue
            if r["output"] == "H vs R_hf":  # HF's own bf16 block: the family's band as is
                band = K.BLOCK_BANDS[many]
            elif r["output"] == "T vs R_t":
                band = _sum(K.BLOCK_BANDS[many], TP_BANDS[("moe", "block")][many])
            else:
                band = _sum(K.HF_ROUNDING_BANDS[many], TP_BANDS[("moe", "block")][many])
        elif r["comparison"] == "stage":
            if r["output"] == "fc2_norm":
                if r["element_ulp"] > 2.0 or r["differing"] > max(1, r["numel"] // 1000):
                    breaks.append(f"stage fc2_norm layer {r['layer']} step {r['step']}: {r}")
                continue
            if r["output"] == "fc2_norm_vs_hf_rounding":
                continue
            band = {
                "fc1": K.STAGE_BANDS["fc1"],
                "experts": _sum(K.STAGE_BANDS["experts"], TP_BANDS[("moe", "latent_sum")][many]),
                "tail": _sum(K.BLOCK_BANDS[many], TP_BANDS[("moe", "shared_sum")][many]),
            }[r["output"]]
        else:
            band = _module_band("moe", r)
        breaks += _breaks([r], lambda _r, band=band: band)
    assert not breaks, "\n".join(breaks[:20])


@pytest.mark.gpu4
def test_the_lm_head_is_the_fp32_gemm_of_the_final_hidden(measured) -> None:
    """lm_head, whole on every rank: its fp32 logits against an IEEE-fp32 GEMM of the same hidden
    rows with the checkpoint's lm_head, in fp32 ulps (the band `cublas_mm` certifies its
    fp32-output cells with, `K.LOGITS_BAND`). That every rank forms the same logits, that the
    forward returns this GEMM and that it reads the final hidden are flags."""
    records = _records(measured, kind="lm_head", comparison="op", output="logits")
    assert records
    print(f"logits: {_worst(records)} fp32 ulp")
    breaks = _breaks(records, lambda r: K.LOGITS_BAND)
    assert not breaks, "\n".join(breaks[:20])


@pytest.mark.gpu4
@pytest.mark.parametrize(
    "kind,rival", [(kind, rival) for kind in KINDS for rival in J.RIVALS[kind]]
)
def test_the_gates_reject_the_tp_rivals(measured, kind, rival) -> None:
    """One plausible tensor-parallel mistake, built from what the ranks really computed: a rank
    keeping its own partial (no all-reduce), its heads or slices paired with the next rank's
    weight columns, `fc2_norm` applied to each rank's latent partial before the sum, or the four
    partials added in bf16 as they come. Each must break its block's band (tp1's plus the TP
    term's) in every step it was planted in -- except the bf16 running sum, a mistake too small
    for any module band, which the bitwise all-reduce gate rejects instead."""
    output = "block vs R_t" if kind == "moe" else "update"
    planted = _records(measured, kind=kind, comparison=f"rival_{rival}", output=output)
    assert planted, f"{kind} rival {rival} was never planted"

    def band(r):
        if kind == "moe":
            return _sum(K.BLOCK_BANDS[_many(r)], TP_BANDS[("moe", "block")][_many(r)])
        return _module_band(kind, r)

    steps = sorted({r["step"] for r in planted})
    caught = {step: _breaks([r for r in planted if r["step"] == step], band) for step in steps}
    print(
        f"{kind} {rival}: {_worst(planted)}; steps caught by the module band "
        f"{[s for s in steps if caught[s]]} of {steps}"
    )
    if rival == "bf16_running_sum":
        flags = _records(measured, kind=kind, comparison="flag", output=f"rival_{rival}:ar_bitwise")
        assert flags and not any(f["ok"] for f in flags), (
            "the bf16 running sum equals the certified sum"
        )
        return
    missed = [step for step in steps if not caught[step]]
    assert not missed, f"{kind} rival {rival} passes the module band at steps {missed}"
