# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of the nemotron4 tp1 target's GDP path, on the checkpoint's real weights.

What runs is the target's own block code: the `_streams_step` read before the mixer,
`PrefillTarget._gdp` or `DecodeTarget._gdp`, and the `_streams_step` write after it, for two
blocks -- layer 0, the first `linear_attention` block, and layer 6, the mixer of the first
`shortcut_moe_gdp` block (its routed MoE is the MoE block test's). Both targets are built by their
own constructors on a stand-in core whose weight mapping loads, through the target's weight table
and its load-time transforms, only the parameters these blocks read. The state slots,
`has_initial_state` and the padding marks come from the target's `Nemotron4StateMetadata`,
prepared on stand-in steps, over pools in the hybrid cache manager's packed per-slot layout whose
every slot starts out holding another request's (non-zero) state.

The inputs are real. Every request is a stretch of the fixed prompt set, and a block's input is
what the plain-torch reference computes for those tokens: the embedding for layer 0, the
reference's layers 0-5 for layer 6.

The scenario, per block (`_STEPS`):

* S0: a 2000-token first prefill chunk of one request; 2000 tokens are 6000 Householder sub-steps,
  so the chunk ends inside a 64-step block of the chunk kernel;
* S1: a varlen batch of five prefills -- 5 tokens, 2 tokens (fewer than the conv's 3-input window),
  the 1500-token continuation of S0 (`has_initial_state`), 89 and 1000 tokens;
* D0-D15: sixteen decode steps of the five requests, eager, states carried;
* the same sixteen decode steps replayed in a CUDA graph of 8 rows, three of them padding;
* G7: one more graph step in which a fresh one-token prompt is promoted into a decode row on the
  slot a finished request left behind.

The reference (`_reference_mixer`) is `_nemotron4_reference.gdp_core` with the short conv rounded
where the target's conv entries round it: SiLU of the fp32 tap sum, rounded once to bf16. The
checkpoint's code rounds the conv output to bf16 and rounds again after SiLU; the gap that makes is
reported as its own rows (`hf-order`), never folded into a band. The reference takes the target's
mixer input (the read is compared on its own), runs the per-token fp32 recurrence of the
checkpoint's loop, the gated norm and out_proj, and carries its own state through the whole
scenario. It runs twice: on its own fp32 in_proj ("reference"), and on the target's in_proj output
("ref@in_proj"), which leaves the path after in_proj alone to compare; the in_proj GEMM is then
compared on its own.

The bands (`_BANDS`) are derived from the entries' own measured errors, before this test ran:

* mixer output u, its out_proj input h and the recurrent state, any step: the chunked prefill is
  the loosest entry in the path. `flashinfer_gdp_chunk` lands within 13.1 bf16 ulps element-wise
  and 1.24 of relative RMS of the fp32 per-token recurrence in its output, and 3.3 / 1.09 in its
  final state (its test's 240 runs at the checkpoint's decay ranges), against the bands it
  certifies, 32 / 2 and 8 / 2.
  Downstream of it the gated norm normalizes each head on its own, which keeps a head's relative
  error, and out_proj is a GEMM. The decode steps start from the chunk-built states: a step
  multiplies the state's error by exp(g) <= 1 and by Householder factors (I - beta k k^T), which
  do not expand it, and adds gdp_recurrent's own fp32-level error. Every other op after in_proj
  rounds where the reference rounds: the conv entries flip 6e-5 of their outputs by one bf16 step,
  the gated norm 4.1e-5, out_proj (cuBLAS, or a low-M kernel within cublas_mm's band in a decode
  step) 0.09% (cublas_mm's measured cells). So the bands are the
  chunk entry's own: 32 / 2 for u and h, 8 / 2 bf16 ulps for the state, against "ref@in_proj".
  Against "reference" u and h keep those bands and the state its RMS band. Its state elements
  are reported, not gated: one bf16 step that in_proj's summation order flips in a recurrence
  input moves single state elements by a large share of themselves, which the reference itself
  shows when fed one in_proj output or the other ("state ref" vs "ref@in_proj": up to 7.35 ulps
  on one element of layer 6, RMS 0.05). There one value element of head 17 differs by a step at
  D8, which moves a state element 0.1367 -> 0.1351; at D9 the element is 0.050 and the difference
  stays.
* the recurrence entries on the block's own inputs, teacher-forced at their call (the inputs each
  call received, run through the entry's own fp32 reference): their certified bands, 32 / 2
  and 8 / 2 (bf16) for `flashinfer_gdp_chunk`, and for `gdp_recurrent` 2 / 0.25 for its output
  and 8 / 1 fp32 ulps for its state -- the decode path alone is that tight.
* the read: `wide_residual`'s normed-read band, 4 / 0.5; the write: bit-exact (it is bit-exact
  against the checkpoint's ops); in_proj: cublas_mm's band, 4 / 2.
* the conv state: a verbatim copy of the last three in_proj outputs of the conv channels, so
  bit-exact against the target's own in_proj, and within cublas_mm's band (4 / 2) of the
  reference's.

Element errors of a recurrent state and of a recurrence output are scored against the row's
operand scale (`_recurrence`), not its final largest value. A Householder step that overwrites
what the token's earlier steps wrote leaves a state column far smaller than it was a sub-step
before; the rounding errors of any implementation are the size of the values it rounded. On
layer 0 head 9, whose keys lean on one dimension (55) with beta near one, a column ends up to 10x
below its in-token peak: against the final largest value the chunk's state reads 19.2 bf16 ulps
there and the fp32 reference itself 25 fp32 ulps from fp64. Against the largest value each column
held, the chunk reads at most 1.94 on every head and the fp32 reference 1.51 on head 9; the
operand scale, that maximum decayed as the state decays, is stricter (3.15 on layer 0). The
final-value figure stays in the table (`elem`), ungated; the band numbers are the entries' own.

The mixer output u carries a massive channel (column 47 reaches magnitudes of 40 to 110 in both
blocks), so its element gate, scaled by the row's largest value, is loose there; h, normalized per
head, and the RMS gates are what see a mistake in the recurrence.

Measured, both blocks, every step (bf16 ulps, element / relative RMS): against "ref@in_proj" u
2.0 / 1.02, h 2.24 / 0.71, the state 3.82 / 0.70; against "reference" the same for u and h and
for the state's RMS, its elements up to 7.57, where the two references differ by 7.35 themselves.
The recurrence entries on their recorded inputs: the chunk entry 0.95 / 0.65 out and 3.82 / 0.61
state, `gdp_recurrent` 0.11 / 0.07 out and 4.24 / 0.50 fp32 ulps state. The graph's replays
read and carry the states bit for bit as the eager steps do, and their mixer output is the eager
step's out_proj input through the GEMM the graph's row count picks; G7's recycled slot gives a
zeroed one's bits. The checkpoint's double-rounded conv moves the reference's own u by up to
1.09 of RMS (`hf-order`).

`test_the_bands_reject_planted_mistakes` plants the nearest mistakes in the reference and shows
each band rejects them.

Run with `-s` to see the tables. `NEMOTRON4_GDP_BLOCK_REPORT=<file>` appends every row as a JSON
line.
"""

import dataclasses
import json
import os
import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_direct_mm import lowm_direct_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_splitk_mm import lowm_splitk_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1.weights import (  # noqa: E501
    MODEL_WEIGHTS,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

__extra_import_path__ = ["../../../..", "../.."]
import _nemotron4_prompts as P  # noqa: E402
import _nemotron4_reference as R  # noqa: E402
from _validating import ULP, validating  # noqa: E402

assert torch.cuda.is_available(), "the GDP block runs the target's CUDA entries"

BF16, FP32 = ULP[torch.bfloat16], ULP[torch.float32]

#: The entries the block calls, guarded by their `is_valid` on every eager step.
_ENTRIES = (
    M.wide_residual,
    M.cublas_mm,
    lowm_direct_mm,
    lowm_splitk_mm,
    M.gdp_gates,
    M.triton_causal_conv1d_varlen,
    M.causal_conv1d_update,
    M.flashinfer_gdp_chunk,
    M.gdp_recurrent,
    M.gated_rmsnorm_multirow,
)

#: The modeling name of the chunked-prefill recurrence entry the prefill target calls.
_CHUNK = "flashinfer_gdp_chunk"
#: The modeling name of the GDP output norm.
_NORM = "gated_rmsnorm_multirow"

#: The blocks under test: the first `linear_attention` block, the first `shortcut_moe_gdp` one.
_LAYERS = (0, 6)


# --------------------------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Row:
    """One request: a stretch of a prompt of the fixed set, prefilled and then decoded."""

    prompt: str
    request_id: int
    slot: int
    first_chunk: tuple[int, int] | None  # an earlier prefill chunk, S0
    prefill: tuple[int, int]  # its tokens in the S1 batch
    decode_tokens: int  # tokens after the prefill: D0-D15, then the G7 step's


_DECODE_STEPS = 16
_ROWS = (
    _Row("gsm8k_multiturn", 101, 7, None, (0, 5), _DECODE_STEPS + 1),
    _Row("gpqa_explain", 102, 2, None, (0, 2), _DECODE_STEPS + 1),
    _Row("gsm8k_fewshot", 103, 9, (0, 2000), (2000, 3500), _DECODE_STEPS + 1),
    _Row("haiku", 104, 4, None, (0, 89), _DECODE_STEPS),
    _Row("news_digest", 105, 11, None, (0, 1000), _DECODE_STEPS + 1),
)
#: The request that finishes after D15; G7's fresh prompt gets its slot.
_FINISHED = 3
#: G7's fresh one-token prompt: the first token of `capital`.
_G7 = SimpleNamespace(prompt="capital", request_id=106, token=0)
_NUM_SLOTS = 12
#: The padding dummy's own slot; a padding row must leave it alone.
_DUMMY_SLOT = 0
_GRAPH_ROWS = 8


@dataclasses.dataclass(frozen=True)
class _Step:
    name: str
    rows: tuple[int, ...]  # indices into _ROWS, contexts first
    spans: tuple[tuple[int, int], ...]  # each row's tokens, positions in its sequence
    num_contexts: int


def _steps() -> tuple[_Step, ...]:
    steps = [_Step("S0", (2,), (_ROWS[2].first_chunk,), 1)]
    steps.append(_Step("S1", tuple(range(len(_ROWS))), tuple(r.prefill for r in _ROWS), len(_ROWS)))
    for t in range(_DECODE_STEPS):
        spans = tuple((r.prefill[1] + t, r.prefill[1] + t + 1) for r in _ROWS)
        steps.append(_Step(f"D{t}", tuple(range(len(_ROWS))), spans, 0))
    return tuple(steps)


_STEPS = _steps()
_DECODE = tuple(s for s in _STEPS if s.num_contexts == 0)
_G7_ROWS = tuple(i for i in range(len(_ROWS)) if i != _FINISHED)


def _offsets(step: _Step) -> list[slice]:
    """Each row's tokens within the step's packed tokens."""
    out, start = [], 0
    for a, b in step.spans:
        out.append(slice(start, start + b - a))
        start += b - a
    return out


# --------------------------------------------------------------------------------------------
# Statistics, bands and the report
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Band:
    element: float  # ulps of the row's scale: its largest |reference|, or its operand scale
    rms: float  # ulps of relative RMS
    unit: float  # one ulp: BF16 or FP32
    rests_on: str


_BANDS = {
    "u": _Band(32.0, 2.0, BF16, "the chunk entry's out band; later ops keep relative error"),
    "h": _Band(32.0, 2.0, BF16, "the chunk entry's out band; the gated norm keeps a head's error"),
    "state": _Band(8.0, 2.0, BF16, "the chunk entry's state band; decode steps do not expand it"),
    # The independent reference's state: element errors there also carry in_proj's summation
    # order, through the recurrence's inputs (the "state ref vs ref@in_proj" rows); its RMS
    # stays gated.
    "state, own in_proj": _Band(float("inf"), 2.0, BF16, "the chunk entry's state RMS band"),
    "conv_state": _Band(4.0, 2.0, BF16, "cublas_mm's band: the window is in_proj outputs"),
    "x": _Band(4.0, 0.5, BF16, "wide_residual's normed-read band"),
    "chunk.out": _Band(32.0, 2.0, BF16, "flashinfer_gdp_chunk's certified out band"),
    "chunk.state": _Band(8.0, 2.0, BF16, "flashinfer_gdp_chunk's certified state band"),
    "recurrent.out": _Band(2.0, 0.25, BF16, "gdp_recurrent's certified out band"),
    "recurrent.state": _Band(8.0, 1.0, FP32, "gdp_recurrent's certified state band"),
}

#: An element is "significant" for the max-relative column when |ref| is at least this share
#: of its row's largest |ref|; below that a relative error says nothing about the op.
_SIGNIFICANT = 1.0 / 16


def _stats(
    out: torch.Tensor, ref: torch.Tensor, unit: float, op_scale: torch.Tensor | None = None
) -> dict:
    """out against ref, rows of their last dim: max / mean absolute error, the max relative
    error over significant elements, the element error against the row's largest |ref| (`elem`)
    and against its operand scale (`elem_op`, when `op_scale` gives one per row), the relative
    RMS (in `unit` ulps, as `assert_within_ulp` scores them), the share of elements that differ,
    and the worst element (row, column, out, ref) by the gated element error."""
    o = out.detach().double().reshape(-1, out.shape[-1])
    r = ref.detach().double().reshape(-1, ref.shape[-1])
    assert o.shape == r.shape, (tuple(out.shape), tuple(ref.shape))
    d = (o - r).abs()
    scale = r.abs().amax(dim=1, keepdim=True)
    row_rel = d / scale.clamp_min(1e-30)
    gated = row_rel
    elem_op = None
    if op_scale is not None:
        op = op_scale.detach().double().reshape(-1, 1)
        assert op.shape[0] == o.shape[0], (tuple(op_scale.shape), tuple(out.shape))
        gated = d / torch.maximum(scale, op).clamp_min(1e-30)
        elem_op = float(gated.max()) / unit
    worst = int(gated.argmax())
    row, col = divmod(worst, o.shape[1])
    significant = r.abs() >= _SIGNIFICANT * scale
    rel = torch.where(significant & (r != 0), d / r.abs().clamp_min(1e-30), torch.zeros_like(d))
    rms_ref = float(r.pow(2).mean().sqrt())
    return dict(
        max_abs=float(d.max()),
        mean_abs=float(d.mean()),
        max_rel=float(rel.max()),
        elem=float(row_rel.max()) / unit,
        elem_op=elem_op,
        rms=(float((o - r).pow(2).mean().sqrt()) / max(rms_ref, 1e-30)) / unit,
        mismatch=float((d != 0).double().mean()),
        worst=(row, col, float(o[row, col]), float(r[row, col])),
        unit="bf16" if unit == BF16 else "fp32",
    )


class _Report:
    """Rows of one test's comparisons, printed as a table; failures collected, then asserted."""

    def __init__(self, title: str):
        self.title, self.rows, self.failures = title, [], []

    def add(self, layer, step, what, built_by, versus, stats, band: _Band | None = None):
        ok = None
        if band is not None:
            assert stats["unit"] == ("bf16" if band.unit == BF16 else "fp32")
            elem = stats["elem"] if stats["elem_op"] is None else stats["elem_op"]
            ok = elem <= band.element and stats["rms"] <= band.rms
            if not ok:
                self.failures.append(
                    f"layer {layer} {step} {what} vs {versus}: element {elem:.3g} / rms "
                    f"{stats['rms']:.3g} {stats['unit']} ulp outside {band.element} / {band.rms} "
                    f"({band.rests_on}); worst {stats['worst']}"
                )
        self.rows.append(
            dict(
                layer=layer,
                step=step,
                what=what,
                built_by=built_by,
                versus=versus,
                band=None if band is None else (band.element, band.rms),
                ok=ok,
                **stats,
            )
        )

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def flush(self) -> None:
        print(f"\n== {self.title} ==")
        print(
            f"{'layer':>5} {'step':>5} {'output':<16} {'built by':<22} {'vs':<14} {'max abs':>9} "
            f"{'mean abs':>9} {'max rel':>9} {'elem':>8} {'elem/op':>8} {'rms':>8} {'unit':>4} "
            f"{'differ':>8} {'band':>11} worst (row, col: out / ref)"
        )
        for r in self.rows:
            band = "" if r["band"] is None else f"{r['band'][0]:g}/{r['band'][1]:g}"
            mark = "" if r["ok"] is None else (" ok" if r["ok"] else " FAIL")
            elem_op = "-" if r["elem_op"] is None else f"{r['elem_op']:.3g}"
            row, col, o, ref = r["worst"]
            print(
                f"{r['layer']:>5} {r['step']:>5} {r['what']:<16} {r['built_by']:<22} "
                f"{r['versus']:<14} {r['max_abs']:9.3g} {r['mean_abs']:9.3g} {r['max_rel']:9.3g} "
                f"{r['elem']:8.3g} {elem_op:>8} {r['rms']:8.3g} {r['unit']:>4} "
                f"{r['mismatch']:8.2e} {band + mark:>11} ({row}, {col}: {o:.6g} / {ref:.6g})"
            )
        path = os.environ.get("NEMOTRON4_GDP_BLOCK_REPORT")
        if path:
            with open(path, "a") as f:
                for r in self.rows:
                    f.write(json.dumps(dict(r, test=self.title)) + "\n")
        assert not self.failures, "\n".join(self.failures)


# --------------------------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _RefState:
    """The reference's state of one sequence after its last token."""

    conv: torch.Tensor  # [conv_dim, kernel - 1] bf16, the last conv inputs, oldest first
    recurrent: torch.Tensor  # [heads, key_dim, value_dim] fp32, the checkpoint's layout
    scale: torch.Tensor  # [heads, value_dim] fp32, each state column's operand scale


def _recurrence(q, k, v, g, beta, initial_state=None, initial_scale=None):
    """`_nemotron4_reference.gated_delta_rule_recurrent` (fp32, q and k l2-normalized) op for op,
    plus the operand scales of its state columns and of its outputs.

    A state column `S[:, v]` (a row of the pool's `[V, K]` layout) has as operand scale the largest
    magnitude it has held, decayed since by the decays the state went through: the size of the
    values the recurrence rounded on its way to the column's current value. An output's operand
    scale is its head's largest column scale times `|q|_1`, which bounds `sum_k |S[k, v] q_k|`.

    Returns `(out [L, H, V], state [H, K, V], scale [H, V], out_scale [L, H])`.
    """
    dtype = torch.float32
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))
    q = R.l2norm(q)
    k = R.l2norm(k)
    q = q / (q.shape[-1] ** 0.5)
    num_steps, heads, key_dim = k.shape
    if initial_state is None:
        state = torch.zeros(heads, key_dim, v.shape[-1], dtype=dtype, device=v.device)
    else:
        state = initial_state.to(dtype).clone()
    scale = state.abs().amax(dim=-2)
    if initial_scale is not None:
        scale = torch.maximum(scale, initial_scale)
    decay = g.exp()[:, :, None, None]
    q_norm1 = q.abs().sum(dim=-1)
    out = torch.zeros_like(v)
    out_scale = torch.zeros(num_steps, heads, dtype=dtype, device=v.device)
    for i in range(num_steps):
        state = state * decay[i]
        scale = scale * decay[i, :, :, 0]
        kv_mem = (state * k[i, :, :, None]).sum(dim=-2)
        delta = (v[i] - kv_mem) * beta[i, :, None]
        state = state + k[i, :, :, None] * delta[:, None, :]
        scale = torch.maximum(scale, state.abs().amax(dim=-2))
        out[i] = (state * q[i, :, :, None]).sum(dim=-2)
        out_scale[i] = scale.amax(dim=-1) * q_norm1[i]
    return out, state, scale, out_scale


def _reference_mixer(x, w, config, state, *, conv="once", rival=None, proj=None):
    """`Nemotron4GDPMixer` over one sequence's tokens, continuing from `state`.

    `_nemotron4_reference.gdp_mixer` op for op, except where `conv` says: "once" applies SiLU to
    the fp32 tap sum and rounds once, where the target's conv entries round; "hf" rounds the conv
    output and then SiLU's, where the checkpoint's code rounds (and is then `gdp_mixer` itself,
    bit for bit: `test_the_reference_transcribes_ref_torch`). `rival` plants a mistake:
    "reversed_householder" runs a token's Householder steps last to first, "decay_every_step"
    applies the decay on every sub-step, "early_read" reads the output one step early.

    Returns `(u, h, state, proj)`: the mixer output, the gated norm's output (out_proj's input),
    the `_RefState` after the last token and the in_proj output.
    """
    if proj is None:
        proj = R.linear(x, w["in_proj.weight"])
    num_tokens = proj.shape[0]
    heads, groups = config.gdp_num_heads, config.gdp_num_groups
    head_v, head_k, n_h = config.gdp_head_dim, config.gdp_key_head_dim, config.gdp_num_householder
    value_dim, key_dim, conv_dim = config.gdp_value_dim, config.gdp_key_dim, config.gdp_conv_dim
    gate, mixed, b, a = proj.split([value_dim, conv_dim, n_h * heads, heads], dim=-1)
    taps = w["conv1d.weight"].reshape(conv_dim, -1).float()
    width = taps.shape[-1]
    prior = mixed.new_zeros(conv_dim, width - 1) if state is None else state.conv
    window = torch.cat([prior.t().to(mixed.dtype), mixed], dim=0)
    acc = window[0:num_tokens].float() * taps[:, 0]
    for i in range(1, width):
        acc = acc + window[i : i + num_tokens].float() * taps[:, i]
    if conv == "once":
        mixed = F.silu(acc).to(proj.dtype)
    else:
        assert conv == "hf", conv
        mixed = F.silu(acc.to(proj.dtype))
    conv_state = window[num_tokens:].t().contiguous()
    value, key, query = mixed.split([n_h * value_dim, n_h * key_dim, key_dim], dim=-1)
    value = value.reshape(num_tokens, n_h, heads, head_v)
    key = key.reshape(num_tokens, n_h, groups, head_k).repeat_interleave(heads // groups, dim=-2)
    query = query.reshape(num_tokens, groups, head_k).repeat_interleave(heads // groups, dim=-2)
    beta = b.sigmoid().reshape(num_tokens, n_h, heads)
    g = -w["A_log"].float().exp() * F.softplus(a.float() + w["dt_bias"].float())
    if rival == "reversed_householder":
        value, key, beta = value.flip(1), key.flip(1), beta.flip(1)
    read = n_h - 2 if rival == "early_read" else n_h - 1
    expanded_query = query.new_zeros(num_tokens, n_h, heads, head_k)
    expanded_query[:, read] = query
    expanded_g = g.new_zeros(num_tokens, n_h, heads)
    if rival == "decay_every_step":
        expanded_g[:] = g[:, None]
    else:
        expanded_g[:, 0] = g
    out, recurrent, scale, _ = _recurrence(
        expanded_query.flatten(0, 1),
        key.flatten(0, 1),
        value.flatten(0, 1),
        expanded_g.flatten(0, 1),
        beta.flatten(0, 1),
        None if state is None else state.recurrent,
        None if state is None else state.scale,
    )
    out = out.to(proj.dtype).reshape(num_tokens, n_h, heads, head_v)[:, read]
    h = R.gated_rms_norm(
        out.reshape(num_tokens, value_dim),
        gate,
        w["norm.weight"],
        value_dim // groups,
        config.gdp_norm_eps,
    )
    return R.linear(h, w["out_proj.weight"]), h, _RefState(conv_state, recurrent, scale), proj


def _reference_run(run, conv="once", rival=None, target_in_proj=False) -> dict:
    """The reference over S0, S1 and D0-D15, fed the target's mixer input, carrying its own
    states. With `target_in_proj` it also takes the target's in_proj output in place of its own
    fp32 GEMM. `rival` also takes the scenario-level mistakes "continuation_from_zero" (S0's
    state dropped at S1) and "decode_conv_from_zero" (each decode step's conv window dropped).

    Returns `{(step, row): (u, h, state, proj)}`.
    """
    states = [None] * len(_ROWS)
    out = {}
    mixer_rival = (
        rival if rival in ("reversed_householder", "decay_every_step", "early_read") else None
    )
    with torch.no_grad():
        for step in _STEPS:
            x = run.target[step.name]["x"]
            for row, part in zip(step.rows, _offsets(step)):
                state = states[row]
                if rival == "continuation_from_zero" and step.name == "S1":
                    state = None
                if rival == "decode_conv_from_zero" and step.num_contexts == 0:
                    state = dataclasses.replace(state, conv=torch.zeros_like(state.conv))
                proj = run.target[step.name]["proj"][part] if target_in_proj else None
                result = _reference_mixer(
                    x[part], run.ref_w, run.config, state, conv=conv, rival=mixer_rival, proj=proj
                )
                states[row] = result[2]
                out[(step.name, row)] = result
    return out


# --------------------------------------------------------------------------------------------
# The target, on a stand-in core
# --------------------------------------------------------------------------------------------

_MOE_PARAMS = frozenset(
    (
        "router",
        "router_bias",
        "fc1_latent",
        "experts_up",
        "experts_down",
        "fc2_norm",
        "fc2_latent",
        "shared_up",
        "shared_down",
    )
)
_READ_PARAMS = frozenset(("res_read", "input_norm", "shortcut_read", "shortcut_norm", "mix_read"))


class _BlockWeights(dict):
    """The target's parameter mapping, holding real tensors only for one block.

    The target's constructors build views of all 44 layers' parameters; this mapping loads, on
    first access, layer `layer`'s parameters except its MoE's, and the next layer's reads (the
    call that writes a block's output makes the next block's reads). Each is filled the way
    `MODEL_WEIGHTS.load` fills it -- the table's checkpoint sources, transforms, shape and
    dtype. Every other parameter is an empty meta tensor, which no kernel accepts.
    """

    def __init__(self, checkpoint: R.Checkpoint, core, layer: int):
        super().__init__()
        self._checkpoint, self._layer = checkpoint, layer
        self._d = MODEL_WEIGHTS.dims(core)
        self._manifest = MODEL_WEIGHTS.manifest(core)
        self._entries = {entry.name: entry for entry in MODEL_WEIGHTS.WEIGHTS}

    def __missing__(self, key: str) -> torch.Tensor:
        match = re.fullmatch(r"l(\d+)_(\w+)", key)
        wanted = match is not None and (
            (int(match[1]) == self._layer and match[2] not in _MOE_PARAMS)
            or (int(match[1]) == self._layer + 1 and match[2] in _READ_PARAMS)
        )
        self[key] = self._load(key, match[2]) if wanted else torch.empty(0, 0, device="meta")
        return self[key]

    def _load(self, key: str, name: str) -> torch.Tensor:
        entry = self._entries[name]
        param = torch.empty(
            *entry.shape(self._d), dtype=entry.dtype or self._d.dtype, device="cuda"
        )
        for ckpt_key, index, transform in self._manifest[key]:
            dst = param if index is None else param[index]
            src = self._checkpoint.tensor(ckpt_key)
            if transform is not None:
                src = transform(src, self._d)
            assert dst.shape == src.shape and dst.dtype == src.dtype, (key, ckpt_key)
            dst.copy_(src)
        return param


def _core(checkpoint: R.Checkpoint, layer: int):
    """What the targets' constructors read off the core: the text config, the layer types, the
    parameter mapping, and what the core sets without speculative decoding (no MTP layer, one
    token a generation row)."""
    with open(os.path.join(checkpoint.path, "config.json")) as f:
        text = SimpleNamespace(**json.load(f)["text_config"])
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(text_config=text),
            torch_dtype=torch.bfloat16,
            spec_config=None,
        ),
        layer_types=tuple(text.layer_types),
        mtp_layer=None,
        tokens_per_gen_step=1,
    )
    core.w = _BlockWeights(checkpoint, core, layer)
    return core


class _Pools:
    """One GDP layer's state in the hybrid cache manager's packed layout: a guard block, then one
    block per slot holding the recurrent state ([24, 64, 128] fp32, K innermost) and then the
    conv state ([16896, 3] bf16), so both views are strided per slot with dense inner dims.
    Every float starts out non-zero: what a finished request left in a slot."""

    SSM = 24 * 64 * 128
    CONV = 16896 * 3 * 2 // 4

    def __init__(self, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.raw = torch.randn(_NUM_SLOTS + 1, self.SSM + self.CONV, generator=g, device="cuda")
        self.ssm = self.raw[1:, : self.SSM].view(_NUM_SLOTS, 24, 64, 128)
        self.conv = self.raw[1:, self.SSM :].view(torch.bfloat16).view(_NUM_SLOTS, 16896, 3)
        self.ssm.mul_(0.5)
        self.conv.copy_(torch.randn(self.conv.shape, generator=g, device="cuda") * 0.5)

    def snapshot(self) -> torch.Tensor:
        return self.raw.clone()

    def owned(self, slots) -> torch.Tensor:
        """A mask of `raw`: the state blocks of `slots`."""
        mask = torch.zeros_like(self.raw, dtype=torch.bool)
        for s in slots:
            mask[1 + s] = True
        return mask


class _Manager:
    """What `Nemotron4StateMetadata.prepare` and the step read off a hybrid cache manager."""

    def __init__(self, layer: int, pools: _Pools):
        self.layer, self.pools = layer, pools
        self.mamba_layer_offsets = {layer: 0}
        self.speculative_num_draft_tokens = 0
        self.slot_of = {r.request_id: r.slot for r in _ROWS}
        self.slot_of[CUDA_GRAPH_DUMMY_REQUEST_ID] = _DUMMY_SLOT

    def get_state_indices(self, request_ids, is_padding):
        return [self.slot_of[rid] for rid in request_ids]

    def get_ssm_states(self, layer: int) -> torch.Tensor:
        assert layer == self.layer, layer
        return self.pools.ssm

    def get_conv_states(self, layer: int) -> torch.Tensor:
        assert layer == self.layer, layer
        return self.pools.conv


def _metadata_step(manager, request_ids, lens, cached, num_contexts):
    """The fields of an engine step that `Nemotron4StateMetadata.prepare` reads."""
    seq_lens = torch.tensor(lens, dtype=torch.int32)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=num_contexts,
        num_ctx_tokens=int(seq_lens[:num_contexts].sum()),
        kv_cache_manager=manager,
        request_ids=list(request_ids),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=list(cached)),
    )


def _state_args(
    md, manager, num_rows: int, num_contexts: int, num_ctx_tokens: int, max_ctx_len: int = 0
) -> dict:
    """The state fields the targets' `step_args` project out of the prepared metadata;
    `max_ctx_len` is the longest context row's token count."""
    layer = manager.layer
    step = dict(
        state_slots=md.state_slots[:num_rows],
        ssm_states={layer: manager.get_ssm_states(layer)},
        conv_states={layer: manager.get_conv_states(layer)},
    )
    if num_contexts:
        step.update(
            num_rows=num_rows,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            max_ctx_len=max_ctx_len,
            query_start_loc=md.query_start_loc,
            query_start_loc_long=md.query_start_loc_long,
            has_initial_state=md.has_initial_states[:num_contexts],
        )
    return step


def _in_proj(target, x: torch.Tensor, view) -> torch.Tensor:
    """The block's in_proj output as the target computes it for these rows: the decode target
    through the GEMM it picks for the row count (`DecodeTarget._mm`), the prefill target through
    `cublas_mm`."""
    mm = target._mm if isinstance(target, M.DecodeTarget) else cublas_mm
    return mm(x, view.mixer.in_proj)


def _block(target, view, reads, streams, step):
    """The target's forward over one block: the read before the mixer, the mixer, the write
    (which also makes the next reads). `streams` is written in place. Returns `(x, u)`."""
    _, x = target._streams_step(streams, None, None, reads)
    u = target._gdp(x, view.mixer, step)
    after = (view.shared_read,) if view.moe is not None else view.next_reads
    target._streams_step(streams, u, view.write, after)
    return x, u


class _Recorder:
    """Stands in for a module-level entry of the target's modeling code: records what each call
    received and returned (the pool before and after), then defers to the entry."""

    def __init__(self, op):
        self.op, self.calls = op, []

    def __call__(self, *args):
        q, k, v, g, beta, pool, slots, *rest = args
        before = pool.clone()
        out = self.op(*args)
        self.calls.append(
            dict(
                args=[t.clone() if isinstance(t, torch.Tensor) else t for t in args],
                pool_before=before,
                pool_after=pool.clone(),
                out=out.clone(),
            )
        )
        return out


def _entry_scales(name: str, args: list) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """The operand scales of one recorded chunk / `gdp_recurrent` call, from the inputs it
    received (`args`, with the pool as it was before the call): `({slot: [H, V]}, [T, H])`."""
    q, k, v, g, beta, pool, slots = args[:7]
    num_tokens, n_h, heads, _ = v.shape
    if name == _CHUNK:
        bounds, resume = args[7].tolist(), args[8].tolist()
    else:
        bounds, resume = list(range(num_tokens + 1)), [True] * num_tokens
    state_scales = {}
    out_scales = torch.zeros(num_tokens, heads, device=v.device)
    for n, slot in enumerate(slots.tolist()):
        a, b = bounds[n], bounds[n + 1]
        initial = pool[slot].transpose(-1, -2) if slot >= 0 and resume[n] else None
        expanded_q = q.new_zeros(b - a, n_h, heads, q.shape[-1])
        expanded_q[:, -1] = q[a:b]
        expanded_g = g.new_zeros(b - a, n_h, heads)
        expanded_g[:, 0] = g[a:b]
        _, _, scale, out_scale = _recurrence(
            expanded_q.flatten(0, 1),
            k[a:b].flatten(0, 1),
            v[a:b].flatten(0, 1),
            expanded_g.flatten(0, 1),
            beta[a:b].flatten(0, 1),
            initial,
        )
        if slot >= 0:
            state_scales[slot] = scale
        out_scales[a:b] = out_scale.reshape(b - a, n_h, heads)[:, -1]
    return state_scales, out_scales


def _entry_rows(report, layer, step, name: str, call: dict, rows: slice | None = None) -> None:
    """One recorded recurrence call against the entry's own fp32 reference on the same inputs
    (its certified bands), each element scored against its operand scale. `rows` picks the
    request rows (all by default)."""
    entry = getattr(M, name)
    args = list(call["args"])
    args[5] = call["pool_before"]
    ref_out, ref_pool = entry.reference(*args)
    state_scales, out_scales = _entry_scales(name, args)
    kind = "chunk" if name == _CHUNK else "recurrent"
    slots = args[6].tolist()
    if rows is None:
        rows = slice(0, len(slots))
    if name == _CHUNK:  # outputs are tokens; take the chosen requests' tokens
        bounds = args[7].tolist()
        tokens = slice(bounds[rows.start], bounds[rows.stop])
    else:
        tokens = rows
    live = [s for s in slots[rows] if s >= 0]
    report.add(
        layer,
        step,
        f"{kind}.out",
        "teacher-forced",
        "entry ref",
        _stats(call["out"][tokens], ref_out[tokens], BF16, out_scales[tokens]),
        _BANDS[f"{kind}.out"],
    )
    band = _BANDS[f"{kind}.state"]
    report.add(
        layer,
        step,
        f"{kind}.state",
        "teacher-forced",
        "entry ref",
        _stats(
            call["pool_after"][live],
            ref_pool[live],
            band.unit,
            torch.stack([state_scales[s] for s in live]),
        ),
        band,
    )


@contextmanager
def _recording():
    """Record every chunk / `gdp_recurrent` / gated-norm call the block makes."""
    recorders = {
        _CHUNK: _Recorder(getattr(M, _CHUNK)),
        "gdp_recurrent": _Recorder(M.gdp_recurrent),
    }
    norm_outputs = []
    real_norm = getattr(M, _NORM)

    def norm(*args, **kwargs):
        out = real_norm(*args, **kwargs)
        norm_outputs.append(out.clone())
        return out

    saved = {name: getattr(M, name) for name in (*recorders, _NORM)}
    try:
        for name, recorder in recorders.items():
            setattr(M, name, recorder)
        setattr(M, _NORM, norm)
        yield recorders, norm_outputs
    finally:
        for name, op in saved.items():
            setattr(M, name, op)


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


def _checkpoint_or_skip() -> str:
    root = os.environ.get("LLM_MODELS_ROOT")
    path = os.path.join(root, "nemotron4-nano-preview") if root else None
    if not path or not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
        pytest.skip("needs the checkpoint at $LLM_MODELS_ROOT/nemotron4-nano-preview")
    return path


#: The test's measured peak is 9.4 GiB: layers 0-6's weights, the embedding, a 2596-token step.
_MIN_FREE_GIB = 12


@pytest.fixture(scope="module")
def checkpoint() -> R.Checkpoint:
    path = _checkpoint_or_skip()
    free, _ = torch.cuda.mem_get_info()
    if free < _MIN_FREE_GIB * 2**30:
        pytest.skip(f"needs {_MIN_FREE_GIB} GiB of free GPU memory, {free / 2**30:.1f} free")
    return R.Checkpoint(path)


@pytest.fixture(scope="module")
def prompts(checkpoint) -> dict[str, tuple[int, ...]]:
    tokenizer = P.load_tokenizer(checkpoint.path)
    return {p.name: p.token_ids for p in P.prompt_set(tokenizer)}


@pytest.fixture(scope="module")
def block_inputs(checkpoint, prompts) -> dict:
    """`{layer: {row: streams}}` with `row` an index of `_ROWS` or "g7": each request's whole
    sequence (first chunk, prefill and decode tokens) as the block's input streams, computed by
    the reference from the embedding through the layers before the block."""
    config = checkpoint.config
    deepest = max(_LAYERS)
    sequences = {
        i: prompts[r.prompt][: r.prefill[1] + r.decode_tokens] for i, r in enumerate(_ROWS)
    }
    sequences["g7"] = prompts[_G7.prompt][_G7.token : _G7.token + 1]
    for i, r in enumerate(_ROWS):
        assert len(sequences[i]) == r.prefill[1] + r.decode_tokens, r
    embed = checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")
    layers = {i: checkpoint.layer(i) for i in range(deepest)}
    out = {layer: {} for layer in _LAYERS}
    with torch.no_grad():
        for name, ids in sequences.items():
            streams = embed[torch.tensor(ids, device="cuda")].repeat(
                1, config.wide_residual_num_streams
            )
            for i in range(deepest + 1):
                if i in _LAYERS:
                    out[i][name] = streams.clone()
                if i < deepest:
                    streams, _ = R.decoder_layer(config.layer_types[i], streams, layers[i], config)
    del layers, embed
    torch.cuda.empty_cache()
    return out


def _step_streams(inputs: dict, step: _Step) -> torch.Tensor:
    return torch.cat([inputs[row][a:b] for row, (a, b) in zip(step.rows, step.spans)])


def _connection(layer_type: str) -> str:
    return (
        "residual_connection" if layer_type == "linear_attention" else "mixer_residual_connection"
    )


@pytest.fixture(scope="module", params=_LAYERS, ids=lambda layer: f"layer{layer}")
def block_run(request, checkpoint, block_inputs):
    """The eager run of one block over S0, S1 and D0-D15, with every entry guarded and the
    recurrence calls recorded."""
    layer = request.param
    config = checkpoint.config
    core = _core(checkpoint, layer)
    prefill, decode = M.PrefillTarget(core), M.DecodeTarget(core)
    view = prefill._layers[layer]
    assert isinstance(view.mixer, M._Gdp) and view.mixer.layer == layer
    reads = M._block_reads(core.w, core.layer_types, layer)
    prefix = f"{R.TEXT_PREFIX}layers.{layer}."
    ref_w = {
        k[len(prefix) :]: checkpoint.tensor(k)
        for k in checkpoint.keys(prefix)
        if not k[len(prefix) :].startswith("moe.")
    }
    inputs = block_inputs[layer]
    pools = _Pools(seed=1000 + layer)
    manager = _Manager(layer, pools)
    md = M.Nemotron4StateMetadata(_GRAPH_ROWS, 128, max_num_tokens=8192)
    run = SimpleNamespace(
        layer=layer,
        layer_type=config.layer_types[layer],
        config=config,
        core=core,
        prefill=prefill,
        decode=decode,
        view=view,
        reads=reads,
        ref_w={k[len("mixer.") :]: v for k, v in ref_w.items() if k.startswith("mixer.")},
        block_w=ref_w,
        inputs=inputs,
        pools=pools,
        manager=manager,
        target={},
        snapshots={"initial": pools.snapshot()},
        records={},
    )
    with torch.no_grad(), validating(*_ENTRIES), _recording() as (recorders, norms):
        for step in _STEPS:
            target = prefill if step.num_contexts else decode
            ids = [_ROWS[i].request_id for i in step.rows]
            lens = [b - a for a, b in step.spans]
            cached = [a for a, _ in step.spans]
            md.prepare(_metadata_step(manager, ids, lens, cached, step.num_contexts))
            n_ctx_tokens = sum(lens[: step.num_contexts])
            args = _state_args(
                md,
                manager,
                len(ids),
                step.num_contexts,
                n_ctx_tokens,
                max(lens[: step.num_contexts], default=0),
            )
            streams_in = _step_streams(inputs, step)
            streams = streams_in.clone()
            calls_before = {k: len(r.calls) for k, r in recorders.items()}
            norms_before = len(norms)
            x, u = _block(target, view, reads, streams, args)
            torch.cuda.synchronize()
            run.target[step.name] = dict(
                x=x.clone(),
                u=u.clone(),
                streams_in=streams_in,
                streams=streams.clone(),
                h=norms[norms_before],
                proj=_in_proj(target, x, view),
            )
            run.records[step.name] = {
                k: r.calls[calls_before[k] :]
                for k, r in recorders.items()
                if len(r.calls) > calls_before[k]
            }
            run.snapshots[step.name] = pools.snapshot()
    run.reference = _reference_run(run)
    run.reference_tf = _reference_run(run, target_in_proj=True)
    return run


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


def _built_by(step: _Step, row: int) -> str:
    if step.name == "S0":
        return "chunk (fresh)"
    if step.name == "S1":
        return "chunk after chunk" if _ROWS[row].first_chunk else "chunk (fresh)"
    return f"chunk + {int(step.name[1:]) + 1} recurrent"


def test_gdp_block_against_the_reference(block_run):
    """Every step of the eager scenario, per request: the mixer input, the mixer output and its
    pre-out_proj value, the recurrent and conv state, the write; the recurrence entries on their
    own recorded inputs; and the gap to the checkpoint's rounding order.

    Two references, both carrying their own state: "reference" runs its own fp32 in_proj, "ref
    @in_proj" takes the target's in_proj output -- the path after in_proj alone, with the in_proj
    GEMM compared on its own. "ref vs ref@in_proj" is the reference's own sensitivity to in_proj's
    summation order."""
    run = block_run
    L = run.layer
    report = _Report(f"GDP block, layer {L} ({run.layer_type}): eager steps")
    hf = _reference_run(run, conv="hf")
    conn = _connection(run.layer_type)
    conv_windows = {}
    references = (("reference", run.reference), ("ref@in_proj", run.reference_tf))
    for step in _STEPS:
        t = run.target[step.name]
        parts = _offsets(step)
        built = _built_by(step, step.rows[0])
        # The read and the write, each against the reference's own op on the same input.
        x_ref = R._connection_read(
            t["streams_in"], run.block_w, conn, run.block_w["input_layernorm.weight"], run.config
        )
        report.add(
            L, step.name, "x (read)", "-", "reference", _stats(t["x"], x_ref, BF16), _BANDS["x"]
        )
        written = R._connection_write(t["streams_in"], t["u"], run.block_w, conn, run.config)
        report.require(
            torch.equal(t["streams"], written),
            f"layer {L} {step.name}: the stream write is not bit-exact against the reference's",
        )
        snapshot = run.snapshots[step.name]
        ssm = snapshot[1:, : _Pools.SSM].view(_NUM_SLOTS, 24, 64, 128)
        conv = snapshot[1:, _Pools.SSM :].view(torch.bfloat16).view(_NUM_SLOTS, 16896, 3)
        conv_cols = t["proj"][:, run.prefill._d.conv]
        proj_ref = torch.cat([run.reference[(step.name, row)][3] for row in step.rows])
        report.add(
            L,
            step.name,
            "in_proj",
            "-",
            "reference",
            _stats(t["proj"], proj_ref, BF16),
            _BANDS["conv_state"],
        )
        u_of = {k: torch.cat([ref[(step.name, r)][0] for r in step.rows]) for k, ref in references}
        h_of = {k: torch.cat([ref[(step.name, r)][1] for r in step.rows]) for k, ref in references}
        for versus, _ in references:
            report.add(
                L,
                step.name,
                "u (mixer out)",
                built,
                versus,
                _stats(t["u"], u_of[versus], BF16),
                _BANDS["u"],
            )
            report.add(
                L,
                step.name,
                "h (out_proj in)",
                built,
                versus,
                _stats(t["h"], h_of[versus], BF16),
                _BANDS["h"],
            )
        report.add(
            L,
            step.name,
            "u ref",
            "-",
            "ref@in_proj",
            _stats(u_of["reference"], u_of["ref@in_proj"], BF16),
        )
        u_hf = torch.cat([hf[(step.name, row)][0] for row in step.rows])
        report.add(L, step.name, "u (mixer out)", "-", "hf-order", _stats(t["u"], u_hf, BF16))
        report.add(
            L, step.name, "u ref (once)", "-", "hf-order", _stats(u_of["reference"], u_hf, BF16)
        )
        for row, part in zip(step.rows, parts):
            slot, label, built_row = _ROWS[row].slot, f"{step.name}r{row}", _built_by(step, row)
            if step.num_contexts:
                for versus, _ in references:
                    report.add(
                        L,
                        label,
                        "u (mixer out)",
                        built_row,
                        versus,
                        _stats(t["u"][part], u_of[versus][part], BF16),
                        _BANDS["u"],
                    )
            for versus, ref in references:
                state = ref[(step.name, row)][2]
                report.add(
                    L,
                    label,
                    "state",
                    built_row,
                    versus,
                    _stats(ssm[slot], state.recurrent.transpose(-1, -2), BF16, state.scale),
                    _BANDS["state" if versus == "ref@in_proj" else "state, own in_proj"],
                )
            ref_state, tf_state = (
                run.reference[(step.name, row)][2],
                run.reference_tf[(step.name, row)][2],
            )
            report.add(
                L,
                label,
                "state ref",
                "-",
                "ref@in_proj",
                _stats(
                    ref_state.recurrent.transpose(-1, -2),
                    tf_state.recurrent.transpose(-1, -2),
                    BF16,
                    tf_state.scale,
                ),
            )
            report.add(
                L,
                label,
                "conv state",
                built_row,
                "reference",
                _stats(conv[slot].t(), ref_state.conv.t(), BF16),
                _BANDS["conv_state"],
            )
            hf_state = hf[(step.name, row)][2]
            report.add(
                L,
                label,
                "state",
                built_row,
                "hf-order",
                _stats(ssm[slot], hf_state.recurrent.transpose(-1, -2), BF16, hf_state.scale),
            )
            # The conv state is the last three conv inputs: the target's own in_proj outputs.
            prior = conv_windows.get(row)
            if prior is None:
                prior = torch.zeros(3, 16896, dtype=torch.bfloat16, device="cuda")
            window = torch.cat([prior, conv_cols[part]])[-3:]
            conv_windows[row] = window
            report.require(
                torch.equal(conv[slot].t(), window) and torch.equal(tf_state.conv.t(), window),
                f"layer {L} {step.name} row {row}: the conv state is not the last three in_proj "
                "conv inputs",
            )
        # The recurrence entries, each on what its calls received, against its own reference.
        for name, calls in run.records[step.name].items():
            for call in calls:
                _entry_rows(report, L, step.name, name, call)
    # Nothing outside the steps' slots moved: the other slots, the dummy's, the guard block.
    rows_slots = [r.slot for r in _ROWS]
    untouched = ~run.pools.owned(rows_slots)
    report.require(
        torch.equal(run.snapshots["D15"][untouched], run.snapshots["initial"][untouched]),
        f"layer {L}: the block wrote outside its requests' slots",
    )
    print(f"\npeak GPU memory so far: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    report.flush()


def _pool_views(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(ssm [slots, 24, 64, 128], conv [slots, 16896, 3])` over a `_Pools.raw` snapshot."""
    ssm = raw[1:, : _Pools.SSM].view(_NUM_SLOTS, 24, 64, 128)
    conv = raw[1:, _Pools.SSM :].view(torch.bfloat16).view(_NUM_SLOTS, 16896, 3)
    return ssm, conv


@pytest.fixture(scope="module")
def graph_run(block_run):
    """D0-D15 again, from the pools as S1 left them, as replays of one CUDA graph of
    `_GRAPH_ROWS` rows: the five requests, then padding rows carrying the padding dummy's id."""
    run = block_run
    pools = _Pools(seed=0)
    pools.raw.copy_(run.snapshots["S1"])
    manager = _Manager(run.layer, pools)
    md = M.Nemotron4StateMetadata(_GRAPH_ROWS, 128, max_num_tokens=8192)
    n, live = _GRAPH_ROWS, len(_ROWS)
    pad = n - live
    static = torch.empty(n, run.inputs[0].shape[1], dtype=torch.bfloat16, device="cuda")
    args = _state_args(md, manager, n, 0, 0)

    def fill(real: torch.Tensor) -> None:
        static[: real.shape[0]].copy_(real)
        static[real.shape[0] :].copy_(real[:1].expand(n - real.shape[0], -1))

    with torch.no_grad():
        # Warm up with every row padding: no slot is read or written.
        md.prepare(_metadata_step(manager, [CUDA_GRAPH_DUMMY_REQUEST_ID] * n, [1] * n, [0] * n, 0))
        fill(_step_streams(run.inputs, _DECODE[0]))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _block(run.decode, run.view, run.reads, static, args)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            x_out, u_out = _block(run.decode, run.view, run.reads, static, args)
        torch.cuda.synchronize()
        assert torch.equal(pools.raw, run.snapshots["S1"]), "warm-up or capture touched the pools"
        steps = {}
        for step in _DECODE:
            ids = [_ROWS[i].request_id for i in step.rows] + [CUDA_GRAPH_DUMMY_REQUEST_ID] * pad
            cached = [a for a, _ in step.spans] + [0] * pad
            md.prepare(_metadata_step(manager, ids, [1] * n, cached, 0))
            fill(_step_streams(run.inputs, step))
            graph.replay()
            torch.cuda.synchronize()
            steps[step.name] = dict(
                x=x_out[:live].clone(),
                u=u_out[:live].clone(),
                streams=static[:live].clone(),
                slots=md.state_slots[:n].tolist(),
                pool=pools.snapshot(),
            )
    return SimpleNamespace(
        graph=graph,
        static=static,
        x=x_out,
        u=u_out,
        md=md,
        manager=manager,
        pools=pools,
        steps=steps,
        fill=fill,
    )


def _out_proj_at(rows: int, h: torch.Tensor, run) -> torch.Tensor:
    """out_proj of `h`'s rows through the GEMM the decode target picks for a step of `rows` rows
    (`DecodeTarget._mm`), the rows past `h`'s zero. Each row of a GEMM's output depends on that
    row alone, so the first rows are what a step of `rows` rows computes from them."""
    padded = h.new_zeros(rows, h.shape[1])
    padded[: h.shape[0]] = h
    return run.decode._mm(padded, run.view.mixer.out_proj)[: h.shape[0]]


def test_decode_steps_replay_in_a_cuda_graph_with_padding(block_run, graph_run):
    """The graph's replays give the eager steps' reads and states bit for bit, and its padding
    rows -- `PAD_SLOT_ID` in the slots the entries read -- touch no slot: neither the padding
    dummy's own, nor any other outside the five requests'.

    The decode target picks each GEMM's kernel by the row count (`DecodeTarget._mm`), and the
    graph runs `_GRAPH_ROWS` rows where the eager steps ran five, so its out_proj may run
    another kernel: the graph's mixer output must be the eager step's out_proj input through
    the GEMM a step of `_GRAPH_ROWS` rows picks, bit for bit, and its stream write must be that
    output's. How far that moves the mixer output from the eager step's is reported."""
    run, gr = block_run, graph_run
    L = run.layer
    report = _Report(f"GDP block, layer {L}: D0-D15 replayed in a CUDA graph of {_GRAPH_ROWS} rows")
    owned = run.pools.owned([r.slot for r in _ROWS])
    conn = _connection(run.layer_type)
    for step in _DECODE:
        e, g = run.target[step.name], gr.steps[step.name]
        report.require(
            g["slots"] == [r.slot for r in _ROWS] + [PAD_SLOT_ID] * (_GRAPH_ROWS - len(_ROWS)),
            f"layer {L} {step.name}: slots {g['slots']}",
        )
        report.require(
            torch.equal(e["x"], g["x"]),
            f"layer {L} {step.name}: the graph's x differs from the eager step's",
        )
        report.require(
            torch.equal(g["u"], _out_proj_at(_GRAPH_ROWS, e["h"], run)),
            f"layer {L} {step.name}: the graph's u is not out_proj of the eager step's h at "
            f"{_GRAPH_ROWS} rows",
        )
        written = R._connection_write(e["streams_in"], g["u"], run.block_w, conn, run.config)
        report.require(
            torch.equal(g["streams"], written),
            f"layer {L} {step.name}: the graph's stream write is not its u's",
        )
        report.require(
            torch.equal(g["pool"][owned], run.snapshots[step.name][owned]),
            f"layer {L} {step.name}: the graph's states differ from the eager step's",
        )
        report.add(L, step.name, "u (mixer out)", "graph", "eager", _stats(g["u"], e["u"], BF16))
        g_ssm, _ = _pool_views(g["pool"])
        e_ssm, _ = _pool_views(run.snapshots[step.name])
        slots = [r.slot for r in _ROWS]
        report.add(
            L, step.name, "state", "graph", "eager", _stats(g_ssm[slots], e_ssm[slots], FP32)
        )
    last = gr.steps[_DECODE[-1].name]["pool"]
    report.require(
        torch.equal(last[~owned], run.snapshots["S1"][~owned]),
        f"layer {L}: a padding row wrote outside the requests' slots",
    )
    report.flush()


def test_g7_fresh_prompt_promoted_onto_a_recycled_slot(block_run, graph_run):
    """G7. The runner promotes a fresh one-token prompt into a CUDA-graph decode row; here it
    lands on the slot of the request that finished after D15, which still holds that request's
    state. `Nemotron4StateMetadata.prepare` must zero it: the step's outputs and states equal the
    same step on a slot zeroed beforehand, bit for bit. Without the reset (the row passed as one
    with a cached token) the prompt's output moves, so the stale state is there to be read.

    The prompt's state is built by the decode path alone; it is also compared with the
    reference, with `gdp_recurrent`'s own reference on the inputs the eager step recorded, and
    with the same prompt run as a one-token prefill (`flashinfer_gdp_chunk`), which the runner
    does when CUDA graphs are off. The eager step runs five rows where the graph runs
    `_GRAPH_ROWS`, so their mixer outputs are compared as the graph-replay test compares them:
    the graph's is the eager step's out_proj input through the GEMM the graph's row count
    picks."""
    run, gr = block_run, graph_run
    L = run.layer
    report = _Report(f"GDP block, layer {L}: G7, a fresh one-token prompt on a recycled slot")
    slot = _ROWS[_FINISHED].slot
    gr.manager.slot_of[_G7.request_id] = slot
    after_decode = gr.pools.snapshot()
    _, conv_after = _pool_views(after_decode)
    ssm_after, _ = _pool_views(after_decode)
    assert bool(ssm_after[slot].abs().gt(0).any()) and bool(conv_after[slot].abs().gt(0).any())
    rows = list(_G7_ROWS)
    live = len(rows) + 1
    ids = [_ROWS[i].request_id for i in rows] + [_G7.request_id]
    ids += [CUDA_GRAPH_DUMMY_REQUEST_ID] * (_GRAPH_ROWS - live)
    positions = [_ROWS[i].prefill[1] + _DECODE_STEPS for i in rows]
    inputs = torch.cat(
        [run.inputs[i][p : p + 1] for i, p in zip(rows, positions)] + [run.inputs["g7"]]
    )
    owned = gr.pools.owned([_ROWS[i].slot for i in rows] + [slot])

    def replay(g7_cached: int, zero_first: bool) -> dict:
        gr.pools.raw.copy_(after_decode)
        if zero_first:
            gr.pools.ssm[slot].zero_()
            gr.pools.conv[slot].zero_()
        cached = positions + [g7_cached] + [0] * (_GRAPH_ROWS - live)
        gr.md.prepare(_metadata_step(gr.manager, ids, [1] * _GRAPH_ROWS, cached, 0))
        gr.fill(inputs)
        gr.graph.replay()
        torch.cuda.synchronize()
        return dict(
            x=gr.x[:live].clone(),
            u=gr.u[:live].clone(),
            streams=gr.static[:live].clone(),
            pool=gr.pools.snapshot(),
        )

    with torch.no_grad():
        recycled = replay(0, zero_first=False)
        zeroed = replay(0, zero_first=True)
        stale = replay(1, zero_first=False)
    for what in ("x", "u", "streams"):
        report.require(
            torch.equal(recycled[what], zeroed[what]),
            f"layer {L} G7: {what} on the recycled slot differs from the zeroed slot's",
        )
    report.require(
        torch.equal(recycled["pool"][owned], zeroed["pool"][owned]),
        f"layer {L} G7: states on the recycled slot differ from the zeroed slot's",
    )
    report.require(
        torch.equal(recycled["pool"][~owned], after_decode[~owned]),
        f"layer {L} G7: the step wrote outside its rows' slots",
    )
    report.add(
        L,
        "G7",
        "u (no reset)",
        "stale slot",
        "zeroed slot",
        _stats(stale["u"][-1:], zeroed["u"][-1:], BF16),
    )
    report.require(
        not torch.equal(stale["u"][-1], zeroed["u"][-1]),
        f"layer {L} G7: the stale slot's state does not reach the output; the check is blind",
    )

    # The same step, eager, with the entries guarded and the recurrence call recorded.
    eager_pools = _Pools(seed=0)
    eager_pools.raw.copy_(after_decode)
    eager_manager = _Manager(L, eager_pools)
    eager_manager.slot_of[_G7.request_id] = slot
    md = M.Nemotron4StateMetadata(_GRAPH_ROWS, 128, max_num_tokens=8192)
    with torch.no_grad(), validating(*_ENTRIES), _recording() as (recorders, norms):
        md.prepare(_metadata_step(eager_manager, ids[:live], [1] * live, positions + [0], 0))
        streams = inputs.clone()
        x, u = _block(
            run.decode, run.view, run.reads, streams, _state_args(md, eager_manager, live, 0, 0)
        )
        torch.cuda.synchronize()
    report.require(
        torch.equal(recycled["u"], _out_proj_at(_GRAPH_ROWS, norms[-1], run))
        and torch.equal(eager_pools.raw[owned], recycled["pool"][owned]),
        f"layer {L} G7: the graph step differs from the eager one",
    )
    report.add(L, "G7", "u (mixer out)", "graph", "eager", _stats(recycled["u"], u, BF16))
    (call,) = recorders["gdp_recurrent"].calls
    _entry_rows(report, L, "G7", "gdp_recurrent", call, rows=slice(live - 1, live))

    # Against the references: the prompt alone, from a zero state, through the reference's own
    # fp32 in_proj and through the target's in_proj output (the same GEMM at the eager step's rows).
    x_g7 = recycled["x"][-1:]
    proj_g7 = _in_proj(run.decode, recycled["x"], run.view)[-1:]
    g_ssm, g_conv = _pool_views(recycled["pool"])
    with torch.no_grad():
        refs = (
            ("reference", _reference_mixer(x_g7, run.ref_w, run.config, None)),
            ("ref@in_proj", _reference_mixer(x_g7, run.ref_w, run.config, None, proj=proj_g7)),
        )
    for versus, (u_ref, _, st_ref, _) in refs:
        ref_ssm = st_ref.recurrent.transpose(-1, -2)
        state_band = _BANDS["state" if versus == "ref@in_proj" else "state, own in_proj"]
        report.add(
            L,
            "G7",
            "u (mixer out)",
            "1 recurrent",
            versus,
            _stats(recycled["u"][-1:], u_ref, BF16),
            _BANDS["u"],
        )
        report.add(
            L,
            "G7",
            "state",
            "1 recurrent",
            versus,
            _stats(g_ssm[slot], ref_ssm, BF16, st_ref.scale),
            state_band,
        )
        report.add(
            L,
            "G7",
            "conv state",
            "1 recurrent",
            versus,
            _stats(g_conv[slot].t(), st_ref.conv.t(), BF16),
            _BANDS["conv_state"],
        )

    # The same prompt as a one-token prefill: the chunk entry and the varlen conv instead.
    prefill_pools = _Pools(seed=0)
    prefill_pools.raw.copy_(after_decode)
    prefill_manager = _Manager(L, prefill_pools)
    prefill_manager.slot_of[_G7.request_id] = slot
    md = M.Nemotron4StateMetadata(_GRAPH_ROWS, 128, max_num_tokens=8192)
    with torch.no_grad(), validating(*_ENTRIES):
        md.prepare(_metadata_step(prefill_manager, [_G7.request_id], [1], [0], 1))
        streams = run.inputs["g7"].clone()
        x_p, u_p = _block(
            run.prefill, run.view, run.reads, streams, _state_args(md, prefill_manager, 1, 1, 1, 1)
        )
        torch.cuda.synchronize()
    report.require(torch.equal(x_p, x_g7), f"layer {L} G7: the prefill path reads another x")
    p_ssm, p_conv = _pool_views(prefill_pools.raw)
    report.require(
        torch.equal(p_conv[slot], g_conv[slot]),
        f"layer {L} G7: the one-token prefill leaves another conv state than the decode row",
    )
    report.add(
        L,
        "G7",
        "u (mixer out)",
        "1-token chunk",
        "graph decode",
        _stats(u_p, recycled["u"][-1:], BF16),
    )
    u_tf, _, st_tf, _ = refs[1][1]
    report.add(
        L,
        "G7",
        "state",
        "1-token chunk",
        "graph decode",
        _stats(p_ssm[slot], g_ssm[slot], BF16, st_tf.scale),
    )
    report.add(
        L,
        "G7",
        "u (mixer out)",
        "1-token chunk",
        "ref@in_proj",
        _stats(u_p, u_tf, BF16),
        _BANDS["u"],
    )
    report.add(
        L,
        "G7",
        "state",
        "1-token chunk",
        "ref@in_proj",
        _stats(p_ssm[slot], st_tf.recurrent.transpose(-1, -2), BF16, st_tf.scale),
        _BANDS["state"],
    )
    report.flush()


def test_the_reference_transcribes_ref_torch(block_run):
    """`_reference_mixer(conv="hf")` is `_nemotron4_reference.gdp_mixer` bit for bit (outputs,
    recurrent and conv state) on S1's requests, the continued one from its S0 state; "once"
    moves only the rounding of the conv, so the conv state stays bit-exact."""
    run = block_run
    step = _STEPS[1]
    x = run.target["S1"]["x"]
    with torch.no_grad():
        for row, part in zip(step.rows, _offsets(step)):
            state = run.reference[("S0", row)][2] if _ROWS[row].first_chunk else None
            mine = _reference_mixer(x[part], run.ref_w, run.config, state, conv="hf")
            u, theirs = R.gdp_mixer(x[part], run.ref_w, run.config, state)
            assert torch.equal(mine[0], u), row
            assert torch.equal(mine[2].recurrent, theirs.recurrent), row
            assert torch.equal(mine[2].conv, theirs.conv), row
            once = _reference_mixer(x[part], run.ref_w, run.config, state, conv="once")
            assert torch.equal(once[2].conv, theirs.conv), row


#: The nearest mistakes, planted in the reference: if the target made one, its distance from
#: the reference would be the rival's.
_RIVALS = (
    "reversed_householder",
    "decay_every_step",
    "early_read",
    "continuation_from_zero",
    "decode_conv_from_zero",
)


def test_the_bands_reject_planted_mistakes(block_run):
    """Each planted mistake breaks the mixer-output or state band at the first step it touches."""
    run = block_run
    L = run.layer
    report = _Report(f"GDP block, layer {L}: planted mistakes against the bands")
    first = dict(continuation_from_zero=("S1",), decode_conv_from_zero=("D0", "D1"))
    for rival in _RIVALS:
        planted = _reference_run(run, rival=rival)
        rejected = False
        for name in first.get(rival, ("S1", "D0")):
            step = next(s for s in _STEPS if s.name == name)
            rows = (2,) if rival == "continuation_from_zero" else step.rows
            true = [run.reference[(name, r)] for r in rows]
            planted_rows = [planted[(name, r)] for r in rows]
            u_true = torch.cat([t[0] for t in true])
            u_rival = torch.cat([p[0] for p in planted_rows])
            h_true = torch.cat([t[1] for t in true])
            h_rival = torch.cat([p[1] for p in planted_rows])
            # States as the pool lays them out, [H, V, K], scored as the module test scores them.
            s_true = torch.stack([t[2].recurrent.transpose(-1, -2) for t in true])
            s_rival = torch.stack([p[2].recurrent.transpose(-1, -2) for p in planted_rows])
            scale = torch.stack([t[2].scale for t in true])
            su, sh = _stats(u_rival, u_true, BF16), _stats(h_rival, h_true, BF16)
            ss = _stats(s_rival, s_true, BF16, scale)
            report.add(L, name, f"u: {rival}", "-", "reference", su)
            report.add(L, name, f"h: {rival}", "-", "reference", sh)
            report.add(L, name, f"state: {rival}", "-", "reference", ss)
            for stats, band in ((su, _BANDS["u"]), (sh, _BANDS["h"]), (ss, _BANDS["state"])):
                elem = stats["elem"] if stats["elem_op"] is None else stats["elem_op"]
                rejected |= elem > band.element or stats["rms"] > band.rms
        report.require(rejected, f"layer {L}: no band rejects the planted '{rival}'")
    report.flush()
