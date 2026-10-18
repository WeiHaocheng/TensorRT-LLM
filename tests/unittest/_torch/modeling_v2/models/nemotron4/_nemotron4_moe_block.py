# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent MoE block's inputs, references, measurements and bands, for a nemotron4 target's
module-accuracy test on the checkpoint's own weights.

Nothing here knows a target. A target's test runs its own block (`Nemotron4MoE`: router,
`noaux_tc_op` top-10, `fc1_latent_proj`, the routed latent experts, `fc2_norm`,
`fc2_latent_proj`, plus the shared expert), observes what reached each op and what came back,
and hands that here; the references, the measurements and the gates are the checkpoint's.

* `real_moe_inputs` -- the routed and shared inputs the model itself produces: the plain-torch
  reference's blocks (`_nemotron4_reference`) run over the fixed prompt set, each MoE call's two
  inputs captured as the reference makes it. Real hidden states put the router's top-10/11
  margins where the model puts them, which random activations would not.
* references -- routing: `route`, `Nemotron4TopkRouter` with IEEE fp32 logits; `R_hf`
  (`reference_chain` with `rms_norm`): `latent_moe`'s chain at HF's rounding points with fp32
  GEMMs; `R_t` (`reference_chain` with `rms_norm_once`): the same chain with `fc2_norm` rounded
  once, as `flashinfer_rmsnorm` rounds it; `H` (`hf_bf16_block`): `Nemotron4MoE.forward` in its
  own bf16 order. The block references take the target's routing (ids and fp32 weights), so a
  near-tie the two routers break differently costs no row; the routing is checked on its own.
* `routing`, `block_and_stages` -- the measurements, scored by `distance` as `assert_within_ulp`
  scores them; `failures` -- the gates a measured batch breaks, with the bands under "Bands".
"""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

__extra_import_path__ = ["."]
import _nemotron4_prompts as P  # noqa: E402 — needs the path declared above
import _nemotron4_reference as R  # noqa: E402

_BF16_ULP = 2.0**-8
_FP32_ULP = 2.0**-23


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------


def real_moe_inputs(checkpoint: R.Checkpoint, layers, blocks) -> tuple[dict, int]:
    """Each layer's MoE inputs over the prompt set, as the reference's own blocks make them.

    Runs blocks 0..max(layers) of the reference over every prompt, long prompts first, and
    captures `(routed, shared)` at each `latent_moe` call. The GDP mixers run the reference's
    chunked (WY) form of the recurrence, its independent fp32 formulation of the per-token loop,
    because the loop over 8K-token prompts is slow; the inputs only have to be the model's own
    hidden states, not bit-exact ones.

    Args:
        blocks: the weights of layers 0..max(layers), `checkpoint.layer(i)` each.

    Returns:
        ``({layer: (routed [N, hidden], shared [N, hidden]) bf16}, the number of long-prompt
        rows that open N)``.
    """
    config = checkpoint.config
    prompts = P.prompt_set(P.load_tokenizer(checkpoint.path))
    prompts = [p for p in prompts if p.long] + [p for p in prompts if not p.long]
    last = max(layers)
    embed = checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")
    captured = {i: [] for i in layers}
    at = {}
    latent_moe = R.latent_moe

    def record(routed, shared, weights, cfg):
        if at["layer"] in captured:
            captured[at["layer"]].append((routed, shared))
        return latent_moe(routed, shared, weights, cfg)

    with torch.inference_mode(), pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "latent_moe", record)
        mp.setattr(
            R, "gdp_mixer", functools.partial(R.gdp_mixer, recurrence=R.gated_delta_rule_chunked)
        )
        for prompt in prompts:
            ids = torch.tensor(prompt.token_ids, device="cuda")
            streams = embed[ids].repeat(1, config.wide_residual_num_streams)
            for i in range(last + 1):
                at["layer"] = i
                streams, _ = R.decoder_layer(config.layer_types[i], streams, blocks[i], config)
    del embed
    inputs = {
        i: (torch.cat([r for r, _ in pairs]), torch.cat([s for _, s in pairs]))
        for i, pairs in captured.items()
    }
    num_long = sum(len(p.token_ids) for p in prompts if p.long)
    return inputs, num_long


# ----------------------------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------------------------


def rms_norm_once(x, weight, eps):
    """`fc2_norm` rounded once, as `flashinfer_rmsnorm` rounds it (its own reference)."""
    return flashinfer_rmsnorm.reference(x, weight, eps)


def reference_chain(routed, shared, mw, config, ids, weights, fc2_norm):
    """The reference's `latent_moe` past its router, given the routing; `fc2_norm` pluggable.

    `linear` (an IEEE fp32 GEMM rounded once), `latent_experts`, then `fc2_norm`, `linear` and the
    shared `mlp`, each rounding where the checkpoint's code rounds.

    Args:
        routed, shared: [N, hidden] bf16, the block's two inputs.
        mw: the layer's ``moe.`` weights as the checkpoint stores them, every expert included.
        ids, weights: [N, top_k] expert ids and fp32 routing weights.
        fc2_norm: `(x, weight, eps) -> bf16`, the latent norm's rounding.

    Returns:
        [N, hidden] bf16.
    """
    clamp = config.activation_clamp_value
    latent = R.linear(routed, mw["fc1_latent_proj.weight"])
    h = R.latent_experts(
        latent, ids, weights, mw["experts.up_proj"], mw["experts.down_proj"], clamp
    )
    h = R.linear(
        fc2_norm(h, mw["fc2_norm.weight"], config.rms_norm_eps), mw["fc2_latent_proj.weight"]
    )
    up, down = mw["shared_experts.up_proj.weight"], mw["shared_experts.down_proj.weight"]
    return h + R.mlp(shared, up, down, clamp)


def hf_bf16_block(routed, shared, mw, config, ids, weights):
    """`Nemotron4MoE.forward` past its router, line for line in the checkpoint's own bf16 order.

    Every projection is a bf16 `F.linear` (torch's cuBLAS GEMM, fp32 accumulation, rounded to
    bf16). The experts run `Nemotron4Experts.forward`'s loop: the experts hit in ascending id,
    each one's (slot, token) pairs from `torch.where` over the slot-major one-hot mask, the fp32
    routing weight multiplied after the activation and rounded before `down_proj`, `index_add_`
    into an fp32 buffer. `Nemotron4RMSNorm` rounds twice; the shared expert is `Nemotron4MLP`.
    Arguments as `reference_chain`'s.
    """
    clamp, eps = config.activation_clamp_value, config.rms_norm_eps

    def act(h):  # Nemotron4ClampedActivation, relu2
        h = (clamp * torch.tanh(h.float() / clamp)).to(h.dtype)
        return torch.square(F.relu(h))

    def rms_norm(x, weight):  # Nemotron4RMSNorm
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        return weight * h.to(x.dtype)

    hidden = F.linear(routed, mw["fc1_latent_proj.weight"])
    final = torch.zeros_like(hidden, dtype=weights.dtype)
    mask = F.one_hot(ids.long(), num_classes=config.n_routed_experts).permute(2, 1, 0)
    for e in (mask.sum(dim=(-1, -2)) > 0).nonzero().squeeze(-1).tolist():
        pos, tok = torch.where(mask[e])
        cur = act(F.linear(hidden[tok], mw["experts.up_proj"][e]))
        cur = (cur * weights[tok, pos, None]).to(hidden.dtype)
        cur = F.linear(cur, mw["experts.down_proj"][e])
        final.index_add_(0, tok, cur.to(final.dtype))
    hidden = final.to(hidden.dtype)
    hidden = F.linear(rms_norm(hidden, mw["fc2_norm.weight"]), mw["fc2_latent_proj.weight"])
    up, down = mw["shared_experts.up_proj.weight"], mw["shared_experts.down_proj.weight"]
    return hidden + F.linear(act(F.linear(shared, up)), down)


# ----------------------------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------------------------


def distance(out: torch.Tensor, ref: torch.Tensor, ulp: float = _BF16_ULP) -> dict:
    """How far `out` sits from `ref`, scored as `assert_within_ulp` scores it.

    `element_ulp`: max |out - ref| over each row's max |ref|, in ulps of `ulp` (bf16's 2^-8 by
    default, also for fp32 sums of bf16 values, whose own ulp would measure the container);
    `rms_ulp`: the relative RMS in the same unit; `max_abs`; `max_rel`: max |out - ref| / |ref|
    over the elements at least a tenth of their row's max; `identical`: the bit-identical
    fraction and `differing` the count of the others, of `numel`, and `differing_rows` the rows
    holding them; `worst`: the element behind `element_ulp`.
    """
    o, r = out.float().reshape(-1, out.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    diff = (o - r).abs()
    scale = r.abs().amax(-1, keepdim=True).clamp_min(1e-9)
    element = diff / scale
    row, col = divmod(int(element.argmax()), o.shape[-1])
    rel = (diff / r.abs().clamp_min(1e-30))[r.abs() >= 0.1 * scale]
    rms = float((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9))
    unequal = o != r
    return dict(
        element_ulp=float(element.max()) / ulp,
        rms_ulp=rms / ulp,
        max_abs=float(diff.max()),
        max_rel=float(rel.max()) if rel.numel() else 0.0,
        identical=float((~unequal).float().mean()),
        differing=int(unequal.sum()),
        differing_rows=int(unequal.any(-1).sum()),
        numel=o.numel(),
        worst=dict(row=row, col=col, out=float(o[row, col]), ref=float(r[row, col])),
    )


def routing(logits, bias, ids, weights, routed, mw, config) -> tuple[dict, torch.Tensor]:
    """The target's routing against `Nemotron4TopkRouter` (`route`) on IEEE fp32 logits.

    Args:
        logits, bias: what reached `noaux_tc_op`: the target's fp32 router logits [N, experts]
            and the correction bias.
        ids, weights: what `noaux_tc_op` returned, [N, top_k].
        routed: [N, hidden] bf16, the block's routed input.
        mw: the layer's ``moe.`` weights as the checkpoint stores them.

    What is measured:

    * Selection, as top-k id SETS: `noaux_tc_op` returns its ids sorted by score, HF's
      `topk(sorted=False)` does not. A row whose sets differ is a flip. A flip is legal only at a
      near-tie: with `delta` the row's largest |difference| between the target's selection
      scores (its own logits through the kernel's tanh-form sigmoid, plus the bias) and the
      reference's, swapping a reference choice `a` for a non-choice `b` needs
      `choice(a) - choice(b) <= 2 * delta`, and that difference is at least the reference's
      margin between its k-th and (k+1)-th choice. So a flip whose margin exceeds `2 * delta`
      is a selection the logits cannot explain.
    * Weights, per expert, on rows whose sets agree. With the routing scale `c`, the reference's
      chosen scores `s` summing to `S`, and `d` the row's largest |score difference| among them,
      HF's `c * s_i / S` moves by at most `c * 11 d / (S - 10 d)` (first order; the sum moves by
      at most `10 d`), plus each side's fp32 rounding of the division, the sum and the scale --
      at most 24 roundings of 2^-24, under `c * 2^-19`. A weight outside that is a different
      formula: a bias leaking into the weights, a missing renormalization, another scale.
    * The selection itself is `noaux_tc_op`'s certified arithmetic (`reference`) on the target's
      own logits, bitwise.

    Returns:
        ``(metrics, agree)``; `agree` [N] bool marks the rows whose sets agree.
    """
    top_k, scale = config.num_experts_per_tok, config.routed_scaling_factor
    ref_bias = mw["gate.e_score_correction_bias"]
    logits_r, ids_r, w_r = R.route(routed, mw["gate.weight"], ref_bias, top_k, scale)
    scores_t = 0.5 * torch.tanh(0.5 * logits) + 0.5
    scores_r = logits_r.sigmoid()
    choice_r = scores_r + ref_bias
    delta = ((scores_t + ref_bias) - choice_r).abs().amax(-1)
    ranked = choice_r.sort(-1, descending=True).values
    margin = ranked[:, top_k - 1] - ranked[:, top_k]
    agree = (ids.long().sort(-1).values == ids_r.sort(-1).values).all(-1)
    flips = (~agree).nonzero().squeeze(-1)

    num_rows, experts = choice_r.shape
    dense_t = torch.zeros(num_rows, experts, device=logits.device).scatter_(1, ids.long(), weights)
    dense_r = torch.zeros(num_rows, experts, device=logits.device).scatter_(1, ids_r, w_r)
    chosen = dense_r > 0
    weight_err = (dense_t - dense_r).abs()
    row_err = weight_err.amax(-1) / scale
    d = ((scores_t - scores_r).abs() * chosen).amax(-1)
    total = (scores_r * chosen).sum(-1)
    room = total - 10 * d
    bound = torch.where(room > 0, 11 * d / room.clamp_min(1e-30), torch.inf) + 2.0**-19
    weight_rel = (weight_err / dense_r.clamp_min(1e-30))[chosen & agree[:, None]]
    significant = chosen & agree[:, None] & (dense_r >= 1e-3 * scale)
    kernel_w, kernel_ids = noaux_tc_op.reference(logits, bias, 1, 1, top_k, scale)
    logit_diff = logits - logits_r
    logit_scale = logits_r.abs().amax(-1, keepdim=True)

    metrics = dict(
        rows=num_rows,
        experts_touched=int(ids.unique().numel()),
        agree=float(agree.float().mean()),
        flips=int(len(flips)),
        flip_margins=[float(margin[i]) for i in flips],
        flip_bounds=[float(2 * delta[i]) for i in flips],
        flips_beyond_near_tie=int((margin[flips] > 2 * delta[flips]).sum()),
        thinnest_margin=float(margin.min()),
        rows_below_1e5=int((margin < 1e-5).sum()),
        choice_delta_max=float(delta.max()),
        logit_max_abs=float(logit_diff.abs().max()),
        logit_element_ulp=float((logit_diff.abs() / logit_scale).max()) / _FP32_ULP,
        logit_rms_ulp=float(logit_diff.pow(2).mean().sqrt() / logits_r.pow(2).mean().sqrt())
        / _FP32_ULP,
        weight_err_of_scale=float(row_err[agree].max()) if bool(agree.any()) else 0.0,
        weight_bound_min=float(bound[agree].min()) if bool(agree.any()) else 0.0,
        weights_beyond_bound=int((agree & (row_err > bound)).sum()),
        weight_max_rel=float(weight_rel.max()) if weight_rel.numel() else 0.0,
        weight_max_rel_significant=(
            float((weight_err / dense_r.clamp_min(1e-30))[significant].max())
            if bool(significant.any())
            else 0.0
        ),
        weight_sum_rel_dev=float((weights.sum(-1) - scale).abs().max()) / scale,
        selection_is_the_kernels=bool(
            torch.equal(kernel_ids, ids) and torch.equal(kernel_w, weights)
        ),
    )
    return metrics, agree


def block_and_stages(out, routed, shared, mw, config, seen, agree) -> tuple[dict, dict]:
    """The block output against every reference, and each op on the target's own input.

    Args:
        out: [N, hidden] bf16, the target's block output.
        routed, shared: [N, hidden] bf16, the block's two inputs as the references take them.
        mw: the layer's ``moe.`` weights as the checkpoint stores them, every expert included.
        seen: what the target passed and got back, on these N rows: `ids`, `weights` (its
            routing), `latent` (`fc1_latent_proj`'s output), `experts` (the routed experts'
            bf16 sum, `fc2_norm`'s input) and `fc2_norm` (its output).
        agree: [N] bool, the rows whose routing agrees with the reference's (`routing`).

    Returns:
        ``(block, stages)``: `block` maps `T` (the target) against `R_t`, `R_hf` and `H`, `H`
        against `R_hf`, `R_t` against `R_hf`, and the target against `R_own` -- the reference
        with its own routing, end to end, on the rows whose routing agrees; `stages` maps
        `fc1`, `experts`, `fc2_norm` (against its single rounding and, as
        `fc2_norm_vs_hf_rounding`, HF's two) and `tail` (`fc2_latent_proj`, the shared expert
        and the add) to their distances.
    """
    ids, weights = seen.ids, seen.weights
    r_t = reference_chain(routed, shared, mw, config, ids, weights, rms_norm_once)
    r_hf = reference_chain(routed, shared, mw, config, ids, weights, R.rms_norm)
    hf = hf_bf16_block(routed, shared, mw, config, ids, weights)
    r_own = R.latent_moe(routed, shared, mw, config)

    clamp, eps = config.activation_clamp_value, config.rms_norm_eps
    up, down = mw["experts.up_proj"], mw["experts.down_proj"]
    shared_up = mw["shared_experts.up_proj.weight"]
    shared_down = mw["shared_experts.down_proj.weight"]
    tail = R.linear(seen.fc2_norm, mw["fc2_latent_proj.weight"])
    stages = dict(
        fc1=distance(seen.latent, R.linear(routed, mw["fc1_latent_proj.weight"])),
        experts=distance(
            seen.experts, R.latent_experts(seen.latent, ids, weights, up, down, clamp)
        ),
        fc2_norm=distance(seen.fc2_norm, rms_norm_once(seen.experts, mw["fc2_norm.weight"], eps)),
        fc2_norm_vs_hf_rounding=distance(
            seen.fc2_norm, R.rms_norm(seen.experts, mw["fc2_norm.weight"], eps)
        ),
        tail=distance(out, tail + R.mlp(shared, shared_up, shared_down, clamp)),
    )
    block = {
        "T vs R_t": distance(out, r_t),
        "H vs R_hf": distance(hf, r_hf),
        "R_t vs R_hf": distance(r_t, r_hf),
        "T vs R_hf": distance(out, r_hf),
        "T vs H": distance(out, hf),
    }
    if bool(agree.any()):
        block["T vs R_own (agreeing rows)"] = distance(out[agree], r_own[agree])
    return block, stages


# ----------------------------------------------------------------------------------------------
# Bands
# ----------------------------------------------------------------------------------------------
#
# Every block band rests on one measurement over all 28 MoE layers of the checkpoint: 232 batches,
# 153,232 rows of the model's own inputs through the tp1 target's block -- layers 5 and 6 over the
# prompt set, layers 24 and 39 the same way (the reference prefix up to 39), and every MoE layer
# over the six short prompts, rebuilt from the ref-torch goldens' per-layer streams.

LOGITS_BAND = (1024.0, 512.0)
"""Router logits against IEEE fp32, in fp32 ulps (2^-23): element against the row's max |logit|,
and relative RMS. The band the `cublas_mm` entry certifies its fp32-output router cells with: the
products of two bf16 values are exact in fp32, so the sides differ in summation order alone.
Measured on the model's inputs: at most 59.2 element and 27.3 RMS; the logits differ by at
most 9.3e-5."""

MANY_ROWS = 64
"""From this many rows on, a batch's relative RMS is a stable statistic; below it, one flipped
rounding at an amplifying spot can dominate a few 3072-wide rows."""

BLOCK_BANDS = {False: (12.0, 1.0), True: (12.0, 0.6)}
"""The block against `R_t` -- fp32 GEMMs, rounding where the target rounds -- and HF's own bf16
implementation `H` against `R_hf`, in bf16 ulps (element against the row's max |ref|, relative
RMS), keyed by `rows >= MANY_ROWS`.

Within each pair both sides round at the same points and differ in GEMM accumulation order alone.
That tips a bf16 rounding now and then, and the clamped relu² (slope up to 2c = 32) carries a
flipped up projection into the down projection. Measured: the target at most 5.95 element and
0.421 RMS from 64 rows (0.738 below); HF's implementation at most 5.95 and 0.412 (0.684 below) --
the same worst element, an output channel where both cuBLAS orders round alike, and from 64 rows
the two RMS figures within 1.2x of each other on every batch (0.93x to 1.19x). A target sits as
far from the fp32-GEMM reference as the checkpoint's own code does. So: element 12, twice the
measured worst; RMS 0.6 from 64 rows (1.4x either side's worst), 1.0 below (1.35x)."""

HF_ROUNDING_BANDS = {False: (12.0, 2.6), True: (12.0, 1.7)}
"""The block against the references that round `fc2_norm` twice, as `Nemotron4RMSNorm` does
(`R_hf`, `H`, `R_own`): the block band plus the measured size of that one rounding difference.

`flashinfer_rmsnorm` rounds once; the two land on neighbouring bf16 values on 20-29% of
`fc2_norm`'s outputs, and `fc2_latent_proj` spreads each flip over a whole output row. `R_t`
against `R_hf` measures at most 4.10 element and 1.096 RMS from 64 rows, 2.51 and 1.551 below.
So RMS 0.6 + 1.1 = 1.7 from 64 rows and 1.0 + 1.6 = 2.6 below; element 12."""

STAGE_BANDS = dict(fc1=(4.0, 2.0), experts=(4.0, 0.4))
"""Two ops on the target's own input against their references: `cublas_mm`'s certified band
(bf16 out, 4 / 2) for `fc1_latent_proj`, measured at most 1.90 / 0.039; the routed experts' ten
expert outputs summed and rounded once, at 4 / 0.4, measured at most 1.97 / 0.268. The tail --
`fc2_latent_proj`, the shared expert and the add, on the target's own `fc2_norm` output -- takes
the block band (measured at most 5.95 / 0.155 from 64 rows, 0.74 / 0.219 below), and `fc2_norm`
its entry's gate: within one bf16 spacing everywhere, and at most max(1, n / 1000) outputs not
bit-identical (measured: at most one in 3,000)."""

WEIGHT_SUM_TOLERANCE = 2.0**-20
"""|sum of a row's routing weights - routed_scaling_factor|, relative to the factor: ten weights
each rounded once to fp32 (2^-23 of the factor all told) and their fp32 sum (nine additions, at
most 9 * 2^-24 of the factor) stay under 5.5 * 2^-23 < 2^-20. Measured: at most 2.3e-7
(7.2e-7 absolute). A missing scale or renormalization misses by orders."""


def failures(metrics: dict) -> list[str]:
    """The gates a measured batch (`routing` as `routing`, `block` and `stages` as
    `block_and_stages`) breaks, by name; empty when every gate holds."""
    broken = []
    r = metrics["routing"]
    many = r["rows"] >= MANY_ROWS
    for name, ok in (
        ("routing: selection is noaux_tc_op's own arithmetic", r["selection_is_the_kernels"]),
        (
            "routing: logits",
            r["logit_element_ulp"] <= LOGITS_BAND[0] and r["logit_rms_ulp"] <= LOGITS_BAND[1],
        ),
        ("routing: flips only at near-ties", r["flips_beyond_near_tie"] == 0),
        ("routing: weights", r["weights_beyond_bound"] == 0),
        ("routing: weight sums", r["weight_sum_rel_dev"] <= WEIGHT_SUM_TOLERANCE),
    ):
        if not ok:
            broken.append(name)
    bands = dict(STAGE_BANDS, tail=BLOCK_BANDS[many])
    checks = [(f"stage: {stage}", metrics["stages"][stage], band) for stage, band in bands.items()]
    for key, band in (
        ("T vs R_t", BLOCK_BANDS[many]),
        ("H vs R_hf", BLOCK_BANDS[many]),
        ("T vs R_hf", HF_ROUNDING_BANDS[many]),
        ("T vs H", HF_ROUNDING_BANDS[many]),
        ("T vs R_own (agreeing rows)", HF_ROUNDING_BANDS[many]),
    ):
        if key in metrics["block"]:
            checks.append((f"block: {key}", metrics["block"][key], band))
    for name, d, (element, rms) in checks:
        if d["element_ulp"] > element or d["rms_ulp"] > rms:
            broken.append(name)
    d = metrics["stages"]["fc2_norm"]
    if d["element_ulp"] > 2.0 or d["differing"] > max(1, d["numel"] // 1000):
        broken.append("stage: fc2_norm")
    return broken
