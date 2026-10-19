# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of the tp1 target's latent MoE block, on the checkpoint's own weights.

What runs is the target's own block: `DecodeTarget._moe` at the CUDA-graph decode batch sizes
and `PrefillTarget._moe` at the prefill chunk sizes -- router GEMM, `noaux_tc_op` top-10,
`fc1_latent_proj`, `latent_moe_experts_fused_act`, `fc2_norm`, `fc2_latent_proj`, plus the shared
expert. The decode block runs its router through `dsv3_router_gemm` up to 8 rows and each bf16
GEMM through the low-M kernel certified for its row count where there is one
(`DecodeTarget._mm`); the prefill block runs cuBLAS.
Both targets come from their own constructors over the target's own weight table
(`weights.MODEL_WEIGHTS`): every parameter is declared on the meta device, and the MoE layers
under test are filled from the checkpoint through the table's manifest, the way `load` fills
them. So the views `_layer_views` takes, and the arguments `_moe` passes each entry, are the
served engine's.

The inputs are the routed and shared inputs the model itself produces: the plain-torch
reference's blocks (`_nemotron4_reference`) run over the fixed prompt set, and each MoE call's
two inputs are captured as the reference makes it. Real hidden states put the router's
top-10/11 margins where the model puts them, which random activations would not.

Two MoE layers: 5, the first `shortcut_moe_attn` block, and 6, the first `shortcut_moe_gdp`
block. The MoE is the same module in both; the cheap prefix to them (the embedding, three GDP
mixers, two dense MLPs) keeps the test at ~31 GiB of GPU memory.

References (`_nemotron4_reference`, transcribed from the checkpoint's `modeling_nemotron4.py`):

* routing: `route`, `Nemotron4TopkRouter` with IEEE fp32 logits;
* `R_hf`: `latent_moe`'s chain at HF's rounding points with fp32 GEMMs -- the fp32-GEMM
  reference;
* `R_t`: the same chain with `fc2_norm` rounded once, as `flashinfer_rmsnorm` rounds it
  (`flashinfer_rmsnorm.reference`) -- a reference that rounds where the target rounds;
* `H`: `Nemotron4MoE.forward` in its own bf16 order, every projection a bf16 `F.linear` -- the
  checkpoint's own implementation.

The block references take the target's routing (ids and fp32 weights), so a near-tie the two
routers break differently costs no row; the routing is checked on its own.

What each batch must hold (`failures`; the bands and what they rest on are under "Bands"):

* routing -- the top-10 sets agree with HF's router except at near-ties the logits' summation
  order can explain, the weights are HF's formula (normalized, scaled, unbiased) up to what the
  scores' difference allows, and the selection is `noaux_tc_op`'s certified arithmetic on the
  target's own logits. The weights are gated directly because the block output cannot see them:
  `fc2_norm` divides any per-token scale of the routed experts back out;
* the block -- the target against `R_t`, as close as HF's own implementation `H` is to `R_hf`
  (both gated with one band), and against the HF-rounding references within that band plus the
  measured effect of rounding `fc2_norm` twice;
* each op on the target's own input (`fc1_latent_proj`, the experts, `fc2_norm`, the tail), which
  localizes a block-level miss.

`test_the_gates_refuse_plausible_wrong_blocks` plants one mistake at a time and shows the gates
named for it break; `test_the_decode_block_replays_in_a_cuda_graph` captures the decode block.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.softclamp_relu2 import (
    softclamp_relu2,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.dsv3_router_gemm import (
    dsv3_router_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_direct_mm import lowm_direct_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_splitk_mm import lowm_splitk_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts_fused_act import (  # noqa: E501
    latent_moe_experts_fused_act,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1 import (  # noqa: E501
    weights as W,
)

__extra_import_path__ = ["../..", "../../../.."]
import _nemotron4_prompts as P  # noqa: E402 — needs the path declared above
import _nemotron4_reference as R  # noqa: E402
from _validating import ULP, validating  # noqa: E402

assert torch.cuda.is_available(), "the nemotron4 MoE block test requires a CUDA device"

LAYERS = (5, 6)
"""The MoE layers under test: the first shortcut attention block and the first shortcut GDP block."""

DECODE_TOKENS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
"""`DecodeTarget`'s token counts: the CUDA-graph decode batch sizes up to 1024, and 3 -- an eager
batch off the power-of-two ladder, and one request's MTP verify."""

PREFILL_TOKENS = (37, 2048, 8192, 16384)
"""`PrefillTarget`'s token counts: a short prompt's tail, a 2048-token chunk, the 8192-token
`max_num_tokens` baseline and the 16384-token budget."""

CELLS = tuple(("decode", t) for t in DECODE_TOKENS) + tuple(("prefill", t) for t in PREFILL_TOKENS)

_BF16_ULP = ULP[torch.bfloat16]
_FP32_ULP = ULP[torch.float32]
_GUARDED = (
    cublas_mm,
    lowm_direct_mm,
    lowm_splitk_mm,
    dsv3_router_gemm,
    noaux_tc_op,
    latent_moe_experts_fused_act,
    flashinfer_rmsnorm,
    softclamp_relu2,
)


def _checkpoint_or_skip() -> str:
    root = os.environ.get("LLM_MODELS_ROOT")
    path = os.path.join(root, "nemotron4-nano-preview") if root else None
    if not path or not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
        pytest.skip("needs the checkpoint at $LLM_MODELS_ROOT/nemotron4-nano-preview")
    return path


# ----------------------------------------------------------------------------------------------
# The target, its weights and its inputs
# ----------------------------------------------------------------------------------------------


def build_targets(checkpoint: R.Checkpoint, layers) -> tuple[M.PrefillTarget, M.DecodeTarget]:
    """The two targets from their own constructors, `layers`' weights real and the rest meta.

    The text config is the one the engine hands the target (the checkpoint's own
    `configuration_nemotron4.py`). Each real parameter is filled from the checkpoint through the
    weight table's manifest -- its checkpoint keys, index and transform -- as `load` fills it.
    """
    from transformers import AutoConfig

    text_config = AutoConfig.from_pretrained(checkpoint.path, trust_remote_code=True).text_config
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(text_config=text_config),
            torch_dtype=torch.bfloat16,
            spec_config=None,
        )
    )
    table = W.MODEL_WEIGHTS
    d = table.dims(core)
    with torch.device("meta"):
        w = table.declare(d)
    wanted = {f"l{i}" for i in layers}
    for key, sources in table.manifest(core).items():
        if key.split("_", 1)[0] not in wanted:
            continue
        param = torch.empty(w[key].shape, dtype=w[key].dtype, device="cuda")
        for ckpt_key, index, transform in sources:
            src = checkpoint.tensor(ckpt_key)
            if transform is not None:
                src = transform(src, d)
            dst = param if index is None else param[index]
            assert (dst.shape, dst.dtype) == (src.shape, src.dtype), (
                ckpt_key,
                dst.shape,
                src.shape,
            )
            dst.copy_(src)
        w[key] = nn.Parameter(param, requires_grad=False)
    core.w, core.layer_types = w, d.layer_types
    # What the core sets without speculative decoding: no MTP layer, one token a generation row.
    core.mtp_layer, core.tokens_per_gen_step = None, 1
    return M.PrefillTarget(core), M.DecodeTarget(core)


def real_moe_inputs(checkpoint: R.Checkpoint, layers) -> tuple[dict, dict, int]:
    """Each layer's MoE inputs over the prompt set, as the reference's own blocks make them.

    Runs blocks 0..max(layers) of the reference over every prompt, long prompts first, and
    captures `(routed, shared)` at each `latent_moe` call. The GDP mixers run the reference's
    chunked (WY) form of the recurrence, its independent fp32 formulation of the per-token loop,
    because the loop over 8K-token prompts is slow; the inputs only have to be the model's own
    hidden states, not bit-exact ones.

    Returns:
        ``({layer: (routed [N, hidden], shared [N, hidden]) bf16}, {layer: the layer's ``moe.``
        weights as the checkpoint stores them}, the number of long-prompt rows that open N)``.
    """
    config = checkpoint.config
    prompts = P.prompt_set(P.load_tokenizer(checkpoint.path))
    prompts = [p for p in prompts if p.long] + [p for p in prompts if not p.long]
    last = max(layers)
    embed = checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")
    blocks = [checkpoint.layer(i) for i in range(last + 1)]
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
    inputs = {
        i: (torch.cat([r for r, _ in pairs]), torch.cat([s for _, s in pairs]))
        for i, pairs in captured.items()
    }
    weights = {i: R.sub(blocks[i], "moe.") for i in layers}
    num_long = sum(len(p.token_ids) for p in prompts if p.long)
    return inputs, weights, num_long


def cell_rows(phase: str, tokens: int, layer: int, num_rows: int, num_long: int) -> torch.Tensor:
    """Which rows of the captured inputs a cell runs.

    Prefill rows are contiguous, as a chunk's tokens are: the three long cells take disjoint
    stretches of the long prompts, and 37 tokens the tail of the last short prompt. Decode rows
    are a fixed random draw over every row: each is another request's current token.
    """
    if phase == "prefill":
        starts = {16384: 0, 8192: 16384, 2048: 24576}
        start = starts.get(tokens, num_rows - tokens)
        assert start + tokens <= (num_long if tokens in starts else num_rows)
        return torch.arange(start, start + tokens, device="cuda")
    g = torch.Generator(device="cpu").manual_seed(1000 * layer + tokens)
    return torch.randperm(num_rows, generator=g)[:tokens].to("cuda")


# ----------------------------------------------------------------------------------------------
# Running the target with its calls observed
# ----------------------------------------------------------------------------------------------

#: The calls of `_moe` the test observes: modeling's name for the entry, the entry, the record.
_OBSERVED = (
    ("noaux_tc_op", noaux_tc_op, "route"),
    ("latent_moe_experts_fused_act", latent_moe_experts_fused_act, "experts"),
    ("flashinfer_rmsnorm", flashinfer_rmsnorm, "fc2_norm"),
)


def _observed(call, signature: inspect.Signature, name: str, seen: dict):
    """`call`, passing through, recording its bound arguments and its result under `name`."""

    def observed(*args, **kwargs):
        out = call(*args, **kwargs)
        seen[name] = (signature.bind(*args, **kwargs).arguments, out)
        return out

    return observed


def run_target(target, layer: int, routed: torch.Tensor, shared: torch.Tensor, plant=None):
    """The target's `_moe` on one batch, with every entry's guard on.

    The router selection, the experts and `fc2_norm` are observed on the way -- what reached each
    and what came back -- which is what localizes a block-level miss to one op. `plant` maps a
    modeling name of `_OBSERVED` to a stand-in for that entry: how a rival plants its mistake.

    Returns:
        ``(out [N, hidden] bf16, the observed calls)``.
    """
    seen = {}
    plant = plant or {}
    with pytest.MonkeyPatch.context() as mp, validating(*_GUARDED):
        for attr, entry, name in _OBSERVED:
            call = plant.get(attr, entry)
            mp.setattr(M, attr, _observed(call, inspect.signature(entry.reference), name, seen))
        out = target._moe(routed, shared, target._layers[layer].moe)
    route_args, (weights, ids) = seen["route"]
    experts_args, experts = seen["experts"]
    _, fc2_norm = seen["fc2_norm"]
    return out, SimpleNamespace(
        logits=route_args["router_logits"],
        bias=route_args["bias"],
        weights=weights,
        ids=ids,
        latent=experts_args["x"],
        experts=experts,
        fc2_norm=fc2_norm,
    )


# ----------------------------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------------------------


def reference_chain(routed, shared, mw, config, ids, weights, fc2_norm):
    """The reference's `latent_moe` past its router, given the routing; `fc2_norm` pluggable.

    `linear` (an IEEE fp32 GEMM rounded once), `latent_experts`, then `fc2_norm`, `linear` and the
    shared `mlp`, each rounding where the checkpoint's code rounds.

    Args:
        routed, shared: [N, hidden] bf16, the block's two inputs.
        mw: the layer's ``moe.`` weights as the checkpoint stores them.
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


def _rms_norm_once(x, weight, eps):
    """`fc2_norm` rounded once, as `flashinfer_rmsnorm` rounds it (its own reference)."""
    return flashinfer_rmsnorm.reference(x, weight, eps)


# ----------------------------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------------------------


def distance(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """How far `out` sits from `ref`, scored as `assert_within_ulp` scores it.

    `element_ulp`: max |out - ref| over each row's max |ref|, in bf16 ulps (2^-8); `rms_ulp`:
    the relative RMS in the same unit; `max_abs`; `max_rel`: max |out - ref| / |ref| over the
    elements at least a tenth of their row's max; `identical`: the bit-identical fraction and
    `differing` the count of the others, of `numel`; `worst`: the element behind `element_ulp`.
    """
    o, r = out.float().reshape(-1, out.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    diff = (o - r).abs()
    scale = r.abs().amax(-1, keepdim=True).clamp_min(1e-9)
    element = diff / scale
    row, col = divmod(int(element.argmax()), o.shape[-1])
    rel = (diff / r.abs().clamp_min(1e-30))[r.abs() >= 0.1 * scale]
    rms = float((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9))
    return dict(
        element_ulp=float(element.max()) / _BF16_ULP,
        rms_ulp=rms / _BF16_ULP,
        max_abs=float(diff.max()),
        max_rel=float(rel.max()) if rel.numel() else 0.0,
        identical=float((o == r).float().mean()),
        differing=int((o != r).sum()),
        numel=o.numel(),
        worst=dict(row=row, col=col, out=float(o[row, col]), ref=float(r[row, col])),
    )


def routing(seen, routed, mw, config) -> tuple[dict, torch.Tensor]:
    """The target's routing against `Nemotron4TopkRouter` (`route`) on IEEE fp32 logits.

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
    bias = mw["gate.e_score_correction_bias"]
    logits_r, ids_r, w_r = R.route(routed, mw["gate.weight"], bias, top_k, scale)
    scores_t = 0.5 * torch.tanh(0.5 * seen.logits) + 0.5
    scores_r = logits_r.sigmoid()
    choice_r = scores_r + bias
    delta = ((scores_t + bias) - choice_r).abs().amax(-1)
    ranked = choice_r.sort(-1, descending=True).values
    margin = ranked[:, top_k - 1] - ranked[:, top_k]
    agree = (seen.ids.long().sort(-1).values == ids_r.sort(-1).values).all(-1)
    flips = (~agree).nonzero().squeeze(-1)

    num_rows, experts = choice_r.shape
    dense_t = torch.zeros(num_rows, experts, device="cuda").scatter_(
        1, seen.ids.long(), seen.weights
    )
    dense_r = torch.zeros(num_rows, experts, device="cuda").scatter_(1, ids_r, w_r)
    chosen = dense_r > 0
    weight_err = (dense_t - dense_r).abs()
    row_err = weight_err.amax(-1) / scale
    d = ((scores_t - scores_r).abs() * chosen).amax(-1)
    total = (scores_r * chosen).sum(-1)
    room = total - 10 * d
    bound = torch.where(room > 0, 11 * d / room.clamp_min(1e-30), torch.inf) + 2.0**-19
    weight_rel = (weight_err / dense_r.clamp_min(1e-30))[chosen & agree[:, None]]
    significant = chosen & agree[:, None] & (dense_r >= 1e-3 * scale)
    kernel_w, kernel_ids = noaux_tc_op.reference(seen.logits, seen.bias, 1, 1, top_k, scale)
    logit_diff = seen.logits - logits_r
    logit_scale = logits_r.abs().amax(-1, keepdim=True)

    metrics = dict(
        rows=num_rows,
        experts_touched=int(seen.ids.unique().numel()),
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
        weight_sum_rel_dev=float((seen.weights.sum(-1) - scale).abs().max()) / scale,
        selection_is_the_kernels=bool(
            torch.equal(kernel_ids, seen.ids) and torch.equal(kernel_w, seen.weights)
        ),
    )
    return metrics, agree


def measure(target, layer: int, routed, shared, mw, config, plant=None, target_shared=None):
    """One batch through the target and every reference.

    Args:
        target: `PrefillTarget` or `DecodeTarget`.
        routed, shared: [N, hidden] bf16, the block's two inputs.
        mw: the layer's ``moe.`` weights as the checkpoint stores them.
        plant, target_shared: a rival's planted entry (see `run_target`) and the shared input
            handed to the target in place of `shared`.

    Returns:
        The routing metrics (`routing`); the block distances (`block`: `T` the target, `R_t`,
        `R_hf` and `H` as in the module docstring, all on the target's routing, and `R_own`, the
        reference with its own routing, on the rows whose routing agrees); and each op given the
        target's own input (`stages`).
    """
    with torch.inference_mode():
        out, seen = run_target(
            target, layer, routed, shared if target_shared is None else target_shared, plant
        )
        route, agree = routing(seen, routed, mw, config)
        ids, weights = seen.ids, seen.weights
        r_t = reference_chain(routed, shared, mw, config, ids, weights, _rms_norm_once)
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
            fc2_norm=distance(
                seen.fc2_norm, _rms_norm_once(seen.experts, mw["fc2_norm.weight"], eps)
            ),
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
    return dict(routing=route, block=block, stages=stages)


# ----------------------------------------------------------------------------------------------
# Bands
# ----------------------------------------------------------------------------------------------
#
# Every block band rests on one measurement over all 28 MoE layers of the checkpoint: 232 batches,
# 153,232 rows of the model's own inputs -- this file's layers 5 and 6 over the prompt set, layers 24
# and 39 the same way (the reference prefix up to 39), and every MoE layer over the six short
# prompts, rebuilt from the ref-torch goldens' per-layer streams.

LOGITS_BAND = (1024.0, 512.0)
"""Router logits against IEEE fp32, in fp32 ulps (2^-23): element against the row's max |logit|,
and relative RMS. The band `cublas_mm` and `dsv3_router_gemm` certify their fp32-output router
cells with: the products of two bf16 values are exact in fp32, so the sides differ in summation
order alone.
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
the two RMS figures within 1.2x of each other on every batch (0.93x to 1.19x). The target sits as far from the
fp32-GEMM reference as the checkpoint's own code does. So: element 12, twice the measured worst;
RMS 0.6 from 64 rows (1.4x either side's worst), 1.0 below (1.35x)."""

HF_ROUNDING_BANDS = {False: (12.0, 2.6), True: (12.0, 1.7)}
"""The block against the references that round `fc2_norm` twice, as `Nemotron4RMSNorm` does
(`R_hf`, `H`, `R_own`): the block band plus the measured size of that one rounding difference.

`flashinfer_rmsnorm` rounds once; the two land on neighbouring bf16 values on 20-29% of
`fc2_norm`'s outputs (op-norm-cells measured ~26% with these weights), and `fc2_latent_proj`
spreads each flip over a whole output row. `R_t` against `R_hf` measures at most 4.10 element and
1.096 RMS from 64 rows, 2.51 and 1.551 below. So RMS 0.6 + 1.1 = 1.7 from 64 rows and
1.0 + 1.6 = 2.6 below; element 12."""

STAGE_BANDS = dict(fc1=(4.0, 2.0), experts=(4.0, 0.4))
"""Two ops on the target's own input against their references, at the entries' certified bands:
`cublas_mm`'s (bf16 out, 4 / 2), which the low-M entries certify too, for `fc1_latent_proj`,
measured at most 1.90 / 0.039; `latent_moe_experts_fused_act` (`latent_moe_experts`' bits and
band, 4 / 0.4), measured at most 1.97 / 0.268. The tail -- `fc2_latent_proj`, the
shared expert and the add, on the target's own `fc2_norm` output -- takes the block band (measured
at most 5.95 / 0.155 from 64 rows, 0.74 / 0.219 below), and `fc2_norm` its entry's gate: within
one bf16 spacing everywhere, and at most max(1, n / 1000) outputs not bit-identical (measured: at
most one in 3,000)."""

WEIGHT_SUM_TOLERANCE = 2.0**-20
"""|sum of a row's routing weights - routed_scaling_factor|, relative to the factor: ten weights
each rounded once to fp32 (2^-23 of the factor all told) and their fp32 sum (nine additions, at
most 9 * 2^-24 of the factor) stay under 5.5 * 2^-23 < 2^-20. Measured: at most 2.3e-7
(7.2e-7 absolute). A missing scale or renormalization misses by orders."""


def failures(metrics: dict) -> list[str]:
    """The gates a measured batch breaks, by name; empty when every gate holds."""
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


# ----------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint() -> R.Checkpoint:
    return R.Checkpoint(_checkpoint_or_skip())


@pytest.fixture(scope="module")
def inputs(checkpoint):
    return real_moe_inputs(checkpoint, LAYERS)


@pytest.fixture(scope="module")
def targets(checkpoint):
    prefill, decode = build_targets(checkpoint, LAYERS)
    return dict(prefill=prefill, decode=decode)


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("phase, tokens", CELLS, ids=[f"{p}{t}" for p, t in CELLS])
@pytest.mark.parametrize("layer", LAYERS, ids=[f"layer{i}" for i in LAYERS])
def test_moe_block(layer, phase, tokens, checkpoint, inputs, targets) -> None:
    """The target's block on one batch of the model's own inputs: the routing, the block output
    and each op within their bands (`failures`). The measured figures are printed (`-rP`)."""
    moe_inputs, moe_weights, num_long = inputs
    routed, shared = moe_inputs[layer]
    rows = cell_rows(phase, tokens, layer, routed.shape[0], num_long)
    metrics = measure(
        targets[phase], layer, routed[rows], shared[rows], moe_weights[layer], checkpoint.config
    )
    record = json.dumps(dict(layer=layer, phase=phase, tokens=tokens, **metrics))
    print(record)
    broken = failures(metrics)
    assert not broken, f"layer {layer}, {phase} at {tokens} tokens breaks {broken}: {record}"


@pytest.mark.parametrize("tokens", DECODE_TOKENS)
@pytest.mark.parametrize("layer", LAYERS, ids=[f"layer{i}" for i in LAYERS])
def test_the_decode_block_replays_in_a_cuda_graph(layer, tokens, inputs, targets) -> None:
    """`DecodeTarget._moe` captured once at a decode batch size, then replayed on other requests'
    rows: bitwise the eager call. The decode target runs inside CUDA graphs, so a host sync or a
    buffer sized from device data anywhere in the block would show here."""
    moe_inputs, _, _ = inputs
    routed, shared = moe_inputs[layer]
    decode = targets["decode"]
    moe = decode._layers[layer].moe
    g = torch.Generator(device="cpu").manual_seed(7 * tokens + layer)
    draws = [torch.randperm(routed.shape[0], generator=g)[:tokens].to("cuda") for _ in range(3)]
    with torch.inference_mode():
        static_routed, static_shared = routed[draws[0]], shared[draws[0]]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                decode._moe(static_routed, static_shared, moe)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = decode._moe(static_routed, static_shared, moe)
        for rows in draws[1:]:
            static_routed.copy_(routed[rows])
            static_shared.copy_(shared[rows])
            graph.replay()
            eager = decode._moe(routed[rows], shared[rows], moe)
            assert torch.equal(out, eager), f"layer {layer}: a replay at {tokens} tokens differs"


def _unscaled(router_logits, bias, n_group, topk_group, topk, routed_scaling_factor):
    return noaux_tc_op(router_logits, bias, n_group, topk_group, topk, 1.0)


def _bf16_weights(router_logits, bias, n_group, topk_group, topk, routed_scaling_factor):
    weights, ids = noaux_tc_op(
        router_logits, bias, n_group, topk_group, topk, routed_scaling_factor
    )
    return weights.to(torch.bfloat16).float(), ids


def _unbiased(router_logits, bias, n_group, topk_group, topk, routed_scaling_factor):
    zero = torch.zeros_like(bias)
    return noaux_tc_op(router_logits, zero, n_group, topk_group, topk, routed_scaling_factor)


def _weight_after_down_proj(x, topk_ids, topk_weights, up_proj, down_proj, clamp, offset, count):
    """The generic MoE order: each expert unweighted, its output then scaled by its weight."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for k in range(topk_ids.shape[1]):
        ids = topk_ids[:, k : k + 1].contiguous()
        unweighted = torch.ones_like(topk_weights[:, k : k + 1])
        o = latent_moe_experts_fused_act(
            x, ids, unweighted, up_proj, down_proj, clamp, offset, count
        )
        out += o.float() * topk_weights[:, k : k + 1]
    return out.to(x.dtype)


def _no_fc2_norm(x, weight, eps):
    return x


def _fc2_norm_rounded_twice(x, weight, eps):
    return R.rms_norm(x, weight, eps)


#: rival -> (entries planted in the target, whether the target's shared expert reads the routed
#: input, the gates that must break). Each is one plausible way to get the block wrong.
RIVALS = {
    "routing weights not scaled by routed_scaling_factor": (
        dict(noaux_tc_op=_unscaled),
        False,
        {"routing: weights", "routing: weight sums"},
    ),
    "routing weights rounded to bf16": (
        dict(noaux_tc_op=_bf16_weights),
        False,
        {"routing: weights", "routing: weight sums"},
    ),
    "selection without the correction bias": (
        dict(noaux_tc_op=_unbiased),
        False,
        {"routing: flips only at near-ties"},
    ),
    "the routing weight applied after down_proj": (
        dict(latent_moe_experts_fused_act=_weight_after_down_proj),
        False,
        {"block: T vs R_t", "stage: experts"},
    ),
    "fc2_norm left out": (dict(flashinfer_rmsnorm=_no_fc2_norm), False, {"block: T vs R_t"}),
    "fc2_norm rounded twice, as HF rounds it": (
        dict(flashinfer_rmsnorm=_fc2_norm_rounded_twice),
        False,
        {"block: T vs R_t", "stage: fc2_norm"},
    ),
    "the shared expert reading the routed input": ({}, True, {"block: T vs R_t", "stage: tail"}),
}


@pytest.mark.parametrize("rival", list(RIVALS))
def test_the_gates_refuse_plausible_wrong_blocks(rival, checkpoint, inputs, targets) -> None:
    """Each rival plants one mistake in the target's block -- an entry wrapped, or the shared
    expert handed the wrong input -- and the gates named in `RIVALS` must break.

    The unscaled routing weights also show why the weights are gated directly: `fc2_norm` divides
    any per-token scale of the routed experts back out, so the block output against the
    reference with its own, scaled routing stays within its band (measured on this batch: 1.16
    ulp of relative RMS against 0.87 for the target, inside the 1.7 of `HF_ROUNDING_BANDS`).
    """
    plant, shared_reads_routed, must_break = RIVALS[rival]
    moe_inputs, moe_weights, num_long = inputs
    layer = LAYERS[0]
    routed, shared = moe_inputs[layer]
    rows = cell_rows("prefill", 2048, layer, routed.shape[0], num_long)
    routed, shared = routed[rows], shared[rows]
    metrics = measure(
        targets["prefill"],
        layer,
        routed,
        shared,
        moe_weights[layer],
        checkpoint.config,
        plant=plant,
        target_shared=routed if shared_reads_routed else None,
    )
    broken = set(failures(metrics))
    print(json.dumps(dict(rival=rival, broken=sorted(broken), **metrics)))
    assert must_break <= broken, f"{rival}: {sorted(must_break - broken)} held"
    if rival == "routing weights not scaled by routed_scaling_factor":
        assert not {b for b in broken if b.startswith("block")}, sorted(broken)
