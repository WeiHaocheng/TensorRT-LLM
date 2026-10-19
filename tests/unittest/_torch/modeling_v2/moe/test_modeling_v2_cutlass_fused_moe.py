# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cutlass_fused_moe catalog entry: TensorRT-LLM's CUTLASS fused MoE on
nemotron4-nano-preview's routed experts, with the tanh soft-capped relu^2 (`Relu2` plus
`swiglu_limit`), at tp1's whole expert range, a dep4 rank's window and a tp4 rank's slice.

The entry owns what is certified: `CELLS`, `reference`, `is_valid`. This file turns a cell into
the call a target makes -- a latent batch, top-10 routing over 512 experts, the rank's expert
weights, the AutoTuner's tactics -- and holds what a cell cannot show: that the band refuses
relu^2 without the cap, that the four expert-parallel windows add up to the whole, that the
output stays close to the checkpoint's own loop (`latent_moe_experts.reference`), and the
output dtype the guard refuses.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.cutlass_fused_moe import (
    cutlass_fused_moe as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
    latent_moe_experts,
)
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.utils import ActivationType

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "cutlass_fused_moe requires a CUDA device"

#: The token budget the AutoTuner tunes to: the largest cell, a dep4 rank's gathered prefill.
_TUNE_MAX = 32768


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Two grouped GEMMs with the soft-capped relu^2 between them, against a reference that
    rounds h and the activation to bf16 as the kernel does and keeps the down-projection, the
    weighting and the sum in fp32 with fp32 GEMMs.

    What is left: the order the GEMMs accumulate in, the kernel's tanh.approx (relative error
    below 2^-10.9), and the fused finalize's sum over a token's ten experts, which does not
    round once at the end as the reference does. Measured on GB300 over every cell, each at its
    own seed and five more: at most 5.19 ulp element-wise against the row scale and 1.26 ulp of
    relative RMS (tp1 and tp4 cells 1.19-1.26, the dep4 windows 0.82-0.97); four windows'
    partials added in fp32 and rounded once, 3.26 and 0.95.

    So: 8 ulp element-wise, 1.6 of relative RMS. The plausible wrong answer is the backend's
    own relu^2 without the cap: at least 172 ulp element-wise and 194 of RMS on every cell
    (`test_the_band_refuses_relu2_without_the_cap`).
    """
    assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


_ELEMENT_ULP = 8.0
_RMS_ULP = 1.6


def _build(spec, seed):
    """One cell as the entry's keyword arguments.

    up_proj is scaled so h has a standard deviation of ~8: about one value in twenty sits past
    the cap's knee (|h| > 16), so the tanh is exercised and not only its linear middle.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, experts, top_k = spec["tokens"], spec["num_experts"], spec["top_k"]
    hidden, inter = spec["hidden"], spec["intermediate"]
    n_local = experts // spec["ep_size"]
    x = torch.randn(tokens, hidden, generator=g, device="cuda").to(torch.bfloat16)
    scores = torch.rand(tokens, experts, generator=g, device="cuda")
    ids = scores.topk(top_k, dim=-1).indices.to(torch.int32).contiguous()
    w = torch.rand(tokens, top_k, generator=g, device="cuda") + 0.05
    w = (w / w.sum(-1, keepdim=True) * 3.16).contiguous()
    up = torch.randn(n_local, inter, hidden, generator=g, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(n_local, hidden, inter, generator=g, device="cuda", dtype=torch.bfloat16)
    return dict(
        input=x,
        token_selected_experts=ids,
        token_final_scales=w,
        fc1_expert_weights=(up * (8.0 / hidden**0.5)).contiguous(),
        fc1_expert_biases=None,
        fc2_expert_weights=(down * (1.0 / inter**0.5)).contiguous(),
        fc2_expert_biases=None,
        output_dtype=torch.bfloat16,
        quant_scales=[],
        swiglu_limit=torch.full((n_local,), spec["limit"], dtype=torch.float32, device="cuda"),
        tp_size=spec["tp_size"],
        tp_rank=spec["tp_rank"],
        ep_size=spec["ep_size"],
        ep_rank=spec["ep_rank"],
        tune_max_num_tokens=_TUNE_MAX,
        activation_type=int(ActivationType.Relu2),
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _run(args):
    """The call a target makes after the engine's warm-up: tactics tuned, then the op."""
    with autotune():
        op(**args)
    return op(**args)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))
    out = _run(args)
    assert out.shape == args["input"].shape and out.dtype == torch.bfloat16
    _compare(out, op.reference(**args))


@pytest.mark.parametrize("tokens", [1, 512, 8192])
def test_the_band_refuses_relu2_without_the_cap(tokens) -> None:
    """The plausible wrong answer: the backend's own relu^2, the cap ignored. Without the limit
    the op is plain relu^2, and that stays what it was, within its own reference's band: without
    the cap the activation reaches ~(3 h)^2, a wider range for the finalize's sum. Measured over
    tokens 1 / 37 / 512 / 8192 at eight seeds each: at most 5.22 ulp element-wise and 1.65 of
    relative RMS (1.43-1.65 at one token, whose ten rows make the RMS noisy; 1.19-1.20 at the
    others). So 8 and 2.2."""
    cell = op.CELLS[0]
    args = _build(dict(cell.spec, tokens=tokens), seed=tokens)
    capped_ref = op.reference(**args)
    plain = _run(dict(args, swiglu_limit=None))
    assert_within_ulp(
        plain, op.reference(**dict(args, swiglu_limit=None)), element_ulp=8.0, rms_ulp=2.2
    )
    with pytest.raises(AssertionError):
        _compare(plain, capped_ref)


def test_the_four_windows_add_up_to_the_whole() -> None:
    """Four expert-parallel windows' bf16 partials, added in fp32 and rounded once, against the
    whole range in one call: each window's slots are the whole call's, rounded once more."""
    spec = dict(op.CELLS[4].spec, tokens=2048)
    args = _build(spec, seed=7)
    whole = _run(args)
    total = torch.zeros(whole.shape, dtype=torch.float32, device="cuda")
    for r in range(4):
        sl = slice(128 * r, 128 * (r + 1))
        part = _run(
            dict(
                args,
                fc1_expert_weights=args["fc1_expert_weights"][sl],
                fc2_expert_weights=args["fc2_expert_weights"][sl],
                swiglu_limit=args["swiglu_limit"][sl],
                ep_size=4,
                ep_rank=r,
            )
        )
        total += part.float()
    _compare(total.to(torch.bfloat16), op.reference(**args))


@pytest.mark.parametrize("tokens", [1, 512, 8192])
def test_close_to_the_checkpoints_loop(tokens) -> None:
    """Against `latent_moe_experts.reference` (the checkpoint's loop: routing weight before
    down_proj, bf16 rounding at each step): the same sum, rounded at other points. Measured on
    GB300 over every cell: at most 5.57 ulp element-wise and 1.70 of relative RMS; this entry's
    own reference sits 1.30-1.35 ulp of RMS from that loop, the rounding points alone. So 8 and
    2.2."""
    cell = op.CELLS[0]
    args = _build(dict(cell.spec, tokens=tokens), seed=100 + tokens)
    out = _run(args)
    hf = latent_moe_experts.reference(
        args["input"],
        args["token_selected_experts"],
        args["token_final_scales"],
        args["fc1_expert_weights"],
        args["fc2_expert_weights"],
        cell.spec["limit"],
        0,
        cell.spec["num_experts"],
    )
    assert_within_ulp(out, hf, element_ulp=_HF_ELEMENT_ULP, rms_ulp=_HF_RMS_ULP)


_HF_ELEMENT_ULP = 8.0
_HF_RMS_ULP = 2.2


def test_guard_refuses_an_fp32_output_on_bf16_activations() -> None:
    """The hazard `is_valid` exists for: accepted, no error, and the values are wrong (measured:
    relative RMS error 1.2-1.7 against the reference at 1, 64 and 4096 tokens, whole range or an
    expert-parallel window, tuned or not)."""
    args = _build(dict(op.CELLS[0].spec, tokens=64), seed=3)
    ref = op.reference(**dict(args, output_dtype=torch.float32))
    unguarded = op(**dict(args, output_dtype=torch.float32)).float()
    rel = (unguarded - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert not bool(torch.isfinite(unguarded).all()) or rel > 0.5, (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="returns wrong values"):
        with validating(op):
            op(**dict(args, output_dtype=torch.float32))
