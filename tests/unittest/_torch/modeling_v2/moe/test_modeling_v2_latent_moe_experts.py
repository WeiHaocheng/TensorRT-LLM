# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the latent_moe_experts entry of nemotron4_nano_preview__sm_103__tp1.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into what the target hands the entry -- a latent batch, top-10
routing over 512 experts, the window's expert weights -- and holds what a cell cannot show:
the activation pass on every bf16 input, rows that do not depend on their batch, CUDA-graph
replay, the band's distance from the plausible wrong orders, and the input the guard refuses.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
    _clamped_relu2_weighted_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
    latent_moe_experts as op,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "latent_moe_experts requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Two GEMMs with an activation between them, against a reference whose GEMMs are fp32.

    Entry and reference round at the same points -- h, the activation, the weighted y, each
    expert's o, the sum -- and differ only in the order the grouped GEMMs accumulate in fp32.
    Now and then that tips an h or an o across a bf16 rounding boundary, and the activation
    (slope 2c, up to 32 at the clamp) carries a flipped h into y. Measured over every cell on
    these inputs: at most 1.95 ulp element-wise against the row scale and 0.160 ulp of relative
    RMS (other seeds: up to 2.63 and 0.159) -- and the checkpoint's own loop, its GEMMs on
    cuBLAS, sits the same distance from this reference (1.95 and 0.158).

    So: 4 ulp element-wise, 0.4 of relative RMS. The RMS gate is the one that separates the
    plausible wrong order: weighting after down_proj measures 0.91 to 0.93 ulp of RMS on every
    cell while its element error stays within 2.2 ulp; relu² without the clamp misses both gates
    by two orders (202 ulp and more). `test_the_band_refuses_the_plausible_wrong_orders` drives
    both.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.4)


def _routing(spec, g):
    """x, ids, weights: a latent batch and top-k routing, uniform over the experts.

    Ids are distinct per token; weights normalized and scaled by 3.16 as the router's are.
    """
    tokens, experts, top_k = spec["tokens"], spec["num_experts"], spec["top_k"]
    x = torch.randn(tokens, spec["latent"], generator=g, device="cuda").to(torch.bfloat16)
    scores = torch.rand(tokens, experts, generator=g, device="cuda")
    ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    w = torch.rand(tokens, top_k, generator=g, device="cuda") + 0.05
    return x, ids, w / w.sum(-1, keepdim=True) * 3.16


def _build(spec, seed):
    """One cell as the entry's keyword arguments.

    up_proj is scaled so h has a standard deviation of ~8: about one value in twenty sits past
    the clamp's knee (|h| > 16), so the tanh is exercised and not only its linear middle.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x, ids, w = _routing(spec, g)
    experts, latent, inter = spec["num_experts"], spec["latent"], spec["intermediate"]
    up = torch.randn(experts, inter, latent, generator=g, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(experts, latent, inter, generator=g, device="cuda", dtype=torch.bfloat16)
    off, n = spec["local_expert_offset"], spec["num_local_experts"]
    return dict(
        x=x,
        topk_ids=ids,
        topk_weights=w,
        up_proj=(up * (8.0 / latent**0.5))[off : off + n],
        down_proj=(down * (1.0 / inter**0.5))[off : off + n],
        activation_clamp=spec["activation_clamp"],
        local_expert_offset=off,
        num_local_experts=n,
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))
    out = op(**args)
    assert out.shape == args["x"].shape and out.dtype == torch.bfloat16
    _compare(out, op.reference(**args))


@pytest.mark.parametrize("weight", [1.0, 0.31640625, 3.16, 0.123456789])
def test_the_activation_pass_is_the_checkpoints_on_every_bf16_value(weight) -> None:
    """Every bf16 bit pattern through the entry's activation pass and through the checkpoint's
    torch ops (`Nemotron4ClampedActivation`, relu2, the product with the fp32 routing weight):
    the same bits, and NaN exactly where the checkpoint gives NaN.

    A cell samples the activation; this covers its whole input space, which leaves the GEMMs'
    accumulation order as the only difference the band has to hold.
    """
    vals = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16)
    vals = vals.view(torch.bfloat16).reshape(16, 4096)
    w = torch.full((16,), weight, dtype=torch.float32, device="cuda")
    got = vals.clone()
    _clamped_relu2_weighted_kernel[(16,)](
        got,
        w,
        torch.arange(16, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        got.stride(0),
        N=4096,
        CLAMP=16.0,
        BLOCK_N=4096,
        num_warps=8,
    )
    clamped = (16.0 * torch.tanh(vals.float() / 16.0)).to(torch.bfloat16)
    want = (torch.square(torch.relu(clamped)) * w[:, None]).to(torch.bfloat16)
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan)
    differ = got.view(torch.int16)[~nan] != want.view(torch.int16)[~nan]
    assert not differ.any(), f"{int(differ.sum())} of {int((~nan).sum())} bf16 inputs differ"


def test_a_rows_result_does_not_depend_on_its_batch() -> None:
    """The first n tokens computed alone and inside a 2048-token batch: bit-identical.

    Batch composition and CUDA-graph padding change what surrounds a request, and with it the
    grouped GEMMs' group sizes. A row whose bits moved with them would make greedy decoding
    depend on its neighbours, so this is measured rather than assumed.
    """
    args = _build(dict(op.CELLS[6].spec), seed=123)
    full = op(**args)
    for n in (1, 8, 37, 64, 256, 1024):
        part = op(
            **dict(
                args,
                x=args["x"][:n],
                topk_ids=args["topk_ids"][:n],
                topk_weights=args["topk_weights"][:n],
            )
        )
        assert torch.equal(part, full[:n]), f"the first {n} tokens change when computed alone"


@pytest.mark.parametrize("tokens", [1, 64, 256])
def test_a_captured_graph_replays_the_eager_result(tokens) -> None:
    """Captured once at a decode size, replayed on new activations and routing: bitwise the
    eager call. Nothing on the path syncs the host or sizes a buffer from device data."""
    spec = dict(op.CELLS[0].spec, tokens=tokens)
    static = _build(spec, seed=7)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            op(**static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**static)
    for seed in (11, 12):
        x, ids, w = _routing(spec, torch.Generator(device="cuda").manual_seed(seed))
        static["x"].copy_(x)
        static["topk_ids"].copy_(ids)
        static["topk_weights"].copy_(w)
        graph.replay()
        eager = op(**dict(static, x=x, topk_ids=ids, topk_weights=w))
        assert torch.equal(out, eager), f"replay {seed} differs from the eager call"


def _rival(args, which: str) -> torch.Tensor:
    """`reference` with one plausible mistake, everything else identical."""
    x, ids, w = args["x"], args["topk_ids"], args["topk_weights"]
    clamp = args["activation_clamp"]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    local = ids.long() - args["local_expert_offset"]
    for e in range(args["num_local_experts"]):
        tok, pos = torch.where(local == e)
        if tok.numel() == 0:
            continue
        with true_fp32_matmul():
            h = (x[tok].float() @ args["up_proj"][e].float().t()).to(torch.bfloat16)
        if which != "no_clamp":
            h = (clamp * torch.tanh(h.float() / clamp)).to(torch.bfloat16)
        s = torch.square(torch.relu(h))
        wk = w[tok, pos, None]
        y = s if which == "weight_after_down_proj" else (s * wk).to(torch.bfloat16)
        with true_fp32_matmul():
            o = (y.float() @ args["down_proj"][e].float().t()).to(torch.bfloat16).float()
        out.index_add_(0, tok, o * wk if which == "weight_after_down_proj" else o)
    return out.to(x.dtype)


@pytest.mark.parametrize("which", ["weight_after_down_proj", "no_clamp"])
def test_the_band_refuses_the_plausible_wrong_orders(which) -> None:
    """The two easiest mistranscriptions of `Nemotron4Experts` fail `_compare`.

    Weighting after down_proj is the generic-MoE order the checkpoint's code says it departs
    from; relu² without the clamp is the activation TensorRT-LLM's fused-MoE kernels offer.
    """
    args = _build(dict(op.CELLS[3].spec), seed=_seed(op.CELLS[3]))
    ref = op.reference(**args)
    _compare(op(**args), ref)
    with pytest.raises(AssertionError):
        _compare(_rival(args, which), ref)


def test_a_window_no_slot_falls_into_gives_zeros() -> None:
    """An expert-parallel rank whose experts this step's tokens all missed: every group empty,
    and the output exactly zero rather than whatever the buffers held."""
    args = _build(dict(op.CELLS[0].spec), seed=5)
    args["topk_ids"] = torch.arange(10, dtype=torch.int32, device="cuda")[None, :]
    window = dict(local_expert_offset=448, num_local_experts=64)
    out = op(
        **dict(args, up_proj=args["up_proj"][448:], down_proj=args["down_proj"][448:], **window)
    )
    assert torch.equal(out, torch.zeros_like(out))


def test_guard_refuses_routing_weights_laid_out_against_the_ids() -> None:
    """[K, T] weights hold the right values in the wrong places; read flat, they are taken."""
    args = _build(dict(op.CELLS[3].spec), seed=_seed(op.CELLS[3]))
    swapped = args["topk_weights"].t().contiguous()
    unguarded = op(**dict(args, topk_weights=swapped))
    with pytest.raises(AssertionError):
        _compare(unguarded, op.reference(**args))
    with pytest.raises(AssertionError, match="must have topk_ids' shape"):
        with validating(op):
            op(**dict(args, topk_weights=swapped))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_guard_refuses_activations_and_weights_in_another_dtype(dtype) -> None:
    """fp16 or fp32 throughout runs, with bf16's rounding inside: outside the band, in that
    dtype's own ulp, of what the checkpoint's code computes in that dtype (`reference` follows
    the dtype as the checkpoint's code does)."""
    args = _build(dict(op.CELLS[3].spec), seed=_seed(op.CELLS[3]))
    cast = dict(
        args,
        x=args["x"].to(dtype),
        up_proj=args["up_proj"][:16].to(dtype),
        down_proj=args["down_proj"][:16].to(dtype),
        num_local_experts=16,
    )
    unguarded = op(**cast)
    with pytest.raises(AssertionError):
        _compare(unguarded, op.reference(**cast))
    with pytest.raises(AssertionError, match="must be bf16"):
        with validating(op):
            op(**cast)
