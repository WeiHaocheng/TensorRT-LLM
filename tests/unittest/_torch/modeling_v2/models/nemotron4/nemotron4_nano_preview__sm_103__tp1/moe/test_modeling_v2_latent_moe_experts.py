# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the latent_moe_experts entry of nemotron4_nano_preview__sm_103__tp1.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into what the target hands the entry -- a latent batch, top-10
routing over 512 experts, the window's expert weights -- and holds what a cell cannot show:
the activation pass on every bf16 input, rows that do not depend on their batch, CUDA-graph
replay, the band's distance from the plausible wrong orders, what the fp32 output is, the
four expert-parallel windows adding up to the whole, and the input the guard refuses.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1.ops.moe.latent_moe_experts import (  # noqa: E501
    _clamped_relu2_weighted_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1.ops.moe.latent_moe_experts import (  # noqa: E501
    latent_moe_experts as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
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
    RMS over the whole expert range (other seeds: up to 2.63 and 0.159), and 2.74 and 0.123 on
    the expert-parallel windows (five other seeds each: up to 3.28, and 0.203 at 4 tokens,
    where 3072 values make the RMS noisy) -- and the checkpoint's own loop, its GEMMs on
    cuBLAS, sits the same distance from this reference (1.95 and 0.158; 2.74 and 0.118).

    So: 6 ulp element-wise, 0.4 of relative RMS. A window's row sums two or three experts
    rather than ten, so one flipped o is a larger share of the row's scale, and a 32768-token
    batch draws that tail 25 million times: the element gate sits at about twice the windows'
    worst on their own inputs. A real slip -- a slot added to the wrong token, an expert's rows
    shifted -- moves its elements by a whole expert's contribution, tens of ulp and more. The
    RMS gate is the one that separates the plausible wrong order: weighting after down_proj
    measures 0.87 to 0.99 ulp of RMS on every cell and seed while its element error stays
    within 3.4 ulp; relu² without the clamp misses both gates by two orders (200 ulp and more).
    `test_the_band_refuses_the_plausible_wrong_orders` drives both.
    """
    assert_within_ulp(out, ref, element_ulp=6.0, rms_ulp=0.4)


def _compare_fp32_sum(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The fp32 output: the sum `_compare` gates, without its final rounding.

    Entry and reference differ by the same flipped h and o roundings, now without the output's
    own rounding on top. Scored in bf16 ulp rather than fp32's: every operand is bf16, so
    fp32's ulp would measure the container. Measured over every fp32 cell on these inputs: at
    most 1.95 ulp element-wise and 0.107 of relative RMS (five other seeds each: up to 2.02 and
    0.133); the checkpoint's own loop on cuBLAS, its fp32 buffer taken before the rounding, sits
    at 1.95 and 0.102.

    So: 4 ulp element-wise, 0.3 of relative RMS. Without the output rounding the plausible
    wrong order moves less too -- weighting after down_proj measures 0.67 to 0.76 ulp of RMS --
    so the RMS gate sits lower than `_compare`'s, about 2.2 times from either side.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.3, ulp=ULP[torch.bfloat16])


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
        out_dtype=spec["out_dtype"],
    )


def _window(args, offset: int, count: int, **over):
    """`args` (built for the whole expert range) narrowed to one window, as a rank holds it."""
    return dict(
        args,
        up_proj=args["up_proj"][offset : offset + count],
        down_proj=args["down_proj"][offset : offset + count],
        local_expert_offset=offset,
        num_local_experts=count,
        **over,
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))
    out = op(**args)
    fp32 = args["out_dtype"] == torch.float32
    assert out.shape == args["x"].shape
    assert out.dtype == (torch.float32 if fp32 else torch.bfloat16)
    (_compare_fp32_sum if fp32 else _compare)(out, op.reference(**args))


_FP32_CELLS = [c for c in op.CELLS if c.spec["out_dtype"] == torch.float32]


@pytest.mark.parametrize("cell", _FP32_CELLS, ids=[c.why[:44] for c in _FP32_CELLS])
def test_the_bf16_output_is_the_fp32_sum_rounded_once(cell) -> None:
    """What the fp32 output is: the sum the default output rounds, before that rounding.

    Rounded to bf16 it gives the default call's bits, on every cell certified in fp32 -- so a
    target that switches to fp32 partials changes the rounding and nothing else.
    """
    args = _build(cell.spec, seed=_seed(cell))
    fp32 = op(**args)
    assert torch.equal(fp32.to(torch.bfloat16), op(**dict(args, out_dtype=None)))


#: The four ranks' windows of a 4-rank expert-parallel target, (128 r, 128).
_EP_WINDOWS = [(128 * r, 128) for r in range(4)]


def _sum_of_partials(args, windows) -> torch.Tensor:
    """Each window's fp32 partial sums, added in fp32 in rank order."""
    parts = [op(**_window(args, off, n, out_dtype=torch.float32)) for off, n in windows]
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def _compare_reassociated(total: torch.Tensor, full: torch.Tensor) -> None:
    """Four fp32 partial sums added in fp32, against the full window's one fp32 sum.

    When every slot's down-projection row has the bits the full-window call gives it, both
    add the same bf16 values, grouped differently, and fp32 holds most such sums exactly.
    Measured at each gathered size over nine seeds: at least 99.96% of the elements are
    identical, the rest at most 0.90 fp32 ulp of the row scale apart, 0.011 of relative RMS.

    So: 4 fp32 ulp element-wise, 0.05 of relative RMS. A row computed with other bits would
    sit a bf16 rounding away, thousands of fp32 ulp; an expert that falls between two windows,
    or into two, misses by five million (`test_the_partition_band_refuses_a_boundary_slip`).
    """
    assert_within_ulp(total, full, element_ulp=4.0, rms_ulp=0.05, ulp=ULP[torch.float32])


@pytest.mark.parametrize("tokens", [4, 64, 1024, 8192, 32768])
def test_the_four_windows_partition_the_experts(tokens) -> None:
    """A 4-rank expert-parallel target's reduction, done in fp32: the four windows' partial
    sums, added, are the full window's sum up to fp32 reassociation. Every slot is computed in
    exactly one window, with the bits the full-window call gives it -- so the partials,
    rounded once after the reduction, give the full window's bf16 result.
    """
    args = _build(dict(op.CELLS[0].spec, tokens=tokens), seed=tokens)
    full = op(**dict(args, out_dtype=torch.float32))
    _compare_reassociated(_sum_of_partials(args, _EP_WINDOWS), full)


@pytest.mark.parametrize(
    "windows",
    [
        [(0, 128), (129, 127), (256, 128), (384, 128)],
        [(0, 128), (127, 129), (256, 128), (384, 128)],
    ],
    ids=["expert 128 in no window", "expert 127 in two windows"],
)
def test_the_partition_band_refuses_a_boundary_slip(windows) -> None:
    """A window boundary off by one drops an expert's contributions or counts them twice;
    at 1024 tokens every expert is routed to, and the sum misses the band by orders."""
    args = _build(dict(op.CELLS[0].spec, tokens=1024), seed=1024)
    full = op(**dict(args, out_dtype=torch.float32))
    with pytest.raises(AssertionError):
        _compare_reassociated(_sum_of_partials(args, windows), full)


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


@pytest.mark.parametrize(
    "offset, count, out_dtype",
    [(0, 512, None), (384, 128, torch.float32)],
    ids=["tp1 window, bf16", "dep4 rank 3 window, fp32"],
)
def test_a_rows_result_does_not_depend_on_its_batch(offset, count, out_dtype) -> None:
    """The first n tokens computed alone and inside a 2048-token batch: bit-identical.

    Batch composition and CUDA-graph padding change what surrounds a request, and with it the
    grouped GEMMs' group sizes -- and on an expert-parallel rank, every other rank's tokens
    join the gathered batch too. A row whose bits moved with them would make greedy decoding
    depend on its neighbours, so this is measured rather than assumed.
    """
    spec = dict(
        op.CELLS[6].spec, local_expert_offset=offset, num_local_experts=count, out_dtype=out_dtype
    )
    args = _build(spec, seed=123)
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


@pytest.mark.parametrize(
    "tokens, offset, count, out_dtype",
    [(1, 0, 512, None), (64, 0, 512, None), (256, 0, 512, None), (64, 384, 128, torch.float32)],
    ids=["1", "64", "256", "64 gathered, dep4 rank 3, fp32"],
)
def test_a_captured_graph_replays_the_eager_result(tokens, offset, count, out_dtype) -> None:
    """Captured once at a decode size, replayed on new activations and routing: bitwise the
    eager call. Nothing on the path syncs the host or sizes a buffer from device data."""
    spec = dict(
        op.CELLS[0].spec,
        tokens=tokens,
        local_expert_offset=offset,
        num_local_experts=count,
        out_dtype=out_dtype,
    )
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
    out_dtype = args["out_dtype"] if args["out_dtype"] is not None else x.dtype
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
    return out.to(out_dtype)


#: The fp32 cell the rivals are driven at: dep4 rank 3's window at 64 gathered tokens.
_FP32_RIVAL_CELL = next(
    c for c in _FP32_CELLS if c.spec["tokens"] == 64 and c.spec["local_expert_offset"] == 384
)


@pytest.mark.parametrize("which", ["weight_after_down_proj", "no_clamp"])
@pytest.mark.parametrize(
    "cell, compare",
    [(op.CELLS[3], _compare), (_FP32_RIVAL_CELL, _compare_fp32_sum)],
    ids=["bf16", "fp32"],
)
def test_the_band_refuses_the_plausible_wrong_orders(which, cell, compare) -> None:
    """The two easiest mistranscriptions of `Nemotron4Experts` fail the band, in each output.

    Weighting after down_proj is the generic-MoE order the checkpoint's code says it departs
    from; relu² without the clamp is the activation TensorRT-LLM's fused-MoE kernels offer.
    """
    args = _build(dict(cell.spec), seed=_seed(cell))
    ref = op.reference(**args)
    compare(op(**args), ref)
    with pytest.raises(AssertionError):
        compare(_rival(args, which), ref)


@pytest.mark.parametrize("out_dtype", [None, torch.float32], ids=["bf16", "fp32"])
def test_a_window_no_slot_falls_into_gives_zeros(out_dtype) -> None:
    """An expert-parallel rank whose experts this step's tokens all missed: every group empty,
    and the output exactly zero rather than whatever the buffers held."""
    args = _build(dict(op.CELLS[0].spec), seed=5)
    args["topk_ids"] = torch.arange(10, dtype=torch.int32, device="cuda")[None, :]
    out = op(**_window(args, 384, 128, out_dtype=out_dtype))
    assert out.dtype == (out_dtype or torch.bfloat16)
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
