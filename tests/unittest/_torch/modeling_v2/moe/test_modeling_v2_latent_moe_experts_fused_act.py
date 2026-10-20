# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the latent_moe_experts_fused_act catalog entry.

The entry claims `latent_moe_experts`' bits on that entry's cells, so each cell is checked twice: against
the shared reference within `latent_moe_experts`' band, and bit for bit against `latent_moe_experts`
itself. This file also holds what a cell cannot show: the approximate tanh giving the exact activation on
every bf16 input, which batches take the fused-activation path, its bits at the sizes, windows, slices and
clamps between the cells, rows that do not depend on their batch across the switch between the paths,
CUDA-graph replay, an empty window, and the inputs the guard refuses.
"""

import zlib

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
    _FUSED_TILES,
    _clamped_relu2_weighted,
    _fp32_reciprocal,
    latent_moe_experts,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts_fused_act import (
    _APPROX_TANH_CLAMPS,
    _clamped_relu2_weighted_approx,
    _takes_fused_act,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts_fused_act import (
    latent_moe_experts_fused_act as op,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "latent_moe_experts_fused_act requires a CUDA device"

#: The largest batch `latent_moe_experts`' decode path takes; past it, this entry's fused-activation path.
_DECODE_MOST = _FUSED_TILES[-1][0]


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """`latent_moe_experts`' band for the bf16 output: 6 ulp element-wise, 0.4 of relative RMS.

    Derived there (`test_modeling_v2_latent_moe_experts.py`, `_compare`) from the same reference, the same
    inputs and that entry's measured distances, and it applies unchanged: every cell here also asserts that
    this entry's output is that entry's, bit for bit, so the two sit at exactly the same distance.
    """
    assert_within_ulp(out, ref, element_ulp=6.0, rms_ulp=0.4)


def _compare_fp32_sum(out: torch.Tensor, ref: torch.Tensor) -> None:
    """`latent_moe_experts`' band for the fp32 output: 4 ulp element-wise, 0.3 of relative RMS, in bf16
    ulp (every operand is bf16); inherited for the same reason as `_compare`'s."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.3, ulp=ULP[torch.bfloat16])


def _routing(spec, g):
    """x, ids, weights: a latent batch and top-k routing, uniform over the experts; distinct ids per token,
    weights normalized and scaled by 3.16 as the router's are (`latent_moe_experts`' test's inputs)."""
    tokens, experts, top_k = spec["tokens"], spec["num_experts"], spec["top_k"]
    x = torch.randn(tokens, spec["latent"], generator=g, device="cuda").to(torch.bfloat16)
    scores = torch.rand(tokens, experts, generator=g, device="cuda")
    ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    w = torch.rand(tokens, top_k, generator=g, device="cuda") + 0.05
    return x, ids, w / w.sum(-1, keepdim=True) * 3.16


def _build(spec, seed):
    """One cell as the entry's keyword arguments, built as `latent_moe_experts`' test builds them: up_proj
    scaled so h has a standard deviation of ~8, about one value in twenty past the clamp's knee."""
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


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _spec(tokens, offset=0, count=512, out_dtype=None, intermediate=4096, clamp=16.0):
    return dict(
        op.CELLS[0].spec,
        tokens=tokens,
        local_expert_offset=offset,
        num_local_experts=count,
        out_dtype=out_dtype,
        intermediate=intermediate,
        activation_clamp=clamp,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))
    out = op(**args)
    fp32 = args["out_dtype"] == torch.float32
    assert out.shape == args["x"].shape
    assert out.dtype == (torch.float32 if fp32 else torch.bfloat16)
    assert torch.equal(out, latent_moe_experts(**args)), "not latent_moe_experts' bits"
    (_compare_fp32_sum if fp32 else _compare)(out, op.reference(**args))


@triton.jit
def _both_activations_kernel(
    bits_ptr,
    exact_ptr,
    approx_ptr,
    w,
    n,
    CLAMP: tl.constexpr,
    INV_CLAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    h = tl.load(bits_ptr + offs, mask=m, other=0).to(tl.bfloat16, bitcast=True).to(tl.float32)
    exact = _clamped_relu2_weighted(h, w, CLAMP, INV_CLAMP)
    approx = _clamped_relu2_weighted_approx(h, w, CLAMP, INV_CLAMP)
    tl.store(exact_ptr + offs, exact.to(tl.int16, bitcast=True), mask=m)
    tl.store(approx_ptr + offs, approx.to(tl.int16, bitcast=True), mask=m)


@pytest.mark.parametrize("weight", [1.0, 0.31640625, 3.16, 0.123456789])
@pytest.mark.parametrize("clamp", sorted(_APPROX_TANH_CLAMPS), ids=lambda c: f"clamp {c:g}")
def test_the_approx_epilogue_is_the_exact_activation_on_every_bf16_value(clamp, weight) -> None:
    """The fused-activation epilogue's tanh.approx.f32 against the grouped path's libdevice tanhf, on all
    65536 bf16 values of h, NaNs and infinities included: the same bits for every one.

    The weighted result depends on h only through s = bf16(c²): the clamp c itself does come out one
    subnormal ulp apart on subnormal h (639 values at clamp 16, measured), and c² flushes both to the same
    zero. So equality here holds for any fp32 weight; the four weights are the router's typical, a bf16
    value, the scale, and one off every grid. A clamp missing from `_APPROX_TANH_CLAMPS` is not covered,
    which is why such a clamp takes the exact epilogue.
    """
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32, device="cuda").to(torch.int16)
    exact = torch.empty_like(bits)
    approx = torch.empty_like(bits)
    n = bits.numel()
    _both_activations_kernel[(triton.cdiv(n, 1024),)](
        bits, exact, approx, weight, n, CLAMP=clamp, INV_CLAMP=_fp32_reciprocal(clamp), BLOCK=1024
    )
    # The comparison can see the activation at all: a clamp's worth of distinct outputs, not a constant.
    assert torch.unique(exact).numel() > 1000
    differ = exact != approx
    assert not differ.any(), f"{int(differ.sum())} of {n} bf16 inputs differ"


def test_the_fused_act_path_takes_the_prompt_sized_batches() -> None:
    """Which calls the fused-activation path takes: every batch past the decode path's largest, on bf16
    operands with a contiguous up_proj; anything else runs `latent_moe_experts`' own path for it."""
    args = _build(_spec(_DECODE_MOST + 1), seed=3)
    assert _takes_fused_act(args["x"], args["topk_ids"], args["up_proj"], args["down_proj"])
    small = dict(args, x=args["x"][:_DECODE_MOST], topk_ids=args["topk_ids"][:_DECODE_MOST])
    assert not _takes_fused_act(small["x"], small["topk_ids"], args["up_proj"], args["down_proj"])
    strided = args["up_proj"][:, :2048]
    assert not strided.is_contiguous()
    assert not _takes_fused_act(
        args["x"], args["topk_ids"], strided, args["down_proj"][:, :, :2048]
    )
    half = args["x"].half()
    assert not _takes_fused_act(
        half, args["topk_ids"], args["up_proj"].half(), args["down_proj"].half()
    )
    for cell in op.CELLS:
        spec = cell.spec
        x = torch.empty(spec["tokens"], spec["latent"], dtype=torch.bfloat16, device="cuda")
        ids = torch.empty(spec["tokens"], spec["top_k"], dtype=torch.int32, device="cuda")
        up = torch.empty(
            1, spec["intermediate"], spec["latent"], dtype=torch.bfloat16, device="cuda"
        )
        down = torch.empty(
            1, spec["latent"], spec["intermediate"], dtype=torch.bfloat16, device="cuda"
        )
        assert _takes_fused_act(x, ids, up, down) == (spec["tokens"] > _DECODE_MOST), cell.why


@pytest.mark.parametrize(
    "tokens, offset, count, out_dtype, intermediate, clamp",
    [
        (_DECODE_MOST + 1, 0, 512, None, 4096, 16.0),
        (3000, 0, 512, None, 4096, 16.0),
        (4096, 128, 128, torch.float32, 4096, 16.0),
        (4096, 256, 128, None, 4096, 16.0),
        (2048, 0, 512, torch.float32, 1024, 16.0),
        (2048, 0, 512, None, 4096, 10.0),
        (2048, 0, 512, None, 4096, 12.0),
    ],
    ids=[
        "just past the decode path",
        "3000, groups off every tile multiple",
        "4096 gathered, dep4 rank 1, fp32",
        "4096 gathered, dep4 rank 2, bf16",
        "2048, tp4 slice, fp32",
        "clamp 10, the other approximate clamp",
        "clamp 12, the exact epilogue",
    ],
)
def test_the_fused_act_path_gives_latent_moe_experts_bits(
    tokens, offset, count, out_dtype, intermediate, clamp
) -> None:
    """The fused-activation path between and beside the cells: the windows the cells skip, a group size off
    every tile multiple, both approximate clamps and one that takes the exact epilogue -- each
    `latent_moe_experts`' output, bit for bit."""
    args = _build(
        _spec(tokens, offset, count, out_dtype, intermediate, clamp), seed=tokens + offset
    )
    assert _takes_fused_act(args["x"], args["topk_ids"], args["up_proj"], args["down_proj"])
    assert torch.equal(op(**args), latent_moe_experts(**args))


@pytest.mark.parametrize(
    "tokens, intermediate", [(2048, 4096), (8192, 1024)], ids=["tp1 2048", "tp4 rank's slice 8192"]
)
def test_a_repeated_call_launches_the_cached_kernels_with_the_same_bits(
    tokens, intermediate
) -> None:
    """The fused-activation path's kernels go through `Launch`: a call whose specialization is
    cached goes straight to the compiled kernels, with the up GEMM's host-built TMA descriptors
    among its arguments. The first call of a fresh process compiles through Triton's own launch;
    the second and later calls take the cached path. Both give `latent_moe_experts`' bits."""
    args = _build(_spec(tokens, intermediate=intermediate), seed=tokens + intermediate)
    assert _takes_fused_act(args["x"], args["topk_ids"], args["up_proj"], args["down_proj"])
    want = latent_moe_experts(**args)
    for _ in range(3):
        assert torch.equal(op(**args), want)


@pytest.mark.parametrize(
    "offset, count, out_dtype, intermediate",
    [(0, 512, None, 4096), (384, 128, torch.float32, 4096), (0, 512, None, 1024)],
    ids=["tp1 window, bf16", "dep4 rank 3 window, fp32", "tp4 rank's slice, bf16"],
)
def test_a_rows_result_does_not_depend_on_its_batch(offset, count, out_dtype, intermediate) -> None:
    """The first n tokens computed alone and inside a larger batch: bit-identical, across the switch from
    `latent_moe_experts`' decode path (n up to its largest) to this entry's fused-activation path (past it),
    as a decode step's rows and a prefill's must be."""
    args = _build(_spec(2 * _DECODE_MOST, offset, count, out_dtype, intermediate), seed=123)
    full = op(**args)
    for n in (1, 37, _DECODE_MOST, _DECODE_MOST + 1, 1500):
        part = dict(
            args,
            x=args["x"][:n],
            topk_ids=args["topk_ids"][:n],
            topk_weights=args["topk_weights"][:n],
        )
        assert _takes_fused_act(
            part["x"], part["topk_ids"], args["up_proj"], args["down_proj"]
        ) == (n > _DECODE_MOST)
        assert torch.equal(op(**part), full[:n]), f"the first {n} tokens change when computed alone"


@pytest.mark.parametrize(
    "tokens, offset, count, out_dtype, intermediate",
    [
        (2048, 0, 512, None, 4096),
        (4096, 384, 128, torch.float32, 4096),
        (2048, 0, 512, None, 1024),
    ],
    ids=["2048", "4096 gathered, dep4 rank 3, fp32", "2048, tp4 slice"],
)
def test_a_captured_graph_replays_the_eager_result(
    tokens, offset, count, out_dtype, intermediate
) -> None:
    """Captured once at a prompt size, replayed on new activations and routing: bitwise the eager call.
    The persistent grid is the SM count and the work list lives on the device, so nothing syncs the host
    or sizes a launch from device data; the descriptors are baked into the graph with their buffers."""
    spec = _spec(tokens, offset, count, out_dtype, intermediate)
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


@pytest.mark.parametrize("out_dtype", [None, torch.float32], ids=["bf16", "fp32"])
def test_a_window_no_slot_falls_into_gives_zeros(out_dtype) -> None:
    """A prompt-sized batch whose slots all miss an expert-parallel rank's window: an empty work list, and
    the output exactly zero rather than whatever the buffers held."""
    tokens = 2 * _DECODE_MOST
    args = _build(_spec(tokens, 384, 128, out_dtype), seed=5)
    args["topk_ids"] = (
        torch.arange(10, dtype=torch.int32, device="cuda")[None, :].expand(tokens, 10).contiguous()
    )
    assert _takes_fused_act(args["x"], args["topk_ids"], args["up_proj"], args["down_proj"])
    out = op(**args)
    assert out.dtype == (out_dtype or torch.bfloat16)
    assert torch.equal(out, torch.zeros_like(out))


def test_guard_refuses_routing_weights_laid_out_against_the_ids() -> None:
    """[K, T] weights hold the right values in the wrong places; read flat, they are taken -- on the
    fused-activation path too, which reads them in its epilogue."""
    args = _build(_spec(_DECODE_MOST + 1), seed=9)
    swapped = args["topk_weights"].t().contiguous()
    unguarded = op(**dict(args, topk_weights=swapped))
    with pytest.raises(AssertionError):
        _compare(unguarded, op.reference(**args))
    with pytest.raises(AssertionError, match="must have topk_ids' shape"):
        with validating(op):
            op(**dict(args, topk_weights=swapped))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_guard_refuses_activations_and_weights_in_another_dtype(dtype) -> None:
    """fp16 or fp32 throughout runs (on `latent_moe_experts`' path, which this entry hands it), with
    bf16's rounding inside: outside the band, in that dtype's own ulp, of what the checkpoint's code
    computes in that dtype."""
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
