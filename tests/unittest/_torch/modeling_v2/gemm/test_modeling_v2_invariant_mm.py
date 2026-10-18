# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the invariant_mm entry of nemotron4_nano_preview__sm_103__tp1.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the guard.
This file turns a cell into the operands a target hands the op, proves the property the entry
exists for -- a row's bits do not depend on how many rows share the call, or where in the call the
row sits -- against `cublas_mm` as the rival that lacks it, and holds the inputs the guard exists
to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.invariant_mm import (
    invariant_mm as op,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "invariant_mm requires a CUDA device"

BF16, FP32 = torch.bfloat16, torch.float32


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of summation-order difference, in the output's own precision.

    Kernel and reference consume the same bf16 operands, whose products are exact in fp32, so the
    two differ only in how they sum over K: the kernel's tcgen05 accumulation against torch's IEEE
    fp32 one with TF32 off. Both gates are scaled by the row (`assert_within_ulp`): random operands
    put a few outputs near zero, where any reordering reads as a large relative difference.

    bf16 out: 4 ulp (2^-8) of the row scale element-wise, 0.5 of relative RMS. The two fp32 sums
    agree to a few 1e-6 relative, so what is left is a rounding flip -- one bf16 spacing, at most 2 of
    these ulps at the top binade of a row. Measured over every cell: worst element 1.98 (that
    flip), worst RMS 0.070. The reference is the more accurate side: 0.019 RMS from a float64
    product, against the kernel's 0.070.

    fp32 out (the router and the lm_head): 1024 ulp (2^-23) element-wise, 512 of relative RMS. Here
    the fp32 sums are the output. Measured worst: 60.5 element and 27.8 RMS, the kernel again the
    less accurate side (27.7 RMS from float64, the reference 4.3). A result rounded to bf16 and
    widened back scores at least 18737 and 13689, so the gate refuses it by 18x and 27x; a band in
    bf16 ulp would let it through, which `test_fp32_out_refuses_a_bf16_rounded_result` keeps true.
    """
    if out.dtype is FP32:
        assert_within_ulp(out, ref, element_ulp=1024.0, rms_ulp=512.0, ulp=ULP[FP32])
    else:
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.5, ulp=ULP[BF16])


def _operands(rows: int, k: int, n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Activations of unit scale, as an RMS-normed input is, and a weight [N, K] as the checkpoint
    stores it, scaled by 1/sqrt(K) so every output sits near unit scale too."""
    x = torch.randn(rows, k, generator=g, device="cuda").to(BF16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(BF16)
    return x, weight


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x, weight = _operands(spec["m"], spec["k"], spec["n"], g)
    out = op(x, weight, spec["out_dtype"])
    assert out.shape == (spec["m"], spec["n"])
    assert out.dtype is (spec["out_dtype"] or BF16)
    _compare(out, op.reference(x, weight, spec["out_dtype"]))


def test_fp32_out_refuses_a_bf16_rounded_result() -> None:
    """fp32 out has to carry fp32's precision, and the gate has to see it.

    A product rounded to bf16 and widened is an fp32 tensor of the right shape whose every value
    sits within half a bf16 ulp of the truth: inside any band scored in bf16 ulp.
    """
    cell = next(c for c in op.CELLS if c.why.startswith("nemotron4 MoE router m=1024"))
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x, weight = _operands(spec["m"], spec["k"], spec["n"], g)
    ref = op.reference(x, weight, spec["out_dtype"])
    _compare(op(x, weight, spec["out_dtype"]), ref)
    with pytest.raises(AssertionError, match="not close"):
        _compare(ref.to(BF16).float(), ref)


#: Every row count the invariance test compares, a superset of the cells' row counts: decode
#: batches, MTP verify batches (3 per request) and prefill chunks, across every boundary at which
#: the entry changes its tiling.
INVARIANCE_ROWS = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    16,
    24,
    48,
    96,
    192,
    256,
    512,
    1024,
    1536,
    3072,
    8192,
    16384,
)

#: The distinct (k, n, out_dtype) the cells certify, named by their first call site.
SHAPES = tuple(
    {
        (c.spec["k"], c.spec["n"], c.spec["out_dtype"]): c.why.split(" m=")[0] for c in op.CELLS
    }.items()
)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit for bit, not value for value: `torch.equal` would let -0.0 stand for +0.0."""
    as_int = {BF16: torch.int16, FP32: torch.int32}[a.dtype]
    return a.dtype == b.dtype and torch.equal(a.view(as_int), b.view(as_int))


def _rows_that_move(mm, k: int, n: int, out_dtype, seed: int) -> list[str]:
    """Where a row's bits differ from the same row inside one call over all `INVARIANCE_ROWS[-1]`.

    Each row count is drawn twice: as a contiguous window at a random offset (a batch slice,
    every row at a new position within its tile) and as rows gathered from random positions (a
    batch in another order). Returns one label per call that disagreed.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = INVARIANCE_ROWS[-1]
    x, weight = _operands(total, k, n, g)
    full = mm(x, weight, out_dtype)
    moved = []
    for rows in INVARIANCE_ROWS[:-1]:
        start = int(torch.randint(0, total - rows + 1, (1,), generator=g, device="cuda").item())
        if not _same_bits(
            mm(x[start : start + rows], weight, out_dtype), full[start : start + rows]
        ):
            moved.append(f"{rows} rows at {start}")
        picked = torch.randperm(total, generator=g, device="cuda")[:rows]
        if not _same_bits(mm(x[picked], weight, out_dtype), full[picked]):
            moved.append(f"{rows} gathered rows")
    return moved


@pytest.mark.parametrize("shape", SHAPES, ids=[name for _, name in SHAPES])
def test_rows_do_not_depend_on_the_batch(shape) -> None:
    """A token's output row is the same bits whatever else shares the call.

    CUDA-graph padding changes the row count of a decode step, and MTP verify runs three rows per
    request where decode runs one; greedy identity across those needs every row computed alike.
    The windows and gathers span every tile change the entry makes, so this also proves its tile
    configurations agree bit for bit with one another.
    """
    (k, n, out_dtype), name = shape
    moved = _rows_that_move(op, k, n, out_dtype, seed=_seed(name))
    assert not moved, f"{name}: rows depend on the batch: {', '.join(moved)}"


#: The shapes on which `cublas_mm`'s rows were measured to depend on the row count.
CUBLAS_MOVES = (
    "o_proj and shared down",
    "dense MLP up",
    "dense MLP down",
    "MoE router",
    "fc1_latent_proj",
    "MTP eh_proj",
)


@pytest.mark.parametrize(
    "shape",
    [s for s in SHAPES if any(s[1].endswith(site) for site in CUBLAS_MOVES)],
    ids=lambda s: s[1],
)
def test_cublas_mm_rows_depend_on_the_batch(shape) -> None:
    """The rival, through the same test: cuBLAS picks its kernel by the row count.

    This is the failure the entry exists to remove, and the control that the test above can see
    it. If it stops failing, the hazard is gone for this shape, not the test.
    """
    (k, n, out_dtype), name = shape
    moved = _rows_that_move(
        lambda x, w, od: cublas_mm(x, w.t(), None, od), k, n, out_dtype, seed=_seed(name)
    )
    assert moved, (
        f"{name}: cublas_mm's rows no longer depend on the batch; the rival this test catches is gone"
    )


def test_graph_replay_is_the_eager_call() -> None:
    """A captured call replays the eager call's bits, at a decode and at a prefill row count."""
    (k, n, out_dtype), _ = SHAPES[0]
    g = torch.Generator(device="cuda").manual_seed(7)
    for rows in (3, 3072):
        x, weight = _operands(rows, k, n, g)
        eager = op(x, weight, out_dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(x, weight, out_dtype)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = op(x, weight, out_dtype)
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(captured, eager), f"{rows} rows: graph replay differs from the eager call"


def _ok():
    g = torch.Generator(device="cuda").manual_seed(3)
    return _operands(8, 256, 512, g)


def test_guard_refuses_a_weight_of_another_width() -> None:
    """K comes from `x`; the weight is read through a descriptor of its own shape.

    A narrower weight's missing columns read as zeros and a wider one's extra columns are never
    read: either way the op returns a product of the wrong matrices without complaint, which is
    what driving it unguarded shows first.
    """
    x, weight = _ok()
    narrow = weight[:, :192].contiguous()
    unguarded = op(x, narrow, None)
    assert unguarded.shape == (x.shape[0], weight.shape[0]), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="share K"):
        with validating(op):
            op(x, narrow, None)
    wide = torch.cat([weight, weight[:, :64]], dim=1)
    with pytest.raises(AssertionError, match="share K"):
        with validating(op):
            op(x, wide, None)


def test_guard_refuses_fp32_operands() -> None:
    """fp32 operands would run through the tensor cores as TF32, not the precision certified."""
    x, weight = _ok()
    with pytest.raises(AssertionError, match="must be bf16"):
        with validating(op):
            op(x.float(), weight.float(), None)


def test_out_dtype_other_than_bf16_or_fp32_is_refused() -> None:
    """Refused by the op itself rather than rounded through bf16 into another dtype."""
    x, weight = _ok()
    with pytest.raises(ValueError, match="bf16 or fp32"):
        op(x, weight, torch.float16)
