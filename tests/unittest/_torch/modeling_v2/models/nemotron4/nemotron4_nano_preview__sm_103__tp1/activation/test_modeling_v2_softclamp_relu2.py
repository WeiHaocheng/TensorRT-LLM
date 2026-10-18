# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the softclamp_relu2 entry of nemotron4_nano_preview__sm_103__tp1.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the up-projection output a target hands the op, drives every
bf16 bit pattern through both sides, and holds the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1.ops.activation.softclamp_relu2 import (  # noqa: E501
    softclamp_relu2 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "softclamp_relu2 requires a CUDA device"

BF16 = torch.bfloat16

#: The checkpoint's `activation_clamp_value`.
CLAMP = 16.0

#: Scale of the cells' Gaussian inputs: about 5% of them lie past the clamp, so the tanh's
#: linear, curved and saturated stretches are all sampled at every size. Every value a bf16
#: input can take is driven by `test_every_bf16_input` regardless.
INPUT_SCALE = 8.0


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _assert_bitwise(x: torch.Tensor, out: torch.Tensor, ref: torch.Tensor) -> None:
    """The same bits as the reference wherever it is a number, and NaN wherever it is NaN.

    There is no band to derive: kernel and reference run the same fp32 operations in the same
    order -- the multiply by fl32(1 / c), tanhf, the multiply by c, the rounding to x's dtype,
    relu, one fp32 square rounded once -- so any difference is a different computation. The
    two tanhf (libdevice's, and the one torch's CUDA kernel calls) were measured bit-identical
    on 2^24 fp32 arguments at each of five scales; they part only on subnormal arguments,
    which libdevice flushes to zero, and an argument that small squares to zero on both sides.
    NaN payloads are left out of the claim (they were measured identical, 0x7fff on both
    sides); NaN-ness is not.
    """
    assert out.shape == ref.shape and out.dtype == ref.dtype, (out.shape, out.dtype, ref.dtype)
    ref_nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(out), ref_nan), "NaN where HF has a number, or the reverse"
    differ = (_bits(out) != _bits(ref)) & ~ref_nan
    count = int(differ.sum())
    if count:
        at = differ.reshape(-1).nonzero()[:4].squeeze(-1)
        pairs = zip(
            x.reshape(-1)[at].tolist(), out.reshape(-1)[at].tolist(), ref.reshape(-1)[at].tolist()
        )
        raise AssertionError(
            f"{count} of {out.numel()} outputs differ from HF's; (x, out, ref): {list(pairs)}"
        )


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(spec["tokens"], spec["width"], generator=g, device="cuda")
    return x.mul_(INPUT_SCALE).to(BF16), spec["activation_clamp"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, clamp = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = x.clone()
    out = op(x, clamp)
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    assert torch.equal(x, before), "the op must not write its input"
    _assert_bitwise(x, out, op.reference(x, clamp))


def _every_bf16() -> torch.Tensor:
    """All 65536 bf16 bit patterns: zeros, subnormals, normals, both infinities, 254 NaNs."""
    return (
        torch.arange(-(2**15), 2**15, dtype=torch.int32, device="cuda").to(torch.int16).view(BF16)
    )


def test_every_bf16_input(certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """The whole input domain at the checkpoint's clamp, through the op and through HF's order.

    An element-wise op on a 16-bit input has no other inputs, so this certifies the arithmetic
    for every value; the cells certify the launch geometry around it.

    The controls show that bitwise agreement is a claim these inputs can refute. HF's rounding
    of the clamp to bf16 before relu² is observable: keeping fp32 to the end and rounding once
    moves 447 of the 65282 finite outputs, by one or two ulps. The clamp itself moves 16328.
    The floors below sit well under those counts and far above zero.
    """
    certifying(op)
    x = _every_bf16()
    ref = op.reference(x, CLAMP)
    _assert_bitwise(x, op(x, CLAMP), ref)

    finite = torch.isfinite(ref)
    fp32_to_the_end = torch.square(torch.relu(CLAMP * torch.tanh(x.float() / CLAMP))).to(BF16)
    unclamped = torch.square(torch.relu(x))
    rounding_moves = int(((_bits(fp32_to_the_end) != _bits(ref)) & finite).sum())
    clamp_moves = int(((_bits(unclamped) != _bits(ref)) & finite).sum())
    assert rounding_moves >= 100, f"HF's clamp rounding changed only {rounding_moves} outputs"
    assert clamp_moves >= 10000, f"the clamp changed only {clamp_moves} outputs"


def test_rows_do_not_depend_on_the_batch() -> None:
    """A row comes back as the same bits whatever batch it rides in.

    CUDA-graph padding and MTP verify change T for the same request, and the identity gates
    compare such runs bitwise. Element-wise, this holds by construction; it is pinned anyway.
    """
    x, clamp = _build(dict(tokens=1024, width=12288, activation_clamp=CLAMP), seed=5)
    full = op(x, clamp)
    for start, stop in ((0, 1), (37, 40), (1021, 1024)):
        assert torch.equal(op(x[start:stop], clamp), full[start:stop]), (start, stop)


def test_a_captured_graph_replays_bitwise() -> None:
    """Captured once, replayed on new inputs, the same bits as eager and as HF's order."""
    spec = dict(tokens=64, width=4096, activation_clamp=CLAMP)
    static_x, clamp = _build(spec, seed=6)
    op(static_x, clamp)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_x, clamp)
    for seed in (7, 8):
        fresh, _ = _build(spec, seed=seed)
        static_x.copy_(fresh)
        graph.replay()
        assert torch.equal(static_out, op(fresh, clamp)), f"replay differs from eager (seed {seed})"
        _assert_bitwise(fresh, static_out, op.reference(fresh, clamp))


def test_zero_tokens_launch_nothing() -> None:
    x = torch.empty(0, 12288, dtype=BF16, device="cuda")
    out = op(x, CLAMP)
    assert out.shape == x.shape and out.dtype == BF16


def test_a_partial_program_is_masked() -> None:
    """Both of the checkpoint's widths fill whole programs; a ragged size must be exact too."""
    g = torch.Generator(device="cuda").manual_seed(9)
    x = (torch.randn(3, 1000, generator=g, device="cuda") * INPUT_SCALE).to(BF16)
    _assert_bitwise(x, op(x, CLAMP), op.reference(x, CLAMP))


def test_the_division_is_torchs_scalar_division() -> None:
    """HF divides by a Python float, which torch computes as a multiply by fl32(1 / c).

    At the checkpoint's power-of-two clamp that equals the IEEE quotient, and on bf16 inputs
    the two forms were measured to agree for every bit pattern at other clamps too. On fp32
    inputs at a clamp that is not a power of two they part (over a mix of input scales, about
    a tenth of the outputs at c = 3), so that is where the kernel's choice of form is visible.
    No target runs fp32 through this op; the test pins the arithmetic the kernel states it does.
    """
    g = torch.Generator(device="cuda").manual_seed(10)
    x = torch.randn(1 << 20, generator=g, device="cuda").mul_(37.0)
    clamp = 3.0
    ref = op.reference(x, clamp)
    _assert_bitwise(x, op(x, clamp), ref)
    ieee = torch.square(torch.relu(clamp * torch.tanh(torch.div(x, torch.full_like(x, clamp)))))
    moved = int((_bits(ieee) != _bits(ref)).sum())
    assert moved > x.numel() // 100, f"IEEE division moved only {moved} outputs: no control"


@pytest.mark.parametrize("layout", ["column slice", "transpose"])
def test_guard_refuses_a_non_contiguous_input(layout) -> None:
    """Read flat from data_ptr(), any other layout comes back as other elements' activations."""
    g = torch.Generator(device="cuda").manual_seed(11)
    buf = (torch.randn(64, 8192, generator=g, device="cuda") * INPUT_SCALE).to(BF16)
    x = buf[:, :4096] if layout == "column slice" else buf.t()
    assert not x.is_contiguous()
    unguarded = op(x, CLAMP)
    assert not torch.equal(unguarded, op.reference(x, CLAMP)), (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="contiguous"), validating(op):
        op(x, CLAMP)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float64], ids=["fp8", "fp64"])
def test_guard_refuses_an_unmeasured_dtype(dtype) -> None:
    """Both run without complaint, and neither answer is HF's.

    An fp8 x -- an activation someone forgot to dequantize -- comes back as a plausible fp8
    relu² (HF's relu raises on it). An fp64 x has its square taken in fp32, where HF takes it
    in fp64.
    """
    g = torch.Generator(device="cuda").manual_seed(12)
    x = (torch.randn(4, 4096, generator=g, device="cuda") * INPUT_SCALE).to(dtype)
    unguarded = op(x, CLAMP)
    assert unguarded.dtype == dtype, "the hazard this guard covers is gone"
    if dtype == torch.float64:
        assert not torch.equal(unguarded, op.reference(x, CLAMP)), (
            "the hazard this guard covers is gone"
        )
    with pytest.raises(AssertionError, match="bf16, fp16 or fp32"), validating(op):
        op(x, CLAMP)
