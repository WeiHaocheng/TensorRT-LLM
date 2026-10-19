# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gated_rmsnorm_multirow catalog entry, at nemotron4-nano-preview's tp1 and
tp4-rank shapes.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the tensors a target hands the op -- the recurrence output
and the gate columns of an in_proj-shaped buffer -- and holds the inputs the guard exists to
refuse.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.gated_rmsnorm import gated_rmsnorm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.gated_rmsnorm_multirow import (
    _MANY_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.gated_rmsnorm_multirow import (
    gated_rmsnorm_multirow as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gated_rmsnorm_multirow requires a CUDA device"

#: Bound on the fraction of outputs that may differ from the reference at all; see `_compare`.
MAX_MISMATCH_FRACTION = 1e-3

#: The head counts the cells certify: tp1's 24 and a tp4 rank's 6.
HEADS = sorted({c.spec["num_heads"] for c in op.CELLS}, reverse=True)


def _cell_spec(num_heads: int, tokens: int):
    """The spec of the cell at this head count and token count."""
    return next(
        c.spec for c in op.CELLS if c.spec["num_heads"] == num_heads and c.spec["tokens"] == tokens
    )


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise distance between two bf16 tensors in ulps, counted on the bit patterns."""

    def index(t: torch.Tensor) -> torch.Tensor:
        # Adjacent bf16 values map to adjacent integers, and +0 and -0 to the same one.
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (index(a) - index(b)).abs()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The final rounding may differ, and nothing else.

    Kernel and reference compute the same fp32 expression from the same bf16 inputs and round
    once to bf16. The kernel's sigmoid, its 64-term tree sum and its `1 / sqrt` are not torch's
    `F.silu`, `mean` and `rsqrt`, but they differ by a few fp32 ulps, ~1e-7 relative, so an
    output can only land on the neighbouring bf16 value, and only when its fp32 value sits on a
    rounding boundary. Measured over every cell at five seeds (survey-small-ops): the 1- and
    3-token cells are bit-identical; at most 4.1e-5 of the outputs differ at all (one element of
    the 64-token cell at 6 heads), 1.3e-5 to 2.5e-5 from 1024 tokens up, each by exactly one ulp of
    its own value. Against `gated_rmsnorm`, the entry it replaces, every output is within one ulp.

    Two gates, `gated_rmsnorm`'s:
      * every output within one bf16 ulp of its own value, counted on the bit patterns. The
        structural rivals that entry's test lists miss by far more: the gate after the norm by
        124 ulps or more, one 1536-wide group by 858 or more, a sigmoid gate by 32661 or more.
      * at most 1e-3 of the outputs not bit-identical: 25x the worst measured. Rounding the gate
        product to bf16 before the norm, the cheapest wrong order, changes 34 to 38% of them.
    """
    assert out.dtype == ref.dtype and out.shape == ref.shape, (out.dtype, out.shape, ref.shape)
    distance = _ulp_distance(out, ref)
    worst = int(distance.max().item())
    assert worst <= 1, f"an output is {worst} bf16 ulps from the reference"
    mismatch = (distance != 0).float().mean().item()
    assert mismatch <= MAX_MISMATCH_FRACTION, (
        f"{mismatch:.2e} of the outputs differ from the reference at all"
    )


def _build(spec, seed):
    """Materialize one cell as `(proj, kwargs)`, as `gated_rmsnorm`'s test does.

    x is the recurrence output, a contiguous `[T, heads * 64]`; z is the first `heads * 64`
    columns of an in_proj-shaped `[T, proj_width]` buffer. Every (token, head) group of x gets its
    own magnitude, log-uniform over [1e-3, 3]: the smallest groups' mean square sits below eps,
    where eps sets the output. The gate pre-activations are N(0, 4), so silu's negative tail is
    covered. The weight follows the checkpoint's 32 `mixer.norm.weight` tensors: mean 1.06,
    standard deviation 0.26, every value in [0.044, 3.03].
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, heads, group = spec["tokens"], spec["num_heads"], spec["group_size"]
    dtype = spec["dtype"]
    width = heads * group
    log_scale = torch.empty(tokens, heads, 1, device="cuda")
    log_scale.uniform_(math.log(1e-3), math.log(3.0), generator=g)
    x = torch.randn(tokens, heads, group, generator=g, device="cuda") * log_scale.exp()
    x = x.to(dtype).view(tokens, width)
    proj = (2.0 * torch.randn(tokens, spec["proj_width"], generator=g, device="cuda")).to(dtype)
    weight = (1.0 + 0.25 * torch.randn(width, generator=g, device="cuda")).clamp(0.04, 3.0)
    kwargs = dict(
        x=x, z=proj[:, :width], weight=weight.to(dtype), eps=spec["eps"], group_size=group
    )
    return proj, kwargs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    proj, kwargs = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    x_before, proj_before = kwargs["x"].clone(), proj.clone()
    ref = op.reference(**kwargs)
    out = op(**kwargs)
    assert out.shape == kwargs["x"].shape and out.dtype == kwargs["x"].dtype
    assert torch.equal(kwargs["x"], x_before), "x was written"
    assert torch.equal(proj, proj_before), "the in_proj buffer the gate views was written"
    _compare(out, ref)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_within_an_ulp_of_gated_rmsnorm(cell) -> None:
    """The catalog's `gated_rmsnorm` computes the same expression, so a target can swap one for
    the other: every output within one bf16 ulp of that entry's."""
    _, kwargs = _build(cell.spec, seed=3)
    out = op(**kwargs)
    old = gated_rmsnorm(
        kwargs["x"],
        kwargs["weight"],
        None,
        kwargs["eps"],
        z=kwargs["z"],
        group_size=kwargs["group_size"],
        norm_before_gate=False,
        is_rms_norm=True,
    )
    assert int(_ulp_distance(out, old).max().item()) <= 1


@pytest.mark.parametrize("num_heads", HEADS)
def test_both_launches_agree_within_an_ulp(num_heads: int) -> None:
    """A row normalized by the many-rows launch (a 16384-token call) and by the few-rows one (the
    same rows alone) lands within one ulp: the launch moves only the sum's tree, and both are
    within one ulp of the reference."""
    _, kwargs = _build(_cell_spec(num_heads, 16384), seed=7)
    assert 16384 * num_heads >= _MANY_ROWS > 3 * num_heads, "the slices must cross the switch"
    full = op(**kwargs)
    for begin, end in ((0, 1), (7, 10), (5000, 5064), (16383, 16384)):
        part = op(**dict(kwargs, x=kwargs["x"][begin:end], z=kwargs["z"][begin:end]))
        assert int(_ulp_distance(part, full[begin:end]).max().item()) <= 1, (
            f"rows [{begin}:{end}] alone are more than one ulp from a 16384-row call's"
        )
    assert torch.equal(op(**kwargs), full), "two identical calls differ"


@pytest.mark.parametrize("num_heads", HEADS)
def test_a_packed_gate_gives_the_same_bits(num_heads: int) -> None:
    """The gate's token stride is only an address: a packed copy of the in_proj slice gives the
    same bits, so a target may pass either."""
    _, kwargs = _build(_cell_spec(num_heads, 1024), seed=11)
    sliced = op(**kwargs)
    packed = op(**dict(kwargs, z=kwargs["z"].contiguous()))
    assert torch.equal(sliced, packed)


def _small():
    _, kwargs = _build(_cell_spec(24, 64), seed=5)
    return kwargs


def test_guard_refuses_a_strided_x() -> None:
    """Read as dense rows: a column slice of a wider buffer is normalized from the wrong memory."""
    kwargs = _small()
    wide = torch.randn(64, 2048, device="cuda").to(torch.bfloat16)
    wide[:, :1536] = kwargs["x"]
    x = wide[:, :1536]
    unguarded = op(**dict(kwargs, x=x))
    assert int(_ulp_distance(unguarded, op.reference(**dict(kwargs, x=x))).max().item()) > 1, (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="must be a contiguous"):
        with validating(op):
            op(**dict(kwargs, x=x))


def test_guard_refuses_a_group_that_does_not_divide_the_channels() -> None:
    kwargs = _small()
    with pytest.raises(AssertionError, match="does not divide"):
        with validating(op):
            op(**dict(kwargs, group_size=100))


def test_guard_refuses_a_gate_with_a_channel_stride() -> None:
    """z is read with unit channel stride whatever its strides say."""
    kwargs = _small()
    z = torch.randn(64, 1536 * 2, device="cuda").to(torch.bfloat16)[:, ::2]
    with pytest.raises(AssertionError, match="unit channel stride"):
        with validating(op):
            op(**dict(kwargs, z=z))


def test_guard_refuses_one_weight_vector_for_every_head() -> None:
    """TensorRT-LLM's multi-row kernel takes one [64] weight for all heads; this one reads 1536."""
    kwargs = _small()
    with pytest.raises(AssertionError, match="weight must be a contiguous"):
        with validating(op):
            op(**dict(kwargs, weight=kwargs["weight"][:64].contiguous()))
