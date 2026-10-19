# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gated_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list and `reference` the gate. This file turns a
cell into the tensors a target hands the op -- the recurrence output and the gate columns of an
in_proj-shaped buffer -- and pins what the kernel shares with `_layer_norm_fwd`, the TensorRT-LLM
op it replaced: the bits.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.gated_rmsnorm import (
    gated_rmsnorm as op,
)
from tensorrt_llm._torch.modules.mamba.layernorm_gated import _layer_norm_fwd

__extra_import_path__ = [".."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gated_rmsnorm requires a CUDA device"

#: Bound on the fraction of outputs that may differ from the reference at all; see `_compare`.
MAX_MISMATCH_FRACTION = 1e-3


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
    once to bf16. The kernel's sigmoid (`1 / (1 + exp(-z))`), its 64-term sum and its
    `1 / sqrt` are not torch's `F.silu`, `mean` and `rsqrt`, but they differ by a few fp32 ulps,
    ~1e-7 relative. So an output can only land on the neighbouring bf16 value, and only when its
    fp32 value sits on a rounding boundary. Measured over every cell at five seeds each: the 1- and
    3-token cells are bit-identical, and from 1024 tokens up at most 1.53e-5 of the outputs
    differ at all, each by exactly one ulp of its own value.

    Two gates:
      * every output within one bf16 ulp of its own value, counted on the bit patterns. Each
        structural rival misses by far more: the gate after the norm by 124 or more, one
        1536-wide group or groups of 128 by 858 or more, a sigmoid gate by 32661 or more, a
        `(1 + w)` weight by 312 or more. eps 2e-5 in place of 1e-5 misses by 66 or more,
        because `_build` puts groups on both sides of eps.
      * at most 1e-3 of the outputs not bit-identical: 65x the worst measured. Rounding the
        normalized value to bf16 before the weight multiply, as the checkpoint's plain
        `Nemotron4RMSNorm` does, keeps every output within one ulp but changes about a quarter
        of them. Rounding the gate product to bf16 before the norm changes 34 to 38%, by up to 3
        ulps. This gate is what tells them apart from the op.
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
    """Materialize one cell as `(proj, kwargs)`.

    x is the recurrence output, a contiguous `[T, 24 * 64]`; z is the first 1536 columns of an
    in_proj-shaped `[T, 18528]` buffer, whose other columns stand for the conv, beta and decay
    inputs. Every (token, head) group of x gets its own magnitude, log-uniform over [1e-3, 3]:
    the smallest groups' mean square sits below eps, where eps sets the output, and the largest
    are far above it. The gate pre-activations are N(0, 4), so silu's negative tail is covered.
    The weight follows the checkpoint's: its 32 `mixer.norm.weight` tensors have mean 1.06,
    standard deviation 0.26, and every value in [0.044, 3.03].
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
        x=x,
        weight=weight.to(dtype),
        bias=None,
        eps=spec["eps"],
        z=proj[:, :width],
        group_size=group,
        norm_before_gate=False,
        is_rms_norm=True,
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


def test_rows_do_not_depend_on_the_batch() -> None:
    """A row comes out the same bits whatever batch it runs in.

    CUDA-graph padding changes a decode step's row count, and an MTP verify step normalizes
    three rows per request where a step without drafts normalizes one. The identity criteria
    compare runs that differ in exactly that way, so this pins it bitwise on slices of the
    largest cell. The one- and three-token slices take the one-row-per-warp launch and the full
    call the four-row one, so it also pins that the launch a row count picks moves no bit.
    """
    _, kwargs = _build(op.CELLS[-1].spec, seed=7)
    full = op(**kwargs)
    for begin, end in ((0, 1), (7, 10), (100, 1124), (5000, 13192), (16383, 16384)):
        part = op(**dict(kwargs, x=kwargs["x"][begin:end], z=kwargs["z"][begin:end]))
        assert torch.equal(part, full[begin:end]), (
            f"rows [{begin}:{end}] alone differ from the same rows of a 16384-row call"
        )
    assert torch.equal(op(**kwargs), full), "two identical calls differ"


def test_a_packed_gate_gives_the_same_bits() -> None:
    """The gate's row stride is only an address: a packed copy of the in_proj slice gives the
    same bits, so a target may pass either."""
    _, kwargs = _build(op.CELLS[2].spec, seed=11)
    sliced = op(**kwargs)
    packed = op(**dict(kwargs, z=kwargs["z"].contiguous()))
    assert torch.equal(sliced, packed)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_bits_equal_layer_norm_fwd(cell) -> None:
    """Every output has the bits of `_layer_norm_fwd`, the op this entry called before its kernel.

    The kernel keeps that op's per-group arithmetic and reduction order -- two adjacent channels
    per lane, then a butterfly over one warp -- so whatever was measured with it (goldens, batch
    invariance, CUDA-graph and MTP identity) holds unchanged. A different order shows up here at
    once: the reference's own order already parts from the op on up to 1.53e-5 of the outputs.
    """
    _, kwargs = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    expected, _, _ = _layer_norm_fwd(**kwargs)
    assert torch.equal(op(**kwargs), expected)


def test_refuses_every_other_form() -> None:
    """The kernel computes one form; a call asking for another gets a ValueError, not that form."""
    _, kwargs = _build(op.CELLS[0].spec, seed=13)
    others = dict(
        z=None,
        bias=torch.zeros_like(kwargs["weight"]),
        out=torch.empty_like(kwargs["x"]),
        norm_before_gate=True,
        is_rms_norm=False,
        fp8_scale=torch.ones((), dtype=torch.float32, device="cuda"),
        gate_activation="sigmoid",
        weight_is_delta=True,
    )
    for name, value in others.items():
        with pytest.raises(ValueError, match="computes one form"):
            op(**dict(kwargs, **{name: value}))
    with pytest.raises(ValueError, match="power of two"):
        op(**dict(kwargs, group_size=48))


def test_row_offsets_past_int32_are_exact() -> None:
    """A row 2**31 elements or more into x is read where it is.

    Rows strided by 3 * 2**29 elements put row 2's offset at 3 * 2**30. An int32 offset wraps that
    to -2**30 -- which `_layer_norm_fwd`'s kernel does, normalizing the memory 2**30 elements
    before x's start in row 2's place. Starting x 2**30 elements into a buffer of 2**32 elements
    and one row puts a sentinel row exactly there, so a wrapped read would return the sentinel's
    norm. The kernel forms offsets in int64, and row 2 is the reference's.
    """
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 12 * 2**30:
        pytest.skip("needs about 9 GB of free device memory for a 2**32-element bf16 buffer")
    width, rows, stride = 1536, 3, 3 * 2**29
    base = 2**32 - 2 * stride
    buf = torch.empty(base + 2 * stride + width, dtype=torch.bfloat16, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(5)
    x = buf.as_strided((rows, width), (stride, 1), storage_offset=base)
    x.copy_(torch.randn(rows, width, generator=g, device="cuda"))
    sentinel = buf[:width]
    sentinel.copy_(100.0 * torch.randn(width, generator=g, device="cuda"))
    kwargs = dict(
        x=x,
        weight=torch.ones(width, dtype=torch.bfloat16, device="cuda"),
        bias=None,
        eps=1e-5,
        z=torch.randn(rows, width, generator=g, device="cuda").to(torch.bfloat16),
        group_size=64,
        norm_before_gate=False,
        is_rms_norm=True,
    )
    expected = op.reference(**kwargs)
    sentinel_norm = op.reference(**dict(kwargs, x=sentinel[None], z=kwargs["z"][2:]))[0]
    assert _ulp_distance(sentinel_norm, expected[2]).max() > 1, (
        "the sentinel's norm cannot be told from row 2's"
    )
    assert _ulp_distance(op(**kwargs), expected).max() <= 1, "a row past int32 offsets is misread"
    del buf, x, sentinel
    torch.cuda.empty_cache()
