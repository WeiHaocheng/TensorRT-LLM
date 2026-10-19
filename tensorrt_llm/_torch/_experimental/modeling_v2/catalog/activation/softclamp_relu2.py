# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The soft-clamped relu² of the checkpoint's dense MLPs: relu(c * tanh(x / c))², rounded as HF rounds."""

from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .._op import Arch, Cell, OpWrapper

#: Elements per program and warps per program. 8 elements per thread is one 16-byte bf16 vector;
#: measured on every cell against 2048, 4096 and 8192 elements at 4 and 8 warps, this is the
#: fastest from one token up and within 2% of the fastest at 8192 tokens.
_BLOCK = 1024
_NUM_WARPS = 4


def _fp32(value: float) -> float:
    """`value` rounded to the nearest fp32, as torch rounds a Python scalar for an fp32 op."""
    return struct.unpack("f", struct.pack("f", value))[0]


@triton.jit
def _softclamp_relu2_kernel(
    x_ptr, y_ptr, numel, CLAMP: tl.constexpr, INV_CLAMP: tl.constexpr, BLOCK: tl.constexpr
):
    """y = relu(round(CLAMP * tanh(x * INV_CLAMP)))², rounded to y's dtype, element-wise.

    The arithmetic is HF's op for op. `x.float() / c` with a Python-float c is, in torch, a
    multiply by fl32(1 / c), so this multiplies by INV_CLAMP rather than dividing (for a
    power-of-two clamp the two agree; for others IEEE division does not reproduce torch).
    libdevice's tanh agrees with the tanhf torch's CUDA kernel calls bit for bit, except that
    it flushes a subnormal argument to zero; an argument that small squares to zero on both
    sides. The clamp is rounded to the tensor dtype, relu² squares that rounded value in fp32
    (exact for a bf16 or fp16 value) and rounds once, as torch.square does on a bf16 tensor.
    `c < 0` rather than a max keeps a NaN a NaN, as torch.relu does.
    """
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    dt = y_ptr.dtype.element_ty
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = (CLAMP * libdevice.tanh(x * INV_CLAMP)).to(dt).to(tl.float32)
    r = tl.where(c < 0.0, 0.0, c)
    tl.store(y_ptr + offs, (r * r).to(dt), mask=mask)


class SoftclampRelu2(OpWrapper):
    """`relu(bf16(c * tanh(x / c)))²`, element-wise, one Triton launch, new tensor out.

    The activation of every dense MLP the checkpoint runs outside the routed experts: the four
    `mlp` layers' MLP (`Nemotron4MLP`, intermediate 12288) and every MoE block's shared expert
    (intermediate 4096), between the up and the down projection. HF's
    `Nemotron4ClampedActivation` with `mlp_hidden_act` relu2 and `activation_clamp_value` c.

    Fusion boundary: the activation only. No projection, no routing weight (the routed experts'
    activation is fused into `latent_moe_experts`), no gate. `x` is not mutated.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The two widths the activation runs at, each at the token counts that differ in kind:
    # one decode token (a launch of a handful of programs), the largest CUDA-graph decode batch,
    # an 8192-token prefill chunk (the bandwidth end), and a 16384-token prefill or mixed step,
    # the largest a rank's step reaches (201M elements at 12288). Then the same at a tp4 rank's
    # quarter of each width: 3072 of the dense MLP's 12288 and 1024 of the shared expert's 4096,
    # where one decode token is three programs and a single one. Every width is a multiple of the
    # block, so no cell has a partial program; the masked tail is exercised by the test
    # separately, since no width this checkpoint has can produce one.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dense MLP decode: one token, intermediate 12288",
            spec=dict(tokens=1, width=12288, activation_clamp=16.0),
        ),
        Cell(
            why="dense MLP at 1024 tokens, the largest CUDA-graph decode batch",
            spec=dict(tokens=1024, width=12288, activation_clamp=16.0),
        ),
        Cell(
            why="dense MLP prefill: an 8192-token chunk, 100M elements",
            spec=dict(tokens=8192, width=12288, activation_clamp=16.0),
        ),
        Cell(
            why="dense MLP prefill: a 16384-token step, 201M elements",
            spec=dict(tokens=16384, width=12288, activation_clamp=16.0),
        ),
        Cell(
            why="shared expert decode: one token, intermediate 4096",
            spec=dict(tokens=1, width=4096, activation_clamp=16.0),
        ),
        Cell(
            why="shared expert at 1024 tokens, the largest CUDA-graph decode batch",
            spec=dict(tokens=1024, width=4096, activation_clamp=16.0),
        ),
        Cell(
            why="shared expert prefill: an 8192-token chunk",
            spec=dict(tokens=8192, width=4096, activation_clamp=16.0),
        ),
        Cell(
            why="shared expert prefill: a 16384-token step",
            spec=dict(tokens=16384, width=4096, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 dense MLP decode: one token, a rank's 3072 of 12288",
            spec=dict(tokens=1, width=3072, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 dense MLP at 1024 tokens, a rank's 3072 of 12288",
            spec=dict(tokens=1024, width=3072, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 dense MLP prefill: an 8192-token chunk, width 3072",
            spec=dict(tokens=8192, width=3072, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 dense MLP prefill: a 16384-token step, width 3072",
            spec=dict(tokens=16384, width=3072, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 shared expert decode: one token, a rank's 1024 of 4096",
            spec=dict(tokens=1, width=1024, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 shared expert at 1024 tokens, a rank's 1024 of 4096",
            spec=dict(tokens=1024, width=1024, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 shared expert prefill: an 8192-token chunk, width 1024",
            spec=dict(tokens=8192, width=1024, activation_clamp=16.0),
        ),
        Cell(
            why="tp4 shared expert prefill: a 16384-token step, width 1024",
            spec=dict(tokens=16384, width=1024, activation_clamp=16.0),
        ),
    )

    note = """
    Bit-exact with HF, not merely close: the test drives every bf16 bit pattern at the
    checkpoint's clamp, NaN payloads included, besides the cells.

    The clamp is a compile-time constant of the kernel, so each distinct value compiles once;
    the checkpoint has one (16.0). Other clamps, and fp16 and fp32 inputs, are computed with
    HF's rounding at the input dtype and were measured bit-exact (every fp16 bit pattern, and
    sampled fp32, at clamps 16, 3, 10 and 0.1), but no shipped target uses them, so they are
    not cells.

    Compute-bound rather than bandwidth-bound: libdevice's tanhf is most of the instructions,
    and the 8192-token cells move their bytes at about half the HBM peak. Fusing this into the
    up projection's epilogue removes the round trip altogether; that is a perf task's.
    """

    def __call__(self, x: torch.Tensor, activation_clamp: float) -> torch.Tensor:
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        numel = x.numel()
        if numel:
            clamp = _fp32(activation_clamp)
            _softclamp_relu2_kernel[(triton.cdiv(numel, _BLOCK),)](
                x,
                y,
                numel,
                CLAMP=clamp,
                INV_CLAMP=_fp32(1.0 / clamp),
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
        return y

    def reference(self, x: torch.Tensor, activation_clamp: float) -> torch.Tensor:
        """HF's `Nemotron4ClampedActivation` then transformers' relu2, op for op.

        The two roundings are the checkpoint's own: the clamp is computed in fp32 and rounded to
        x's dtype, and relu² then runs on that dtype (torch.square rounds its fp32 product once).
        They are the specification here rather than an inaccuracy to avoid, which is why this
        does not keep fp32 to the end.
        """
        clamped = (activation_clamp * torch.tanh(x.float() / activation_clamp)).to(x.dtype)
        return torch.square(torch.relu(clamped))

    def is_valid(self, x: torch.Tensor, activation_clamp: float) -> None:
        # The kernel reads x and writes the output flat, from data_ptr(); a view with gaps or a
        # permuted layout is read as if it were dense and comes back as the wrong elements.
        assert x.is_contiguous(), "x must be contiguous; the kernel reads it flat from data_ptr()"
        # Other dtypes load, convert and round without complaint: an fp8 x comes back as a
        # plausible fp8 relu² where HF's relu raises, and an fp64 x is squared in fp32 where HF
        # squares in fp64 (half the outputs differ). Only the dtypes measured are admitted.
        assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), (
            f"x must be bf16, fp16 or fp32, got {x.dtype}; dequantize before the activation"
        )


softclamp_relu2 = SoftclampRelu2()
