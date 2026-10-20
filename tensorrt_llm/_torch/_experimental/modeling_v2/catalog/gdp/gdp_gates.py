# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDP mixer's per-token gates: the step sizes beta = sigmoid(b) and the log decay g."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .._launch import Launch
from .._op import Arch, Cell, OpWrapper

# Every GDP mixer of the checkpoint (`Nemotron4GDPMixer`, 32 layers) has one geometry: 24 heads
# (`gdp_num_heads`) and 3 Householder steps per token (`gdp_num_householder`). Its in_proj output
# is [T, 18528] = [z 1536 | conv 16896 | b0..b2 72 | a 24], so b and a are column slices of it at
# offsets 18432 and 18504, row stride 18528.
_GDP = dict(heads=24, householder=3, proj_width=18528, b_offset=18432, a_offset=18504)
# A tp4 rank holds 6 of the heads, and its in_proj output keeps the segments in that order at a
# quarter of each width: [T, 4632] = [z 384 | conv 4224 | b0..b2 18 | a 6], so b and a sit at
# offsets 4608 and 4626. Each row of a then starts 4 bytes past a 16-byte boundary, where tp1's
# start on one, and Triton compiles another variant of the kernel for that pointer.
_GDP_TP4 = dict(heads=6, householder=3, proj_width=4632, b_offset=4608, a_offset=4626)

# torch's softplus at its defaults (beta 1, threshold 20): the input itself above the threshold.
_SOFTPLUS_THRESHOLD = 20.0


def _cell(why: str, tokens: int, geometry: dict = _GDP) -> Cell:
    return Cell(why=why, spec=dict(tokens=tokens, **geometry))


@triton.jit
def _gdp_gates_kernel(
    b_ptr,
    a_ptr,
    neg_a_ptr,
    dt_bias_ptr,
    beta_ptr,
    g_ptr,
    stride_b,
    stride_a,
    NB: tl.constexpr,
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """One token: its n_h * H step sizes and its H log decays.

    The arithmetic is torch's, op for op. `sigmoid` on a bf16 tensor is `1 / (1 + exp(-x))` in
    fp32, rounded to bf16; `softplus` is `x > threshold ? x : log1p(exp(x))` in fp32 (beta 1, so
    its `* beta` and `/ beta` are exact); the decay is the fp32 product of `-exp(A_log)` with it.
    libdevice's exp and log1p are the expf and log1pf torch's CUDA kernels call, and `div_rn` is
    IEEE division (Triton's `/` is an approximate reciprocal).
    """
    row = tl.program_id(0).to(tl.int64)
    cb = tl.arange(0, BLOCK_B)
    mb = cb < NB
    x = tl.load(b_ptr + row * stride_b + cb, mask=mb, other=0.0).to(tl.float32)
    beta = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-x))
    tl.store(beta_ptr + row * NB + cb, beta.to(beta_ptr.dtype.element_ty), mask=mb)

    ch = tl.arange(0, BLOCK_H)
    mh = ch < H
    a = tl.load(a_ptr + row * stride_a + ch, mask=mh, other=0.0).to(tl.float32)
    bias = tl.load(dt_bias_ptr + ch, mask=mh, other=0.0).to(tl.float32)
    neg_a = tl.load(neg_a_ptr + ch, mask=mh, other=0.0)
    u = a + bias
    softplus = tl.where(u > THRESHOLD, u, libdevice.log1p(libdevice.exp(u)))
    tl.store(g_ptr + row * H + ch, neg_a * softplus, mask=mh)


#: The kernel launched straight to its compiled form once compiled (`Launch`).
_GDP_GATES = Launch(_gdp_gates_kernel)


class GdpGates(OpWrapper):
    """`Nemotron4GDPMixer.forward` between the conv and the recurrence, for its gates: from the
    in_proj output's `b` and `a` columns, `beta = sigmoid(b)` reshaped to `[T, n_h, H]` and the
    log decay `g = -exp(A_log) * softplus(a + dt_bias)` `[T, H]`. One Triton launch, two new
    tensors out.

    `b` `[T, n_h * H]` and `a` `[T, H]` are bf16 views with unit-stride rows -- the in_proj output's
    column slices, so no copy runs. `neg_a` is `-A_log.float().exp()` `[H]` fp32, formed once at
    load (`weights.py`), and `dt_bias` `[H]` the checkpoint's bf16. `beta` comes back in b's dtype,
    as HF's `beta.sigmoid()` rounds it, and `g` in fp32: exactly what `gdp_chunk` and
    `gdp_recurrent` take (beta post-sigmoid, g the fp32 log decay). Fusion boundary: no
    Householder expansion and no l2 norm -- those are the recurrence entries' -- and no read of
    the conv columns.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Token counts that differ in kind: one decode token, one request's MTP verify (1 + 2 drafts),
    # a count no power of two divides, the largest CUDA-graph decode batch, and the two prefill
    # chunk budgets, at tp1's geometry and a tp4 rank's. The kernel is one program per token, so
    # the counts differ only in the grid.
    CELLS: tuple[Cell, ...] = (
        _cell("decode: one token", 1),
        _cell("one request's MTP verify: 3 tokens", 3),
        _cell("37 tokens: a count that is not a multiple of any tile", 37),
        _cell("1024 tokens: the largest CUDA-graph decode batch", 1024),
        _cell("8192 tokens: a prefill chunk at max_num_tokens 8192", 8192),
        _cell("16384 tokens: a prefill chunk at max_num_tokens 16384", 16384),
        _cell("tp4 rank, decode: one token", 1, _GDP_TP4),
        _cell("tp4 rank, one request's MTP verify: 3 tokens", 3, _GDP_TP4),
        _cell("tp4 rank, 37 tokens", 37, _GDP_TP4),
        _cell("tp4 rank, 1024 tokens: the largest graph batch", 1024, _GDP_TP4),
        _cell("tp4 rank, 8192 tokens: a max_num_tokens 8192 chunk", 8192, _GDP_TP4),
        _cell("tp4 rank, 16384 tokens: a 16384-token chunk", 16384, _GDP_TP4),
    )

    note = """
    Bit-exact with HF's torch ops, not merely close: besides the cells, the test drives every
    bf16 bit pattern through each gate (b for beta; a, at several heads' dt_bias and A_log, for g),
    NaN included, at both certified head counts. One exception: where exp(a + dt_bias) falls below
    2^-126 (a + dt_bias under about -87.3), libdevice flushes it to zero, so the kernel's decay is a
    zero of HF's sign while HF's is the subnormal softplus times exp(A_log) -- a subnormal, or
    where exp(A_log) > 1 a normal number below 2^-126 * exp(A_log) (2.2e-37 at the checkpoint's
    largest). The recurrences read g through exp(), which maps both to exactly 1.0.

    A token's results depend on that token alone, so a CUDA-graph padding row only computes
    something nobody reads. CUDA-graph capturable: one launch, no host
    sync, nothing allocated besides the two outputs. `T = 0` launches nothing.

    Certified: 24 heads (tp1) and 6 (a tp4 rank), 3 Householder steps, each as its target's in_proj
    column slices. Uncertified: another head count or Householder count, fp16 or fp32 b and a, a
    bf16 `neg_a`.
    """

    def __call__(
        self, b: torch.Tensor, a: torch.Tensor, neg_a: torch.Tensor, dt_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, heads = a.shape
        width = b.shape[1]
        beta = torch.empty(num_tokens, width // heads, heads, dtype=b.dtype, device=b.device)
        g = torch.empty(num_tokens, heads, dtype=torch.float32, device=a.device)
        if num_tokens:
            _GDP_GATES[(num_tokens,)](
                b,
                a,
                neg_a,
                dt_bias,
                beta,
                g,
                b.stride(0),
                a.stride(0),
                NB=width,
                H=heads,
                BLOCK_B=triton.next_power_of_2(width),
                BLOCK_H=triton.next_power_of_2(heads),
                THRESHOLD=_SOFTPLUS_THRESHOLD,
                num_warps=1,
            )
        return beta, g

    def reference(
        self, b: torch.Tensor, a: torch.Tensor, neg_a: torch.Tensor, dt_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HF's ops, op for op (`Nemotron4GDPMixer.forward`): `beta.sigmoid().reshape(...)` and
        `-self.A_log.float().exp() * F.softplus(decay.float() + self.dt_bias.float())`, with the
        first factor passed in. They are the specification, which is why this does not keep
        beta in fp32."""
        beta = b.sigmoid().reshape(b.shape[0], -1, a.shape[1])
        g = neg_a.float() * F.softplus(a.float() + dt_bias.float())
        return beta, g

    def is_valid(
        self, b: torch.Tensor, a: torch.Tensor, neg_a: torch.Tensor, dt_bias: torch.Tensor
    ) -> None:
        # The kernel reads a row's columns as consecutive elements from its row start; a
        # transposed or column-strided view is read as if it were dense.
        assert b.stride(-1) == 1 and a.stride(-1) == 1, (
            "b and a must have unit-stride rows (column slices of the in_proj output)"
        )
        # The kernel takes the head count from a and the group count from b's width: a b that is
        # not whole groups of H, or per-head vectors of another length, are read silently short.
        num_tokens, heads = a.shape
        assert b.shape[0] == num_tokens and b.shape[1] % heads == 0, (
            f"b must be [T, n_h * H] = [{num_tokens}, n_h * {heads}]; got {tuple(b.shape)}"
        )
        assert neg_a.shape == (heads,) and dt_bias.shape == (heads,), (
            f"neg_a and dt_bias must be [H] = [{heads}]; got {tuple(neg_a.shape)}, "
            f"{tuple(dt_bias.shape)}"
        )
        # Another dtype loads and converts without complaint, and beta comes back in it -- not the
        # bf16 rounding HF's sigmoid makes on its bf16 in_proj output.
        assert b.dtype == a.dtype == torch.bfloat16, (
            f"b and a must be bfloat16 (the in_proj output); got {b.dtype}, {a.dtype}"
        )
        # HF forms -exp(A_log) in fp32; a rounded factor moves every decay of its head.
        assert neg_a.dtype == torch.float32, f"neg_a must be float32; got {neg_a.dtype}"
        # A_log itself is accepted and gives a decay of the wrong sign wherever it is positive,
        # which grows the state instead of decaying it.
        assert bool((neg_a < 0).all()), "neg_a must be -exp(A_log) < 0: pass -exp(A_log), not A_log"


gdp_gates = GdpGates()
