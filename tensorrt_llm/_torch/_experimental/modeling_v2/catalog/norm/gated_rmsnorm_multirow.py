# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDP mixer's output norm on TensorRT-LLM's multi-row gated RMSNorm layout: gate by silu(z),
then RMS-normalize each head's channels, several (token, head) rows per program."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper

# Every GDP layer of the checkpoint (12 `linear_attention`, 20 `shortcut_moe_gdp`) builds the same
# `Nemotron4RMSNormGated(value_dim, group_size=value_dim // gdp_num_groups, eps=gdp_norm_eps)`:
# 1536 channels in 24 groups of 64 -- one group per head -- and eps 1e-5. x is the recurrence
# output, a contiguous [T, 1536]; z is the gate, the first 1536 columns of the target's in_proj
# output [T, 18528] = [z 1536 | conv 16896 | beta 72 | decay 24], passed as that column slice
# (row stride 18528) so no copy runs.
_GDP_NORM = dict(num_heads=24, group_size=64, eps=1e-5, dtype=torch.bfloat16, proj_width=18528)
# A tp4 rank normalizes its 6 heads: x a contiguous [T, 384] and z the first 384 columns of its
# in_proj output [T, 4632] = [z 384 | conv 4224 | b 18 | a 6].
_GDP_NORM_TP4 = dict(_GDP_NORM, num_heads=6, proj_width=4632)

#: (token, head) rows per program below `_MANY_ROWS` rows, and from it up. Four warps either way.
#: Measured on GB300 at 1 to 16384 tokens and 24 and 6 heads against 4 to 64 rows at 4 and 8
#: warps (survey-small-ops): below 8192 rows 8 is the fastest or within 0.1 us of it; from 8192
#: rows 16 is, and 32 or 64 gain nothing more.
_FEW_ROWS, _MANY_ROWS_PER_PROGRAM, _MANY_ROWS = 8, 16, 8192
_WARPS = 4


@triton.jit
def _gated_rmsnorm_multirow_kernel(
    X,
    Z,
    W,
    Y,
    stride_z_tok,
    M,
    eps,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
):
    """ROWS (token, head) rows of N channels per program; row r is token r // HEADS, head r % HEADS.

    TensorRT-LLM's `_rms_norm_gated_fwd_multirow_kernel` (`modules/mamba/layernorm_gated.py`) with
    the two differences this checkpoint's norm needs: the gate multiplies x *before* the mean
    square (`Nemotron4RMSNormGated`; that kernel normalizes first), and the weight is one value
    per (head, channel), 1536 of them, where that kernel reads one [N] vector for every head.
    x and y are dense [M, N]; z is read token-major through its token stride. Offsets are int64.
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M
    rows = rows.to(tl.int64)
    cols = tl.arange(0, N)
    mask = row_mask[:, None]
    tok = rows // HEADS
    head = rows % HEADS
    x = tl.load(X + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    z = tl.load(
        Z + tok[:, None] * stride_z_tok + head[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    x = x * (z * tl.sigmoid(z))
    var = tl.sum(x * x, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + head[:, None] * N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w
    tl.store(Y + rows[:, None] * N + cols[None, :], y.to(Y.dtype.element_ty), mask=mask)


class GatedRmsnormMultirow(OpWrapper):
    """`x * silu(z)`, RMS-normalized per group of `group_size` channels, times `weight`.

    For row `t` and group `g` (channels `[g * G, (g + 1) * G)`): `h = x[t] * silu(z[t])`,
    `y[t, c] = h[c] / sqrt(mean_g(h ** 2) + eps) * weight[c]`, in fp32 from the stored inputs and
    rounded once to x's dtype -- the checkpoint's `Nemotron4RMSNormGated`, what `gated_rmsnorm`
    computes. One Triton launch, a new `[T, H * G]` tensor out.

    `x` is the recurrence output, a contiguous `[T, H * G]`; `z` has unit channel stride and any
    token stride (the in_proj output's column slice); `weight` is `[H * G]`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The token counts a GDP layer normalizes per call, at tp1's width and a tp4 rank's: a decode
    # step of one request, one request's MTP verify (1 + max_draft_len 2), a CUDA-graph batch, the
    # largest graph batch and its verify, the 8192-token prefill chunk and a mixed step past it,
    # and the 16384-token step. Up to 64 tokens run the few-rows launch at both widths, 1024 and
    # up the many-rows one at 24 heads (24,576 rows and more), 3072 and up at 6 (18,432).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"{name}: {why}", spec=dict(tokens=tokens, **geo))
        for name, geo in (("tp1 24 heads", _GDP_NORM), ("tp4 rank 6 heads", _GDP_NORM_TP4))
        for tokens, why in (
            (1, "decode, one token"),
            (3, "one request's MTP verify"),
            (64, "a 64-request graph batch"),
            (1024, "the largest CUDA-graph decode batch"),
            (3072, "MTP verify of 1024 requests"),
            (8192, "an 8192-token prefill chunk"),
            (8960, "a mixed step: 8192 + 256 verify rows"),
            (16384, "the 16384-token step"),
        )
    )

    note = """
    Fusion boundary: the gate, the per-group norm and the weight -- `gated_rmsnorm`'s computation,
    nothing more. No bias, no fp8 output, no sigmoid gate, no `(1 + w)` weight, no gate after the
    norm (TensorRT-LLM's multi-row kernel's order).

    Certified: bf16 x, z and weight; group 64 over 1536 channels (tp1, a dep4 rank) or 384 (a tp4
    rank); eps 1e-5; z the in_proj column slice at its target's row stride. Group sizes are a
    compile-time power of two; another one compiles but is uncertified.

    Launch: `_FEW_ROWS` (8) rows per program below `_MANY_ROWS` (8192) rows, `_MANY_ROWS_PER_PROGRAM`
    (16) from it up. A row's arithmetic does not depend on the batch, but its sum of squares is a
    tree over the program's [ROWS, 64] tile whose shape follows the launch, so the two sides can
    round a row differently: within one bf16 ulp of each other, each side within one of the
    reference. Calls of the same shape are bit-identical (no atomics).

    Time on GB300 (survey-small-ops, CUDA-graph replay, us per call, 24 heads): 1.08 at 1 token,
    1.45 at 64, 2.62 at 1024 and 5.15 at 3072, against `gated_rmsnorm`'s 1.50, 1.97, 13.54 and
    38.24 and perf-gated-rmsnorm's Gluon kernel's 1.29, 1.51, 3.59 and 7.80. Eager GPU time at
    8192 tokens 13.1 against 101.2 and 20.0, 1.2x the bytes' bound (x, z and y at the HBM copy
    rate). A tp4 rank's 6 heads: 1.29, 1.39, 1.70 and 2.50 in a graph, 5.4 eager at 8192.
    Eager host issue ~27 us against `_layer_norm_fwd`'s ~48.
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int,
    ) -> torch.Tensor:
        tokens, width = x.shape
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        rows = tokens * (width // group_size)
        if rows:
            per_program = _MANY_ROWS_PER_PROGRAM if rows >= _MANY_ROWS else _FEW_ROWS
            _gated_rmsnorm_multirow_kernel[(triton.cdiv(rows, per_program),)](
                x,
                z,
                weight,
                y,
                z.stride(0),
                rows,
                eps,
                N=group_size,
                ROWS=per_program,
                HEADS=width // group_size,
                num_warps=_WARPS,
            )
        return y

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int,
    ) -> torch.Tensor:
        """`Nemotron4RMSNormGated.forward`, transcribed: fp32 from the stored inputs, one rounding."""
        hidden = x.float() * F.silu(z.float())
        grouped = hidden.reshape(*hidden.shape[:-1], -1, group_size)
        grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
        return (grouped.reshape_as(hidden) * weight.float()).to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int,
    ) -> None:
        # The kernel reads x and writes y as dense [rows, group_size], z through its token
        # stride alone, and the weight as [heads, group_size]; it checks no shape or stride.
        tokens, width = x.shape
        assert x.is_contiguous(), (
            f"x must be a contiguous [T, H * G]: the kernel reads row r at r * G; got strides "
            f"{tuple(x.stride())}"
        )
        assert width % group_size == 0, (
            f"group_size {group_size} does not divide the {width} channels: the trailing channels "
            "would never be normalized"
        )
        assert tuple(z.shape) == (tokens, width) and z.stride(-1) == 1, (
            f"z must be [T, H * G] = [{tokens}, {width}] with unit channel stride; got shape "
            f"{tuple(z.shape)}, strides {tuple(z.stride())}"
        )
        assert weight.is_contiguous() and weight.numel() == width, (
            f"weight must be a contiguous [{width}]: the kernel reads head h's G values at h * G; "
            f"got shape {tuple(weight.shape)}"
        )


gated_rmsnorm_multirow = GatedRmsnormMultirow()
