# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDP mixer's output norm: gate by silu(z), then RMS-normalize each head's channels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .._op import Arch, Cell, OpWrapper

# Every GDP layer of the checkpoint (12 `linear_attention`, 20 `shortcut_moe_gdp`) builds the same
# `Nemotron4RMSNormGated(value_dim, group_size=value_dim // gdp_num_groups, eps=gdp_norm_eps)`, so
# there is one geometry: 1536 channels in 24 groups of 64 -- one group per head -- and eps 1e-5.
# x is the recurrence output, which the GDP entries return as a contiguous [T, 24, 64] (viewed
# [T, 1536]); z is the gate, the first 1536 columns of the target's in_proj output [T, 18528] =
# [z 1536 | conv 16896 | beta 72 | decay 24], passed as that column slice (row stride 18528) so no
# copy runs. The cells build exactly those two views.
_GDP_NORM = dict(
    num_heads=24,
    group_size=64,
    eps=1e-5,
    dtype=torch.bfloat16,
    proj_width=18528,
)

#: Warps per program.
_WARPS = 4
#: From this many (token, group) rows up a warp normalizes four at once, which fills the grid with
#: fewer, longer programs. Below it a warp takes one, which spreads a small batch over as many SMs
#: as it has rows; there a program's latency is the op's time. Either way a row gets the same bits.
_MANY_ROWS = 4096


@gluon.jit
def _gated_rmsnorm_kernel(
    X,
    Y,
    W,
    Z,
    stride_x,
    stride_y,
    stride_z,
    num_rows,
    group_cols,
    eps,
    GROUPS: gl.constexpr,
    GROUP: gl.constexpr,
    WARPS: gl.constexpr,
    ROWS_PER_WARP: gl.constexpr,
):
    """One row is one (token, group) pair: GROUP channels of x and z, normalized on their own.

    Each lane holds GROUP // 32 adjacent channels of a row and one warp holds the whole row, so a
    row's sum of squares runs in the order `_layer_norm_fwd_1pass_kernel` (one warp per row and
    group) runs it: the lane's own channels first, then a butterfly over the 32 lanes. With the
    same expressions in the same order, every output has that kernel's bits. The layout is stated
    rather than left to the compiler because the order it fixes is the result. A program holds
    WARPS * ROWS_PER_WARP rows, ROWS_PER_WARP to a warp, so each warp has that many rows of loads
    in flight; `group_cols` (= GROUP) is a runtime value so that the mean divides the way that
    kernel's does.
    """
    L: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, GROUP // 32],
        threads_per_warp=[1, 32],
        warps_per_cta=[WARPS, 1],
        order=[1, 0],
    )
    ROWS: gl.constexpr = WARPS * ROWS_PER_WARP
    cols = gl.arange(0, GROUP, layout=gl.SliceLayout(0, L))
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, L))
    live = (row < num_rows)[:, None]
    token = (row // GROUPS).to(gl.int64)[:, None]
    channel = (row % GROUPS)[:, None] * GROUP + cols[None, :]
    x = gl.load(X + token * stride_x + channel, mask=live, other=0.0).to(gl.float32)
    z = gl.load(Z + token * stride_z + channel, mask=live, other=0.0).to(gl.float32)
    gate = 1 / (1 + gl.exp(-z))
    gate *= z
    x *= gate
    var = gl.sum(x * x, axis=1) / group_cols
    rstd = 1 / gl.sqrt(var + eps)
    w = gl.load(W + channel).to(gl.float32)
    y = x * rstd[:, None] * w
    gl.store(Y + token * stride_y + channel, y.to(Y.dtype.element_ty), mask=live)


class GatedRmsnorm(OpWrapper):
    """`x * silu(z)`, RMS-normalized per group of channels, times `weight`.

    For row `t` and group `g` (channels `[g * G, (g + 1) * G)`, `G = group_size`):
    `h = x[t] * silu(z[t])`, `y[t, c] = h[c] / sqrt(mean_g(h ** 2) + eps) * weight[c]`, computed
    in fp32 from the stored inputs and rounded once to the output dtype -- the checkpoint's
    `Nemotron4RMSNormGated`. The gate acts *before* the norm.

    One launch of `_gated_rmsnorm_kernel`: four warps per program, each warp one (token, group)
    row, or four rows from `_MANY_ROWS` rows up. It computes exactly what TensorRT-LLM's gated
    `RMSNorm` (`_layer_norm_fwd` in `modules/mamba/layernorm_gated.py`) computes for
    `is_rms_norm=True, norm_before_gate=False`, bit for bit, without that path's one single-warp
    program per (row, group), which leaves it bound by its grid rather than by memory. The
    signature is `_layer_norm_fwd`'s, so a call site reads the same; every argument outside that
    one form is refused.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The token counts a GDP layer normalizes per call: a decode step of one request, one
    # request's MTP verify (1 + max_draft_len 2), the largest CUDA-graph decode batch, and the two
    # prefill budgets. A row's arithmetic is the same whatever the count. The first two take the
    # one-row-per-warp launch and the rest the four-row one, so the cells drive both.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: one token", spec=dict(tokens=1, **_GDP_NORM)),
        Cell(why="one request's MTP verify: 1 + 2 draft tokens", spec=dict(tokens=3, **_GDP_NORM)),
        Cell(why="the largest CUDA-graph decode batch, 1024", spec=dict(tokens=1024, **_GDP_NORM)),
        Cell(why="prefill: one max_num_tokens 8192 chunk", spec=dict(tokens=8192, **_GDP_NORM)),
        Cell(why="prefill: the larger 16384-token budget", spec=dict(tokens=16384, **_GDP_NORM)),
    )

    note = """
    Certified: `group_size` 64 over 1536 channels, eps 1e-5, bf16 x, z and weight, a contiguous x
    and z the in_proj column slice. Accepted but not certified, because no target here reaches
    them: other power-of-two group sizes from 32 to 256, fp16 or fp32 inputs, other row strides.
    Refused with a ValueError: the gate after the norm, LayerNorm, the sigmoid gate, a bias, fp8
    output, a delta-form weight, an explicit `out`, no `z`.

    A row is computed from its own inputs only, by the same arithmetic in the same order whatever
    the row count, which picks only how many rows a warp holds: its result does not depend on the
    batch it runs in. A packed copy of the gate gives the same bits as the in_proj slice, and the
    output equals `_layer_norm_fwd`'s bit for bit. The test pins all three. Row offsets are formed
    in int64, so a strided x or z of any size is read where it is.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        z: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        is_rms_norm: bool = False,
        fp8_scale: torch.Tensor | None = None,
        gate_activation: str = "silu",
        weight_is_delta: bool = False,
    ) -> torch.Tensor:
        """Returns the normalized `[rows, channels]` tensor, newly allocated in x's dtype."""
        if (
            z is None
            or bias is not None
            or out is not None
            or norm_before_gate
            or not is_rms_norm
            or fp8_scale is not None
            or gate_activation != "silu"
            or weight_is_delta
        ):
            raise ValueError(
                "gated_rmsnorm computes one form: a silu(z) gate before an RMS norm, no bias, an "
                "absolute weight, a new output (z given, bias=None, out=None, "
                "norm_before_gate=False, is_rms_norm=True, fp8_scale=None, gate_activation='silu', "
                "weight_is_delta=False)"
            )
        rows, width = x.shape
        group = width if group_size is None else group_size
        if group not in (32, 64, 128, 256) or width % group:
            raise ValueError(
                f"group_size must be a power of two from 32 to 256 that divides the {width} "
                f"channels; got {group}"
            )
        if z.shape != x.shape or weight.shape != (width,):
            raise ValueError(
                f"x {tuple(x.shape)}, z {tuple(z.shape)} and weight {tuple(weight.shape)} must be "
                f"[rows, channels], [rows, channels] and [channels]"
            )
        if x.stride(1) != 1 or z.stride(1) != 1 or weight.stride(0) != 1:
            raise ValueError("x, z and weight must each have unit stride along the channels")
        groups = width // group
        num_rows = rows * groups
        if num_rows >= 2**31:
            raise ValueError(f"{rows} rows of {groups} groups exceed the kernel's int32 row index")
        y = torch.empty((rows, width), dtype=x.dtype, device=x.device)
        if rows == 0:
            return y
        per_warp = 4 if num_rows >= _MANY_ROWS else 1
        with torch.cuda.device(x.device.index):
            _gated_rmsnorm_kernel[(triton.cdiv(num_rows, _WARPS * per_warp),)](
                x,
                y,
                weight,
                z,
                x.stride(0),
                y.stride(0),
                z.stride(0),
                num_rows,
                group,
                eps,
                GROUPS=groups,
                GROUP=group,
                WARPS=_WARPS,
                ROWS_PER_WARP=per_warp,
                num_warps=_WARPS,
            )
        return y

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        z: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        is_rms_norm: bool = False,
        fp8_scale: torch.Tensor | None = None,
        gate_activation: str = "silu",
        weight_is_delta: bool = False,
    ) -> torch.Tensor:
        """`Nemotron4RMSNormGated.forward`, transcribed: fp32 from the stored inputs, one rounding.

        Covers the certified form only -- the silu gate before an RMS norm, no bias, no fp8
        output, an absolute weight -- and asserts it. `out` sets only the output dtype: this
        returns a new tensor.
        """
        assert z is not None and not norm_before_gate and is_rms_norm, (
            "the reference covers a silu(z) gate applied before an RMS norm only"
        )
        assert (
            bias is None and fp8_scale is None and gate_activation == "silu" and not weight_is_delta
        ), "the reference covers no bias, no fp8 output and an absolute silu-gated weight only"
        hidden = x.float() * F.silu(z.float())
        grouped = hidden.reshape(*hidden.shape[:-1], -1, group_size or hidden.shape[-1])
        variance = grouped.square().mean(dim=-1, keepdim=True)
        grouped = grouped * torch.rsqrt(variance + eps)
        y = grouped.reshape_as(hidden) * weight.float()
        return y.to(x.dtype if out is None else out.dtype)


gated_rmsnorm = GatedRmsnorm()
