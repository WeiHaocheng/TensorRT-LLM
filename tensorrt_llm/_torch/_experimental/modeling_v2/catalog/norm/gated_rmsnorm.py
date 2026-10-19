# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDP mixer's output norm: gate by silu(z), then RMS-normalize each head's channels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.mamba.layernorm_gated import _layer_norm_fwd

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


class GatedRmsnorm(OpWrapper):
    """`x * silu(z)`, RMS-normalized per group of channels, times `weight`.

    For row `t` and group `g` (channels `[g * G, (g + 1) * G)`, `G = group_size`):
    `h = x[t] * silu(z[t])`, `y[t, c] = h[c] / sqrt(mean_g(h ** 2) + eps) * weight[c]`, computed
    in fp32 from the stored inputs and rounded once to the output dtype -- the checkpoint's
    `Nemotron4RMSNormGated`. The gate acts *before* the norm.

    One call of `_layer_norm_fwd`, the Triton path behind TensorRT-LLM's gated `RMSNorm`
    (`modules/mamba/layernorm_gated.py`). With `is_rms_norm=True` and `norm_before_gate=False` it
    runs `_layer_norm_fwd_1pass_kernel`, one program per (row, group). The op's defaults are the
    other order and LayerNorm, so a call site passes both flags.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The token counts a GDP layer normalizes per call: a decode step of one request, one
    # request's MTP verify (1 + max_draft_len 2), the largest CUDA-graph decode batch, and the two
    # prefill budgets. The kernel's grid is (rows, groups) with no tiling over rows, so the counts
    # differ in the grid they launch, not in the path through the kernel.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: one token", spec=dict(tokens=1, **_GDP_NORM)),
        Cell(why="one request's MTP verify: 1 + 2 draft tokens", spec=dict(tokens=3, **_GDP_NORM)),
        Cell(why="the largest CUDA-graph decode batch, 1024", spec=dict(tokens=1024, **_GDP_NORM)),
        Cell(why="prefill: one max_num_tokens 8192 chunk", spec=dict(tokens=8192, **_GDP_NORM)),
        Cell(why="prefill: the larger 16384-token budget", spec=dict(tokens=16384, **_GDP_NORM)),
    )

    note = """
    Certified: `is_rms_norm=True`, `norm_before_gate=False`, the silu gate, no bias, no
    `fp8_scale`, `weight_is_delta=False`, `group_size` 64 over 1536 channels, eps 1e-5, bf16 x, z
    and weight, a contiguous x and z the in_proj column slice, no `out`. Not certified, because no
    target here reaches it: the gate after the norm (the op's default, which also selects the
    multi-row kernel), LayerNorm, the sigmoid gate, a bias, fp8 output, a delta-form weight, other
    group sizes, an explicit `out`.

    A row is computed from its own inputs only, by one compiled kernel whatever the row count:
    its result does not depend on the batch it runs in. A packed copy of the gate gives the same
    bits as the in_proj slice. The test pins both bitwise.
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
        """Returns the normalized `[rows, channels]` tensor -- `out` when one is given. The op's
        other two results, the per-group mean and 1/std, are dropped."""
        y, _, _ = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            out=out,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
            fp8_scale=fp8_scale,
            gate_activation=gate_activation,
            weight_is_delta=weight_is_delta,
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

    def is_valid(
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
    ) -> None:
        # Only what the op takes and answers wrongly. The shapes of x, z, weight and out, their
        # unit last-dim strides and the group size dividing the channels it checks itself.
        #
        # The kernel forms the row offsets of x and of the output as `row * stride + group *
        # group_size` in int32 -- only z's is widened to int64 -- so once they pass 2**31 - 1 they
        # wrap, and the op reads (or writes) another row's memory and returns. A stride of 2**31
        # or more is passed to the kernel as int64 and does not wrap. Without `out` the op
        # allocates a dense output, one row every `width` elements.
        rows, width = x.shape[0], x.shape[-1]
        group = group_size if group_size is not None else width
        output_stride = width if out is None else out.stride(0)
        for name, stride in (("x", x.stride(0)), ("out", output_stride)):
            last = (rows - 1) * stride + width - group
            assert stride >= 2**31 or last < 2**31, (
                f"{name}'s row offsets reach {last} elements ({rows} rows, row stride {stride}): "
                "the kernel forms them in int32, so past 2**31 - 1 they wrap and it reads or "
                "writes another row's memory"
            )


gated_rmsnorm = GatedRmsnorm()
