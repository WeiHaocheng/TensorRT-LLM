# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One decode step of the GDP mixer's short causal conv, updating each request's conv state by slot."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

# One geometry, as for `causal_conv1d_varlen`: every GDP layer of the checkpoint builds the same
# `Nemotron4GDPMixer`, whose conv runs over conv_dim = 3 * 1536 (V) + 4 * 3072 (K and Q) = 16896
# channels, width 4 (`gdp_conv_kernel`), no bias, SiLU. x is the conv-column slice of the
# target's in_proj output [batch, 18528] = [z 1536 | conv 16896 | beta 72 | decay 24], one row
# per request, so no copy runs. Each layer's conv state sits in the hybrid cache pool in one of
# two forms, both strided per slot with dense inner dims: `CppMambaHybridCacheManager._setup_states`
# packs each slot's GDP recurrent state (24 x 64 x 128 fp32) and conv state into one block ("v1"),
# and `MambaHybridCacheManagerV2._get_state_buffer` exposes this layer's page of slots that
# coalesce every GDP layer's conv state ("v2"). The cells build both.
_GDP = dict(
    dim=16896,
    width=4,
    dtype=torch.bfloat16,
    proj_width=18528,
    proj_offset=1536,
    pool_ssm_bytes=24 * 64 * 128 * 4,
    silu_activation=True,
)
# A tp4 rank convolves the channels of its 6 GDP heads, a quarter of every segment: 4224 channels,
# the conv columns of its in_proj output [batch, 4632] = [z 384 | conv 4224 | b 18 | a 6], and
# pool slots whose recurrent state is 6 x 64 x 128 fp32. Every rank keeps all 32 GDP layers.
_GDP_TP4 = dict(_GDP, dim=4224, proj_width=4632, proj_offset=384, pool_ssm_bytes=6 * 64 * 128 * 4)


def _cell(
    why: str, batch: int, pad_rows: int, pool: str, graph: bool = False, geometry: dict = _GDP
) -> Cell:
    """`batch` rows of one token each, the last `pad_rows` of them CUDA-graph padding; `pool` is
    "v1" or "v2" (above); `graph` also captures the call in a CUDA graph and replays it."""
    return Cell(
        why=why, spec=dict(batch=batch, pad_rows=pad_rows, pool=pool, graph=graph, **geometry)
    )


class CausalConv1dUpdate(OpWrapper):
    """One step of a depthwise causal conv + SiLU per row, from the row's saved conv state.

    For row `b` with slot `s = conv_state_indices[b]`, channel `c`, width `W` and the window
    `win = [conv_state[s, c, -(W - 1):], x[b, c, :]]`:
    `x[b, c, t] = silu(sum_k weight[c, k] * win[t + k] (+ bias_[c]))`. Afterwards
    `conv_state[s, c]` holds the last `state_len` inputs of `[conv_state[s, c], x[b, c]]`, oldest
    first: with `state_len = W - 1` and one token, the old state shifted left by one with the new
    input appended. A row whose slot is `pad_slot_id` is skipped -- its x keeps its values and no
    slot is touched.

    `x` is `[batch, dim, T]` and is overwritten with the output; `conv_state` is the pool
    `[slots, dim, state_len]`, addressed through its strides. Every launch shape is fixed by the
    tensors' metadata and the slots are read on the device, so the call is capturable in a CUDA
    graph and replays on whatever x, the slot list and the pool hold at replay time.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The CUDA-graph decode batch sizes the targets capture -- tp1 up to its max_batch_size, 1024
    # for a DP rank's batch -- each with its padding rows at the end, plus 3, a size no graph is
    # captured at, which runs eagerly without padding. A partly filled graph batch (8 holding 5,
    # 256 holding 129) puts many padding rows next to live ones. "v2" is the default manager's
    # pool; the cells alternate the two forms over the sizes. The same twelve batches run at a
    # tp4 rank's width: tensor parallelism splits the channels, not the requests.
    CELLS: tuple[Cell, ...] = (
        _cell("graph batch 1: one request, no padding row", 1, 0, "v1"),
        _cell("graph batch 2: one request and one padding row", 2, 1, "v2"),
        _cell("eager batch 3: a size no CUDA graph is captured at", 3, 0, "v2"),
        _cell("graph batch 4: three requests and a padding row", 4, 1, "v1"),
        _cell("graph batch 8 holding 5 requests: 3 padding rows", 8, 3, "v2"),
        _cell("graph batch 16: 15 requests and a padding row", 16, 1, "v1"),
        _cell("graph batch 32: 31 requests and a padding row", 32, 1, "v2"),
        _cell("graph batch 64, also captured and replayed", 64, 1, "v2", graph=True),
        _cell("graph batch 128: 127 requests and a padding row", 128, 1, "v1"),
        _cell("graph batch 256 holding 129 requests: 127 padding rows", 256, 127, "v2"),
        _cell("graph batch 512: around tp1's max_batch_size", 512, 1, "v1"),
        _cell("graph batch 1024: a DP rank's largest batch", 1024, 1, "v1"),
        _cell("tp4 rank, graph batch 1: no padding row", 1, 0, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 2: one padding row", 2, 1, "v2", geometry=_GDP_TP4),
        _cell("tp4 rank, eager batch 3", 3, 0, "v2", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 4: a padding row", 4, 1, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 8 holding 5 requests", 8, 3, "v2", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 16: a padding row", 16, 1, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 32: a padding row", 32, 1, "v2", geometry=_GDP_TP4),
        _cell(
            "tp4 rank, graph batch 64, captured and replayed",
            64,
            1,
            "v2",
            graph=True,
            geometry=_GDP_TP4,
        ),
        _cell("tp4 rank, graph batch 128: a padding row", 128, 1, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 256 holding 129 requests", 256, 127, "v2", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 512: a padding row", 512, 1, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, graph batch 1024: a padding row", 1024, 1, "v1", geometry=_GDP_TP4),
    )

    note = """
    Conv state: one layer's `conv_state` is a `[slots, dim, W - 1]` view in x's dtype, oldest
    input first -- what `get_conv_states(layer_idx)` of the Mamba cache managers returns and what
    `causal_conv1d_varlen` leaves in a slot after a prefill chunk. The op addresses it through its
    strides alone, so both managers' per-slot-strided views are fine. Pass the pool's own view,
    never a `.contiguous()` copy: the update would land in the copy. The op has no
    `has_initial_state`: it always reads the slot, so a request that reaches a decode row without
    a prefill having written its slot -- a fresh one-token prompt that
    `_make_single_token_context_graph_batch` promotes into a decode row on a recycled slot --
    needs that slot zeroed first.

    Padding: the cache managers pad a CUDA-graph batch with rows that carry the dummy request's
    real slot. Pass `pad_slot_id` (`PAD_SLOT_ID`, -1) for them instead: such a row returns before
    reading or writing anything, so its x keeps its values and no slot moves. `is_valid` refuses
    live rows that share a slot.

    Certified: the decode kernel `causal_conv1d_update_kernel_sl1`, which the op picks when T is
    1, `cache_seqlens_` is None, SiLU is on and `state_len == W - 1`; x as the in_proj output's
    conv columns, `[batch, 16896, 1]` with row stride 18528, or a tp4 rank's `[batch, 4224, 1]`
    with row stride 4632; no bias; `conv_state_indices` with
    the padding rows at the end; both pool forms; one call captured in a CUDA graph and replayed
    on new inputs, slots and states. Not certified, because no target here reaches it yet: T > 1
    (an MTP verify step) runs the general kernel, whose states match T one-token steps bit for
    bit but about 6e-6 of whose outputs were measured one ulp away from theirs -- a verify step
    through this op does not reproduce decode bitwise. Nor are `state_len > W - 1`, a bias, SiLU
    off, widths 2 and 3, or fp16 and fp32.

    Rounding: the decode kernel sums the four taps as an fma chain from the bias, applies SiLU as
    `v * __frcp_rn(1 + __expf(-v))` in fp32 and rounds once -- the arithmetic of
    `causal_conv1d_fwd`'s kernels at the default `TRTLLM_CAUSAL_CONV1D_SILU_MODE`, a variable
    that reaches the prefill kernels only. A token's output and state therefore do not depend on
    whether it ran as a one-token context or as a decode row; the test pins that bitwise.

    The checkpoint's `Nemotron4GDPMixer` rounds the conv output to bf16 before its SiLU; this op
    rounds once. Measured over a dense grid of pre-activations on [-16, 16], HF's extra rounding
    moves the result by at most one bf16 ulp of the output on [-3.2, 2) and [8, 16], two on
    [-4, -3.2) and [2, 8), four on [-8, -4) and eight below -8, and it changes about a quarter of
    a decode step's outputs. A check against an HF-faithful reference has to allow for that.
    """

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias_: torch.Tensor | None,
        silu_activation: bool,
        cache_seqlens_: torch.Tensor | None,
        conv_state_indices: torch.Tensor | None,
        pad_slot_id: int,
    ) -> torch.Tensor:
        """Writes the output into `x` and the live rows' slots of `conv_state` in place, and
        returns `x`. Both lose their prior contents, so a caller that wants them has to have kept
        its own copy beforehand."""
        torch.ops.trtllm.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias_,
            silu_activation,
            cache_seqlens_,
            conv_state_indices,
            pad_slot_id,
        )
        return x

    def reference(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias_: torch.Tensor | None,
        silu_activation: bool,
        cache_seqlens_: torch.Tensor | None,
        conv_state_indices: torch.Tensor | None,
        pad_slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x after the call, conv_state after the call)`, in fp32 and rounded once.

        Returns rather than mutates, so a cell can drive both sides from the same inputs: the
        first is a new `[batch, dim, T]` tensor, holding a padding row's x as it is now; the
        second a copy of the whole pool with the live rows' slots replaced, which lets a test see
        that nothing outside them moved. Covers the linear state only: `cache_seqlens_` has to be
        None.
        """
        assert cache_seqlens_ is None, "the reference covers the linear, oldest-first state only"
        batch, _, tokens = x.shape
        width = weight.shape[-1]
        state_len = conv_state.shape[-1]
        slots = list(range(batch)) if conv_state_indices is None else conv_state_indices.tolist()
        live = [b for b in range(batch) if slots[b] != pad_slot_id]

        out = x.to(torch.float32, copy=True)
        states = conv_state.clone()
        if live:
            rows = torch.tensor(live, device=x.device)
            owned = torch.tensor([slots[b] for b in live], device=x.device)
            seq = torch.cat([conv_state[owned].float(), x[rows].float()], dim=-1)
            window = seq[..., state_len - (width - 1) :]
            taps = weight.float()
            acc = sum(taps[None, :, k : k + 1] * window[..., k : k + tokens] for k in range(width))
            if bias_ is not None:
                acc = acc + bias_.float()[None, :, None]
            out[rows] = F.silu(acc) if silu_activation else acc
            states[owned] = seq[..., -state_len:].to(states.dtype)
        return out.to(x.dtype), states

    def is_valid(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias_: torch.Tensor | None,
        silu_activation: bool,
        cache_seqlens_: torch.Tensor | None,
        conv_state_indices: torch.Tensor | None,
        pad_slot_id: int,
    ) -> None:
        # Only what the op takes and answers wrongly. The dtypes of x, weight, conv_state and
        # bias_, the width, the shapes of x (3-D), weight, bias_, conv_state and
        # conv_state_indices, and conv_state_indices' dtype and contiguity it checks itself.
        assert cache_seqlens_ is None, (
            "cache_seqlens_ makes the op treat conv_state as a ring buffer read from "
            "cache_seqlens % state_len, but the pool holds each slot's inputs oldest first: the "
            "step would read a rotated window and write the new input out of order"
        )
        if conv_state_indices is None:
            return
        live = [s for s in conv_state_indices.tolist() if s != pad_slot_id]
        outside = sorted({s for s in live if not 0 <= s < conv_state.shape[0]})
        assert not outside, (
            f"slots {outside} reach outside conv_state's {conv_state.shape[0]} slots: "
            "the op reads and writes there unchecked"
        )
        shared = sorted(s for s, n in Counter(live).items() if n > 1)
        assert not shared, (
            f"live rows share slots {shared}: rows sharing a slot race to read and write it "
            "(a CUDA-graph padding row must carry pad_slot_id, not the dummy request's slot)"
        )


causal_conv1d_update = CausalConv1dUpdate()
