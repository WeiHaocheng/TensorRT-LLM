# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An MTP verify step of the GDP mixer's short causal conv: T tokens per request in one launch,
each token's conv window stored for the engine's commit, the pool only read."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper

#: Channels per program and warps per program: one channel per thread. Measured at batch 1, 8,
#: 64, 256, 512 and 1024, T = 2 and 3, against 64 to 512 channels per program at one channel per
#: thread and 2 or 4 channels per thread: within 2% of the fastest everywhere, and 6-20% faster
#: than any layout with 2 or 4. With one channel per thread a warp's stride-3 accesses to the
#: pool and the window buffer cover 192 contiguous bytes.
_BLOCK = 128
_NUM_WARPS = 4


@triton.jit
def _causal_conv1d_verify_kernel(
    x,
    weight,
    conv_state,
    conv_state_indices,
    window,
    pad_slot_id,
    stride_x_b,
    stride_x_d,
    stride_x_t,
    stride_w_d,
    stride_w_k,
    stride_s_s,
    stride_s_d,
    stride_s_k,
    stride_i_b,
    stride_i_t,
    stride_i_d,
    stride_i_k,
    DIM: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (batch row, BLOCK channels); width 4, state length 3, SiLU, no bias.

    Token t's output is the decode kernel's arithmetic on the window (c0, c1, c2, x_t): an fp32
    fma chain over the taps in order, from +0, then `v * (1 / (1 + exp(-v)))` with a correctly
    rounded reciprocal, rounded once to x's dtype. The window then shifts by one and is stored
    as position t of the row's window buffer. A row whose slot is pad_slot_id returns before
    reading or writing anything.
    """
    b = tl.program_id(0)
    slot = tl.load(conv_state_indices + b)
    if slot == pad_slot_id:
        return
    slot = slot.to(tl.int64)
    row = b.to(tl.int64)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = d < DIM

    p_state = conv_state + slot * stride_s_s + d * stride_s_d
    c0 = tl.load(p_state, mask=m, other=0.0).to(tl.float32)
    c1 = tl.load(p_state + stride_s_k, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(p_state + 2 * stride_s_k, mask=m, other=0.0).to(tl.float32)
    p_w = weight + d * stride_w_d
    w0 = tl.load(p_w, mask=m, other=0.0).to(tl.float32)
    w1 = tl.load(p_w + stride_w_k, mask=m, other=0.0).to(tl.float32)
    w2 = tl.load(p_w + 2 * stride_w_k, mask=m, other=0.0).to(tl.float32)
    w3 = tl.load(p_w + 3 * stride_w_k, mask=m, other=0.0).to(tl.float32)

    p_x = x + row * stride_x_b + d * stride_x_d
    p_win = window + row * stride_i_b + d * stride_i_d
    for t in tl.static_range(T):
        # Each thread reads x_t of its channels before overwriting them, and no other thread
        # touches those channels: writing the output in place needs no barrier.
        xt = tl.load(p_x + t * stride_x_t, mask=m, other=0.0).to(tl.float32)
        acc = tl.fma(c0, w0, 0.0)
        acc = tl.fma(c1, w1, acc)
        acc = tl.fma(c2, w2, acc)
        acc = tl.fma(xt, w3, acc)
        y = acc * tl.math.div_rn(1.0, 1.0 + tl.exp(-acc))
        tl.store(p_x + t * stride_x_t, y.to(x.dtype.element_ty), mask=m)
        c0 = c1
        c1 = c2
        c2 = xt
        p_t = p_win + t * stride_i_t
        tl.store(p_t, c0.to(window.dtype.element_ty), mask=m)
        tl.store(p_t + stride_i_k, c1.to(window.dtype.element_ty), mask=m)
        tl.store(p_t + 2 * stride_i_k, c2.to(window.dtype.element_ty), mask=m)


# One geometry, as for `causal_conv1d_update`: every GDP layer of the checkpoint builds the same
# `Nemotron4GDPMixer`, whose conv runs over conv_dim = 3 * 1536 (V) + 4 * 3072 (K and Q) = 16896
# channels, width 4 (`gdp_conv_kernel`), no bias, SiLU. A verify step lays each generation
# request's 1 + D tokens out together, so x is the conv-column slice of the in_proj output's
# generation rows [batch * T, 18528] = [z 1536 | conv 16896 | beta 72 | decay 24], viewed
# [batch, 16896, T]. The conv state is one layer's view of the hybrid cache pool in either form
# `causal_conv1d_update` certifies ("v1" packed per slot after the recurrent state, "v2" one page
# per GDP layer). The window buffer is one layer's view of the manager's
# `intermediate_conv_states` [GDP layers, max_batch_size, 1 + max_draft_len, 16896, 3]: with
# max_draft_len 2 every row has three positions, of which a T = 2 step fills two; with the
# max_draft_len 3 tp1 also runs, four (`_GDP_DRAFT3`).
_GDP = dict(
    dim=16896,
    width=4,
    dtype=torch.bfloat16,
    proj_width=18528,
    proj_offset=1536,
    pool_ssm_bytes=24 * 64 * 128 * 4,
    window_positions=3,
)
# A tp4 rank's conv: its 6 of the 24 GDP heads, 4224 channels, the conv columns of its in_proj
# output [z 384 | conv 4224 | beta 18 | decay 6] = 4632 wide; a v1 slot holds the rank's
# [6, 64, 128] fp32 recurrent state before the conv state.
_GDP_TP4 = dict(_GDP, dim=4224, proj_width=4632, proj_offset=384, pool_ssm_bytes=6 * 64 * 128 * 4)
# tp1's conv under max_draft_len 3: four positions a row in the window buffer.
_GDP_DRAFT3 = dict(_GDP, window_positions=4)


def _cell(
    why: str,
    tokens: int,
    batch: int,
    pad_rows: int,
    pool: str,
    graph: bool = False,
    geometry: dict = _GDP,
) -> Cell:
    """`batch` rows of `tokens` tokens each, the last `pad_rows` rows CUDA-graph padding; `pool` is
    "v1" or "v2" (above); `graph` also captures the call in a CUDA graph and replays it;
    `geometry` is tp1's conv or a tp4 rank's."""
    return Cell(
        why=why,
        spec=dict(
            tokens=tokens, batch=batch, pad_rows=pad_rows, pool=pool, graph=graph, **geometry
        ),
    )


class CausalConv1dVerify(OpWrapper):
    """T steps of a depthwise causal conv + SiLU per row from the row's saved conv state, in one
    launch, storing the window after every step and leaving the state where it was.

    For row `b` with slot `s = conv_state_indices[b]`, channel `c` and the sequence
    `seq = [conv_state[s, c, 0], conv_state[s, c, 1], conv_state[s, c, 2], x[b, c, 0], ...,
    x[b, c, T - 1]]`: `x[b, c, t] = silu(sum_k weight[c, k] * seq[t + k])` and
    `intermediate_conv_window[b, t, c] = seq[t + 1 : t + 4]`, the conv state after token t,
    oldest first. `conv_state` is not written: the engine's commit (`update_mamba_states`) copies
    the accepted position's window into the slot. A row whose slot is `pad_slot_id` is skipped --
    its x keeps its values and its window row is not written.

    `x` is `[batch, dim, T]` and is overwritten with the output; `conv_state` is the pool
    `[slots, dim, 3]` and `intermediate_conv_window` the buffer `[rows >= batch, positions >= T,
    dim, 3]`, all addressed through their strides. Every launch shape is fixed by the tensors'
    metadata and the slots are read on the device, so the call is capturable in a CUDA graph.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The verify shapes of the targets' CUDA-graph batches: T = 1 + max_draft_len tokens per row
    # (3 for the checkpoint's trained depth, 2 for one draft token, 4 for the three drafts tp1
    # also runs), at every decode batch size the targets capture -- tp1 up to its max_batch_size,
    # 1024 for a DP rank's batch -- each with its padding rows at the end, plus 3, a size no graph
    # is captured at, which runs eagerly without padding. Each batch size meets both pool forms,
    # one per T.
    CELLS: tuple[Cell, ...] = (
        _cell("T=3, graph batch 1: one request, no padding row", 3, 1, 0, "v1"),
        _cell("T=3, graph batch 2: one request and a padding row", 3, 2, 1, "v2"),
        _cell("T=3, eager batch 3: no graph is captured at 3", 3, 3, 0, "v2"),
        _cell("T=3, graph batch 4: three requests and a padding row", 3, 4, 1, "v1"),
        _cell("T=3, graph batch 8 holding 5 requests: 3 padding rows", 3, 8, 3, "v2"),
        _cell("T=3, graph batch 16: 15 requests and a padding row", 3, 16, 1, "v1"),
        _cell("T=3, graph batch 32: 31 requests and a padding row", 3, 32, 1, "v2"),
        _cell("T=3, graph batch 64, also captured and replayed", 3, 64, 1, "v2", graph=True),
        _cell("T=3, graph batch 128: 127 requests and a padding row", 3, 128, 1, "v1"),
        _cell("T=3, graph batch 256 holding 129 requests: 127 padding rows", 3, 256, 127, "v2"),
        _cell("T=3, graph batch 512: around tp1's max_batch_size", 3, 512, 1, "v1"),
        _cell("T=3, graph batch 1024: a DP rank's largest batch", 3, 1024, 1, "v1"),
        _cell("T=2, graph batch 1: one draft token, one request", 2, 1, 0, "v2"),
        _cell("T=2, graph batch 2: one request and a padding row", 2, 2, 1, "v1"),
        _cell("T=2, eager batch 3: no graph is captured at 3", 2, 3, 0, "v1"),
        _cell("T=2, graph batch 4: three requests and a padding row", 2, 4, 1, "v2"),
        _cell("T=2, graph batch 8 holding 5 requests: 3 padding rows", 2, 8, 3, "v1"),
        _cell("T=2, graph batch 16: 15 requests and a padding row", 2, 16, 1, "v2"),
        _cell("T=2, graph batch 32: 31 requests and a padding row", 2, 32, 1, "v1"),
        _cell("T=2, graph batch 64: 63 requests and a padding row", 2, 64, 1, "v1"),
        _cell("T=2, graph batch 128: 127 requests and a padding row", 2, 128, 1, "v2"),
        _cell("T=2, graph batch 256 holding 129 requests: 127 padding rows", 2, 256, 127, "v1"),
        _cell("T=2, graph batch 512: around tp1's max_batch_size", 2, 512, 1, "v2"),
        _cell("T=2, graph batch 1024: a DP rank's largest batch", 2, 1024, 1, "v2"),
        _cell(
            "T=4, graph batch 1: one request, no padding row", 4, 1, 0, "v2", geometry=_GDP_DRAFT3
        ),
        _cell(
            "T=4, graph batch 2: one request and a padding row", 4, 2, 1, "v1", geometry=_GDP_DRAFT3
        ),
        _cell("T=4, eager batch 3: no graph is captured at 3", 4, 3, 0, "v1", geometry=_GDP_DRAFT3),
        _cell(
            "T=4, graph batch 4: three requests and a padding row",
            4,
            4,
            1,
            "v2",
            geometry=_GDP_DRAFT3,
        ),
        _cell(
            "T=4, graph batch 8 holding 5 requests: 3 padding rows",
            4,
            8,
            3,
            "v1",
            geometry=_GDP_DRAFT3,
        ),
        _cell(
            "T=4, graph batch 16: 15 requests and a padding row",
            4,
            16,
            1,
            "v2",
            geometry=_GDP_DRAFT3,
        ),
        _cell(
            "T=4, graph batch 32: 31 requests and a padding row",
            4,
            32,
            1,
            "v1",
            geometry=_GDP_DRAFT3,
        ),
        _cell(
            "T=4, graph batch 64, also captured and replayed",
            4,
            64,
            1,
            "v1",
            graph=True,
            geometry=_GDP_DRAFT3,
        ),
        _cell(
            "T=4, graph batch 128: 127 requests and a padding row",
            4,
            128,
            1,
            "v2",
            geometry=_GDP_DRAFT3,
        ),
        _cell(
            "T=4, graph batch 256 holding 129 requests: 127 padding rows",
            4,
            256,
            127,
            "v1",
            geometry=_GDP_DRAFT3,
        ),
        _cell("T=4, graph batch 448: tp1's max_batch_size", 4, 448, 1, "v2", geometry=_GDP_DRAFT3),
        _cell("tp4 rank, T=3, graph batch 1: one request", 3, 1, 0, "v2", geometry=_GDP_TP4),
        _cell("tp4 rank, T=3, graph batch 2 with a padding row", 3, 2, 1, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, T=3, eager batch 3", 3, 3, 0, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, T=3, graph batch 8 holding 5 requests", 3, 8, 3, "v2", geometry=_GDP_TP4),
        _cell(
            "tp4 rank, T=3, graph batch 64, also captured and replayed",
            3,
            64,
            1,
            "v1",
            graph=True,
            geometry=_GDP_TP4,
        ),
        _cell(
            "tp4 rank, T=3, graph batch 256 with a padding row", 3, 256, 1, "v2", geometry=_GDP_TP4
        ),
        _cell(
            "tp4 rank, T=3, graph batch 512 with a padding row", 3, 512, 1, "v1", geometry=_GDP_TP4
        ),
        _cell(
            "tp4 rank, T=3, graph batch 1024 with a padding row",
            3,
            1024,
            1,
            "v2",
            geometry=_GDP_TP4,
        ),
        _cell("tp4 rank, T=2, graph batch 1: one request", 2, 1, 0, "v1", geometry=_GDP_TP4),
        _cell("tp4 rank, T=2, eager batch 3", 2, 3, 0, "v2", geometry=_GDP_TP4),
        _cell(
            "tp4 rank, T=2, graph batch 64 with a padding row", 2, 64, 1, "v2", geometry=_GDP_TP4
        ),
        _cell(
            "tp4 rank, T=2, graph batch 1024 with a padding row",
            2,
            1024,
            1,
            "v1",
            geometry=_GDP_TP4,
        ),
    )

    note = """
    Fusion boundary: the GDP conv of a verify step and the per-token windows, nothing else.
    Width 4, state length 3, SiLU, no bias -- the checkpoint's conv and nothing more general.
    The commit is the engine's: after acceptance, `update_mamba_states` copies window
    `intermediate_conv_window[g, num_accepted[g] - 1]` of generation row g into the slot the
    manager's `state_indices` names. Row b of the window buffer is batch row b's, which is where
    that copy reads it (`intermediate_state_indices` is an arange). Positions >= T of a row are
    not written. Pass the manager's own view of the buffer, never a `.contiguous()` copy: the
    windows would land in the copy.

    Padding: pass `pad_slot_id` (`PAD_SLOT_ID`, -1) for a CUDA-graph batch's padding rows. Such
    a row reads nothing and writes nothing, so its window row keeps what it held; the commit
    copies it into whatever slot the manager's `state_indices` gives that row. The pool is only
    read, so live rows sharing a slot compute correctly here; what the commit then does with two
    windows for one slot is the engine's.

    Rounding: every token takes the arithmetic of `causal_conv1d_update`'s decode kernel
    (`causal_conv1d_update_kernel_sl1`): the four taps as an fma chain from +0 -- which keeps a
    zero output +0 where products of zeros and negative taps are -0 -- SiLU as
    `v * (1 / (1 + exp(-v)))` with a correctly rounded reciprocal, one rounding. The test holds
    output and windows to the reference within the decode entry's band. Like the decode entry,
    this rounds the conv output once; the checkpoint's `Nemotron4GDPMixer` rounds it to bf16
    before SiLU too (see that entry's note).

    Certified: T = 2 and 3, and T = 4 (tp1's three drafts) at tp1's width; x as the in_proj
    output's conv columns, `[batch, 16896, T]` with channel stride 1 and token stride 18528
    (tp1), or `[batch, 4224, T]` with token stride 4632 (a tp4 rank's 6 heads); both pool forms;
    the window buffer as one layer's view of the manager's dense `intermediate_conv_states`;
    padding rows at the end; one call captured in a CUDA graph and replayed on new inputs, slots
    and states. Not certified: T > 4, T = 4 at a tp4 rank's width, T = 1 (a decode step, which
    `causal_conv1d_update` covers), other widths, a bias, SiLU off, fp16 and fp32.

    Time: one launch, 6 * (1 + T) + 4 * T bytes of traffic per channel and row. GPU time per
    call under CUDA-graph replay, buffers rotated past L2, measured at T = 3 (T = 2): 3.3 (3.1)
    us at batch 1, 4.0 (3.2) at 8, 12.8 (9.7) at 64, 43 (31) at 256, 84 (60) at 512 and 164
    (117) at 1024, 3.8 TB/s at the large end. The per-position alternative that also leaves
    the pool alone -- gather the slots' states into window position 0, then per token a copy
    and a `causal_conv1d_update` call -- measured 2.2 to 5.4x slower. The stride-3 accesses to
    the pool and the window buffer hold the kernel to about 60% of a copy of the same bytes
    with ideal access. An eager call costs about 16 us of host time, Triton's launch, which
    sets its floor up to batch 128.
    """

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        intermediate_conv_window: torch.Tensor,
        pad_slot_id: int,
    ) -> torch.Tensor:
        """Writes the output into `x` and the live rows' first T windows into
        `intermediate_conv_window` in place, and returns `x`. Both lose their prior contents
        there, so a caller that wants them has to have kept its own copy beforehand."""
        batch, dim, tokens = x.shape
        if batch == 0:
            return x
        _causal_conv1d_verify_kernel[(batch, triton.cdiv(dim, _BLOCK))](
            x,
            weight,
            conv_state,
            conv_state_indices,
            intermediate_conv_window,
            pad_slot_id,
            *x.stride(),
            *weight.stride(),
            *conv_state.stride(),
            *intermediate_conv_window.stride(),
            DIM=dim,
            T=tokens,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        return x

    def reference(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        intermediate_conv_window: torch.Tensor,
        pad_slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x after the call, intermediate_conv_window after the call)`, in fp32 and rounded once.

        Returns rather than mutates, so a cell can drive both sides from the same inputs: the
        first is a new `[batch, dim, T]` tensor, holding a padding row's x as it is now; the
        second a copy of the whole window buffer with the live rows' first T positions replaced,
        which lets a test see that nothing else moved. The pool is not part of the result: the
        op only reads it.
        """
        batch, _, tokens = x.shape
        width = weight.shape[-1]
        slots = conv_state_indices.tolist()
        live = [b for b in range(batch) if slots[b] != pad_slot_id]

        out = x.to(torch.float32, copy=True)
        windows = intermediate_conv_window.clone()
        if live:
            rows = torch.tensor(live, device=x.device)
            owned = torch.tensor([slots[b] for b in live], device=x.device)
            seq = torch.cat([conv_state[owned].float(), x[rows].float()], dim=-1)
            taps = weight.float()
            acc = sum(taps[None, :, k : k + 1] * seq[..., k : k + tokens] for k in range(width))
            out[rows] = F.silu(acc)
            for t in range(tokens):
                windows[rows, t] = seq[..., t + 1 : t + width].to(windows.dtype)
        return out.to(x.dtype), windows

    def is_valid(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        intermediate_conv_window: torch.Tensor,
        pad_slot_id: int,
    ) -> None:
        # The kernel checks nothing: it reads every tensor through x's shape and its own
        # strides, so a disagreeing shape reads or writes the wrong elements without a word.
        assert x.dim() == 3, f"x must be [batch, dim, T]; got shape {tuple(x.shape)}"
        batch, dim, tokens = x.shape
        assert tuple(weight.shape) == (dim, 4), (
            f"weight must be [{dim}, 4]: the kernel reads four taps per channel; got "
            f"{tuple(weight.shape)}"
        )
        assert conv_state.dim() == 3 and tuple(conv_state.shape[1:]) == (dim, 3), (
            f"conv_state must be [slots, {dim}, 3], the width-4 state oldest first; got "
            f"{tuple(conv_state.shape)}"
        )
        window = intermediate_conv_window
        assert window.dim() == 4 and tuple(window.shape[2:]) == (dim, 3), (
            f"intermediate_conv_window must be [rows, positions, {dim}, 3]; got "
            f"{tuple(window.shape)}"
        )
        assert window.shape[0] >= batch, (
            f"intermediate_conv_window has {window.shape[0]} rows for a batch of {batch}: row b "
            "is written for batch row b, past the buffer's end"
        )
        assert window.shape[1] >= tokens, (
            f"intermediate_conv_window has {window.shape[1]} positions for T = {tokens}: the "
            "last windows would land in the next row's first positions"
        )
        assert tuple(conv_state_indices.shape) == (batch,), (
            f"conv_state_indices must hold one slot per batch row ({batch}); got shape "
            f"{tuple(conv_state_indices.shape)}"
        )
        live = [s for s in conv_state_indices.tolist() if s != pad_slot_id]
        outside = sorted({s for s in live if not 0 <= s < conv_state.shape[0]})
        assert not outside, (
            f"slots {outside} reach outside conv_state's {conv_state.shape[0]} slots: "
            "the op reads there unchecked"
        )


causal_conv1d_verify = CausalConv1dVerify()
