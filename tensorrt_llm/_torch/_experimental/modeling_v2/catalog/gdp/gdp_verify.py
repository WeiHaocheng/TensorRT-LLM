# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gated delta product over an MTP verify step: T tokens per request, the state kept after each."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

from .._op import Arch, Cell, OpWrapper

# The decode entry's tile, warp count and grid cap. Each token's update has to
# round exactly as one `gdp_recurrent` call does, and the order of its two
# reductions over K follows from the tile and the warps.
from .gdp_recurrent import _BLOCK_V, _MAX_GRID_ROWS, _NUM_WARPS


@triton.jit(do_not_specialize=["num_rows", "num_slots"])
def _gdp_verify_kernel(
    q,
    k,
    v,
    g,
    beta,
    out,
    state_pool,
    slots,
    intermediate,
    scale,
    num_rows,
    num_slots,
    stride_q_n,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_t,
    stride_k_j,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_t,
    stride_v_j,
    stride_v_h,
    stride_v_d,
    stride_g_n,
    stride_g_t,
    stride_g_h,
    stride_b_n,
    stride_b_t,
    stride_b_j,
    stride_b_h,
    stride_s_s,
    stride_s_h,
    stride_s_v,
    stride_s_k,
    stride_i_n,
    stride_i_t,
    stride_i_h,
    stride_i_v,
    stride_i_k,
    T: tl.constexpr,
    H: tl.constexpr,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """One program per (BV-column tile, request x head row), grid-strided over rows.

    The state tile is loaded once from the request's slot and carried through
    the T tokens in fp32 registers. Each token is `_gdp_recurrent_kernel`'s
    body, statement for statement; the state after it goes to
    `intermediate[request, token]`, and the slot itself is never written.
    """
    i_v = tl.program_id(0)
    i_row = tl.program_id(1)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]

    while i_row < num_rows:
        i_n = (i_row // H).to(tl.int64)
        i_h = i_row % H
        slot = tl.load(slots + i_n).to(tl.int64)
        # Element [k, v] of the HF state lives at pool[slot, h, v, k].
        p_state = (
            state_pool
            + slot * stride_s_s
            + i_h * stride_s_h
            + o_k[:, None] * stride_s_k
            + o_v[None, :] * stride_s_v
        )
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if slot >= 0:
            tl.device_assert(slot < num_slots, "slot out of range")
            b_h = tl.load(p_state, mask=mask_kv, other=0.0).to(tl.float32)

        for t in tl.static_range(T):
            b_g = tl.load(g + i_n * stride_g_n + t * stride_g_t + i_h * stride_g_h).to(tl.float32)
            b_h *= tl.exp(b_g)
            for j in tl.static_range(NH):
                b_k = tl.load(
                    k
                    + i_n * stride_k_n
                    + t * stride_k_t
                    + j * stride_k_j
                    + i_h * stride_k_h
                    + o_k * stride_k_d,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k) + 1e-6)
                b_v = tl.load(
                    v
                    + i_n * stride_v_n
                    + t * stride_v_t
                    + j * stride_v_j
                    + i_h * stride_v_h
                    + o_v * stride_v_d,
                    mask=mask_v,
                    other=0.0,
                ).to(tl.float32)
                b_beta = tl.load(
                    beta + i_n * stride_b_n + t * stride_b_t + j * stride_b_j + i_h * stride_b_h
                ).to(tl.float32)
                b_v = (b_v - tl.sum(b_h * b_k[:, None], 0)) * b_beta
                b_h += b_k[:, None] * b_v[None, :]

            b_q = tl.load(
                q + i_n * stride_q_n + t * stride_q_t + i_h * stride_q_h + o_k * stride_q_d,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(
                out + ((i_n * T + t) * H + i_h) * V + o_v,
                b_o.to(out.dtype.element_ty),
                mask=mask_v,
            )
            if slot >= 0:
                p_snapshot = (
                    intermediate
                    + i_n * stride_i_n
                    + t * stride_i_t
                    + i_h * stride_i_h
                    + o_k[:, None] * stride_i_k
                    + o_v[None, :] * stride_i_v
                )
                tl.store(p_snapshot, b_h.to(intermediate.dtype.element_ty), mask=mask_kv)
        i_row += tl.num_programs(1)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """HF `l2norm`: `x * rsqrt(sum(x * x) + 1e-6)` over the head dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


class GdpVerify(OpWrapper):
    """T tokens per request through `torch_recurrent_gated_delta_rule`'s
    recurrence, Householder-expanded the way `Nemotron4GDPMixer.forward`
    expands it, from the request's fp32 state in the pool. The state after
    every token is written to the request's row of `intermediate`. The pool is
    only read.

    Per (request, head), with `S` the HF `[K, V]` state read from the slot and
    `l2n(x) = x * rsqrt(sum(x*x) + 1e-6)`:

        for t in range(T):
            S *= exp(g[t])                           # once per token, before its first step
            for j in range(n_h):                     # the token's Householder steps
                kj = l2n(k[t, j]); u = beta[t, j] * (v[t, j] - S^T kj); S += kj u^T
            out[t] = S^T (l2n(q[t]) * scale)         # read after the token's last step
            intermediate[request, t] = S

    A request's token t output and state are, bit for bit, what the t-th of
    T consecutive `gdp_recurrent` calls returns and leaves in the slot -- the
    MTP-off decode of the same tokens -- so promoting snapshot `t` commits
    exactly the state that decode would have reached. Returns `out`
    `[B, T, H, V]` in `q`'s dtype.
    """

    ARCHS = frozenset({Arch.SM_103})

    # nemotron4-nano-preview's GDP mixer, the shape every one of its 32 GDP and
    # linear_attention layers runs: gdp_num_heads 24 (== gdp_num_groups, so no
    # head repeat), gdp_key_head_dim 128, gdp_head_dim 64, gdp_num_householder 3.
    # `scale` is what the target passes for HF's `query / sqrt(key_head_dim)`.
    # q, k and v are views of the in_proj output's conv columns, a token per row:
    # a token's row stride is that output's width, 18528, a request's T times it,
    # and the conv columns start at 1536. `decode_bits`: a cell whose tokens must
    # get the bits of T `gdp_recurrent` calls on decode's views of the same rows.
    _NEMOTRON4 = dict(
        heads=24,
        householder=3,
        key_dim=128,
        value_dim=64,
        scale=128**-0.5,
        proj_width=18528,
        conv_offset=1536,
        decode_bits=True,
    )
    # A tp4 rank runs the same mixer on 6 of the 24 heads, its in_proj output
    # [z 384 | conv 4224 | b 18 | a 6] = 4632 wide. Triton specializes both
    # kernels on each stride's divisibility by 16. Decode's row stride 4632 is
    # not divisible, and neither is a draft-2 request's 3 * 4632 = 13896, so a
    # token there rounds as its decode step does. A draft-1 request's 9264 is
    # divisible: that compiled variant rounds the reductions otherwise, within
    # the band of the reference but not with decode's bits (measured: in every
    # batch of two rows or more some output or state moved).
    _NEMOTRON4_TP4 = dict(_NEMOTRON4, heads=6, proj_width=4632, conv_offset=384)
    _NEMOTRON4_TP4_DRAFT1 = dict(_NEMOTRON4_TP4, decode_bits=False)

    # tokens is 1 + max_draft_len: 3 for the checkpoint's trained depth (draft
    # length 2), 2 for draft length 1. The batch sizes are the decode CUDA graphs
    # the target captures plus the eager size 3; a graph batch ends in one
    # PAD_SLOT_ID padding row. The pool is a GDP layer's view in one of the two
    # hybrid managers: "v1" (`CppMambaHybridCacheManager`) follows each slot's
    # ssm state with its conv state, a slot stride of 221952 floats; "v2"
    # (`MambaHybridCacheManagerV2`, the default) coalesces the 32 layers' ssm
    # states per slot, a stride of 6291456 floats, so a slot past 341 sits more
    # than 2^31 elements in. The two alternate along the batch sizes so each
    # draft length meets both; at batch 512 both draft lengths run on v2, with
    # slots that far in.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="draft 2, concurrency 1: one request, v2 pool",
            spec=dict(tokens=3, batch=1, pad_rows=0, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 2: one request and a padding row, v1 pool",
            spec=dict(tokens=3, batch=2, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, eager batch 3 (not a power of two), v1 pool",
            spec=dict(tokens=3, batch=3, pad_rows=0, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 4 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=4, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 8 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=8, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 16 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=16, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 32 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=32, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 64 (the CUDA-graph test's), v2 pool",
            spec=dict(tokens=3, batch=64, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 128 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=128, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 256 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 512: v2 slots past 2^31 elements",
            spec=dict(tokens=3, batch=512, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2, graph batch 1024 (a DP rank's batch), v1 pool",
            spec=dict(tokens=3, batch=1024, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, concurrency 1: one request, v1 pool",
            spec=dict(tokens=2, batch=1, pad_rows=0, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 2: one request and a padding row, v2 pool",
            spec=dict(tokens=2, batch=2, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, eager batch 3 (not a power of two), v2 pool",
            spec=dict(tokens=2, batch=3, pad_rows=0, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 4 with a padding row, v1 pool",
            spec=dict(tokens=2, batch=4, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 8 with a padding row, v2 pool",
            spec=dict(tokens=2, batch=8, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 16 with a padding row, v1 pool",
            spec=dict(tokens=2, batch=16, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 32 with a padding row, v2 pool",
            spec=dict(tokens=2, batch=32, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 64 with a padding row, v1 pool",
            spec=dict(tokens=2, batch=64, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 128 with a padding row, v2 pool",
            spec=dict(tokens=2, batch=128, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 256 with a padding row, v1 pool",
            spec=dict(tokens=2, batch=256, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 512: v2 slots past 2^31 elements",
            spec=dict(tokens=2, batch=512, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1, graph batch 1024 (a DP rank's batch), v1 pool",
            spec=dict(tokens=2, batch=1024, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="tp4 rank, draft 2, concurrency 1: one request, v2 pool",
            spec=dict(tokens=3, batch=1, pad_rows=0, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 2 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=2, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, eager batch 3, v1 pool",
            spec=dict(tokens=3, batch=3, pad_rows=0, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 4 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=4, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 8 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=8, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 16 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=16, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 32 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=32, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 64 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=64, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 128 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=128, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 256 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 512 with a padding row, v2 pool",
            spec=dict(tokens=3, batch=512, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 2, graph batch 1024 with a padding row, v1 pool",
            spec=dict(tokens=3, batch=1024, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, draft 1, concurrency 1: one request, v1 pool",
            spec=dict(tokens=2, batch=1, pad_rows=0, pool="v1", **_NEMOTRON4_TP4_DRAFT1),
        ),
        Cell(
            why="tp4 rank, draft 1, graph batch 2 with a padding row, v2 pool",
            spec=dict(tokens=2, batch=2, pad_rows=1, pool="v2", **_NEMOTRON4_TP4_DRAFT1),
        ),
        Cell(
            why="tp4 rank, draft 1, eager batch 3, v2 pool",
            spec=dict(tokens=2, batch=3, pad_rows=0, pool="v2", **_NEMOTRON4_TP4_DRAFT1),
        ),
        Cell(
            why="tp4 rank, draft 1, graph batch 64 with a padding row, v1 pool",
            spec=dict(tokens=2, batch=64, pad_rows=1, pool="v1", **_NEMOTRON4_TP4_DRAFT1),
        ),
        Cell(
            why="tp4 rank, draft 1, graph batch 256 with a padding row, v2 pool",
            spec=dict(tokens=2, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4_TP4_DRAFT1),
        ),
        Cell(
            why="tp4 rank, draft 1, graph batch 1024 with a padding row, v1 pool",
            spec=dict(tokens=2, batch=1024, pad_rows=1, pool="v1", **_NEMOTRON4_TP4_DRAFT1),
        ),
    )

    note = """
    Inputs: `gdp_recurrent`'s, one more leading dim for the T tokens of a
    request -- q `[B, T, H, K]` and k `[B, T, n_h, H, K]` not yet
    l2-normalized, v `[B, T, n_h, H, V]`, g `[B, T, H]` the fp32 log decay, beta
    `[B, T, n_h, H]` after sigmoid. Any strides: a verify step's generation
    rows are request-major, so the target passes `[B, T, ...]` views of the conv
    output.

    `state_pool` is `[num_slots, H, V, K]` fp32, K innermost, any slot stride,
    and is never written: the engine's `update_mamba_states` copies the
    accepted token's snapshot into the slot after acceptance. `intermediate` is
    a GDP layer's view of the manager's `intermediate_ssm_states`, `[rows, T,
    H, V, K]` fp32 with rows >= B; request `b` writes row `b`, position `t` holding
    the state after token `t`. Rows past B are not touched.

    Padding: a row whose slot is PAD_SLOT_ID (-1) starts from a zero state,
    carries it through its T tokens and writes nothing, its intermediate row
    included; its output is computed and meaningless. The commit still copies
    that row's snapshot -- whatever it held -- into the slot
    `update_mamba_states` is handed for the row. That has to be the engine's
    own, the dummy request's: its copy kernel does not skip a negative slot,
    so this entry's PAD_SLOT_ID there would be written before slot 0. Rows
    that share a slot are harmless here, unlike in `gdp_recurrent`: the slot
    is only read.

    Bit-identical to T `gdp_recurrent` calls on the same tokens, outputs and
    states, for every real request: each token is that kernel's body with its
    tile and warp count, and the state between tokens stays in fp32 registers,
    the value the pool round trip would have stored and reloaded. A change to
    either kernel's arithmetic or tiling breaks that, and so can a change in
    how Triton compiles the same statements: the token loop rolled
    (`tl.range`) instead of unrolled moved the snapshots' bits at a single
    request. The entry's test compares the two bit for bit. A padding row
    differs from its second token on, since a decode call restarts it from
    zero and keeps nothing.

    The bits hold only where both kernels compile alike: Triton specializes
    them on each stride's divisibility by 16. Every stride a token's q, k and v
    are read through -- the request's here, the row's in a decode call -- has
    to agree in it with decode's row stride. tp1's in_proj width 18528 and every
    multiple of it are divisible; a tp4 rank's 4632 is not, and neither is a
    draft-2 request's 13896, but a draft-1 request's 9264 is (and a draft-3
    request's 18528). There the verify stays within the reference's band but
    is not bitwise decode's: measured, and pinned by the entry's test. A target
    that needs those bits at an even T has to lay its requests out at a stride
    that is not a multiple of 16.

    Uncertified: T other than 2 and 3, an intermediate whose token dim exceeds
    T, a pool other than the two managers' layouts, and the shapes
    `gdp_recurrent` leaves uncertified. CUDA-graph capturable: no host sync and
    no allocation besides `out`.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        slots: torch.Tensor,
        intermediate: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Writes each real request's T states to `intermediate` and returns `out`."""
        batch, tokens, num_householder, num_heads, value_dim = v.shape
        key_dim = k.shape[-1]
        out = torch.empty((batch, tokens, num_heads, value_dim), dtype=q.dtype, device=q.device)
        num_rows = batch * num_heads
        if num_rows == 0:
            return out
        grid = (triton.cdiv(value_dim, _BLOCK_V), min(num_rows, _MAX_GRID_ROWS))
        _gdp_verify_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            out,
            state_pool,
            slots,
            intermediate,
            scale,
            num_rows,
            state_pool.shape[0],
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *g.stride(),
            *beta.stride(),
            *state_pool.stride(),
            *intermediate.stride(),
            T=tokens,
            H=num_heads,
            NH=num_householder,
            K=key_dim,
            V=value_dim,
            BK=triton.next_power_of_2(key_dim),
            BV=_BLOCK_V,
            num_warps=_NUM_WARPS,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        slots: torch.Tensor,
        intermediate: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, intermediate as the op leaves it)`, fp32 throughout; mutates nothing.

        Transcribed from `torch_recurrent_gated_delta_rule` with
        `use_qk_l2norm_in_kernel=True` over the request's T tokens, each
        expanded as `Nemotron4GDPMixer.forward` expands it: the decay rides on
        the token's first step only and the query is read only after its last.
        HF divides the l2-normalized query by `sqrt(key_dim)`; `scale` is that
        factor.

        The returned intermediate is a copy with each real request's row
        holding its T states; padding rows and rows past B keep what they held.
        A PAD_SLOT_ID row's output is computed from a zero state.
        """
        tokens = q.shape[1]
        pad = slots < 0
        rows = torch.where(pad, torch.zeros_like(slots), slots).long()
        state = state_pool[rows].float().transpose(-1, -2)  # [B, H, K, V]
        state = torch.where(pad[:, None, None, None], torch.zeros_like(state), state)

        query = _l2norm(q.float()) * scale
        key = _l2norm(k.float())
        value, step_beta, decay = v.float(), beta.float(), g.float().exp()
        outs, states = [], []
        for t in range(tokens):
            state = state * decay[:, t, :, None, None]
            for j in range(v.shape[2]):
                kv_mem = (state * key[:, t, j, :, :, None]).sum(dim=-2)
                delta = (value[:, t, j] - kv_mem) * step_beta[:, t, j, :, None]
                state = state + key[:, t, j, :, :, None] * delta[:, :, None, :]
            outs.append((state * query[:, t, :, :, None]).sum(dim=-2))
            states.append(state.transpose(-1, -2))

        after = intermediate.clone()
        real = (~pad).nonzero().squeeze(-1)
        after[real, :tokens] = torch.stack(states, 1)[real].to(intermediate.dtype)
        return torch.stack(outs, 1).to(q.dtype), after

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        slots: torch.Tensor,
        intermediate: torch.Tensor,
        scale: float,
    ) -> None:
        # The kernel takes its sizes from `v` and `k` and reads and writes every
        # other tensor through raw strides, so a tensor of the wrong shape is
        # addressed as whatever those sizes say rather than rejected.
        batch, tokens, num_householder, num_heads, value_dim = v.shape
        key_dim = k.shape[-1]
        assert k.shape == (batch, tokens, num_householder, num_heads, key_dim), (
            f"k must be [B, T, n_h, H, K] = "
            f"{(batch, tokens, num_householder, num_heads, key_dim)}; got {tuple(k.shape)}"
        )
        assert q.shape == (batch, tokens, num_heads, key_dim), (
            f"q must be [B, T, H, K] = {(batch, tokens, num_heads, key_dim)}, one query per "
            f"token; got {tuple(q.shape)}"
        )
        assert g.shape == (batch, tokens, num_heads), (
            f"g must be [B, T, H] = {(batch, tokens, num_heads)}, one decay per token; "
            f"got {tuple(g.shape)}"
        )
        assert beta.shape == (batch, tokens, num_householder, num_heads), (
            f"beta must be [B, T, n_h, H] = {(batch, tokens, num_householder, num_heads)}; "
            f"got {tuple(beta.shape)}"
        )
        assert slots.shape == (batch,) and not slots.is_floating_point(), (
            f"slots must be one integer per request; got {slots.dtype} {tuple(slots.shape)}"
        )
        # HF's [H, K, V] state has the same element count; read through these
        # strides it comes back transposed.
        assert tuple(state_pool.shape[1:]) == (num_heads, value_dim, key_dim), (
            f"state_pool must be [num_slots, H, V, K] = [*, {num_heads}, {value_dim}, {key_dim}] "
            f"with K innermost; got {tuple(state_pool.shape)}"
        )
        # A bf16 pool is loaded without complaint, starting every verify from a
        # rounded state; HF keeps it in fp32.
        assert state_pool.dtype == torch.float32, (
            f"state_pool holds the recurrent state in float32; got {state_pool.dtype}"
        )
        # Fewer rows than requests, or fewer positions than tokens, and the
        # stores run on into the next row -- another request's snapshots, or
        # past the buffer.
        assert (
            intermediate.dim() == 5
            and intermediate.shape[0] >= batch
            and intermediate.shape[1] >= tokens
            and tuple(intermediate.shape[2:]) == (num_heads, value_dim, key_dim)
        ), (
            f"intermediate must be [rows >= {batch}, {tokens}, {num_heads}, {value_dim}, "
            f"{key_dim}], a row of T states per request with K innermost; "
            f"got {tuple(intermediate.shape)}"
        )
        # A bf16 snapshot is stored without complaint, and the commit copies it
        # into the fp32 pool: the state rounded at every accepted step.
        assert intermediate.dtype == torch.float32, (
            f"intermediate holds the snapshots the commit copies into the pool in float32; "
            f"got {intermediate.dtype}"
        )
        # HF computes the decay in fp32. Rounding g to bf16 moves it by up to
        # |g| * 2^-9, so exp(g) by that much relative -- percents on the heads
        # that forget fastest -- and the state compounds it every token.
        assert g.dtype == torch.float32, f"g (the log decay) must be float32; got {g.dtype}"
        # The two mistakes the kernel cannot see: a positive g grows the state
        # every token instead of decaying it, and a pre-sigmoid beta over-writes it.
        assert bool((g <= 0).all()), (
            "g must be the log decay -exp(A_log) * softplus(a + dt_bias) <= 0"
        )
        assert bool(((beta >= 0) & (beta <= 1)).all()), (
            "beta must be post-sigmoid, in [0, 1]: pass sigmoid(b), not b"
        )
        # An out-of-range slot reads outside the pool; any negative slot is
        # treated as padding, so only PAD_SLOT_ID is the padding value.
        assert bool(((slots >= PAD_SLOT_ID) & (slots < state_pool.shape[0])).all()), (
            f"slots must be in [0, {state_pool.shape[0]}) or PAD_SLOT_ID ({PAD_SLOT_ID})"
        )


gdp_verify = GdpVerify()
