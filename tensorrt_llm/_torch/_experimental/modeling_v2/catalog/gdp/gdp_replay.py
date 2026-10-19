# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A generation step of the gated delta product with replay-style MTP state: drafts are kept as cached updates."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

from .._op import Arch, Cell, OpWrapper

#: Columns of the [K, V] state one program holds, and its warp count: the
#: state-stream tile (4 columns, one warp: 2 KiB of fp32 state in registers).
_BLOCK_V = 4
#: A tp4 rank's MTP-off decode from this many request x head rows up takes 8
#: columns a program (measured faster there; 4 below it).
_WIDE_DECODE_ROWS = 1536
#: Register cap of the replay / verify variant: it holds the state tile, a
#: token's inputs and the replayed drafts' cache lines, and above 64 registers
#: fewer warps fit an SM than its latency needs (measured: 80 uncapped).
_REPLAY_MAXNREG = 64
#: Rows of programs per SM in the prefetching decode grid; past that each
#: program grid-strides, loading its next row's state while it computes.
_ROWS_PER_SM = 16
#: CUDA's limit on the second grid dimension.
_MAX_GRID_ROWS = 65535


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit(do_not_specialize=["num_rows", "num_slots"])
def _gdp_replay_kernel(
    q,
    k,
    v,
    g,
    beta,
    out,
    state_pool,
    slots,
    pending,
    parity,
    cache_k,
    cache_u,
    cache_g,
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
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    """One program per (BV-column tile, request x head row), grid-strided over rows.

    The state tile is loaded once. The row's pending drafts (the previous
    step's accepted ones, cached as k-hat, u and the decay) are folded in as
    rank-one updates; then the T tokens run through the Householder steps in
    fp32 registers. The state after the first token is stored into the slot,
    and tokens 1..T-1 leave their k-hat, u and decay in the slot's other cache
    buffer instead of a state.

    The cache loads are masked by the slot alone, like the state's, so they
    issue with it; the pending count only gates their values (a draft not
    accepted contributes exp(0) and u = 0). Cache offsets are int32 (the
    wrapper checks they fit).
    """
    i_v = tl.program_id(0)
    i_row = tl.program_id(1)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]
    # The caches are contiguous [num_slots, 2, D, NH, H, K | V] and [num_slots, 2, D, H].
    ck_j: tl.constexpr = H * K
    ck_d: tl.constexpr = NH * ck_j
    cu_j: tl.constexpr = H * V
    cu_d: tl.constexpr = NH * cu_j

    if PREFETCH:
        n_next = (i_row // H).to(tl.int64)
        h_next = i_row % H
        slot_next = tl.load(slots + n_next, mask=i_row < num_rows, other=-1)
        b_next = tl.load(
            state_pool
            + slot_next.to(tl.int64) * stride_s_s
            + h_next * stride_s_h
            + o_k[:, None] * stride_s_k
            + o_v[None, :] * stride_s_v,
            mask=mask_kv & (slot_next >= 0),
            other=0.0,
            eviction_policy="evict_first",
        )
    while i_row < num_rows:
        if PREFETCH:
            i_n = n_next
            i_h = h_next
            slot = slot_next
            b_h = b_next.to(tl.float32)
            nxt = i_row + tl.num_programs(1)
            n_next = (nxt // H).to(tl.int64)
            h_next = nxt % H
            slot_next = tl.load(slots + n_next, mask=nxt < num_rows, other=-1)
            b_next = tl.load(
                state_pool
                + slot_next.to(tl.int64) * stride_s_s
                + h_next * stride_s_h
                + o_k[:, None] * stride_s_k
                + o_v[None, :] * stride_s_v,
                mask=mask_kv & (slot_next >= 0),
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            i_n = (i_row // H).to(tl.int64)
            i_h = i_row % H
            slot = tl.load(slots + i_n)
            b_h = tl.load(
                state_pool
                + slot.to(tl.int64) * stride_s_s
                + i_h * stride_s_h
                + o_k[:, None] * stride_s_k
                + o_v[None, :] * stride_s_v,
                mask=mask_kv & (slot >= 0),
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        live = slot >= 0
        tl.device_assert(slot < num_slots, "slot out of range")
        # Element [k, v] of the HF state lives at pool[slot, h, v, k].
        p_state = (
            state_pool
            + slot.to(tl.int64) * stride_s_s
            + i_h * stride_s_h
            + o_k[:, None] * stride_s_k
            + o_v[None, :] * stride_s_v
        )

        if D > 0:
            cnt = tl.load(pending + slot, mask=live, other=0)
            rb = tl.load(parity + slot, mask=live, other=0)
            rbase = (slot * 2 + rb) * D
            wbase = (slot * 2 + 1 - rb) * D
            for d in tl.static_range(D):
                f = d < cnt
                b_gd = tl.load(cache_g + (rbase + d) * H + i_h, mask=live, other=0.0)
                b_h *= tl.exp(tl.where(f, b_gd, 0.0))
                for j in tl.static_range(NH):
                    b_kk = tl.load(
                        cache_k + (rbase + d) * ck_d + j * ck_j + i_h * K + o_k,
                        mask=mask_k & live,
                        other=0.0,
                    )
                    b_uu = tl.load(
                        cache_u + (rbase + d) * cu_d + j * cu_j + i_h * V + o_v,
                        mask=mask_v & live,
                        other=0.0,
                    )
                    b_h += b_kk[:, None] * tl.where(f, b_uu, 0.0)[None, :]

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
                if D > 0:
                    if t >= 1:
                        tl.store(
                            cache_u + (wbase + t - 1) * cu_d + j * cu_j + i_h * V + o_v,
                            b_v,
                            mask=mask_v & live,
                            cache_modifier=".cs",
                        )
                        tl.store(
                            cache_k + (wbase + t - 1) * ck_d + j * ck_j + i_h * K + o_k,
                            b_k,
                            mask=mask_k & live & (i_v == 0),
                            cache_modifier=".cs",
                        )
            if D > 0:
                if t >= 1:
                    tl.store(cache_g + (wbase + t - 1) * H + i_h, b_g, mask=live & (i_v == 0))
            b_q = tl.load(
                q + i_n * stride_q_n + t * stride_q_t + i_h * stride_q_h + o_k * stride_q_d,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(
                out + ((i_n * T + t) * H + i_h) * V + o_v, b_o.to(out.dtype.element_ty), mask=mask_v
            )
            if t == 0:
                tl.store(
                    p_state,
                    b_h.to(state_pool.dtype.element_ty),
                    mask=mask_kv & live,
                    cache_modifier=".cs",
                )
        i_row += tl.num_programs(1)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """HF `l2norm`: `x * rsqrt(sum(x * x) + 1e-6)` over the head dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


class GdpReplay(OpWrapper):
    """A generation step of `torch_recurrent_gated_delta_rule`'s recurrence,
    Householder-expanded the way `Nemotron4GDPMixer.forward` expands it, with
    the MTP drafts kept as cached updates rather than states.

    Per (request, head), with `S` the HF `[K, V]` state read from the slot,
    `l2n(x) = x * rsqrt(sum(x*x) + 1e-6)` and `P` = `parity[slot]`:

        for d in range(pending[slot]):               # the previous step's accepted drafts
            S *= exp(cache_g[slot, P, d])
            for j in range(n_h):
                S += cache_k[slot, P, d, j] cache_u[slot, P, d, j]^T
        for t in range(T):                           # this step's tokens
            S *= exp(g[t])
            for j in range(n_h):
                kj = l2n(k[t, j]); u = beta[t, j] * (v[t, j] - S^T kj); S += kj u^T
                if t >= 1: cache_k[slot, 1-P, t-1, j] = kj; cache_u[slot, 1-P, t-1, j] = u
            if t >= 1: cache_g[slot, 1-P, t-1] = g[t]
            out[t] = S^T (l2n(q[t]) * scale)
            if t == 0: pool[slot] = S                # the commit: the first token is always kept

    A cached draft replays as exactly the update its verify made, because its
    `u` was computed from the state its accepted prefix leads to. So the slot
    always holds the state after a step's first token, and the next step folds
    in the drafts the sampler kept. That replaces per-position snapshots and the
    commit's copy: a step reads and writes one state per request, where the
    snapshot design reads one, writes 1 + D and copies one more.

    T = 1 with no caches (D = 0) is plain MTP-off decode: one token, in place.
    Returns `out` `[B, T, H, V]` in `q`'s dtype.
    """

    ARCHS = frozenset({Arch.SM_103})

    # nemotron4-nano-preview's GDP mixer, the shape every one of its 32 GDP and
    # linear_attention layers runs: gdp_num_heads 24 (== gdp_num_groups, so no
    # head repeat), gdp_key_head_dim 128, gdp_head_dim 64, gdp_num_householder 3.
    # `scale` is what a target passes for HF's `query / sqrt(key_head_dim)`. q,
    # k and v are views of the in_proj output's conv columns, a token per row: a
    # token's row stride is that output's width, 18528, a request's T times it,
    # and the conv columns start at 1536.
    _NEMOTRON4 = dict(
        heads=24,
        householder=3,
        key_dim=128,
        value_dim=64,
        scale=128**-0.5,
        proj_width=18528,
        conv_offset=1536,
    )
    # A tp4 rank runs the same mixer on 6 of the 24 heads, its in_proj output
    # [z 384 | conv 4224 | b 18 | a 6] = 4632 wide.
    _NEMOTRON4_TP4 = dict(_NEMOTRON4, heads=6, proj_width=4632, conv_offset=384)

    # tokens is 1 + the step's draft length; drafts (D) the caches' depth, the
    # engine's max_draft_len: 2 (the checkpoint's trained depth) or 1. A verify
    # step's rows replay 0..D pending drafts each, drawn per slot; a decode step
    # (tokens 1) runs with no cache under MTP off, and with one -- folding in a
    # verify's accepted drafts -- when a step without drafts follows a verify.
    # The batch sizes are the CUDA graphs the targets capture (to 1024, a DP
    # rank's batch) and the eager 3; a graph batch ends in a PAD_SLOT_ID row. The
    # pool is a GDP layer's view in one of the two hybrid managers: "v1"
    # (`CppMambaHybridCacheManager`) follows each slot's ssm state with its conv
    # state; "v2" (`MambaHybridCacheManagerV2`, the default) puts the 32 layers'
    # ssm states of a slot side by side, so slots past 341 sit beyond 2^31
    # elements. The two alternate along the cells.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, MTP off: concurrency 1, no padding row",
            spec=dict(tokens=1, drafts=0, batch=1, pad_rows=0, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="decode, MTP off: graph batch 2, a padding row",
            spec=dict(tokens=1, drafts=0, batch=2, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="decode, MTP off: eager batch 3 (not a power of two)",
            spec=dict(tokens=1, drafts=0, batch=3, pad_rows=0, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="decode, MTP off: graph batch 64",
            spec=dict(tokens=1, drafts=0, batch=64, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="decode, MTP off: graph batch 256",
            spec=dict(tokens=1, drafts=0, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="decode, MTP off: graph batch 1024 (a DP rank's), v2 past 2^31",
            spec=dict(tokens=1, drafts=0, batch=1024, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="decode folding a verify's pending drafts in, batch 4",
            spec=dict(tokens=1, drafts=2, batch=4, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="decode folding pending drafts in, graph batch 64",
            spec=dict(tokens=1, drafts=2, batch=64, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, concurrency 1",
            spec=dict(tokens=3, drafts=2, batch=1, pad_rows=0, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, graph batch 2 with a padding row",
            spec=dict(tokens=3, drafts=2, batch=2, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, eager batch 3",
            spec=dict(tokens=3, drafts=2, batch=3, pad_rows=0, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, graph batch 16",
            spec=dict(tokens=3, drafts=2, batch=16, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, graph batch 64",
            spec=dict(tokens=3, drafts=2, batch=64, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, graph batch 256",
            spec=dict(tokens=3, drafts=2, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, graph batch 512, v2 slots past 2^31",
            spec=dict(tokens=3, drafts=2, batch=512, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 2 verify, graph batch 1024 (a DP rank's)",
            spec=dict(tokens=3, drafts=2, batch=1024, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1 verify, concurrency 1",
            spec=dict(tokens=2, drafts=1, batch=1, pad_rows=0, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1 verify, graph batch 64",
            spec=dict(tokens=2, drafts=1, batch=64, pad_rows=1, pool="v1", **_NEMOTRON4),
        ),
        Cell(
            why="draft 1 verify, graph batch 256",
            spec=dict(tokens=2, drafts=1, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4),
        ),
        Cell(
            why="tp4 rank decode, MTP off: concurrency 1",
            spec=dict(tokens=1, drafts=0, batch=1, pad_rows=0, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank decode, MTP off: eager batch 3",
            spec=dict(tokens=1, drafts=0, batch=3, pad_rows=0, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank decode, MTP off: graph batch 64",
            spec=dict(tokens=1, drafts=0, batch=64, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank decode, MTP off: graph batch 256 (8 columns a program)",
            spec=dict(tokens=1, drafts=0, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank decode, MTP off: graph batch 1024",
            spec=dict(tokens=1, drafts=0, batch=1024, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank decode folding pending drafts in, batch 16",
            spec=dict(tokens=1, drafts=2, batch=16, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank draft 2 verify, concurrency 1",
            spec=dict(tokens=3, drafts=2, batch=1, pad_rows=0, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank draft 2 verify, graph batch 2 with a padding row",
            spec=dict(tokens=3, drafts=2, batch=2, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank draft 2 verify, eager batch 3",
            spec=dict(tokens=3, drafts=2, batch=3, pad_rows=0, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank draft 2 verify, graph batch 64",
            spec=dict(tokens=3, drafts=2, batch=64, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank draft 2 verify, graph batch 256",
            spec=dict(tokens=3, drafts=2, batch=256, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank draft 2 verify, graph batch 1024",
            spec=dict(tokens=3, drafts=2, batch=1024, pad_rows=1, pool="v1", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank draft 1 verify, graph batch 64",
            spec=dict(tokens=2, drafts=1, batch=64, pad_rows=1, pool="v2", **_NEMOTRON4_TP4),
        ),
    )

    note = """
    Inputs: `gdp_verify`'s -- q `[B, T, H, K]` and k `[B, T, n_h, H, K]` not yet
    l2-normalized, v `[B, T, n_h, H, V]`, g `[B, T, H]` the fp32 log decay
    `-exp(A_log) * softplus(a + dt_bias)`, beta `[B, T, n_h, H]` after sigmoid,
    any strides (the targets pass views of the conv output).

    State: `state_pool` `[num_slots, H, V, K]` fp32, K innermost, any slot
    stride. The caches belong to one GDP layer and are contiguous:
    `cache_k [num_slots, 2, D, n_h, H, K]` and `cache_u [num_slots, 2, D, n_h, H,
    V]` fp32 (normalized keys and the updates' u), `cache_g [num_slots, 2, D, H]`
    fp32 (the decay). `pending [num_slots]` and `parity [num_slots]` are int32
    and shared by all layers: how many cached drafts of the slot's last verify
    are still to be folded in (0..D), and which buffer holds them. This op
    reads both and writes neither -- after acceptance the caller sets `pending`
    to the step's accepted draft count (num_accepted - 1; 0 after a step that
    cached none) and flips `parity` of every slot whose step cached drafts.
    Under one-model MTP that is `SpecWorkerBase.commit_auxiliary_speculative_states`,
    which runs inside the verify step's CUDA graph after sampling. A new
    request's slot starts with `pending` 0.

    The pool holds the state after a step's first token, so whatever reads the
    slot between steps (another op, a disaggregated transfer) has to fold the
    pending drafts in first. The MTP layer reads no GDP state.

    Padding: a PAD_SLOT_ID row starts from a zero state, replays nothing,
    computes meaningless outputs and writes nothing. Real rows must have
    distinct slots: rows sharing one race on its state and caches.

    Speed (survey-gdp-recurrent, GB300, HBM-cold, per layer at B = 256): draft-2
    verify 243 us at 24 heads and 65 us at 6, against 248 / 86 for the snapshot
    design's verify plus the engine's commit (`gdp_verify` with the state-stream
    tile, `_promote_mamba_state_triton`). MTP-off decode takes that tile's
    prefetching grid at 24 heads (71 us) and 8 columns a program at a tp4 rank's
    B >= 256 (25 us).

    Uncertified: T > 3 or D > 2, other head counts, head dims or Householder
    count, a bf16 state pool or caches, a disaggregated or chunked hand-over of
    a slot with pending drafts. CUDA-graph capturable: no host sync and no
    allocation besides `out`.
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
        pending: torch.Tensor | None,
        parity: torch.Tensor | None,
        cache_k: torch.Tensor | None,
        cache_u: torch.Tensor | None,
        cache_g: torch.Tensor | None,
        scale: float,
    ) -> torch.Tensor:
        """Folds the slots' pending drafts in, runs the T tokens, commits the first and caches the rest."""
        batch, tokens, num_householder, num_heads, value_dim = v.shape
        key_dim = k.shape[-1]
        drafts = 0 if cache_k is None else cache_k.shape[2]
        out = torch.empty((batch, tokens, num_heads, value_dim), dtype=q.dtype, device=q.device)
        num_rows = batch * num_heads
        if num_rows == 0:
            return out
        if drafts == 0:
            pending = parity = slots
            cache_k = cache_u = cache_g = state_pool
        plain_decode = tokens == 1 and drafts == 0
        block_v = _BLOCK_V
        prefetch = False
        rows = min(num_rows, _MAX_GRID_ROWS)
        launch = {}
        if not plain_decode:
            launch["maxnreg"] = _REPLAY_MAXNREG
        elif num_heads >= 16:
            prefetch = True
            rows = min(rows, _ROWS_PER_SM * _num_sms(q.device.index))
        elif num_rows >= _WIDE_DECODE_ROWS:
            block_v = 2 * _BLOCK_V
        grid = (triton.cdiv(value_dim, block_v), rows)
        _gdp_replay_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            out,
            state_pool,
            slots,
            pending,
            parity,
            cache_k,
            cache_u,
            cache_g,
            scale,
            num_rows,
            state_pool.shape[0],
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *g.stride(),
            *beta.stride(),
            *state_pool.stride(),
            T=tokens,
            D=drafts,
            H=num_heads,
            NH=num_householder,
            K=key_dim,
            V=value_dim,
            BK=triton.next_power_of_2(key_dim),
            BV=block_v,
            PREFETCH=prefetch,
            num_warps=1,
            **launch,
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
        pending: torch.Tensor | None,
        parity: torch.Tensor | None,
        cache_k: torch.Tensor | None,
        cache_u: torch.Tensor | None,
        cache_g: torch.Tensor | None,
        scale: float,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """`(out, the pool, cache_k, cache_u, cache_g as the op leaves them)`, fp32 throughout; mutates nothing.

        Transcribed from `torch_recurrent_gated_delta_rule` with
        `use_qk_l2norm_in_kernel=True` over the step's tokens, each expanded as
        `Nemotron4GDPMixer.forward` expands it (the decay on a token's first
        Householder step, the query read after its last), preceded by the
        pending drafts' cached updates. `pending` and `parity` are read only.
        A PAD_SLOT_ID row's output comes from a zero state.
        """
        tokens = q.shape[1]
        pad = slots < 0
        rows = torch.where(pad, torch.zeros_like(slots), slots).long()
        state = state_pool[rows].float().transpose(-1, -2)  # [B, H, K, V]
        state = torch.where(pad[:, None, None, None], torch.zeros_like(state), state)
        if cache_k is not None:
            n_pending = torch.where(pad, 0, pending[rows]).long()
            read = parity[rows].long()
            for d in range(cache_k.shape[2]):
                keep = (d < n_pending)[:, None, None, None]
                upd = state * cache_g[rows, read, d].float().exp()[..., None, None]
                # Elementwise rank-one terms in the kernel's order: an fp32 matmul
                # here runs in TF32 and makes the reference the inexact side.
                for j in range(cache_k.shape[3]):
                    upd = upd + (
                        cache_k[rows, read, d, j].float()[..., :, None]
                        * cache_u[rows, read, d, j].float()[..., None, :]
                    )
                state = torch.where(keep, upd, state)

        query = _l2norm(q.float()) * scale
        key = _l2norm(k.float())
        value, step_beta, step_decay = v.float(), beta.float(), g.float().exp()
        outs, us = [], []
        committed = None
        for t in range(tokens):
            state = state * step_decay[:, t, :, None, None]
            token_u = []
            for j in range(v.shape[2]):
                kv_mem = (state * key[:, t, j, :, :, None]).sum(dim=-2)
                delta = (value[:, t, j] - kv_mem) * step_beta[:, t, j, :, None]
                state = state + key[:, t, j, :, :, None] * delta[:, :, None, :]
                token_u.append(delta)
            us.append(torch.stack(token_u, 1))  # [B, n_h, H, V]
            outs.append((state * query[:, t, :, :, None]).sum(dim=-2))
            if t == 0:
                committed = state

        real = ~pad
        pool_after = state_pool.clone()
        pool_after[slots[real].long()] = committed[real].transpose(-1, -2).to(state_pool.dtype)
        caches_after = (None, None, None)
        if cache_k is not None:
            ck, cu, cg = cache_k.clone(), cache_u.clone(), cache_g.clone()
            if tokens > 1:
                write = 1 - parity[slots[real].long()].long()
                real_slots = slots[real].long()
                for t in range(1, tokens):
                    ck[real_slots, write, t - 1] = key[real, t].to(ck.dtype)
                    cu[real_slots, write, t - 1] = us[t][real].to(cu.dtype)
                    cg[real_slots, write, t - 1] = g[real, t].float().to(cg.dtype)
            caches_after = (ck, cu, cg)
        return (torch.stack(outs, 1).to(q.dtype), pool_after, *caches_after)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        slots: torch.Tensor,
        pending: torch.Tensor | None,
        parity: torch.Tensor | None,
        cache_k: torch.Tensor | None,
        cache_u: torch.Tensor | None,
        cache_g: torch.Tensor | None,
        scale: float,
    ) -> None:
        # The kernel takes its sizes from `v`, `k` and the caches and reads every
        # tensor through raw strides or computed offsets, so a tensor of the
        # wrong shape is addressed as whatever those sizes say.
        batch, tokens, num_householder, num_heads, value_dim = v.shape
        key_dim = k.shape[-1]
        assert k.shape == (batch, tokens, num_householder, num_heads, key_dim), (
            f"k must be [B, T, n_h, H, K] = "
            f"{(batch, tokens, num_householder, num_heads, key_dim)}; got {tuple(k.shape)}"
        )
        assert q.shape == (batch, tokens, num_heads, key_dim), (
            f"q must be [B, T, H, K] = {(batch, tokens, num_heads, key_dim)}; got {tuple(q.shape)}"
        )
        assert g.shape == (batch, tokens, num_heads), (
            f"g must be [B, T, H] = {(batch, tokens, num_heads)}; got {tuple(g.shape)}"
        )
        assert beta.shape == (batch, tokens, num_householder, num_heads), (
            f"beta must be [B, T, n_h, H] = {(batch, tokens, num_householder, num_heads)}; "
            f"got {tuple(beta.shape)}"
        )
        assert slots.shape == (batch,) and slots.dtype == torch.int32, (
            f"slots must be int32, one per request; got {slots.dtype} {tuple(slots.shape)}"
        )
        num_slots = state_pool.shape[0]
        assert tuple(state_pool.shape[1:]) == (num_heads, value_dim, key_dim), (
            f"state_pool must be [num_slots, H, V, K] = [*, {num_heads}, {value_dim}, {key_dim}] "
            f"with K innermost; got {tuple(state_pool.shape)}"
        )
        # A bf16 pool or cache is loaded and stored without complaint, rounding
        # the recurrent state (or a draft's update) at every step; HF keeps fp32.
        assert state_pool.dtype == torch.float32, (
            f"state_pool holds the recurrent state in float32; got {state_pool.dtype}"
        )
        caches = (cache_k, cache_u, cache_g)
        assert all(c is None for c in caches) or all(c is not None for c in caches), (
            "cache_k, cache_u and cache_g go together"
        )
        if cache_k is None:
            # Without caches there is nowhere to keep tokens 1..T-1: their updates would be lost.
            assert tokens == 1, f"a step of {tokens} tokens needs the draft caches"
        else:
            drafts = cache_k.shape[2]
            assert drafts >= tokens - 1, (
                f"the caches hold {drafts} drafts; a step of {tokens} tokens caches {tokens - 1}"
            )
            assert cache_k.shape == (num_slots, 2, drafts, num_householder, num_heads, key_dim), (
                f"cache_k must be [num_slots, 2, D, n_h, H, K] = "
                f"{(num_slots, 2, drafts, num_householder, num_heads, key_dim)}; got {tuple(cache_k.shape)}"
            )
            assert cache_u.shape == (num_slots, 2, drafts, num_householder, num_heads, value_dim), (
                f"cache_u must be [num_slots, 2, D, n_h, H, V] = "
                f"{(num_slots, 2, drafts, num_householder, num_heads, value_dim)}; got {tuple(cache_u.shape)}"
            )
            assert cache_g.shape == (num_slots, 2, drafts, num_heads), (
                f"cache_g must be [num_slots, 2, D, H] = {(num_slots, 2, drafts, num_heads)}; "
                f"got {tuple(cache_g.shape)}"
            )
            # The kernel computes the cache offsets from the shapes alone.
            assert all(c.is_contiguous() for c in caches), "the caches must be contiguous"
            assert all(c.dtype == torch.float32 for c in caches), "the caches hold float32"
            assert num_slots * 2 * drafts * num_householder * num_heads * key_dim < 2**31, (
                "the cache offsets are int32"
            )
            assert pending is not None and parity is not None, "the caches need pending and parity"
            assert pending.shape == (num_slots,) and pending.dtype == torch.int32, (
                f"pending must be int32 [num_slots]; got {pending.dtype} {tuple(pending.shape)}"
            )
            assert parity.shape == (num_slots,) and parity.dtype == torch.int32, (
                f"parity must be int32 [num_slots]; got {parity.dtype} {tuple(parity.shape)}"
            )
            used = slots[slots >= 0].long()
            # More pending drafts than the caches hold replay the next slot's lines.
            assert bool(((pending[used] >= 0) & (pending[used] <= drafts)).all()), (
                f"pending must be in [0, {drafts}] for every slot in the step"
            )
            # Any other parity addresses outside the slot's two buffers.
            assert bool(((parity[used] == 0) | (parity[used] == 1)).all()), "parity must be 0 or 1"
        # HF computes the decay in fp32; a bf16 one compounds through the state.
        assert g.dtype == torch.float32, f"g (the log decay) must be float32; got {g.dtype}"
        assert bool((g <= 0).all()), (
            "g must be the log decay -exp(A_log) * softplus(a + dt_bias) <= 0"
        )
        assert bool(((beta >= 0) & (beta <= 1)).all()), (
            "beta must be post-sigmoid, in [0, 1]: pass sigmoid(b), not b"
        )
        assert bool(((slots >= PAD_SLOT_ID) & (slots < num_slots)).all()), (
            f"slots must be in [0, {num_slots}) or PAD_SLOT_ID ({PAD_SLOT_ID})"
        )
        real = slots[slots >= 0]
        assert real.unique().numel() == real.numel(), (
            "two rows share a slot and race on its state and caches; give padding rows PAD_SLOT_ID"
        )


gdp_replay = GdpReplay()
