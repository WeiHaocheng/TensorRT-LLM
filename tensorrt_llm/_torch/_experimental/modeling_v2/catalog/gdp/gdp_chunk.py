# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gated delta product over a varlen batch of prefill sequences, final states written back by slot."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from tensorrt_llm._torch.modules.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
from tensorrt_llm._torch.modules.fla.op import exp, safe_exp
from tensorrt_llm._torch.modules.fla.solve_tril import (
    merge_16x16_to_64x64_inverse_kernel,
    solve_tril_16x16_kernel,
)
from tensorrt_llm._torch.modules.fla.wy_fast import recompute_w_u_fwd_kernel

from .._op import Arch, Cell, OpWrapper

#: The vendored chunk kernels' block length, counted in Householder sub-steps.
_BT = 64
#: `solve_tril`'s diagonal blocks, inverted before they are merged into a chunk's.
_BT_DIAG = 16


def _jit(kernel):
    """The `triton.jit` function under a vendored kernel's heuristics and autotune
    wrappers, so the entry can launch it with one fixed config."""
    while not isinstance(kernel, triton.runtime.jit.JITFunction):
        kernel = kernel.fn
    return kernel


#: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` without its autotuner: tuned
#: per process, it would pick its config from timings that move between
#: processes, and its list stops at BV 32 where 16 is faster at these shapes.
_FWD_H = _jit(chunk_gated_delta_rule_fwd_kernel_h_blockdim64)

#: Every stage's launch config, fixed: the fastest of each kernel's configs at the
#: 8K prefill, timed again at 2K, 16K and multi-sequence batches.
_CONFIG = dict(
    cumsum=dict(num_warps=1),
    kkt=dict(BK=64, num_warps=2, num_stages=3),
    solve=dict(num_warps=1, num_stages=4),
    merge=dict(num_warps=2, num_stages=3),
    w_u=dict(BK=32, BV=32, num_warps=4, num_stages=1),
    fwd_h=dict(BV=16, num_warps=2, num_stages=3),
    o=dict(BK=64, BV=64, num_warps=4, num_stages=3),
)


@triton.jit
def _chunk_list_kernel(
    cu_steps,
    chunk_indices,
    chunk_offsets,
    num_seqs,
    num_chunks,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BN: tl.constexpr,
):
    """`prepare_chunk_indices` and `prepare_chunk_offsets` on the device.

    Entry `c` of `chunk_indices` is `(sequence, chunk within it)` for the batch's
    `c`-th BT-sub-step chunk, sequences in order; `chunk_offsets[n]` is sequence
    n's first chunk. Each program covers BC chunk ids and walks the sequences BN
    at a time: a chunk's sequence is the number of sequences that end at or
    before it, its first chunk the last sequence start at or before it.

    `num_chunks` is a bound, not the count. The ids past the real chunks get the
    last sequence and a chunk past its end, which every stage masks out.
    """
    i_c = tl.program_id(0)
    o_c = i_c * BC + tl.arange(0, BC)
    seq = tl.zeros([BC], dtype=tl.int32)
    first = tl.zeros([BC], dtype=tl.int32)
    carry = tl.sum(tl.zeros([BN], dtype=tl.int32), axis=0)
    for n0 in range(0, num_seqs, BN):
        o_n = n0 + tl.arange(0, BN)
        m_n = o_n < num_seqs
        bos = tl.load(cu_steps + o_n, mask=m_n, other=0).to(tl.int32)
        eos = tl.load(cu_steps + o_n + 1, mask=m_n, other=0).to(tl.int32)
        count = tl.cdiv(eos - bos, BT)
        end = carry + tl.cumsum(count, axis=0)
        start = end - count
        if i_c == 0:
            tl.store(chunk_offsets + o_n, start, mask=m_n)
            tl.store(chunk_offsets + o_n + 1, end, mask=m_n)
        ended = m_n[None, :] & (end[None, :] <= o_c[:, None])
        seq += tl.sum(ended.to(tl.int32), axis=1)
        started = m_n[None, :] & (start[None, :] <= o_c[:, None])
        first = tl.maximum(first, tl.max(tl.where(started, start[None, :], 0), axis=1))
        carry += tl.sum(count, axis=0)
    seq = tl.minimum(seq, num_seqs - 1)
    m_c = o_c < num_chunks
    tl.store(chunk_indices + o_c * 2, seq, mask=m_c)
    tl.store(chunk_indices + o_c * 2 + 1, o_c - first, mask=m_c)


def _chunk_list(
    cu_steps: torch.Tensor, steps: int, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(chunk_indices, chunk_offsets)` for `chunk_size`-sub-step chunks, built on the device.

    The list has `cdiv(steps, chunk_size) + num_seqs - 1` entries, a bound the
    shapes give: rounding up, each sequence after the first adds at most one chunk
    to `cdiv(steps, chunk_size)`. So no batch has more chunks, and one sequence has
    exactly that many.
    """
    num_seqs = cu_steps.shape[0] - 1
    num_chunks = triton.cdiv(steps, chunk_size) + num_seqs - 1
    chunk_indices = cu_steps.new_empty((num_chunks, 2))
    chunk_offsets = cu_steps.new_empty((num_seqs + 1,))
    block = 64
    _chunk_list_kernel[(triton.cdiv(num_chunks, block),)](
        cu_steps,
        chunk_indices,
        chunk_offsets,
        num_seqs,
        num_chunks,
        BT=chunk_size,
        BC=block,
        BN=block,
        num_warps=4,
    )
    return chunk_indices, chunk_offsets


@triton.jit
def _expand_kernel(
    q,
    k,
    v,
    beta,
    q_norm,
    k_steps,
    v_steps,
    beta_steps,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_j,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_j,
    stride_v_h,
    stride_v_d,
    stride_b_t,
    stride_b_j,
    stride_b_h,
    H: tl.constexpr,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """One program per token: l2-normalize its query and its NH keys in fp32, and
    lay out the keys, values and betas as NH consecutive sub-steps of `[steps, H,
    *]`, the strided views the target passes made dense. The query stays one row
    per token, `[T, H, K]`: only a token's last sub-step reads it."""
    i_t = tl.program_id(0).to(tl.int64)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_h = o_h < H
    mask = m_h[:, None] & (o_k < K)[None, :]
    mask_v = m_h[:, None] & (o_v < V)[None, :]
    o_step = o_h[:, None] * K + o_k[None, :]
    o_step_v = o_h[:, None] * V + o_v[None, :]
    b_q = tl.load(
        q + i_t * stride_q_t + o_h[:, None] * stride_q_h + o_k[None, :] * stride_q_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q, axis=1) + 1e-6)[:, None]
    tl.store(q_norm + i_t * H * K + o_step, b_q.to(q_norm.dtype.element_ty), mask=mask)
    for j in tl.static_range(NH):
        i_s = i_t * NH + j
        b_k = tl.load(
            k
            + i_t * stride_k_t
            + j * stride_k_j
            + o_h[:, None] * stride_k_h
            + o_k[None, :] * stride_k_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k, axis=1) + 1e-6)[:, None]
        tl.store(k_steps + i_s * H * K + o_step, b_k.to(k_steps.dtype.element_ty), mask=mask)
        b_v = tl.load(
            v
            + i_t * stride_v_t
            + j * stride_v_j
            + o_h[:, None] * stride_v_h
            + o_v[None, :] * stride_v_d,
            mask=mask_v,
            other=0.0,
        )
        tl.store(v_steps + i_s * H * V + o_step_v, b_v, mask=mask_v)
        b_b = tl.load(beta + i_t * stride_b_t + j * stride_b_j + o_h * stride_b_h, mask=m_h)
        tl.store(beta_steps + i_s * H + o_h, b_b, mask=m_h)


@triton.jit
def _decay_cumsum_kernel(
    g,
    g_cumsum,
    cu_steps,
    chunk_indices,
    stride_g_t,
    stride_g_h,
    H: tl.constexpr,
    NH: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    """One program per chunk: the running sum of the log decay over the chunk's
    sub-steps, restarting at every chunk, added one sub-step at a time.

    Sequential on purpose. Every term is <= 0, so a running sum never rises and a
    sub-step that adds zero repeats its predecessor exactly; the vendored kernels
    weight a pair of sub-steps by `safe_exp(later - earlier)`, which is exactly one
    across a token's sub-steps here. A parallel scan rounds each prefix its own
    way, and the one-ulp rises it leaves on those flat runs are zeroed by
    `safe_exp`, dropping the coupling between a token's Householder steps.
    """
    i_c = tl.program_id(0)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int64)
    i_b = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int64)
    bos = tl.load(cu_steps + i_n).to(tl.int64)
    eos = tl.load(cu_steps + i_n + 1).to(tl.int64)
    o_s = tl.arange(0, BT)
    o_h = tl.arange(0, BH)
    s = bos + i_b * BT + o_s
    live = s < eos
    mask_h = o_h < H
    # The decay rides on a token's first sub-step; the others add zero.
    b_g = tl.load(
        g + (s // NH)[:, None] * stride_g_t + o_h[None, :] * stride_g_h,
        mask=(live & (s % NH == 0))[:, None] & mask_h[None, :],
        other=0.0,
    ).to(tl.float32)
    b_acc = tl.zeros([BH], dtype=tl.float32)
    b_out = tl.zeros([BT, BH], dtype=tl.float32)
    for i in tl.static_range(BT):
        # Row i alone is non-zero in the selection, so its sum is exact.
        b_acc += tl.sum(tl.where(o_s[:, None] == i, b_g, 0.0), axis=0)
        b_out = tl.where(o_s[:, None] == i, b_acc[None, :], b_out)
    tl.store(
        g_cumsum + s[:, None] * H + o_h[None, :],
        b_out,
        mask=live[:, None] & mask_h[None, :],
    )


@triton.jit
def _last_step_o_kernel(
    q_norm,
    k_steps,
    v_new,
    h,
    g_cumsum,
    out,
    cu_steps,
    chunk_indices,
    scale,
    H: tl.constexpr,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """`chunk_fwd_kernel_o` for the rows that are a token's last sub-step only.

    Program (V block, chunk, head). A chunk's rows are its sub-steps `s` with `s %
    NH == NH - 1` (a sequence starts on a multiple of NH): at most `cdiv(BT, NH)`
    of them, `NH` apart, so row `r` is token `first // NH + r` and the query is
    read from `q_norm` one row per token, the output written to `out` the same way.
    Per row the arithmetic is the vendored kernel's: the state entering the chunk
    read through the query, plus the chunk's own earlier sub-steps through the
    causal, decay-weighted `q k^T`, accumulated over K in BK blocks.
    """
    i_v, i_c, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    i_b = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu_steps + i_n).to(tl.int32)
    eos = tl.load(cu_steps + i_n + 1).to(tl.int32)
    s0 = bos + i_b * BT
    s_end = tl.minimum(s0 + BT, eos)
    first = s0 + (NH - 1 - s0 % NH)
    # The kept rows are tokens t_first .. t_end - 1; past them every block below
    # reads zeros and the store writes nothing.
    t_first = first // NH
    t_end = tl.maximum(t_first, s_end // NH)
    o_r = tl.arange(0, BR)
    s_r = first + o_r * NH
    s_j = s0 + tl.arange(0, BT)

    h += (i_c * H + i_h).to(tl.int64) * K * V
    b_o = tl.zeros([BR, BV], dtype=tl.float32)
    b_A = tl.zeros([BR, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q_norm + i_h * K, (t_end, K), (H * K, 1), (t_first, i_k * BK), (BR, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k_steps + i_h * K, (K, s_end), (1, H * K), (i_k * BK, s0), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BR, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    p_g_r = tl.make_block_ptr(
        g_cumsum + first * H + i_h, (t_end - t_first,), (NH * H,), (0,), (BR,), (0,)
    )
    p_g_j = tl.make_block_ptr(g_cumsum + i_h, (s_end,), (H,), (s0,), (BT,), (0,))
    b_g_r = tl.load(p_g_r, boundary_check=(0,))
    b_g_j = tl.load(p_g_j, boundary_check=(0,))
    b_o = b_o * exp(b_g_r)[:, None]
    b_A = b_A * safe_exp(b_g_r[:, None] - b_g_j[None, :])
    b_A = tl.where(s_r[:, None] >= s_j[None, :], b_A, 0)

    p_v = tl.make_block_ptr(
        v_new + i_h * V, (s_end, V), (H * V, 1), (s0, i_v * BV), (BT, BV), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    p_o = tl.make_block_ptr(
        out + i_h * V, (t_end, V), (H * V, 1), (t_first, i_v * BV), (BR, BV), (1, 0)
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _gdp_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Expand each token into its NH Householder sub-steps and run the vendored
    gated-delta-rule chunk stages over the expanded sequences.

    The stages' kernels are launched one by one rather than through the vendored
    wrapper for four reasons, all measured: its `chunk_local_cumsum` is the
    parallel scan `_decay_cumsum_kernel` replaces; its indexed in-place state path
    is autotuned without restoring the state, so the first call per shape
    advances the pool on every benchmark run of every config it tries; each
    stage's wrapper builds its chunk list on the host, two device round trips
    per list; and its output stage computes every sub-step's output where only a
    token's last is kept. Here the states are gathered into a buffer the kernel
    only reads and the final states scattered back after it, the chunk lists come
    from `_chunk_list`, and `_last_step_o_kernel` computes the kept rows straight
    into `out`. Every stage is launched over a grid sized from a bound instead of
    the count, with its config from `_CONFIG`, none of them autotuned.
    """
    num_tokens, num_householder, num_heads, value_dim = v.shape
    key_dim = k.shape[-1]
    out = torch.empty((num_tokens, num_heads, value_dim), dtype=q.dtype, device=q.device)
    if num_tokens == 0:
        return out
    steps = num_tokens * num_householder
    num_seqs = cu_seqlens.shape[0] - 1
    cu_steps = cu_seqlens.to(torch.long) * num_householder
    chunks, chunk_offsets = _chunk_list(cu_steps, steps, _BT)
    diag_blocks, _ = _chunk_list(cu_steps, steps, _BT_DIAG)
    num_chunks = chunks.shape[0]

    q_norm = torch.empty((num_tokens, num_heads, key_dim), dtype=q.dtype, device=q.device)
    k_steps = torch.empty((1, steps, num_heads, key_dim), dtype=k.dtype, device=q.device)
    v_steps = torch.empty((1, steps, num_heads, value_dim), dtype=v.dtype, device=q.device)
    beta_steps = torch.empty((1, steps, num_heads), dtype=beta.dtype, device=q.device)
    _expand_kernel[(num_tokens,)](
        q,
        k,
        v,
        beta,
        q_norm,
        k_steps,
        v_steps,
        beta_steps,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *beta.stride(),
        H=num_heads,
        NH=num_householder,
        K=key_dim,
        V=value_dim,
        BH=triton.next_power_of_2(num_heads),
        BK=triton.next_power_of_2(key_dim),
        BV=triton.next_power_of_2(value_dim),
        num_warps=4,
    )

    g_cumsum = torch.empty((1, steps, num_heads), dtype=torch.float32, device=q.device)
    _decay_cumsum_kernel[(num_chunks,)](
        g,
        g_cumsum,
        cu_steps,
        chunks,
        *g.stride(),
        H=num_heads,
        NH=num_householder,
        BT=_BT,
        BH=triton.next_power_of_2(num_heads),
        **_CONFIG["cumsum"],
    )

    # chunk_scaled_dot_kkt_fwd: beta * K K^T within each chunk, strictly lower.
    a = torch.empty((1, steps, num_heads, _BT), dtype=torch.float32, device=q.device)
    chunk_scaled_dot_kkt_fwd_kernel[(num_chunks, num_heads)](
        k=k_steps,
        beta=beta_steps,
        g_cumsum=g_cumsum,
        A=a,
        cu_seqlens=cu_steps,
        chunk_indices=chunks,
        T=steps,
        H=num_heads,
        Hg=num_heads,
        K=key_dim,
        BT=_BT,
        **_CONFIG["kkt"],
    )
    # solve_tril: (I + A)^-1, the 16-sub-step diagonal blocks first, then merged.
    a_diag = torch.empty((1, steps, num_heads, _BT_DIAG), dtype=torch.float32, device=q.device)
    solve_tril_16x16_kernel[(diag_blocks.shape[0], num_heads)](
        A=a,
        Ad=a_diag,
        cu_seqlens=cu_steps,
        chunk_indices=diag_blocks,
        T=steps,
        H=num_heads,
        BT=_BT,
        **_CONFIG["solve"],
    )
    a_inv = torch.empty((1, steps, num_heads, _BT), dtype=k_steps.dtype, device=q.device)
    merge_16x16_to_64x64_inverse_kernel[(num_chunks, num_heads)](
        A=a,
        Ad=a_diag,
        Ai=a_inv,
        cu_seqlens=cu_steps,
        chunk_indices=chunks,
        T=steps,
        H=num_heads,
        BT=_BT,
        **_CONFIG["merge"],
    )
    # recompute_w_u_fwd: the WY factors.
    u = torch.empty_like(v_steps)
    w = k_steps.new_empty(1, steps, num_heads, key_dim)
    recompute_w_u_fwd_kernel[(num_chunks, num_heads)](
        k=k_steps,
        v=v_steps,
        beta=beta_steps,
        w=w,
        u=u,
        A=a_inv,
        g=g_cumsum,
        cu_seqlens=cu_steps,
        chunk_indices=chunks,
        T=steps,
        H=num_heads,
        Hg=num_heads,
        K=key_dim,
        V=value_dim,
        BT=_BT,
        **_CONFIG["w_u"],
    )
    rows = slots.to(torch.long)
    # A fresh sequence's slot holds whatever its previous owner left there.
    initial = torch.where(has_initial_state[:, None, None, None], state_pool[rows], 0.0)
    # chunk_gated_delta_rule_fwd_h: the state entering each chunk, one program per
    # sequence and head, so its grid needs only the sequence count.
    h = k_steps.new_empty(1, num_chunks, num_heads, key_dim, value_dim)
    v_new = torch.empty_like(u)
    final = torch.empty(
        (num_seqs, num_heads, value_dim, key_dim), dtype=torch.float32, device=q.device
    )
    _FWD_H[(triton.cdiv(value_dim, _CONFIG["fwd_h"]["BV"]), num_seqs * num_heads)](
        k=k_steps,
        v=u,
        w=w,
        v_new=v_new,
        g=g_cumsum,
        h=h,
        h0=initial,
        h0_i=None,
        ht=final,
        cu_seqlens=cu_steps,
        chunk_offsets=chunk_offsets,
        T=steps,
        stride_h0=initial.stride(0),
        H=num_heads,
        Hg=num_heads,
        K=key_dim,
        V=value_dim,
        BT=_BT,
        USE_G=True,
        USE_INITIAL_STATE=True,
        USE_INDEXED_STATE=False,
        STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=True,
        IS_VARLEN=True,
        **_CONFIG["fwd_h"],
    )
    # The output of each token's last sub-step, the only one the mixer keeps.
    _last_step_o_kernel[(triton.cdiv(value_dim, _CONFIG["o"]["BV"]), num_chunks, num_heads)](
        q_norm,
        k_steps,
        v_new,
        h,
        g_cumsum,
        out,
        cu_steps,
        chunks,
        scale,
        H=num_heads,
        NH=num_householder,
        K=key_dim,
        V=value_dim,
        BT=_BT,
        BR=triton.next_power_of_2(triton.cdiv(_BT, num_householder)),
        **_CONFIG["o"],
    )
    state_pool[rows] = final.to(state_pool.dtype)
    return out


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """HF `l2norm`: `x * rsqrt(sum(x * x) + 1e-6)` over the head dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


class GdpChunk(OpWrapper):
    """A varlen batch of prefill sequences through `torch_recurrent_gated_delta_rule`'s
    recurrence, Householder-expanded the way `Nemotron4GDPMixer.forward` expands
    it, each sequence starting from zero or from its slot's saved state and
    leaving its final state in that slot.

    Per sequence and head, with `S` the HF `[K, V]` state (zero unless the
    sequence's `has_initial_state` is set, then the slot's) and `l2n(x) = x *
    rsqrt(sum(x*x) + 1e-6)`, for each of its tokens in order:

        S *= exp(g)                                  # once, before the first step
        for j in range(n_h):                         # the token's Householder steps
            kj = l2n(k[j]); u = beta[j] * (v[j] - S^T kj); S += kj u^T
        out = S^T (l2n(q) * scale)                   # read after the last step

    Returns `out` `[T, H, V]` in `q`'s dtype -- the per-head output before
    `Nemotron4RMSNormGated` -- and leaves each sequence's final state in
    `state_pool[slots[n]]`; a caller that wants the old states has to have kept
    its own copy.
    """

    ARCHS = frozenset({Arch.SM_103})

    # nemotron4-nano-preview's GDP mixer, the one shape all 32 of its GDP and
    # linear_attention layers share: gdp_num_heads 24 (== gdp_num_groups, so no
    # head repeat), gdp_key_head_dim 128, gdp_head_dim 64, gdp_num_householder 3.
    # `scale` is what the target passes for HF's `query / sqrt(key_head_dim)`.
    _NEMOTRON4 = dict(heads=24, householder=3, key_dim=128, value_dim=64, scale=128**-0.5)
    # A tp4 rank runs the same mixer on 6 of the 24 heads.
    _NEMOTRON4_TP4 = dict(_NEMOTRON4, heads=6)

    # A token is 3 sub-steps and a chunk 64, so a chunk boundary falls inside a
    # token in 2 of every 3 chunks. Lengths run from one token to the 16384-token
    # prefill budget; the two batches are prefill mixes with sequences that span
    # less than one chunk, exactly one, and many. "fresh" sequences start from
    # zero over slots holding a stale state they must not read; "continued" ones
    # resume a saved state (chunked prefill); "mixed" alternates the two in one
    # batch. The same eighteen batches run at both geometries: tensor parallelism
    # splits the heads, not the tokens.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="1 token, fresh: 3 sub-steps, one partial chunk",
            spec=dict(seq_lens=(1,), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="1 token, continued: a one-token prefill resume",
            spec=dict(seq_lens=(1,), init="continued", **_NEMOTRON4),
        ),
        Cell(
            why="7 tokens, fresh: 21 sub-steps in one chunk",
            spec=dict(seq_lens=(7,), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="7 tokens, continued",
            spec=dict(seq_lens=(7,), init="continued", **_NEMOTRON4),
        ),
        Cell(
            why="64 tokens, fresh: 192 sub-steps, exactly 3 chunks",
            spec=dict(seq_lens=(64,), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="64 tokens, continued",
            spec=dict(seq_lens=(64,), init="continued", **_NEMOTRON4),
        ),
        Cell(
            why="512 tokens, fresh: 24 chunks",
            spec=dict(seq_lens=(512,), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="512 tokens, continued",
            spec=dict(seq_lens=(512,), init="continued", **_NEMOTRON4),
        ),
        Cell(
            why="2048 tokens, fresh: 96 chunks",
            spec=dict(seq_lens=(2048,), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="2048 tokens, continued",
            spec=dict(seq_lens=(2048,), init="continued", **_NEMOTRON4),
        ),
        Cell(
            why="8192 tokens, fresh: the max_num_tokens budget",
            spec=dict(seq_lens=(8192,), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="8192 tokens, continued: a resumed prefill chunk",
            spec=dict(seq_lens=(8192,), init="continued", **_NEMOTRON4),
        ),
        Cell(
            why="16384 tokens, fresh: the larger prefill budget",
            spec=dict(seq_lens=(16384,), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="16384 tokens, continued",
            spec=dict(seq_lens=(16384,), init="continued", **_NEMOTRON4),
        ),
        Cell(
            why="4 sequences summing to 8192, all fresh",
            spec=dict(seq_lens=(3, 509, 2600, 5080), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="4 sequences summing to 8192, even ones continued",
            spec=dict(seq_lens=(3, 509, 2600, 5080), init="mixed", **_NEMOTRON4),
        ),
        Cell(
            why="8 sequences summing to 2048, all fresh: 1 and 2 tokens to 1024",
            spec=dict(seq_lens=(1, 2, 17, 64, 129, 300, 511, 1024), init="fresh", **_NEMOTRON4),
        ),
        Cell(
            why="8 sequences summing to 2048, even ones continued",
            spec=dict(seq_lens=(1, 2, 17, 64, 129, 300, 511, 1024), init="mixed", **_NEMOTRON4),
        ),
        Cell(
            why="tp4 rank, 1 token, fresh",
            spec=dict(seq_lens=(1,), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 1 token, continued",
            spec=dict(seq_lens=(1,), init="continued", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 7 tokens, fresh: one chunk",
            spec=dict(seq_lens=(7,), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 7 tokens, continued",
            spec=dict(seq_lens=(7,), init="continued", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 64 tokens, fresh: exactly 3 chunks",
            spec=dict(seq_lens=(64,), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 64 tokens, continued",
            spec=dict(seq_lens=(64,), init="continued", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 512 tokens, fresh: 24 chunks",
            spec=dict(seq_lens=(512,), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 512 tokens, continued",
            spec=dict(seq_lens=(512,), init="continued", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 2048 tokens, fresh: 96 chunks",
            spec=dict(seq_lens=(2048,), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 2048 tokens, continued",
            spec=dict(seq_lens=(2048,), init="continued", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 8192 tokens, fresh: max_num_tokens",
            spec=dict(seq_lens=(8192,), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 8192 tokens, continued",
            spec=dict(seq_lens=(8192,), init="continued", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 16384 tokens, fresh",
            spec=dict(seq_lens=(16384,), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 16384 tokens, continued",
            spec=dict(seq_lens=(16384,), init="continued", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 4 sequences summing to 8192, all fresh",
            spec=dict(seq_lens=(3, 509, 2600, 5080), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 4 sequences summing to 8192, even ones continued",
            spec=dict(seq_lens=(3, 509, 2600, 5080), init="mixed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 8 sequences summing to 2048, all fresh",
            spec=dict(seq_lens=(1, 2, 17, 64, 129, 300, 511, 1024), init="fresh", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, 8 sequences summing to 2048, even ones continued",
            spec=dict(seq_lens=(1, 2, 17, 64, 129, 300, 511, 1024), init="mixed", **_NEMOTRON4_TP4),
        ),
    )

    note = """
    Inputs, shared with `gdp_recurrent`: what `Nemotron4GDPMixer.forward` hands
    `torch_recurrent_gated_delta_rule` (after in_proj, the short conv + SiLU,
    sigmoid and the softplus decay), before its n_h-fold expansion -- q `[T, H,
    K]` and k `[T, n_h, H, K]` not yet l2-normalized, v `[T, n_h, H, V]`, g `[T,
    H]` the fp32 log decay `-exp(A_log) * softplus(a + dt_bias)`, beta `[T, n_h,
    H]` after sigmoid. Any strides: the target passes views of the conv output.
    Sequence n is tokens `cu_seqlens[n]:cu_seqlens[n+1]`, its state slot
    `slots[n]`; `has_initial_state[n]` False starts it from zero without reading
    the slot, which may hold a previous request's state.

    State layout: `state_pool` is `[num_slots, H, V, K]` fp32 with K innermost,
    the pool layout of the vendored FLA kernels (`modules/fla/chunk_delta_h.py`,
    `modules/fla/fused_recurrent.py`) -- HF's `[H, K, V]` state transposed. Any
    strides: the slots are gathered and scattered by index.

    Numerics: the vendored chunk stages carry the l2-normalized q and k, the WY
    factors, the per-chunk states and the corrected values in bf16 between
    stages and the running state in fp32; the result is a few bf16 ulps from
    the fp32 per-token recurrence, which is what the test derives its band from.

    Two vendored-path hazards this entry routes around (the test pins both):
    `chunk_local_cumsum` is a parallel scan whose rounding drops the coupling of
    a token's Householder steps (see `_decay_cumsum_kernel`), and the indexed
    in-place state path of `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is
    autotuned without `restore_value`, so its first call per shape advances
    the pool on every benchmark run.

    No host syncs: the chunk lists the vendored stages index are built on the
    device (`_chunk_list`) and every grid is sized from the token and sequence
    counts, never from the lengths, where each stage's wrapper reads the lengths
    on the host with `prepare_chunk_indices` (every call, every layer). So the
    call can be captured in a CUDA graph once its kernels are compiled, and one
    capture serves any split of the same token count into the same number of
    sequences; the test replays one. Prefill only.

    One config per stage, no autotuning: every launch config is fixed, chosen by
    timing at this op's shapes, so two processes compute the same bits (the test
    runs two). `_last_step_o_kernel` computes only the rows the mixer keeps, a
    token's last sub-step, with the vendored output kernel `chunk_fwd_kernel_o`'s
    arithmetic.

    Certified head counts: 24 (tp1) and 6 (a tp4 rank). Uncertified: any other
    head count, head dims or Householder count, a bf16 state pool, query/key
    heads fewer than value heads, fp16 or fp32 q/k/v, MTP verify (several
    tokens per request with per-position states).
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
        cu_seqlens: torch.Tensor,
        has_initial_state: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Writes the sequences' final states into `state_pool[slots]` and returns `out`."""
        return _gdp_chunk_fwd(
            q, k, v, g, beta, state_pool, slots, cu_seqlens, has_initial_state, scale
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        slots: torch.Tensor,
        cu_seqlens: torch.Tensor,
        has_initial_state: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, the pool as the op leaves it)`, fp32 throughout; mutates nothing.

        Transcribed from `torch_recurrent_gated_delta_rule` with
        `use_qk_l2norm_in_kernel=True`, token by token over each sequence, the
        token's steps as `Nemotron4GDPMixer.forward` expands them: the decay rides
        on the first step only (the others' `g` is zero, and `exp(0)` is exactly
        one) and the query is read only after the last (the others' query is
        zero and their output is dropped). HF divides the l2-normalized query by
        `sqrt(key_dim)`; `scale` is that factor. HF starts every sequence from
        zero; a sequence with `has_initial_state` starts from its slot's state,
        as the cached path resumes a chunked prefill.
        """
        query = _l2norm(q.float()) * scale
        key = _l2norm(k.float())
        value, step_beta, decay = v.float(), beta.float(), g.float().exp()
        out = torch.empty(v.shape[0], v.shape[2], v.shape[3], dtype=torch.float32, device=v.device)
        after = state_pool.clone()
        bounds = cu_seqlens.tolist()
        for n, (slot, resume) in enumerate(zip(slots.tolist(), has_initial_state.tolist())):
            state = state_pool[slot].float().transpose(-1, -2)  # [H, K, V]
            if not resume:
                state = torch.zeros_like(state)
            for t in range(bounds[n], bounds[n + 1]):
                state = state * decay[t, :, None, None]
                for j in range(v.shape[1]):
                    kv_mem = (state * key[t, j, :, :, None]).sum(dim=-2)
                    delta = (value[t, j] - kv_mem) * step_beta[t, j, :, None]
                    state = state + key[t, j, :, :, None] * delta[:, None, :]
                out[t] = (state * query[t, :, :, None]).sum(dim=-2)
            after[slot] = state.transpose(-1, -2).to(state_pool.dtype)
        return out.to(q.dtype), after

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        slots: torch.Tensor,
        cu_seqlens: torch.Tensor,
        has_initial_state: torch.Tensor,
        scale: float,
    ) -> None:
        # The kernels take their sizes from `v` and `k` and read q, k and g
        # through raw strides, so a tensor of the wrong shape is read as
        # whatever those sizes say rather than rejected.
        num_tokens, num_householder, num_heads, value_dim = v.shape
        key_dim = k.shape[-1]
        assert k.shape == (num_tokens, num_householder, num_heads, key_dim), (
            f"k must be [T, n_h, H, K] = {(num_tokens, num_householder, num_heads, key_dim)}; "
            f"got {tuple(k.shape)}"
        )
        assert q.shape == (num_tokens, num_heads, key_dim), (
            f"q must be [T, H, K] = {(num_tokens, num_heads, key_dim)}, one query per token; "
            f"got {tuple(q.shape)}"
        )
        assert g.shape == (num_tokens, num_heads), (
            f"g must be [T, H] = {(num_tokens, num_heads)}, one decay per token; "
            f"got {tuple(g.shape)}"
        )
        assert beta.shape == (num_tokens, num_householder, num_heads), (
            f"beta must be [T, n_h, H] = {(num_tokens, num_householder, num_heads)}; "
            f"got {tuple(beta.shape)}"
        )
        # The chunk kernels multiply bf16 tiles; fp32 inputs reach them as TF32
        # products and fp16 ones as another rounding, neither of them measured.
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16, (
            f"q, k and v must be bfloat16; got {q.dtype}, {k.dtype}, {v.dtype}"
        )
        # HF's [H, K, V] state has the same element count; read through these
        # strides it comes back transposed.
        assert tuple(state_pool.shape[1:]) == (num_heads, value_dim, key_dim), (
            f"state_pool must be [num_slots, H, V, K] = [*, {num_heads}, {value_dim}, {key_dim}] "
            f"with K innermost; got {tuple(state_pool.shape)}"
        )
        # A bf16 pool would round the recurrent state at every prefill chunk;
        # HF keeps it in fp32.
        assert state_pool.dtype == torch.float32, (
            f"state_pool holds the recurrent state in float32; got {state_pool.dtype}"
        )
        # HF computes the decay in fp32; exp() of a bf16-rounded decay is off by
        # up to 2^-9 relative per token, and the state compounds it.
        assert g.dtype == torch.float32, f"g (the log decay) must be float32; got {g.dtype}"
        # A positive g grows the state instead of decaying it, and the chunk
        # kernels' safe_exp silently zeroes the pairs it would weight above one.
        assert bool((g <= 0).all()), (
            "g must be the log decay -exp(A_log) * softplus(a + dt_bias) <= 0"
        )
        assert bool(((beta >= 0) & (beta <= 1)).all()), (
            "beta must be post-sigmoid, in [0, 1]: pass sigmoid(b), not b"
        )
        num_seqs = slots.shape[0]
        assert cu_seqlens.shape == (num_seqs + 1,) and not cu_seqlens.is_floating_point(), (
            f"cu_seqlens must be {num_seqs + 1} integer offsets, one more than slots; "
            f"got {cu_seqlens.dtype} {tuple(cu_seqlens.shape)}"
        )
        bounds = cu_seqlens.tolist()
        assert bounds[0] == 0 and bounds[-1] == num_tokens, (
            f"cu_seqlens must run from 0 to the token count {num_tokens}; got {bounds}"
        )
        assert all(a <= b for a, b in zip(bounds, bounds[1:])), (
            f"cu_seqlens must be non-decreasing; got {bounds}"
        )
        assert has_initial_state.shape == (num_seqs,) and has_initial_state.dtype == torch.bool, (
            f"has_initial_state must be one bool per sequence; got {has_initial_state.dtype} "
            f"{tuple(has_initial_state.shape)}"
        )
        # A negative slot indexes from the end of the pool: PAD_SLOT_ID would
        # read and overwrite the last slot rather than be skipped.
        assert slots.shape == (num_seqs,) and not slots.is_floating_point(), (
            f"slots must be one integer per sequence; got {slots.dtype} {tuple(slots.shape)}"
        )
        assert bool(((slots >= 0) & (slots < state_pool.shape[0])).all()), (
            f"slots must be in [0, {state_pool.shape[0]}): prefill rows have no padding slot"
        )
        assert slots.unique().numel() == num_seqs, (
            "two sequences share a slot; the last one written wins"
        )


gdp_chunk = GdpChunk()
