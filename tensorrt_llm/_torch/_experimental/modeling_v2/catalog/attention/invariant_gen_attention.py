# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged generation attention (decode and MTP verify) whose rows do not depend on the batch."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Keys one program attends over. Fixed: a split never follows the batch size,
#: the tokens per request or the request's own length, so a query token's keys
#: always fall into the same splits and are reduced in the same order.
_SPLIT = 512
#: Keys per tile: two 32-token pages.
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 3
_LOG2E = 1.4426950408889634

#: nemotron4-nano-preview's attention layers (8 trunk layers and the MTP
#: layer): 16 query heads over one KV head of 256, NoPE, q_scaling 1 (scale
#: 1/16), a bf16 pool of 32-token pages, max_seq_len 9344 as the target runs.
_NEMOTRON4 = dict(
    num_heads=16,
    num_kv_heads=1,
    head_size=256,
    tokens_per_block=32,
    max_seq_len=9344,
    q_scaling=1.0,
)
#: The same layers on one rank of the tp4 target: 4 of the 16 query heads, over
#: the one KV head, which every rank keeps whole.
_NEMOTRON4_TP4 = dict(_NEMOTRON4, num_heads=4)

#: Every (B, padding rows) a cell runs, and what it stands for.
_BATCHES = (
    (1, 0, "one request, history max_seq_len"),
    (2, 1, "graph batch, one request and one padding row"),
    (3, 0, "eager batch, mixed histories"),
    (4, 2, "graph batch with two padding rows"),
    (8, 2, "graph batch with two padding rows"),
    (64, 2, "graph batch with two padding rows"),
    (256, 2, "graph batch with two padding rows"),
    (512, 2, "graph batch with two padding rows"),
    (1024, 2, "graph batch with two padding rows"),
)


def _cells(geometry: dict, label: str, largest: str) -> tuple[Cell, ...]:
    """Every T x B cell at one head geometry; `largest` says what the largest batch is there."""
    return tuple(
        Cell(
            why=f"{label}T={tokens} B={batch}: {kind}{f' ({largest})' if batch == 1024 else ''}",
            spec=dict(tokens=tokens, batch=batch, pad_rows=pad_rows, **geometry),
        )
        for tokens in (1, 2, 3)
        for batch, pad_rows, kind in _BATCHES
    )


class _DevicePtr:
    """A raw device address as a Triton pointer argument.

    Triton takes any object with `data_ptr()` and `dtype`. The KV pool is
    reachable only as the address `host_kv_cache_pool_pointers` holds, the
    way the attention op itself reaches it.
    """

    def __init__(self, address: int, dtype: torch.dtype) -> None:
        self._address = address
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self._address


@triton.jit(do_not_specialize=["tokens_per_seq", "stride_qkv_row", "stride_bo_seq", "stride_bo_kv"])
def _append_kv_kernel(
    qkv,
    pool,
    block_offsets,
    seq_lens,
    tokens_per_seq,
    stride_qkv_row,
    stride_bo_seq,
    stride_bo_kv,
    Q_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    D: tl.constexpr,
    TPB: tl.constexpr,
):
    """One program per new token: its K and V rows to their slot in the pool.

    Token j of a request of length L (this call's tokens included) sits at
    position L - T + j; a page holds `[kv heads, TPB, D]` and the page's offset
    counts pages of that size from the layer's base, as `KVBlockArray` reads it.
    """
    i_tok = tl.program_id(0)
    i_seq = i_tok // tokens_per_seq
    pos = tl.load(seq_lens + i_seq) - tokens_per_seq + i_tok % tokens_per_seq
    if pos >= 0:
        row = block_offsets + i_seq.to(tl.int64) * stride_bo_seq + pos // TPB
        page_k = tl.load(row).to(tl.int64) * (KV_WIDTH * TPB)
        page_v = tl.load(row + stride_bo_kv).to(tl.int64) * (KV_WIDTH * TPB)
        o = tl.arange(0, KV_WIDTH)
        in_page = (o // D) * (TPB * D) + (pos % TPB) * D + o % D
        src = qkv + i_tok.to(tl.int64) * stride_qkv_row + Q_WIDTH + o
        tl.store(pool + page_k + in_page, tl.load(src))
        tl.store(pool + page_v + in_page, tl.load(src + KV_WIDTH))


@triton.jit(do_not_specialize=["tokens_per_seq", "stride_qkv_row", "stride_bo_seq", "stride_bo_kv"])
def _split_kernel(
    qkv,
    pool,
    block_offsets,
    seq_lens,
    part_acc,
    part_m,
    part_l,
    qk_scale,
    tokens_per_seq,
    stride_qkv_row,
    stride_bo_seq,
    stride_bo_kv,
    stride_acc_tok,
    stride_acc_split,
    stride_ml_tok,
    stride_ml_split,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    TPB: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (query token, split): the token's heads against the
    split's keys, one KV head. Writes the unnormalized partial (row max in log2
    units, row sum, accumulator), all fp32.

    Token j of a request of length L sees keys 0 .. L - T + j: its program is
    the one a one-token call over L - T + j + 1 keys runs, same code, same
    tiles, same masks. A split past the token's keys does nothing, so the grid
    can be sized once from max_seq_len.
    """
    i_seq = tl.program_id(2)
    i_split = tl.program_id(1)
    i_tok = i_seq * tokens_per_seq + tl.program_id(0)
    visible = tl.load(seq_lens + i_seq) - tokens_per_seq + tl.program_id(0) + 1
    start = i_split * SPLIT
    if start < visible:
        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, D)
        head_ok = offs_h < H
        q_row = qkv + i_tok.to(tl.int64) * stride_qkv_row
        q = tl.load(q_row + offs_h[:, None] * D + offs_d[None, :], mask=head_ok[:, None], other=0.0)
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, D], tl.float32)
        pages = block_offsets + i_seq.to(tl.int64) * stride_bo_seq
        for n0 in range(start, tl.minimum(start + SPLIT, visible), BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            key_ok = offs_n < visible
            in_page = (offs_n % TPB) * D
            page_k = tl.load(pages + offs_n // TPB, mask=key_ok, other=0).to(tl.int64)
            page_v = tl.load(pages + stride_bo_kv + offs_n // TPB, mask=key_ok, other=0).to(
                tl.int64
            )
            k = tl.load(
                pool + (page_k * (TPB * D) + in_page)[:, None] + offs_d[None, :],
                mask=key_ok[:, None],
                other=0.0,
            )
            v = tl.load(
                pool + (page_v * (TPB * D) + in_page)[:, None] + offs_d[None, :],
                mask=key_ok[:, None],
                other=0.0,
            )
            # Every tile holds at least one key the token sees, so m_new is finite.
            s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(key_ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
            m_i = m_new
        rows = i_tok.to(tl.int64) * stride_acc_tok + i_split * stride_acc_split + offs_h * D
        tl.store(part_acc + rows[:, None] + offs_d[None, :], acc, mask=head_ok[:, None])
        ml = i_tok.to(tl.int64) * stride_ml_tok + i_split * stride_ml_split + offs_h
        tl.store(part_m + ml, m_i, mask=head_ok)
        tl.store(part_l + ml, l_i, mask=head_ok)


@triton.jit(do_not_specialize=["tokens_per_seq", "stride_out_row"])
def _combine_kernel(
    part_acc,
    part_m,
    part_l,
    seq_lens,
    out,
    tokens_per_seq,
    stride_out_row,
    stride_acc_tok,
    stride_acc_split,
    stride_ml_tok,
    stride_ml_split,
    D: tl.constexpr,
    SPLIT: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    """One program per (head, query token): the token's splits merged in one
    fixed-order reduction over a split axis sized from max_seq_len, the splits
    holding none of its keys masked to zero."""
    i_head = tl.program_id(0)
    i_tok = tl.program_id(1)
    i_seq = i_tok // tokens_per_seq
    visible = tl.load(seq_lens + i_seq) - tokens_per_seq + i_tok % tokens_per_seq + 1
    offs_s = tl.arange(0, MAX_SPLITS)
    offs_d = tl.arange(0, D)
    used = offs_s < tl.cdiv(tl.maximum(visible, 0), SPLIT)
    ml = i_tok.to(tl.int64) * stride_ml_tok + offs_s * stride_ml_split + i_head
    m = tl.load(part_m + ml, mask=used, other=float("-inf"))
    w = tl.where(used, tl.exp2(m - tl.max(m, 0)), 0.0)
    den = tl.sum(tl.load(part_l + ml, mask=used, other=0.0) * w, 0)
    rows = i_tok.to(tl.int64) * stride_acc_tok + offs_s * stride_acc_split + i_head * D
    acc = tl.load(part_acc + rows[:, None] + offs_d[None, :], mask=used[:, None], other=0.0)
    num = tl.sum(acc * w[:, None], 0)
    o = tl.where(den > 0.0, num / den, 0.0)
    dst = out + i_tok.to(tl.int64) * stride_out_row + i_head * D + offs_d
    tl.store(dst, o.to(out.dtype.element_ty))


class InvariantGenAttention(OpWrapper):
    """Generation attention for T tokens per request over the paged KV pool,
    appending the call's K/V first, with every query token's output a function
    of that token's query and keys only.

    For request b of length L (this call's T tokens included) and its token j:

        pool[L - T + j] = (k_j, v_j)                        # the append
        out[b, j, h] = softmax(q_jh . K[0 .. L-T+j]^T * scale) V[0 .. L-T+j]

    one KV head shared by all query heads, `scale = 1 / (q_scaling *
    sqrt(head_size))`, no positional encoding: causal within the request's
    group of T, which is decode at T = 1 and MTP verify at T > 1.

    Writes rows `[:B*T]` of `output` and returns it, the same buffer, as
    `thop_attention` does.
    """

    ARCHS = frozenset({Arch.SM_103})

    # T is decode (1) and MTP verify (1 + max_draft_len: 3 at the trained
    # depth, 2 under a one-draft schedule). B is every decode CUDA-graph batch
    # the target captures that differs in kind -- 1, 2, 4, 8 at the latency
    # end, 64, 256, 512, 1024 where the kernel is bandwidth-bound -- plus the
    # eager size 3. Every batch mixes history lengths up to max_seq_len; graph
    # batches end in padding rows that share one dummy request's pages, as the
    # CUDA-graph runner pads. The heads are the tp1 and dep4 targets' 16 and a
    # tp4 rank's 4, whose program fills 4 of its 16 head rows.
    CELLS: tuple[Cell, ...] = _cells(_NEMOTRON4, "", "a DP rank") + _cells(
        _NEMOTRON4_TP4, "tp4 rank, 4 heads: ", "the largest graph batch"
    )

    note = """
    Why it exists: `thop_attention`'s generation kernels change with T and
    with B (a multi-CTA KV split that follows the batch, and different
    trtllm-gen variants for decode and verify), so a token's attention output
    differs in its last bits between a padded CUDA-graph batch, an eager batch
    and position j of an MTP verify group. Here one program computes one query
    token's heads (16 rows: all 16 heads on tp1, a tp4 rank's 4 and 12 masked
    rows) over one fixed 512-key split (64-key tiles, fp32 online softmax), and
    one program per (head, token) merges the token's splits in one fixed-order
    reduction. T rides as a runtime argument, so one compiled kernel serves
    every call: a token gets the same bits alone, inside any batch, and as
    token j of a T-token group as the same token decoded alone over the same
    keys -- that program is literally the decode program.

    Cost of that choice: the T tokens of a verify group each read the
    request's pages (through L2), so T = 3 verify takes 2.15-2.3x the time of
    `thop_attention`'s verify kernels from B 64 up, while decode takes
    1.09-1.2x of its decode kernels there. One program holding all T tokens
    reads the pages once but spends its registers on T 16x256 fp32
    accumulators: measured 2-4% faster at B 256-512 and slower below it. The
    64-row tensor-core tile that would hold them (with or without warp
    specialization, TMA page loads or a two-pass softmax) measured about
    3 TB/s at both T.

    Context (prefill) rows stay on `thop_attention`. The KV append writes the
    bytes `thop_attention` writes for generation rows (bf16 K/V copied into
    `[kv heads, tokens_per_block, head_size]` pages through the same block
    offsets and pool pointers), so the two share one pool.

    Batch state is read the way `thop_attention` reads it: lengths from the
    device tensor `sequence_length` only (the overlap scheduler corrects them
    on device), pages from `kv_cache_block_offsets[pool, num_contexts + b]`,
    the layer's base from `host_kv_cache_pool_pointers` and
    `host_kv_cache_pool_mapping[local_layer_idx]`. Keys past a token's length
    are never read, so stale or NaN bytes in a last page's tail cannot reach
    the output. The grid is sized from `max_seq_len` and the call allocates
    fp32 partials of `B*T x cdiv(max_seq_len, 512) x heads x head_size` --
    about 320 MB at B 1024 / T 1 and 480 MB at B 512 / T 3 for max_seq_len
    9344 -- so `max_seq_len` must bound every length the call sees. CUDA-graph capturable: no host sync,
    allocations only.

    CUDA-graph padding rows share the dummy request's pages and race on its
    append slot; their outputs are meaningless and real rows never read them.

    Uncertified: more than one KV head, more than 16 query heads, an fp8 or
    fp4 pool, sliding windows, attention sinks, any positional encoding, tree
    masks, T above 3.
    """

    def __call__(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        sequence_length: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        local_layer_idx: int,
        num_contexts: int,
        tokens_per_seq: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: int,
        max_seq_len: int,
        q_scaling: float,
    ) -> torch.Tensor:
        """Appends the call's K/V, writes `output[:B*T]` in place and returns it.

        `q` is the fused qkv `[B*T, (num_heads + 2*num_kv_heads) * head_size]`
        of this call's generation tokens only, request-major. `sequence_length`
        and `kv_cache_block_offsets` are the step's whole tensors, indexed from
        `num_contexts` (the first generation request), as `thop_attention`
        indexes them.
        """
        num_tokens = q.shape[0]
        num_seqs = num_tokens // tokens_per_seq
        if num_seqs == 0:
            return output
        pool_index = int(host_kv_cache_pool_mapping[local_layer_idx, 0])
        layer_in_pool = int(host_kv_cache_pool_mapping[local_layer_idx, 1])
        page_bytes = tokens_per_block * num_kv_heads * head_size * q.element_size()
        pool = _DevicePtr(
            int(host_kv_cache_pool_pointers[pool_index, 0]) + layer_in_pool * 2 * page_bytes,
            q.dtype,
        )
        block_offsets = kv_cache_block_offsets[pool_index, num_contexts : num_contexts + num_seqs]
        seq_lens = sequence_length[num_contexts : num_contexts + num_seqs]
        num_splits = triton.cdiv(max_seq_len, _SPLIT)
        part_acc = torch.empty(
            (num_tokens, num_splits, num_heads, head_size), dtype=torch.float32, device=q.device
        )
        part_m = torch.empty(
            (num_tokens, num_splits, num_heads), dtype=torch.float32, device=q.device
        )
        part_l = torch.empty_like(part_m)

        _append_kv_kernel[(num_tokens,)](
            q,
            pool,
            block_offsets,
            seq_lens,
            tokens_per_seq,
            q.stride(0),
            block_offsets.stride(0),
            block_offsets.stride(1),
            Q_WIDTH=num_heads * head_size,
            KV_WIDTH=num_kv_heads * head_size,
            D=head_size,
            TPB=tokens_per_block,
            num_warps=1,
        )
        _split_kernel[(tokens_per_seq, num_splits, num_seqs)](
            q,
            pool,
            block_offsets,
            seq_lens,
            part_acc,
            part_m,
            part_l,
            _LOG2E / (q_scaling * math.sqrt(head_size)),
            tokens_per_seq,
            q.stride(0),
            block_offsets.stride(0),
            block_offsets.stride(1),
            part_acc.stride(0),
            part_acc.stride(1),
            part_m.stride(0),
            part_m.stride(1),
            H=num_heads,
            BLOCK_H=max(16, triton.next_power_of_2(num_heads)),
            D=head_size,
            TPB=tokens_per_block,
            SPLIT=_SPLIT,
            BLOCK_N=_BLOCK_N,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        _combine_kernel[(num_heads, num_tokens)](
            part_acc,
            part_m,
            part_l,
            seq_lens,
            output,
            tokens_per_seq,
            output.stride(0),
            part_acc.stride(0),
            part_acc.stride(1),
            part_m.stride(0),
            part_m.stride(1),
            D=head_size,
            SPLIT=_SPLIT,
            MAX_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=4,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_lens: Sequence[int],
        tokens_per_seq: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        q_scaling: float,
    ) -> torch.Tensor:
        """Masked softmax attention in fp32 (TF32 off), rounded once to q's dtype.

        The paged-cache exception to mirroring `__call__`, as for
        `thop_attention.reference`: the output is a function of what the pool
        holds, so `k` and `v` are each request's whole history
        `[sum(kv_lens), num_kv_heads * head_size]`, request-major, this call's
        tokens included -- gathered by the test from the pool. Only the query
        columns of the fused `q` are read. Token j of a request of length L
        sees keys 0 .. L - T + j.
        """
        width = num_heads * head_size
        group = num_heads // num_kv_heads
        scale = 1.0 / (q_scaling * math.sqrt(head_size))
        out = torch.empty(q.shape[0], width, dtype=torch.float32, device=q.device)
        start = 0
        with true_fp32_matmul():
            for i, kv_len in enumerate(kv_lens):
                tokens = slice(i * tokens_per_seq, (i + 1) * tokens_per_seq)
                qi = q[tokens, :width].float().view(tokens_per_seq, num_kv_heads, group, head_size)
                ki = k[start : start + kv_len].float().view(kv_len, num_kv_heads, head_size)
                vi = v[start : start + kv_len].float().view(kv_len, num_kv_heads, head_size)
                scores = torch.einsum("tgrd,kgd->grtk", qi, ki) * scale
                last = kv_len - tokens_per_seq + torch.arange(tokens_per_seq, device=q.device)
                keys = torch.arange(kv_len, device=q.device)
                scores = scores.masked_fill(keys[None, :] > last[:, None], float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[tokens] = torch.einsum("grtk,kgd->tgrd", probs, vi).reshape(tokens_per_seq, -1)
                start += kv_len
        return out.to(q.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        sequence_length: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        local_layer_idx: int,
        num_contexts: int,
        tokens_per_seq: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: int,
        max_seq_len: int,
        q_scaling: float,
    ) -> None:
        # One program holds every query head of a token against one KV head: a
        # second KV head would be read as the first's, and heads past the
        # 16-row tile would never be computed.
        assert num_kv_heads == 1, (
            f"num_kv_heads must be 1 (one KV head per request); got {num_kv_heads}"
        )
        assert num_heads <= 16, f"num_heads must fit the 16-row tile; got {num_heads}"
        # The kernels read q, the pool and output through raw strides at the
        # widths these sizes imply, so a tensor of another shape -- the q
        # slice instead of the fused qkv, say -- is read rather than rejected.
        assert q.dim() == 2 and q.shape[0] % tokens_per_seq == 0, (
            f"q must be [B*T, ...] with T = tokens_per_seq = {tokens_per_seq}; got {tuple(q.shape)}"
        )
        num_seqs = q.shape[0] // tokens_per_seq
        qkv_width = (num_heads + 2 * num_kv_heads) * head_size
        assert q.shape[1] == qkv_width and q.stride(1) == 1, (
            f"q must be the fused qkv [B*T, {qkv_width}] with unit column stride; got "
            f"{tuple(q.shape)} strides {q.stride()}"
        )
        assert output.shape[0] >= q.shape[0] and output.shape[1] == num_heads * head_size, (
            f"output must be [>= {q.shape[0]}, {num_heads * head_size}]; got {tuple(output.shape)}"
        )
        assert output.stride(1) == 1, f"output must have unit column stride; got {output.stride()}"
        # The pool is read and written as q's dtype; the append copies q's K/V bytes.
        assert q.dtype == torch.bfloat16 and output.dtype == q.dtype, (
            f"q, output and the pool are bfloat16; got q {q.dtype}, output {output.dtype}"
        )
        # Lengths: each token needs itself among its keys, and the grid covers
        # max_seq_len keys only -- a longer history loses its tail.
        lens = sequence_length[num_contexts : num_contexts + num_seqs]
        assert lens.numel() == num_seqs, (
            f"sequence_length holds {sequence_length.numel()} lengths; the call needs "
            f"num_contexts + B = {num_contexts + num_seqs}"
        )
        assert bool((lens >= tokens_per_seq).all()), (
            f"every sequence_length must be >= tokens_per_seq ({tokens_per_seq}): it counts "
            "this call's tokens"
        )
        assert bool((lens <= max_seq_len).all()), (
            f"a sequence_length exceeds max_seq_len ({max_seq_len}); the grid covers "
            "max_seq_len keys and would drop the rest"
        )
        # The page table is read flat: a request needing more pages than a
        # row holds reads the next request's offsets.
        max_pages = triton.cdiv(int(lens.max()), tokens_per_block)
        assert max_pages <= kv_cache_block_offsets.shape[-1], (
            f"a request needs {max_pages} pages; kv_cache_block_offsets rows hold "
            f"{kv_cache_block_offsets.shape[-1]}"
        )
        assert kv_cache_block_offsets.shape[1] >= num_contexts + num_seqs, (
            f"kv_cache_block_offsets holds {kv_cache_block_offsets.shape[1]} sequences; the call "
            f"needs num_contexts + B = {num_contexts + num_seqs}"
        )
        assert kv_cache_block_offsets.stride(-1) == 1, (
            "kv_cache_block_offsets must have unit stride over pages"
        )


invariant_gen_attention = InvariantGenAttention()
