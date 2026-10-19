# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent MoE's routed experts: soft-clamped relu², routing weight applied before down_proj."""

from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

# Every MoE block of the checkpoint (`Nemotron4Experts`, 28 shortcut layers plus the MTP layer)
# has one geometry: 512 routed experts, top-10, experts working in the 768-wide latent space
# (`moe_latent_size`) with intermediate 4096 (`moe_intermediate_size`), and the activation
# 16 * tanh(x / 16) before relu² (`activation_clamp_value`, `mlp_hidden_act` relu2).
_LATENT_MOE = dict(num_experts=512, top_k=10, latent=768, intermediate=4096, activation_clamp=16.0)


def _cell(
    why: str,
    tokens: int,
    offset: int = 0,
    count: int = 512,
    out_dtype: torch.dtype | None = None,
) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            tokens=tokens,
            local_expert_offset=offset,
            num_local_experts=count,
            out_dtype=out_dtype,
            **_LATENT_MOE,
        ),
    )


# An expert-parallel rank over 4 ranks owns the window (128 r, 128) and runs it over every
# rank's tokens, all-gathered: T_all is the 4 ranks' tokens together. What each gathered size
# puts on the window's 128 experts, at 10 slots per token of which a quarter land in a window.
_EP_LOADS = {
    4: "one decode token per rank, ~10 slots in the window, one row per expert touched",
    64: "16 decode tokens per rank, ~160 slots, about one row per window expert",
    1024: "256 decode tokens per rank: every window expert touched, ~20 rows each",
    8192: "a prefill chunk's worth over the ranks: ~160 rows per window expert",
    32768: "four 8192-token prefill chunks: ~640 rows per expert, twice tp1's largest",
}

# The ranks whose windows touch the ends of the expert range: rank 0's starts at expert 0, rank
# 3's ends at the last one. Ranks 1 and 2 differ from them only in the offset value, which
# the (128, 128) cell above already drives.
_EP_RANKS = (0, 3)

# bf16 is the partial sum rounded as a bf16 reduce-scatter would take it; fp32 is the same sum
# unrounded, for ranks that reduce their partials before the one rounding.
_EP_OUT_DTYPES = {torch.bfloat16: "bf16", torch.float32: "fp32"}


def _ep_cells() -> tuple[Cell, ...]:
    return tuple(
        _cell(
            f"dep4 rank {rank}, {tokens} gathered tokens, {name} out: experts "
            f"{128 * rank}..{128 * rank + 127}; {load}",
            tokens,
            offset=128 * rank,
            count=128,
            out_dtype=out_dtype,
        )
        for rank in _EP_RANKS
        for tokens, load in _EP_LOADS.items()
        for out_dtype, name in _EP_OUT_DTYPES.items()
    )


# The fused path's tiles by batch size: (most tokens, BLOCK_M, up tile, down tile), each GEMM's
# tile (BLOCK_N, BLOCK_K, num_warps, num_stages). Up to BLOCK_M rows of one expert are one unit of
# work for both GEMMs. A batch of more tokens than the last row takes the grouped-GEMM path,
# which is the faster one there once routing concentrates rows on fewer experts. Each row's
# tiles were measured the fastest, within a few percent, over uniform and concentrated routing
# at the batch sizes it covers. A tile decides how the work is split, not the order an output
# element accumulates in, so every row gives the grouped-GEMM path's bits (tested).
_FUSED_TILES = (
    (8, 16, (32, 128, 4, 3), (64, 128, 4, 4)),
    (128, 16, (64, 128, 4, 3), (64, 128, 4, 5)),
    (512, 16, (128, 128, 8, 3), (128, 128, 4, 4)),
    (1024, 16, (128, 64, 4, 3), (128, 128, 4, 3)),
)


def _fp32_reciprocal(clamp: float) -> float:
    """fp32(1 / fp32(clamp)), the factor torch multiplies by when it divides a CUDA tensor by a
    Python scalar -- which is how the checkpoint computes h / clamp. For a power of two such as
    the checkpoint's 16 the product is the quotient itself."""
    c = struct.unpack("f", struct.pack("f", clamp))[0]
    return struct.unpack("f", struct.pack("f", 1.0 / c))[0]


@triton.jit
def _clamped_relu2_weighted(h, w, CLAMP: tl.constexpr, INV_CLAMP: tl.constexpr):
    """bf16(w * bf16(relu(bf16(c*tanh(h/c)))^2)), for a bf16 `h` held in fp32 and an fp32 `w`.

    Each rounding is the checkpoint's: the clamp is computed in fp32 from the bf16 GEMM output
    and rounded, relu² runs on bf16 tensors (one fp32 product, rounded), and the bf16 result times
    the fp32 routing weight is an fp32 product rounded once. h/c is h times INV_CLAMP, c's fp32
    reciprocal (`_fp32_reciprocal`), as torch computes the checkpoint's division by a scalar.
    libdevice's tanh is the same tanhf torch's CUDA kernel calls.

    The relu is taken first, on h: tanh keeps the sign and maps 0 to 0, so the clamp of relu(h)
    is the relu of the clamp, for every input. A max that propagates NaN keeps a NaN a NaN, as
    torch.relu does. Both reorderings -- the reciprocal for the division, the relu first -- are
    measured faster and pinned bitwise on every bf16 input by the entry's test.
    """
    r = tl.maximum(h, 0.0, propagate_nan=tl.PropagateNan.ALL)
    c = (CLAMP * libdevice.tanh(r * INV_CLAMP)).to(tl.bfloat16).to(tl.float32)
    s = (c * c).to(tl.bfloat16).to(tl.float32)
    return (s * w).to(tl.bfloat16)


@triton.jit
def _clamped_relu2_weighted_kernel(
    h_ptr,
    w_ptr,
    order_ptr,
    n_valid_ptr,
    stride_h,
    N: tl.constexpr,
    CLAMP: tl.constexpr,
    INV_CLAMP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BLOCK_N columns of one row of the up-projection output, in place:
    `_clamped_relu2_weighted` with the row's routing weight. The grouped-GEMM path's activation
    pass; a row at or past `n_valid` is left as it is."""
    row = tl.program_id(0)
    if row < tl.load(n_valid_ptr):
        w = tl.load(w_ptr + tl.load(order_ptr + row)).to(tl.float32)
        offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        ptrs = h_ptr + row.to(tl.int64) * stride_h + offs
        h = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, _clamped_relu2_weighted(h, w, CLAMP, INV_CLAMP), mask=mask)


def _activate_rows(
    h: torch.Tensor,
    topk_weights: torch.Tensor,
    order: torch.Tensor,
    n_valid: torch.Tensor,
    activation_clamp: float,
) -> None:
    """The grouped-GEMM path's activation pass over `h` [rows, N], in place: row r is slot
    `order[r]`'s, weighted by `topk_weights` read flat at that slot; rows from `n_valid` (a
    one-element device tensor) on are skipped. One program per row and up to 4096 columns, four
    warps -- 32 elements a thread at the checkpoint's 4096 -- the fastest configuration measured
    on the target's own prefill inputs."""
    n = h.shape[1]
    block_n = min(4096, triton.next_power_of_2(n))
    _clamped_relu2_weighted_kernel[(h.shape[0], triton.cdiv(n, block_n))](
        h,
        topk_weights,
        order,
        n_valid,
        h.stride(0),
        N=n,
        CLAMP=float(activation_clamp),
        INV_CLAMP=_fp32_reciprocal(activation_clamp),
        BLOCK_N=block_n,
        num_warps=4,
    )


@triton.jit
def _sort_key(ids_ptr, slot, n_slots, local_offset, num_local):
    """Each slot's key: its local expert, `num_local` for an expert outside the window, and
    `num_local + 1` past the last slot, where no program places anything."""
    valid = slot < n_slots
    local = tl.load(ids_ptr + slot, mask=valid, other=0).to(tl.int32) - local_offset
    key = tl.where((local >= 0) & (local < num_local), local, num_local)
    return tl.where(valid, key, num_local + 1)


@triton.jit
def _count_sort_kernel(
    ids_ptr,
    order_ptr,
    inv_ptr,
    offs_ptr,
    n_slots,
    local_offset,
    num_local,
    BLOCK: tl.constexpr,
):
    """`_sort_slots`' layout in one kernel: program p places the slots whose key is p.

    It counts the slots with a smaller key -- the rows before p's -- then walks the slots in
    order and gives each of p's the next row, so within an expert the rows keep slot order, as
    the stable sort leaves them. The last program places the slots outside the window, after
    every local expert's rows. Every program reads every id twice: at decode sizes, a few tens
    of KB from L2.
    """
    p = tl.program_id(0)
    row = tl.full((), 0, tl.int32)
    for b0 in range(0, n_slots, BLOCK):
        key = _sort_key(ids_ptr, b0 + tl.arange(0, BLOCK), n_slots, local_offset, num_local)
        row += tl.sum((key < p).to(tl.int32), axis=0)
    for b0 in range(0, n_slots, BLOCK):
        slot = b0 + tl.arange(0, BLOCK)
        hit = (_sort_key(ids_ptr, slot, n_slots, local_offset, num_local) == p).to(tl.int32)
        dest = row + tl.cumsum(hit, axis=0) - 1
        tl.store(order_ptr + dest, slot, mask=hit != 0)
        tl.store(inv_ptr + slot, dest, mask=hit != 0)
        row += tl.sum(hit, axis=0)
    if p < num_local:
        tl.store(offs_ptr + p, row)


@triton.jit
def _tile_map_kernel(offs_ptr, tiles_ptr, num_local, BLOCK_M: tl.constexpr, NE: tl.constexpr):
    """The fused GEMMs' work list: an entry per BLOCK_M rows of one local expert, in expert
    order, and none for an expert no slot routes to.

    `tiles[0]` is the number of entries; entry t, `tiles[1 + 3t : 4 + 3t]`, is (expert, first
    row, end row). Each entry is one program's rows, so a much-routed expert's rows spread over
    as many programs as they fill rather than queueing in one, and the programs past the last
    entry -- the launch is sized for the most entries a batch can need -- return at once.
    """
    e = tl.arange(0, NE)
    local = e < num_local
    end = tl.load(offs_ptr + e, mask=local, other=0)
    start = tl.load(offs_ptr + e - 1, mask=local & (e > 0), other=0)
    count = tl.where(local, (end - start + BLOCK_M - 1) // BLOCK_M, 0)
    first = tl.cumsum(count, axis=0) - count
    tl.store(tiles_ptr, tl.sum(count, axis=0))
    for j in range(0, tl.max(count, axis=0)):
        has = j < count
        row = start + j * BLOCK_M
        entry = tiles_ptr + 1 + 3 * (first + j)
        tl.store(entry, e, mask=has)
        tl.store(entry + 1, row, mask=has)
        tl.store(entry + 2, tl.minimum(row + BLOCK_M, end), mask=has)


@triton.jit
def _up_act_kernel(
    x_ptr,
    up_ptr,
    w_ptr,
    order_ptr,
    tiles_ptr,
    h_ptr,
    stride_xt,
    stride_xk,
    stride_ue,
    stride_un,
    stride_uk,
    stride_h,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    CLAMP: tl.constexpr,
    INV_CLAMP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One work-list entry's rows times one BLOCK_N slice of their expert's up projection,
    activated and weighted as the tile leaves the accumulator: `_clamped_relu2_weighted` of the
    product rounded to bf16, which is the grouped-GEMM path's h after its activation pass.

    A row's x is read from the token its slot belongs to; nothing is gathered first. Every
    output element accumulates over K in fp32, in ascending order, as the grouped GEMM's does.
    """
    pid = tl.program_id(0)
    t = pid // (N // BLOCK_N)
    if t >= tl.load(tiles_ptr):
        return
    e = tl.load(tiles_ptr + 1 + 3 * t)
    rows = tl.load(tiles_ptr + 2 + 3 * t) + tl.arange(0, BLOCK_M)
    rmask = rows < tl.load(tiles_ptr + 3 + 3 * t)
    slot = tl.load(order_ptr + rows, mask=rmask, other=0)
    offs_n = (pid % (N // BLOCK_N)) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = x_ptr + (slot // TOPK).to(tl.int64)[:, None] * stride_xt + offs_k[None, :] * stride_xk
    b_ptrs = (
        up_ptr
        + e.to(tl.int64) * stride_ue
        + offs_n[None, :] * stride_un
        + offs_k[:, None] * stride_uk
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K * stride_xk
        b_ptrs += BLOCK_K * stride_uk
    w = tl.load(w_ptr + slot, mask=rmask, other=0.0).to(tl.float32)
    y = _clamped_relu2_weighted(acc.to(tl.bfloat16).to(tl.float32), w[:, None], CLAMP, INV_CLAMP)
    tl.store(
        h_ptr + rows.to(tl.int64)[:, None] * stride_h + offs_n[None, :], y, mask=rmask[:, None]
    )


@triton.jit
def _down_kernel(
    h_ptr,
    down_ptr,
    tiles_ptr,
    o_ptr,
    stride_h,
    stride_de,
    stride_dn,
    stride_dk,
    stride_o,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One work-list entry's rows times one BLOCK_N slice of their expert's down projection,
    rounded to bf16: the grouped-GEMM path's o, accumulated over K in the same ascending order."""
    pid = tl.program_id(0)
    t = pid // (N // BLOCK_N)
    if t >= tl.load(tiles_ptr):
        return
    e = tl.load(tiles_ptr + 1 + 3 * t)
    rows = tl.load(tiles_ptr + 2 + 3 * t) + tl.arange(0, BLOCK_M)
    rmask = rows < tl.load(tiles_ptr + 3 + 3 * t)
    offs_n = (pid % (N // BLOCK_N)) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = h_ptr + rows.to(tl.int64)[:, None] * stride_h + offs_k[None, :]
    b_ptrs = (
        down_ptr
        + e.to(tl.int64) * stride_de
        + offs_n[None, :] * stride_dn
        + offs_k[:, None] * stride_dk
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_dk
    tl.store(
        o_ptr + rows.to(tl.int64)[:, None] * stride_o + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit
def _sum_slots_kernel(
    o_ptr,
    inv_ptr,
    n_valid_ptr,
    out_ptr,
    stride_o,
    stride_out,
    L: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """One token: the down-projection rows of its in-window slots, summed in fp32, rounded once
    to the output's dtype (an fp32 output is the sum itself).

    The rows of one expert are contiguous and the experts ascend with the row index, so taking
    the token's rows smallest first adds its experts in ascending id order -- the order the
    checkpoint's loop `index_add_`s them into its fp32 buffer. Slots outside the window sit at
    or past `n_valid` and add nothing; a token with none in the window comes out as zeros.
    """
    t = tl.program_id(0)
    offs_l = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)
    lmask = offs_l < L
    n_valid = tl.load(n_valid_ptr)
    k = tl.arange(0, TOPK_P2)
    rows = tl.load(inv_ptr + t * TOPK + k, mask=k < TOPK, other=n_valid)
    rows = tl.where(rows < n_valid, rows, n_valid)
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    for _ in tl.static_range(TOPK):
        r = tl.min(rows, axis=0)
        rows = tl.where(rows == r, n_valid, rows)
        part = tl.load(
            o_ptr + r.to(tl.int64) * stride_o + offs_l, mask=lmask & (r < n_valid), other=0.0
        )
        acc += part.to(tl.float32)
    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t.to(tl.int64) * stride_out + offs_l, out, mask=lmask)


@triton.jit
def _gather_rows_kernel(
    x_ptr,
    order_ptr,
    inv_ptr,
    xg_ptr,
    n_rows,
    stride_xt,
    stride_xk,
    stride_g,
    L: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """BLOCK_R rows of the grouped GEMMs' input, after the sort: row r holds slot `order[r]`, so
    it takes the row of x of that slot's token, and `inv[order[r]] = r`. The row gather and the
    inverse permutation in one pass."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    slot = tl.load(order_ptr + rows, mask=rmask, other=0)
    tl.store(inv_ptr + slot, rows.to(tl.int32), mask=rmask)
    src = x_ptr + (slot // TOPK).to(tl.int64)[:, None] * stride_xt
    dst = xg_ptr + rows.to(tl.int64)[:, None] * stride_g
    for l0 in tl.static_range(0, L, BLOCK_L):
        cols = l0 + tl.arange(0, BLOCK_L)
        mask = rmask[:, None] & (cols < L)[None, :]
        row = tl.load(src + cols[None, :] * stride_xk, mask=mask)
        tl.store(dst + cols[None, :], row, mask=mask)


def _sort_keys(
    topk_ids: torch.Tensor, local_expert_offset: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The stable sort of the (token, k) slots by local expert: `order` (int64 [T*K], the slot
    each row holds) and `offs` (int32 [num_local_experts], each expert's end row -- the form
    `torch._grouped_mm` takes). A slot whose expert is outside the window is keyed past the last
    local expert, so it lands at or past `offs[-1]`, where nothing reads it. The keys are sorted
    as int16 when they fit: the same stable order from half the radix sort's passes."""
    local = topk_ids.reshape(-1).to(torch.int32) - local_expert_offset
    in_window = (local >= 0) & (local < num_local_experts)
    key = torch.where(in_window, local, num_local_experts)
    if num_local_experts < torch.iinfo(torch.int16).max:
        key = key.to(torch.int16)
    sorted_key, order = torch.sort(key, stable=True)
    bounds = torch.arange(num_local_experts, dtype=key.dtype, device=key.device)
    offs = torch.searchsorted(sorted_key, bounds, right=True, out_int32=True)
    return order, offs


def _sort_slots(
    topk_ids: torch.Tensor, local_expert_offset: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lay the (token, k) slots out as grouped-GEMM rows: grouped by local expert, ascending.

    Returns `order` and `offs` (`_sort_keys`) and `inv` (int32 [T*K], the row each slot went
    to). The sort is stable: within an expert the rows keep slot order. Device-side throughout,
    no host sync. The layout both paths' sorts give, in torch ops.
    """
    order, offs = _sort_keys(topk_ids, local_expert_offset, num_local_experts)
    inv = torch.empty(order.numel(), dtype=torch.int32, device=order.device)
    inv.scatter_(0, order, torch.arange(order.numel(), dtype=torch.int32, device=order.device))
    return order, inv, offs


def _sort_and_gather(
    x: torch.Tensor, topk_ids: torch.Tensor, local_expert_offset: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The grouped-GEMM path's layout: `_sort_slots`' `order`, `inv` and `offs`, the same
    values, plus each row's x (`xg` [T*K, latent], x's dtype) -- the sort, then one pass
    (`_gather_rows_kernel`) for the inverse permutation and the gather."""
    order, offs = _sort_keys(topk_ids, local_expert_offset, num_local_experts)
    rows = order.numel()
    inv = torch.empty(rows, dtype=torch.int32, device=x.device)
    xg = torch.empty(rows, x.shape[1], dtype=x.dtype, device=x.device)
    block_r = 8
    _gather_rows_kernel[(triton.cdiv(rows, block_r),)](
        x,
        order,
        inv,
        xg,
        rows,
        x.stride(0),
        x.stride(1),
        xg.stride(0),
        L=x.shape[1],
        TOPK=topk_ids.shape[1],
        BLOCK_R=block_r,
        BLOCK_L=min(256, triton.next_power_of_2(x.shape[1])),
        num_warps=4,
    )
    return order, inv, offs, xg


def _count_sort(
    topk_ids: torch.Tensor, local_expert_offset: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_sort_slots`' `order`, `inv` and `offs`, the same values, from one kernel launch;
    `order` comes back int32."""
    ids = topk_ids.reshape(-1)
    n = ids.numel()
    order = torch.empty(n, dtype=torch.int32, device=ids.device)
    inv = torch.empty(n, dtype=torch.int32, device=ids.device)
    offs = torch.empty(num_local_experts, dtype=torch.int32, device=ids.device)
    block = min(max(16, triton.next_power_of_2(n)), 2048)
    _count_sort_kernel[(num_local_experts + 1,)](
        ids,
        order,
        inv,
        offs,
        n,
        local_expert_offset,
        num_local_experts,
        BLOCK=block,
        num_warps=4 if block <= 1024 else 8,
    )
    return order, inv, offs


def _tiles_for(
    x: torch.Tensor, topk_ids: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor
) -> tuple | None:
    """The fused path's (BLOCK_M, up tile, down tile) for this call, or None for the
    grouped-GEMM path: a batch past `_FUSED_TILES`' last row, or dtypes and widths the fused
    kernels are not built for (they take bf16, and widths their tiles divide)."""
    tiles = next((t[1:] for t in _FUSED_TILES if topk_ids.shape[0] <= t[0]), None)
    if tiles is None or not (x.dtype == up_proj.dtype == down_proj.dtype == torch.bfloat16):
        return None
    _, (up_n, up_k, _, _), (down_n, down_k, _, _) = tiles
    latent, intermediate = x.shape[1], up_proj.shape[1]
    if intermediate % up_n or latent % up_k or latent % down_n or intermediate % down_k:
        return None
    return tiles


def _experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation_clamp: float,
    local_expert_offset: int,
    num_local_experts: int,
    out_dtype: torch.dtype | None,
    tiles: tuple | None,
) -> torch.Tensor:
    """The routed experts by one path: the fused Triton GEMMs with `tiles` = (BLOCK_M, up tile,
    down tile), or the grouped-GEMM path when `tiles` is None. Both give every row the same bits."""
    num_tokens, top_k = topk_ids.shape
    latent, intermediate = x.shape[1], up_proj.shape[1]
    w = topk_weights.reshape(-1)
    if tiles is None:
        order, inv, offs, xg = _sort_and_gather(x, topk_ids, local_expert_offset, num_local_experts)
        h = torch._grouped_mm(xg, up_proj.transpose(1, 2), offs=offs)
        _activate_rows(h, w, order, offs[-1:], activation_clamp)
        o = torch._grouped_mm(h, down_proj.transpose(1, 2), offs=offs)
    else:
        block_m, (up_n, up_k, up_warps, up_stages), (dn_n, dn_k, dn_warps, dn_stages) = tiles
        order, inv, offs = _count_sort(topk_ids, local_expert_offset, num_local_experts)
        rows = order.numel()
        max_tiles = triton.cdiv(rows, block_m) + min(num_local_experts, rows)
        work = torch.empty(1 + 3 * max_tiles, dtype=torch.int32, device=x.device)
        _tile_map_kernel[(1,)](
            offs,
            work,
            num_local_experts,
            BLOCK_M=block_m,
            NE=triton.next_power_of_2(num_local_experts),
            num_warps=4,
        )
        h = torch.empty(rows, intermediate, dtype=x.dtype, device=x.device)
        _up_act_kernel[(max_tiles * (intermediate // up_n),)](
            x,
            up_proj,
            w,
            order,
            work,
            h,
            x.stride(0),
            x.stride(1),
            up_proj.stride(0),
            up_proj.stride(1),
            up_proj.stride(2),
            h.stride(0),
            N=intermediate,
            K=latent,
            TOPK=top_k,
            CLAMP=float(activation_clamp),
            INV_CLAMP=_fp32_reciprocal(activation_clamp),
            BLOCK_M=block_m,
            BLOCK_N=up_n,
            BLOCK_K=up_k,
            num_warps=up_warps,
            num_stages=up_stages,
        )
        o = torch.empty(rows, latent, dtype=x.dtype, device=x.device)
        _down_kernel[(max_tiles * (latent // dn_n),)](
            h,
            down_proj,
            work,
            o,
            h.stride(0),
            down_proj.stride(0),
            down_proj.stride(1),
            down_proj.stride(2),
            o.stride(0),
            N=latent,
            K=intermediate,
            BLOCK_M=block_m,
            BLOCK_N=dn_n,
            BLOCK_K=dn_k,
            num_warps=dn_warps,
            num_stages=dn_stages,
        )
    out = torch.empty(
        num_tokens,
        latent,
        dtype=out_dtype if out_dtype is not None else x.dtype,
        device=x.device,
    )
    block_l = min(1024, triton.next_power_of_2(latent))
    _sum_slots_kernel[(num_tokens, triton.cdiv(latent, block_l))](
        o,
        inv,
        offs[-1:],
        out,
        o.stride(0),
        out.stride(0),
        L=latent,
        TOPK=top_k,
        TOPK_P2=triton.next_power_of_2(top_k),
        BLOCK_L=block_l,
        num_warps=4,
    )
    return out


class LatentMoeExperts(OpWrapper):
    """`Nemotron4Experts`: each token's top-k latent experts, weighted, summed. New tensor out.

    For every slot (token t, choice k) whose expert e = `topk_ids[t, k]` lies in the window
    `[local_expert_offset, local_expert_offset + num_local_experts)`:
    h = bf16(x_t @ up_e^T), y = bf16(w_tk * act(h)), o = bf16(y @ down_e^T), with
    act = relu(bf16(c * tanh(h / c)))² rounded to bf16; then out_t = bf16(Σ_e fp32(o)), the
    experts added in ascending id order. The routing weight multiplies after the activation and
    before down_proj (the checkpoint's Megatron order, not the generic-MoE one), and the experts
    are accumulated in fp32.

    Layouts are the checkpoint's: `x` [T, latent] bf16 (the `fc1_latent_proj` output),
    `topk_ids` [T, K] int32 global expert ids and `topk_weights` [T, K] fp32 (the router's,
    already normalized and scaled), `up_proj` [n, intermediate, latent] and `down_proj`
    [n, latent, intermediate] bf16 -- the window's experts, `experts.up_proj[offset:offset + n]`
    as stored. On tp1 the window is every expert, (0, 512); an expert-parallel rank passes its
    own, gets its experts' partial sums, and a slot outside the window adds nothing.

    `out_dtype` None (the default) returns x's dtype, bf16: the sum rounded once.
    `torch.float32` returns the fp32 sum itself, the same additions without that rounding, so
    expert-parallel ranks can add their partial sums before rounding once, as the full window
    does.

    Two paths, both device-side and CUDA-graph capturable, chosen by the batch size
    (`_FUSED_TILES`). Decode-sized batches: one counting-sort kernel lays the slots out by local
    expert, a one-program kernel lists the GEMMs' work (up to BLOCK_M rows of one expert per
    entry), a Triton grouped GEMM computes the up projection straight from each slot's token row
    and applies the activation and routing weight as the tile leaves the accumulator, and a
    Triton grouped GEMM computes the down projection. Larger batches: a stable torch sort of the
    slots, one Triton pass gathering each row's x and writing the inverse permutation,
    `torch._grouped_mm` for up_proj, a Triton pass applying the activation and the routing
    weight in place, and `torch._grouped_mm` for down_proj. Either way a Triton pass then sums
    each token's rows in fp32. The paths round at the same points and accumulate every GEMM
    output over K in fp32 in the same order, so a row has the same bits whichever path computed
    it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Token counts the tp1 target runs this block at, each a different load on the grouped
    # GEMMs: rows per expert go from one (decode, latency-bound) to ~320 (a 16K prefill,
    # compute-bound), and the experts touched from 10 to all 512. Then a 4-rank expert-parallel
    # target's: the end windows at each gathered size, as bf16 and as fp32 partial sums.
    CELLS: tuple[Cell, ...] = (
        _cell("decode, one token: ten experts, one row each, latency-bound", 1),
        _cell("decode 8: ~75 experts touched, mostly one row each", 8),
        _cell("37 tokens: group sizes off every tile multiple", 37),
        _cell("decode 64: ~370 experts, the weight stream dominates", 64),
        _cell("decode 256: every expert touched, ~5 rows each", 256),
        _cell("decode 1024, the largest CUDA-graph batch: ~20 rows per expert", 1024),
        _cell("prefill chunk 2048: ~40 rows per expert", 2048),
        _cell("prefill 8192, the max_num_tokens baseline: ~160 rows per expert", 8192),
        _cell("prefill 16384, the larger token budget: ~320 rows per expert", 16384),
        _cell(
            "an EP rank's window, experts 128..255 of 512: other slots add nothing",
            256,
            offset=128,
            count=128,
        ),
        *_ep_cells(),
    )

    note = """
    Certified: bf16 x and weights, fp32 routing weights, int32 ids, top-10 of 512, latent 768,
    intermediate 4096, clamp 16, the window (0, 512) up to 16384 tokens, and the 4-rank
    expert-parallel windows (128 r, 128): r = 0 and 3 from 4 to 32768 gathered tokens with bf16
    and fp32 outputs, r = 1 at 256 tokens. Not certified, because no target here reaches it:
    another top-k or width, another window size, fp16 or fp64 outputs, a zero-token batch, and
    a token that names the same expert twice (the checkpoint's `index_add_` gives no order for
    that case either). A window none of the batch's slots fall into returns zeros (tested, not
    a cell).

    Expert parallelism: a window gives each of its slots the bits the full-window call gives
    it, so the four windows' fp32 partial sums, added in fp32, differ from the full window's
    fp32 sum only by fp32 reassociation (tested). Rounded once after the reduction they are the
    full window's bf16 result, except where that reassociation crosses a bf16 rounding boundary
    (measured: one element in ten million). Rounding each partial to bf16 first adds a rounding
    per rank: measured, about 0.7 ulp of relative RMS from the full window's result, which is
    outside this entry's band.

    The routing weights are the router's fp32 values. bf16 weights are taken and multiplied as
    the checkpoint would multiply them, but their own rounding moves the result about one ulp
    of RMS from the fp32-weight one -- more than the band -- so a target passes fp32.

    Ids outside the window are skipped, which is what a window means. On tp1 the window is every
    expert, so an id outside [0, 512) is dropped silently; the router's contract keeps ids in
    range, and this entry does not check it.

    `torch._grouped_mm` is a private torch API: its schema and layout rules can change with
    torch. The weights go in as `up_proj.transpose(1, 2)` -- each expert's matrix column-major,
    which is how the checkpoint's [n, out, in] storage already reads -- so nothing is copied.

    The grouped GEMMs accumulate in fp32 in another order than cuBLAS does for the checkpoint's
    `F.linear`, so neither this entry nor the checkpoint's loop is bitwise the fp32-GEMM
    reference; measured on every cell, both sit the same distance from it. The activation is
    bit-exact against the checkpoint's torch ops on every finite bf16 input. A row's result does
    not depend on the rest of the batch (bitwise), and a captured CUDA graph replays it bitwise.

    The two paths' bit-identity rests on Triton's `tl.dot` and `torch._grouped_mm` accumulating
    each output element over K in the same steps, in ascending order -- which neither promises.
    It is measured on every cell the fused path takes and at every row of `_FUSED_TILES`, and a
    row computed alone and inside a batch on the other path is compared bitwise; a change in
    either kernel library shows up there, as a failing test, rather than as rows whose bits
    depend on their batch.

    Memory per call, at T tokens: the down-projection output [T*10, 768] bf16 and the
    intermediate [T*10, 4096] bf16, plus the gathered x [T*10, 768] on the grouped-GEMM path --
    1.3 GB at T = 16384. A window sizes them for every slot too, in window or not: the in-window
    count is only known on the device.
    """

    def __call__(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        activation_clamp: float,
        local_expert_offset: int,
        num_local_experts: int,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return _experts(
            x,
            topk_ids,
            topk_weights,
            up_proj,
            down_proj,
            activation_clamp,
            local_expert_offset,
            num_local_experts,
            out_dtype,
            _tiles_for(x, topk_ids, up_proj, down_proj),
        )

    def reference(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        activation_clamp: float,
        local_expert_offset: int,
        num_local_experts: int,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The checkpoint's loop (`Nemotron4Experts.forward`) with each GEMM in true fp32.

        Experts in ascending id order, each GEMM computed in fp32 and rounded to x's dtype where
        the checkpoint's `F.linear` rounds, the activation and weighting in the checkpoint's own
        torch ops, the experts `index_add_`ed into an fp32 buffer and rounded once to
        `out_dtype` (x's dtype when None; an fp32 output is the buffer itself). Written for any
        dtype, as the checkpoint's code is, so a guard test can show what another dtype should
        have given; the entry itself is certified at bf16 only.
        """
        dt = x.dtype
        out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
        local = topk_ids.long() - local_expert_offset
        for e in range(num_local_experts):
            tok, pos = torch.where(local == e)
            if tok.numel() == 0:
                continue
            with true_fp32_matmul():
                h = (x[tok].float() @ up_proj[e].float().t()).to(dt)
            c = (activation_clamp * torch.tanh(h.float() / activation_clamp)).to(dt)
            y = (torch.square(torch.relu(c)) * topk_weights[tok, pos, None]).to(dt)
            with true_fp32_matmul():
                o = (y.float() @ down_proj[e].float().t()).to(dt)
            out.index_add_(0, tok, o.float())
        return out.to(out_dtype if out_dtype is not None else dt)

    def is_valid(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        activation_clamp: float,
        local_expert_offset: int,
        num_local_experts: int,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        # The activation pass rounds to bf16 whatever it is handed: fp16 or fp32 activations
        # and weights run, and come back with bf16's rounding inside. Mixed dtypes are refused
        # by the grouped GEMM itself.
        assert x.dtype == up_proj.dtype == down_proj.dtype == torch.bfloat16, (
            f"x, up_proj and down_proj must be bf16, got {x.dtype}, {up_proj.dtype} and "
            f"{down_proj.dtype}: the activation pass and the sum round to bf16 regardless"
        )
        # The routing weight of slot (t, k) is read flat, at t * K + k: weights laid out any other
        # way, e.g. [K, T], are read without complaint and multiply the wrong slots.
        assert topk_weights.shape == topk_ids.shape, (
            f"topk_weights {tuple(topk_weights.shape)} must have topk_ids' shape "
            f"{tuple(topk_ids.shape)}; they are read flat, slot by slot"
        )


latent_moe_experts = LatentMoeExperts()
