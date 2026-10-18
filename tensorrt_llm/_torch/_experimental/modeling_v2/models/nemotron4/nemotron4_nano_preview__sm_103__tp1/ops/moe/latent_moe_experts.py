# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent MoE's routed experts: soft-clamped relu², routing weight applied before down_proj."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

# Every MoE block of the checkpoint (`Nemotron4Experts`, 28 shortcut layers plus the MTP layer)
# has one geometry: 512 routed experts, top-10, experts working in the 768-wide latent space
# (`moe_latent_size`) with intermediate 4096 (`moe_intermediate_size`), and the activation
# 16 * tanh(x / 16) before relu² (`activation_clamp_value`, `mlp_hidden_act` relu2).
_LATENT_MOE = dict(num_experts=512, top_k=10, latent=768, intermediate=4096, activation_clamp=16.0)


def _cell(why: str, tokens: int, offset: int = 0, count: int = 512) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            tokens=tokens, local_expert_offset=offset, num_local_experts=count, **_LATENT_MOE
        ),
    )


@triton.jit
def _clamped_relu2_weighted_kernel(
    h_ptr,
    w_ptr,
    order_ptr,
    n_valid_ptr,
    stride_h,
    N: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One row of the up-projection output, in place: bf16(w * bf16(relu(bf16(c*tanh(h/c)))^2)).

    Each rounding is the checkpoint's: the clamp is computed in fp32 from the bf16 GEMM output
    and rounded, relu² runs on bf16 tensors (one fp32 product, rounded), and the bf16 result times
    the fp32 routing weight is an fp32 product rounded once. `div_rn` and libdevice's tanh are the
    correctly rounded division and the same tanhf torch's CUDA kernel calls. `c < 0` rather than a
    max keeps a NaN a NaN, as torch.relu does.
    """
    row = tl.program_id(0)
    if row < tl.load(n_valid_ptr):
        w = tl.load(w_ptr + tl.load(order_ptr + row)).to(tl.float32)
        base = h_ptr + row.to(tl.int64) * stride_h
        for n0 in tl.static_range(0, N, BLOCK_N):
            offs = n0 + tl.arange(0, BLOCK_N)
            mask = offs < N
            h = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
            c = (CLAMP * libdevice.tanh(tl.math.div_rn(h, CLAMP))).to(tl.bfloat16).to(tl.float32)
            r = tl.where(c < 0.0, 0.0, c)
            s = (r * r).to(tl.bfloat16).to(tl.float32)
            tl.store(base + offs, (s * w).to(tl.bfloat16), mask=mask)


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
    """One token: the down-projection rows of its in-window slots, summed in fp32, rounded once.

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
    tl.store(out_ptr + t.to(tl.int64) * stride_out + offs_l, acc.to(tl.bfloat16), mask=lmask)


def _sort_slots(
    topk_ids: torch.Tensor, local_expert_offset: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lay the (token, k) slots out as grouped-GEMM rows: grouped by local expert, ascending.

    Returns `order` (int64 [T*K], the slot each row holds), `inv` (int32 [T*K], the row each
    slot went to) and `offs` (int32 [num_local_experts], each expert's end row -- the form
    `torch._grouped_mm` takes). A slot whose expert is outside the window is keyed past the last
    local expert, so it lands at or past `offs[-1]`, where nothing reads it. The sort is stable:
    within an expert the rows keep slot order. Device-side throughout, no host sync.
    """
    local = topk_ids.reshape(-1).to(torch.int32) - local_expert_offset
    in_window = (local >= 0) & (local < num_local_experts)
    key = torch.where(in_window, local, num_local_experts)
    sorted_key, order = torch.sort(key, stable=True)
    bounds = torch.arange(num_local_experts, dtype=key.dtype, device=key.device)
    offs = torch.searchsorted(sorted_key, bounds, right=True, out_int32=True)
    inv = torch.empty(order.numel(), dtype=torch.int32, device=order.device)
    inv.scatter_(0, order, torch.arange(order.numel(), dtype=torch.int32, device=order.device))
    return order, inv, offs


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

    The path, all device-side and CUDA-graph capturable: a stable sort of the slots by local
    expert, a row gather of `x`, `torch._grouped_mm` for up_proj, a Triton pass applying the
    activation and the routing weight in place, `torch._grouped_mm` for down_proj, and a Triton
    pass summing each token's rows in fp32.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Token counts the tp1 target runs this block at, each a different load on the grouped
    # GEMMs: rows per expert go from one (decode, latency-bound) to ~320 (a 16K prefill,
    # compute-bound), and the experts touched from 10 to all 512.
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
    )

    note = """
    Certified: bf16 x and weights, fp32 routing weights, int32 ids, top-10 of 512, latent 768,
    intermediate 4096, clamp 16, the window (0, 512) and one expert-parallel window (128, 128).
    Not certified, because no target here reaches it: another top-k or width, a zero-token
    batch, and a token that names the same expert twice (the checkpoint's `index_add_` gives no
    order for that case either). A window none of the batch's slots fall into returns zeros
    (tested, not a cell: no shipped target runs expert parallelism yet).

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
    reference; measured on every cell, both sit the same distance from it. The activation pass is
    bit-exact against the checkpoint's torch ops on every finite bf16 input. A row's result does
    not depend on the rest of the batch (bitwise), and a captured CUDA graph replays it bitwise.

    Memory per call, at T tokens: the gathered x and the down-projection output [T*10, 768] bf16
    and the intermediate [T*10, 4096] bf16 -- 1.3 GB at T = 16384.
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
    ) -> torch.Tensor:
        num_tokens, top_k = topk_ids.shape
        latent, intermediate = x.shape[1], up_proj.shape[1]
        order, inv, offs = _sort_slots(topk_ids, local_expert_offset, num_local_experts)
        n_valid = offs[-1:]
        h = torch._grouped_mm(x.index_select(0, order // top_k), up_proj.transpose(1, 2), offs=offs)
        _clamped_relu2_weighted_kernel[(order.numel(),)](
            h,
            topk_weights.reshape(-1),
            order,
            n_valid,
            h.stride(0),
            N=intermediate,
            CLAMP=float(activation_clamp),
            BLOCK_N=min(4096, triton.next_power_of_2(intermediate)),
            num_warps=8,
        )
        o = torch._grouped_mm(h, down_proj.transpose(1, 2), offs=offs)
        out = torch.empty(num_tokens, latent, dtype=x.dtype, device=x.device)
        block_l = min(1024, triton.next_power_of_2(latent))
        _sum_slots_kernel[(num_tokens, triton.cdiv(latent, block_l))](
            o,
            inv,
            n_valid,
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
    ) -> torch.Tensor:
        """The checkpoint's loop (`Nemotron4Experts.forward`) with each GEMM in true fp32.

        Experts in ascending id order, each GEMM computed in fp32 and rounded to x's dtype where
        the checkpoint's `F.linear` rounds, the activation and weighting in the checkpoint's own
        torch ops, the experts `index_add_`ed into an fp32 buffer and rounded once. Written for
        any dtype, as the checkpoint's code is, so a guard test can show what another dtype
        should have given; the entry itself is certified at bf16 only.
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
        return out.to(dt)

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
    ) -> None:
        # The activation pass and the sum round to bf16 whatever they are handed: fp16 or fp32
        # activations and weights run, and come back with bf16's rounding inside. Mixed dtypes
        # are refused by the grouped GEMM itself.
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
