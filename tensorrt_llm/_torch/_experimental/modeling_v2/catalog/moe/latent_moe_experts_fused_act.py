# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent MoE's routed experts with the activation in the up GEMM's epilogue: `latent_moe_experts`' bits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .._launch import Launch
from .._op import Arch, Cell, OpWrapper
from .latent_moe_experts import (
    _FUSED_TILES,
    _SUM_SLOTS,
    _TILE_MAP,
    _clamped_relu2_weighted,
    _experts,
    _fp32_reciprocal,
    _sort_and_gather,
    _tiles_for,
    latent_moe_experts,
)

#: The up GEMM's tile: BLOCK_M rows of one local expert times BLOCK_N intermediate channels, K walked in BLOCK_K
#: steps. 8 warps and 4 stages; the fastest of the tiles measured on the target's own prompt-sized routing.
_BLOCK_M, _BLOCK_N, _BLOCK_K, _NUM_WARPS, _NUM_STAGES = 128, 256, 64, 8, 4

#: Clamps at which `tanh.approx.f32` gives the activation the exact epilogue gives, on every bf16 input and every
#: routing weight: measured exhaustively by the entry's test. Any other clamp takes libdevice's tanhf.
_APPROX_TANH_CLAMPS = frozenset({16.0, 10.0})


@triton.jit
def _tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _clamped_relu2_weighted_approx(h, w, CLAMP: tl.constexpr, INV_CLAMP: tl.constexpr):
    """`_clamped_relu2_weighted` with the hardware's tanh approximation in place of libdevice's tanhf.

    The clamp c = bf16(CLAMP * tanh(relu(h) / CLAMP)) can come out one subnormal ulp from the exact one on a
    subnormal h, and nowhere else; c² then underflows to the same zero. So s = bf16(c²), and the weighted result,
    are the exact ones on every bf16 h at the clamps in `_APPROX_TANH_CLAMPS` (tested on all 65536 values).
    """
    r = tl.maximum(h, 0.0, propagate_nan=tl.PropagateNan.ALL)
    c = (CLAMP * _tanh_approx(r * INV_CLAMP)).to(tl.bfloat16).to(tl.float32)
    s = (c * c).to(tl.bfloat16).to(tl.float32)
    return (s * w).to(tl.bfloat16)


@triton.jit
def _activate_tile(acc, w, CLAMP: tl.constexpr, INV_CLAMP: tl.constexpr, APPROX: tl.constexpr):
    """The grouped path's h for one accumulator tile: the GEMM output rounded to bf16, activated, weighted."""
    h = acc.to(tl.bfloat16).to(tl.float32)
    if APPROX:
        y = _clamped_relu2_weighted_approx(h, w[:, None], CLAMP, INV_CLAMP)
    else:
        y = _clamped_relu2_weighted(h, w[:, None], CLAMP, INV_CLAMP)
    return y


@triton.jit(do_not_specialize=["num_programs"])
def _up_act_kernel(
    x_desc,
    up_desc,
    h_ptr,
    w_ptr,
    order_ptr,
    tiles_ptr,
    num_programs,
    stride_h,
    N: tl.constexpr,
    K: tl.constexpr,
    CLAMP: tl.constexpr,
    INV_CLAMP: tl.constexpr,
    APPROX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """The up projection of every work-list entry's rows, activated and weighted in the epilogue.

    One program per SM walks the (entry, BLOCK_N slice) pairs; an entry is up to BLOCK_M rows of one local
    expert (`_tile_map_kernel`), and its slices are consecutive, so the rows of one expert share each weight
    slice while it is in L2. The loads and the MMAs run in their own warps. Each output element accumulates
    over K from 0 in fp32, in order, as the grouped GEMM's does. The epilogue rounds the tile to bf16, applies
    the activation and the row's routing weight, and stores the rows that belong to the entry's expert: a
    partial entry's TMA loads read the next expert's rows, which are computed and never stored.
    """
    start = tl.program_id(0)
    NUM_N: tl.constexpr = N // BLOCK_N
    num_items = tl.load(tiles_ptr) * NUM_N
    # The epilogue recomputes its item's ids rather than carrying the loop's, so the stores of one item can
    # overlap the loads of the next.
    item_c = start - num_programs
    for item in tl.range(start, num_items, num_programs, flatten=True, warp_specialize=True):
        t = item // NUM_N
        e = tl.load(tiles_ptr + 1 + 3 * t)
        r0 = tl.load(tiles_ptr + 2 + 3 * t)
        n0 = (item % NUM_N) * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(K // BLOCK_K):
            a = x_desc.load([r0, ki * BLOCK_K])
            b = up_desc.load([e * N + n0, ki * BLOCK_K])
            acc = tl.dot(a, b.T, acc)
        item_c += num_programs
        t_c = item_c // NUM_N
        r0_c = tl.load(tiles_ptr + 2 + 3 * t_c)
        r1_c = tl.load(tiles_ptr + 3 + 3 * t_c)
        n0_c = (item_c % NUM_N) * BLOCK_N
        rows = r0_c + tl.arange(0, BLOCK_M)
        rmask = rows < r1_c
        slot = tl.load(order_ptr + rows, mask=rmask, other=0)
        w = tl.load(w_ptr + slot, mask=rmask, other=0.0).to(tl.float32)
        # Two column halves: half the registers of the whole tile at a time.
        halves = tl.permute(tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2)), (0, 2, 1))
        lo, hi = tl.split(halves)
        cols = n0_c + tl.arange(0, BLOCK_N // 2)
        ptrs = h_ptr + rows.to(tl.int64)[:, None] * stride_h + cols[None, :]
        tl.store(ptrs, _activate_tile(lo, w, CLAMP, INV_CLAMP, APPROX), mask=rmask[:, None])
        tl.store(
            ptrs + BLOCK_N // 2,
            _activate_tile(hi, w, CLAMP, INV_CLAMP, APPROX),
            mask=rmask[:, None],
        )


#: The up GEMM launched straight to its compiled form once compiled (`Launch`), as the entry's
#: other kernels are: a tp4 prompt step makes one call per MoE block, and its host time counts.
_UP_ACT = Launch(_up_act_kernel)


def _takes_fused_act(
    x: torch.Tensor, topk_ids: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor
) -> bool:
    """Whether this call takes the fused-activation path: the batch sizes `latent_moe_experts` runs on its
    grouped GEMMs, with bf16 operands, an up_proj the descriptor can read flat, and widths the tile divides.
    Anything else runs `latent_moe_experts`' own path for it."""
    return (
        topk_ids.shape[0] > _FUSED_TILES[-1][0]
        and x.dtype == up_proj.dtype == down_proj.dtype == torch.bfloat16
        and up_proj.is_contiguous()
        and up_proj.shape[1] % _BLOCK_N == 0
        and x.shape[1] % _BLOCK_K == 0
    )


def _experts_fused_act(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation_clamp: float,
    local_expert_offset: int,
    num_local_experts: int,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    """`latent_moe_experts`' grouped path with its up GEMM and activation pass as one kernel: the sort and
    gather, the activated up projection (`_up_act_kernel`), `torch._grouped_mm` for down_proj, and the sum."""
    num_tokens, top_k = topk_ids.shape
    latent, intermediate = x.shape[1], up_proj.shape[1]
    order, inv, offs, xg = _sort_and_gather(x, topk_ids, local_expert_offset, num_local_experts)
    rows = order.numel()
    max_tiles = triton.cdiv(rows, _BLOCK_M) + min(num_local_experts, rows)
    work = torch.empty(1 + 3 * max_tiles, dtype=torch.int32, device=x.device)
    _TILE_MAP[(1,)](
        offs,
        work,
        num_local_experts,
        BLOCK_M=_BLOCK_M,
        NE=triton.next_power_of_2(num_local_experts),
        num_warps=4,
    )
    h = torch.empty(rows, intermediate, dtype=x.dtype, device=x.device)
    programs = torch.cuda.get_device_properties(x.device).multi_processor_count
    _UP_ACT[(programs,)](
        TensorDescriptor.from_tensor(xg, [_BLOCK_M, _BLOCK_K]),
        TensorDescriptor.from_tensor(up_proj.view(-1, latent), [_BLOCK_N, _BLOCK_K]),
        h,
        topk_weights.reshape(-1),
        order,
        work,
        programs,
        h.stride(0),
        N=intermediate,
        K=latent,
        CLAMP=float(activation_clamp),
        INV_CLAMP=_fp32_reciprocal(activation_clamp),
        APPROX=float(activation_clamp) in _APPROX_TANH_CLAMPS,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    o = torch._grouped_mm(h, down_proj.transpose(1, 2), offs=offs)
    out = torch.empty(
        num_tokens,
        latent,
        dtype=out_dtype if out_dtype is not None else x.dtype,
        device=x.device,
    )
    block_l = min(1024, triton.next_power_of_2(latent))
    _SUM_SLOTS[(num_tokens, triton.cdiv(latent, block_l))](
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


class LatentMoeExpertsFusedAct(OpWrapper):
    """`latent_moe_experts` -- the same arguments, the same result, bit for bit -- with the prompt-sized
    batches' up GEMM and activation pass done by one kernel. New tensor out.

    For every slot (token t, choice k) whose expert e = `topk_ids[t, k]` lies in the window
    `[local_expert_offset, local_expert_offset + num_local_experts)`: h = bf16(x_t @ up_e^T),
    y = bf16(w_tk * act(h)), o = bf16(y @ down_e^T), act = relu(bf16(c * tanh(h / c)))² rounded to bf16;
    out_t = bf16(Σ_e fp32(o)) over the experts in ascending id order, or the fp32 sum itself for
    `out_dtype` torch.float32. Layouts, windows and slices are `latent_moe_experts`' (read its docstring).

    Batches past `latent_moe_experts`' decode tiles (more than 1024 tokens) take the fused-activation path:
    the grouped path's sort and gather, then one persistent Triton GEMM for up_proj whose epilogue applies
    the activation and the routing weight as each tile leaves the accumulator, then `torch._grouped_mm` for
    down_proj and the per-token sum. The intermediate is written once and read once, where the grouped path
    writes it, reads and writes it again in the activation pass, and reads it. Smaller batches, and inputs
    the fused-activation path is not built for, run `latent_moe_experts`' own path.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The configurations `latent_moe_experts` is certified over, since this entry serves the same call:
    # its decode cells run that entry's own fused path, and the prompt-sized ones (2048 tokens and up on the
    # whole window, the expert-parallel windows from 8192 gathered tokens, the tensor-parallel slice from
    # 8192) run this entry's.
    CELLS: tuple[Cell, ...] = latent_moe_experts.CELLS

    note = """
    Certified: `latent_moe_experts`' cells, each giving that entry's bits (tested bitwise on every cell, and
    against the shared reference within that entry's band), so its certified range and its list of what is
    not certified are this entry's.

    The bits rest on two measured properties, each tested on every commit:
      * the up GEMM accumulates each output element over K from 0 in fp32, in tcgen05's steps of 16, in
        order, and never splits K -- as `torch._grouped_mm` does, so h is the grouped path's h before the
        activation, as `invariant_mm` finds for its tiles;
      * `tanh.approx.f32` in place of libdevice's tanhf changes the clamp c only on subnormal h, by one
        subnormal ulp, which c² then flushes to the same zero: the weighted activation is the exact one on
        every bf16 h and every fp32 weight at the clamps in `_APPROX_TANH_CLAMPS` (16, the checkpoint's, and
        10), all 65536 values checked per clamp. Any other clamp takes the exact epilogue, libdevice's tanhf,
        which is slower: its tanh is most of the epilogue's instructions.

    Measured on the target's prompt-sized routing (sm_103, CUDA-graph replay, every weight streamed from
    HBM): the call takes 13 to 19% less time than `latent_moe_experts` on the whole window from 1025 to 16384
    tokens, 15 to 25% less on the expert-parallel windows from 4096 to 65536 gathered tokens (10 to 12% at
    2048), and 14 to 17% less on the tensor-parallel slice. The up GEMM with its activation runs within 2% of
    the same kernel storing the raw product. Forced onto decode-sized batches it is slower than
    `latent_moe_experts`' decode path at every size measured, up to the 1024 tokens where the paths meet.

    What is still the grouped path's: the intermediate [T*10, intermediate] bf16 is materialized (written by
    the up GEMM, read by down_proj's `_grouped_mm`), and so is the gathered x. Memory per call is
    `latent_moe_experts`'.

    The fused-activation path needs bf16 operands, a contiguous up_proj (read through a TMA descriptor as
    [n * intermediate, latent]), latent a multiple of 64 and intermediate a multiple of 256; anything else
    runs `latent_moe_experts`' path. Its two descriptors are built on the host per call: host time on an
    eager call, nothing under CUDA-graph replay.
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
        if _takes_fused_act(x, topk_ids, up_proj, down_proj):
            return _experts_fused_act(
                x,
                topk_ids,
                topk_weights,
                up_proj,
                down_proj,
                activation_clamp,
                local_expert_offset,
                num_local_experts,
                out_dtype,
            )
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
        """The checkpoint's loop (`Nemotron4Experts.forward`) with each GEMM in true fp32:
        `latent_moe_experts`' reference, which this entry shares as it shares the op."""
        return latent_moe_experts.reference(
            x,
            topk_ids,
            topk_weights,
            up_proj,
            down_proj,
            activation_clamp,
            local_expert_offset,
            num_local_experts,
            out_dtype,
        )

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
        # `latent_moe_experts`' guards, for the same reasons: the activation rounds to bf16 whatever it is
        # handed, and the routing weights are read flat, slot by slot.
        assert x.dtype == up_proj.dtype == down_proj.dtype == torch.bfloat16, (
            f"x, up_proj and down_proj must be bf16, got {x.dtype}, {up_proj.dtype} and "
            f"{down_proj.dtype}: the activation pass and the sum round to bf16 regardless"
        )
        assert topk_weights.shape == topk_ids.shape, (
            f"topk_weights {tuple(topk_weights.shape)} must have topk_ids' shape "
            f"{tuple(topk_ids.shape)}; they are read flat, slot by slot"
        )


latent_moe_experts_fused_act = LatentMoeExpertsFusedAct()
