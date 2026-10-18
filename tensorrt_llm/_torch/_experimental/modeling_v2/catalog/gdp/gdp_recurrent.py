# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One decode step of the gated delta product, each request's state updated in place by slot."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

from .._op import Arch, Cell, OpWrapper

#: Columns of the [K, V] state one program holds. Eight 128-float columns are
#: 4 KiB of fp32 state in one warp's registers, the tile the vendored
#: fused-recurrent kernels use for the same update.
_BLOCK_V = 8
_NUM_WARPS = 1
#: CUDA's limit on the second grid dimension; rows past it are grid-strided.
_MAX_GRID_ROWS = 65535


@triton.jit(do_not_specialize=["num_rows", "num_slots"])
def _gdp_recurrent_kernel(
    q,
    k,
    v,
    g,
    beta,
    out,
    state_pool,
    slots,
    scale,
    num_rows,
    num_slots,
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
    stride_g_t,
    stride_g_h,
    stride_b_t,
    stride_b_j,
    stride_b_h,
    stride_s_s,
    stride_s_h,
    stride_s_v,
    stride_s_k,
    H: tl.constexpr,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """One program per (BV-column tile, request x head row), grid-strided over rows.

    The state tile is loaded once, carried through the token's NH Householder
    steps in fp32 registers and stored once: decay before the first step, the
    query read after the last -- exactly the order the HF expansion feeds the
    per-step recurrence.
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

        b_g = tl.load(g + i_n * stride_g_t + i_h * stride_g_h).to(tl.float32)
        b_h *= tl.exp(b_g)
        for j in tl.static_range(NH):
            b_k = tl.load(
                k + i_n * stride_k_t + j * stride_k_j + i_h * stride_k_h + o_k * stride_k_d,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k) + 1e-6)
            b_v = tl.load(
                v + i_n * stride_v_t + j * stride_v_j + i_h * stride_v_h + o_v * stride_v_d,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            b_beta = tl.load(beta + i_n * stride_b_t + j * stride_b_j + i_h * stride_b_h).to(
                tl.float32
            )
            b_v = (b_v - tl.sum(b_h * b_k[:, None], 0)) * b_beta
            b_h += b_k[:, None] * b_v[None, :]

        b_q = tl.load(
            q + i_n * stride_q_t + i_h * stride_q_h + o_k * stride_q_d, mask=mask_k, other=0.0
        ).to(tl.float32)
        b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(out + (i_n * H + i_h) * V + o_v, b_o.to(out.dtype.element_ty), mask=mask_v)
        if slot >= 0:
            tl.store(p_state, b_h.to(state_pool.dtype.element_ty), mask=mask_kv)
        i_row += tl.num_programs(1)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """HF `l2norm`: `x * rsqrt(sum(x * x) + 1e-6)` over the head dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


class GdpRecurrent(OpWrapper):
    """One decode token per request through `torch_recurrent_gated_delta_rule`'s
    recurrence, Householder-expanded the way `Nemotron4GDPMixer.forward` expands
    it, reading and writing each request's fp32 state in place in the pool.

    Per (request, head), with `S` the HF `[K, V]` state and `l2n(x) = x *
    rsqrt(sum(x*x) + 1e-6)`:

        S *= exp(g)                                  # once, before the first step
        for j in range(n_h):                         # the token's Householder steps
            kj = l2n(k[j]); u = beta[j] * (v[j] - S^T kj); S += kj u^T
        out = S^T (l2n(q) * scale)                   # read after the last step

    Returns the new `out` `[B, H, V]` in `q`'s dtype -- the per-head output
    before `Nemotron4RMSNormGated` -- and leaves the advanced state in
    `state_pool` rows `slots`; a caller that wants the old state has to have
    kept its own copy.
    """

    ARCHS = frozenset({Arch.SM_103})

    # nemotron4-nano-preview's GDP mixer, the one shape all 32 of its GDP and
    # linear_attention layers share: gdp_num_heads 24 (== gdp_num_groups, so no
    # head repeat), gdp_key_head_dim 128, gdp_head_dim 64, gdp_num_householder 3.
    # `scale` is what the target passes for HF's `query / sqrt(key_head_dim)`.
    # In decode q, k and v are views of the in_proj output's conv columns, so
    # their row stride is that output's width, 18528, and the conv columns start
    # at 1536.
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
    # [z 384 | conv 4224 | b 18 | a 6] = 4632 wide. That row stride is not a
    # multiple of 16, so Triton compiles another variant of the kernel for it
    # than for 18528.
    _NEMOTRON4_TP4 = dict(_NEMOTRON4, heads=6, proj_width=4632, conv_offset=384)

    # The batch sizes are the decode CUDA graphs the targets capture (1 ... 1024,
    # 1024 covering the per-rank batch of a DP topology) plus the eager
    # non-power-of-two size 3, at both geometries. Tensor parallelism splits the
    # heads, not the batch, so a tp4 rank runs every request of the step.
    # Padding rows sit at the end of a graph batch and carry PAD_SLOT_ID.
    # "packed" is the hybrid cache manager's pool, whose per-layer view of a
    # slot is followed by that slot's conv state, so the slot stride exceeds
    # H*V*K; "dense" is a [num_slots, H, V, K] tensor.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="concurrency 1: a single request, no padding row",
            spec=dict(batch=1, pad_rows=0, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 2: one request and one padding row",
            spec=dict(batch=2, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="eager batch 3 (not a power of two) over a dense per-layer pool",
            spec=dict(batch=3, pad_rows=0, pool="dense", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 4 with a padding row",
            spec=dict(batch=4, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 8 with a padding row",
            spec=dict(batch=8, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 16 with a padding row",
            spec=dict(batch=16, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 32 with a padding row",
            spec=dict(batch=32, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 64: the batch the CUDA-graph test captures",
            spec=dict(batch=64, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 128 with a padding row",
            spec=dict(batch=128, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 256 with a padding row",
            spec=dict(batch=256, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 512, around tp1's max_batch_size",
            spec=dict(batch=512, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="graph batch 1024 (a DP rank's batch): 24576 request x head rows",
            spec=dict(batch=1024, pad_rows=1, pool="packed", **_NEMOTRON4),
        ),
        Cell(
            why="tp4 rank, concurrency 1: 6 request x head rows, no padding row",
            spec=dict(batch=1, pad_rows=0, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 2: one request and one padding row",
            spec=dict(batch=2, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, eager batch 3 over a dense per-layer pool",
            spec=dict(batch=3, pad_rows=0, pool="dense", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 4 with a padding row",
            spec=dict(batch=4, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 8 with a padding row",
            spec=dict(batch=8, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 16 with a padding row",
            spec=dict(batch=16, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 32 with a padding row",
            spec=dict(batch=32, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 64: the CUDA-graph test's batch",
            spec=dict(batch=64, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 128 with a padding row",
            spec=dict(batch=128, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 256 with a padding row",
            spec=dict(batch=256, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 512 with a padding row",
            spec=dict(batch=512, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
        Cell(
            why="tp4 rank, graph batch 1024: 6144 request x head rows",
            spec=dict(batch=1024, pad_rows=1, pool="packed", **_NEMOTRON4_TP4),
        ),
    )

    note = """
    Inputs, shared with `gdp_chunk`: what `Nemotron4GDPMixer.forward` hands
    `torch_recurrent_gated_delta_rule` (after in_proj, the short conv + SiLU,
    sigmoid and the softplus decay), before its n_h-fold expansion -- q `[B, H,
    K]` and k `[B, n_h, H, K]` not yet l2-normalized, v `[B, n_h, H, V]`, g `[B,
    H]` the fp32 log decay `-exp(A_log) * softplus(a + dt_bias)`, beta `[B, n_h,
    H]` after sigmoid. Any strides: the targets pass views of the conv output,
    which the decode conv writes in place into the in_proj output's conv
    columns, so a token's q, k and v sit in one in_proj row.

    State layout: `state_pool` is `[num_slots, H, V, K]` fp32 with K innermost,
    the pool layout of the vendored FLA kernels (`modules/fla/fused_recurrent.py`,
    `modules/fla/chunk_delta_h.py`) -- HF's `[H, K, V]` state transposed. Any
    slot stride: the hybrid cache manager's per-layer view (`_setup_states`)
    packs each slot's conv state after its ssm state.

    The l2 norm is HF's `x * rsqrt(sum(x*x) + 1e-6)`; the vendored
    fused-recurrent kernels compute `x / (sqrt(sum(x*x)) + 1e-6)`, ~1e-6/|x|
    apart.

    Padding: a row whose slot is PAD_SLOT_ID (-1) starts from a zero state and
    writes nothing; its output is computed and meaningless. The cache managers'
    `get_state_indices` give CUDA-graph padding rows the dummy request's slot,
    once per padding row -- rows that share a slot race on it, so the target
    passes PAD_SLOT_ID for its padding rows instead.

    Every other row continues the state its slot holds: there is no
    `has_initial_state` here. A recycled slot keeps the last request's state,
    so a request that reaches a decode row without a prefill (a one-token
    prompt promoted into a graph batch) needs its slot zeroed first.

    A row's result does not depend on the batch it runs in: one tile shape
    and one reduction order for every batch size, so the same request gets the
    same bits in a padded graph batch and in an eager one.

    Certified head counts: 24 (tp1) and 6 (a tp4 rank), each with its own
    in_proj row stride. Uncertified: more than one token per request (MTP
    verify, per-position states), any other head count, head dims or
    Householder count, a bf16 state pool, query/key heads fewer than value
    heads. CUDA-graph capturable: no host sync and no allocation besides `out`.
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
        scale: float,
    ) -> torch.Tensor:
        """Advances `state_pool[slots]` by one token in place and returns the new `out`."""
        num_tokens, num_householder, num_heads, value_dim = v.shape
        key_dim = k.shape[-1]
        out = torch.empty((num_tokens, num_heads, value_dim), dtype=q.dtype, device=q.device)
        num_rows = num_tokens * num_heads
        if num_rows == 0:
            return out
        grid = (triton.cdiv(value_dim, _BLOCK_V), min(num_rows, _MAX_GRID_ROWS))
        _gdp_recurrent_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            out,
            state_pool,
            slots,
            scale,
            num_rows,
            state_pool.shape[0],
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *g.stride(),
            *beta.stride(),
            *state_pool.stride(),
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
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, the pool as the op leaves it)`, fp32 throughout; mutates nothing.

        Transcribed from `torch_recurrent_gated_delta_rule` with
        `use_qk_l2norm_in_kernel=True`, over the token's steps as
        `Nemotron4GDPMixer.forward` expands them: the decay rides on the first
        step only (the others' `g` is zero, and `exp(0)` is exactly one) and the
        query is read only after the last (the others' query is zero and their
        output is dropped). HF divides the l2-normalized query by
        `sqrt(key_dim)`; `scale` is that factor.

        The returned pool is a copy with the real rows' slots advanced; a
        PAD_SLOT_ID row's output is computed from a zero state.
        """
        pad = slots < 0
        rows = torch.where(pad, torch.zeros_like(slots), slots).long()
        state = state_pool[rows].float().transpose(-1, -2)  # [B, H, K, V]
        state = torch.where(pad[:, None, None, None], torch.zeros_like(state), state)

        query = _l2norm(q.float()) * scale
        key = _l2norm(k.float())
        value, step_beta = v.float(), beta.float()
        state = state * g.float().exp()[..., None, None]
        for j in range(v.shape[1]):
            kv_mem = (state * key[:, j, :, :, None]).sum(dim=-2)
            delta = (value[:, j] - kv_mem) * step_beta[:, j, :, None]
            state = state + key[:, j, :, :, None] * delta[:, :, None, :]
        out = (state * query[..., None]).sum(dim=-2)

        after = state_pool.clone()
        real = ~pad
        after[slots[real].long()] = state[real].transpose(-1, -2).to(state_pool.dtype)
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
        scale: float,
    ) -> None:
        # The kernel takes its sizes from `v` and `k` and reads every other
        # tensor through raw strides, so a tensor of the wrong shape is read
        # as whatever those sizes say rather than rejected.
        num_tokens, num_householder, num_heads, value_dim = v.shape
        key_dim = k.shape[-1]
        assert k.shape == (num_tokens, num_householder, num_heads, key_dim), (
            f"k must be [B, n_h, H, K] = {(num_tokens, num_householder, num_heads, key_dim)}; "
            f"got {tuple(k.shape)}"
        )
        assert q.shape == (num_tokens, num_heads, key_dim), (
            f"q must be [B, H, K] = {(num_tokens, num_heads, key_dim)}, one query per token; "
            f"got {tuple(q.shape)}"
        )
        assert g.shape == (num_tokens, num_heads), (
            f"g must be [B, H] = {(num_tokens, num_heads)}, one decay per token; "
            f"got {tuple(g.shape)}"
        )
        assert beta.shape == (num_tokens, num_householder, num_heads), (
            f"beta must be [B, n_h, H] = {(num_tokens, num_householder, num_heads)}; "
            f"got {tuple(beta.shape)}"
        )
        assert slots.shape == (num_tokens,) and not slots.is_floating_point(), (
            f"slots must be one integer per request; got {slots.dtype} {tuple(slots.shape)}"
        )
        # HF's [H, K, V] state has the same element count; read through these
        # strides it comes back transposed.
        assert tuple(state_pool.shape[1:]) == (num_heads, value_dim, key_dim), (
            f"state_pool must be [num_slots, H, V, K] = [*, {num_heads}, {value_dim}, {key_dim}] "
            f"with K innermost; got {tuple(state_pool.shape)}"
        )
        # A bf16 pool is loaded and stored without complaint, rounding the
        # recurrent state at every decode step; HF keeps it in fp32.
        assert state_pool.dtype == torch.float32, (
            f"state_pool holds the recurrent state in float32; got {state_pool.dtype}"
        )
        # HF computes the decay in fp32. Rounding g to bf16 moves it by up to
        # |g| * 2^-9, so exp(g) by that much relative -- percents on the heads
        # that forget fastest -- and the state compounds it every token.
        assert g.dtype == torch.float32, f"g (the log decay) must be float32; got {g.dtype}"
        # The two mistakes the kernel cannot see: a positive g grows the state
        # every step instead of decaying it, and a pre-sigmoid beta over-writes it.
        assert bool((g <= 0).all()), (
            "g must be the log decay -exp(A_log) * softplus(a + dt_bias) <= 0"
        )
        assert bool(((beta >= 0) & (beta <= 1)).all()), (
            "beta must be post-sigmoid, in [0, 1]: pass sigmoid(b), not b"
        )
        # An out-of-range slot reads and writes outside the pool; any negative
        # slot is treated as padding, so only PAD_SLOT_ID is the padding value.
        assert bool(((slots >= PAD_SLOT_ID) & (slots < state_pool.shape[0])).all()), (
            f"slots must be in [0, {state_pool.shape[0]}) or PAD_SLOT_ID ({PAD_SLOT_ID})"
        )
        real = slots[slots >= 0]
        assert real.unique().numel() == real.numel(), (
            "two rows share a slot and race on it; give padding rows PAD_SLOT_ID"
        )


gdp_recurrent = GdpRecurrent()
