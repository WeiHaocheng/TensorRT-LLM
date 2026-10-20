# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gated delta product over a varlen batch of prefill sequences through FlashInfer's
Blackwell GDN prefill kernel, final states written back by slot."""

from __future__ import annotations

import flashinfer
import torch
import triton
import triton.language as tl

from .._launch import Launch
from .._op import Arch, Cell, OpWrapper

#: FlashInfer's Blackwell GDN prefill takes one head size for q, k and v, and only 128.
_HEAD = 128

#: Where the context-parallel kernel starts to beat the per-(sequence, head) one, measured on
#: sm_103 (152 SMs) over the expanded sequences: see `_use_cp`.
_CP_MIN_LEN = 1024
_CP_LOW_PARALLELISM = 64
_CP_LONG_LEN = 4096


def _use_cp(max_seqlen: int, num_tokens: int, num_seqs: int, num_heads: int) -> bool:
    """Whether FlashInfer's context-parallel (CP) path is the faster one for this batch.

    The non-CP kernel runs one program per (sequence, head) along its whole expanded chain, so
    its time follows the longest sequence; CP splits each sequence into chunks and recombines
    their states, which pays a fixed ~0.04 ms and loses when several mid-length sequences
    already fill the GPU. Measured kernel time, ms, 24 heads, CP / non-CP: one sequence of 32
    tokens 0.039 / 0.016, 1024 0.094 / 0.095, 8192 0.334 / 0.675; (8160, 32) 0.351 / 0.674;
    (8192, 8192) 0.619 / 0.675; 4 x 2048 0.471 / 0.183; 8 x 1024 0.512 / 0.200; 6 heads:
    (4096, 4096) 0.152 / 0.343, 8 x 1024 0.219 / 0.099. So CP when the longest sequence is
    long and either holds most of the tokens or the batch leaves the GPU mostly idle. The
    choice moves only time: both paths compute the same recurrence and are certified alike.
    """
    if max_seqlen < _CP_MIN_LEN:
        return False
    if 2 * max_seqlen > num_tokens:
        return True
    return num_seqs * num_heads <= _CP_LOW_PARALLELISM and max_seqlen >= _CP_LONG_LEN


@triton.jit
def _expand_kernel(
    q,
    k,
    v,
    beta,
    g,
    q_steps,
    k_steps,
    v_steps,
    alpha_steps,
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
    stride_g_t,
    stride_g_h,
    H: tl.constexpr,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
):
    """One program per token: its NH Householder sub-steps the way FlashInfer reads a
    sequence. The query is zero but at the last sub-step, where it is l2-normalized; each key
    is l2-normalized; each value is zero-padded from V to D; the linear decay is exp(g) at the
    first sub-step and exactly one at the others; beta goes to fp32. Normalization is HF's
    `x * rsqrt(sum(x * x) + 1e-6)` in fp32, rounded once to the inputs' dtype."""
    i_t = tl.program_id(0).to(tl.int64)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, K)
    o_d = tl.arange(0, D)
    m_h = o_h < H
    m_k = m_h[:, None] & (o_k < K)[None, :]
    m_d = m_h[:, None] & (o_d < D)[None, :]
    b_q = tl.load(
        q + i_t * stride_q_t + o_h[:, None] * stride_q_h + o_k[None, :] * stride_q_d,
        mask=m_k,
        other=0.0,
    ).to(tl.float32)
    b_q = b_q * tl.math.rsqrt(tl.sum(b_q * b_q, axis=1) + 1e-6)[:, None]
    b_g = tl.load(g + i_t * stride_g_t + o_h * stride_g_h, mask=m_h, other=0.0).to(tl.float32)
    for j in tl.static_range(NH):
        i_s = i_t * NH + j
        p_q = q_steps + i_s * H * K + o_h[:, None] * K + o_k[None, :]
        if j == NH - 1:
            tl.store(p_q, b_q.to(q_steps.dtype.element_ty), mask=m_k)
        else:
            tl.store(p_q, tl.zeros([BH, K], dtype=q_steps.dtype.element_ty), mask=m_k)
        b_k = tl.load(
            k
            + i_t * stride_k_t
            + j * stride_k_j
            + o_h[:, None] * stride_k_h
            + o_k[None, :] * stride_k_d,
            mask=m_k,
            other=0.0,
        ).to(tl.float32)
        b_k = b_k * tl.math.rsqrt(tl.sum(b_k * b_k, axis=1) + 1e-6)[:, None]
        tl.store(
            k_steps + i_s * H * K + o_h[:, None] * K + o_k[None, :],
            b_k.to(k_steps.dtype.element_ty),
            mask=m_k,
        )
        b_v = tl.load(
            v
            + i_t * stride_v_t
            + j * stride_v_j
            + o_h[:, None] * stride_v_h
            + o_d[None, :] * stride_v_d,
            mask=m_h[:, None] & (o_d < V)[None, :],
            other=0.0,
        )
        tl.store(v_steps + i_s * H * D + o_h[:, None] * D + o_d[None, :], b_v, mask=m_d)
        b_b = tl.load(
            beta + i_t * stride_b_t + j * stride_b_j + o_h * stride_b_h, mask=m_h, other=0.0
        )
        tl.store(beta_steps + i_s * H + o_h, b_b.to(tl.float32), mask=m_h)
        if j == 0:
            tl.store(alpha_steps + i_s * H + o_h, tl.exp(b_g), mask=m_h)
        else:
            tl.store(alpha_steps + i_s * H + o_h, tl.full([BH], 1.0, tl.float32), mask=m_h)


@triton.jit
def _gather_state_kernel(
    state_pool,
    slots,
    has_initial_state,
    cu_seqlens,
    initial,
    cu_steps,
    stride_p_n,
    stride_p_h,
    stride_p_v,
    stride_p_k,
    H: tl.constexpr,
    NH: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (sequence, head): the slot's `[V, K]` state into the first V of the
    D rows of the sequence's packed initial state, zeros below it and everywhere for a fresh
    sequence (its slot is not read). Head 0's program also writes the sequence's end in
    sub-steps, the offsets FlashInfer walks."""
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    o_d = tl.arange(0, D)
    o_k = tl.arange(0, K)
    resume = tl.load(has_initial_state + i_n).to(tl.int1)
    slot = tl.load(slots + i_n).to(tl.int64)
    b_s = tl.load(
        state_pool
        + slot * stride_p_n
        + i_h * stride_p_h
        + o_d[:, None] * stride_p_v
        + o_k[None, :] * stride_p_k,
        mask=((o_d < V)[:, None] & (o_k < K)[None, :]) & resume,
        other=0.0,
    )
    tl.store(initial + ((i_n * H + i_h) * D + o_d[:, None]) * K + o_k[None, :], b_s)
    if i_h == 0:
        tl.store(cu_steps + i_n + 1, (tl.load(cu_seqlens + i_n + 1) * NH).to(tl.int32))
        if i_n == 0:
            tl.store(cu_steps, (tl.load(cu_seqlens) * NH).to(tl.int32))


@triton.jit
def _scatter_state_kernel(
    final,
    state_pool,
    slots,
    stride_p_n,
    stride_p_h,
    stride_p_v,
    stride_p_k,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (sequence, head): the first V of the final state's D rows back into
    the slot. The rows past V were zero going in and come out zero; they are not written."""
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    o_d = tl.arange(0, D)
    o_k = tl.arange(0, K)
    mask = (o_d < V)[:, None] & (o_k < K)[None, :]
    slot = tl.load(slots + i_n).to(tl.int64)
    b_s = tl.load(final + ((i_n * H + i_h) * D + o_d[:, None]) * K + o_k[None, :], mask=mask)
    tl.store(
        state_pool
        + slot * stride_p_n
        + i_h * stride_p_h
        + o_d[:, None] * stride_p_v
        + o_k[None, :] * stride_p_k,
        b_s.to(state_pool.dtype.element_ty),
        mask=mask,
    )


#: Each glue kernel launched straight to its compiled form once compiled (`Launch`): an eager
#: prefill step makes one call per GDP layer, so each launch's host time counts.
_EXPAND = Launch(_expand_kernel)
_GATHER_STATE = Launch(_gather_state_kernel)
_SCATTER_STATE = Launch(_scatter_state_kernel)


def _flashinfer_gdp_chunk_fwd(
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
    use_cp: bool,
) -> torch.Tensor:
    num_tokens, num_householder, num_heads, value_dim = v.shape
    key_dim = k.shape[-1]
    out = torch.empty((num_tokens, num_heads, value_dim), dtype=q.dtype, device=q.device)
    if num_tokens == 0:
        return out
    steps = num_tokens * num_householder
    num_seqs = slots.shape[0]
    q_steps = q.new_empty(steps, num_heads, key_dim)
    k_steps = k.new_empty(steps, num_heads, key_dim)
    v_steps = v.new_empty(steps, num_heads, _HEAD)
    alpha_steps = torch.empty(steps, num_heads, dtype=torch.float32, device=q.device)
    beta_steps = torch.empty(steps, num_heads, dtype=torch.float32, device=q.device)
    _EXPAND[(num_tokens,)](
        q,
        k,
        v,
        beta,
        g,
        q_steps,
        k_steps,
        v_steps,
        alpha_steps,
        beta_steps,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *beta.stride(),
        *g.stride(),
        H=num_heads,
        NH=num_householder,
        K=key_dim,
        V=value_dim,
        D=_HEAD,
        BH=triton.next_power_of_2(num_heads),
        num_warps=4,
    )
    initial = torch.empty(
        (num_seqs, num_heads, _HEAD, key_dim), dtype=torch.float32, device=q.device
    )
    cu_steps = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
    _GATHER_STATE[(num_seqs, num_heads)](
        state_pool,
        slots,
        has_initial_state,
        cu_seqlens,
        initial,
        cu_steps,
        *state_pool.stride(),
        H=num_heads,
        NH=num_householder,
        V=value_dim,
        K=key_dim,
        D=_HEAD,
        num_warps=8,
    )
    o_steps = torch.empty_like(q_steps)
    final = torch.empty_like(initial)
    flashinfer.chunk_gated_delta_rule(
        q_steps,
        k_steps,
        v_steps,
        alpha_steps,
        beta_steps,
        scale=scale,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu_steps,
        output=o_steps,
        output_state=final,
        use_cp=use_cp,
    )
    out.copy_(o_steps.view(num_tokens, num_householder, num_heads, _HEAD)[:, -1, :, :value_dim])
    _SCATTER_STATE[(num_seqs, num_heads)](
        final,
        state_pool,
        slots,
        *state_pool.stride(),
        H=num_heads,
        V=value_dim,
        K=key_dim,
        D=_HEAD,
        num_warps=8,
    )
    return out


#: nemotron4-nano-preview's GDP mixer, the one shape all 32 of its GDP and linear_attention
#: layers share: gdp_num_heads 24 (== gdp_num_groups, so no head repeat), gdp_key_head_dim 128,
#: gdp_head_dim 64, gdp_num_householder 3. `scale` is what the target passes for HF's
#: `query / sqrt(key_head_dim)`.
_NEMOTRON4 = dict(heads=24, householder=3, key_dim=128, value_dim=64, scale=128**-0.5)
#: A tp4 rank runs the same mixer on 6 of the 24 heads.
_NEMOTRON4_TP4 = dict(_NEMOTRON4, heads=6)

#: gdp_chunk's batches, whose lengths cross both of FlashInfer's paths (`_use_cp` picks the
#: non-CP kernel through 512 tokens and for the 8-sequence mix, CP from 2048 tokens and for the
#: 4-sequence one), plus the two batches the serving protocol adds: a prompt's first chunk
#: beside another's 32-token tail, and two 8192-token prompts in one step (a dep4 rank's
#: 16384-token budget). A token is 3 sub-steps and a chunk of FlashInfer's 64, so chunk
#: boundaries fall inside tokens. "fresh" sequences start from zero over slots holding a stale
#: state they must not read; "continued" ones resume a saved state; "mixed" alternates the two,
#: even ones continued; "first+tail" resumes only the last.
_BATCHES = (
    ((1,), "fresh", "1 token, fresh: one partial chunk"),
    ((1,), "continued", "1 token, continued: a one-token prefill resume"),
    ((7,), "fresh", "7 tokens, fresh: 21 sub-steps"),
    ((7,), "continued", "7 tokens, continued"),
    ((64,), "fresh", "64 tokens, fresh: exactly 3 chunks"),
    ((64,), "continued", "64 tokens, continued: a short tail"),
    ((512,), "fresh", "512 tokens, fresh: the longest non-CP single"),
    ((512,), "continued", "512 tokens, continued"),
    ((2048,), "fresh", "2048 tokens, fresh: context-parallel"),
    ((2048,), "continued", "2048 tokens, continued"),
    ((8192,), "fresh", "8192 tokens, fresh: the max_num_tokens budget"),
    ((8192,), "continued", "8192 tokens, continued: a resumed prefill chunk"),
    ((16384,), "fresh", "16384 tokens, fresh: the larger budget"),
    ((16384,), "continued", "16384 tokens, continued"),
    ((3, 509, 2600, 5080), "fresh", "4 sequences summing to 8192, all fresh"),
    ((3, 509, 2600, 5080), "mixed", "4 sequences summing to 8192, even ones continued"),
    ((1, 2, 17, 64, 129, 300, 511, 1024), "fresh", "8 sequences summing to 2048, fresh"),
    ((1, 2, 17, 64, 129, 300, 511, 1024), "mixed", "8 sequences summing to 2048, mixed"),
    ((8160, 32), "first+tail", "a fresh 8160-token chunk beside a 32-token continued tail"),
    ((8192, 8192), "fresh", "two 8192-token prompts: 16384 tokens in one step"),
)


def _cells() -> tuple[Cell, ...]:
    """Every batch at both certified geometries: tensor parallelism splits heads, not tokens."""
    tp1 = tuple(
        Cell(why=why, spec=dict(seq_lens=lens, init=init, **_NEMOTRON4))
        for lens, init, why in _BATCHES
    )
    tp4 = tuple(
        Cell(why=f"tp4 rank, {why}", spec=dict(seq_lens=lens, init=init, **_NEMOTRON4_TP4))
        for lens, init, why in _BATCHES
    )
    return tp1 + tp4


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """HF `l2norm`: `x * rsqrt(sum(x * x) + 1e-6)` over the head dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


class FlashinferGdpChunk(OpWrapper):
    """`gdp_chunk`'s computation through `flashinfer.chunk_gated_delta_rule`: a varlen batch
    of prefill sequences through `torch_recurrent_gated_delta_rule`'s recurrence,
    Householder-expanded the way `Nemotron4GDPMixer.forward` expands it, each sequence starting
    from zero or from its slot's saved state and leaving its final state in that slot.

    Per sequence and head, with `S` the HF `[K, V]` state (zero unless the sequence's
    `has_initial_state` is set, then the slot's) and `l2n(x) = x * rsqrt(sum(x*x) + 1e-6)`, for
    each of its tokens in order:

        S *= exp(g)                                  # once, before the first step
        for j in range(n_h):                         # the token's Householder steps
            kj = l2n(k[j]); u = beta[j] * (v[j] - S^T kj); S += kj u^T
        out = S^T (l2n(q) * scale)                   # read after the last step

    FlashInfer's kernel runs the gated delta rule over the expanded sequence, one sub-step per
    row: the decay as a linear gate, one at every sub-step but a token's first; the query zero
    but at its last. It takes one head size, 128, for q, k and v alike, so v and the state are
    zero-padded from V to 128, and a zero column of v keeps its column of the state exactly
    zero. Returns `out` `[T, H, V]` in `q`'s dtype and leaves each sequence's final state in
    `state_pool[slots[n]]`; a caller that wants the old states has to have kept its own copy.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Inputs and state pool are `gdp_chunk`'s, argument for argument, so a target swaps one call
    for the other; `max_seqlen` is added: the longest sequence's token count as a host int,
    which the caller knows without a device round trip (the attention metadata's host
    `seq_lens`). It chooses between FlashInfer's two kernels (`_use_cp`) and moves only time:
    a wrong value costs speed, never a result, and a device tensor in its place would make
    the choice sync.

    Numerics: the l2-normalized q and k, the padded v and the output are bf16, the gates and
    the state fp32; FlashInfer keeps the state fp32 between chunks. Both paths sit within the
    bands the test derives with errors of `gdp_chunk`'s size on the same inputs (worst output
    element 13.1 against 13.8 ulp over the cells at six seeds; relative RMS and the states
    lower), and on real layer 0 and 6 inputs of four ~8K prompts their worst output element
    equals `gdp_chunk`'s (both round the normalized keys to bf16) with smaller state errors.

    Cost beyond FlashInfer's kernel: the expansion writes the query, keys and padded values
    of every sub-step, 3x the tokens at 128 wide (~0.11 ms at 8192 tokens and 24 heads), and
    the kept rows are copied out of an output 6x the size of `out`. FlashInfer compiles its
    kernels on the first call of a process for each head count (seconds, measured ~4.8 s for
    CP and ~3.7 s non-CP at 24 heads), and never again for any token count.

    No host syncs: grids and the CP choice come from shapes and `max_seqlen`. Prefill only.

    Certified head counts: 24 (tp1, a dep4 rank) and 6 (a tp4 rank). Uncertified: any other
    head count or Householder count, key_dim other than 128 (FlashInfer refuses it), value_dim
    above 128, a bf16 state pool, fp16 q/k/v, MTP verify (per-position states).
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
        max_seqlen: int,
    ) -> torch.Tensor:
        """Writes the sequences' final states into `state_pool[slots]` and returns `out`."""
        return _flashinfer_gdp_chunk_fwd(
            q,
            k,
            v,
            g,
            beta,
            state_pool,
            slots,
            cu_seqlens,
            has_initial_state,
            scale,
            _use_cp(max_seqlen, v.shape[0], slots.shape[0], v.shape[2]),
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
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, the pool as the op leaves it)`, fp32 throughout; mutates nothing.

        Transcribed from `torch_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel=True`,
        token by token over each sequence, the token's steps as `Nemotron4GDPMixer.forward`
        expands them: the decay rides on the first step only and the query is read only after
        the last. HF divides the l2-normalized query by `sqrt(key_dim)`; `scale` is that
        factor. HF starts every sequence from zero; a sequence with `has_initial_state` starts
        from its slot's state, as the cached path resumes a chunked prefill. `max_seqlen`
        chooses a kernel, not a result, and is ignored here.
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
        max_seqlen: int,
    ) -> None:
        # The expansion takes its sizes from `v` and `k` and reads q, k, g and beta through
        # raw strides, so a tensor of the wrong shape is read as whatever those sizes say
        # rather than rejected.
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
        # The padding stops at FlashInfer's head size: wider values would be cut, not refused.
        assert value_dim <= _HEAD, f"value_dim must be at most {_HEAD}; got {value_dim}"
        # fp16 inputs reach FlashInfer as another rounding, never measured here.
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16, (
            f"q, k and v must be bfloat16; got {q.dtype}, {k.dtype}, {v.dtype}"
        )
        # HF's [H, K, V] state has the same element count; read through these strides it
        # comes back transposed.
        assert tuple(state_pool.shape[1:]) == (num_heads, value_dim, key_dim), (
            f"state_pool must be [num_slots, H, V, K] = [*, {num_heads}, {value_dim}, {key_dim}] "
            f"with K innermost; got {tuple(state_pool.shape)}"
        )
        # A bf16 pool would round the recurrent state at every prefill chunk; HF keeps it
        # in fp32.
        assert state_pool.dtype == torch.float32, (
            f"state_pool holds the recurrent state in float32; got {state_pool.dtype}"
        )
        # HF computes the decay in fp32; exp() of a bf16-rounded decay is off by up to 2^-9
        # relative per token, and the state compounds it.
        assert g.dtype == torch.float32, f"g (the log decay) must be float32; got {g.dtype}"
        # A positive g grows the state instead of decaying it.
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
        # A negative slot indexes from the end of the pool: PAD_SLOT_ID would read and
        # overwrite the last slot rather than be skipped.
        assert slots.shape == (num_seqs,) and not slots.is_floating_point(), (
            f"slots must be one integer per sequence; got {slots.dtype} {tuple(slots.shape)}"
        )
        assert bool(((slots >= 0) & (slots < state_pool.shape[0])).all()), (
            f"slots must be in [0, {state_pool.shape[0]}): prefill rows have no padding slot"
        )
        assert slots.unique().numel() == num_seqs, (
            "two sequences share a slot; the last one written wins"
        )


flashinfer_gdp_chunk = FlashinferGdpChunk()
