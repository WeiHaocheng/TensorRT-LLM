# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDP mixer's short causal conv over a varlen context batch on TensorRT-LLM's Triton varlen
kernel layout, with fp32 products: channel-last in, channel-last out, per-slot conv state."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .._launch import Launch
from .._op import Arch, Cell, OpWrapper

# One geometry per target width, as for `causal_conv1d_varlen`: every GDP layer builds the same
# `Nemotron4GDPMixer`, whose conv runs over 3 * 1536 (V) + 4 * 3072 (K and Q) = 16896 channels,
# width 4 (`gdp_conv_kernel`), no bias, SiLU. x is the transpose of the in_proj output's conv
# columns [T, 18528] = [z 1536 | conv 16896 | beta 72 | decay 24]; each slot of the state pool
# is one block holding the GDP recurrent state (24 x 64 x 128 fp32) and then this conv state.
_GDP = dict(
    dim=16896,
    width=4,
    dtype=torch.bfloat16,
    proj_width=18528,
    proj_offset=1536,
    pool_ssm_bytes=24 * 64 * 128 * 4,
    silu_activation=True,
)
# A tp4 rank's 6 GDP heads: 4224 channels, in_proj [T, 4632] = [z 384 | conv 4224 | b 18 | a 6].
_GDP_TP4 = dict(_GDP, dim=4224, proj_width=4632, proj_offset=384, pool_ssm_bytes=6 * 64 * 128 * 4)

#: Tokens per program. Channels per program and warps follow the width: 512 channels on 4 warps
#: from `_WIDE` channels up, 128 on 2 below, which gives both certified widths 33 channel blocks.
#: Measured on GB300 at 8192 fresh tokens (survey-small-ops) over 4 to 32 tokens x 128 to 512
#: channels x 2 to 8 warps: the fastest at 16896 channels (153 us) and at 4224 (47 us); 16 x
#: 256 on 4 warps, one choice for both, costs 4% and 15%.
_BLOCK_M = 16
_WIDE = 8192


# TensorRT-LLM's `_causal_conv1d_fwd_kernel` (`modules/mamba/causal_conv1d_triton.py`), copied
# statement for statement with one change: each tap's product is formed in fp32 (`tl.fma` on the
# operands converted to fp32), where that kernel multiplies the bf16 operands in bf16 and adds the
# bf16 product. The copy is verbatim otherwise: the kernel's speed is sensitive to its form, and a
# width-4 rewrite of the same algorithm measured 7% slower at 16896 channels and 16% at 4224
# (survey-small-ops). One program per (sequence, BLOCK_M tokens, BLOCK_N channels); the sequence's
# chunk 0 is the only program that reads or writes its slot.
@triton.jit()
def _triton_causal_conv1d_varlen_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # No cached prefix: start the convolution window from zeros.
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if state_len <= seqlen:  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                # need this due to the bug in tl.where not enforcing this when data is the result of
                # another tl.load
                tl.debug_barrier()
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc = tl.fma(matrix_x.to(tl.float32), matrix_w.to(tl.float32), acc)  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


#: The kernel launched straight to its compiled form once compiled (`Launch`).
_TRITON_CAUSAL_CONV1D_VARLEN = Launch(_triton_causal_conv1d_varlen_kernel)


def _cell(why: str, seq_lens: tuple[int, ...], init: str, geometry: dict = _GDP) -> Cell:
    """`init`: "fresh" (no sequence resumes), "continued" (every one does), "mixed" (even ones do)."""
    return Cell(why=why, spec=dict(seq_lens=seq_lens, init=init, **geometry))


class TritonCausalConv1dVarlen(OpWrapper):
    """Depthwise causal conv + SiLU over each sequence of a packed context batch, on Triton.

    What `causal_conv1d_varlen` computes, argument for argument: for channel `c` and token `t` of
    one sequence, `out[c, t] = silu(sum_k weight[c, k] * x[c, t - 3 + k])`, the 3 inputs before
    a sequence's first token being `conv_states[slot]` when its `has_initial_state` is set and
    zero otherwise; afterwards `conv_states[slot]` holds the last 3 inputs of [that state, the
    sequence], oldest first. `x` is `[dim, total_tokens]`, sequences concatenated as
    `query_start_loc` (int32, `[n + 1]`, from 0) delimits them; sequence `i` owns slot
    `cache_indices[i]`. Writes `out` and the batch's slots in place.

    The grid covers the batch's token count for every sequence and a chunk past its sequence
    returns at once, so the launch needs no host copy of the lengths and is capturable in a
    CUDA graph.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `causal_conv1d_varlen`'s cells, so the two entries are certified over the same batches:
    # single sequences from 1 token (shorter than the state) to the 16384-token step, fresh and
    # continued, and two varlen batches with sequences shorter than the state next to long ones,
    # at tp1's 16896 channels and a tp4 rank's 4224. 16-token chunks: 1 and 7 tokens are one
    # partial chunk, 64 four whole ones; the batches put chunk 0 of a 1- or 2-token sequence,
    # which reads and writes its slot in one program, beside multi-chunk ones.
    CELLS: tuple[Cell, ...] = tuple(
        _cell(f"{prefix}{why}", lens, init, geo)
        for prefix, geo in (("", _GDP), ("tp4 rank, ", _GDP_TP4))
        for why, lens, init in (
            ("1 token, fresh: shorter than the 3-input state", (1,), "fresh"),
            ("1 token, continued: the saved state shifts by one", (1,), "continued"),
            ("7 tokens, fresh: one partial chunk", (7,), "fresh"),
            ("7 tokens, continued: the state seeds 3 outputs", (7,), "continued"),
            ("64 tokens, fresh: four chunks, a halo between each", (64,), "fresh"),
            ("64 tokens, continued", (64,), "continued"),
            ("512 tokens, fresh", (512,), "fresh"),
            ("512 tokens, continued", (512,), "continued"),
            ("2048 tokens, fresh", (2048,), "fresh"),
            ("2048 tokens, continued", (2048,), "continued"),
            ("8192 tokens, fresh: max_num_tokens", (8192,), "fresh"),
            ("8192 tokens, continued: a resumed prefill chunk", (8192,), "continued"),
            ("16384 tokens, fresh: the larger prefill budget", (16384,), "fresh"),
            ("16384 tokens, continued", (16384,), "continued"),
            ("4 sequences summing to 8192, all fresh", (3, 509, 2600, 5080), "fresh"),
            ("4 sequences summing to 8192, even ones continue", (3, 509, 2600, 5080), "mixed"),
            (
                "8 sequences summing to 2048, all fresh",
                (1, 2, 17, 64, 129, 300, 511, 1024),
                "fresh",
            ),
            (
                "8 sequences summing to 2048, even ones continue",
                (1, 2, 17, 64, 129, 300, 511, 1024),
                "mixed",
            ),
        )
    )

    note = """
    Certified: width 4, no bias, SiLU on; x the in_proj output's conv columns transposed
    (channel stride 1, token stride 18528 at tp1 or 4632 at a tp4 rank) and an explicit
    channel-last `out`; every sequence owning a real slot; the packed per-slot state block of
    `CppMambaHybridCacheManager`. The kernel reads every tensor through its strides, so other
    layouts compute the same values but are uncertified. Widths other than 4, a bias, SiLU off,
    no pool or no `query_start_loc` raise ValueError. `pad_slot_id` entries are skipped (their
    tokens and slots untouched) but uncertified.

    Accuracy: the products and sums are `causal_conv1d_varlen`'s (fp32, fma chain from +0); SiLU's
    division is the fast one TensorRT-LLM's Triton kernel uses rather than the CUDA kernel's
    correctly rounded reciprocal, which moves an fp32 value by a few ulps. Measured over every cell
    at five seeds (survey-small-ops): states exact; at most 2.4e-4 of a cell's outputs differ from
    the fp32 reference (one output), 3.7e-6 to 3.4e-5 elsewhere, each by one bf16 ulp of its own
    value -- `causal_conv1d_varlen`'s level. TensorRT-LLM's kernel as shipped, with bf16 products,
    puts 37% of them one ulp off (relative RMS 0.70 ulp); a correctly rounded division instead ran
    29% slower (211 against 163 us at 8192 tokens, 16896 channels, the shipped tiles).

    Compile-time constants, as in TensorRT-LLM's kernel: the width, the strides, the pool's slot
    count and the padding id, so one compile per layer geometry and pool.

    Time on GB300 (survey-small-ops, eager GPU time per call, against `causal_conv1d_varlen`):
    16896 channels, 153 us against 248 at 8192 tokens, 295 against 487 at 16384, 163 against 252
    for a 4-sequence 8192-token batch (whose chunks past each sequence's end cost 4% against a
    grid sized from host lengths); a tp4 rank's 4224, 46 against 78, 84 against 149 and 53 against
    98. About 2x the bytes' bound at the HBM copy rate; SiLU's arithmetic over 138M elements is
    most of the rest. Host issue time through `JITFunction.run` is 22 to 27 us eager against the
    C++ op's 8 to 9 us (both with a preallocated `out`); the entry launches through `Launch`,
    which skips `run` once the call's specialization is compiled.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias_: torch.Tensor | None,
        conv_states: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        cache_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        silu_activation: bool,
        pad_slot_id: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Writes `out` and the batch's slots of `conv_states` in place and returns `out`. The
        slots' prior contents are destroyed, so a caller that wants them has to have kept its own
        copy beforehand."""
        dim, total = x.shape
        if weight.shape[-1] != 4 or bias_ is not None or not silu_activation:
            raise ValueError("only the certified form runs: width 4, no bias, SiLU on")
        if conv_states is None or query_start_loc is None or cache_indices is None:
            raise ValueError("conv_states, query_start_loc and cache_indices are required")
        if out is None:
            out = torch.empty(total, dim, dtype=x.dtype, device=x.device).t()
        if has_initial_state is None:
            has_initial_state = torch.zeros(
                cache_indices.shape[0], dtype=torch.bool, device=x.device
            )
        num_seqs = query_start_loc.shape[0] - 1
        if total == 0 or num_seqs == 0:
            return out
        wide = dim >= _WIDE
        block_n, warps = (512, 4) if wide else (128, 2)
        _TRITON_CAUSAL_CONV1D_VARLEN[
            (num_seqs, triton.cdiv(total, _BLOCK_M), triton.cdiv(dim, block_n))
        ](
            x,
            weight,
            None,
            conv_states,
            cache_indices,
            has_initial_state,
            query_start_loc,
            out,
            dim,
            total,
            conv_states.shape[0],
            0,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            conv_states.stride(0),
            conv_states.stride(1),
            conv_states.stride(2),
            0,
            out.stride(0),
            out.stride(1),
            pad_slot_id,
            HAS_BIAS=False,
            KERNEL_WIDTH=4,
            SILU_ACTIVATION=True,
            HAS_INITIAL_STATES=True,
            HAS_CACHE=True,
            IS_CONTINUOUS_BATCHING=True,
            USE_PAD_SLOT=True,
            NP2_STATELEN=4,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=block_n,
            num_warps=warps,
            num_stages=2,
        )
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias_: torch.Tensor | None,
        conv_states: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        cache_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        silu_activation: bool,
        pad_slot_id: int,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, conv_states after the call)`, in fp32 and rounded once.

        Returns rather than mutates: the first a new `[dim, total_tokens]` tensor, the second a
        copy of the whole pool with the batch's slots replaced. `bias_`, `silu_activation` and
        `out` are taken for the signature; the certified form has no bias and SiLU on.
        """
        width = weight.shape[-1]
        taps = weight.float()
        starts = query_start_loc.tolist()
        slots = cache_indices.tolist()
        resume = [False] * len(slots) if has_initial_state is None else has_initial_state.tolist()
        y = torch.zeros(weight.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
        states = conv_states.clone()
        for i in range(len(starts) - 1):
            begin, end = starts[i], starts[i + 1]
            if slots[i] == pad_slot_id:
                continue
            prior = (
                conv_states[slots[i]].float()
                if resume[i]
                else torch.zeros(weight.shape[0], width - 1, dtype=torch.float32, device=x.device)
            )
            window = torch.cat([prior, x[:, begin:end].float()], dim=1)
            length = end - begin
            acc = sum(taps[:, k : k + 1] * window[:, k : k + length] for k in range(width))
            y[:, begin:end] = F.silu(acc)
            states[slots[i]] = window[:, length:].to(states.dtype)
        return y.to(x.dtype), states

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias_: torch.Tensor | None,
        conv_states: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        cache_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        silu_activation: bool,
        pad_slot_id: int,
        out: torch.Tensor | None = None,
    ) -> None:
        # Only what the op takes and answers wrongly; the forms it does not run raise in
        # __call__. The kernel reads every tensor through its strides and checks nothing.
        dim, total = x.shape
        starts = query_start_loc.tolist()
        assert (
            starts[0] == 0
            and starts[-1] == total
            and all(a <= b for a, b in zip(starts, starts[1:]))
        ), (
            f"query_start_loc must rise from 0 to x's {total} tokens, got {starts}: the op reads "
            "the sequences where it says and never checks it against x"
        )
        assert tuple(weight.shape) == (dim, 4), (
            f"weight must be [{dim}, 4]: the kernel reads four taps per channel of x; got "
            f"{tuple(weight.shape)}"
        )
        assert conv_states.dim() == 3 and tuple(conv_states.shape[1:]) == (dim, 3), (
            f"conv_states must be [slots, dim, 3] = [slots, {dim}, 3], got "
            f"{tuple(conv_states.shape)}: the op addresses it through its strides alone"
        )
        if out is not None:
            assert tuple(out.shape) == (dim, total), (
                f"out must be [{dim}, {total}] like x; got {tuple(out.shape)}"
            )
        for name, t in (("x", x), ("out", out)):
            if t is not None:
                last = (total - 1) * t.stride(1) + (dim - 1) * t.stride(0)
                assert last < 2**31, (
                    f"{name}'s last element sits {last} elements in: the kernel forms token "
                    "offsets in int32, so past 2**31 - 1 they wrap and it reads or writes another "
                    "token's memory"
                )
        slots = cache_indices.tolist()
        assert len(slots) == len(starts) - 1, (
            f"{len(slots)} slots for {len(starts) - 1} sequences: the kernel reads one per sequence"
        )
        live = [s for s in slots if s != pad_slot_id]
        assert all(0 <= s < conv_states.shape[0] for s in live), (
            f"slots {live} reach outside conv_states' {conv_states.shape[0]} slots: "
            "the op reads and writes there unchecked"
        )
        assert len(set(live)) == len(live), (
            f"slots {live} repeat: sequences sharing a slot race to read and write it"
        )


triton_causal_conv1d_varlen = TritonCausalConv1dVarlen()
