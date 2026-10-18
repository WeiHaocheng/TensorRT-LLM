# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDP mixer's short causal conv over a varlen context batch, with per-slot conv state."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

# Every GDP layer of the checkpoint (12 `linear_attention`, 20 `shortcut_moe_gdp`) builds the same
# `Nemotron4GDPMixer`, so there is one geometry: conv_dim = 3 * 1536 (V) + 4 * 3072 (K and Q)
# = 16896 channels, width 4 (`gdp_conv_kernel`), no bias, SiLU. The input is the target's
# in_proj output [T, 18528] = [z 1536 | conv 16896 | beta 72 | decay 24], passed as the
# transpose of its conv columns -- channel-last, token stride 18528 -- so no copy runs. Each
# slot of the state pool is one block holding the GDP recurrent state (24 x 64 x 128 fp32) and
# then this conv state, as `CppMambaHybridCacheManager._setup_states` packs them, so the conv
# view a target passes is strided per slot with dense inner dims (the view
# `MambaHybridCacheManagerV2._get_state_buffer` exposes has the same form). The cells build
# exactly those two views.
_GDP = dict(
    dim=16896,
    width=4,
    dtype=torch.bfloat16,
    proj_width=18528,
    proj_offset=1536,
    pool_ssm_bytes=24 * 64 * 128 * 4,
    silu_activation=True,
)


def _cell(why: str, seq_lens: tuple[int, ...], init: str) -> Cell:
    """`init`: "fresh" (no sequence resumes), "continued" (every one does), "mixed" (even ones do)."""
    return Cell(why=why, spec=dict(seq_lens=seq_lens, init=init, **_GDP))


class CausalConv1dVarlen(OpWrapper):
    """Depthwise causal conv + SiLU over each sequence of a packed context batch.

    For channel `c` and token `t` of one sequence, with `W` the width:
    `out[c, t] = silu(sum_k weight[c, k] * x[c, t - (W - 1) + k] (+ bias_[c]))`. The `W - 1`
    inputs before a sequence's first token are `conv_states[slot]` when its `has_initial_state`
    is set and zero otherwise. Afterwards `conv_states[slot]` holds the last `W - 1` inputs of
    the state it started from followed by the sequence -- its own last `W - 1` tokens once it has
    that many -- oldest first: the state the request's next chunk or decode step resumes from.

    `x` is `[dim, total_tokens]`, sequences concatenated along the tokens as `query_start_loc`
    (int32, `[n + 1]`, from 0) delimits them; sequence `i` owns slot `cache_indices[i]`. The
    kernel is chosen by the layout: `x.stride(0) == 1` -- the transpose of a `[tokens, channels]`
    activation -- runs the channel-last kernel, which needs an explicit, non-overlapping
    channel-last `out`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The batches of the GDP recurrence entries, so a chunk of prefill is certified end to end.
    # Single sequences run on 32-token chunks up to 2048 tokens and on 64-token chunks from 8192
    # (`causal_conv1d_channellast_fwd_dispatch` takes 64 once channel blocks x 64-token chunks x
    # sequences reach 1024); the 4- and 8-sequence batches run on 64-token chunks and put
    # sequences shorter than the 3-input state next to long ones.
    CELLS: tuple[Cell, ...] = (
        _cell("1 token, fresh: shorter than the 3-input state", (1,), "fresh"),
        _cell("1 token, continued: the saved state shifts by one", (1,), "continued"),
        _cell("7 tokens, fresh: one unrolled step and a remainder", (7,), "fresh"),
        _cell("7 tokens, continued: the state seeds 3 outputs", (7,), "continued"),
        _cell("64 tokens, fresh: two chunks, one halo between", (64,), "fresh"),
        _cell("64 tokens, continued", (64,), "continued"),
        _cell("512 tokens, fresh: sixteen chunks", (512,), "fresh"),
        _cell("512 tokens, continued", (512,), "continued"),
        _cell("2048 tokens, fresh: the longest on 32-token chunks", (2048,), "fresh"),
        _cell("2048 tokens, continued", (2048,), "continued"),
        _cell("8192 tokens, fresh: max_num_tokens, 64-token chunks", (8192,), "fresh"),
        _cell("8192 tokens, continued: a resumed prefill chunk", (8192,), "continued"),
        _cell("16384 tokens, fresh: the larger prefill budget", (16384,), "fresh"),
        _cell("16384 tokens, continued", (16384,), "continued"),
        _cell("4 sequences summing to 8192, all fresh", (3, 509, 2600, 5080), "fresh"),
        _cell("4 sequences summing to 8192, even ones continue", (3, 509, 2600, 5080), "mixed"),
        _cell(
            "8 sequences summing to 2048, all fresh: 1 and 2 tokens among them",
            (1, 2, 17, 64, 129, 300, 511, 1024),
            "fresh",
        ),
        _cell(
            "8 sequences summing to 2048, even ones continue",
            (1, 2, 17, 64, 129, 300, 511, 1024),
            "mixed",
        ),
    )

    note = """
    Conv state: one layer's `conv_states` is a `[slots, dim, W - 1]` view in x's dtype, oldest
    input first -- the layout `get_conv_states(layer_idx)` of the Mamba cache managers returns
    (`[slot_size, conv_dim, d_conv - 1]`). The op addresses it through its strides alone, so a
    view strided per slot is fine. Pass the pool's own view, never a `.contiguous()` copy: the
    write-back would land in the copy. A sequence whose `has_initial_state` is False ignores
    its slot's old contents and still writes its final state there.

    Certified: the channel-last kernel, reached by passing the in_proj output's conv columns
    transposed, with an explicit channel-last `out`, no bias, SiLU on, and every sequence owning
    a real slot. Not certified, because no target here reaches it: the channel-major kernel
    (a token-contiguous `[dim, T]` x, which also runs in place), the batched `(B, dim, L)` form
    without `query_start_loc`, a bias, `pad_slot_id` entries, a zero-token batch, widths 2 and 3,
    and a row stride or offset off the 16-byte grid (the dispatcher then falls back to scalar
    accesses).

    `TRTLLM_CAUSAL_CONV1D_SILU_MODE` selects the SiLU flavour, once per process. The cells are
    certified at the default (mode 1, the exact form); modes 2 and 3 are opt-in approximations.

    The checkpoint's `Nemotron4GDPMixer` rounds the conv output to bf16 before its SiLU; this op
    applies SiLU to the fp32 accumulator and rounds once. Measured over a dense grid of
    pre-activations on [-16, 16], HF's extra rounding moves the result by at most one bf16 ulp
    of the output between -3.2 and 2, by up to two from 2 or below -3.2, and by up to eight below
    -4, where SiLU shrinks the value. A check against an HF-faithful reference has to allow for
    that.
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
        """Writes `out` (x itself when `out` is None) and the batch's slots of `conv_states` in
        place, and returns the tensor it wrote. The slots' prior contents are destroyed, so a
        caller that wants them has to have kept its own copy beforehand."""
        torch.ops.trtllm.causal_conv1d_fwd(
            x,
            weight,
            bias_,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            silu_activation,
            pad_slot_id,
            out,
        )
        return x if out is None else out

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`(out, conv_states after the call)`, in fp32 and rounded once.

        Returns rather than mutates, so a cell can drive both sides from the same inputs: the
        first is a new `[dim, total_tokens]` tensor, the second a copy of the whole pool with the
        batch's slots replaced (None without a pool), which lets a test see that nothing outside
        those slots moved. A padded entry (`cache_indices[i] == pad_slot_id`) keeps the
        destination's rows as they are now and its slot untouched. Covers the varlen form only:
        `query_start_loc` is required.
        """
        assert query_start_loc is not None, "the reference covers the varlen form only"
        dim, width = weight.shape
        taps = weight.float()
        bias = None if bias_ is None else bias_.float()
        starts = query_start_loc.tolist()
        num_seqs = len(starts) - 1
        slots = list(range(num_seqs)) if cache_indices is None else cache_indices.tolist()
        resume = [False] * num_seqs if has_initial_state is None else has_initial_state.tolist()
        dst = x if out is None else out

        y = torch.empty(dim, x.shape[-1], dtype=torch.float32, device=x.device)
        states = None if conv_states is None else conv_states.clone()
        for i in range(num_seqs):
            begin, end = starts[i], starts[i + 1]
            if slots[i] == pad_slot_id:
                y[:, begin:end] = dst[:, begin:end].float()
                continue
            if resume[i] and conv_states is not None:
                prior = conv_states[slots[i]].float()
            else:
                prior = torch.zeros(dim, width - 1, dtype=torch.float32, device=x.device)
            window = torch.cat([prior, x[:, begin:end].float()], dim=1)
            length = end - begin
            acc = sum(taps[:, k : k + 1] * window[:, k : k + length] for k in range(width))
            if bias is not None:
                acc = acc + bias[:, None]
            y[:, begin:end] = F.silu(acc) if silu_activation else acc
            if states is not None:
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
        # Only what the op takes and answers wrongly. The dtypes of bias_, conv_states and out,
        # the devices, the shapes of x, weight, bias_, cache_indices and has_initial_state, and
        # a channel-last x without a separate channel-last out it rejects itself.
        dim, width = weight.shape[0], weight.shape[-1]
        assert 2 <= width <= 4, (
            f"width {width} has no kernel: the op launches nothing and returns with out unwritten"
        )
        assert weight.dtype == x.dtype, (
            f"weight is {weight.dtype} but x is {x.dtype}: the op instantiates both on x's dtype "
            "and reinterprets weight's bytes"
        )
        assert x.stride(-2) == 1 or x.stride(-1) == 1, (
            f"x has strides {tuple(x.stride())}: without a unit-stride channel or token axis the "
            "channel-major kernel reads each channel's tokens as if contiguous"
        )
        starts = None if query_start_loc is None else query_start_loc.tolist()
        if starts is not None:
            assert (
                starts[0] == 0
                and starts[-1] == x.shape[-1]
                and all(a <= b for a, b in zip(starts, starts[1:]))
            ), (
                f"query_start_loc must rise from 0 to x's {x.shape[-1]} tokens, got {starts}: "
                "the op reads the sequences where it says and never checks it against x"
            )
        num_seqs = x.shape[0] if starts is None else len(starts) - 1
        if conv_states is None:
            assert has_initial_state is None or not bool(has_initial_state.any()), (
                "has_initial_state asks for saved state but conv_states is None: "
                "the op starts those sequences from zeros"
            )
            return
        assert conv_states.dim() == 3 and tuple(conv_states.shape[1:]) == (dim, width - 1), (
            f"conv_states must be [slots, dim, width - 1] = [slots, {dim}, {width - 1}], got "
            f"{tuple(conv_states.shape)}: the op addresses it through its strides alone"
        )
        slots = list(range(num_seqs)) if cache_indices is None else cache_indices.tolist()
        live = [s for s in slots if s != pad_slot_id]
        assert all(0 <= s < conv_states.shape[0] for s in live), (
            f"slots {live} reach outside conv_states' {conv_states.shape[0]} slots: "
            "the op reads and writes there unchecked"
        )
        assert len(set(live)) == len(live), (
            f"slots {live} repeat: sequences sharing a slot race to read and write it"
        )


causal_conv1d_varlen = CausalConv1dVarlen()
