# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum every rank's partial and write the sum into the wide residual, then read it: one MNNVL op."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import allreduce
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.wide_residual import (
    wide_residual,
)

# The checkpoint's wide residual: 3 streams of its 3072-wide hidden state, every norm at
# `rms_norm_eps` 1e-5.
_GEOMETRY = dict(streams=3, hidden=3072, eps=1e-5)

# The rows tp4 sends through a mixer's all-reduce in one step.
_ROWS = (
    (1, "decode at C = 1"),
    (3, "one request's MTP verify (1 + 2 drafts)"),
    (4, "decode at C = 4"),
    (12, "MTP verify at C = 4"),
    (16, "decode at C = 16"),
    (42, "the most rows the one-shot kernel takes at width 3072 on 4 ranks"),
    (43, "the fewest rows the two-shot kernel takes at width 3072 on 4 ranks"),
    (48, "MTP verify at C = 16"),
    (64, "the largest decode graph"),
    (1024, "a 1024-token prefill chunk"),
    (8192, "a full 8192-token prefill chunk, tp4's max_num_tokens"),
)

# The two calls a mixer's all-reduce feeds (`_streams_step` right after it).
_READS = (
    (1, "a shortcut block's shared read, or the next GDP / MLP block's input"),
    (2, "the next shortcut block's routed and mixer reads"),
)


class AllreduceWideResidual(OpWrapper):
    """`allreduce` then `wide_residual` with a write, as one op: the sum of every rank's `input`
    is the update the streams are written with, and is never stored on its own.

    The sum is `allreduce`'s, bit for bit: fp32, ascending rank order, one rounding. The write and
    the reads are `wide_residual`'s: `S_k <- bf16(bf16(r_k * S_k) + w_k * sum)` in place, then
    each read `x = bf16(sum_k f_k * S_k)` returned as `bf16(g * bf16(x * rsqrt(mean(x^2) + eps)))`.
    Returns `(streams, normed, normed2)`, `normed2` None for one read; the reads are new tensors.

    The streams come back bit-identical to the unfused pair's. A normed read differs from the
    pair's only where its variance, summed in another order, moves a value across a bf16 rounding
    boundary. Each row is one CTA of `hidden / 8` threads at every row count, on the one-shot and
    the two-shot path alike, so a row's outputs do not depend on the rest of the call, bit for
    bit, and a captured CUDA graph replays the eager call bit for bit.

    `comm_buffer` and `buffer_flags` are the TP group's MNNVL workspace, as `allreduce` takes it.
    The op runs the one-shot kernel with the step as its epilogue while `rows * hidden * ranks *
    2` is at most 1 MiB (`allreduce`'s own switch), and above that the two-shot all-reduce then a
    second kernel that waits for each row's sum and runs the step.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"T={rows}, write + {reads} read{'s' if reads == 2 else ''}: {rwhy}; {why_reads}",
            spec=dict(tokens=rows, reads=reads, dtype=torch.bfloat16, **_GEOMETRY),
        )
        for rows, rwhy in _ROWS
        for reads, why_reads in _READS
    )

    note = """
    Certified at tp4's world size, 4 ranks over MPI, on one GB300 node. The op also launches for 2
    and 8 ranks; neither is a cell.

    The op takes bf16 only, 3 streams only, a hidden width that is a multiple of 8 up to 3072, and
    bf16 coefficient vectors and norm weights, and refuses anything else loudly. `wide_residual`
    also takes fp32 coefficients and rounds them; here the caller passes them as they are stored.

    PDL follows `TRTLLM_ENABLE_PDL` (default on), as every MNNVL kernel does; it changes
    scheduling, not results, and is not a cell axis. Under it the norm weights and coefficients
    are read before the kernel ahead of the call has finished, so that kernel must not write them;
    nothing in a forward does.

    Every rank must issue the same calls, in the same order, at the same shapes, with every other
    call on the same workspace: the Lamport protocol cannot see a disagreement, and one rank cannot
    check what the others do. Plain `allreduce` calls may share the workspace in between.
    """

    def __call__(
        self,
        input: torch.Tensor,
        streams: torch.Tensor,
        retention: torch.Tensor,
        write: torch.Tensor,
        read_factors: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        read_factors2: Optional[torch.Tensor],
        norm_weight2: Optional[torch.Tensor],
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Writes `streams` in place and returns it -- the op destroys the streams' prior contents,
        so a caller that wants them has to have kept its own copy."""
        reads = torch.ops.trtllm.mnnvl_allreduce_wide_residual(
            input,
            streams,
            retention,
            write,
            read_factors,
            norm_weight,
            eps,
            read_factors2,
            norm_weight2,
            comm_buffer,
            buffer_flags,
        )
        return streams, reads[0], reads[1] if len(reads) == 2 else None

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        streams: torch.Tensor,
        retention: torch.Tensor,
        write: torch.Tensor,
        read_factors: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        read_factors2: Optional[torch.Tensor] = None,
        norm_weight2: Optional[torch.Tensor] = None,
        comm_buffer: Optional[torch.Tensor] = None,
        buffer_flags: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """The unfused pair's math: `allreduce.reference` of the group's inputs, then
        `wide_residual.reference` with that sum as the update; `streams` is not touched.

        The collective exception to mirroring `__call__`: the first argument is the group's inputs
        in ascending rank order, since a rank's output is not a function of its own. The workspace
        arguments keep their names and are ignored.
        """
        update = allreduce.reference(per_rank_inputs)
        return wide_residual.reference(
            streams,
            update,
            retention,
            write,
            read_factors,
            norm_weight,
            eps,
            read_factors2,
            norm_weight2,
        )

    def is_valid(
        self,
        input: torch.Tensor,
        streams: torch.Tensor,
        retention: torch.Tensor,
        write: torch.Tensor,
        read_factors: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        read_factors2: Optional[torch.Tensor],
        norm_weight2: Optional[torch.Tensor],
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
    ) -> None:
        # The op refuses wrong dtypes, shapes, strides and alignments itself. It refuses neither the
        # workspace conditions `allreduce` guards, nor an input that shares memory with the streams:
        # the one-shot kernel loads a row's input before it writes that row's streams, but another
        # row's program may already have written them.
        allreduce.is_valid(input, comm_buffer, buffer_flags)
        lo, hi = input.data_ptr(), input.data_ptr() + input.numel() * input.element_size()
        s_lo = streams.data_ptr()
        s_hi = s_lo + (streams.shape[0] - 1) * streams.stride(0) * streams.element_size()
        s_hi += streams.shape[1] * streams.element_size()
        assert input.numel() == 0 or streams.numel() == 0 or hi <= s_lo or s_hi <= lo, (
            "input overlaps the streams the op writes in place; another row's program may write "
            "them before this row's input is read"
        )


allreduce_wide_residual = AllreduceWideResidual()
