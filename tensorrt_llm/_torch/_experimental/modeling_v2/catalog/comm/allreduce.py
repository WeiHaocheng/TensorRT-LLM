# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum every rank's partial on every rank, via the trtllm MNNVL all-reduce op."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm.functional import AllReduceFusionOp

from .._op import Arch, Cell, OpWrapper

# The rows tp4 reduces per call. Up to 16 rows, the decode step's batch at
# concurrency C = 1 ... 16; from 64, prefill chunks up to a 16384-token step.
_ROWS = (
    (1, "decode at C = 1, the floor every row-parallel layer pays per step"),
    (2, "decode at C = 2"),
    (4, "decode at C = 4"),
    (8, "decode at C = 8"),
    (16, "decode at C = 16"),
    (64, "a short prefill"),
    (256, "a 256-token prefill chunk"),
    (1024, "a 1024-token prefill chunk"),
    (8192, "a full 8192-token prefill chunk"),
    (16384, "a 16384-token step, two 8192-token prompts prefilled together"),
)

# The widths tp4's row-parallel layers produce: the hidden size (GDP out_proj,
# attention o_proj, the dense MLP's down projection); the MoE block's latent
# partial (the routed experts' sliced output, in latent space); and the MoE
# block's latent and shared-expert partials packed into one call.
_WIDTHS = (
    (3072, "hidden"),
    (768, "MoE latent partial"),
    (3840, "MoE latent + shared-expert partials packed"),
)


class Allreduce(OpWrapper):
    """`sum over ranks of input`, on every rank, as a new tensor; `input` is left untouched.

    How the sum is formed, which is what the cells certify bit for bit: in
    fp32, in ascending rank order, starting from +0.0, rounded once to
    `input.dtype` -- `round((((+0.0 + x_0) + x_1) + x_2) + x_3)`. Every rank
    applies that same formula, so every rank receives the same bits; and every
    element is reduced on its own, so a row's sum depends neither on how many
    rows share the call nor on where the row sits among them. A CUDA graph
    captured at one height and an eager call at another give a row the same
    bits.

    `comm_buffer` and `buffer_flags` are the group's MNNVL workspace, which
    also fixes the group: `ws = get_or_scale_allreduce_mnnvl_workspace(mapping,
    dtype, buffer_size_bytes=...)`, built once and outside any graph capture,
    then `ws["uc_buffer"].view(dtype).view(3, -1)` and `ws["buffer_flags"]`.
    The op runs its one-shot kernel while `rows * width * ranks * itemsize` is
    at most 1 MiB and its two-shot kernel above that; both form the sum the
    same way, and the cells sit on both sides of the switch.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"[{rows}, {width}] {wname}: {rwhy}",
            spec=dict(rows=rows, width=width, dtype=torch.bfloat16),
        )
        for width, wname in _WIDTHS
        for rows, rwhy in _ROWS
    )

    note = """
    Pinned to MNNVL rather than the custom ONESHOT / TWOSHOT kernels or NCCL.
    ONESHOT and TWOSHOT form the sum the same way, but
    `torch.ops.trtllm.allreduce` silently hands any message over 64 MiB
    (`TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE`) to NCCL, whose sum is formed
    differently and depends on the message size: a T-invariant forward would
    stop being one at a large prefill. TWOSHOT is also measured to take hundreds
    of microseconds on decode-sized messages. The per-strategy timings live with
    the run that measured them.

    The workspace must hold the largest message the target sends:
    `MNNVLAllReduce.get_required_workspace_size(rows, width, tp_size, dtype)`
    bytes per stage, sized for the most rows at the widest width. The op does
    not check; `is_valid` does.

    Every rank must issue the same calls, in the same order, at the same
    shapes. The workspace's Lamport protocol has no way to see a disagreement,
    and one rank cannot check what the others do.

    -0.0 is the protocol's empty-slot marker, so the kernel reads a -0.0 input
    as +0.0 and a sum of zeros is +0.0. `reference` starts from +0.0, which
    reproduces that.

    Not cells: the fused patterns this op also runs (residual + RMSNorm, with or
    without FP8 / NVFP4 quantization), dtypes other than bf16, and a workspace
    built over a torch ProcessGroup (`TLLM_DISABLE_MPI=1`). The matrix runs MPI
    ranks, which is what a tp4 serving engine starts.
    """

    def __call__(
        self, input: torch.Tensor, comm_buffer: torch.Tensor, buffer_flags: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.trtllm.mnnvl_fusion_allreduce(
            input=input,
            gamma=None,
            residual=None,
            epsilon=None,
            comm_buffer=comm_buffer,
            buffer_flags=buffer_flags,
            rmsnorm_fusion=False,
            scale=None,
            fusion_op=int(AllReduceFusionOp.NONE),
        )[0]

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        comm_buffer: Optional[torch.Tensor] = None,
        buffer_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The sum the collective is supposed to make, formed the way the cells certify.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds, so this takes the
        group's inputs in ascending rank order. The workspace arguments keep
        their names and are ignored.
        """
        acc = torch.zeros(
            per_rank_inputs[0].shape, dtype=torch.float32, device=per_rank_inputs[0].device
        )
        for x in per_rank_inputs:
            acc = acc + x.float()
        return acc.to(per_rank_inputs[0].dtype)

    def is_valid(
        self, input: torch.Tensor, comm_buffer: torch.Tensor, buffer_flags: torch.Tensor
    ) -> None:
        # The op itself refuses a comm_buffer that is not a workspace's base
        # pointer, a non-contiguous input, fewer than 2 dimensions and a width
        # that is not a multiple of 16 bytes. It refuses neither condition below.
        if input.dim() < 2:
            return
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

        found = [
            (mapping, ws)
            for mapping, ws in MNNVLAllReduce.allreduce_mnnvl_workspaces.items()
            if ws["uc_buffer"].data_ptr() == comm_buffer.data_ptr()
        ]
        assert found, (
            "comm_buffer is not the unicast buffer of a workspace built by "
            "get_or_scale_allreduce_mnnvl_workspace; only that path resets the Lamport "
            "protocol the kernel relies on and records how large its stages are"
        )
        mapping, ws = found[0]
        assert buffer_flags.data_ptr() == ws["buffer_flags"].data_ptr(), (
            "buffer_flags must be the control words of the workspace comm_buffer belongs "
            "to; the op takes any uint32 tensor and indexes the stages with whatever it holds"
        )
        width = input.shape[-1]
        rows = input.numel() // width
        need = MNNVLAllReduce.get_required_workspace_size(rows, width, mapping.tp_size, input.dtype)
        assert need <= ws["buffer_size_bytes"], (
            f"a [{rows}, {width}] {input.dtype} message needs {need} bytes per workspace "
            f"stage and the workspace has {ws['buffer_size_bytes']}; the op does not check "
            "and writes past the stage"
        )


allreduce = Allreduce()
