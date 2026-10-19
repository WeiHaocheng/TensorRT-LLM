# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks and keep this rank's slice, via the trtllm reducescatter op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: The dtypes whose NCCL mapping performs the sum the caller means by it.
#: `getDtypeMap` in `cpp/tensorrt_llm/common/opUtils.cpp` is the whole table:
#: these entries land on a native NCCL float or int type, so `ncclSum` is a
#: float or integer sum of the values. Every other entry, and every dtype the
#: table omits, is refused below -- see `is_valid` for the two distinct ways
#: they go wrong.
_SUMMABLE_DTYPES = frozenset(
    {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.uint8,
    }
)

#: nemotron4-nano-preview's MoE combine: per-rank token counts its dep4 step reduces, as
#: (busiest rank's slice, every rank's slice in rank order or None for an even split, what the
#: height covers). They are the heights its all-gather is certified at, because one step's
#: gather and reduce-scatter carry the same per-rank counts.
_NEMOTRON4_COMBINE_SPLITS = (
    (1, None, "decode, one token per rank"),
    (32, None, "32 tokens per rank"),
    (256, None, "256 tokens per rank"),
    (1024, None, "1024 tokens per rank"),
    (2048, None, "prefill chunk, 2048 tokens per rank"),
    (32, [32, 17, 1, 5], "ragged decode, busiest rank 0"),
    (256, [1, 256, 129, 33], "ragged at 256 tokens, busiest rank 1"),
    (1024, [513, 65, 1024, 1], "ragged at 1024 tokens, busiest rank 2"),
    (2048, [1, 3, 1, 2048], "mixed step, rank 3 prefills 2048 tokens"),
)

#: The per-rank token counts of its dep4 serving steps, in the same form: the heights its
#: all-gather is certified at up to the step budgets its serve configs set (one 8192-token
#: prompt plus a decode batch per rank; 16384 for two prompts).
_NEMOTRON4_SERVING_SPLITS = (
    (4096, None, "4096 tokens per rank, over an MTP decode batch of 1024 x 3"),
    (8192, None, "one prompt per rank, 8192 tokens"),
    (9216, None, "one prompt and 1024 decodes per rank, 9216 tokens"),
    (16384, None, "two prompts per rank, 16384 tokens"),
    (9088, [9088, 896, 1, 8193], "ragged serving step, busiest rank 0"),
    (9216, [513, 9216, 8192, 1], "ragged serving step, busiest rank 1"),
    (16384, [8193, 1024, 16384, 4096], "ragged serving step, busiest rank 2"),
    (8192, [1, 1, 1, 8192], "rank 3 prefills one prompt, the others idle"),
)


def _nemotron4_combine(
    why: str, rows_per_rank: int, dtype: torch.dtype, sizes: Optional[List[int]] = None
) -> Cell:
    return Cell(
        why=why,
        spec=dict(rows_per_rank=rows_per_rank, trailing=(768,), dtype=dtype, sizes=sizes),
    )


class Reducescatter(OpWrapper):
    """Sum `input` across every rank in `group`, then keep this rank's slice.

    Every rank passes a tensor of the same full shape; the elementwise sum is
    split along dim 0 in ascending rank order and rank `i` of the group gets
    slice `i`. `sizes` gives each rank's slice height in that order, or is
    `None` when the split is even. The result is a freshly allocated tensor on
    every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # A cell's `rows_per_rank` is its busiest rank's slice. `sizes` is None for
    # an even split, or lists every rank's slice height in rank order (the ragged
    # form, which the op serves by a different path); the input holds their sum.
    #
    # deepseek-r1 at dep4 pads to a common height first and always passes
    # `sizes=None`. Its row counts are the full heights an attention-DP step
    # reduces, which is the rank count times a slice.
    #
    # nemotron4-nano-preview at dep4 reduce-scatters each MoE block's partial
    # expert sums. Every rank applies its 128-expert window to every rank's
    # tokens in the 768-wide latent space, and each token's sum goes back to the
    # rank that owns the token. bf16 and fp32 partials are both certified at
    # every decode and chunk height, so the target can round once after the sum
    # rather than before it. The serving cells carry those heights up to the
    # step budgets its serve configs set, in fp32 only: that is what the dep4
    # forward reduces.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank after the split, hidden 7168",
            spec=dict(rows_per_rank=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch",
            spec=dict(rows_per_rank=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows_per_rank=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        *(
            _nemotron4_combine(f"nemotron4 dep4 {what}, {name} partials", rows, dtype, sizes)
            for rows, sizes, what in _NEMOTRON4_COMBINE_SPLITS
            for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32))
        ),
        *(
            _nemotron4_combine(f"nemotron4 dep4 {what}, fp32 partials", rows, torch.float32, sizes)
            for rows, sizes, what in _NEMOTRON4_SERVING_SPLITS
        ),
    )

    note = """
    Certified dtypes: bf16 and fp32. fp16 and the integer dtypes in
    `_SUMMABLE_DTYPES` reduce correctly in the op matrix, but no target passes
    them, so they are not cells.

    The cells' payloads make every partial sum exact, so they are gated
    bitwise. On general inputs the op sums in the input dtype, in the order
    NCCL picks. For bf16 the matrix pins that order as a sequential chain in
    ring order (`check_reduction_is_deterministic_and_accumulates_in_the_input_dtype`).
    A bf16 call therefore rounds to bf16 after every partial, and an fp32 call
    stays in fp32 until the caller casts.

    Every cell is certified on both symbols `process_group` selects: the `_pg`
    symbol, by ranks that rendezvous over torch.distributed with MPI disabled,
    and the MPI-session symbol (`process_group=None`, the call a dep4 serving
    world makes), by ranks that `mpirun` starts. On each, a cell is driven
    eagerly and replayed from a CUDA graph on fresh partials. The summation
    order above is pinned on the `_pg` symbol only.

    CUDA graph behaviour is carried by the matrix's own named checks rather
    than as a cell axis; those checks drive the `_pg` symbol only.
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> torch.Tensor:
        """`process_group` selects the communicator flavor, mirroring the
        product router in `_torch/distributed/ops.py`: None runs the
        MPI-session symbol -- the one a dep4 serving world calls -- and a boxed
        c10d ProcessGroup runs the `_pg` symbol the non-MPI backends use.
        """
        if process_group is None:
            return torch.ops.trtllm.reducescatter(input, sizes, group)
        return torch.ops.trtllm.reducescatter_pg(input, sizes, group, process_group)

    def reference(
        self,
        contributions_to_this_slice: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """The sum of what every rank sent to this rank's slice.

        The documented exception to mirroring `__call__`, for the same reason as
        allgather: a collective's output is not a function of what the calling
        rank holds. It takes the one block each rank contributed to *this*
        position, in ascending rank order -- not their whole inputs, because a
        rank can regenerate exactly those WORLD blocks from the shared seed
        convention and never has to materialize anybody else's full tensor.

        Accumulated in fp32 (int64 for integer dtypes) and rounded once. The op
        sums in the input dtype in whatever order NCCL picks; on a cell's payload
        that costs nothing, because every partial sum there is exact in the input
        dtype either way -- which is what lets the matrix gate it bitwise.
        """
        first = contributions_to_this_slice[0]
        acc_dtype = torch.float32 if first.dtype.is_floating_point else torch.int64
        acc = torch.zeros(first.shape, dtype=acc_dtype, device=first.device)
        for block in contributions_to_this_slice:
            acc += block.to(acc_dtype)
        return acc.to(first.dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption, a
        # wedge, or a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "ReducescatterOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently reduced over the wrong elements"
        )
        # An allowlist rather than a list of known-bad dtypes, because the two
        # ways a dtype goes wrong here are both silent and the second is not
        # confined to this call:
        #
        #   mapped to the wrong arithmetic -- float8_e4m3fn and bool both land
        #   on ncclInt8, so ncclSum adds their raw bytes. fp8 comes back as
        #   nonsense and bool behaves like OR, neither with an error.
        #
        #   not in the map at all -- float64, quint4x2 and the rest reach
        #   `TorchUtils::dataType()`, which throws. That call is *inside* the
        #   group: `ReducescatterOp::run_list` in reducescatterOp.cpp converts
        #   the dtype after its `ncclGroupStart()`, so the matching
        #   `ncclGroupEnd()` is never reached. The communicator is left
        #   mid-group, and the collectives after this one return invalid data.
        #   A guard here turns that into a Python error before the group opens.
        assert input.dtype in _SUMMABLE_DTYPES, (
            f"{input.dtype} is not summable by this op; it is accepted and then "
            f"either summed as raw bytes or thrown on inside an open NCCL group, "
            f"which corrupts later collectives. Reduce in one of "
            f"{sorted(str(d) for d in _SUMMABLE_DTYPES)} and cast afterwards"
        )
        if sizes is None:
            assert input.shape[0] % len(group) == 0, (
                f"input.shape[0]={input.shape[0]} must be divisible by "
                f"len(group)={len(group)} when sizes is None; the op takes "
                "shape[0] // len(group) rows per rank and silently drops the remainder"
            )
        else:
            assert len(sizes) == len(group), (
                f"len(sizes)={len(sizes)} must equal len(group)={len(group)}; a short "
                "list raises on the highest rank and wedges the others, a long one "
                "reduces more rows than the input holds and returns garbage"
            )
            assert sum(sizes) == input.shape[0], (
                f"sum(sizes)={sum(sizes)} must equal input.shape[0]={input.shape[0]}; "
                "the op sizes the reduction from `sizes` alone, so a short split "
                "silently drops the trailing rows and a long one reads past the input"
            )


reducescatter = Reducescatter()
