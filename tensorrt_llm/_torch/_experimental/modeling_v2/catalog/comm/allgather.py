# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: nemotron4-nano-preview's MoE dispatch row: one routed token's 768 bf16 latent values, then
#: its 10 int32 expert ids, then its 10 fp32 routing weights, gathered as raw bytes.
_NEMOTRON4_DISPATCH_ROW_BYTES = 768 * 2 + 10 * 4 + 10 * 4

#: The per-rank token counts its dep4 step gathers, as (busiest rank's rows, every rank's rows in
#: rank order or None when they are equal, what the height covers).
_NEMOTRON4_DISPATCH_SPLITS = (
    (1, None, "decode, one token per rank"),
    (32, None, "32 tokens per rank, ~52 KB each"),
    (256, None, "256 tokens per rank, ~0.4 MB each"),
    (1024, None, "1024 tokens per rank, ~1.7 MB each"),
    (2048, None, "prefill chunk, 2048 tokens per rank"),
    (32, [32, 17, 1, 5], "ragged decode, busiest rank 0"),
    (256, [1, 256, 129, 33], "ragged at 256 tokens, busiest rank 1"),
    (1024, [513, 65, 1024, 1], "ragged at 1024 tokens, busiest rank 2"),
    (2048, [1, 3, 1, 2048], "mixed step, rank 3 prefills 2048 tokens"),
)

#: The per-rank token counts of its dep4 serving steps, in the same form. A serve config prefills
#: unchunked under a per-rank step budget of one 8192-token prompt plus a decode batch (tp1's
#: serves 9088), so a step that carries a prompt gathers 8192 rows and up per rank; 16384 is the
#: budget of two prompts. A ragged step is what attention DP makes of that: ranks prefilling a
#: prompt beside ranks that only decode, and idle ranks running their one-token dummy request.
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


#: nemotron4-nano-preview's tp4 logits: each rank's fp32 slice of the vocabulary, 262272 / 4
#: columns, gathered at the row counts the target forms vocab-parallel, as (rows, what they are).
_NEMOTRON4_TP4_LOGITS_WIDTH = 262272 // 4
_NEMOTRON4_TP4_LOGITS_ROWS = (
    (1, "one decode row or one draft row"),
    (3, "one request's MTP verify, its token and two drafts"),
    (48, "MTP verify at 16 requests"),
    (64, "the most rows the tp4 target gathers"),
)


def _nemotron4_dispatch(why: str, rows: int, sizes: Optional[List[int]] = None) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            rows=rows, trailing=(_NEMOTRON4_DISPATCH_ROW_BYTES,), dtype=torch.uint8, sizes=sizes
        ),
    )


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # A cell's `rows` is its busiest rank's height. `sizes` is None when every
    # rank holds `rows` rows (the padded form) or lists every rank's height in
    # rank order (the ragged form, which the op serves by a different path).
    #
    # deepseek-r1 at dep4 pads every rank to a common height first and always
    # passes `sizes=None`. Its row counts span what an attention-DP rank feeds
    # its expert-parallel MoE call, from one decode token to a full prefill chunk.
    #
    # nemotron4-nano-preview at dep4 packs each routed token into one
    # `_NEMOTRON4_DISPATCH_ROW_BYTES` row, so a MoE block makes one gather
    # rather than three. Attention DP gives ranks different token counts, so its
    # ragged form is certified beside the padded one. Every rank holds at least
    # one row, because an idle attention-DP rank runs a dummy request
    # (`_pad_attention_dp_dummy_request`). That is also why one token per rank
    # has no ragged cell. The ragged cells move the busiest rank through every
    # position. The reduce-scatter entry certifies the same per-rank counts,
    # because one step's gather and reduce-scatter carry them both. The serving
    # cells carry those counts up to the step budgets its serve configs set.
    #
    # nemotron4-nano-preview at tp4 gathers each rank's fp32 slice of the
    # vocabulary's logits, up to its vocab-parallel row limit
    # (`_VOCAB_PARALLEL_ROWS`). Every rank forms logits for the same rows, so its
    # cells are the padded form.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank, hidden 7168",
            spec=dict(rows=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch, the common case between decode and prefill",
            spec=dict(rows=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        *(
            _nemotron4_dispatch(f"nemotron4 dep4 {what}, packed dispatch rows", rows, sizes)
            for rows, sizes, what in _NEMOTRON4_DISPATCH_SPLITS + _NEMOTRON4_SERVING_SPLITS
        ),
        *(
            Cell(
                why=f"nemotron4 tp4 logits, {what}",
                spec=dict(
                    rows=rows,
                    trailing=(_NEMOTRON4_TP4_LOGITS_WIDTH,),
                    dtype=torch.float32,
                    sizes=None,
                ),
            )
            for rows, what in _NEMOTRON4_TP4_LOGITS_ROWS
        ),
    )

    note = """
    Certified dtypes: bf16 rows, the uint8 bytes of nemotron4's packed
    dispatch row, and the fp32 rows of nemotron4 tp4's logits. fp16, int32 and
    fp8 move bitwise in the op matrix, but no target passes them, so they are
    not cells. A packed row is only bytes to this op. The caller takes the
    fields back as views of the gathered rows:
    `[:, :1536].view(torch.bfloat16)`, `[:, 1536:1576].view(torch.int32)` and
    `[:, 1576:].view(torch.float32)`. Those views are strided, with one row
    every 1616 bytes, not contiguous.

    Every cell is certified on both symbols `process_group` selects: the `_pg`
    symbol, by ranks that rendezvous over torch.distributed with MPI disabled,
    and the MPI-session symbol (`process_group=None`, the call a dep4 or tp4
    serving world makes), by ranks that `mpirun` starts. On each, a cell is driven
    eagerly and replayed from a CUDA graph on fresh rows.

    CUDA graph behaviour -- capture at every engine batch size, replays
    surviving eager calls of other shapes, the captured sizes vector being held
    -- is not a cell axis either: it is about the op across capture and replay
    rather than about an input configuration, and the matrix carries it as its
    own named checks. Those checks drive the `_pg` symbol only.
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
            return torch.ops.trtllm.allgather(input, sizes, group)
        return torch.ops.trtllm.allgather_pg(input, sizes, group, process_group)

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        sizes: Optional[List[int]] = None,
        group: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The concatenation the collective is supposed to make.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds, so a reference given
        only `input` could not state anything at all. It takes the group's
        inputs in ascending rank order instead. The matrix builds that list on
        one rank from the seed convention every rank shares, which is what keeps
        the reference arithmetic rather than a second collective.
        """
        return torch.cat(list(per_rank_inputs), dim=0)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption or
        # a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "AllgatherOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently gathered from the wrong elements"
        )
        assert sizes is None or len(sizes) == len(group), (
            f"len(sizes)={len(sizes) if sizes is not None else None} must equal "
            f"len(group)={len(group)}; the op sizes its output from `sizes` alone, "
            "so a short list silently drops the trailing ranks and a long one "
            "appends uninitialized rows"
        )


allgather = Allgather()
