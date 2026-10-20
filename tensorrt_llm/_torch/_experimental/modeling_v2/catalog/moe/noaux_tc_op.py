# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped top-k expert routing by biased sigmoid score, via the trtllm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class NoauxTcOp(OpWrapper):
    """Route each token to `topk` experts by biased sigmoid score.

    Selection uses `sigmoid(router_logits) + bias`; the combine weights are
    gathered from the *unbiased* `sigmoid(router_logits)`, renormalized to sum
    to 1 and scaled by `routed_scaling_factor`. The bias steers which experts
    are chosen and never reaches the weights -- getting that backwards is the
    single easiest way to misread this op.

    Returns `(topk_weights [T, topk] in router_logits.dtype, topk_ids [T, topk]
    int32)`, both newly allocated and contiguous.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Two routers call this, and they take different kernels (see `note`).
    # deepseek-r1: 256 experts in 8 groups, 4 groups kept, 8 experts per token,
    # scaled by 2.5; its trunk's router GEMM emits bf16 logits and its MTP
    # layer's emits fp32. nemotron4: 512 experts in one group, 10 per token,
    # scaled by 3.16, fp32 logits -- its router (`Nemotron4TopkRouter`)
    # computes them in fp32. The weights come back in the logits' dtype, so
    # the fp32 cells are the ones that see the renormalization's last bit.
    # The token counts are the ones each target routes at once: one per decode
    # step, a decode batch, the chunk a prefill is split into, and for
    # nemotron4 the largest step a rank runs, 16384 tokens. Its expert-parallel
    # target routes each rank's own tokens before gathering them, so no
    # nemotron4 call routes more rows than that.
    #
    # Every bias is fp32. nemotron4's are drawn at its checkpoint's scale
    # (per-layer std 0.025-0.064), which leaves the choice to the sigmoid
    # scores and the top-10/11 margins thin: the 8192-token cell's narrowest
    # is 6e-8, one fp32 ulp, against 1.4e-6 with a unit-scale bias. Exact
    # selection has to hold there.
    _DEEPSEEK = dict(experts=256, n_group=8, topk_group=4, topk=8, scale=2.5, bias_std=1.0)
    _NEMOTRON4 = dict(
        experts=512,
        n_group=1,
        topk_group=1,
        topk=10,
        scale=3.16,
        bias_std=0.05,
        logits_dtype=torch.float32,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one token through the full 256-expert router",
            spec=dict(tokens=1, logits_dtype=torch.bfloat16, **_DEEPSEEK),
        ),
        Cell(
            why="prefill chunk: the batch the target routes at once",
            spec=dict(tokens=2048, logits_dtype=torch.bfloat16, **_DEEPSEEK),
        ),
        Cell(
            why="a token count that is not a multiple of any tile",
            spec=dict(tokens=37, logits_dtype=torch.bfloat16, **_DEEPSEEK),
        ),
        Cell(
            why="deepseek MTP router: fp32 logits through the grouped kernel",
            spec=dict(tokens=2048, logits_dtype=torch.float32, **_DEEPSEEK),
        ),
        Cell(
            why="nemotron4 decode: one token, 512 experts in one group, top-10, fp32",
            spec=dict(tokens=1, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 at 37 tokens, not a multiple of any tile",
            spec=dict(tokens=37, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 decode batch of 1024, the top of the CUDA-graph sizes",
            spec=dict(tokens=1024, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 prefill chunk of 8192 tokens, the max_num_tokens budget",
            spec=dict(tokens=8192, **_NEMOTRON4),
        ),
        Cell(
            why="nemotron4 prefill or mixed step of 16384 tokens, the larger budget",
            spec=dict(tokens=16384, **_NEMOTRON4),
        ),
    )

    note = """
    The kernel evaluates sigmoid as `0.5 * tanh(x/2) + 0.5`, which saturates to
    exactly 1.0 at x >= ~17 and exactly 0.0 at x <= ~-18.5. `reference` uses the
    same form; the 1/(1+exp(-x)) form disagrees in the tails. In fp32 the two
    differ by up to 2.4e-6 relative for |x| <= 5 and 3.3e-4 for |x| <= 10, so a
    router written with `torch.sigmoid` -- nemotron4's HF router is -- can pick
    another expert where a top-k margin is that thin: measured, 1 row in 16384
    random-logit rows with that checkpoint's own bias.

    Ties in the selection score are broken toward the lower expert id, which
    `reference` reproduces with a stable descending sort.

    Two kernels sit behind the op, and the cells drive both. `n_group > 1`
    takes the grouped one: at most 256 experts, at most 32 per group, at most
    256 across the kept groups, `topk` at most 8. `n_group <= 1` takes the
    single-group one: at most 1024 experts and `topk` at most 32, and
    `topk_group` is never read. Any other configuration raises
    ("invokeNoAuxTc: unsupported configuration"), so neither limit needs a
    guard here.
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.noaux_tc_op(
            router_logits, bias, n_group, topk_group, topk, routed_scaling_factor
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native torch, in the kernel's own arithmetic.

        Three details are the kernel's rather than the obvious choice, and all
        three are load-bearing: sigmoid is the tanh form; the picked scores are
        summed in fp32 in the kernel's warp order; and the scale-and-divide is
        evaluated in fp64, rounding once. Against fp32 weights (the fp32-logit
        cells) an all-fp32 renormalization disagrees on about a third of the
        elements, and a sum in sequential or ascending-stride order on a quarter
        to half of the rows; this one is bit-exact on every cell. bf16 weights
        hide the renormalization's precision almost entirely (1 element in
        16384 for the all-fp32 one).
        """
        num_experts = router_logits.shape[-1]
        scores = 0.5 * torch.tanh(0.5 * router_logits.float()) + 0.5
        choice = scores + bias.float()
        if n_group > 1:
            grouped = choice.view(-1, n_group, num_experts // n_group)
            group_score = torch.topk(grouped, k=2, dim=-1).values.sum(-1)
            keep = torch.sort(group_score, dim=-1, descending=True, stable=True).indices[
                :, :topk_group
            ]
            mask = torch.zeros_like(group_score).scatter_(-1, keep, 1.0)
            mask = mask.unsqueeze(-1).expand_as(grouped).reshape(choice.shape)
            choice = torch.where(
                mask.bool(), choice, torch.tensor(float("-inf"), device=choice.device)
            )
        ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk]
        weights = torch.gather(scores, 1, ids)
        # One warp sums the picked scores, lane i holding the i-th pick and the
        # lanes past `topk` holding 0, pairing lanes at strides 16, 8, 4, 2, 1.
        lanes = torch.nn.functional.pad(weights, (0, 32 - topk))
        lane = torch.arange(32, device=lanes.device)
        for stride in (16, 8, 4, 2, 1):
            lanes = lanes + lanes[:, lane ^ stride]
        total = lanes[:, :1].double()
        weights = weights.double() * routed_scaling_factor / (total + 1e-20)
        return weights.to(router_logits.dtype), ids.to(torch.int32)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
    ) -> None:
        # Three domains the op does not police. All three were observed to
        # return a plausible-looking wrong answer instead of raising: the kernel
        # addresses both tensors from data_ptr() as dense buffers and ignores
        # strides, and topk past the end of a row emits expert ids outside the
        # expert range at non-zero weight.
        assert router_logits.is_contiguous(), (
            "router_logits must be contiguous; a strided view is read as a dense "
            "[num_tokens, num_experts] buffer and silently gives wrong routing"
        )
        assert bias.is_contiguous(), (
            "bias must be contiguous; a strided view is read as a dense "
            "[num_experts] buffer and silently gives wrong routing"
        )
        assert topk <= router_logits.shape[-1], (
            f"topk ({topk}) exceeds num_experts ({router_logits.shape[-1]}); the "
            "kernel reads out of bounds and emits out-of-range expert ids"
        )


noaux_tc_op = NoauxTcOp()
