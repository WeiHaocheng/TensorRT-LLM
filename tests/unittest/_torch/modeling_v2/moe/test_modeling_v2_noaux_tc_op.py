# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the noaux_tc_op catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op as op

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "noaux_tc_op requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        spec["tokens"], spec["experts"], generator=g, device="cuda", dtype=torch.float32
    ).to(spec["logits_dtype"])
    bias = torch.randn(spec["experts"], generator=g, device="cuda", dtype=torch.float32)
    bias = bias * spec["bias_std"]
    return logits, bias, spec["n_group"], spec["topk_group"], spec["topk"], spec["scale"]


def _cell_args(cell):
    return _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _cell_args(cell)
    weights, ids = op(*args)
    ref_weights, ref_ids = op.reference(*args)

    # Ids are an exact claim: routing either picks the same experts or it does
    # not, and a near-miss here is a different set of experts, not a rounding.
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert ids.dtype is torch.int32
    # So are the weights, and the band is zero by construction: `reference`
    # repeats every rounding the kernel makes -- the tanh-form sigmoid, the
    # fp32 bias add, the fp32 warp-order sum, the fp64 scale-and-divide, one
    # rounding to the logits' dtype -- so any difference is a change in the
    # kernel's arithmetic. Measured on every cell: none.
    assert weights.dtype is args[0].dtype
    assert torch.equal(weights, ref_weights), "combine weights differ from the reference"


def test_the_fp32_cells_see_the_renormalization() -> None:
    """The fp64 scale-and-divide shows only in fp32 weights; hold that it does.

    An all-fp32 renormalization moves bf16 weights almost never, so the
    exactness above would hold it to fp64 only through the fp32 cells -- and
    only if they can see the difference. This drives the rival on nemotron4's
    prefill cell: it has to miss the op's weights where `reference` meets them
    (measured: about a third of the elements, by 1-2 ulp).
    """
    cell = next(c for c in op.CELLS if c.spec["experts"] == 512 and c.spec["tokens"] == 8192)
    logits, bias, n_group, topk_group, topk, scale = _cell_args(cell)
    weights, ids = op(logits, bias, n_group, topk_group, topk, scale)
    picked = torch.gather(0.5 * torch.tanh(0.5 * logits) + 0.5, 1, ids.long())
    rival = picked / (picked.sum(-1, keepdim=True) + 1e-20) * scale
    missed = (rival != weights).float().mean().item()
    assert missed > 0.1, f"an all-fp32 renormalization reproduced {1 - missed:.1%} of the weights"


def test_heavy_ties_route_exactly() -> None:
    """More than 32 experts at the threshold: the single-group kernel's sort path, held exact.

    The kernel takes as candidates the experts at or above a threshold drawn
    from its lanes' maxima and ranks them one per lane; when ties put more than
    32 experts there, it sorts each lane's experts and picks one at a time
    instead. Continuous logits never get there, so no cell drives that path.
    Logits on a five-value grid with a zero bias put about 100 experts on the
    top score of a row; those rows alternate with continuous ones, so both
    paths run in one launch and the tie order (lower expert first) decides
    every pick past the first.
    """
    g = torch.Generator(device="cuda").manual_seed(17)
    tokens, experts, topk, scale = 64, 512, 10, 3.16
    logits = torch.randn(tokens, experts, generator=g, device="cuda", dtype=torch.float32)
    grid = torch.randint(-2, 3, (tokens, experts), generator=g, device="cuda").float() * 0.5
    logits[::2] = grid[::2]
    bias = torch.zeros(experts, dtype=torch.float32, device="cuda")
    scores = 0.5 * torch.tanh(0.5 * logits) + 0.5
    at_top = (scores == scores.max(-1, keepdim=True).values).sum(-1)
    assert (at_top[::2] > 32).all(), "every grid row must put more than 32 experts on its top score"

    weights, ids = op(logits, bias, 1, 1, topk, scale)
    ref_weights, ref_ids = op.reference(logits, bias, 1, 1, topk, scale)
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert torch.equal(weights, ref_weights), "combine weights differ from the reference"


def test_topk_group_is_not_read_with_one_group() -> None:
    """The note's claim about the single-group kernel, held: nemotron4 passes
    `topk_group=1`, and any other value routes identically."""
    cell = next(c for c in op.CELLS if c.spec["experts"] == 512 and c.spec["tokens"] == 37)
    logits, bias, n_group, _, topk, scale = _cell_args(cell)
    want_weights, want_ids = op(logits, bias, n_group, 1, topk, scale)
    for topk_group in (0, 4, 32):
        weights, ids = op(logits, bias, n_group, topk_group, topk, scale)
        assert torch.equal(ids, want_ids) and torch.equal(weights, want_weights), topk_group


def test_the_sigmoid_is_the_tanh_form() -> None:
    """The entry's reference claims a specific sigmoid; hold the op to it.

    `1/(1+exp(-x))` and `0.5*tanh(x/2)+0.5` agree everywhere a random logit
    lands and diverge in the tails, where the tanh form saturates exactly. A
    cell over random inputs cannot tell them apart, so this drives the tails
    directly -- otherwise `reference` could encode the wrong sigmoid and every
    cell would still pass.
    """
    experts = 8
    logits = torch.full((4, experts), -30.0, dtype=torch.bfloat16, device="cuda")
    logits[:, 0] = 30.0
    bias = torch.zeros(experts, dtype=torch.float32, device="cuda")
    weights, _ = op(logits, bias, 1, 1, 2, 1.0)
    ref_weights, _ = op.reference(logits, bias, 1, 1, 2, 1.0)
    torch.testing.assert_close(weights, ref_weights)


def _ok():
    logits = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(16, dtype=torch.float32, device="cuda")
    return logits, bias


def test_guard_refuses_strided_logits() -> None:
    """Read as a dense buffer from data_ptr(); strides are ignored."""
    logits = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")[:, ::2]
    _, bias = _ok()
    with pytest.raises(AssertionError, match="router_logits must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_a_strided_bias() -> None:
    logits, _ = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")[::2]
    with pytest.raises(AssertionError, match="bias must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_topk_past_the_expert_count() -> None:
    """Reads out of bounds and emits ids outside the expert range."""
    logits, bias = _ok()
    with pytest.raises(AssertionError, match="exceeds num_experts"):
        with validating(op):
            op(logits, bias, 1, 1, 17, 1.0)
