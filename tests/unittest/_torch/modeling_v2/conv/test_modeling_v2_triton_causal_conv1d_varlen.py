# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the triton_causal_conv1d_varlen catalog entry, at nemotron4-nano-preview's tp1 and
tp4-rank shapes.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the buffers a target hands the op -- an in_proj-shaped
activation whose conv columns are x, a packed per-slot state block whose conv part is the
pool -- as `causal_conv1d_varlen`'s test does, and holds the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_varlen import (
    causal_conv1d_varlen,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.triton_causal_conv1d_varlen import (
    triton_causal_conv1d_varlen as op,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "triton_causal_conv1d_varlen requires a CUDA device"

#: Slots in each test's state pool; a batch's slots are drawn from it at random.
POOL_SLOTS = 64

#: Bound on the fraction of outputs that may differ from the reference at all; see `_compare`.
MAX_MISMATCH_FRACTION = 1e-3


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The final rounding may differ, and nothing else.

    Kernel and reference sum the same four bf16 products in fp32 -- the kernel as an fma chain
    from +0, the reference as products then adds -- and apply SiLU in fp32, the kernel with a
    fast exp and a fast division. Those fp32 differences are a few ulps, ~1e-7 relative, so an
    output can only land on the neighbouring bf16 value, and only when its fp32 value sits on a
    rounding boundary. Measured over every cell at five seeds (survey-small-ops): the states are
    exact on every cell; the 1-token cells are bit-identical but for one of a tp4 rank's 4224
    outputs (2.4e-4 of them, relative RMS 0.019 ulp); elsewhere 3.7e-6 to 3.4e-5 of the outputs
    differ, each by one ulp of its own value, the worst element 1.65 ulp at its row's scale and the
    worst relative RMS 0.0071 ulp.

    `causal_conv1d_varlen`'s three gates, a row being one token:
      * element, 2 ulp at the row scale: one ulp of an element's own magnitude reads as up to
        two of the nominal 2**-8 at the row's largest magnitude (measured worst 1.65).
      * relative RMS, 0.1 ulp: 5x the worst measured. TensorRT-LLM's Triton kernel as shipped,
        which multiplies in bf16, sits at 0.70; rounding the conv output to bf16 before SiLU, as
        the checkpoint's HF module does, at 0.74.
      * the fraction of outputs not bit-identical, 1e-3: 4x the worst measured. The bf16
        products change 37% of them.
    """
    out_t, ref_t = out.t(), ref.t()
    assert_within_ulp(out_t, ref_t, element_ulp=2.0, rms_ulp=0.1, ulp=ULP[torch.bfloat16])
    mismatch = (out_t != ref_t).float().mean().item()
    assert mismatch <= MAX_MISMATCH_FRACTION, (
        f"{mismatch:.2e} of the outputs differ from the reference at all"
    )


def _build(spec, seed, pool_slots=POOL_SLOTS):
    """Materialize one cell as `(proj, block, kwargs)`, as `causal_conv1d_varlen`'s test does.

    `x` is the transposed conv-column slice of an in_proj-shaped buffer and `conv_states` the
    conv part of a packed per-slot state block. Every slot starts random: a fresh sequence must
    ignore its slot's stale contents and a resumed one must read them. `out` starts NaN, so a
    token the op skips cannot pass for one it computed.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, width, dtype = spec["dim"], spec["width"], spec["dtype"]
    lens = spec["seq_lens"]
    total = sum(lens)
    proj = torch.randn(total, spec["proj_width"], generator=g, device="cuda").to(dtype)
    x = proj[:, spec["proj_offset"] : spec["proj_offset"] + dim].t()
    # The checkpoint's conv taps have a standard deviation of about 0.1.
    weight = (0.1 * torch.randn(dim, width, generator=g, device="cuda")).to(dtype)
    ssm_bytes = spec["pool_ssm_bytes"]
    conv_bytes = dim * (width - 1) * dtype.itemsize
    block = torch.randint(
        0, 256, (pool_slots, ssm_bytes + conv_bytes), generator=g, device="cuda", dtype=torch.uint8
    )
    conv_states = block[:, ssm_bytes:].view(dtype).view(pool_slots, dim, width - 1)
    conv_states.copy_(torch.randn(conv_states.shape, generator=g, device="cuda"))
    num_seqs = len(lens)
    slots = torch.randperm(pool_slots, generator=g, device="cuda")[:num_seqs].to(torch.int32)
    resume = {
        "fresh": [False] * num_seqs,
        "continued": [True] * num_seqs,
        "mixed": [i % 2 == 0 for i in range(num_seqs)],
    }[spec["init"]]
    starts = [0]
    for length in lens:
        starts.append(starts[-1] + length)
    out = torch.full((total, dim), float("nan"), dtype=dtype, device="cuda").t()
    kwargs = dict(
        x=x,
        weight=weight,
        bias_=None,
        conv_states=conv_states,
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        cache_indices=slots,
        has_initial_state=torch.tensor(resume, dtype=torch.bool, device="cuda"),
        silu_activation=spec["silu_activation"],
        pad_slot_id=PAD_SLOT_ID,
        out=out,
    )
    return proj, block, kwargs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    proj, block, kwargs = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    proj_before, block_before = proj.clone(), block.clone()
    ref_out, ref_states = op.reference(**kwargs)  # first: the call rewrites the slots it reads
    out = op(**kwargs)
    assert out is kwargs["out"], "the op writes the out it is given"
    assert torch.equal(proj, proj_before), "the in_proj buffer x views was written"
    # A state is a copy of inputs, never arithmetic, so it is exact -- and the whole pool is
    # compared, which pins every slot outside the batch too.
    assert torch.equal(kwargs["conv_states"], ref_states), "conv states differ from the reference"
    ssm_bytes = cell.spec["pool_ssm_bytes"]
    assert torch.equal(block[:, :ssm_bytes], block_before[:, :ssm_bytes]), (
        "the recurrent state sharing each slot's block was written"
    )
    _compare(out, ref_out)


@pytest.mark.parametrize("cell", op.CELLS[::3], ids=[c.why[:44] for c in op.CELLS[::3]])
def test_a_drop_in_for_causal_conv1d_varlen(cell) -> None:
    """Same arguments, same slots, outputs within one ulp of the CUDA entry's: a target swaps the
    two by name."""
    _, _, mine = _build(cell.spec, seed=3)
    _, _, cuda = _build(cell.spec, seed=3)
    op(**mine)
    causal_conv1d_varlen(**cuda)
    assert torch.equal(mine["conv_states"], cuda["conv_states"]), "the two entries' states differ"
    assert_within_ulp(mine["out"].t(), cuda["out"].t(), element_ulp=2.0, rms_ulp=0.1)


def test_captured_in_a_cuda_graph_and_replayed() -> None:
    """The launch reads no host lengths: one call captured on a 4-sequence batch replays the
    same arithmetic on new tokens, a new slot list, new resume flags and a refilled pool."""
    spec = dict(op.CELLS[14].spec)  # 4 sequences summing to 8192, all fresh
    _, _, static = _build(spec, seed=21)
    _, _, fresh = _build(dict(spec, init="mixed"), seed=22)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(**static)  # compile outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**static)
    # New inputs in the captured buffers: tokens, states, slots and resume flags.
    static["x"].copy_(fresh["x"])
    static["conv_states"].copy_(fresh["conv_states"])
    static["cache_indices"].copy_(fresh["cache_indices"])
    static["has_initial_state"].copy_(fresh["has_initial_state"])
    ref_out, ref_states = op.reference(**static)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static["conv_states"], ref_states)
    _compare(static["out"], ref_out)


def _small(**over):
    """A small valid call -- two sequences, the first resuming -- with `over` applied to its spec."""
    spec = dict(
        op.CELLS[0].spec,
        dim=64,
        proj_width=96,
        proj_offset=16,
        pool_ssm_bytes=64,
        seq_lens=(5, 11),
        init="mixed",
    )
    spec.update(over)
    return _build(spec, seed=7)[2]


@pytest.mark.parametrize(
    "change, what",
    [
        (dict(bias_=torch.zeros(64, dtype=torch.bfloat16, device="cuda")), "a bias"),
        (dict(silu_activation=False), "SiLU off"),
        (dict(conv_states=None), "no pool"),
    ],
)
def test_forms_it_does_not_run_raise(change, what) -> None:
    kwargs = _small()
    with pytest.raises(ValueError):
        op(**dict(kwargs, **change))


def test_guard_refuses_query_start_loc_that_stops_short_of_x() -> None:
    """The op takes the sequences where query_start_loc puts them; tokens past its end are
    never written."""
    kwargs = _small()
    total = kwargs["x"].shape[-1]
    kwargs["query_start_loc"] = torch.tensor([0, 5, total - 4], dtype=torch.int32, device="cuda")
    op(**kwargs)
    assert kwargs["out"][:, total - 4 :].isnan().all(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="query_start_loc must"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_state_pool_narrower_than_x() -> None:
    """conv_states is addressed through its strides alone, so a slot half as wide as x spills
    its upper channels into the next slot -- another request's state."""
    kwargs = _small(seq_lens=(7,), init="fresh")
    dim = kwargs["x"].shape[0]
    pool = torch.zeros(4, dim // 2, 3, dtype=torch.bfloat16, device="cuda")
    kwargs.update(
        conv_states=pool,
        cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([False], device="cuda"),
    )
    op(**kwargs)
    assert pool[1].abs().sum().item() > 0, "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="conv_states must be"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_slot_outside_the_pool() -> None:
    """Not driven unguarded: the write would land outside the pool's memory."""
    kwargs = _small()
    slots = kwargs["conv_states"].shape[0]
    kwargs["cache_indices"] = torch.tensor([0, slots], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="reach outside"), validating(op):
        op(**kwargs)


def test_guard_refuses_two_sequences_sharing_a_slot() -> None:
    """Not driven unguarded: which sequence's state survives is a race."""
    kwargs = _small()
    kwargs["cache_indices"] = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="repeat"), validating(op):
        op(**kwargs)
