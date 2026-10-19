# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_varlen catalog entry, at nemotron4-nano-preview's tp1 and
tp4-rank shapes.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the buffers a target hands the op -- an in_proj-shaped
activation whose conv columns are x, a packed per-slot state block whose conv part is the
pool -- and holds the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_varlen import (
    causal_conv1d_varlen as op,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "causal_conv1d_varlen requires a CUDA device"

#: Slots in each test's state pool; a batch's slots are drawn from it at random.
POOL_SLOTS = 64

#: Bound on the fraction of outputs that may differ from the reference at all; see `_compare`.
MAX_MISMATCH_FRACTION = 1e-3


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The final rounding may differ, and nothing else.

    Kernel and reference sum the same four bf16 products in fp32 -- the kernel as an fma chain,
    the reference as products then adds -- apply SiLU in fp32 (the kernel with `__expf` and a
    reciprocal), and round once to bf16. Those fp32 differences are ~1e-7 relative, so an output
    can only land on the neighbouring bf16 value, and only when its fp32 value sits on a rounding
    boundary. Measured over every cell: at most 5.9e-5 of the outputs differ at all (one of the
    1-token cell's 16896; about 4e-6 on the large cells), each by exactly one ulp of its own
    magnitude. On a tp4 rank's 4224 channels, five seeds of each cell: the 1- and 7-token cells
    bit-identical, at most 1.1e-5 of the outputs differ elsewhere, the worst element 1.46 ulp at
    its row's scale and the relative RMS 0.0055.

    Three gates, a row being one token:
      * element, 2 ulp at the row scale: one ulp of an element's own magnitude reads as up to
        two of the nominal 2**-8 at the row's largest magnitude (measured worst 1.46). A
        resumed sequence computed as if fresh misses by 210 to 260.
      * relative RMS, 0.1 ulp: measured worst 0.019, that single flip. Rounding the conv output
        to bf16 before SiLU, as the checkpoint's HF module does, sits at 0.74.
      * the fraction of outputs not bit-identical, 1e-3: 17x the worst measured. On these cells
        that double rounding changes 25% of them while staying within one ulp of each, so this
        gate and the RMS one are what tell it apart from the op.
    """
    out_t, ref_t = out.t(), ref.t()
    assert_within_ulp(out_t, ref_t, element_ulp=2.0, rms_ulp=0.1, ulp=ULP[torch.bfloat16])
    mismatch = (out_t != ref_t).float().mean().item()
    assert mismatch <= MAX_MISMATCH_FRACTION, (
        f"{mismatch:.2e} of the outputs differ from the reference at all"
    )


def _build(spec, seed):
    """Materialize one cell as `(proj, block, kwargs)`.

    `x` is the transposed conv-column slice of an in_proj-shaped buffer and `conv_states` the
    conv part of a packed per-slot state block -- the two views a target passes, built the way
    `CppMambaHybridCacheManager._setup_states` builds the pool. Their buffers come back too, so a
    test can see that nothing else in them moved. Every slot starts random: a fresh sequence
    must ignore its slot's stale contents and a resumed one must read them.
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
        0, 256, (POOL_SLOTS, ssm_bytes + conv_bytes), generator=g, device="cuda", dtype=torch.uint8
    )
    conv_states = block[:, ssm_bytes:].view(dtype).view(POOL_SLOTS, dim, width - 1)
    conv_states.copy_(torch.randn(conv_states.shape, generator=g, device="cuda"))

    num_seqs = len(lens)
    slots = torch.randperm(POOL_SLOTS, generator=g, device="cuda")[:num_seqs].to(torch.int32)
    resume = {
        "fresh": [False] * num_seqs,
        "continued": [True] * num_seqs,
        "mixed": [i % 2 == 0 for i in range(num_seqs)],
    }[spec["init"]]
    starts = [0]
    for length in lens:
        starts.append(starts[-1] + length)
    # NaN until written, so a token the op skips cannot pass for one it computed.
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


def _mostly_wrong(out: torch.Tensor, expected: torch.Tensor) -> bool:
    """More than half the outputs far outside any band (NaN counts as wrong)."""
    close = torch.isclose(out.float(), expected.float(), rtol=2**-6, atol=2**-6)
    return (~close).float().mean().item() > 0.5


def test_guard_refuses_a_width_with_no_kernel() -> None:
    """Widths 2 to 4 have kernels; any other launches nothing and returns as if it had run."""
    kwargs = _small(width=5)
    op(**kwargs)  # unguarded: is_valid is not on the call path
    assert kwargs["out"].isnan().all(), "the hazard this guard covers is gone: width 5 now writes"
    with pytest.raises(AssertionError, match="has no kernel"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_weight_in_another_dtype() -> None:
    """fp32 taps with a bf16 x: both are instantiated on x's dtype, so the fp32 bytes are read
    as bf16 values."""
    kwargs = _small()
    expected, _ = op.reference(**kwargs)
    kwargs["weight"] = kwargs["weight"].float()
    op(**kwargs)
    assert _mostly_wrong(kwargs["out"], expected), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="reinterprets weight"), validating(op):
        op(**kwargs)


def test_guard_refuses_an_x_with_no_unit_stride_axis() -> None:
    """Every other token of a channel-major buffer: the channel-major kernel reads each
    channel's tokens as if contiguous."""
    kwargs = _small(seq_lens=(64,), init="fresh")
    dim, total = kwargs["x"].shape
    x = torch.randn(dim, 2 * total, device="cuda").to(torch.bfloat16)[:, ::2]
    out = torch.full((dim, 2 * total), float("nan"), device="cuda").to(torch.bfloat16)[:, ::2]
    kwargs.update(x=x, out=out)
    expected, _ = op.reference(**kwargs)
    op(**kwargs)
    assert _mostly_wrong(out, expected), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="unit-stride"), validating(op):
        op(**kwargs)


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
        conv_states=pool, cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda")
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


def test_guard_refuses_has_initial_state_without_conv_states() -> None:
    """Asked to resume with no pool to resume from, the op starts from zeros and says nothing."""
    kwargs = _small(init="continued")
    kwargs.update(conv_states=None, cache_indices=None)
    fresh, _ = op.reference(**{**kwargs, "has_initial_state": None})
    op(**kwargs)
    _compare(kwargs["out"], fresh)
    with pytest.raises(AssertionError, match="asks for saved state"), validating(op):
        op(**kwargs)
