# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_update catalog entry, at nemotron4-nano-preview's tp1 and
tp4-rank shapes.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the buffers a target hands the op -- the conv columns of an
in_proj-shaped activation as x, one layer's view of a hybrid-cache state pool as conv_state --
and holds the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_update import (
    causal_conv1d_update as op,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "causal_conv1d_update requires a CUDA device"

#: GDP layers whose conv states `MambaHybridCacheManagerV2` coalesces into each slot, and the
#: page of each slot the tested layer owns.
V2_LAYERS, V2_LAYER = 32, 7

#: Bound on the fraction of outputs that may differ from the reference at all; see `_compare`.
MAX_MISMATCH_FRACTION = 1e-3

GRAPH_CELLS = [c for c in op.CELLS if c.spec["graph"]]

#: The conv widths the cells certify: tp1's 16896 channels and a tp4 rank's 4224.
DIMS = sorted({c.spec["dim"] for c in op.CELLS}, reverse=True)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The final rounding may differ, and nothing else.

    `out` and `ref` are `[rows, dim]`, a row being one request's token. Kernel and reference sum
    the same four bf16 products in fp32 -- the kernel as an fma chain, the reference as products
    then adds -- apply SiLU in fp32 (the kernel with `__expf` and a reciprocal) and round once to
    bf16. Those fp32 differences are ~1e-7 relative, so an output can only land on the
    neighbouring bf16 value, and only when its fp32 value sits on a rounding boundary. Measured
    over every cell, five seeds each: only batch 2 bit-identical; at most 6e-5 of the outputs
    differ (one of batch 1's 16896), each by exactly one ulp of its own magnitude; the worst
    element 0.74 ulp at its row's scale, the relative RMS 0.011 ulp. With the checkpoint's own
    taps (all 32 GDP layers, batch 64) the three measure 0.41, 0.0054 and 1.1e-5. On a tp4
    rank's 4224 channels, five seeds of each cell: batches 1, 2, 4 and 8 bit-identical, at most
    7.9e-5 of the outputs differ (one of batch 3's 12672), the worst element 0.52 ulp, the
    relative RMS 0.012 ulp.

    Three gates:
      * element, 2 ulp at the row scale: one ulp of an element's own magnitude reads as up to two
        of the nominal 2**-8 at the row's largest magnitude. A step that ignores the saved state
        misses by 167 ulp or more, one that reads it newest first by 203, one that reads it as a
        ring buffer rotated by one (`cache_seqlens_` 1) by 206.
      * relative RMS, 0.1 ulp: 8x the worst measured. Rounding the conv output to bf16 before
        SiLU, as the checkpoint's HF module does, sits at 0.70 or more.
      * the fraction of outputs not bit-identical, 1e-3: 12x the worst measured. That double
        rounding changes a quarter of them while staying inside the element gate (1.985 ulp at
        most), so this gate and the RMS one are what tell it apart from the op.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.1, ulp=ULP[torch.bfloat16])
    mismatch = (out != ref).float().mean().item()
    assert mismatch <= MAX_MISMATCH_FRACTION, (
        f"{mismatch:.2e} of the outputs differ from the reference at all"
    )


def _view(raw: torch.Tensor, spec) -> torch.Tensor:
    """The `[slots, dim, W - 1]` conv-state view of `raw` a target passes, strided per slot."""
    dim, width, dtype = spec["dim"], spec["width"], spec["dtype"]
    if spec["pool"] == "v1":
        return raw[:, spec["pool_ssm_bytes"] :].view(dtype).view(raw.shape[0], dim, width - 1)
    page = dim * (width - 1)
    return raw[V2_LAYER:].as_strided(
        (raw.shape[0] // V2_LAYERS, dim, width - 1), (V2_LAYERS * page, width - 1, 1)
    )


def _raw_pool(spec, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """The memory one layer's conv states live in, every byte of it holding something.

    "v1" is `CppMambaHybridCacheManager._setup_states`' per-slot block as uint8: the GDP recurrent
    state (fp32), then the conv state. "v2" is `MambaHybridCacheManagerV2`'s pages: each slot is
    `V2_LAYERS` consecutive `[dim, W - 1]` pages, one per GDP layer, the tested layer's at
    `V2_LAYER`. Every conv state starts random, as a slot a request has written does.
    """
    dim, width, dtype = spec["dim"], spec["width"], spec["dtype"]
    if spec["pool"] == "v1":
        block = spec["pool_ssm_bytes"] + dim * (width - 1) * dtype.itemsize
        raw = torch.randint(
            0, 256, (num_slots, block), generator=g, device="cuda", dtype=torch.uint8
        )
    else:
        pages = (num_slots * V2_LAYERS, dim, width - 1)
        raw = torch.randn(pages, generator=g, device="cuda").to(dtype)
    states = _view(raw, spec)
    states.copy_(torch.randn(states.shape, generator=g, device="cuda"))
    return raw


def _build(spec, seed):
    """Materialize one cell as `(proj, raw, kwargs)`.

    `x` is the conv-column slice of an in_proj-shaped buffer, one row per request, and
    `conv_state` one layer's view of a state pool -- the two views a target passes. Their buffers
    come back too, so a test can see that nothing else in them moved. The live rows draw distinct
    random slots from a pool of `2 * batch + 3`, so they are scattered through slots they do not
    fill; the padding rows sit at the end of the batch with PAD_SLOT_ID, where a CUDA-graph batch
    puts them.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, width, dtype, batch = spec["dim"], spec["width"], spec["dtype"], spec["batch"]
    proj = torch.randn(batch, spec["proj_width"], generator=g, device="cuda").to(dtype)
    lo = spec["proj_offset"]
    x = proj[:, lo : lo + dim].unsqueeze(-1)
    # The checkpoint's conv taps have a standard deviation of 0.09 to 0.12 per GDP layer.
    weight = (0.1 * torch.randn(dim, width, generator=g, device="cuda")).to(dtype)
    num_slots = 2 * batch + 3
    raw = _raw_pool(spec, num_slots, g)
    live = torch.randperm(num_slots, generator=g, device="cuda")[: batch - spec["pad_rows"]]
    pad = torch.full((spec["pad_rows"],), PAD_SLOT_ID, device="cuda")
    kwargs = dict(
        x=x,
        conv_state=_view(raw, spec),
        weight=weight,
        bias_=None,
        silu_activation=spec["silu_activation"],
        cache_seqlens_=None,
        conv_state_indices=torch.cat([live, pad]).to(torch.int32),
        pad_slot_id=PAD_SLOT_ID,
    )
    return proj, raw, kwargs


def _check(proj, raw, kwargs, ref_out, ref_states, proj_before, raw_before, spec) -> None:
    """The live rows' outputs against `reference`; the live slots' states exactly the
    reference's (a state is a copy of inputs, never arithmetic); every other element of the
    in_proj buffer -- padding rows, columns outside the conv slice -- and of the pool's memory --
    other slots, the recurrent state or the other layers' pages beside each conv state -- as it
    was before the call."""
    slots = kwargs["conv_state_indices"]
    live = slots != PAD_SLOT_ID
    rows = live.nonzero().squeeze(1)
    lo, hi = spec["proj_offset"], spec["proj_offset"] + spec["dim"]
    _compare(proj[rows, lo:hi], ref_out[rows, :, 0])
    written = torch.zeros_like(proj, dtype=torch.bool)
    written[rows, lo:hi] = True
    assert torch.equal(proj[~written], proj_before[~written]), (
        "the op wrote the in_proj buffer outside the live rows' conv columns"
    )
    assert torch.equal(kwargs["conv_state"], ref_states), "conv states differ from the reference"
    owned = torch.zeros_like(raw)
    _view(owned, spec)[slots[live].long()] = 1
    untouched = owned == 0
    assert torch.equal(raw[untouched], raw_before[untouched]), (
        "the op wrote the pool outside the live rows' conv states"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    proj, raw, kwargs = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    proj_before, raw_before = proj.clone(), raw.clone()
    ref_out, ref_states = op.reference(**kwargs)  # first: the call rewrites what it reads
    out = op(**kwargs)
    assert out is kwargs["x"], "the op writes x in place and returns it"
    _check(proj, raw, kwargs, ref_out, ref_states, proj_before, raw_before, cell.spec)


@pytest.mark.parametrize("cell", GRAPH_CELLS, ids=[c.why[:44] for c in GRAPH_CELLS])
def test_cuda_graph_replays_on_new_inputs(cell) -> None:
    """One capture, two replays: each replay reads whatever the in_proj buffer, the slot list and
    the pool hold at replay time, and the second continues the states the first wrote -- the same
    slots in another row order. The capture calls the op unguarded: `is_valid` reads the slots
    back to the host, which a capture forbids."""
    spec = cell.spec
    proj, raw, kwargs = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    g = torch.Generator(device="cuda").manual_seed(spec["batch"])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**kwargs)  # any first-call initialization stays outside the capture
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**kwargs)

    slots = kwargs["conv_state_indices"]
    live = spec["batch"] - spec["pad_rows"]
    real = slots[:live].clone()
    for _ in range(2):
        proj.copy_(torch.randn(proj.shape, generator=g, device="cuda"))
        slots[:live].copy_(real[torch.randperm(live, generator=g, device="cuda")])
        proj_before, raw_before = proj.clone(), raw.clone()
        ref_out, ref_states = op.reference(**kwargs)
        graph.replay()
        _check(proj, raw, kwargs, ref_out, ref_states, proj_before, raw_before, spec)


@pytest.mark.parametrize("dim", DIMS)
def test_decode_steps_reproduce_the_hf_whole_sequence_conv(dim: int) -> None:
    """Eight decode steps from an empty state give the checkpoint module's whole-sequence conv.

    `Nemotron4GDPMixer` runs `nn.Conv1d(conv_dim, conv_dim, 4, padding=3, groups=conv_dim)` over
    the whole sequence and keeps its first L outputs; a target computes the same tokens one decode
    step at a time from the state the last step left. Driving five requests (and three padding
    rows) through eight steps has to give that convolution's outputs -- computed here by torch in
    float64 and rounded once, as the op rounds -- and leave each request's last three inputs in
    its slot. That pins the state's order and the taps' orientation against torch's own
    convolution rather than against `reference`.
    """
    spec = next(c.spec for c in op.CELLS if c.spec["dim"] == dim and c.spec["batch"] == 8)
    proj, raw, kwargs = _build(spec, seed=8)
    width, dtype = spec["width"], spec["dtype"]
    slots = kwargs["conv_state_indices"]
    live = slots != PAD_SLOT_ID
    kwargs["conv_state"][slots[live].long()] = 0  # the module's zero padding before the sequence
    steps = 8
    g = torch.Generator(device="cuda").manual_seed(88)
    seq = torch.randn(spec["batch"], dim, steps, generator=g, device="cuda").to(dtype)

    lo, hi = spec["proj_offset"], spec["proj_offset"] + dim
    outs = []
    for t in range(steps):
        proj[:, lo:hi] = seq[:, :, t]
        with validating(op):
            op(**kwargs)
        outs.append(proj[:, lo:hi].clone())
    out = torch.stack(outs, dim=-1)[live]  # [requests, dim, steps]

    taps = kwargs["weight"].double().unsqueeze(1)  # nn.Conv1d's [out, in / groups, k] layout
    conv = F.conv1d(seq[live].double(), taps, padding=width - 1, groups=dim)[..., :steps]
    expected = F.silu(conv).to(dtype)
    _compare(out.transpose(1, 2).reshape(-1, dim), expected.transpose(1, 2).reshape(-1, dim))
    assert torch.equal(kwargs["conv_state"][slots[live].long()], seq[live][..., -(width - 1) :]), (
        "a slot does not hold its request's last three inputs, oldest first"
    )


def _small(**over):
    """A small valid call -- four rows, the last one padding -- with `over` applied to its spec."""
    spec = dict(
        op.CELLS[0].spec,
        dim=64,
        proj_width=96,
        proj_offset=16,
        pool_ssm_bytes=64,
        batch=4,
        pad_rows=1,
        pool="v1",
    )
    spec.update(over)
    return _build(spec, seed=7)[2]


def _mostly_wrong(out: torch.Tensor, expected: torch.Tensor) -> bool:
    """More than half the outputs far outside any band."""
    close = torch.isclose(out.float(), expected.float(), rtol=2**-6, atol=2**-6)
    return (~close).float().mean().item() > 0.5


def test_guard_refuses_cache_seqlens_on_the_linear_pool() -> None:
    """`cache_seqlens_` turns the pool into a ring buffer: at 1, the step reads its window rotated
    by one and writes the new input into the middle of the state."""
    kwargs = _small()
    expected, expected_states = op.reference(**kwargs)
    kwargs["cache_seqlens_"] = torch.ones(4, dtype=torch.int32, device="cuda")
    op(**kwargs)  # unguarded: is_valid is not on the call path
    live = kwargs["conv_state_indices"] != PAD_SLOT_ID
    owned = kwargs["conv_state_indices"][live].long()
    assert _mostly_wrong(kwargs["x"][live], expected[live]), "the hazard this guard covers is gone"
    assert not torch.equal(kwargs["conv_state"][owned], expected_states[owned]), (
        "the hazard this guard covers is gone: the ring-buffer write lands where the linear one does"
    )
    with pytest.raises(AssertionError, match="ring buffer"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_slot_outside_the_pool() -> None:
    """Not driven unguarded: the step would read and write past the pool's memory."""
    kwargs = _small()
    kwargs["conv_state_indices"][0] = kwargs["conv_state"].shape[0]
    with pytest.raises(AssertionError, match="reach outside"), validating(op):
        op(**kwargs)


def test_guard_refuses_two_rows_sharing_a_slot() -> None:
    """How the cache managers pad a CUDA-graph batch: every padding row gets the dummy request's
    real slot. Not driven unguarded: which row's state survives is a race."""
    kwargs = _small()
    slots = kwargs["conv_state_indices"]
    slots[-1] = slots[0]
    with pytest.raises(AssertionError, match="race"), validating(op):
        op(**kwargs)
