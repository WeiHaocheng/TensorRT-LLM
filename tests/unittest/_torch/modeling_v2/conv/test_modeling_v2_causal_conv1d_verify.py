# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_verify catalog entry, at nemotron4-nano-preview's tp1 and
tp4-rank shapes.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the buffers a target hands the op -- the conv columns of the
in_proj output's generation rows as x, one layer's view of a hybrid-cache state pool as
conv_state, one layer's view of the manager's intermediate conv buffer as the window buffer --
drives the decode entry over the same tokens to pin the verify bit for bit, and holds the inputs
the guard exists to refuse.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_update import (
    causal_conv1d_update as decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_verify import (
    causal_conv1d_verify as op,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "causal_conv1d_verify requires a CUDA device"

#: GDP layers whose conv states `MambaHybridCacheManagerV2` coalesces into each slot, and the
#: page of each slot the tested layer owns.
V2_LAYERS, V2_LAYER = 32, 7

#: GDP layers of the stand-in for the manager's `intermediate_conv_states`, and the tested
#: layer's index in it: one layer's window buffer is a view into the middle of a larger buffer.
WINDOW_LAYERS, WINDOW_LAYER = 3, 1

#: Bound on the fraction of outputs that may differ from the reference at all; see `_compare`.
MAX_MISMATCH_FRACTION = 1e-3

#: Leading channels the bitwise test sets to exact zeros -- state and inputs -- under negative
#: taps, where the products are -0 and only an fma chain started from +0 gives +0.
ZERO_CHANNELS = 64

GRAPH_CELLS = [c for c in op.CELLS if c.spec["graph"]]


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The final rounding may differ, and nothing else -- `causal_conv1d_update`'s band.

    `out` and `ref` are `[rows, dim]`, a row being one token of one request. Kernel and
    reference sum the same four bf16 products in fp32 -- the kernel as an fma chain, the
    reference as products then adds -- apply SiLU in fp32 and round once to bf16, exactly as the
    decode entry and its reference do; each token here is bit for bit a decode step (pinned by
    `test_equals_decode_steps_bit_for_bit`), so the decode entry's band and its derivation carry
    over unchanged. Measured over every cell, five seeds each: only batches of 16 or fewer ever
    came out bit-identical; at most 3.0e-5 of the outputs differ, each by exactly one ulp of its
    own magnitude; the worst element 1.17 ulp at its row's scale, the relative RMS 0.0074 ulp.
    With the checkpoint's own taps (all 32 GDP layers, T = 3, batch 64) the three measure 0.95,
    0.0053 and 7.8e-6.

    Three gates:
      * element, 2 ulp at the row scale: one ulp of an element's own magnitude reads as up to two
        of the nominal 2**-8 at the row's largest magnitude, so this admits a one-ulp flip
        anywhere and nothing wider. Over the cells a verify that starts from a zero state misses
        by 193 ulp or more, one that reads the state newest first by 225, and one that convolves
        every token with the saved window instead of the rolled one by 231.
      * relative RMS, 0.1 ulp: 13x the worst measured. Rounding the conv output to bf16 before
        SiLU, as the checkpoint's HF module does, sits at 0.72 or more.
      * the fraction of outputs not bit-identical, 1e-3: 34x the worst measured. That double
        rounding changes a quarter of them while staying inside the element gate (1.98 ulp at
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
    """Materialize one cell as `(proj, raw, buf, kwargs)`.

    `x` is the conv-column slice of an in_proj-shaped buffer whose rows are the generation
    tokens, each request's T tokens together, viewed `[batch, dim, T]`; `conv_state` one layer's
    view of a state pool; the window buffer one layer's view of an `intermediate_conv_states`
    stand-in `[layers, batch + 2, positions, dim, W - 1]` holding random bytes. Their buffers come
    back too, so a test can see that nothing else in them moved. The live rows draw distinct
    random slots from a pool of `2 * batch + 3`; the padding rows sit at the end of the batch with
    PAD_SLOT_ID, where a CUDA-graph batch puts them.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, width, dtype = spec["dim"], spec["width"], spec["dtype"]
    batch, tokens = spec["batch"], spec["tokens"]
    proj = torch.randn(batch * tokens, spec["proj_width"], generator=g, device="cuda").to(dtype)
    lo = spec["proj_offset"]
    x = proj[:, lo : lo + dim].view(batch, tokens, dim).transpose(1, 2)
    # The checkpoint's conv taps have a standard deviation of 0.09 to 0.12 per GDP layer.
    weight = (0.1 * torch.randn(dim, width, generator=g, device="cuda")).to(dtype)
    num_slots = 2 * batch + 3
    raw = _raw_pool(spec, num_slots, g)
    buf = torch.randn(
        (WINDOW_LAYERS, batch + 2, spec["window_positions"], dim, width - 1),
        generator=g,
        device="cuda",
    ).to(dtype)
    live = torch.randperm(num_slots, generator=g, device="cuda")[: batch - spec["pad_rows"]]
    pad = torch.full((spec["pad_rows"],), PAD_SLOT_ID, device="cuda")
    kwargs = dict(
        x=x,
        conv_state=_view(raw, spec),
        weight=weight,
        conv_state_indices=torch.cat([live, pad]).to(torch.int32),
        intermediate_conv_window=buf[WINDOW_LAYER],
        pad_slot_id=PAD_SLOT_ID,
    )
    return proj, raw, buf, kwargs


def _check(proj, raw, buf, kwargs, ref_out, ref_windows, before, spec) -> None:
    """The live rows' outputs against `reference`; their windows exactly the reference's (a
    window is a copy of inputs, never arithmetic); every other element of the in_proj buffer --
    padding rows, columns outside the conv slice -- and of the window buffer -- padding rows,
    rows past the batch, positions past T, the other layers -- as it was before the call; and
    every byte of the pool's memory untouched."""
    proj_before, raw_before, buf_before = before
    batch, tokens, dim = spec["batch"], spec["tokens"], spec["dim"]
    slots = kwargs["conv_state_indices"]
    live = slots != PAD_SLOT_ID
    rows = live.nonzero().squeeze(1)
    lo = spec["proj_offset"]

    out = kwargs["x"][rows].transpose(1, 2).reshape(-1, dim)
    _compare(out, ref_out[rows].transpose(1, 2).reshape(-1, dim))
    written = torch.zeros(batch, tokens, proj.shape[1], dtype=torch.bool, device="cuda")
    written[rows, :, lo : lo + dim] = True
    written = written.view(proj.shape)
    assert torch.equal(proj[~written], proj_before[~written]), (
        "the op wrote the in_proj buffer outside the live rows' conv columns"
    )

    window = kwargs["intermediate_conv_window"]
    assert torch.equal(window[rows, :tokens], ref_windows[rows, :tokens]), (
        "windows differ from the reference"
    )
    filled = torch.zeros(buf.shape[:3], dtype=torch.bool, device="cuda")
    filled[WINDOW_LAYER, rows, :tokens] = True
    assert torch.equal(buf[~filled], buf_before[~filled]), (
        "the op wrote the window buffer outside the live rows' first T positions"
    )
    assert torch.equal(raw, raw_before), "the op wrote the pool, which it only reads"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    proj, raw, buf, kwargs = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = (proj.clone(), raw.clone(), buf.clone())
    ref_out, ref_windows = op.reference(**kwargs)  # first: the call rewrites what it reads
    out = op(**kwargs)
    assert out is kwargs["x"], "the op writes x in place and returns it"
    _check(proj, raw, buf, kwargs, ref_out, ref_windows, before, cell.spec)


def _decode_steps(kwargs, spec):
    """T one-token calls of the decode entry over the same tokens, on a copy of the pool:
    `(outputs [live, dim, T], windows [live, T, dim, W - 1])` -- each window the slot's conv
    state after that step, what the verify has to store for the commit."""
    slots = kwargs["conv_state_indices"]
    live = slots != PAD_SLOT_ID
    owned = slots[live].long()
    states = kwargs["conv_state"].clone()
    outs, windows = [], []
    for t in range(spec["tokens"]):
        x_t = kwargs["x"][:, :, t : t + 1].clone()
        decode(
            x=x_t,
            conv_state=states,
            weight=kwargs["weight"],
            bias_=None,
            silu_activation=True,
            cache_seqlens_=None,
            conv_state_indices=slots,
            pad_slot_id=PAD_SLOT_ID,
        )
        outs.append(x_t[live])
        windows.append(states[owned].clone())
    return torch.cat(outs, dim=-1), torch.stack(windows, dim=1)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_equals_decode_steps_bit_for_bit(cell) -> None:
    """One verify call gives every token the bits T decode steps give it, output and window.

    MTP-on greedy identity rests on this: the accepted tokens' conv outputs feed the rest of the
    layer, and the accepted window becomes the slot's state, exactly as if the tokens had run
    one decode step each. Compared as bit patterns, so a +0 against a -0 counts: the leading
    channels hold exact zeros under negative taps, where the products are -0 and the decode
    kernel's fma chain from +0 gives +0."""
    spec = cell.spec
    proj, raw, buf, kwargs = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31 + 1)
    x, states, weight = kwargs["x"], kwargs["conv_state"], kwargs["weight"]
    x[:, :ZERO_CHANNELS] = 0
    states[:, :ZERO_CHANNELS] = 0
    weight[:ZERO_CHANNELS] = -(weight[:ZERO_CHANNELS].abs() + 0.01)
    expected_out, expected_windows = _decode_steps(kwargs, spec)

    live = kwargs["conv_state_indices"] != PAD_SLOT_ID
    with validating(op):
        op(**kwargs)
    out = x[live]
    assert torch.equal(out.view(torch.int16), expected_out.view(torch.int16)), (
        f"{(out.view(torch.int16) != expected_out.view(torch.int16)).float().mean().item():.2e} "
        "of the outputs differ from the decode steps' bits"
    )
    assert bool((out[:, :ZERO_CHANNELS].view(torch.int16) == 0).all()), "a zero output is not +0"
    windows = kwargs["intermediate_conv_window"][live.nonzero().squeeze(1), : spec["tokens"]]
    assert torch.equal(windows.view(torch.int16), expected_windows.view(torch.int16)), (
        "a window differs from the decode steps' state after that token"
    )


@pytest.mark.parametrize("cell", GRAPH_CELLS, ids=[c.why[:44] for c in GRAPH_CELLS])
def test_cuda_graph_replays_on_new_inputs(cell) -> None:
    """One capture, two replays: each replay reads whatever the in_proj buffer, the slot list and
    the pool hold at replay time and stores its windows where the buffer is -- new tokens, the
    same slots in another row order, new states. The capture calls the op unguarded: `is_valid`
    reads the slots back to the host, which a capture forbids."""
    spec = cell.spec
    proj, raw, buf, kwargs = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    g = torch.Generator(device="cuda").manual_seed(spec["batch"])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**kwargs)  # any first-call initialization (the Triton compile) stays outside the capture
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**kwargs)

    slots = kwargs["conv_state_indices"]
    live = spec["batch"] - spec["pad_rows"]
    real = slots[:live].clone()
    states = kwargs["conv_state"]
    for _ in range(2):
        proj.copy_(torch.randn(proj.shape, generator=g, device="cuda"))
        slots[:live].copy_(real[torch.randperm(live, generator=g, device="cuda")])
        states.copy_(torch.randn(states.shape, generator=g, device="cuda"))
        before = (proj.clone(), raw.clone(), buf.clone())
        ref_out, ref_windows = op.reference(**kwargs)
        graph.replay()
        _check(proj, raw, buf, kwargs, ref_out, ref_windows, before, spec)


def test_matches_torch_conv1d_over_state_and_tokens() -> None:
    """The outputs are torch's own convolution over the saved state followed by the T tokens.

    `Nemotron4GDPMixer` runs `nn.Conv1d(conv_dim, conv_dim, 4, groups=conv_dim)`; over the
    sequence (state ++ tokens) without padding it gives exactly the T outputs of a verify step,
    computed here in float64 and rounded once, as the op rounds. That pins the state's order, the
    taps' orientation and the windows' positions against torch's convolution rather than against
    `reference`."""
    spec = next(c.spec for c in op.CELLS if c.spec["batch"] == 8 and c.spec["tokens"] == 3)
    proj, raw, buf, kwargs = _build(spec, seed=8)
    dim, width, tokens = spec["dim"], spec["width"], spec["tokens"]
    slots = kwargs["conv_state_indices"]
    live = slots != PAD_SLOT_ID
    seq = torch.cat([kwargs["conv_state"][slots[live].long()], kwargs["x"][live]], dim=-1)
    with validating(op):
        op(**kwargs)

    taps = kwargs["weight"].double().unsqueeze(1)  # nn.Conv1d's [out, in / groups, k] layout
    expected = F.silu(F.conv1d(seq.double(), taps, groups=dim)).to(spec["dtype"])
    out = kwargs["x"][live]
    _compare(out.transpose(1, 2).reshape(-1, dim), expected.transpose(1, 2).reshape(-1, dim))
    windows = kwargs["intermediate_conv_window"][live.nonzero().squeeze(1), :tokens]
    for t in range(tokens):
        assert torch.equal(windows[:, t], seq[..., t + 1 : t + width]), (
            f"window {t} does not hold the three inputs ending at token {t}, oldest first"
        )


def _small(**over):
    """A small valid call -- four rows of T = 3, the last one padding -- with `over` applied."""
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
    return _build(spec, seed=7)


def test_guard_refuses_a_window_buffer_with_too_few_positions() -> None:
    """A T = 3 step into a buffer of two positions: the third window of row b lands on row
    b + 1's first, and nothing says so. Shown on the last live row, whose neighbour is a padding
    row nothing else writes."""
    proj, raw, buf, kwargs = _small(window_positions=2)
    window = kwargs["intermediate_conv_window"]
    last_tokens = kwargs["x"][2].clone()  # row 2's third window is its three tokens
    op(**kwargs)  # unguarded: is_valid is not on the call path; the buffer's spare rows absorb it
    assert torch.equal(window[3, 0], last_tokens), (
        "the hazard this guard covers is gone: row 2's third window did not land on row 3"
    )
    proj, raw, buf, kwargs = _small(window_positions=2)
    with pytest.raises(AssertionError, match="positions for T"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_window_buffer_with_too_few_rows() -> None:
    """Not driven unguarded: the last rows' windows would land past the buffer's end."""
    proj, raw, buf, kwargs = _small()
    kwargs["intermediate_conv_window"] = kwargs["intermediate_conv_window"][:3]
    with pytest.raises(AssertionError, match="rows for a batch"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_slot_outside_the_pool() -> None:
    """Not driven unguarded: the step would read past the pool's memory."""
    proj, raw, buf, kwargs = _small()
    kwargs["conv_state_indices"][0] = kwargs["conv_state"].shape[0]
    with pytest.raises(AssertionError, match="reach outside"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_slot_list_shorter_than_the_batch() -> None:
    """Not driven unguarded: the last rows would read their slots past the list's end."""
    proj, raw, buf, kwargs = _small()
    kwargs["conv_state_indices"] = kwargs["conv_state_indices"][:3]
    with pytest.raises(AssertionError, match="one slot per batch row"), validating(op):
        op(**kwargs)


def test_guard_refuses_taps_of_another_width() -> None:
    """Three taps per channel read as four: every channel takes its neighbour's first tap as
    its fourth, and the last reads one element past the weight."""
    proj, raw, buf, kwargs = _small()
    kwargs["weight"] = kwargs["weight"][:, :3]
    with pytest.raises(AssertionError, match="four taps"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_state_of_another_length() -> None:
    """A width-4 conv keeps three inputs; a longer state would be read from its oldest end."""
    proj, raw, buf, kwargs = _small()
    longer = torch.zeros(kwargs["conv_state"].shape[0], 64, 4, device="cuda", dtype=torch.bfloat16)
    kwargs["conv_state"] = longer
    with pytest.raises(AssertionError, match="width-4 state"), validating(op):
        op(**kwargs)
