# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the wide_residual catalog entry, at the widths every nemotron4-nano-preview
target runs it.

The entry owns what is certified: `CELLS` is the list, `reference` the gate, `is_valid` the
guard. This file turns a cell into the tensors a target hands the op -- streams whose rows span
the magnitudes a residual takes through depth, a branch update, and coefficients formed the way
the checkpoint forms them from its fp32 logits -- and holds the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual import wide_residual as entry
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.wide_residual import (
    wide_residual as op,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "wide_residual requires a CUDA device"

BF16 = torch.bfloat16

#: Bound on the fraction of normed outputs that may differ from the reference at all, and the
#: number always allowed whatever the size of the call; see `_compare`.
MAX_MISMATCH_FRACTION = 1e-4
MIN_ALLOWED_MISMATCHES = 4


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One rounding of the normalized value may go the other way, and nothing else.

    Kernel and reference compute the read bit-identically: the streams and the read are exact
    products summed in the same order and rounded once each. So a normed read can differ only
    through the variance. Both sum the same 3072 exact squares in fp32, in different orders, so
    the variance and `x * rsqrt(var + eps)` move by ~1e-7 relative, and the bf16 rounding of that
    normalized value n can go the other way, by one of its ulps. Scaled by the weight that is
    |g| * ulp(n) <= 2**-7 |y| before the second rounding, and with half an ulp from each side's
    rounding at most 2**-6 |y| after it: four ulps of 2**-8 at the row's largest magnitude, and
    three of the output's own where the two values sit on either side of a power of two.
    Measured on this file's inputs, every cell over five seeds: at most 1.4e-5 of a call's
    outputs differ at all, by at most 3 own ulps and 2.8 at the row scale, with a relative RMS of
    at most 0.0074 ulp; no one- or three-token call differed at all. Summing the variance in fp64
    instead, another legal order, scores the same (4.1e-6, 2, 1.0, 0.0031).

    Three gates, a row being one token:
      * element, 4 ulp at the row scale: the single-flip bound above.
      * outputs not bit-identical: at most a 1e-4 fraction, 7x the worst measured, and at least
        4 whatever the size, since one flip is already 3.3e-4 of a one-token row. This is the
        gate a miscomputed op fails: skipping HF's rounding of the normalized value changes 26%
        of the outputs, leaving the read unrounded before the norm 23%, and using the read
        factors unrounded 13 to 34%.
      * relative RMS, 0.5 ulp: measured worst 0.0074 on the large cells. It is set by the small
        ones, where one permitted flip on one of a 3072-wide row's largest elements alone reads
        as up to ~0.4. The rivals above sit at 0.59 to 0.78.
    """
    assert out.dtype == ref.dtype == BF16 and out.shape == ref.shape
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.5, ulp=ULP[BF16])
    mismatches = (out != ref).sum().item()
    allowed = max(MAX_MISMATCH_FRACTION * out.numel(), MIN_ALLOWED_MISMATCHES)
    assert mismatches <= allowed, (
        f"{mismatches} of {out.numel()} normed outputs differ from the reference; {allowed:g} may"
    )


def _coefficients(g: torch.Generator, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(r, w, f)` in the checkpoint's fp32 forms, from fp32 logits drawn so that each spans its
    range over the checkpoint's 72 connections and 101 reads: r in [0.906, 1], w in
    [0.002, 1.9], f in [0.0004, 0.98]."""

    def logits(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(n, generator=g, device="cuda")

    retention = 1.0 - 0.1 * torch.sigmoid(-logits(-3.0, 6.0))
    write = 2.0 * torch.sigmoid(logits(-6.8, 3.4))
    factors = torch.sigmoid(logits(-7.8, 4.0))
    return retention, write, factors


def _build(spec, seed, coefficient_dtype=BF16):
    """Materialize one cell as the keyword arguments of one call.

    Row magnitudes are log-uniform over [2**-10, 2**4]: the small end puts the variance within a
    few orders of `eps`, where the epsilon's own rounding counts. The norm weights sit around 0.9
    with the spread the checkpoint's do.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, n, hidden = spec["tokens"], spec["streams"], spec["hidden"]
    scale = torch.exp2(-10.0 + 14.0 * torch.rand(tokens, 1, generator=g, device="cuda"))
    streams = (torch.randn(tokens, n * hidden, generator=g, device="cuda") * scale).to(BF16)
    retention, write, factors = _coefficients(g, n)
    _, _, factors2 = _coefficients(g, n)
    weight = (0.9 + 0.25 * torch.randn(hidden, generator=g, device="cuda")).to(BF16)
    weight2 = (0.9 + 0.25 * torch.randn(hidden, generator=g, device="cuda")).to(BF16)
    kwargs = dict(
        streams=streams,
        update=None,
        retention=None,
        write=None,
        read_factors=factors.to(coefficient_dtype),
        norm_weight=weight,
        eps=spec["eps"],
    )
    if spec["write"]:
        update = scale * torch.randn(tokens, hidden, generator=g, device="cuda")
        kwargs.update(
            update=update.to(BF16),
            retention=retention.to(coefficient_dtype),
            write=write.to(coefficient_dtype),
        )
    if spec["reads"] == 2:
        kwargs.update(read_factors2=factors2.to(coefficient_dtype), norm_weight2=weight2)
    return kwargs


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    kwargs = _build(cell.spec, seed=_seed(cell))
    before = kwargs["streams"].clone()
    ref_streams, ref, ref2 = op.reference(**kwargs)  # first: the call rewrites the streams
    ref_streams = ref_streams.clone()
    streams, normed, normed2 = op(**kwargs)
    assert streams is kwargs["streams"], "the op writes the streams it is given"
    # The write is two roundings of exact products and sums, the same ones in the same order on
    # both sides: bit-exact. Rounding r * s + w * u once instead changes 18 to 25% of them.
    assert torch.equal(streams, ref_streams), "the streams differ from the checkpoint's write"
    if not cell.spec["write"]:
        assert torch.equal(streams, before), "the streams were written without an update"
    _compare(normed, ref)
    if cell.spec["reads"] == 2:
        _compare(normed2, ref2)
    else:
        assert normed2 is None


def test_rows_do_not_depend_on_the_batch() -> None:
    """A token's outputs are the same bits whatever else is in the call.

    CUDA-graph padding and chunking change T for the same tokens; greedy identity across them
    needs every row computed alike. The launch never chooses the number of warps by T, which
    would reorder each row's variance sum.
    """
    spec = dict(op.CELLS[0].spec, tokens=8192, write=True, reads=2)
    full = _build(spec, seed=21)
    parts = []
    for start, stop in ((0, 1), (0, 3), (5, 1029), (100, 8192)):
        part = dict(full, streams=full["streams"][start:stop].clone())
        part["update"] = full["update"][start:stop]
        parts.append((start, stop, part))
    streams, normed, normed2 = op(**full)
    for start, stop, kwargs in parts:
        s, y, y2 = op(**kwargs)
        rows = f"rows {start}:{stop}"
        assert torch.equal(s, streams[start:stop]), f"the streams of {rows} depend on the batch"
        assert torch.equal(y, normed[start:stop]), f"the first read of {rows} depends on the batch"
        assert torch.equal(y2, normed2[start:stop]), f"the second read of {rows} depends on it"


def test_a_captured_graph_replays_bitwise() -> None:
    """Decode runs the step inside a CUDA graph: the replay is the eager call, bit for bit."""
    spec = dict(op.CELLS[0].spec, tokens=64, write=True, reads=2)
    kwargs = _build(spec, seed=31)
    start = kwargs["streams"].clone()
    eager = op(**{**kwargs, "streams": start.clone()})

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(**{**kwargs, "streams": start.clone()})  # compile and warm up outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kwargs["streams"].copy_(start)
        captured = op(**kwargs)
    kwargs["streams"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(captured, eager):
        assert torch.equal(got, want)


def test_a_programmatic_launch_gives_the_plain_launch_bits(monkeypatch) -> None:
    """PDL moves when a call starts, not what it computes: every row count it is used for gives
    the bits of the same call launched plainly (`TRTLLM_ENABLE_PDL=0`)."""
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    assert entry._launch_with_pdl(entry._PDL_MAX_TOKENS)
    assert not entry._launch_with_pdl(entry._PDL_MAX_TOKENS + 1)
    for tokens, reads in ((1, 1), (3, 2), (entry._PDL_MAX_TOKENS, 2)):
        kwargs = _build(dict(op.CELLS[0].spec, tokens=tokens, write=True, reads=reads), seed=51)
        start = kwargs["streams"].clone()
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
        plain = op(**{**kwargs, "streams": start.clone()})
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
        pdl = op(**{**kwargs, "streams": start.clone()})
        for got, want in zip(pdl, plain):
            assert (got is None and want is None) or torch.equal(got, want), f"T={tokens}"


@triton.jit
def _late_writer(dst_ptr, src_ptr, numel, spins, BLOCK: tl.constexpr):
    """Let a programmatic dependent start at once, spin, then copy `src` over `dst`."""
    tl.extra.cuda.gdc_launch_dependents()
    acc = tl.load(src_ptr).to(tl.float32)
    for _ in range(spins):
        acc = acc * 0.999999 + 1e-7
    offs = tl.arange(0, BLOCK)
    for start in range(0, numel, BLOCK):
        mask = start + offs < numel
        value = tl.load(src_ptr + start + offs, mask=mask, other=0.0)
        # `acc == acc` holds for every finite acc; it makes each store wait for the spin.
        tl.store(dst_ptr + start + offs, tl.where(acc == acc, value, 0.0), mask=mask)


@triton.jit
def _copy_without_waiting(dst_ptr, src_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for start in range(0, numel, BLOCK):
        mask = start + offs < numel
        tl.store(dst_ptr + start + offs, tl.load(src_ptr + start + offs, mask=mask), mask=mask)


def test_a_programmatic_launch_reads_the_streams_and_update_after_the_kernel_before_it(
    monkeypatch,
) -> None:
    """Under PDL the op starts before the kernel ahead of it is done, so it must not read what that
    kernel writes until it is: the streams (the embedding writes them before the first call) and
    `update` (the branch GEMM writes it). Here that kernel lets its dependents start at once, then
    spins before writing both; a dependent that read them without waiting reads them stale."""
    tokens = 4
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    assert entry._launch_with_pdl(tokens)
    kwargs = _build(dict(op.CELLS[0].spec, tokens=tokens, write=True, reads=2), seed=61)
    final_streams, final_update = kwargs["streams"].clone(), kwargs["update"].clone()
    want = op(**{**kwargs, "streams": final_streams.clone()})

    def late(dst, src, spins=1 << 20):  # a dependent chain of a few ms
        _late_writer[(1,)](dst, src, src.numel(), spins, BLOCK=1024)

    # Compiled first, with the launch options used below: a first call's compile would let the
    # writer finish before the dependent was even enqueued.
    scratch = torch.empty_like(final_update)
    late(scratch, final_update, spins=16)
    _copy_without_waiting[(1,)](scratch, final_update, scratch.numel(), BLOCK=1024, launch_pdl=True)
    torch.cuda.synchronize()

    # The hazard is live: a programmatic dependent that does not wait copies the stale values.
    stale = torch.zeros_like(final_update)
    seen = torch.empty_like(final_update)
    late(stale, final_update)
    _copy_without_waiting[(1,)](seen, stale, stale.numel(), BLOCK=1024, launch_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(stale, final_update), "the late writer did not copy its source"
    assert torch.equal(seen, torch.zeros_like(seen)), (
        "the dependent did not start before the writer finished: this test checks nothing"
    )

    for name, final in (("streams", final_streams), ("update", final_update)):
        inputs = dict(streams=final_streams.clone(), update=final_update.clone())
        inputs[name].zero_()
        late(inputs[name], final)  # the kernel right before the op writes this input
        got = op(**{**kwargs, **inputs})
        torch.cuda.synchronize()
        for g, w in zip(got, want):
            assert torch.equal(g, w), f"the op read {name} before the kernel writing it was done"


def test_fp32_coefficients_and_update_are_rounded_as_the_checkpoint_rounds_them() -> None:
    """HF forms r, w and f in fp32 and casts them to bf16 before use, and casts the branch
    update to bf16: the op takes either form and makes the same casts."""
    spec = dict(op.CELLS[0].spec, tokens=257, write=True, reads=2)
    exact = _build(spec, seed=41, coefficient_dtype=torch.float32)
    exact["update"] = exact["update"].float() * (1.0 + 2.0**-12)  # bits below bf16's
    rounded = {
        k: (v.to(BF16) if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v)
        for k, v in exact.items()
    }
    rounded["streams"] = exact["streams"].clone()
    assert not torch.equal(rounded["update"].float(), exact["update"]), "update has no fp32 bits"
    for got, want in zip(op(**exact), op(**rounded)):
        assert torch.equal(got, want)


def test_zero_tokens_launch_nothing() -> None:
    spec = dict(op.CELLS[0].spec, tokens=0, write=True, reads=2)
    streams, normed, normed2 = op(**_build(spec, seed=0))
    assert streams.shape[0] == normed.shape[0] == normed2.shape[0] == 0


def _small(**over):
    """A small valid call -- 64 tokens, a write and two reads -- with `over` applied to its spec."""
    spec = dict(op.CELLS[0].spec, tokens=64, write=True, reads=2)
    spec.update(over)
    return _build(spec, seed=7)


def _mostly_wrong(out: torch.Tensor, expected: torch.Tensor) -> bool:
    """More than half the outputs far outside any band (NaN counts as wrong)."""
    close = torch.isclose(out.float(), expected.float(), rtol=2**-5, atol=2**-5)
    return (~close).float().mean().item() > 0.5


def test_guard_refuses_fp32_streams() -> None:
    """Streams kept in fp32 (`residual_in_fp32`) are written in fp32 by the checkpoint; the op
    rounds every write to bf16 and stores the rounded values as fp32."""
    kwargs = _small()
    kwargs["streams"] = kwargs["streams"].float() * (1.0 + 2.0**-12)
    streams, _, _ = op(**{**kwargs, "streams": kwargs["streams"].clone()})  # unguarded
    assert torch.equal(streams, streams.to(BF16).float()), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="streams must be bf16"), validating(op):
        op(**kwargs)


def test_guard_refuses_streams_of_another_width() -> None:
    """Four streams where the factors name three: the op reads the first three of each row and
    ignores the fourth without a word."""
    kwargs = _small(write=False)
    hidden = kwargs["norm_weight"].numel()
    tokens = kwargs["streams"].shape[0]
    four = torch.randn(tokens, 4 * hidden, device="cuda").to(BF16)
    _, expected, expected2 = op.reference(**{**kwargs, "streams": four[:, : 3 * hidden]})
    _, normed, normed2 = op(**{**kwargs, "streams": four})
    _compare(normed, expected)  # the fourth stream changed nothing: the hazard is still there
    _compare(normed2, expected2)
    with pytest.raises(AssertionError, match=r"streams must be \[T, n \* hidden\]"), validating(op):
        op(**{**kwargs, "streams": four})


def test_guard_refuses_streams_without_unit_stride_rows() -> None:
    """Every other column of a wider buffer: the op reads each row as if contiguous."""
    kwargs = _small(write=False)
    tokens, width = kwargs["streams"].shape
    wide = torch.randn(tokens, 2 * width, device="cuda").to(BF16)
    strided = wide[:, ::2]
    _, expected, _ = op.reference(**{**kwargs, "streams": strided})
    _, normed, _ = op(**{**kwargs, "streams": strided})
    assert _mostly_wrong(normed, expected), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="unit-stride"), validating(op):
        op(**{**kwargs, "streams": strided})


def test_guard_refuses_a_strided_coefficient_vector() -> None:
    """Every other element of a buffer: the op reads the first three elements flat."""
    kwargs = _small()
    buffer = torch.rand(6, device="cuda").to(BF16)
    strided = buffer[::2]
    kwargs["read_factors"] = strided
    expected = op.reference(**{**kwargs, "streams": kwargs["streams"].clone()})
    got = op(**{**kwargs, "streams": kwargs["streams"].clone()})
    assert not torch.equal(got[1], expected[1]), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="read_factors must be contiguous"), validating(op):
        op(**kwargs)


def test_guard_refuses_a_strided_norm_weight() -> None:
    """A norm weight taken from every other element: read flat, the wrong weights scale the row."""
    kwargs = _small(write=False)
    hidden = kwargs["norm_weight"].numel()
    kwargs["norm_weight"] = torch.rand(2 * hidden, device="cuda").to(BF16)[::2]
    _, expected, _ = op.reference(**kwargs)
    _, normed, _ = op(**kwargs)
    assert _mostly_wrong(normed, expected), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="norm_weight must be contiguous"), validating(op):
        op(**kwargs)


def test_guard_refuses_an_update_read_down_its_columns() -> None:
    """A transposed update: the op reads row t of it as contiguous memory, which is not row t."""
    kwargs = _small()
    tokens, hidden = kwargs["update"].shape
    transposed = torch.randn(hidden, tokens, device="cuda").to(BF16).t()
    kwargs["update"] = transposed
    expected, _, _ = op.reference(**{**kwargs, "streams": kwargs["streams"].clone()})
    streams, _, _ = op(**{**kwargs, "streams": kwargs["streams"].clone()})
    differ = (streams != expected).float().mean().item()
    assert differ > 0.5, f"the hazard this guard covers is gone: {differ:.2f} of the streams differ"
    with pytest.raises(AssertionError, match="update must be"), validating(op):
        op(**kwargs)


def test_guard_refuses_an_update_with_fewer_rows() -> None:
    """Not driven unguarded: the last rows of the update would be read past its end."""
    kwargs = _small()
    kwargs["update"] = kwargs["update"][:-1]
    with pytest.raises(AssertionError, match="update must be"), validating(op):
        op(**kwargs)
