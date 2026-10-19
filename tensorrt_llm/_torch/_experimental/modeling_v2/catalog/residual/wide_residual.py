# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wide residual step between two branches: write the branch update, read, RMS-norm."""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from .._launch import Launch
from .._op import Arch, Cell, OpWrapper

# The checkpoint keeps 3 residual streams of its 3072-wide hidden state
# (`wide_residual_num_streams`, `hidden_size`), and every branch norm and the final `norm` use
# `rms_norm_eps` 1e-5.
_GEOMETRY = dict(streams=3, hidden=3072, eps=1e-5)

# Fixed for every T: the warps a row is spread over set the order of its variance sum, so a launch
# that chose them by T would give the same token different bits in batches of different size.
# Four were fastest at every T measured, and one row per program beat two to eight.
_NUM_WARPS = 4

# Up to this many rows a call's time is one row's latency, and launching it as a programmatic
# dependent of the kernel before it (PDL) measured 0.5-0.7 us faster a call behind a cuBLAS GEMM,
# and ~0.2 us in tp1's decode step. From 96 rows it measured no faster, and from 192 slower.
_PDL_MAX_TOKENS = 64


def _launch_with_pdl(num_tokens: int) -> bool:
    """Whether a call of `num_tokens` rows launches with PDL; `TRTLLM_ENABLE_PDL=0` turns it off."""
    return num_tokens <= _PDL_MAX_TOKENS and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


# Why each token count is a cell.
_TOKENS = (
    (1, "one decode token"),
    (3, "one request's MTP verify (1 + 2 drafts), a non-power-of-two batch"),
    (1024, "the largest decode batch a DP rank captures"),
    (8192, "a full prefill chunk at max_num_tokens 8192"),
    (16384, "a full prefill chunk at max_num_tokens 16384"),
)

# Why each call shape is a cell: where target-core makes it.
_SHAPES = (
    (True, 1, "a branch into a GDP/MLP block, the post-mixer shared read, or the readout"),
    (True, 2, "a branch into a shortcut block: its routed and mixer reads"),
    (False, 1, "the model start into layer 0, or an unfused GDP/MLP block entry"),
    (False, 2, "an unfused shortcut block entry"),
)


def _cell(tokens: int, why_tokens: str, write: bool, reads: int, why_shape: str) -> Cell:
    shape = ("write + " if write else "no write, ") + ("2 reads" if reads == 2 else "1 read")
    return Cell(
        why=f"T={tokens}, {shape}: {why_tokens}; {why_shape}",
        spec=dict(tokens=tokens, write=write, reads=reads, **_GEOMETRY),
    )


@triton.jit
def _gdc_wait():
    """`griddepcontrol.wait`, with a memory clobber so that no load is scheduled above it."""
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _rmsnorm_row(acc, g, y_ptr, cols, mask, inv_hidden, eps):
    """Round the read to bf16, then `Nemotron4RMSNorm`: normalize in fp32, round, scale, round.

    The variance is HF's `pow(2).mean(-1)`: squares (exact) summed in fp32 and scaled by the
    fp32 1/hidden, as torch's mean projects its sum. Only the order of that sum differs.
    """
    x = acc.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(x * x, axis=0) * inv_hidden
    normed = (x * tl.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + cols, (g.to(tl.bfloat16).to(tl.float32) * normed).to(tl.bfloat16), mask=mask)


@triton.jit
def _wide_residual_kernel(
    s_ptr,
    u_ptr,
    r_ptr,
    w_ptr,
    f0_ptr,
    g0_ptr,
    y0_ptr,
    f1_ptr,
    g1_ptr,
    y1_ptr,
    stride_s,
    stride_u,
    stride_y,
    inv_hidden,
    eps,
    HIDDEN: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE: tl.constexpr,
    TWO_READS: tl.constexpr,
):
    """One token row: every stream written in place, then one or two reads normed.

    Each rounding is the checkpoint's. The write is bf16(bf16(r * s) + u * w): a bf16 product
    rounded once (torch's bf16 mul), then an fp32 sum of it and the exact u * w product rounded
    once (torch's bf16 addcmul). The read is the fp32 sum of the exact f * s products, taken in
    stream order and rounded once. A row's stores land on the elements this thread loaded, so the
    in-place write never races a load of the same row.

    Every load is issued before the first store: a load the compiler cannot prove disjoint from an
    earlier store is issued after it, behind that store's own inputs, one more memory round trip
    a row. The loads come in two groups. The parameters first: the norm weights and the
    coefficients, which nothing in a forward writes. Then the wait for the kernel before this one
    (`griddepcontrol.wait`), and the streams and `update` behind it. Under a programmatic launch
    (PDL) the program starts while that kernel drains and the parameters' loads overlap it; under
    a plain launch the wait returns at once. Either way its memory clobber holds the two groups
    apart: left to interleave them, the compiler's order measured up to 0.9 us slower a call at one
    row, and 21% slower for a write and two reads at 8192 rows and over.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    g0 = tl.load(g0_ptr + cols, mask=mask, other=0.0)
    if TWO_READS:
        g1 = tl.load(g1_ptr + cols, mask=mask, other=0.0)
    f0 = ()
    f1 = ()
    r = ()
    w = ()
    for k in tl.static_range(STREAMS):
        f0 = f0 + (tl.load(f0_ptr + k),)
        if TWO_READS:
            f1 = f1 + (tl.load(f1_ptr + k),)
        if WRITE:
            r = r + (tl.load(r_ptr + k),)
            w = w + (tl.load(w_ptr + k),)
    _gdc_wait()
    s_row = s_ptr + row * stride_s
    s = ()
    for k in tl.static_range(STREAMS):
        s = s + (tl.load(s_row + k * HIDDEN + cols, mask=mask, other=0.0),)
    if WRITE:
        u = tl.load(u_ptr + row * stride_u + cols, mask=mask, other=0.0)
        u = u.to(tl.bfloat16).to(tl.float32)
    tl.extra.cuda.gdc_launch_dependents()
    for k in tl.static_range(STREAMS):
        sk = s[k].to(tl.float32)
        if WRITE:
            kept = (sk * r[k].to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            new = (kept + u * w[k].to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            tl.store(s_row + k * HIDDEN + cols, new, mask=mask)
            sk = new.to(tl.float32)
        p0 = f0[k].to(tl.bfloat16).to(tl.float32) * sk
        if k == 0:
            acc0 = p0
        else:
            acc0 = acc0 + p0
        if TWO_READS:
            p1 = f1[k].to(tl.bfloat16).to(tl.float32) * sk
            if k == 0:
                acc1 = p1
            else:
                acc1 = acc1 + p1
    _rmsnorm_row(acc0, g0, y0_ptr + row * stride_y, cols, mask, inv_hidden, eps)
    if TWO_READS:
        _rmsnorm_row(acc1, g1, y1_ptr + row * stride_y, cols, mask, inv_hidden, eps)


#: The kernel launched straight to its compiled form once compiled (`Launch`).
_WIDE_RESIDUAL = Launch(_wide_residual_kernel)


class WideResidual(OpWrapper):
    """Between two branches: optionally write a branch update into the streams, then read them
    once or twice, each read RMS-normed with its own weight. One kernel launch.

    `streams` is [T, n * hidden] (the n streams side by side, `Nemotron4WideResidualRead`'s
    `unflatten(-1, (n, -1))`). With `update` given, every stream is written in place,
    `S_k <- bf16(bf16(r_k * S_k) + w_k * U)` (`Nemotron4WideResidualConnection.write`); without
    it the streams are only read. Each read is `x = bf16(sum_k f_k * S_k)` over the streams after
    the write (`...Connection.read`, `Nemotron4WideResidualRead`), returned as
    `Nemotron4RMSNorm(x)`: `g * bf16(x * rsqrt(mean(x^2) + eps))`, rounded again. Returns
    `(streams, normed, normed2)`, `normed2` None for one read; the reads are new tensors.

    `r`, `w` and `f` are [n] vectors in HF's own form, each from its fp32 logit:
    `1 - max_forget * sigmoid(-retention_logit)`, `2 * sigmoid(write_logit)`, `sigmoid(logit)`.
    Given in fp32 or already bf16, they are rounded to bf16 by the op as HF's
    `.to(streams.dtype)` rounds them, and so is `update`.

    The streams come back bit-identical to the checkpoint's torch ops. The normed reads differ from
    them only where the fp32 variance, summed in another order, moves a value across a bf16
    rounding boundary. A captured CUDA graph replays the eager call bit for bit.

    A call of up to `_PDL_MAX_TOKENS` rows launches as a programmatic dependent (PDL) of the kernel
    before it: it starts while that kernel drains and reads the streams and `update` only once that
    kernel is done. That moves when it runs, not what it computes.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        _cell(tokens, why_tokens, write, reads, why_shape)
        for tokens, why_tokens in _TOKENS
        for write, reads, why_shape in _SHAPES
    )

    note = """
    The streams are bf16, the checkpoint's `residual_in_fp32` false. A checkpoint that keeps them
    in fp32 writes and reads them without the bf16 roundings this op makes; `is_valid` refuses
    fp32 streams.

    Both reads take one `eps`: every norm of the text model, and the final `norm`, uses
    `rms_norm_eps`. A row of `streams` may be strided (a slice of a wider buffer); within a row the
    elements are unit-stride.

    One program per token row, the whole row in registers (the hidden width rounded up to a power
    of two). `T = 0` launches nothing.

    PDL follows `TRTLLM_ENABLE_PDL` (default on) and the row count; it changes scheduling, not
    results, and is not a cell axis. Under it the norm weights and the coefficients are read before
    the kernel ahead of the call has finished, so that kernel must not write them; nothing in a
    forward does. The streams and `update` are read after the wait and may come from it, as the
    streams do from the embedding before the first call.

    The no-write, two-read shape at 8192 rows and over runs ~7% slower in this load order than
    when the compiler orders the loads itself; no nemotron4 target makes that call.
    """

    def __call__(
        self,
        streams: torch.Tensor,
        update: Optional[torch.Tensor],
        retention: Optional[torch.Tensor],
        write: Optional[torch.Tensor],
        read_factors: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        read_factors2: Optional[torch.Tensor] = None,
        norm_weight2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Writes `streams` in place when `update` is given and returns it -- the op destroys the
        streams' prior contents, so a caller that wants them has to have kept its own copy."""
        num_tokens, hidden = streams.shape[0], norm_weight.shape[-1]
        two_reads = read_factors2 is not None
        normed = torch.empty(num_tokens, hidden, dtype=streams.dtype, device=streams.device)
        normed2 = torch.empty_like(normed) if two_reads else None
        if num_tokens:
            pdl = _launch_with_pdl(num_tokens)
            _WIDE_RESIDUAL[(num_tokens,)](
                streams,
                update,
                retention,
                write,
                read_factors,
                norm_weight,
                normed,
                read_factors2,
                norm_weight2,
                normed2,
                streams.stride(0),
                update.stride(0) if update is not None else 0,
                normed.stride(0),
                1.0 / hidden,
                eps,
                HIDDEN=hidden,
                STREAMS=read_factors.numel(),
                BLOCK=triton.next_power_of_2(hidden),
                WRITE=update is not None,
                TWO_READS=two_reads,
                num_warps=_NUM_WARPS,
                launch_pdl=pdl,
            )
        return streams, normed, normed2

    def reference(
        self,
        streams: torch.Tensor,
        update: Optional[torch.Tensor],
        retention: Optional[torch.Tensor],
        write: Optional[torch.Tensor],
        read_factors: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        read_factors2: Optional[torch.Tensor] = None,
        norm_weight2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """The checkpoint's torch ops, rounding where they round; `streams` is not touched.

        The write is `Nemotron4WideResidualConnection.write` verbatim and the norm
        `Nemotron4RMSNorm.forward` verbatim. The read is the one place this is not HF's own op:
        HF's `torch.matmul(f, streams)` is a cuBLAS bf16 GEMM with K = n, whose fp32 accumulation
        order is cuBLAS's to choose. Here the n exact fp32 products are summed in stream order and
        rounded once, which is what that GEMM computes up to the order of its fp32 sum.
        """
        dtype, hidden = streams.dtype, norm_weight.shape[-1]
        s = streams.unflatten(-1, (-1, hidden))
        if update is not None:
            s = s * retention.to(dtype).unsqueeze(-1)
            s = s.addcmul_(update.to(dtype).unsqueeze(-2), write.to(dtype).unsqueeze(-1))
        reads = []
        for factors, weight in ((read_factors, norm_weight), (read_factors2, norm_weight2)):
            if factors is None:
                reads.append(None)
                continue
            f = factors.to(dtype).float()
            x = f[0] * s[:, 0].float()
            for k in range(1, s.shape[1]):
                x = x + f[k] * s[:, k].float()
            x = x.to(dtype).float()
            variance = x.pow(2).mean(-1, keepdim=True)
            reads.append(weight * (x * torch.rsqrt(variance + eps)).to(dtype))
        return s.flatten(-2), reads[0], reads[1]

    def is_valid(
        self,
        streams: torch.Tensor,
        update: Optional[torch.Tensor],
        retention: Optional[torch.Tensor],
        write: Optional[torch.Tensor],
        read_factors: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        read_factors2: Optional[torch.Tensor] = None,
        norm_weight2: Optional[torch.Tensor] = None,
    ) -> None:
        # `eps` is named though nothing here reads it: it mirrors `__call__`.
        #
        # The kernel addresses everything from data_ptr() and two scalars it trusts: n, the
        # length of `read_factors`, and hidden, the length of `norm_weight`.
        n, hidden = read_factors.numel(), norm_weight.numel()
        assert streams.dtype == torch.bfloat16, (
            f"streams must be bf16, got {streams.dtype}: the op rounds every stream write and read "
            "to bf16, which fp32 (residual_in_fp32) streams never are"
        )
        assert streams.dim() == 2 and streams.shape[1] == n * hidden, (
            f"streams must be [T, n * hidden] = [T, {n} * {hidden}], got {tuple(streams.shape)}: "
            "the op reads stream k at column k * hidden of each row"
        )
        assert streams.stride(1) == 1 and streams.stride(0) >= streams.shape[1], (
            f"streams rows must be unit-stride and must not overlap, got strides {streams.stride()}"
        )
        for name, vec, size in (
            ("read_factors", read_factors, n),
            ("read_factors2", read_factors2, n),
            ("retention", retention, n),
            ("write", write, n),
            ("norm_weight", norm_weight, hidden),
            ("norm_weight2", norm_weight2, hidden),
        ):
            if vec is not None:
                assert vec.is_contiguous() and vec.numel() == size, (
                    f"{name} must be contiguous with exactly {size} elements, got shape "
                    f"{tuple(vec.shape)} contiguous={vec.is_contiguous()}: the op reads it flat"
                )
        if update is not None:
            assert update.shape == (streams.shape[0], hidden) and update.stride(1) == 1, (
                f"update must be [T, hidden] = [{streams.shape[0]}, {hidden}] with unit-stride "
                f"rows, got shape {tuple(update.shape)} strides {update.stride()}: the op reads "
                "row t of it for row t of the streams"
            )


wide_residual = WideResidual()
