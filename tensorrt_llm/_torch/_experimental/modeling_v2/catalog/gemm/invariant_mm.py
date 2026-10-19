# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense bf16 GEMM whose output rows do not depend on the batch: `y = x @ weight.T`, fp32 accumulation."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul


class _Tile(NamedTuple):
    """One tiling of the product; every tiling computes the same bits (see `InvariantMm.note`).

    `swap`: one [block_n weight rows, block_m activation rows] tile per program, the weight as the
    MMA's M operand, so a decode batch of a few rows fills a tile of its own size.
    `persistent`: [block_m activation rows, block_n weight rows] tiles over one program per SM.
    """

    kind: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    warp_specialize: bool = False
    subtile: bool = False


def _swap(block_m, block_n, block_k, num_stages, warp_specialize=False) -> _Tile:
    return _Tile("swap", block_m, block_n, block_k, 4, num_stages, warp_specialize)


def _persistent(block_n, block_k, num_warps, num_stages, subtile=False) -> _Tile:
    return _Tile("persistent", 128, block_n, block_k, num_warps, num_stages, True, subtile)


_ANY = 1 << 62

#: The tiles most shapes share between their decode tile and their largest row counts.
_MID = _swap(64, 128, 128, 4, True)
_LARGE = _persistent(256, 64, 8, 4, True)

#: (k, n) -> ((largest row count, tile), ...), ascending; the last entry takes every larger count.
#: Read off a sweep of each shape's candidate tiles on a fine grid of row counts (sm_103, every
#: weight streamed from HBM, under CUDA-graph replay). A tile is a speed choice only.
_TILES: dict[tuple[int, int], tuple[tuple[int, _Tile], ...]] = {
    # GDP in_proj
    (3072, 18528): ((16, _swap(16, 128, 128, 4)), (64, _MID), (_ANY, _LARGE)),
    # GDP out_proj
    (1536, 3072): ((96, _swap(32, 64, 256, 4, True)), (512, _MID), (_ANY, _LARGE)),
    # attention qkv
    (3072, 4608): ((64, _swap(32, 64, 256, 4)), (512, _MID), (_ANY, _LARGE)),
    # attention o_proj, shared-expert down
    (4096, 3072): ((96, _swap(32, 64, 256, 4, True)), (768, _MID), (_ANY, _LARGE)),
    # dense MLP up
    (3072, 12288): ((16, _swap(16, 128, 128, 4, True)), (64, _MID), (_ANY, _LARGE)),
    # dense MLP down
    (12288, 3072): ((96, _swap(32, 64, 128, 8, True)), (512, _MID), (_ANY, _LARGE)),
    # MoE router, fp32 out: the persistent tile's fp32 epilogue leaves room for three stages
    (3072, 512): (
        (256, _swap(16, 64, 256, 4, True)),
        (4096, _MID),
        (_ANY, _persistent(256, 64, 8, 3, True)),
    ),
    # fc1_latent_proj
    (3072, 768): (
        (192, _swap(16, 64, 256, 4, True)),
        (3072, _MID),
        (_ANY, _persistent(256, 64, 4, 4, True)),
    ),
    # fc2_latent_proj
    (768, 3072): ((96, _swap(32, 64, 256, 4)), (512, _MID), (_ANY, _LARGE)),
    # shared expert up
    (3072, 4096): ((64, _swap(32, 64, 256, 4, True)), (512, _MID), (_ANY, _LARGE)),
    # MTP eh_proj
    (6144, 3072): (
        (96, _swap(32, 64, 128, 8, True)),
        (384, _MID),
        (768, _persistent(128, 64, 8, 4)),
        (_ANY, _LARGE),
    ),
    # lm_head, fp32 out
    (3072, 262272): (
        (16, _swap(16, 128, 128, 6, True)),
        (32, _swap(32, 128, 128, 4, True)),
        (_ANY, _persistent(256, 64, 8, 3, True)),
    ),
    # a tp4 rank's GDP in_proj: 6 of the 24 heads
    (3072, 4632): (
        (16, _swap(16, 64, 256, 4, True)),
        (64, _swap(32, 64, 256, 4, True)),
        (128, _swap(64, 64, 128, 6, True)),
        (256, _MID),
        (512, _persistent(128, 128, 8, 3)),
        (_ANY, _persistent(256, 64, 4, 4, True)),
    ),
}

#: Any other (k, n): the same arithmetic, uncertified. Small enough for an fp32 epilogue.
_DEFAULT_TILES: tuple[tuple[int, _Tile], ...] = (
    (32, _swap(32, 64, 256, 4)),
    (_ANY, _persistent(128, 64, 8, 3)),
)

#: nemotron4-nano-preview's dense GEMMs, one row per distinct (k, n, out_dtype): (call sites, k, n,
#: out_dtype). All are bias-free on bf16 operands. The router and the lm_head ask for fp32 out because
#: the checkpoint runs both as fp32 GEMMs on bf16 values; their products are exact in fp32, so fp32
#: accumulation with fp32 out is that math up to summation order. A tp4 rank runs a quarter of most of
#: them; the one slice whose cuBLAS rows move with the row count -- its GDP in_proj -- is a row of
#: its own, and the replicated router and fc1_latent_proj keep the whole model's rows.
_NEMOTRON4_GEMMS: tuple[tuple[str, int, int, Optional[torch.dtype]], ...] = (
    ("GDP in_proj", 3072, 18528, None),
    ("GDP out_proj", 1536, 3072, None),
    ("attention qkv", 3072, 4608, None),
    ("o_proj and shared down", 4096, 3072, None),
    ("dense MLP up", 3072, 12288, None),
    ("dense MLP down", 12288, 3072, None),
    ("MoE router", 3072, 512, torch.float32),
    ("fc1_latent_proj", 3072, 768, None),
    ("fc2_latent_proj", 768, 3072, None),
    ("shared expert up", 3072, 4096, None),
    ("MTP eh_proj", 6144, 3072, None),
    ("lm_head", 3072, 262272, torch.float32),
    ("tp4 GDP in_proj", 3072, 4632, None),
)

#: The row counts every GEMM is certified at, and why each is a cell.
_ROWS = (
    (1, "decode, one token"),
    (3, "one request's MTP verify (1 + 2 drafts)"),
    (8, "a small decode batch"),
    (64, "a mid decode batch"),
    (256, "a large decode batch"),
    (1024, "the largest decode batch a CUDA graph captures"),
    (3072, "1024 requests verifying two drafts each"),
    (8192, "a prefill chunk at max_num_tokens 8192"),
)


def _cells() -> tuple[Cell, ...]:
    """One cell per (GEMM, row count) the forward runs; the lm_head sees only each request's last
    token and its drafts, so its rows stop at 3072."""
    return tuple(
        Cell(
            why=f"nemotron4 {site} m={m}: {k} -> {n}{', fp32 out' if out_dtype else ''}, {rows}",
            spec=dict(m=m, k=k, n=n, out_dtype=out_dtype),
        )
        for site, k, n, out_dtype in _NEMOTRON4_GEMMS
        for m, rows in _ROWS
        if not (site == "lm_head" and m > 3072)
    )


@triton.jit
def _swap_kernel(
    x_desc,
    w_desc,
    y_desc,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    OUT: tl.constexpr,
):
    """One [BLOCK_N weight rows, BLOCK_M activation rows] tile: K walked from 0 in BLOCK_K steps
    into one fp32 accumulator per output, stored transposed into y."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for ki in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
        w = w_desc.load([pid_n * BLOCK_N, ki * BLOCK_K])
        x = x_desc.load([pid_m * BLOCK_M, ki * BLOCK_K])
        acc = tl.dot(w, x.T, acc)
    y_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], tl.trans(acc).to(OUT))


@triton.jit
def _tile_ids(tile_id, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    """Grouped tile order: GROUP_M row blocks share each weight block while it is in L2."""
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(do_not_specialize=["M", "num_programs"])
def _persistent_kernel(
    x_desc,
    w_desc,
    y_desc,
    M,
    N,
    K,
    num_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SUBTILE: tl.constexpr,
    OUT: tl.constexpr,
):
    """[BLOCK_M, BLOCK_N] tiles over `num_programs` programs, each tile's K walked from 0 in
    BLOCK_K steps into one fp32 accumulator per output. The loads and the MMAs run in their own
    warps; with SUBTILE the epilogue stores the tile in two column halves."""
    start = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    # The epilogue recomputes its tile's ids rather than carrying the loop's, so the stores of
    # one tile can overlap the loads of the next.
    tile_id_c = start - num_programs
    for tile_id in tl.range(start, num_tiles, num_programs, flatten=True, warp_specialize=True):
        pid_m, pid_n = _tile_ids(tile_id, num_pid_in_group, num_pid_m, GROUP_M)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            x = x_desc.load([pid_m * BLOCK_M, ki * BLOCK_K])
            w = w_desc.load([pid_n * BLOCK_N, ki * BLOCK_K])
            acc = tl.dot(x, w.T, acc)
        tile_id_c += num_programs
        pid_m, pid_n = _tile_ids(tile_id_c, num_pid_in_group, num_pid_m, GROUP_M)
        if SUBTILE:
            acc = tl.permute(tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2)), (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            y_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc0.to(OUT))
            y_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N + BLOCK_N // 2], acc1.to(OUT))
        else:
            y_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(OUT))


_OUT = {torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


def _tile_for(m: int, k: int, n: int) -> _Tile:
    for largest, tile in _TILES.get((k, n), _DEFAULT_TILES):
        if m <= largest:
            return tile
    raise AssertionError("unreachable: the last tile takes every row count")


class InvariantMm(OpWrapper):
    """`x @ weight.T` with every output row's bits a function of that row of `x` and of `weight`
    alone -- not of how many rows share the call, nor of where in it the row sits.

    `x` is [M, K] bf16 with unit-stride rows; `weight` is [N, K] bf16 as the checkpoint stores it
    (no transpose). Returns a new [M, N] tensor in `out_dtype`: bf16 by default, or fp32.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Every tiling in `_TILES` computes the same bits. An output element is one fp32 accumulator fed
    by tcgen05 bf16 MMAs that walk K from 0 in steps of 16, in order, and K is never split across
    programs or warps. The tile's shape, which operand is the MMA's M side, persistence and warp
    specialization change only which program computes the element and when, so the tile may follow
    the row count -- a decode batch of a few rows takes a tile of its own height, a prefill chunk a
    128 x 256 one -- and the row's bits do not. That is a measured property of the MMA rather than
    a promise of the language: every tiling was compared bit for bit with the others on every
    shape, and the invariance test crosses each bucket boundary on every commit. A Triton that
    lowered some tile to another MMA would fail it loudly.

    cuBLAS's own unsplit kernels compute the same bits: `cublas_mm` agrees with this op wherever
    cuBLAS does not split K. Its rows move with the row count because at some counts it does, or
    picks another kernel -- which is why it is the rival the invariance test is checked against.

    The descriptors are built on the host, three per call: host time on an eager call, nothing
    under CUDA-graph replay. Built on the device they would cost GPU time on every launch.

    `_DEFAULT_TILES` serves any other (k, n) with the same arithmetic, but only the cells are
    certified. K and N must be multiples of 8 (16-byte descriptor rows, refused loudly otherwise).
    bf16 operands only; out_dtype bf16 or fp32. The tiles were chosen by measurement on sm_103;
    choosing others changes speed, never bits.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        m, k = x.shape
        n = weight.shape[0]
        out_dtype = out_dtype if out_dtype is not None else x.dtype
        out = _OUT.get(out_dtype)
        if out is None:
            raise ValueError(f"out_dtype must be bf16 or fp32, got {out_dtype}")
        y = torch.empty((m, n), dtype=out_dtype, device=x.device)
        if m == 0:
            return y
        tile = _tile_for(m, k, n)
        x_desc = TensorDescriptor.from_tensor(x, [tile.block_m, tile.block_k])
        w_desc = TensorDescriptor.from_tensor(weight, [tile.block_n, tile.block_k])
        if tile.kind == "swap":
            _swap_kernel[(triton.cdiv(n, tile.block_n), triton.cdiv(m, tile.block_m))](
                x_desc,
                w_desc,
                TensorDescriptor.from_tensor(y, [tile.block_m, tile.block_n]),
                k,
                BLOCK_M=tile.block_m,
                BLOCK_N=tile.block_n,
                BLOCK_K=tile.block_k,
                WARP_SPECIALIZE=tile.warp_specialize,
                OUT=out,
                num_warps=tile.num_warps,
                num_stages=tile.num_stages,
            )
        else:
            tiles = triton.cdiv(m, tile.block_m) * triton.cdiv(n, tile.block_n)
            programs = min(tiles, torch.cuda.get_device_properties(x.device).multi_processor_count)
            y_block_n = tile.block_n // 2 if tile.subtile else tile.block_n
            _persistent_kernel[(programs,)](
                x_desc,
                w_desc,
                TensorDescriptor.from_tensor(y, [tile.block_m, y_block_n]),
                m,
                n,
                k,
                programs,
                BLOCK_M=tile.block_m,
                BLOCK_N=tile.block_n,
                BLOCK_K=tile.block_k,
                GROUP_M=8,
                SUBTILE=tile.subtile,
                OUT=out,
                num_warps=tile.num_warps,
                num_stages=tile.num_stages,
            )
        return y

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """fp32 products and sums, TF32 off -- see `true_fp32_matmul` -- rounded once to the output."""
        with true_fp32_matmul():
            ref = x.float() @ weight.float().t()
        return ref.to(out_dtype if out_dtype is not None else x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> None:
        # `out_dtype` is named though nothing here reads it: it mirrors `__call__`, which refuses
        # anything but bf16 and fp32 itself.
        #
        # The kernels take K from `x` and read `weight` through a descriptor of its own shape: a
        # narrower weight's missing columns come back as zeros and a wider one's extra columns are
        # never read, so either is a plausible product of the wrong matrices.
        assert weight.shape[1] == x.shape[1], (
            f"x and weight must share K: x is {tuple(x.shape)}, weight {tuple(weight.shape)}"
        )
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
            f"x and weight must be bf16, got {x.dtype} and {weight.dtype}: fp32 operands would "
            "run as TF32, a precision this entry does not certify"
        )


invariant_mm = InvariantMm()
