# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the allreduce catalog entry.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and it starts one MPI rank per device named in
CUDA_VISIBLE_DEVICES under ``mpirun``, with a deadline it enforces by killing
the job. A collective that breaks hangs rather than raising, so the deadline is
what keeps a broken kernel from taking the calling run down with it.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce_op_matrix.py

The ranks are MPI ranks, not torch.distributed ones, because that is what a
tensor-parallel serving engine starts: its workers are an MPI world, the
entry's workspace is set up over a split of that world's communicator, and
`torch.ops.trtllm.mnnvl_fusion_allreduce` runs on it. The all-gather and
reduce-scatter matrices beside this file drive their ops' `_pg` symbols over
torch.distributed instead; an all-reduce certified that way would say nothing
about the path a tp4 rank takes.

What the matrix certifies, beyond each cell's sum: every rank receives the same
bits; a row's sum does not depend on how many rows share the call or where the
row sits among them (the property a CUDA-graph-padded decode step needs to
match an unpadded one); how the sum is formed -- fp32, ascending rank order,
one rounding -- against every rival order on a payload that tells them apart;
CUDA graph capture and replay at every decode height, interleaved with eager
calls of other shapes; and the guards.

Not a pytest module, despite the `check_*` bodies. They are one fixed
sequence inside a single 4-rank job rather than independent cases: each reads
module-global rank state that only `_run_one_rank` binds, and the workspace's
protocol state carries from one to the next. Neither this file's name nor its
function names match pytest's collection patterns.

The collected entry point is
`tests/unittest/_torch/modeling_v2/comm/test_modeling_v2_allreduce_op_matrix.py`:
it starts this job and turns its exit code into an assertion. The launcher
must not import `tensorrt_llm` -- that calls `MPI_Init`, and an MPI-initialized
process cannot start `mpirun` -- and the ranks reach the catalog by absolute
import, so neither half needs a package.
"""

import functools
import hashlib
import itertools
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "MODELING_V2_ALLREDUCE_PID_DIR"
DEADLINE_S = 900

# The decode heights an engine captures graphs at for this entry's cells: one
# graph per height, all alive in one pool, replayed as the served batch moves.
GRAPH_ROWS: Tuple[int, ...] = (1, 2, 4, 8, 16)
# All-reduces per captured graph, each reading its own input: one per
# row-parallel layer of a short stack.
SITES_PER_GRAPH = 4

# The op picks its one-shot kernel while rows * width * ranks * itemsize is at
# most this many bytes, its two-shot kernel above (allreduceOp.cpp,
# mnnvlFusionAllReduce, kOneShotSizeThreshold).
ONESHOT_MAX_BYTES = 64 * 1024 * 8 * 2

# Bound inside the rank body, never in the launcher: importing the entry pulls
# in tensorrt_llm, which calls MPI_Init at import.
entry: Any = None
allreduce: Callable[[torch.Tensor], torch.Tensor] = None  # type: ignore[assignment]
validating: Any = None
COMM: Any = None
RANK = 0
WORLD = 1
WORKSPACE: Dict[str, Any] = {}
BUF: Any = None
FLAGS: Any = None


def _payload(rank: int, rows: int, width: int, seed: int, kind: str = "normal") -> torch.Tensor:
    """One rank's partial: deterministic per (rank, seed, shape), bf16.

    Every rank can regenerate every other rank's payload -- cuRAND is
    deterministic for a given (seed, shape) -- which is what lets the reference
    be arithmetic rather than a second collective.

    `normal` is activation-like. `wide` is sign * 2**e * (1 + m/128) with e in
    [-24, 12]: exact in bf16, and spread so widely that the fp32 sum of four
    values depends on the order they are added in -- the payload that tells the
    rank orders apart.
    """
    g = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    if kind == "normal":
        return (torch.randn(rows, width, generator=g, device="cuda") * 4.0).to(torch.bfloat16)
    assert kind == "wide", kind
    e = torch.randint(-24, 13, (rows, width), generator=g, device="cuda").float()
    m = torch.randint(0, 128, (rows, width), generator=g, device="cuda").float()
    s = torch.randint(0, 2, (rows, width), generator=g, device="cuda").float() * 2 - 1
    return (s * torch.exp2(e) * (1 + m / 128)).to(torch.bfloat16)


def _all_payloads(rows: int, width: int, seed: int, kind: str = "normal") -> List[torch.Tensor]:
    return [_payload(r, rows, width, seed, kind) for r in range(WORLD)]


def _reference(rows: int, width: int, seed: int, kind: str = "normal") -> torch.Tensor:
    """The entry's own claim of what the sum is, from every rank's regenerated payload."""
    return entry.allreduce.reference(_all_payloads(rows, width, seed, kind))


def _assert_bits(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero, on the bits: the claim is a formula, not a band."""
    assert out.dtype == ref.dtype, f"{where}: dtype {out.dtype} != {ref.dtype}"
    assert out.shape == ref.shape, f"{where}: shape {tuple(out.shape)} != {tuple(ref.shape)}"
    a, b = out.contiguous().view(torch.int16), ref.contiguous().view(torch.int16)
    if not torch.equal(a, b):
        bad = (a != b).nonzero()
        first = tuple(bad[0].tolist())
        raise AssertionError(
            f"{where}: {bad.shape[0]} of {a.numel()} elements differ in their bits; first at "
            f"{first}: got {out[first].item()!r}, expected {ref[first].item()!r}"
        )


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _assert_same_bits_on_every_rank(out: torch.Tensor, where: str) -> None:
    digests = COMM.allgather(_digest(out))
    assert len(set(digests)) == 1, f"{where}: ranks received different bits: {digests}"


def _oneshot_max_rows(width: int) -> int:
    return ONESHOT_MAX_BYTES // (width * WORLD * torch.bfloat16.itemsize)


def _widths() -> List[int]:
    return sorted({cell.spec["width"] for cell in entry.allreduce.CELLS})


def _cell_rows(width: int) -> List[int]:
    return sorted(
        cell.spec["rows"] for cell in entry.allreduce.CELLS if cell.spec["width"] == width
    )


def _expect_refusal(call: Callable[[], Any], match: str, where: str) -> None:
    """The guard has to raise -- on every rank, before the op could run."""
    try:
        call()
    except AssertionError as exc:
        assert match in str(exc), f"{where}: refused for another reason: {exc}"
        return
    raise AssertionError(f"{where}: the guard admitted it")


def check_a_cold_first_call_captures_into_a_graph() -> None:
    """The workspace's first-ever call is inside a CUDA graph capture.

    The workspace is set up eagerly, collectively, before any op runs; this
    proves the op itself initializes nothing lazily on its first call that a
    capture could not record.
    """
    rows, width, seed = 4, 3072, 100
    x = _payload(RANK, rows, width, seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = allreduce(x)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bits(out, _reference(rows, width, seed), "first call, captured then replayed")
    COMM.Barrier()


def check_certified_cells() -> None:
    """Drive the entry's own cell list, inside `validating`.

    For every configuration the entry claims, the output is the entry's own
    `reference` bit for bit, on every rank; it is a new tensor of the input's
    shape and dtype; the input is untouched; and the guard admits the input.
    """
    for i, cell in enumerate(entry.allreduce.CELLS):
        rows, width = cell.spec["rows"], cell.spec["width"]
        seed = 1000 + i
        x = _payload(RANK, rows, width, seed)
        before = x.clone()
        with validating(entry.allreduce):
            out = allreduce(x)
        torch.cuda.synchronize()
        assert out.shape == x.shape and out.dtype is cell.spec["dtype"], (
            cell.why,
            out.shape,
            out.dtype,
        )
        assert out.data_ptr() != x.data_ptr(), f"{cell.why}: the op must return a new tensor"
        _assert_bits(x, before, f"{cell.why}: input")
        _assert_bits(out, _reference(rows, width, seed), cell.why)
        _assert_same_bits_on_every_rank(out, cell.why)
        COMM.Barrier()


def check_every_rank_receives_identical_bits() -> None:
    """On the order-revealing payload, at every cell's shape.

    A rank that formed the sum in its own order -- starting from its own value,
    say, as a kernel keeping the local value in registers might -- would get
    different bits on this payload than its peers. Every rank's output is hashed
    and the hashes are compared.
    """
    for i, cell in enumerate(entry.allreduce.CELLS):
        rows, width = cell.spec["rows"], cell.spec["width"]
        out = allreduce(_payload(RANK, rows, width, 2000 + i, "wide"))
        torch.cuda.synchronize()
        _assert_same_bits_on_every_rank(out, f"{cell.why} (wide payload)")
    COMM.Barrier()


def check_a_rows_sum_does_not_depend_on_the_rows_beside_it() -> None:
    """Batch invariance, bitwise, at every width.

    One payload of the most rows any cell has; then, at every cell height and
    on both sides of the one-shot / two-shot switch, a prefix, a window at an
    odd offset and a random subset of those rows are reduced on their own. Each
    row has to come back with the bits the full call gave it. The subset is the
    same on every rank, drawn from a generator every rank seeds alike.
    """
    for width in _widths():
        tmax = max(_cell_rows(width))
        seed = 3000 + width
        x = _payload(RANK, tmax, width, seed)
        full = allreduce(x)
        boundary = _oneshot_max_rows(width)
        heights = sorted(set(_cell_rows(width)) | {boundary, boundary + 1})
        gen = torch.Generator(device="cuda").manual_seed(seed)
        for t in heights:
            where = f"[{t}, {width}]"
            _assert_bits(allreduce(x[:t].contiguous()), full[:t], f"{where} prefix")
            if t + 3 <= tmax:
                _assert_bits(
                    allreduce(x[3 : 3 + t].contiguous()), full[3 : 3 + t], f"{where} at offset 3"
                )
            rows = torch.randperm(tmax, generator=gen, device="cuda")[:t]
            _assert_bits(allreduce(x[rows].contiguous()), full[rows], f"{where} random subset")
        _assert_bits(full, _reference(tmax, width, seed), f"[{tmax}, {width}] full")
        COMM.Barrier()


def check_the_sum_is_fp32_in_ascending_rank_order() -> None:
    """How the sum is formed, not just that it is stable.

    On the wide payload the fp32 sum of four values depends on the order they
    are added in, and a bf16 rounding after every add loses what fp32 keeps.
    The op is gated bitwise against the claim -- fp32, ascending rank order
    from +0.0, one rounding -- and every rival formula has to disagree with the
    claim somewhere on the same payload, or the gate could not tell them apart.
    The rivals: the other eleven distinct orders (the first add commutes, so an
    order and its first-pair swap are one formula), bf16 rounding after every
    add, and a pairwise tree. Heights on both sides of the kernel switch.
    """
    width = 3072
    for rows in (32, 64, 2048):
        seed = 4000 + rows
        xs = _all_payloads(rows, width, seed, "wide")
        out = allreduce(xs[RANK])
        torch.cuda.synchronize()
        claimed = entry.allreduce.reference(xs)
        where = f"[{rows}, {width}] wide payload"
        _assert_bits(out, claimed, where)

        def fp32_in(order: Sequence[int]) -> torch.Tensor:
            acc = torch.zeros(xs[0].shape, dtype=torch.float32, device="cuda")
            for r in order:
                acc = acc + xs[r].float()
            return acc.to(torch.bfloat16)

        rivals: Dict[str, torch.Tensor] = {}
        for order in itertools.permutations(range(WORLD)):
            if order[0] < order[1] and list(order) != list(range(WORLD)):
                rivals["fp32, ranks " + "".join(map(str, order))] = fp32_in(order)
        acc = xs[0]
        for x in xs[1:]:
            acc = (acc.float() + x.float()).to(torch.bfloat16)
        rivals["bf16 after every add"] = acc
        half = WORLD // 2
        lo, hi = xs[0].float(), xs[half].float()
        for r in range(1, half):
            lo = lo + xs[r].float()
        for r in range(half + 1, WORLD):
            hi = hi + xs[r].float()
        rivals["pairwise fp32 tree"] = (lo + hi).to(torch.bfloat16)
        for name, rival in rivals.items():
            assert not torch.equal(rival.view(torch.int16), claimed.view(torch.int16)), (
                f"{where}: the payload cannot tell '{name}' from the claimed formula"
            )
        COMM.Barrier()


def check_signed_zeros_subnormals_and_overflow() -> None:
    """The values a bitwise claim can trip on.

    -0.0 is the Lamport protocol's empty-slot marker, so the kernel reads a
    -0.0 input as +0.0 and a sum of zeros comes back +0.0; the reference's
    +0.0 start says the same. Columns of bf16 subnormals check the kernel keeps
    them -- a kernel built to flush them would return zero there -- and columns
    near bf16's largest value check a sum that overflows becomes the same
    infinity.
    """
    rows, width = 8, 3072
    tiny = torch.finfo(torch.bfloat16).tiny  # smallest normal, 2**-126
    big = torch.finfo(torch.bfloat16).max

    def special(rank: int) -> torch.Tensor:
        x = _payload(rank, rows, width, 5000)
        x[:, 0:64] = -0.0  # every rank -0.0
        x[:, 64:128] = -0.0 if rank % 2 else 0.0  # mixed signs of zero
        x[:, 128:192] = tiny / 32 * (rank + 1)  # subnormal inputs with a subnormal sum
        x[:, 192:256] = -tiny / 4 * (rank + 1)  # subnormal inputs with a normal sum
        x[:, 256:320] = tiny / 4 * (1 if rank % 2 else -1)  # subnormals that cancel
        x[:, 320:384] = big / 2  # overflows to +inf
        x[:, 384:448] = -big / 2  # overflows to -inf
        return x

    xs = [special(r) for r in range(WORLD)]
    out = allreduce(xs[RANK])
    torch.cuda.synchronize()
    ref = entry.allreduce.reference(xs)
    _assert_bits(out, ref, "special values")
    assert torch.equal(
        out[:, 0:128].view(torch.int16), torch.zeros_like(out[:, 0:128]).view(torch.int16)
    ), "a sum of zeros has to be +0.0"
    assert bool((out[:, 128:256] != 0).all()), "subnormal inputs were flushed"
    assert bool(torch.isinf(out[:, 320:448]).all()), "an overflowing sum has to be infinite"
    _assert_same_bits_on_every_rank(out, "special values")
    COMM.Barrier()


def _capture_per_height(widths: Sequence[int], seed: int):
    """One graph per (height, width), `SITES_PER_GRAPH` all-reduces each, one shared pool, all alive.

    Returns (inputs, outputs, graphs), keyed by (rows, width). Each site reads
    its own persistent input buffer and its output is copied into a persistent
    buffer inside the capture, so the payload of every site can be moved
    independently and read after a burst of replays.
    """
    inputs: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    outputs: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    graphs: Dict[Tuple[int, int], Any] = {}
    pool = None
    for width in widths:
        for rows in GRAPH_ROWS:
            key = (rows, width)
            xs = [_payload(RANK, rows, width, seed + 13 * s) for s in range(SITES_PER_GRAPH)]
            persistent = [torch.empty(rows, width, dtype=torch.bfloat16, device="cuda") for _ in xs]

            def body(
                xs: List[torch.Tensor] = xs, persistent: List[torch.Tensor] = persistent
            ) -> None:
                for x, dst in zip(xs, persistent):
                    dst.copy_(allreduce(x))

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            COMM.Barrier()
            graph = torch.cuda.CUDAGraph()
            ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
            with ctx:
                body()
            if pool is None:
                pool = graph.pool()
            inputs[key], outputs[key], graphs[key] = xs, persistent, graph
            COMM.Barrier()
    return inputs, outputs, graphs


def _fresh(
    inputs: Dict[Tuple[int, int], List[torch.Tensor]], key: Tuple[int, int], seed: int
) -> None:
    rows, width = key
    for site, x in enumerate(inputs[key]):
        x.copy_(_payload(RANK, rows, width, seed + 13 * site))


def _verify(outputs, key: Tuple[int, int], seed: int, where: str) -> None:
    rows, width = key
    for site, out in enumerate(outputs[key]):
        _assert_bits(
            out, _reference(rows, width, seed + 13 * site), f"{where} [{rows}, {width}] site {site}"
        )


def check_cuda_graph_at_every_decode_height() -> None:
    """Graphs at every decode height and width, replayed in four orders.

    That is the state an engine's graph runner leaves: one graph per batch
    size, every capture in the first one's pool, none destroyed, replayed
    interleaved as the served batch size moves. Each replay reads a fresh
    payload and every site's output is gated against the reference.
    """
    widths = _widths()
    inputs, outputs, graphs = _capture_per_height(widths, seed=6000)
    keys = list(graphs)
    shuffled = list(keys)
    random.Random(1234).shuffle(shuffled)
    for order_name, order in (
        ("ascending", keys),
        ("descending", list(reversed(keys))),
        ("shuffled", shuffled),
        ("the same graph twice running", [k for k in keys for _ in range(2)]),
    ):
        for step, key in enumerate(order):
            seed = 7000 + 101 * step
            _fresh(inputs, key, seed)
            graphs[key].replay()
            torch.cuda.synchronize()
            _verify(outputs, key, seed, f"replay ({order_name}, step {step})")
        COMM.Barrier()


def check_cuda_graph_replays_survive_eager_calls_of_other_shapes() -> None:
    """A decode graph replayed between eager prefill-sized calls.

    The workspace's Lamport buffers rotate on every call, eager or replayed, so
    an eager call between two replays moves the state the captured kernels
    read. Every replay still has to give the reference, and so does every eager
    call between them.
    """
    width = 3072
    inputs, outputs, graphs = _capture_per_height([width], seed=8000)
    for step, rows in enumerate((1024, 64, 8192, 1, 256)):
        key = (GRAPH_ROWS[step % len(GRAPH_ROWS)], width)
        seed = 8100 + 101 * step
        eager = allreduce(_payload(RANK, rows, width, seed + 50))
        _fresh(inputs, key, seed)
        graphs[key].replay()
        torch.cuda.synchronize()
        _assert_bits(
            eager, _reference(rows, width, seed + 50), f"eager [{rows}, {width}] between replays"
        )
        _verify(outputs, key, seed, f"replay after an eager [{rows}, {width}]")
        COMM.Barrier()


def check_wrapper_guards_a_message_larger_than_the_workspace() -> None:
    """Refused before the op runs: the op would write past the stage."""
    width = max(_widths())
    rows = max(_cell_rows(width)) + WORLD
    x = torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
    with validating(entry.allreduce):
        _expect_refusal(lambda: allreduce(x), "writes past the stage", f"[{rows}, {width}]")
    COMM.Barrier()


def check_wrapper_guards_control_words_from_another_buffer() -> None:
    """Refused: the op indexes its stages with whatever uint32 tensor it is handed."""
    x = _payload(RANK, 4, 3072, 9000)
    stray = torch.zeros_like(FLAGS)
    with validating(entry.allreduce):
        _expect_refusal(
            lambda: entry.allreduce(x, comm_buffer=BUF, buffer_flags=stray),
            "control words",
            "stray buffer_flags",
        )
    COMM.Barrier()


def check_wrapper_guards_a_buffer_from_no_workspace() -> None:
    """Refused: without the workspace's record the stage size cannot be checked."""
    x = _payload(RANK, 4, 3072, 9100)
    stray = torch.empty(1024, dtype=torch.bfloat16, device="cuda")
    with validating(entry.allreduce):
        _expect_refusal(
            lambda: entry.allreduce(x, comm_buffer=stray, buffer_flags=FLAGS),
            "unicast buffer",
            "stray comm_buffer",
        )
    COMM.Barrier()


def check_the_op_itself_refuses_a_derived_comm_buffer() -> None:
    """Why the guard does not check the buffer's base: the op already does, loudly.

    The op finds the workspace by the exact pointer it is handed, so a view
    into the buffer -- one stage of it, say -- is refused before any kernel
    runs, on every rank alike.
    """
    x = _payload(RANK, 4, 3072, 9200)
    try:
        entry.allreduce(x, comm_buffer=BUF[1], buffer_flags=FLAGS)
    except RuntimeError as exc:
        assert "mcastBuffer" in str(exc), f"refused for another reason: {exc}"
    else:
        raise AssertionError("the op took a pointer into the middle of its workspace")
    torch.cuda.synchronize()
    out = allreduce(x)
    torch.cuda.synchronize()
    _assert_bits(out, _reference(4, 3072, 9200), "a call after the refusal")
    COMM.Barrier()


CHECKS = (
    # Stays first: it is the only check that can observe the workspace's first call.
    check_a_cold_first_call_captures_into_a_graph,
    check_certified_cells,
    check_every_rank_receives_identical_bits,
    check_a_rows_sum_does_not_depend_on_the_rows_beside_it,
    check_the_sum_is_fp32_in_ascending_rank_order,
    check_signed_zeros_subnormals_and_overflow,
    check_cuda_graph_at_every_decode_height,
    check_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    check_wrapper_guards_a_message_larger_than_the_workspace,
    check_wrapper_guards_control_words_from_another_buffer,
    check_wrapper_guards_a_buffer_from_no_workspace,
    check_the_op_itself_refuses_a_derived_comm_buffer,
)


def _run_one_rank() -> int:
    """Body of one MPI rank: run every check; a failing rank exits nonzero and
    mpirun ends the job."""
    global entry, allreduce, validating, COMM, RANK, WORLD, WORKSPACE, BUF, FLAGS
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    torch.cuda.set_device(local)

    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK, WORLD = COMM.Get_rank(), COMM.Get_size()
    pid_dir = os.environ.get(_PID_DIR_ENV)
    if pid_dir:
        Path(pid_dir, f"rank{RANK}.pid").write_text(str(os.getpid()))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allreduce as _entry
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from tensorrt_llm.mapping import Mapping

    entry = _entry
    # The workspace a tp4 target builds at init: one per mapping, sized for the
    # largest message any cell sends.
    mapping = Mapping(world_size=WORLD, rank=RANK, tp_size=WORLD, gpus_per_node=WORLD)
    need = max(
        MNNVLAllReduce.get_required_workspace_size(
            c.spec["rows"], c.spec["width"], WORLD, c.spec["dtype"]
        )
        for c in entry.allreduce.CELLS
    )
    WORKSPACE = get_or_scale_allreduce_mnnvl_workspace(
        mapping, torch.bfloat16, buffer_size_bytes=need
    )
    BUF = WORKSPACE["uc_buffer"].view(torch.bfloat16).view(3, -1)
    FLAGS = WORKSPACE["buffer_flags"]
    # One partial instead of a per-call argument, so every check's call site
    # keeps the certified shape. The guard is armed on the entry itself.
    allreduce = functools.partial(entry.allreduce, comm_buffer=BUF, buffer_flags=FLAGS)

    # Run as a spawned rank, not collected by pytest, so the repo's
    # `__extra_import_path__` mechanism is not in play: the sibling helper is
    # reached the plain way, in these rank processes only.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import validating as _validating_cm

    validating = _validating_cm

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective
            # early wedges every sibling in it, and mpirun ends the job the
            # moment it sees this exit.
            os._exit(13)
    COMM.Barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    # Finalize so mpirun counts this as a clean exit, then leave without
    # running teardown: the multicast workspace and the captured graphs are
    # torn down with the process, and these are throwaway processes.
    MPI.Finalize()
    os._exit(0)


def _standalone_env() -> Dict[str, str]:
    """This process's environment, minus any enclosing launcher's identity.

    A launching srun step or MPI job leaves its rank identity in the
    environment (SLURM_*, PMIX_*, PMI_*, OMPI_*); a nested mpirun that inherited
    it would join that job instead of starting its own world.
    """
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    return env


def _spawn_ranks() -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); a collective needs at least 2"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; this matrix runs its ranks as an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="allreduce_matrix_")
    env = dict(_standalone_env(), **{_PID_DIR_ENV: pid_dir})
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(world_size)]
    if os.geteuid() == 0:
        command.insert(1, "--allow-run-as-root")
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    print(f"[launcher] {' '.join(command)}", flush=True)
    job = subprocess.Popen(command, env=env, start_new_session=True)

    def _kill_all() -> None:
        for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 0)):
            try:
                os.killpg(job.pid, sig)
            except ProcessLookupError:
                pass
            try:
                job.wait(timeout=grace or None)
                break
            except subprocess.TimeoutExpired:
                continue
        # Ranks forked by mpirun's daemon may sit in their own process groups.
        for pid_file in Path(pid_dir).glob("rank*.pid"):
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass

    deadline = time.monotonic() + DEADLINE_S
    while job.poll() is None:
        if time.monotonic() > deadline:
            _kill_all()
            raise AssertionError(
                f"the {world_size}-rank run did not finish in {DEADLINE_S}s (wedged)"
            )
        time.sleep(0.5)
    shutil.rmtree(pid_dir, ignore_errors=True)
    if job.returncode != 0:
        raise AssertionError(f"the {world_size}-rank run failed: mpirun exited {job.returncode}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
