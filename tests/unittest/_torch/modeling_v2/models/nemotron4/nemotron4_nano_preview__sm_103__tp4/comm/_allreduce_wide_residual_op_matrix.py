# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the nemotron4 tp4 target's allreduce_wide_residual entry.

A collective cannot be exercised in one process, so this script is its own launcher: run it
plainly and it starts one MPI rank per device named in CUDA_VISIBLE_DEVICES under ``mpirun``, with
a deadline it enforces by killing the job. A collective that breaks hangs rather than raising, so
the deadline is what keeps a broken kernel from taking the calling run down with it.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce_wide_residual_op_matrix.py

The ranks are an MPI world, as a tp4 serving engine's workers are, and the workspace is the one a
tp4 target builds, shared with plain `allreduce` calls as it is in a forward.

What the matrix certifies, cell by cell, against the unfused pair the op replaces (`allreduce`,
then `wide_residual` with its sum as the update, both catalog entries on the same workspace) and
against the entry's own `reference`:

  * the streams, bit for bit;
  * each normed read within `wide_residual`'s own band (its test's `_compare`): the op sums the
    same exact squares as the pair in another order, so a normed value may round the other way,
    by one of its ulps, and nothing else;
  * every rank receives the same bits, and the input is left untouched.

Beyond the cells: a row's outputs do not depend on the rows beside it, across the one-shot /
two-shot switch; CUDA graphs captured at the decode and verify heights replay the eager call bit
for bit, interleaved with eager calls; fused and plain calls of every size share the workspace in
any order; and the guards.

Not a pytest module, despite the `check_*` bodies: they are one fixed sequence inside a single
4-rank job, and the workspace's protocol state carries from one to the next. The collected entry
point is `test_modeling_v2_allreduce_wide_residual.py` beside this file. The launcher must not
import `tensorrt_llm` -- that calls `MPI_Init`, and an MPI-initialized process cannot start
`mpirun` -- and the ranks reach the entry by absolute import.
"""

import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

assert torch.cuda.is_available(), "allreduce_wide_residual requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "MODELING_V2_ALLREDUCE_WIDE_RESIDUAL_PID_DIR"
DEADLINE_S = 900

BF16 = torch.bfloat16

# The decode and MTP-verify heights a tp4 engine captures graphs at for this entry's cells.
GRAPH_ROWS: Tuple[int, ...] = (1, 3, 4, 12, 16, 48)

# `wide_residual`'s band on a normed read against its reference (its test's `_compare`, which
# derives it): 4 ulp at the row's largest magnitude, 0.5 ulp of relative RMS, and at most a 1e-4
# fraction of the outputs not bit-identical (at least 4, since one flip is already 3.3e-4 of a
# 3072-wide row).
ELEMENT_ULP = 4.0
RMS_ULP = 0.5
MAX_MISMATCH_FRACTION = 1e-4
MIN_ALLOWED_MISMATCHES = 4

# Plain all-reduces the tp4 forward sends on the same workspace between fused calls: a mixer's
# hidden-width partial, the MoE latent alone, and the MoE latent + shared partials packed.
PLAIN_WIDTHS = (3072, 768, 3840)

# Bound inside the rank body, never in the launcher: importing the entry pulls in tensorrt_llm,
# which calls MPI_Init at import.
entry: Any = None
allreduce: Any = None
wide_residual: Any = None
assert_within_ulp: Any = None
validating: Any = None
COMM: Any = None
RANK = 0
WORLD = 1
BUF: Any = None
FLAGS: Any = None
STATS: List[Dict[str, Any]] = []


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _step(spec: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """The step's parameters as `wide_residual`'s test draws them: bf16 coefficients from fp32
    logits spanning the checkpoint's ranges, norm weights around 0.9. Same on every rank."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, hidden = spec["streams"], spec["hidden"]

    def logits(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(n, generator=g, device="cuda")

    two = spec["reads"] == 2
    retention = (1.0 - 0.1 * torch.sigmoid(-logits(-3.0, 6.0))).to(BF16)
    write = (2.0 * torch.sigmoid(logits(-6.8, 3.4))).to(BF16)
    factors = torch.sigmoid(logits(-7.8, 4.0)).to(BF16)
    factors2 = torch.sigmoid(logits(-7.8, 4.0)).to(BF16)
    weight = (0.9 + 0.25 * torch.randn(hidden, generator=g, device="cuda")).to(BF16)
    weight2 = (0.9 + 0.25 * torch.randn(hidden, generator=g, device="cuda")).to(BF16)
    return dict(
        retention=retention,
        write=write,
        read_factors=factors,
        norm_weight=weight,
        eps=spec["eps"],
        read_factors2=factors2 if two else None,
        norm_weight2=weight2 if two else None,
    )


def _operands(spec: Dict[str, Any], seed: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Every rank's partial and the streams. Row magnitudes are log-uniform over [2**-10, 2**4],
    as in `wide_residual`'s test, and each rank's partial is half the row's scale, so the sum is
    an update of the row's size. Every rank regenerates every rank's partial for the reference."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, n, hidden = spec["tokens"], spec["streams"], spec["hidden"]
    scale = torch.exp2(-10.0 + 14.0 * torch.rand(tokens, 1, generator=g, device="cuda"))
    streams = (torch.randn(tokens, n * hidden, generator=g, device="cuda") * scale).to(BF16)
    partials = []
    for r in range(WORLD):
        gr = torch.Generator(device="cuda").manual_seed(seed * 977 + r + 1)
        partials.append(
            (0.5 * scale * torch.randn(tokens, hidden, generator=gr, device="cuda")).to(BF16)
        )
    return partials, streams


def _spec(tokens: int, reads: int) -> Dict[str, Any]:
    return dict(tokens=tokens, reads=reads, dtype=BF16, streams=3, hidden=3072, eps=1e-5)


def _fused(x: torch.Tensor, streams: torch.Tensor, step: Dict[str, Any]):
    return entry.allreduce_wide_residual(
        x,
        streams,
        step["retention"],
        step["write"],
        step["read_factors"],
        step["norm_weight"],
        step["eps"],
        step["read_factors2"],
        step["norm_weight2"],
        BUF,
        FLAGS,
    )


def _pair(x: torch.Tensor, streams: torch.Tensor, step: Dict[str, Any]):
    """What the target runs today: the all-reduce, then the wide residual step on its sum."""
    update = allreduce(x, BUF, FLAGS)
    return wide_residual(
        streams,
        update,
        step["retention"],
        step["write"],
        step["read_factors"],
        step["norm_weight"],
        step["eps"],
        step["read_factors2"],
        step["norm_weight2"],
    )


def _assert_bits(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
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


def _compare_normed(out: torch.Tensor, ref: torch.Tensor, where: str) -> int:
    """`wide_residual`'s band; returns how many outputs are not bit-identical."""
    assert out.dtype == ref.dtype == BF16 and out.shape == ref.shape, where
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP, ulp=2.0**-8)
    mismatches = int((out != ref).sum().item())
    allowed = max(MAX_MISMATCH_FRACTION * out.numel(), MIN_ALLOWED_MISMATCHES)
    assert mismatches <= allowed, (
        f"{where}: {mismatches} of {out.numel()} normed outputs differ; {allowed:g} may"
    )
    return mismatches


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _assert_same_bits_on_every_rank(out: torch.Tensor, where: str) -> None:
    digests = COMM.allgather(_digest(out))
    assert len(set(digests)) == 1, f"{where}: ranks received different bits: {digests}"


def _expect_refusal(call: Callable[[], Any], error: type, match: str, where: str) -> None:
    """The guard (or the op) has to raise -- on every rank, before any kernel runs."""
    try:
        call()
    except error as exc:
        assert match in str(exc), f"{where}: refused for another reason: {exc}"
        return
    raise AssertionError(f"{where}: admitted")


def check_certified_cells() -> None:
    """Drive the entry's own cell list, inside `validating`, against the reference and the pair.

    The streams are the reference's and the pair's bit for bit; each normed read is within
    `wide_residual`'s band of both; every rank holds the same bits; the input is untouched.
    """
    for cell in entry.allreduce_wide_residual.CELLS:
        seed = _seed(cell.why)
        step = _step(cell.spec, seed)
        partials, streams0 = _operands(cell.spec, seed + 1)
        x = partials[RANK]
        x_before = x.clone()
        ref_streams, ref, ref2 = entry.allreduce_wide_residual.reference(partials, streams0, **step)
        pair_streams = streams0.clone()
        _, pair, pair2 = _pair(x, pair_streams, step)
        streams = streams0.clone()
        with validating(entry.allreduce_wide_residual):
            out_streams, normed, normed2 = _fused(x, streams, step)
        torch.cuda.synchronize()
        where = cell.why
        assert out_streams is streams, f"{where}: the op must return the streams it writes"
        assert normed.data_ptr() not in (x.data_ptr(), streams.data_ptr()), where
        _assert_bits(x, x_before, f"{where}: input")
        _assert_bits(streams, ref_streams, f"{where}: streams against the reference")
        _assert_bits(streams, pair_streams, f"{where}: streams against the unfused pair")
        record = {"cell": where, "numel": normed.numel()}
        record["vs_reference"] = _compare_normed(normed, ref, f"{where}: read 1 vs reference")
        record["vs_pair"] = _compare_normed(normed, pair, f"{where}: read 1 vs the pair")
        _assert_same_bits_on_every_rank(streams, f"{where}: streams")
        _assert_same_bits_on_every_rank(normed, f"{where}: read 1")
        if cell.spec["reads"] == 2:
            record["vs_reference_2"] = _compare_normed(normed2, ref2, f"{where}: read 2 vs ref")
            record["vs_pair_2"] = _compare_normed(normed2, pair2, f"{where}: read 2 vs the pair")
            _assert_same_bits_on_every_rank(normed2, f"{where}: read 2")
        else:
            assert normed2 is None, f"{where}: one read returns no second"
        STATS.append(record)
        COMM.Barrier()


def check_a_rows_outputs_do_not_depend_on_the_rows_beside_it() -> None:
    """Every row of an 8192-row call (two-shot) against the same rows called alone or in fewer
    rows -- one-shot and two-shot heights, at the start and from an offset: identical bits.

    What a CUDA-graph-padded decode step needs to match an unpadded one, and an MTP verify
    step its decode."""
    for reads in (1, 2):
        spec = _spec(8192, reads)
        step = _step(spec, 41 + reads)
        partials, streams0 = _operands(spec, 43 + reads)
        x = partials[RANK]
        full = streams0.clone()
        _, normed, normed2 = _fused(x, full, step)
        for start, rows in ((0, 1), (0, 3), (5, 16), (0, 42), (7, 43), (100, 48), (0, 1024)):
            stop = start + rows
            part = streams0[start:stop].clone()
            _, y, y2 = _fused(x[start:stop], part, step)
            where = f"rows {start}:{stop} of 8192, {reads} read(s)"
            _assert_bits(part, full[start:stop], f"{where}: streams")
            _assert_bits(y, normed[start:stop], f"{where}: read 1")
            if reads == 2:
                _assert_bits(y2, normed2[start:stop], f"{where}: read 2")
        COMM.Barrier()


def check_cuda_graphs_replay_the_eager_bits() -> None:
    """One graph per (height, reads) in one pool, all alive, replayed in three orders with fresh
    partials and streams each time, with eager calls between replays: the graph's streams and
    reads equal an eager call's on the same inputs, bit for bit."""
    graphs: Dict[Tuple[int, int], Tuple[Any, torch.Tensor, torch.Tensor, Dict, Dict, Dict]] = {}
    pool = None
    for rows in GRAPH_ROWS:
        for reads in (1, 2):
            spec = _spec(rows, reads)
            step = _step(spec, 5000 + 10 * rows + reads)
            partials, streams0 = _operands(spec, 5100 + 10 * rows + reads)
            xbuf, sbuf = partials[RANK].clone(), streams0.clone()
            outs: Dict[str, Optional[torch.Tensor]] = {}

            def body(xbuf=xbuf, sbuf=sbuf, step=step, outs=outs) -> None:
                _, outs["y"], outs["y2"] = _fused(xbuf, sbuf, step)

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
            graphs[(rows, reads)] = (graph, xbuf, sbuf, outs, step, spec)
            COMM.Barrier()
    keys = list(graphs)
    shuffled = list(keys)
    random.Random(4321).shuffle(shuffled)
    for order_name, order in (
        ("ascending", keys),
        ("shuffled", shuffled),
        ("each twice running", [k for k in keys for _ in range(2)]),
    ):
        for i, key in enumerate(order):
            graph, xbuf, sbuf, outs, step, spec = graphs[key]
            partials, streams0 = _operands(spec, 7000 + 97 * i + key[0])
            xbuf.copy_(partials[RANK])
            sbuf.copy_(streams0)
            graph.replay()
            eager_streams = streams0.clone()
            _, y, y2 = _fused(partials[RANK], eager_streams, step)
            torch.cuda.synchronize()
            where = f"graph {key} ({order_name}, step {i})"
            _assert_bits(sbuf, eager_streams, f"{where}: streams")
            _assert_bits(outs["y"], y, f"{where}: read 1")
            if key[1] == 2:
                _assert_bits(outs["y2"], y2, f"{where}: read 2")
        COMM.Barrier()


def check_fused_and_plain_calls_share_the_workspace() -> None:
    """The forward's order and worse: fused and plain all-reduces of every size, one-shot and
    two-shot alternating, on one workspace. Each result against its reference."""
    sequence = [
        ("fused", 4, 1),
        ("plain", 8192, 3840),
        ("fused", 8192, 2),
        ("plain", 1, 768),
        ("fused", 43, 1),
        ("plain", 64, 3072),
        ("fused", 1, 2),
        ("fused", 48, 1),
        ("plain", 3, 3840),
        ("fused", 1024, 2),
        ("fused", 12, 2),
    ]
    for i, (kind, rows, arg) in enumerate(sequence):
        where = f"call {i}: {kind} T={rows}"
        if kind == "plain":
            g = [torch.Generator(device="cuda").manual_seed(900 + 31 * i + r) for r in range(WORLD)]
            parts = [
                (4.0 * torch.randn(rows, arg, generator=gr, device="cuda")).to(BF16) for gr in g
            ]
            out = allreduce(parts[RANK], BUF, FLAGS)
            torch.cuda.synchronize()
            _assert_bits(out, allreduce.reference(parts), where)
        else:
            spec = _spec(rows, arg)
            step = _step(spec, 800 + i)
            partials, streams0 = _operands(spec, 850 + i)
            ref_streams, ref, ref2 = entry.allreduce_wide_residual.reference(
                partials, streams0, **step
            )
            streams = streams0.clone()
            _, y, y2 = _fused(partials[RANK], streams, step)
            torch.cuda.synchronize()
            _assert_bits(streams, ref_streams, f"{where}: streams")
            _compare_normed(y, ref, f"{where}: read 1")
            if arg == 2:
                _compare_normed(y2, ref2, f"{where}: read 2")
        COMM.Barrier()


def check_guards() -> None:
    """Refused before any kernel runs, on every rank alike; the workspace works afterwards."""
    op = entry.allreduce_wide_residual
    step = _step(_spec(4, 1), 9000)
    partials, streams0 = _operands(_spec(4, 1), 9001)
    x = partials[RANK]
    with validating(op):
        stray_flags = torch.zeros_like(FLAGS)
        _expect_refusal(
            lambda: op(x, streams0.clone(), *_positional(step), BUF, stray_flags),
            AssertionError,
            "control words",
            "buffer_flags of another buffer",
        )
        stray_buffer = torch.empty(1024, dtype=BF16, device="cuda")
        _expect_refusal(
            lambda: op(x, streams0.clone(), *_positional(step), stray_buffer, FLAGS),
            AssertionError,
            "unicast buffer",
            "a comm_buffer from no workspace",
        )
        rows = TOO_MANY_ROWS
        big = torch.zeros(rows, 3072, dtype=BF16, device="cuda")
        big_streams = torch.zeros(rows, 3 * 3072, dtype=BF16, device="cuda")
        _expect_refusal(
            lambda: op(big, big_streams, *_positional(step), BUF, FLAGS),
            AssertionError,
            "writes past the stage",
            f"{rows} rows, one more than the workspace holds",
        )
        storage = torch.zeros(4 * 4 * 3072, dtype=BF16, device="cuda")
        aliased_streams = storage[: 4 * 3 * 3072].view(4, 3 * 3072)
        aliased_input = storage[3072 : 3072 + 4 * 3072].view(4, 3072)
        _expect_refusal(
            lambda: op(aliased_input, aliased_streams, *_positional(step), BUF, FLAGS),
            AssertionError,
            "overlaps the streams",
            "an input inside the streams",
        )
    # What the op refuses itself, loudly, which `wide_residual` would have taken and rounded.
    fp32_step = dict(step, retention=step["retention"].float())
    _expect_refusal(
        lambda: _fused(x, streams0.clone(), fp32_step),
        RuntimeError,
        "retention must be a contiguous bf16 tensor",
        "fp32 coefficients",
    )
    torch.cuda.synchronize()
    ref_streams, ref, _ = op.reference(partials, streams0, **step)
    streams = streams0.clone()
    _, y, _ = _fused(x, streams, step)
    torch.cuda.synchronize()
    _assert_bits(streams, ref_streams, "a call after the refusals: streams")
    _compare_normed(y, ref, "a call after the refusals: read 1")
    COMM.Barrier()


def _positional(step: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        step["retention"],
        step["write"],
        step["read_factors"],
        step["norm_weight"],
        step["eps"],
        step["read_factors2"],
        step["norm_weight2"],
    )


CHECKS = (
    check_certified_cells,
    check_a_rows_outputs_do_not_depend_on_the_rows_beside_it,
    check_cuda_graphs_replay_the_eager_bits,
    check_fused_and_plain_calls_share_the_workspace,
    check_guards,
)

#: The workspace is sized for this many rows at the widest width the forward sends (tp4's
#: max_num_tokens); `TOO_MANY_ROWS` is set from it in the rank body.
WORKSPACE_ROWS = 8192
TOO_MANY_ROWS = 0


def _run_one_rank() -> int:
    """Body of one MPI rank: run every check; a failing rank exits nonzero and mpirun ends the
    job."""
    global entry, allreduce, wide_residual, assert_within_ulp, validating
    global COMM, RANK, WORLD, BUF, FLAGS
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    torch.cuda.set_device(local)

    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK, WORLD = COMM.Get_rank(), COMM.Get_size()
    pid_dir = os.environ.get(_PID_DIR_ENV)
    if pid_dir:
        Path(pid_dir, f"rank{RANK}.pid").write_text(str(os.getpid()))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import (
        allreduce as _allreduce,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.wide_residual import (
        wide_residual as _wide_residual,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4.ops.comm import (  # noqa: E501
        allreduce_wide_residual as _entry,
    )
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from tensorrt_llm.mapping import Mapping

    entry, allreduce, wide_residual = _entry, _allreduce, _wide_residual
    # The workspace a tp4 target builds at init (`_allreduce_workspace`): one per mapping, sized
    # for its max_num_tokens rows at the widest width it sends.
    mapping = Mapping(world_size=WORLD, rank=RANK, tp_size=WORLD, gpus_per_node=WORLD)
    need = max(
        MNNVLAllReduce.get_required_workspace_size(WORKSPACE_ROWS, width, WORLD, BF16)
        for width in PLAIN_WIDTHS
    )
    workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, BF16, buffer_size_bytes=need)
    BUF = workspace["uc_buffer"].view(BF16).view(3, -1)
    FLAGS = workspace["buffer_flags"]
    # The fewest hidden-width rows the workspace cannot hold (it is sized for the widest message).
    rows = WORKSPACE_ROWS
    while (
        MNNVLAllReduce.get_required_workspace_size(rows + 1, 3072, WORLD, BF16)
        <= workspace["buffer_size_bytes"]
    ):
        rows += 1
    globals()["TOO_MANY_ROWS"] = rows + 1

    # Run as a spawned rank, not collected by pytest, so the repo's `__extra_import_path__`
    # mechanism is not in play: the shared helper is reached the plain way, in these ranks only.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import assert_within_ulp as _assert_within_ulp
    from _validating import validating as _validating_cm

    assert_within_ulp, validating = _assert_within_ulp, _validating_cm

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective early wedges every
            # sibling in it, and mpirun ends the job the moment it sees this exit.
            os._exit(13)
    COMM.Barrier()
    torch.cuda.synchronize()
    if RANK == 0:
        for record in STATS:
            print("CELL_STATS " + json.dumps(record), flush=True)
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    MPI.Finalize()
    os._exit(0)


def _standalone_env() -> Dict[str, str]:
    """This process's environment, minus any enclosing launcher's identity (a nested mpirun that
    inherited SLURM_* / PMIX_* / OMPI_* would join that job instead of starting its own)."""
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    return env


def _spawn_ranks() -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, f"CUDA_VISIBLE_DEVICES names {world_size} device(s); need 2+"
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; this matrix runs its ranks as an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="allreduce_wide_residual_matrix_")
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
