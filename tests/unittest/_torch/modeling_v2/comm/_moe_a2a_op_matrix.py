# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the moe_a2a_dispatch and moe_a2a_combine catalog entries.

A collective cannot be exercised in one process, so this script is its own launcher: run it plainly and it
starts one MPI rank per device named in CUDA_VISIBLE_DEVICES under ``mpirun``, with a deadline it enforces by
killing the job. A collective that breaks hangs rather than raising, so the deadline is what keeps a broken
kernel from taking the calling run down with it.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _moe_a2a_op_matrix.py

The ranks are MPI ranks, as a dep4 serving engine's are: the entries' workspace (`NVLinkOneSided`'s) is MNNVL
memory set up over that world.

What the matrix certifies, per cell (a vector of per-rank token counts) and on every rank: the dispatch hands
this rank, from every source rank, exactly the tokens that route one of their experts here, each row a
bit-exact copy of every payload (latent, expert ids, routing weights), and marks every other receive row's
expert ids -1; the combine then returns, for each of this rank's tokens, the sum of the partials the ranks
holding it put in their receive rows, within the band `_compare` derives, for fp32 and bf16 partials. Then
both, captured in a CUDA graph at every decode height and replayed on fresh data.

Not a pytest module, despite the `check_*` bodies: one fixed sequence inside a single 4-rank job, reading
module-global rank state that only `_run_one_rank` binds. The collected entry point is
`test_modeling_v2_moe_a2a_op_matrix.py` beside this file. The launcher must not import `tensorrt_llm` (that
calls `MPI_Init`, and an MPI-initialized process cannot start `mpirun`); the ranks reach the catalog by
absolute import.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

assert torch.cuda.is_available(), "moe_a2a requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "MODELING_V2_MOE_A2A_PID_DIR"
DEADLINE_S = 900

# Bound inside the rank body, never in the launcher: importing the entries pulls in tensorrt_llm.
dispatch_entry: Any = None
combine_entry: Any = None
validating: Any = None
assert_within_ulp: Any = None
ULP: Any = None
COMM: Any = None
RANK = 0
WORLD = 1
A2A: Any = None  # the NVLinkOneSided whose workspace the entries are called on


class _Cell:
    """One cell's tokens, every rank's, regenerated identically on every rank."""

    def __init__(
        self, index: int, sizes: Tuple[int, ...], latent: int, top_k: int, num_experts: int
    ):
        self.sizes, self.latent, self.top_k, self.num_experts = (
            list(sizes),
            latent,
            top_k,
            num_experts,
        )
        self.offsets = [sum(self.sizes[:i]) for i in range(WORLD + 1)]
        self.max_tokens = max(self.sizes)
        self.experts, self.payloads = [], []
        for r in range(WORLD):
            g = torch.Generator(device="cuda").manual_seed(1_000 * index + r)
            n = self.sizes[r]
            lat = torch.randn(n, latent, generator=g, device="cuda").to(torch.bfloat16)
            ids = torch.rand(n, num_experts, generator=g, device="cuda").topk(top_k, dim=1).indices
            ids = ids.to(torch.int32).contiguous()
            # Routing weights in (0, 1), distinct per row with overwhelming probability: a received row is found
            # by them, since where the kernel puts it is its own choice.
            wts = torch.rand(n, top_k, generator=g, device="cuda") * 0.999 + 1e-4
            self.experts.append(ids)
            self.payloads.append([lat, ids, wts])
        per_rank = num_experts // WORLD
        self.routes = [
            (torch.cat(self.experts).long() // per_rank == r).any(dim=1) for r in range(WORLD)
        ]
        # Each rank's window partial for every token (zero where the token routes nothing to it), fp32.
        t_all = self.offsets[-1]
        self.partials = []
        for r in range(WORLD):
            g = torch.Generator(device="cuda").manual_seed(5_000_000 + 1_000 * index + r)
            p = (
                torch.randn(t_all, latent, generator=g, device="cuda")
                * self.routes[r].unsqueeze(1).float()
            )
            self.partials.append(p)

    def mine(self, r: int) -> slice:
        return slice(self.offsets[r], self.offsets[r + 1])


def _dispatch(cell: _Cell) -> Tuple[List[torch.Tensor], int]:
    lat, ids, wts = cell.payloads[RANK]
    return dispatch_entry.moe_a2a_dispatch(
        token_selected_experts=ids,
        input_payloads=[lat, ids, wts],
        workspace=A2A.workspace,
        metainfo=A2A.moe_a2a_metainfo,
        runtime_max_tokens_per_rank=cell.max_tokens,
        ep_rank=RANK,
        ep_size=WORLD,
        top_k=cell.top_k,
        num_experts=cell.num_experts,
        expert_id_payload_index=1,
        invalid_token_expert_id=-1,
    )


def _combine(cell: _Cell, payload: torch.Tensor, offset: int) -> torch.Tensor:
    return combine_entry.moe_a2a_combine(
        payload=payload,
        local_num_tokens=cell.sizes[RANK],
        workspace=A2A.workspace,
        metainfo=A2A.moe_a2a_metainfo,
        runtime_max_tokens_per_rank=cell.max_tokens,
        ep_rank=RANK,
        ep_size=WORLD,
        top_k=cell.top_k,
        combine_payload_offset=offset,
        payload_in_workspace=False,
    )


def _row_keys(w: torch.Tensor) -> List[bytes]:
    return [bytes(row) for row in w.contiguous().view(torch.int32).cpu().numpy()]


def _received_tokens(cell: _Cell, recv: List[torch.Tensor]) -> torch.Tensor:
    """Check the dispatch against the entry's reference; return each receive row's global token index (-1 for
    rows no token was written to)."""
    ref = dispatch_entry.moe_a2a_dispatch.reference(
        cell.experts, cell.payloads, RANK, WORLD, cell.num_experts
    )
    lat_r, ids_r, wts_r = (t.view(WORLD, cell.max_tokens, -1) for t in recv)
    token_of_row = torch.full((WORLD, cell.max_tokens), -1, dtype=torch.long)
    for s in range(WORLD):
        want_lat, want_ids, want_wts = ref[s]
        count = want_lat.shape[0]
        # Every row past the count is marked: its expert ids are all -1.
        assert bool((ids_r[s, count:] == -1).all()), (
            f"source {s}: rows past {count} keep expert ids"
        )
        # The first `count` rows hold exactly the routed tokens, each a bit-exact copy, in some order.
        index = {k: i for i, k in enumerate(_row_keys(want_wts))}
        got = _row_keys(wts_r[s, :count])
        order = [index.get(k, -1) for k in got]
        assert -1 not in order and len(set(order)) == count, (
            f"source {s}: received rows are not the routed set"
        )
        perm = torch.tensor(order, dtype=torch.long, device="cuda")
        for name, g, w in (
            ("latent", lat_r, want_lat),
            ("ids", ids_r, want_ids),
            ("weights", wts_r, want_wts),
        ):
            a = g[s, :count].contiguous().view(torch.uint8)
            b = w.index_select(0, perm).contiguous().view(torch.uint8)
            assert torch.equal(a, b), f"source {s}: a received {name} row differs from the sent one"
        routed_idx = torch.nonzero(cell.routes[RANK][cell.mine(s)]).flatten().cpu()
        token_of_row[s, :count] = (
            routed_idx[torch.tensor(order, dtype=torch.long)] + cell.offsets[s]
        )
    return token_of_row


def _payload(cell: _Cell, token_of_row: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    rows = token_of_row.flatten().to("cuda")
    p = cell.partials[RANK].index_select(0, rows.clamp_min(0)) * (rows >= 0).unsqueeze(1).float()
    return p.view(WORLD, cell.max_tokens, cell.latent).to(dtype).contiguous()


def _compare(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, where: str) -> None:
    """The combine against the fp64 sum of the same partials, rounded once.

    fp32: the kernel adds at most four fp32 partials in a pairwise tree, each add rounding once, against an
    exact sum: two roundings deep, so 2 fp32 ulp of the row's scale bounds every element and 1 ulp the
    relative RMS. bf16: the partials are bf16, and four of them add exactly in fp32 unless their exponents span
    more than 16 bits, so the kernel's one rounding to bf16 is the reference's; 1 ulp element-wise allows for
    the rare inexact add. A wrong token, a missing rank or a doubled partial misses either band by orders.
    """
    assert out.dtype == dtype and out.shape == ref.shape, (where, out.dtype, out.shape, ref.shape)
    if out.numel() == 0:
        return
    if dtype == torch.float32:
        assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=1.0, ulp=ULP[torch.float32])
    else:
        assert_within_ulp(out, ref, element_ulp=1.0, rms_ulp=0.5, ulp=ULP[torch.bfloat16])


def _dtypes() -> List[torch.dtype]:
    return sorted({c.spec["dtype"] for c in combine_entry.moe_a2a_combine.CELLS}, key=str)


def check_certified_cells() -> None:
    """Every dispatch cell, then the combine at each of its dtypes, under the entries' guards."""
    combine_cells = {
        (tuple(c.spec["sizes"]), c.spec["dtype"]) for c in combine_entry.moe_a2a_combine.CELLS
    }
    for i, c in enumerate(dispatch_entry.moe_a2a_dispatch.CELLS):
        sp = c.spec
        assert len(sp["sizes"]) == WORLD, (
            f"cell {c.why!r} is certified at world size {len(sp['sizes'])}"
        )
        cell = _Cell(i, sp["sizes"], sp["latent"], sp["top_k"], sp["num_experts"])
        for dtype in _dtypes():
            assert (tuple(sp["sizes"]), dtype) in combine_cells, (
                f"no combine cell for {sp['sizes']} {dtype}"
            )
            with validating(dispatch_entry.moe_a2a_dispatch, combine_entry.moe_a2a_combine):
                recv, offset = _dispatch(cell)
                torch.cuda.synchronize()
                token_of_row = _received_tokens(cell, recv)
                out = _combine(cell, _payload(cell, token_of_row, dtype), offset)
                torch.cuda.synchronize()
            mine = cell.mine(RANK)
            ref = combine_entry.moe_a2a_combine.reference(
                [p[mine].to(dtype) for p in cell.partials], dtype
            )
            _compare(out, ref, dtype, f"cell {c.why!r}, {dtype}")
        COMM.Barrier()


def check_cuda_graph_replay_at_every_decode_height() -> None:
    """Dispatch, a per-row partial, and combine, captured once per decode height, all graphs alive in one pool,
    replayed in a shuffled order on fresh latents written into the captured input on every rank.

    The receive row a token lands in is the kernel's choice and can change from one call to the next, so a
    payload has to be computed from the rows the same dispatch filled -- as the experts call does in a target.
    Here each rank's partial for a received row is the row's latent times (rank + 1): exact in fp32, so the
    combined row is exactly the token's latent times the sum of (r + 1) over the ranks it routes to, and the
    check is bitwise. Before each replay the captured receive buffer and output are zeroed, so a replay that
    skipped either op fails."""
    import random

    heights = [
        c
        for c in dispatch_entry.moe_a2a_dispatch.CELLS
        if len(set(c.spec["sizes"])) == 1 and c.spec["sizes"][0] <= 1024
    ]
    pool = torch.cuda.graph_pool_handle()
    graphs = []

    def step(cell: _Cell):
        recv, offset = _dispatch(cell)
        payload = (recv[0].float() * float(RANK + 1)).view(WORLD, cell.max_tokens, cell.latent)
        return recv, _combine(cell, payload.contiguous(), offset)

    for i, c in enumerate(heights):
        sp = c.spec
        cell = _Cell(100 + i, sp["sizes"], sp["latent"], sp["top_k"], sp["num_experts"])
        step(cell)  # the shapes, eagerly
        torch.cuda.synchronize()
        COMM.Barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            recv_g, out_g = step(cell)
        torch.cuda.synchronize()
        graphs.append((cell, g, recv_g, out_g))
    order = list(range(len(graphs))) * 2
    random.Random(7).shuffle(order)
    for rep, k in enumerate(order):
        cell, g, recv_g, out_g = graphs[k]
        for r in range(WORLD):
            gen = torch.Generator(device="cuda").manual_seed(90_000 + 100 * rep + r)
            fresh = torch.randn(cell.payloads[r][0].shape, generator=gen, device="cuda").to(
                torch.bfloat16
            )
            cell.payloads[r][0].copy_(fresh)  # this rank's copy is the captured input
        recv_g[0].zero_()
        out_g.zero_()
        torch.cuda.synchronize()
        COMM.Barrier()
        g.replay()
        torch.cuda.synchronize()
        _received_tokens(cell, recv_g)
        mine = cell.mine(RANK)
        weight = sum(cell.routes[r][mine].float() * float(r + 1) for r in range(WORLD))
        ref = cell.payloads[RANK][0].float() * weight.unsqueeze(1)
        assert torch.equal(out_g, ref), (
            f"graph replay {rep} at {cell.sizes}: {int((out_g != ref).sum())} of {ref.numel()} "
            f"elements differ from the exact sum"
        )
        COMM.Barrier()


CHECKS = (
    check_certified_cells,
    check_cuda_graph_replay_at_every_decode_height,
)


def _run_one_rank() -> int:
    """Body of one MPI rank: run every check; a failing rank exits nonzero and mpirun ends the job."""
    global dispatch_entry, combine_entry, validating, assert_within_ulp, ULP, COMM, RANK, WORLD, A2A
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    torch.cuda.set_device(local)

    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK, WORLD = COMM.Get_rank(), COMM.Get_size()
    pid_dir = os.environ.get(_PID_DIR_ENV)
    if pid_dir:
        Path(pid_dir, f"rank{RANK}.pid").write_text(str(os.getpid()))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        moe_a2a_combine as _combine_entry,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        moe_a2a_dispatch as _dispatch_entry,
    )
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm.mapping import Mapping

    dispatch_entry, combine_entry = _dispatch_entry, _combine_entry
    # The workspace a dep4 target builds at init, sized for the largest cell; fp32 sizes the combine region for
    # the wider of the two certified partial dtypes.
    cells = dispatch_entry.moe_a2a_dispatch.CELLS
    mapping = Mapping(
        world_size=WORLD,
        rank=RANK,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        enable_attention_dp=True,
        gpus_per_node=WORLD,
    )
    A2A = NVLinkOneSided(
        mapping,
        cells[0].spec["num_experts"],
        cells[0].spec["top_k"],
        max(max(c.spec["sizes"]) for c in cells),
        hidden_size=cells[0].spec["latent"],
        dtype=torch.float32,
    )

    # Run as a spawned rank, not collected by pytest, so `__extra_import_path__` is not in play: the sibling
    # helper is reached the plain way, in these rank processes only.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import ULP as _ULP
    from _validating import assert_within_ulp as _assert_within_ulp
    from _validating import validating as _validating_cm

    validating, assert_within_ulp, ULP = _validating_cm, _assert_within_ulp, _ULP

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective early wedges every sibling in it,
            # and mpirun ends the job the moment it sees this exit.
            os._exit(13)
    COMM.Barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    MPI.Finalize()
    os._exit(0)


def _standalone_env() -> Dict[str, str]:
    """This process's environment, minus any enclosing launcher's identity (a nested mpirun that inherited an
    srun step's or MPI job's rank variables would join that job instead of starting its own world)."""
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    return env


def _spawn_ranks() -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); a collective needs at least 2"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; this matrix runs its ranks as an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="moe_a2a_matrix_")
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
