# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 4-rank job behind `test_modeling_v2_dep4_moe_block.py`: the dep4 target's routed MoE
block, run on every rank of one node and measured there.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _dep4_moe_block_job.py --out <json> [--checkpoint <dir>]

Run plainly, this script is its own launcher: it starts one MPI rank per device under `mpirun`,
with a deadline it enforces by killing the job, because a collective that breaks hangs rather
than raising. The ranks are an MPI world since that is what a dep4 serving engine starts, and
the block's all-gather and reduce-scatter run on the MPI-session symbols over it.

Every rank builds the dep4 core from its own constructor (`Nemotron4NanoPreviewDep4ModelingV2`
with rank r's dep4 `Mapping`) over the target's weight table, every parameter declared on the
meta device. The MoE layers under test are filled from the checkpoint through the table's
manifest, as `load` fills them: the rank's own window of 128 experts of each stack. Then
`PrefillTarget._moe` or `DecodeTarget._moe` runs on the rank's rows of each step, while the other
three ranks run theirs, which is the served step's sequence of collectives: the one-sided
all-to-all on a balanced step, the packed all-gather and the reduce-scatter on a ragged one
(`M._moe_comm`), and a balanced decode step on the kernels a CUDA graph replays (`graph`). The
inputs are the model's own (`_nemotron4_moe_block.real_moe_inputs`), and the references hold every
one of the 512 experts.

What each rank records, for the rows whose output it owns (gated by the test):

* `cells`: a step's per-rank token counts, decode and prefill, padded and ragged. Per rank:
  the routing against HF's router; the block against the references; each op on the target's
  own input (`fc1`, `dispatch`, `window`, `experts`, `ep`, `fc2_norm`, `tail`); and the block
  against its single-GPU twin -- the same `_moe` on its all-gather path with the collectives
  taken out and the window widened to every expert, which is the order HF sums a token's ten
  experts in;
* `idle`: steps where a rank has no request and runs the attention-DP dummy, and graph-padded
  ranks: the real rows' outputs under three contents of every non-real row, each against the
  first's;
* `graph`: `DecodeTarget._moe` captured in a CUDA graph at each decode height on all four ranks
  and replayed on other rows, against the eager call on the same kernels;
* `rivals`: one planted mistake at a time, measured as a cell.

Rank 0 writes every rank's records to `--out` as JSON.

Not a pytest module: the ranks run one fixed sequence of collectives in lockstep, and neither
this file's name nor its functions match pytest's collection patterns. The launcher must not
import `tensorrt_llm` -- that calls `MPI_Init`, and an MPI-initialized process cannot start
`mpirun` -- so everything that does is imported inside the rank body.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "NEMOTRON4_DEP4_MOE_BLOCK_PID_DIR"
DEADLINE_S = 2400
"""Ceiling on the whole job. The test's layers measured 55-87 s in the ranks; deeper layers take
longer, since every rank first runs the reference prefix up to the deepest."""

WORLD_SIZE = 4
"""dep4: attention DP over the four GPUs of one node, the routed experts split four ways."""

LAYERS = (5, 6)
"""The MoE layers the test measures: the first shortcut attention block and the first shortcut
GDP block. The prefix to them (the embedding, three GDP mixers, two dense MLPs) is cheap.
`--layers` measures others; every rank then holds the reference prefix up to the deepest."""

MANY_ROWS = 64
"""A prefill count below this is a short prompt's tail; at or above it, a chunk of a long one."""


@dataclass(frozen=True)
class Step:
    """One step of the attention-DP group, as the routed MoE sees it.

    `counts`: each rank's rows, in rank order -- what `all_rank_num_tokens` carries.
    `phases`: per rank, "P" for `PrefillTarget` (the rank has context rows) or "D" for
    `DecodeTarget` (generation rows only).
    """

    name: str
    counts: tuple[int, ...]
    phases: str


#: Graph decode: every rank padded to one common height, so the collectives take sizes=None.
_DECODE_HEIGHTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

CELLS: tuple[Step, ...] = (
    *(Step(f"graph decode, {b} per rank", (b,) * WORLD_SIZE, "DDDD") for b in _DECODE_HEIGHTS),
    Step("eager decode, busiest rank 0", (32, 17, 1, 5), "DDDD"),
    Step("eager decode, busiest rank 1", (1, 256, 129, 33), "DDDD"),
    Step("eager decode, busiest rank 2", (513, 65, 1024, 1), "DDDD"),
    Step("rank 3 prefills 2048 beside three decoding ranks", (1, 3, 1, 2048), "DDDP"),
    Step("rank 0 prefills a 37-token tail, the others one token", (37, 1, 1, 1), "PDDD"),
    Step("ragged prefill", (2048, 1024, 517, 37), "PPPP"),
    Step("every rank prefills 2048", (2048,) * WORLD_SIZE, "PPPP"),
    Step("every rank prefills 8192", (8192,) * WORLD_SIZE, "PPPP"),
)
"""The steps measured. Decode heights are the CUDA-graph batch sizes up to 1024 per rank; the
eager splits are the comm entries' ragged cells, which move the busiest rank through every
position; prefill takes a 2048-token chunk on one rank or every rank, a ragged mix, and the
8192-token `max_num_tokens` baseline on every rank (32768 gathered rows)."""


@dataclass(frozen=True)
class IdleStep:
    """A step in which some rows are no request's: `real[r]` of rank r's `counts[r]` rows are real
    requests' tokens, the rest an idle rank's attention-DP dummy or CUDA-graph padding."""

    name: str
    counts: tuple[int, ...]
    real: tuple[int, ...]
    phases: str


IDLE_STEPS: tuple[IdleStep, ...] = (
    IdleStep(
        "graph decode, 4 per rank: rank 3 idle, ranks 1 and 2 padded",
        (4,) * 4,
        (4, 2, 1, 0),
        "DDDD",
    ),
    IdleStep(
        "graph decode, 64 per rank: rank 2 idle, the rest padded", (64,) * 4, (64, 33, 0, 7), "DDDD"
    ),
    IdleStep(
        "eager: rank 0 prefills 37, rank 1 decodes, ranks 2 and 3 idle",
        (37, 1, 1, 1),
        (37, 1, 0, 0),
        "PDDD",
    ),
    IdleStep(
        "eager: ranks 0 to 2 prefill 2048, rank 3 idle",
        (2048, 2048, 2048, 1),
        (2048, 2048, 2048, 0),
        "PPPD",
    ),
)

FILLERS = ("pool rows, draw A", "pool rows, draw B", "zeros")
"""What a non-real row holds, one run each: two different hidden states of the model's own (an
idle rank's dummy is a real generation row to the model, so its MoE input is a hidden state),
and zeros (the router then selects on the correction bias alone). Each routes elsewhere, so the
experts' group sizes change between runs."""

GRAPH_HEIGHTS = _DECODE_HEIGHTS
"""Per-rank heights `DecodeTarget._moe` is captured at, as the engine captures its decode
graphs."""

RIVAL_STEP = Step("every rank prefills 2048", (2048,) * WORLD_SIZE, "PPPP")
"""The rivals run at the first measured layer, on this balanced step (the all-to-all)..."""
RAGGED_RIVAL_STEP = Step("ragged prefill", (2048, 1024, 517, 129), "PPPP")
"""...except the all-gather path's own mistakes, which run on this ragged one: every rank holds at
least `MANY_ROWS`, where the block band is the tighter one."""
RIVALS = (
    "routing weights not scaled by routed_scaling_factor",
    "routing weights rounded to bf16",
    "selection without the correction bias",
    "the routing weight applied after down_proj",
    "fc2_norm left out",
    "fc2_norm rounded twice, as HF rounds it",
    "the shared expert reading the routed input",
    "bf16 window partial sums",
    "every rank running experts 0..127",
    "routing weights dispatched as bf16",
    "the window partials kept, not combined",
    "routing weights all-gathered as bf16",
    "the window partials kept, not reduce-scattered",
)
"""One plausible mistake each, planted in the target's block on every rank; the test names the
gates each must break."""
RAGGED_RIVALS = RIVALS[-2:]


# ----------------------------------------------------------------------------------------------
# Launcher (no tensorrt_llm import on this side)
# ----------------------------------------------------------------------------------------------


def _standalone_env() -> dict[str, str]:
    """This process's environment, minus any enclosing launcher's identity.

    A launching srun step or MPI job leaves its rank identity in the environment (SLURM_*,
    PMIX_*, PMI_*, OMPI_*); a nested mpirun that inherited it would join that job instead of
    starting its own world.
    """
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; this file moved "
        "without its parents[] index following"
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH")) if p)
    return env


def spawn_ranks(out: str, checkpoint: str, layers=LAYERS) -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the four devices this run owns"
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    assert len(devices) >= WORLD_SIZE, (
        f"dep4 runs on {WORLD_SIZE} devices; CUDA_VISIBLE_DEVICES names {len(devices)}"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; the ranks are an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="dep4_moe_block_")
    env = dict(
        _standalone_env(),
        CUDA_VISIBLE_DEVICES=",".join(devices[:WORLD_SIZE]),
        **{_PID_DIR_ENV: pid_dir},
    )
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(WORLD_SIZE)]
    if os.geteuid() == 0:
        command.insert(1, "--allow-run-as-root")
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    command += ["--out", out, "--checkpoint", checkpoint, "--layers", *map(str, layers)]
    print(f"[launcher] {' '.join(command)}", flush=True)
    job = subprocess.Popen(command, env=env, start_new_session=True)

    def kill_all() -> None:
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
            kill_all()
            raise AssertionError(
                f"the {WORLD_SIZE}-rank job did not finish in {DEADLINE_S}s (wedged)"
            )
        time.sleep(0.5)
    shutil.rmtree(pid_dir, ignore_errors=True)
    if job.returncode != 0:
        raise AssertionError(f"the {WORLD_SIZE}-rank job failed: mpirun exited {job.returncode}")


# ----------------------------------------------------------------------------------------------
# Rank body
# ----------------------------------------------------------------------------------------------


def _rank_body(out_path: str, checkpoint_path: str, layers) -> None:
    """Everything one rank does; every collective in lockstep with the other three."""
    import inspect
    from types import SimpleNamespace

    import pytest
    import torch
    from mpi4py import MPI
    from torch import nn

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == WORLD_SIZE, f"dep4 needs {WORLD_SIZE} ranks, got {comm.Get_size()}"

    # Spawned by mpirun rather than collected by pytest, so the tree's `__extra_import_path__`
    # is not in play: the family's helpers and `_validating` are reached the plain way.
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here.parents[1]), str(here.parents[3])]

    import _nemotron4_moe_block as K
    import _nemotron4_reference as R
    from _validating import validating
    from transformers import AutoConfig

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.softclamp_relu2 import (
        softclamp_relu2,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.moe_a2a_combine import (
        moe_a2a_combine,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.moe_a2a_dispatch import (
        moe_a2a_dispatch,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import (
        reducescatter,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.dsv3_router_gemm import (
        dsv3_router_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_direct_mm import (
        lowm_direct_mm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.lowm_splitk_mm import (
        lowm_splitk_mm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
        latent_moe_experts,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts_fused_act import (  # noqa: E501
        latent_moe_experts_fused_act,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
        flashinfer_rmsnorm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
        weights as W,
    )
    from tensorrt_llm.mapping import Mapping

    started = time.monotonic()

    def say(message: str) -> None:
        if rank == 0:
            print(f"[job {time.monotonic() - started:7.1f}s] {message}", flush=True)

    # ------------------------------------------------------------------------------------------
    # The core, its weights and the inputs
    # ------------------------------------------------------------------------------------------

    checkpoint = R.Checkpoint(checkpoint_path)
    config = checkpoint.config
    # One rank at a time: the remote-code config is copied into the shared modules cache.
    text_config = None
    for turn in range(WORLD_SIZE):
        if turn == rank:
            text_config = AutoConfig.from_pretrained(
                checkpoint_path, trust_remote_code=True
            ).text_config
        comm.Barrier()

    blocks = [checkpoint.layer(i) for i in range(max(layers) + 1)]
    pool, num_long = K.real_moe_inputs(checkpoint, layers, blocks)
    num_rows = pool[layers[0]][0].shape[0]
    say(f"inputs: {num_rows} rows per layer ({num_long} from the long prompts)")

    def checkpoint_tensor(key: str) -> torch.Tensor:
        """A checkpoint tensor, from the layers already on the device when it is one of theirs."""
        layer, _, suffix = key[len(R.TEXT_PREFIX + "layers.") :].partition(".")
        if key.startswith(R.TEXT_PREFIX + "layers.") and int(layer) < len(blocks):
            return blocks[int(layer)][suffix]
        return checkpoint.tensor(key)

    mapping = Mapping(
        world_size=WORLD_SIZE,
        rank=rank,
        tp_size=WORLD_SIZE,
        moe_ep_size=WORLD_SIZE,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(text_config=text_config, torch_dtype=torch.bfloat16),
        mapping=mapping,
        torch_dtype=torch.bfloat16,
        spec_config=None,
        # The most rows a rank feeds one block here: 8192, every rank's prefill cell.
        max_num_tokens=max(max(step.counts) for step in CELLS),
    )
    with torch.device("meta"):
        core = M.Nemotron4NanoPreviewDep4ModelingV2(model_config)
    table = W.MODEL_WEIGHTS
    dims = table.dims(core)
    wanted = {f"l{i}" for i in layers}
    for key, sources in table.manifest(core).items():
        if key.split("_", 1)[0] not in wanted:
            continue
        param = torch.empty(core.w[key].shape, dtype=core.w[key].dtype, device="cuda")
        for ckpt_key, index, rows, transform in sources:
            src = checkpoint_tensor(ckpt_key)
            if rows is not None:
                src = src[rows]
            if transform is not None:
                src = transform(src, dims)
            dst = param if index is None else param[index]
            assert (dst.shape, dst.dtype) == (src.shape, src.dtype), (
                ckpt_key,
                dst.shape,
                src.shape,
            )
            dst.copy_(src)
        core.w[key] = nn.Parameter(param, requires_grad=False)
    core.post_load_weights()
    targets = dict(P=core._prefill, D=core._decode)
    moe_weights = {i: R.sub(blocks[i], "moe.") for i in layers}
    blocks.clear()
    offset, local = core.expert_offset, core.local_experts
    assert (offset, local) == (rank * 128, 128), (offset, local)
    say(f"core built: rank {rank} holds experts [{offset}, {offset + local})")

    # ------------------------------------------------------------------------------------------
    # Running the block with its calls observed
    # ------------------------------------------------------------------------------------------

    guarded = (
        cublas_mm,
        dsv3_router_gemm,
        lowm_direct_mm,
        lowm_splitk_mm,
        noaux_tc_op,
        latent_moe_experts_fused_act,
        flashinfer_rmsnorm,
        softclamp_relu2,
        allgather,
        reducescatter,
        moe_a2a_dispatch,
        moe_a2a_combine,
    )
    #: The calls of `_moe` observed: modeling's name for the entry, the entry, the record. Each
    #: block calls each of the first two and the last once, and the middle four's one path: the
    #: all-gather's or the all-to-all's.
    observed = (
        ("noaux_tc_op", noaux_tc_op, "route"),
        ("latent_moe_experts_fused_act", latent_moe_experts_fused_act, "window"),
        ("allgather", allgather, "gather"),
        ("reducescatter", reducescatter, "combine"),
        ("moe_a2a_dispatch", moe_a2a_dispatch, "dispatch"),
        ("moe_a2a_combine", moe_a2a_combine, "combine"),
        ("flashinfer_rmsnorm", flashinfer_rmsnorm, "fc2_norm"),
    )
    # Taken before any `validating` block swaps the classes' `__call__` for a guarded one.
    signatures = {attr: inspect.signature(entry.__call__) for attr, entry, _ in observed}

    def observer(call, attr: str, name: str, seen: dict):
        def observe(*args, **kwargs):
            result = call(*args, **kwargs)
            seen[name] = (signatures[attr].bind(*args, **kwargs).arguments, result)
            return result

        return observe

    def run_block(step: Step, layer: int, routed, shared, plant=None, comm=None):
        """This rank's target on its rows of `step`, every entry guarded, the calls observed.

        `comm` is the step's own (`M._moe_comm`) unless given. A decode step whose ranks hold one
        count is a CUDA-graph batch, and runs on the graph's kernels (`graph`)."""
        seen = {}
        plant = plant or {}
        target = targets[step.phases[rank]]
        comm = M._moe_comm(step.counts) if comm is None else comm
        moe = target._layers[layer].moe
        with pytest.MonkeyPatch.context() as mp, validating(*guarded):
            for attr, entry, name in observed:
                mp.setattr(M, attr, observer(plant.get(attr, entry), attr, name, seen))
            if step.phases[rank] == "D":
                graph = len(set(step.counts)) == 1
                out = target._moe(routed, shared, moe, comm, graph)
            else:
                out = target._moe(routed, shared, moe, comm)
        return out, seen

    def own_latent(seen: dict, step: Step):
        """This rank's own rows' latent as the window path took it: its packed rows' latent
        columns in the all-gather (checked bitwise against what it sent), or what it handed the
        all-to-all."""
        if "dispatch" in seen:
            return seen["dispatch"][0]["input_payloads"][0]
        start = sum(step.counts[:rank])
        return seen["window"][0]["x"][start : start + step.counts[rank]]

    up_all = {i: moe_weights[i]["experts.up_proj"] for i in layers}
    down_all = {i: moe_weights[i]["experts.down_proj"] for i in layers}

    def twin_plant(layer: int) -> dict:
        """The block with the collectives taken out and every expert in the window: this rank's
        rows alone, summed in the full window's order -- a single-GPU block on the same GEMMs.
        Run on the all-gather path (`twin_comm`), where taking a collective out leaves its
        input."""

        def own_rows(input, sizes, group, process_group=None):
            return input

        def every_expert(
            x,
            topk_ids,
            topk_weights,
            up_proj,
            down_proj,
            activation_clamp,
            local_expert_offset,
            num_local_experts,
            out_dtype=None,
        ):
            return latent_moe_experts(
                x,
                topk_ids,
                topk_weights,
                up_all[layer],
                down_all[layer],
                activation_clamp,
                0,
                up_all[layer].shape[0],
                out_dtype,
            )

        return dict(
            allgather=own_rows, latent_moe_experts_fused_act=every_expert, reducescatter=own_rows
        )

    def twin_comm(step: Step):
        """The all-gather path for a step: the twin's collectives take nothing else."""
        return M._MoeComm(a2a=False, sizes=None, max_rows=max(step.counts))

    def rows_of(step: Step, index: int, layer: int, counts=None) -> torch.Tensor:
        """Which pool rows this rank runs in a step. Prefill rows are contiguous, as a chunk's
        tokens are: a chunk's stretch follows the step's earlier chunks, from a point that moves
        with the step, wrapping at the pool's end; a tail shorter than `MANY_ROWS` is the last
        short prompt's. Decode rows are a seeded draw over every row: each is another request's
        current token."""
        counts = step.counts if counts is None else counts
        n = counts[rank]
        if step.phases[rank] == "P":
            if n < MANY_ROWS:
                start = num_rows - n
            else:
                earlier = sum(
                    c
                    for c, p in zip(counts[:rank], step.phases[:rank])
                    if p == "P" and c >= MANY_ROWS
                )
                start = 1987 * index + earlier
            return ((start + torch.arange(n)) % num_rows).to("cuda")
        g = torch.Generator(device="cpu").manual_seed(100003 * layer + 1009 * index + rank)
        return torch.randperm(num_rows, generator=g)[:n].to("cuda")

    def by_contents(latent, ids, weights) -> tuple:
        """Rows in a canonical order -- sorted by their routing weights' bytes, distinct per row
        with overwhelming probability -- so two sets of rows compare whatever order holds them."""
        keys = [bytes(row) for row in weights.contiguous().view(torch.int32).cpu().numpy()]
        order = torch.tensor(
            sorted(range(len(keys)), key=keys.__getitem__), dtype=torch.long, device=ids.device
        )
        return latent[order], ids[order], weights[order]

    def dispatch_record(seen: dict, step: Step, ids, weights, experts) -> dict:
        """What reached the window. All-gather: the gathered rows at this rank's place are its
        packed rows, bit for bit, and the ids and weights the window read there are the router's.
        All-to-all: the window was handed ranks x the busiest rank's rows, and the rows it
        received from this rank are exactly this rank's tokens that route an expert here, each
        a bit-exact copy of its latent, ids and weights (which receive row holds a token is the
        kernel's choice, so the rows are compared by contents). Either way the experts' output
        is the combined sum, rounded once."""
        _, combined = seen["combine"]
        window_args, _ = seen["window"]
        rounded = bool(torch.equal(experts, combined.to(experts.dtype)))
        if "dispatch" in seen:
            dispatch_args, (recv, _) = seen["dispatch"]
            latent = dispatch_args["input_payloads"][0]
            here = ((ids.long() // 128) == rank).any(dim=1)
            got_ids = recv[1][rank]
            arrived = (got_ids >= 0).any(dim=1)
            same = bool(here.sum() == arrived.sum())
            if same:
                mine = by_contents(latent[here], ids[here], weights[here])
                theirs = by_contents(
                    recv[0][rank][arrived], got_ids[arrived], recv[2][rank][arrived]
                )
                same = all(torch.equal(a, b) for a, b in zip(mine, theirs))
            return dict(
                gathered_rows=int(window_args["x"].shape[0]),
                gathered_rows_expected=int(WORLD_SIZE * max(step.counts)),
                own_rows_bitwise=same,
                ids_bitwise=same,
                weights_bitwise=same,
                experts_are_the_rounded_sum=rounded,
            )
        gather_args, gathered = seen["gather"]
        n = step.counts[rank]
        mine = slice(sum(step.counts[:rank]), sum(step.counts[:rank]) + n)
        return dict(
            gathered_rows=int(gathered.shape[0]),
            gathered_rows_expected=int(sum(step.counts)),
            own_rows_bitwise=bool(
                gathered.shape[0] == sum(step.counts)
                and torch.equal(gathered[mine], gather_args["input"])
            ),
            ids_bitwise=bool(torch.equal(window_args["topk_ids"][mine], ids)),
            weights_bitwise=bool(torch.equal(window_args["topk_weights"][mine], weights)),
            experts_are_the_rounded_sum=rounded,
        )

    def measure(step: Step, layer: int, routed, shared, plant=None, target_shared=None, twin=True):
        """One step's block on this rank's rows, against every reference.

        Returns the routing (`K.routing`), the block (`K.block_and_stages`, plus `T vs
        single-GPU twin` when `twin`) and the stages: `K.block_and_stages`' and the
        expert-parallel path's own --
        `dispatch`: what reached the window (`dispatch_record`);
        `window`: the window's fp32 partial sums over the rows it was handed, against the entry's
        reference on the same arguments;
        `ep`: the combined sum, rounded, against `latent_moe_experts` over every expert on the
        same rows -- the full window's one fp32 sum, HF's order. Its differing rows are
        `ep_rows`, and `ep_extra_rows` counts the block rows that differ from the twin's outside
        them.
        """
        mw = moe_weights[layer]
        out, seen = run_block(
            step, layer, routed, shared if target_shared is None else target_shared, plant
        )
        route_args, (weights, ids) = seen["route"]
        window_args, partial = seen["window"]
        norm_args, normed = seen["fc2_norm"]
        latent = own_latent(seen, step)
        experts = norm_args["x"]
        dispatch = dispatch_record(seen, step, ids, weights, experts)
        window_ref = latent_moe_experts_fused_act.reference(**window_args)
        full = latent_moe_experts(
            latent.contiguous(),
            ids,
            weights,
            up_all[layer],
            down_all[layer],
            config.activation_clamp_value,
            0,
            up_all[layer].shape[0],
        )
        route, agree = K.routing(
            route_args["router_logits"], route_args["bias"], ids, weights, routed, mw, config
        )
        block, stages = K.block_and_stages(
            out,
            routed,
            shared,
            mw,
            config,
            SimpleNamespace(
                ids=ids, weights=weights, latent=latent, experts=experts, fc2_norm=normed
            ),
            agree,
        )
        ep_rows = (experts != full).any(-1)
        stages.update(
            dispatch=dispatch,
            window=K.distance(partial, window_ref),
            ep=K.distance(experts, full),
        )
        metrics = dict(routing=route, block=block, stages=stages)
        if twin:
            twin_out, _ = run_block(
                step, layer, routed, shared, twin_plant(layer), comm=twin_comm(step)
            )
            block["T vs single-GPU twin"] = K.distance(out, twin_out)
            metrics["ep_extra_rows"] = int(((out != twin_out).any(-1) & ~ep_rows).sum())
        return out, seen, metrics

    records = dict(cells=[], idle=[], graph=[], rivals=[])

    with torch.inference_mode():
        # --------------------------------------------------------------------------------------
        # Cells
        # --------------------------------------------------------------------------------------
        for layer in layers:
            routed_pool, shared_pool = pool[layer]
            for index, step in enumerate(CELLS):
                rows = rows_of(step, index, layer)
                _, _, metrics = measure(step, layer, routed_pool[rows], shared_pool[rows])
                records["cells"].append(
                    dict(
                        layer=layer,
                        step=step.name,
                        counts=list(step.counts),
                        rank=rank,
                        phase=step.phases[rank],
                        rows=int(rows.numel()),
                        comm=M._moe_comm(step.counts)._asdict(),
                        metrics=metrics,
                    )
                )
            say(f"layer {layer}: {len(CELLS)} cells measured")

        # --------------------------------------------------------------------------------------
        # Idle ranks' dummies and graph padding
        # --------------------------------------------------------------------------------------
        for layer in layers:
            routed_pool, shared_pool = pool[layer]
            for index, case in enumerate(IDLE_STEPS):
                step = Step(case.name, case.counts, case.phases)
                real = case.real[rank]
                rows = rows_of(step, len(CELLS) + index, layer, counts=case.real)
                fill = case.counts[rank] - real
                outs, filler_experts, gathered_ids = [], [], []
                for f, filler in enumerate(FILLERS):
                    if filler == "zeros":
                        zero = torch.zeros(
                            fill, routed_pool.shape[1], dtype=routed_pool.dtype, device="cuda"
                        )
                        routed = torch.cat([routed_pool[rows], zero])
                        shared = torch.cat([shared_pool[rows], zero])
                    else:
                        g = torch.Generator(device="cpu").manual_seed(
                            7 * f + 1000 * layer + 10 * index + rank
                        )
                        extra = torch.randint(0, num_rows, (fill,), generator=g).to("cuda")
                        routed = torch.cat([routed_pool[rows], routed_pool[extra]])
                        shared = torch.cat([shared_pool[rows], shared_pool[extra]])
                    out, seen = run_block(step, layer, routed, shared)
                    outs.append(out[:real])
                    # Every rank's routed ids in rank order, whichever path carried them.
                    gathered_ids.append(
                        torch.cat([t.cuda() for t in comm.allgather(seen["route"][1][1].cpu())])
                    )
                    filler_experts.append(
                        sorted(set(seen["route"][1][1][real:].reshape(-1).tolist()))
                    )
                    if f == 0 and real:
                        # The real rows against the single-GPU block on this rank's rows, which
                        # never sees the dummy or another rank: equal wherever the expert sums
                        # equal the full window's (`ep`), which they do up to the EP term.
                        twin_out, _ = run_block(
                            step, layer, routed, shared, twin_plant(layer), comm=twin_comm(step)
                        )
                        (weights, ids), experts = seen["route"][1], seen["fc2_norm"][0]["x"]
                        latent = own_latent(seen, step)
                        full = latent_moe_experts(
                            latent.contiguous(),
                            ids,
                            weights,
                            up_all[layer],
                            down_all[layer],
                            config.activation_clamp_value,
                            0,
                            up_all[layer].shape[0],
                        )
                        real_ep = K.distance(experts[:real], full[:real])
                        ep_rows = (experts[:real] != full[:real]).any(-1)
                        twin_rows = (out[:real] != twin_out[:real]).any(-1)
                        vs_twin = K.distance(out[:real], twin_out[:real])
                        twin_extra = int((twin_rows & ~ep_rows).sum())
                    elif f == 0:
                        real_ep = vs_twin = twin_extra = None
                # Which gathered rows are real, on every rank alike: rank r's first real[r] rows.
                starts = [sum(case.counts[:r]) for r in range(WORLD_SIZE)]
                is_real = torch.zeros(sum(case.counts), dtype=torch.bool, device="cuda")
                for r in range(WORLD_SIZE):
                    is_real[starts[r] : starts[r] + case.real[r]] = True
                real_experts = set(gathered_ids[0][is_real].reshape(-1).tolist())
                non_real = [set(ids[~is_real].reshape(-1).tolist()) for ids in gathered_ids]
                records["idle"].append(
                    dict(
                        layer=layer,
                        step=case.name,
                        counts=list(case.counts),
                        real_counts=list(case.real),
                        rank=rank,
                        phase=case.phases[rank],
                        rows=case.counts[rank],
                        real=real,
                        fillers=list(FILLERS),
                        # Each later filler's real rows against the first's.
                        real_rows_across_fillers=(
                            [K.distance(o, outs[0]) for o in outs[1:]] if real else None
                        ),
                        own_filler_experts=[len(e) for e in filler_experts],
                        non_real_experts=[len(e) for e in non_real],
                        non_real_experts_also_real=[len(e & real_experts) for e in non_real],
                        non_real_experts_differ=[non_real[0] != e for e in non_real[1:]],
                        real_ep=real_ep,
                        real_vs_single_gpu_twin=vs_twin,
                        real_twin_extra_rows=twin_extra,
                    )
                )
            say(f"layer {layer}: {len(IDLE_STEPS)} idle steps x {len(FILLERS)} fillers")

        # --------------------------------------------------------------------------------------
        # CUDA graphs
        # --------------------------------------------------------------------------------------
        decode = core._decode
        for layer in layers:
            routed_pool, shared_pool = pool[layer]
            moe = decode._layers[layer].moe
            for height in GRAPH_HEIGHTS:
                # Every rank at one height: the all-to-all, on the kernels a graph replays.
                graph_comm = M._moe_comm((height,) * WORLD_SIZE)
                g = torch.Generator(device="cpu").manual_seed(7919 * height + 31 * layer + rank)
                draws = [
                    torch.randperm(num_rows, generator=g)[:height].to("cuda") for _ in range(3)
                ]
                static_routed = routed_pool[draws[0]].clone()
                static_shared = shared_pool[draws[0]].clone()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        decode._moe(static_routed, static_shared, moe, graph_comm, True)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = decode._moe(static_routed, static_shared, moe, graph_comm, True)
                equal = []
                for rows in draws[1:]:
                    static_routed.copy_(routed_pool[rows])
                    static_shared.copy_(shared_pool[rows])
                    graph.replay()
                    eager = decode._moe(routed_pool[rows], shared_pool[rows], moe, graph_comm, True)
                    equal.append(bool(torch.equal(captured, eager)))
                records["graph"].append(
                    dict(layer=layer, height=height, rank=rank, replays_bitwise=equal)
                )
                del graph, captured
            say(f"layer {layer}: {len(GRAPH_HEIGHTS)} graph heights captured and replayed")

        # --------------------------------------------------------------------------------------
        # Rivals
        # --------------------------------------------------------------------------------------
        d = decode._d
        layer = layers[0]
        routed_pool, shared_pool = pool[layer]
        rows = rows_of(RIVAL_STEP, len(CELLS) + len(IDLE_STEPS), layer)
        routed, shared = routed_pool[rows], shared_pool[rows]

        def unscaled(router_logits, bias, n_group, topk_group, topk, routed_scaling_factor):
            return noaux_tc_op(router_logits, bias, n_group, topk_group, topk, 1.0)

        def bf16_weights(router_logits, bias, n_group, topk_group, topk, routed_scaling_factor):
            weights, ids = noaux_tc_op(
                router_logits, bias, n_group, topk_group, topk, routed_scaling_factor
            )
            return weights.to(torch.bfloat16).float(), ids

        def unbiased(router_logits, bias, n_group, topk_group, topk, routed_scaling_factor):
            zero = torch.zeros_like(bias)
            return noaux_tc_op(
                router_logits, zero, n_group, topk_group, topk, routed_scaling_factor
            )

        def weight_after_down_proj(
            x,
            topk_ids,
            topk_weights,
            up_proj,
            down_proj,
            activation_clamp,
            local_expert_offset,
            num_local_experts,
            out_dtype=None,
        ):
            """The generic MoE order: each expert unweighted, its output then scaled by its weight."""
            total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
            for k in range(topk_ids.shape[1]):
                one = topk_ids[:, k : k + 1].contiguous()
                unit = torch.ones_like(topk_weights[:, k : k + 1])
                o = latent_moe_experts(
                    x,
                    one,
                    unit,
                    up_proj,
                    down_proj,
                    activation_clamp,
                    local_expert_offset,
                    num_local_experts,
                )
                total += o.float() * topk_weights[:, k : k + 1]
            return total if out_dtype == torch.float32 else total.to(x.dtype)

        def bf16_partials(
            x,
            topk_ids,
            topk_weights,
            up_proj,
            down_proj,
            activation_clamp,
            local_expert_offset,
            num_local_experts,
            out_dtype=None,
        ):
            return latent_moe_experts(
                x,
                topk_ids,
                topk_weights,
                up_proj,
                down_proj,
                activation_clamp,
                local_expert_offset,
                num_local_experts,
                torch.bfloat16,
            )

        def first_window(
            x,
            topk_ids,
            topk_weights,
            up_proj,
            down_proj,
            activation_clamp,
            local_expert_offset,
            num_local_experts,
            out_dtype=None,
        ):
            return latent_moe_experts(
                x,
                topk_ids,
                topk_weights,
                up_proj,
                down_proj,
                activation_clamp,
                0,
                num_local_experts,
                out_dtype,
            )

        def bf16_dispatch(token_selected_experts, input_payloads, *args, **kwargs):
            latent, ids, weights = input_payloads
            payloads = [latent, ids, weights.to(torch.bfloat16).float()]
            return moe_a2a_dispatch(token_selected_experts, payloads, *args, **kwargs)

        def no_combine(payload, *args, **kwargs):
            """Every rank zeroes the partials it computed for other ranks' tokens: each token
            then sums its own rank's window alone."""
            kept = payload.clone()
            kept[[s for s in range(WORLD_SIZE) if s != rank]] = 0
            return moe_a2a_combine(kept, *args, **kwargs)

        def bf16_gather(input, sizes, group, process_group=None):
            rows = input.clone()
            w = rows[:, d.dispatch_weights].view(torch.float32).to(torch.bfloat16).float()
            rows[:, d.dispatch_weights] = w.view(torch.uint8)
            return allgather(rows, sizes, group)

        def no_reduce_scatter(input, sizes, group, process_group=None):
            counts = sizes if sizes is not None else [input.shape[0] // len(group)] * len(group)
            start = sum(counts[:rank])
            return input[start : start + counts[rank]].clone()

        plants = {
            RIVALS[0]: dict(noaux_tc_op=unscaled),
            RIVALS[1]: dict(noaux_tc_op=bf16_weights),
            RIVALS[2]: dict(noaux_tc_op=unbiased),
            RIVALS[3]: dict(latent_moe_experts_fused_act=weight_after_down_proj),
            RIVALS[4]: dict(flashinfer_rmsnorm=lambda x, weight, eps: x),
            RIVALS[5]: dict(flashinfer_rmsnorm=R.rms_norm),
            RIVALS[6]: {},
            RIVALS[7]: dict(latent_moe_experts_fused_act=bf16_partials),
            RIVALS[8]: dict(latent_moe_experts_fused_act=first_window),
            RIVALS[9]: dict(moe_a2a_dispatch=bf16_dispatch),
            RIVALS[10]: dict(moe_a2a_combine=no_combine),
            RIVALS[11]: dict(allgather=bf16_gather),
            RIVALS[12]: dict(reducescatter=no_reduce_scatter),
        }
        assert set(plants) == set(RIVALS)
        ragged_rows = rows_of(RAGGED_RIVAL_STEP, len(CELLS) + len(IDLE_STEPS) + 1, layer)
        for rival in RIVALS:
            step = RAGGED_RIVAL_STEP if rival in RAGGED_RIVALS else RIVAL_STEP
            mine = ragged_rows if rival in RAGGED_RIVALS else rows
            _, _, metrics = measure(
                step,
                layer,
                routed_pool[mine],
                shared_pool[mine],
                plant=plants[rival],
                target_shared=routed_pool[mine] if rival == RIVALS[6] else None,
                twin=False,
            )
            records["rivals"].append(
                dict(rival=rival, step=step.name, layer=layer, rank=rank, metrics=metrics)
            )
        say(f"{len(RIVALS)} rivals measured")

    torch.cuda.synchronize()
    peak_gib = comm.gather(torch.cuda.max_memory_allocated() / 2**30, root=0)
    everything = comm.gather(records, root=0)
    if rank == 0:
        merged = {key: [r for per in everything for r in per[key]] for key in records}
        merged["meta"] = dict(
            world_size=WORLD_SIZE,
            layers=list(layers),
            pool_rows=num_rows,
            long_rows=num_long,
            seconds=time.monotonic() - started,
            device=torch.cuda.get_device_name(),
            peak_gib_per_rank=peak_gib,
        )
        with open(out_path, "w") as f:
            json.dump(merged, f)
        say(f"wrote {out_path}; peak GiB per rank {[round(g, 1) for g in peak_gib]}")
    comm.Barrier()


def _run_one_rank(out: str, checkpoint: str, layers) -> int:
    """Body of one MPI rank: a failing rank exits hard, and mpirun ends the job."""
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    import torch

    torch.cuda.set_device(local)
    pid_dir = os.environ.get(_PID_DIR_ENV)
    if pid_dir:
        Path(pid_dir, f"rank{os.environ.get('OMPI_COMM_WORLD_RANK', local)}.pid").write_text(
            str(os.getpid())
        )
    from mpi4py import MPI

    try:
        _rank_body(out, checkpoint, layers)
    except BaseException:
        print(f"[rank {MPI.COMM_WORLD.Get_rank()}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Exit hard rather than return: a rank that leaves a collective early wedges every
        # sibling in it, and mpirun ends the job the moment it sees this exit.
        os._exit(13)
    sys.stdout.flush()
    # Finalize so mpirun counts this as a clean exit, then leave without running teardown:
    # the captured graphs and the communicators go with the process.
    MPI.Finalize()
    os._exit(0)


def _parse(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(_WORKER_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", required=True, help="where rank 0 writes the records (JSON)")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.environ.get("LLM_MODELS_ROOT", ""), "nemotron4-nano-preview"),
        help="the checkpoint directory (default: $LLM_MODELS_ROOT/nemotron4-nano-preview)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(LAYERS),
        help="the MoE layers to measure (default: the test's)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse(sys.argv[1:])
    if args.rank_worker:
        sys.exit(_run_one_rank(args.out, args.checkpoint, tuple(args.layers)))
    spawn_ranks(os.path.abspath(args.out), args.checkpoint, tuple(args.layers))
    print("OK", flush=True)
