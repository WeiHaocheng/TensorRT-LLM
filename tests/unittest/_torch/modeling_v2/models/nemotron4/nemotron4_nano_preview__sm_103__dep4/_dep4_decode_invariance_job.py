# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 4-rank job behind `test_modeling_v2_dep4_decode_invariance.py`: the dep4 decode step, run on
every rank of one node on real weights and real cache state, and compared bit for bit across batch
shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _dep4_decode_invariance_job.py --out <json> [--checkpoint <dir>]

Run plainly, this script is its own launcher: it starts one MPI rank per device under `mpirun`,
with a deadline it enforces by killing the job, because a collective that breaks hangs rather than
raising. The ranks are an MPI world since that is what a dep4 serving engine starts, and the routed
MoE's all-gather and reduce-scatter run on the MPI-session symbols over it.

What every rank builds, as the engine builds it for its rank: ``ModelConfig`` off the checkpoint
with the rank's dep4 ``Mapping``, the target class, ``load_weights`` over every checkpoint tensor
(the rank's window of 128 experts of each stack) and ``post_load_weights``; the engine's hybrid
cache manager for that mapping (``_create_kv_cache_manager``, the default
``MambaHybridCacheManagerV2``); the engine's TRTLLM attention metadata (``build_attention_metadata``),
filled per step as the runner fills it, ``all_rank_num_tokens`` included; and, for graphs,
``create_cuda_graph_metadata`` and the runner's padding dummy. A rank with no request runs the
attention-DP dummy (``ATTENTION_DP_DUMMY_REQUEST_ID``, a generation request with nothing cached),
as the executor pads an idle rank.

The requests: on every rank, three probes -- the twelve of the tp1 test (the ten prompts of the
fixed set, a history that fills one 512-key attention split exactly and one that ends mid-split),
dealt round-robin -- and fillers, prefilled through the target's own prefill steps on all four ranks
in lockstep, so every request decodes from KV, recurrent and conv state the model wrote. Each then
decodes its greedy next token. Every run starts from the same state: each GDP slot is restored from
a snapshot, and a row's KV append lands at the one position its own attention reads only after the
append.

The runs (`STEPS`), all four ranks in lockstep, each rank's probes compared on that rank:

* the reference: each probe alone on its rank, eagerly, the other three ranks idle;
* eager decode batches on one rank, the others idle (the ragged split, busiest rank anywhere);
* eager decode batches on every rank at once, ragged and even;
* eager decode beside ranks that prefill a 2048-token chunk through PrefillTarget;
* padded CUDA-graph batches on one rank, the others idle (the attention-DP dummy plus padding);
* padded CUDA-graph batches on every rank at once.

Per probe, bitwise: its fp32 logits, the recurrent and conv state of its slot in the 32 GDP layers,
and the K/V at its decoded position in the 8 attention layers (read through the manager's own page
table). After every run, only the real rows' slots differ from the snapshot: placeholder rows write
no state. Beside the model: the reduce-scatter's order on fp32 partials of the MoE width over every
split the runs make (`COMBINE_LAYOUTS`), the attention entry against `thop_attention` on one step,
and two rivals -- the replaced cuBLAS GEMMs and `thop_attention` put back -- that must move with the
batch. Rank 0 writes every rank's records to `--out` as JSON.

Not a pytest module: the ranks run one fixed sequence of collectives in lockstep, and neither this
file's name nor its functions match pytest's collection patterns. The launcher must not import
`tensorrt_llm` -- that calls `MPI_Init`, and an MPI-initialized process cannot start `mpirun` -- so
everything that does is imported inside the rank body.
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
from dataclasses import dataclass, field
from pathlib import Path

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "NEMOTRON4_DEP4_DECODE_INVARIANCE_PID_DIR"
DEADLINE_S = 3600
"""Ceiling on the whole job: weight load, prefill, captures and every run."""

WORLD_SIZE = 4
"""dep4: attention DP over the four GPUs of one node, the routed experts split four ways."""

# The engine sizes this run uses everywhere (task.yaml, the boot gate, check_goldens).
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
TOKENS_PER_BLOCK = 32
NUM_REQUESTS = 256
"""Requests per rank: its three probes and its fillers."""
MAX_BATCH = 264
"""Rows a rank's step may hold; resident requests per rank: the probes, fillers, the loader and
the two dummies."""
KV_TOKENS = 192 * 1024
"""KV tokens of a rank's paged pool: every request stays resident."""
LOADER_TOKENS = 2048
"""A prefilling rank's chunk beside decoding ranks: the comm cells' prefill height."""
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
"""The CUDA-graph batch sizes captured per rank; a step runs in the smallest one that holds the
busiest rank's batch, as the runner pads every rank to it."""

IDLE = 0
"""A rank's count when it has no request: it runs the attention-DP dummy, one row."""
PREFILL = "P"
"""A rank's count when it prefills a `LOADER_TOKENS` chunk through PrefillTarget."""


@dataclass(frozen=True)
class Step:
    """One decode step of the attention-DP group.

    `counts[r]`: rank r's decode rows, `IDLE`, or `PREFILL`. `graph`: replayed in the captured
    graph of the smallest size that holds the busiest rank, every rank padded to it; else eager,
    every rank its own count.
    """

    group: str
    counts: tuple
    graph: bool

    @property
    def name(self) -> str:
        return f"{self.group} {list(self.counts)}"


def _one_rank(count, rank: int) -> tuple:
    return tuple(count if r == rank else IDLE for r in range(WORLD_SIZE))


EAGER_SIZES = (2, 3, 5, 8, 16, 64, 100, 256)
"""Eager decode batches on one rank while the others idle: every ragged split the target makes,
with the busiest rank in each position."""
GRAPH_BATCH_SIZES = (1, 2, 3, 5, 7, 8, 16, 64, 100, 250, 256)
"""Graph decode batches on one rank while the others idle; 3, 5, 7, 100 and 250 leave padding."""

STEPS: tuple[Step, ...] = (
    *(
        Step("eager, others idle", _one_rank(b, r), False)
        for r in range(WORLD_SIZE)
        for b in EAGER_SIZES
    ),
    *(
        Step("eager, others decoding", counts, False)
        for counts in (
            (3, 64, 5, 256),
            (256, 3, 64, 5),
            (5, 256, 3, 64),
            (64, 5, 256, 3),
            (100, 100, 100, 100),
            (1, 7, 16, 250),
            (250, 1, 7, 16),
        )
    ),
    *(
        Step("eager, beside a prefill", counts, False)
        for counts in (
            (PREFILL, 8, IDLE, 64),
            (5, PREFILL, 64, PREFILL),
            (PREFILL, PREFILL, PREFILL, 3),
            (16, 2, PREFILL, 100),
        )
    ),
    *(
        Step("graph, others idle", _one_rank(b, r), True)
        for r in range(WORLD_SIZE)
        for b in GRAPH_BATCH_SIZES
    ),
    *(
        Step("graph, others decoding", counts, True)
        for counts in (
            (3, 64, 5, 50),
            (250, 100, 256, 7),
            (1, 1, 1, 1),
            (2, IDLE, 1, 2),
            (8, 5, IDLE, 3),
            (16, 9, 12, 1),
            (30, IDLE, 17, 32),
            (128, 100, 3, 64),
        )
    ),
)
GROUPS = tuple(dict.fromkeys(s.group for s in STEPS))

RIVALS = ("cublas_mm", "thop_attention")
"""What the decode step ran before it was made invariant: the six GEMM sites on `cublas_mm`, or
generation attention on `thop_attention`."""
RIVAL_SIZES = (2, 3, 8, 64, 256)
"""Eager batch sizes the rivals run at on one rank, the others idle: cuBLAS changes kernels on
its row-variant shapes from two rows up, and thop's KV split follows the batch."""

COMBINE_LAYOUTS: tuple[tuple[int, ...], ...] = (
    *((h,) * WORLD_SIZE for h in (1, 2, 3, 4, 8, 16, 32, 64, 100, 128, 256)),
    (2, 1, 1, 1),
    (1, 1, 1, 256),
    (3, 64, 5, 256),
    (1, 7, 16, 250),
    (LOADER_TOKENS, 8, 1, 64),
    (LOADER_TOKENS, LOADER_TOKENS, LOADER_TOKENS, 3),
    (MAX_NUM_TOKENS, 1, 1, 1),
    (1, MAX_NUM_TOKENS, 64, MAX_NUM_TOKENS),
)
"""Per-rank counts the routed MoE's reduce-scatter combines: every padded graph height, and the
ragged splits of eager decode beside idle, decoding and prefilling ranks (up to the prefill steps'
`MAX_NUM_TOKENS`). Even counts take the padded form (sizes None), the rest the ragged form."""
COMBINE_GRAPH_HEIGHTS = (1, 8, 64, 256)
"""Padded heights the combine is also captured and replayed at, as decode graphs run it."""
COMBINE_WIDTH = 768
"""The latent width the reduce-scatter combines."""


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
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; this file moved "
        "without its parents[] index following"
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH")) if p)
    return env


def spawn_ranks(out: str, checkpoint: str) -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the four devices this run owns"
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    assert len(devices) >= WORLD_SIZE, (
        f"dep4 runs on {WORLD_SIZE} devices; CUDA_VISIBLE_DEVICES names {len(devices)}"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; the ranks are an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="dep4_decode_invariance_")
    env = dict(
        _standalone_env(),
        CUDA_VISIBLE_DEVICES=",".join(devices[:WORLD_SIZE]),
        **{_PID_DIR_ENV: pid_dir},
    )
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(WORLD_SIZE)]
    if os.geteuid() == 0:
        command.insert(1, "--allow-run-as-root")
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    command += ["--out", out, "--checkpoint", checkpoint]
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


@dataclass
class Request:
    """One sequence on one rank: its prompt, and the token its decode step feeds (the prompt's
    greedy next token)."""

    rid: int
    name: str
    prompt: tuple
    token: int = -1


@dataclass
class Run:
    """One lockstep run: this rank's part of it."""

    step: str
    group: str
    graph_size: int | None
    rows: list = field(default_factory=list)  # this rank's Requests, or [] when idle
    probes: list = field(default_factory=list)  # row positions of this rank's probes
    prefill: bool = False


def _rank_body(out_path: str, checkpoint_path: str) -> None:
    """Everything one rank does; every collective in lockstep with the other three."""
    import torch
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == WORLD_SIZE, f"dep4 needs {WORLD_SIZE} ranks, got {comm.Get_size()}"

    # Spawned by mpirun rather than collected by pytest, so the tree's `__extra_import_path__`
    # is not in play: the family's helpers and `_validating` are reached the plain way.
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here.parent), str(here.parents[2])]

    import _nemotron4_prompts as P
    from _validating import ULP

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import (
        reducescatter,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
    from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    started = time.monotonic()

    def say(message: str) -> None:
        if rank == 0:
            print(f"[job {time.monotonic() - started:7.1f}s] {message}", flush=True)

    def bits(x: torch.Tensor) -> torch.Tensor:
        return x.view({4: torch.int32, 2: torch.int16}[x.element_size()])

    group = list(range(WORLD_SIZE))
    records: dict[str, list] = dict(combine=[], reference=[], runs=[], attention=[], rivals=[])

    # ------------------------------------------------------------------------------------------
    # The combine's order, on the MoE width, over every split the runs make
    # ------------------------------------------------------------------------------------------

    def partials(seed: int, r: int, rows: int) -> torch.Tensor:
        g = torch.Generator(device="cuda").manual_seed(seed * 16 + r)
        # Mixed magnitudes, as expert partial sums are.
        scale = torch.exp(torch.randn(rows, 1, generator=g, device="cuda") * 2.0)
        return torch.randn(rows, COMBINE_WIDTH, generator=g, device="cuda") * scale

    def check_combine(name: str, counts, out: torch.Tensor, seed: int) -> None:
        start = sum(counts[:rank])
        mine = slice(start, start + counts[rank])
        parts = [partials(seed, r, sum(counts))[mine] for r in range(WORLD_SIZE)]
        ring = parts[(rank + 1) % WORLD_SIZE].clone()
        for k in range(2, WORLD_SIZE + 1):
            ring = ring + parts[(rank + k) % WORLD_SIZE]
        ascending = ((parts[0] + parts[1]) + parts[2]) + parts[3]
        records["combine"].append(
            dict(
                layout=name,
                counts=list(counts),
                rank=rank,
                rows=counts[rank],
                ring_equal=float((bits(out) == bits(ring)).float().mean()),
                ascending_equal=float((bits(out) == bits(ascending)).float().mean()),
            )
        )

    with torch.inference_mode():
        seed = 0
        for counts in COMBINE_LAYOUTS:
            seed += 1
            sizes = None if len(set(counts)) == 1 else list(counts)
            out = reducescatter(partials(seed, rank, sum(counts)), sizes, group)
            check_combine(f"eager {list(counts)}", counts, out, seed)
        for height in COMBINE_GRAPH_HEIGHTS:
            seed += 1
            counts = (height,) * WORLD_SIZE
            static = partials(seed + 1000, rank, sum(counts))
            reducescatter(static, None, group)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = reducescatter(static, None, group)
            torch.cuda.synchronize()
            comm.Barrier()
            static.copy_(partials(seed, rank, sum(counts)))
            graph.replay()
            torch.cuda.synchronize()
            check_combine(f"graph {list(counts)}", counts, out.clone(), seed)
            del graph
            comm.Barrier()
    say(f"combine: {len(records['combine'])} records on rank 0")

    # ------------------------------------------------------------------------------------------
    # The target, its cache manager and metadata, the requests
    # ------------------------------------------------------------------------------------------

    mapping = Mapping(
        world_size=WORLD_SIZE,
        rank=rank,
        gpus_per_node=WORLD_SIZE,
        tp_size=WORLD_SIZE,
        moe_ep_size=WORLD_SIZE,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    # One rank at a time: the remote-code config is copied into the shared modules cache.
    config = None
    for turn in range(WORLD_SIZE):
        if turn == rank:
            config = ModelConfig.from_pretrained(
                checkpoint_path, trust_remote_code=True, mapping=mapping
            )
            validate_and_set_mamba_ssm_cache_dtype(config, "auto")
        comm.Barrier()
    tokenizer = None
    for turn in range(WORLD_SIZE):
        if turn == rank:
            tokenizer = P.load_tokenizer(checkpoint_path)
        comm.Barrier()

    from safetensors import safe_open

    with open(os.path.join(checkpoint_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {
        shard: safe_open(os.path.join(checkpoint_path, shard), framework="pt", device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    tensors = {key: handles[shard].get_slice(key) for key, shard in weight_map.items()}
    with torch.device("cuda"):
        model = M.ModelingV2Nemotron4NanoPreviewSm103Dep4(config)
    model.load_weights(tensors)
    model.post_load_weights()
    model.eval()
    del tensors, handles
    torch.cuda.synchronize()
    core = model.model
    assert (core.expert_offset, core.local_experts) == (rank * 128, 128)
    weights_gib = torch.cuda.memory_allocated() / 2**30
    say(f"target built on every rank: {weights_gib:.2f} GiB allocated on rank 0")

    kv_cache_config = KvCacheConfig(
        max_tokens=KV_TOKENS, enable_block_reuse=False, free_gpu_memory_fraction=0.5
    )
    manager = _create_kv_cache_manager(
        None,
        get_kv_cache_manager_cls(config, kv_cache_config),
        mapping,
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=config,
        dtype=torch.bfloat16,
        max_cuda_graph_batch_size=max(GRAPH_SIZES),
    )
    md = build_attention_metadata(
        config,
        max_batch_size=MAX_BATCH,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        attention_backend=get_attention_backend("TRTLLM"),
        attention_runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=False,
            cache_reuse=False,
            has_speculative_draft_tokens=False,
            chunk_size=MAX_NUM_TOKENS,
        ),
        mapping=mapping,
        cache_indirection=None,
        kv_cache_manager=manager,
    )
    md.mamba_metadata_cls = M.ModelingV2Nemotron4NanoPreviewSm103Dep4.mamba_metadata_cls
    gdp_layers, attention_layers = core.gdp_layers, core.attention_layers

    # The requests: this rank's probes (the tp1 test's twelve, dealt round-robin), its fillers
    # and its loader, a 2048-token prompt it prefills beside other ranks' decode steps.
    prompts = P.prompt_set(tokenizer)
    by_name = {p.name: p for p in prompts}
    specs = [(p.name, p.token_ids) for p in prompts]
    specs.append(("gpqa_explain[:511]", by_name["gpqa_explain"].token_ids[:511]))
    specs.append(("gsm8k_multiturn[:4096]", by_name["gsm8k_multiturn"].token_ids[:4096]))
    assert len(specs) == 3 * WORLD_SIZE, len(specs)
    probes = [Request(0, n, tuple(t)) for k, (n, t) in enumerate(specs) if k % WORLD_SIZE == rank]
    sources = [
        by_name[n].token_ids
        for n in ("gsm8k_fewshot", "gsm8k_multiturn", "news_digest", "gpqa_explain")
    ]
    fillers = [
        Request(
            0,
            f"filler{k}",
            tuple(sources[(k + rank) % len(sources)][: 24 + (37 * k + 101 * rank) % 600]),
        )
        for k in range(NUM_REQUESTS - len(probes))
    ]
    loader = Request(0, "loader", tuple(sources[rank % len(sources)][:LOADER_TOKENS]))
    requests = probes + fillers
    for rid, r in enumerate(requests + [loader], start=1):
        r.rid = rid
    for r in requests + [loader]:
        added = manager.add_dummy_requests([r.rid], token_nums=[len(r.prompt) + 1])
        assert added, f"rank {rank}: no room for {r.name} ({len(r.prompt)} tokens)"
    # The executor's attention-DP dummy for an idle rank, and the runner's padding dummy: each a
    # generation request whose one token is its own.
    for dummy in (ATTENTION_DP_DUMMY_REQUEST_ID, CUDA_GRAPH_DUMMY_REQUEST_ID):
        added = manager.add_dummy_requests([dummy], is_gen=True)
        assert added, f"rank {rank}: no room for dummy {dummy}"
    ids = [r.rid for r in requests + [loader]]
    slot = dict(zip(ids, manager.get_state_indices(ids)))
    assert len(set(slot.values())) == len(ids)

    # ------------------------------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------------------------------

    pad_tokens = torch.Generator(device="cuda").manual_seed(9 + rank)
    vocab = config.pretrained_config.text_config.vocab_size

    def fill(meta, ctx: list, gen: list, adp_dummy: bool, padding: int, counts: list) -> None:
        """The step's per-request fields, set as the runner's `_prepare_tp_inputs` sets them,
        then `prepare()`, then every rank's token count. Context rows feed their whole prompt;
        generation rows feed one token over the prompt already cached; the attention-DP dummy and
        `padding` rows feed one token over nothing cached (one key: their own page)."""
        placeholders = int(adp_dummy) + padding
        if not meta.is_cuda_graph:
            meta.seq_lens = torch.tensor(
                [len(r.prompt) for r in ctx] + [1] * (len(gen) + placeholders), dtype=torch.int
            )
        meta.beam_width = 1
        meta.request_ids = (
            [r.rid for r in ctx + gen]
            + [ATTENTION_DP_DUMMY_REQUEST_ID] * int(adp_dummy)
            + [CUDA_GRAPH_DUMMY_REQUEST_ID] * padding
        )
        meta.prompt_lens = [len(r.prompt) for r in ctx + gen] + [0] * placeholders
        meta.num_contexts = len(ctx)
        meta.num_chunked_ctx_requests = 0
        meta.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * len(ctx)
            + [len(r.prompt) for r in gen]
            + [0] * placeholders,
            num_extra_kv_tokens=0,
        )
        meta.kv_cache_manager = manager
        meta.padded_num_tokens = None
        meta.prepare()
        # The runner sets this after `prepare()`, every rank's count in rank order.
        meta.all_rank_num_tokens = list(counts)

    def input_ids(ctx: list, gen: list, placeholders: int) -> torch.Tensor:
        """Context rows' prompts, generation rows' decode tokens, then arbitrary tokens on the
        placeholder rows, as the runner leaves whatever its static buffer held there."""
        real = torch.tensor(
            [t for r in ctx for t in r.prompt] + [r.token for r in gen],
            dtype=torch.int,
            device="cuda",
        )
        pad = torch.randint(
            0, vocab, (placeholders,), generator=pad_tokens, dtype=torch.int, device="cuda"
        )
        return torch.cat([real, pad])

    def all_counts(mine: int) -> list:
        return comm.allgather(mine)

    # --- prefill: every request through PrefillTarget, all ranks in lockstep ---
    batches: list[list] = []
    for r in requests:
        if batches and sum(len(b.prompt) for b in batches[-1]) + len(r.prompt) <= MAX_NUM_TOKENS:
            batches[-1].append(r)
        else:
            batches.append([r])
    num_steps = comm.allreduce(len(batches), op=MPI.MAX)
    with torch.inference_mode():
        for s in range(num_steps):
            batch = batches[s] if s < len(batches) else []
            tokens = sum(len(b.prompt) for b in batch) if batch else 1
            counts = all_counts(tokens)
            fill(md, batch, [], adp_dummy=not batch, padding=0, counts=counts)
            logits = model.forward(
                attn_metadata=md, input_ids=input_ids(batch, [], 0 if batch else 1)
            )
            for b, token in zip(batch, logits.argmax(-1).tolist()):
                b.token = token
        torch.cuda.synchronize()
    say(f"prefilled {len(requests)} requests per rank in {num_steps} lockstep steps")

    snapshot = [
        (manager.get_ssm_states(i).clone(), manager.get_conv_states(i).clone()) for i in gdp_layers
    ]
    kv_buffers = {i: manager.get_buffers(i) for i in attention_layers}
    pages = {
        r.rid: {
            i: manager.get_batch_cache_indices([r.rid], layer_idx=i)[0] for i in attention_layers
        }
        for r in probes
    }

    def restore() -> None:
        for i, (ssm, conv) in zip(gdp_layers, snapshot):
            manager.get_ssm_states(i).copy_(ssm)
            manager.get_conv_states(i).copy_(conv)

    def changed_slots() -> set:
        changed = set()
        for i, (ssm, conv) in zip(gdp_layers, snapshot):
            for pool, saved in (
                (manager.get_ssm_states(i), ssm),
                (manager.get_conv_states(i), conv),
            ):
                diff = (bits(pool) != bits(saved)).flatten(1).any(dim=1)
                changed |= set(torch.nonzero(diff).flatten().tolist())
        return changed

    def kv_at(r: Request, layer: int) -> torch.Tensor:
        pos = len(r.prompt)
        page = pages[r.rid][layer][pos // TOKENS_PER_BLOCK]
        return kv_buffers[layer][page, :, pos % TOKENS_PER_BLOCK, 0].clone()

    def row(r: Request, logits: torch.Tensor) -> dict:
        s = slot[r.rid]
        return dict(
            logits=logits.clone(),
            ssm=[manager.get_ssm_states(i)[s].clone() for i in gdp_layers],
            conv=[manager.get_conv_states(i)[s].clone() for i in gdp_layers],
            kv=torch.stack([kv_at(r, i) for i in attention_layers]),
        )

    def differences(got: dict, ref: dict) -> list:
        out = []
        if not torch.equal(bits(got["logits"]), bits(ref["logits"])):
            diff = (got["logits"] - ref["logits"]).abs()
            out.append(
                f"logits: {int((diff > 0).sum())} of {diff.numel()} differ, max |diff| "
                f"{float(diff.max()):.3e}, argmax {int(got['logits'].argmax())} vs "
                f"{int(ref['logits'].argmax())}"
            )
        for name in ("ssm", "conv"):
            bad = [
                layer
                for layer, a, b in zip(gdp_layers, got[name], ref[name])
                if not torch.equal(bits(a), bits(b))
            ]
            if bad:
                out.append(f"{name} state differs in GDP layers {bad}")
        bad = [
            layer
            for layer, a, b in zip(attention_layers, got["kv"], ref["kv"])
            if not torch.equal(bits(a), bits(b))
        ]
        if bad:
            out.append(f"appended K/V differs in attention layers {bad}")
        return out

    graphs: dict[int, tuple] = {}
    graph_pool = torch.cuda.graph_pool_handle()
    static_ids = torch.zeros(max(GRAPH_SIZES), dtype=torch.int, device="cuda")

    def run_eager(run: Run, counts: list) -> torch.Tensor:
        """One eager step: this rank's decode rows, its loader's prefill, or the attention-DP
        dummy; the rows' logits."""
        restore()
        if run.prefill:
            fill(md, [loader], [], adp_dummy=False, padding=0, counts=counts)
            return model.forward(attn_metadata=md, input_ids=input_ids([loader], [], 0))
        idle = not run.rows
        fill(md, [], run.rows, adp_dummy=idle, padding=0, counts=counts)
        return model.forward(attn_metadata=md, input_ids=input_ids([], run.rows, int(idle)))

    def run_graph(run: Run) -> torch.Tensor:
        """One step replayed in the captured graph of `run.graph_size`, this rank's rows (or the
        attention-DP dummy) padded with the padding dummy's rows; the real rows' logits."""
        size = run.graph_size
        idle = not run.rows
        padding = size - len(run.rows) - int(idle)
        counts = [size] * WORLD_SIZE
        if size not in graphs:
            gmd = md.create_cuda_graph_metadata(size, False, 0, get_memory_buffers())
            fill(gmd, [], run.rows, adp_dummy=idle, padding=padding, counts=counts)
            static_ids[:size].copy_(input_ids([], run.rows, int(idle) + padding))
            ids = static_ids[:size]
            for _ in range(2):
                model.forward(attn_metadata=gmd, input_ids=ids)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=graph_pool):
                out = model.forward(attn_metadata=gmd, input_ids=ids)
            torch.cuda.synchronize()
            comm.Barrier()
            graphs[size] = (gmd, graph, out)
        gmd, graph, out = graphs[size]
        restore()
        fill(gmd, [], run.rows, adp_dummy=idle, padding=padding, counts=counts)
        static_ids[:size].copy_(input_ids([], run.rows, int(idle) + padding))
        graph.replay()
        return out[: len(run.rows)]

    # --- the plan: every rank expands every step the same way ---

    offsets = {}

    def batches_of(size: int) -> list[tuple[list, list]]:
        """Batches of `size` decode rows on this rank holding each of its probes at least once,
        with the probes' row positions: spread from the first row to the last when the batch
        holds all three, else taken in turn (reversed every other batch). Fillers fill the rest,
        from an offset of their own per size."""
        offset = offsets.setdefault(size, (size * 7 + 13 * rank) % len(fillers))
        rotated = fillers[offset:] + fillers[:offset]
        if size >= len(probes):
            positions = [round(k * (size - 1) / (len(probes) - 1)) for k in range(len(probes))]
            rows, spare = [], iter(rotated)
            for p in range(size):
                rows.append(probes[positions.index(p)] if p in positions else next(spare))
            return [(rows, positions)]
        out = []
        for start in range(0, len(probes), size):
            chunk = probes[start : start + size]
            if (start // size) % 2:
                chunk = chunk[::-1]
            out.append((chunk + rotated[: size - len(chunk)], list(range(len(chunk)))))
        return out

    def expand(step: Step) -> list[tuple[list, Run]]:
        """`step`'s lockstep runs: (every rank's token count, this rank's part)."""
        decoding = [c for c in step.counts if c not in (IDLE, PREFILL)]
        num_runs = max(len(batches_of(c)) for c in decoding) if decoding else 1
        size = None
        if step.graph:
            busiest = max(1 if c == IDLE else c for c in step.counts)
            size = min(g for g in GRAPH_SIZES if g >= busiest)
        mine = step.counts[rank]
        out = []
        for k in range(num_runs):
            run = Run(step=step.name, group=step.group, graph_size=size)
            if mine == PREFILL:
                run.prefill = True
            elif mine != IDLE:
                options = batches_of(mine)
                run.rows, run.probes = options[k % len(options)]
            counts = [
                LOADER_TOKENS if c == PREFILL else (1 if c == IDLE else c) for c in step.counts
            ]
            out.append((counts, run))
        return out

    def execute(run: Run, counts: list) -> torch.Tensor:
        return run_graph(run) if run.graph_size else run_eager(run, counts)

    # --- the reference: each probe alone on its rank, the other ranks idle ---
    reference = {}
    with torch.inference_mode():
        for owner in range(WORLD_SIZE):
            for k in range(len(probes)):
                run = Run(
                    step=f"reference rank {owner} probe {k}", group="reference", graph_size=None
                )
                if rank == owner:
                    run.rows, run.probes = [probes[k]], [0]
                counts = [1] * WORLD_SIZE
                logits = execute(run, counts)
                if rank == owner:
                    r = probes[k]
                    ref = row(r, logits[0])
                    reference[r.rid] = ref
                    records["reference"].append(
                        dict(
                            rank=rank,
                            probe=r.name,
                            prompt_tokens=len(r.prompt),
                            finite=bool(torch.isfinite(ref["logits"]).all()),
                            ssm_nonzero=all(bool(t.any()) for t in ref["ssm"]),
                            conv_nonzero=all(bool(t.any()) for t in ref["conv"]),
                            kv_nonzero=all(bool(t.any()) for t in ref["kv"]),
                            argmax=int(ref["logits"].argmax()),
                        )
                    )
    say(f"reference: {len(probes)} probes alone on every rank")

    def run_and_compare(run: Run, counts: list, ref: dict, what: str) -> dict:
        """Run `run`, compare its probes with `ref`, and list the slots no real row owns that
        the run wrote."""
        logits = execute(run, counts)
        diffs, differing = [], 0
        for pos in run.probes:
            r = run.rows[pos]
            found = differences(row(r, logits[pos]), ref[r.rid])
            differing += bool(found)
            diffs += [f"{what}, rank {rank}, {r.name} at row {pos}: {d}" for d in found]
        written = changed_slots()
        real = {slot[r.rid] for r in run.rows}
        if run.prefill:
            real.add(slot[loader.rid])
        return dict(
            compared=len(run.probes),
            differing=differing,
            diffs=diffs,
            placeholder_writes=sorted(written - real),
        )

    # --- every step ---
    with torch.inference_mode():
        for step in STEPS:
            for counts, run in expand(step):
                result = run_and_compare(run, counts, reference, step.name)
                records["runs"].append(
                    dict(
                        step=step.name,
                        group=step.group,
                        counts=counts,
                        graph_size=run.graph_size,
                        rank=rank,
                        rows=len(run.rows),
                        **result,
                    )
                )
    say(f"runs: {len(records['runs'])} on rank 0, graphs captured at {sorted(graphs)}")

    # --- the attention entry against thop_attention, on one 64-row step on every rank ---
    d = core._decode._d
    k_cols = slice(d.q_width, d.q_width + d.head_dim)
    v_cols = slice(d.q_width + d.head_dim, d.q_width + 2 * d.head_dim)
    original = M.invariant_gen_attention
    calls = []
    rows64 = batches_of(64)[0][0]

    def both(**kwargs):
        out = original(**kwargs)
        q, num_rows = kwargs["q"], kwargs["q"].shape[0]
        layer = next(
            i for i in attention_layers if manager.layer_offsets[i] == kwargs["local_layer_idx"]
        )
        ours = out[:num_rows].clone()
        appended = torch.stack([kv_at(r, layer) for r in rows64 if r.rid in pages])
        theirs = M.thop_attention(
            q=q,
            output=torch.empty_like(out),
            local_layer_idx=kwargs["local_layer_idx"],
            num_heads=kwargs["num_heads"],
            num_kv_heads=kwargs["num_kv_heads"],
            head_size=kwargs["head_size"],
            **core._prefill.step_args(md)["attention"],
            **M._CALL_CONSTANTS,
        )
        after_thop = torch.stack([kv_at(r, layer) for r in rows64 if r.rid in pages])
        probe_rows = [p for p, r in enumerate(rows64) if r.rid in pages]
        expected = torch.stack([q[probe_rows, k_cols], q[probe_rows, v_cols]], dim=1)
        diff = (ours.float() - theirs[:num_rows].float()).abs()
        scale = theirs[:num_rows].float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
        records["attention"].append(
            dict(
                rank=rank,
                layer=layer,
                appended_is_qkv=bool(torch.equal(bits(appended), bits(expected))),
                thop_reappend_same=bool(torch.equal(bits(after_thop), bits(appended))),
                element_ulp=float((diff / scale).max()) / ULP[torch.bfloat16],
                rms_ulp=float(
                    diff.pow(2).mean().sqrt() / theirs[:num_rows].float().pow(2).mean().sqrt()
                )
                / ULP[torch.bfloat16],
            )
        )
        calls.append(layer)
        return out

    M.invariant_gen_attention = both
    try:
        with torch.inference_mode():
            run = Run(step="attention band", group="attention", graph_size=None)
            run.rows, run.probes = rows64, []
            execute(run, [64] * WORLD_SIZE)
    finally:
        M.invariant_gen_attention = original
    assert len(calls) == len(attention_layers), calls
    say("attention: every rank's 64-row step against thop_attention")

    # --- rivals: the replaced paths put back must move some probe with the batch ---
    for rival in RIVALS:
        if rival == "cublas_mm":
            saved = ("invariant_mm", M.invariant_mm)

            def replaced(x, weight, out_dtype=None):
                return M.cublas_mm(x, weight.t(), None, out_dtype)

        else:
            saved = ("invariant_gen_attention", M.invariant_gen_attention)

            def replaced(q, output, local_layer_idx, num_heads, num_kv_heads, head_size, **unused):
                return M.thop_attention(
                    q=q,
                    output=output,
                    local_layer_idx=local_layer_idx,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    **core._prefill.step_args(md)["attention"],
                    **M._CALL_CONSTANTS,
                )

        setattr(M, saved[0], replaced)
        try:
            with torch.inference_mode():
                rival_ref = {}
                for owner in range(WORLD_SIZE):
                    for k in range(len(probes)):
                        run = Run(step="rival reference", group="rival", graph_size=None)
                        if rank == owner:
                            run.rows, run.probes = [probes[k]], [0]
                        logits = execute(run, [1] * WORLD_SIZE)
                        if rank == owner:
                            rival_ref[probes[k].rid] = row(probes[k], logits[0])
                differ, compared, differing = [], 0, 0
                for owner in range(WORLD_SIZE):
                    for size in RIVAL_SIZES:
                        step = Step("rival", _one_rank(size, owner), False)
                        for counts, run in expand(step):
                            result = run_and_compare(run, counts, rival_ref, f"B={size}")
                            compared += result["compared"]
                            differing += result["differing"]
                            differ += result["diffs"]
            records["rivals"].append(
                dict(rival=rival, rank=rank, compared=compared, differing=differing, differ=differ)
            )
        finally:
            setattr(M, *saved)
        say(f"rival {rival} done")

    # ------------------------------------------------------------------------------------------
    # Every rank's records to rank 0
    # ------------------------------------------------------------------------------------------

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    gathered = comm.gather(dict(records=records, peak_gib=peak_gib), root=0)
    if rank == 0:
        merged = {key: [r for per in gathered for r in per["records"][key]] for key in records}
        merged["peak_gib_per_rank"] = [per["peak_gib"] for per in gathered]
        merged["seconds"] = time.monotonic() - started
        with open(out_path, "w") as f:
            json.dump(merged, f)
        say(
            f"wrote {out_path}; peak GiB per rank {[round(g, 1) for g in merged['peak_gib_per_rank']]}"
        )
    graphs.clear()
    comm.Barrier()


def _run_one_rank(out: str, checkpoint: str) -> int:
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
        _rank_body(out, checkpoint)
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse(sys.argv[1:])
    if args.rank_worker:
        sys.exit(_run_one_rank(args.out, args.checkpoint))
    spawn_ranks(args.out, args.checkpoint)
