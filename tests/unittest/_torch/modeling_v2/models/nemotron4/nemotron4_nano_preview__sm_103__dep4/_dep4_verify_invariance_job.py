# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 4-rank job behind `test_modeling_v2_dep4_verify_invariance.py`: the dep4 MTP verify step,
run on every rank of one node on real weights and real cache state, and compared with the MTP-off
decode steps of the same tokens.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _dep4_verify_invariance_job.py --out <json> [--checkpoint <dir>]

Run plainly, this script is its own launcher: it starts one MPI rank per device under `mpirun`,
with a deadline it enforces by killing the job, because a collective that breaks hangs rather than
raising. The ranks are an MPI world since that is what a dep4 serving engine starts, and the routed
MoE's all-gather and reduce-scatter run on the MPI-session symbols over it.

What every rank builds, as the engine builds it for its rank under ``MTPDecodingConfig(max_draft_len
=2)``: the spec config resolved against the checkpoint (one-model MTP-Eagle over the one repeated MTP
layer); ``ModelConfig`` off the checkpoint with the rank's dep4 ``Mapping`` and that spec config; the
target class with every weight loaded (the rank's window of 128 experts of each stack, the MTP
layer's included) and ``post_load_weights``; the engine's hybrid cache manager for that mapping
(``_create_kv_cache_manager``, the default ``MambaHybridCacheManagerV2``), which under attention DP
holds the MTP layer's KV beside the trunk's and the verify's per-position buffers; the engine's
TRTLLM attention metadata (``build_attention_metadata``), filled per step as the runner and the
speculative step's ``update_spec_dec_param`` fill it, ``all_rank_num_tokens`` included; and, for
graphs, ``create_cuda_graph_metadata`` at draft length 2 and the runner's MTP padding dummy
(``CUDA_GRAPH_DUMMY_REQUEST_ID - 2``). A rank with no request runs the attention-DP dummy
(``ATTENTION_DP_DUMMY_REQUEST_ID``), a generation request carrying the step's drafts, as the
executor pads an idle rank under speculative decoding.

The requests: on every rank, three probes -- the twelve of the tp1 test (the ten prompts of the
fixed set, a history that fills one 512-key attention split exactly and one that ends mid-split),
dealt round-robin -- and fillers, prefilled through the target's own prefill steps on all four ranks
in lockstep, each followed by the MTP layer's first draft step over the same rows as the worker runs
it, so every request verifies from KV, recurrent and conv state the model wrote. A probe's
reference is its three MTP-off decode steps alone on its rank, eagerly, the other three ranks idle:
over t0, its greedy next token, then t1 and t2, the greedy tokens after it -- the drafts a perfect
drafter would propose. Every run starts from the same state: each GDP slot is restored from a
snapshot, and a row's KV appends land at the positions its own attention reads only after them.

The runs (`STEPS`), all four ranks in lockstep, each rank's probes compared on that rank: its
verify over [t0, t1, t2] alone (the guards on) and inside eager verify batches of every size while
the other ranks idle, verify, decode or prefill a 2048-token chunk; and inside padded CUDA-graph
verify batches with the other ranks idle or verifying. A verify runs the entries a decode step
runs, over 1 + D tokens a row: cuBLAS at 3B rows where decode has B, ``thop_attention``'s
generation kernels at three tokens a row, the GDP and conv verify entries, and the routed MoE over
every rank's tokens. Each picks its kernel by the step or sums in its own order, so a verify token
sits near its decode step's bits rather than on them. Per probe and token j, as tp1's test compares
them: within the bands (`LOGIT_BAND` and the rest), its fp32 logits, the state the verify stored
for it in every GDP layer and the K/V it appended in every attention layer; reported, whether its
logits move from its verify alone. A verify writes no slot.

Beside them: which entry the attention layers of a decode step and of a verify call; the worker's
commit (``update_mamba_states``) after a padded graph verify, on every rank and with one rank idle;
a request at the length limit, whose verify reaches past ``max_seq_len`` by the drafts; a verify
from a state one token too far, which must leave the bands; the drift, MTP on along MTP off's own
greedy tokens for as many tokens as dep4's MTP identity gate generates, each probe alone on its
rank, which that gate's near-tie band and the acceptance gate's verify term rest on; and the MTP
layer's draft step through both targets and ``MTPLayer``. The CUDA graphs are captured as the
engine captures them: one eager forward at every size first, which grows the attention workspace
all of them share to what the largest call needs, then the captures, which resize nothing. Rank 0
writes every rank's records to `--out` as JSON.

Not a pytest module: the ranks run one fixed sequence of collectives in lockstep, and neither this
file's name nor its functions match pytest's collection patterns. The launcher must not import
`tensorrt_llm` -- that calls `MPI_Init`, and an MPI-initialized process cannot start `mpirun` -- so
everything that does is imported inside the rank body.
"""

from __future__ import annotations

import argparse
import contextlib
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
_PID_DIR_ENV = "NEMOTRON4_DEP4_VERIFY_INVARIANCE_PID_DIR"
DEADLINE_S = 3600
"""Ceiling on the whole job: weight load, prefill, captures and every run."""

WORLD_SIZE = 4
"""dep4: attention DP over the four GPUs of one node, the routed experts split four ways."""

# The engine sizes this run uses everywhere (task.yaml, the boot gate, check_goldens).
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
TOKENS_PER_BLOCK = 32
DRAFT = 2
"""The checkpoint's trained draft depth; a verify row carries 1 + DRAFT tokens."""
TOKENS = 1 + DRAFT
NUM_REQUESTS = 256
"""Requests per rank: its three probes and its fillers."""
MAX_BATCH = 264
"""Rows a rank's step may hold; resident requests per rank: the probes, fillers, the loader, the
request at the length limit and the two dummies."""
KV_TOKENS = 192 * 1024
"""KV tokens of a rank's paged pool: every request stays resident."""
LOADER_TOKENS = 2048
"""A prefilling rank's chunk beside verifying ranks: the comm cells' prefill height."""
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
"""The CUDA-graph verify batch sizes captured per rank; a step runs in the smallest one that holds
the busiest rank's batch, as the runner pads every rank to it."""

IDLE = 0
"""A rank's entry when it has no request: it runs the attention-DP dummy, one row."""
PREFILL = "P"
"""A rank's entry when it prefills a `LOADER_TOKENS` chunk through PrefillTarget."""


@dataclass(frozen=True)
class Decode:
    """A rank's entry when it runs `rows` MTP-off decode rows (one token each) beside verifying
    ranks: a configuration a static draft length never makes, run to measure how far a verify
    token moves with the other ranks' token counts."""

    rows: int


@dataclass(frozen=True)
class Step:
    """One verify step of the attention-DP group.

    `counts[r]`: rank r's verify rows, `IDLE`, `PREFILL` or `Decode(n)`. `graph`: replayed in the
    captured verify graph of the smallest size that holds the busiest rank, every rank padded to it;
    else eager, every rank its own count.
    """

    group: str
    counts: tuple
    graph: bool

    @property
    def name(self) -> str:
        return f"{self.group} {[str(c) for c in self.counts]}"


def _one_rank(count, rank: int) -> tuple:
    return tuple(count if r == rank else IDLE for r in range(WORLD_SIZE))


EAGER_SIZES = (2, 3, 5, 8, 16, 64, 100, 256)
"""Eager verify batches on one rank while the others idle: every ragged split the target makes,
with the busiest rank in each position."""
GRAPH_BATCH_SIZES = (1, 2, 3, 5, 7, 8, 16, 64, 100, 250, 256)
"""Graph verify batches on one rank while the others idle; 3, 5, 7, 100 and 250 leave padding."""

STEPS: tuple[Step, ...] = (
    *(Step("eager alone, others idle", _one_rank(1, r), False) for r in range(WORLD_SIZE)),
    *(
        Step("eager, others idle", _one_rank(b, r), False)
        for r in range(WORLD_SIZE)
        for b in EAGER_SIZES
    ),
    *(
        Step("eager, others verifying", counts, False)
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
        Step("eager, others decoding", counts, False)
        for counts in (
            (3, Decode(64), Decode(5), Decode(256)),
            (Decode(256), 5, Decode(3), Decode(1)),
            (Decode(8), Decode(1), 64, IDLE),
            (Decode(7), Decode(16), Decode(250), 100),
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
        Step("graph, others verifying", counts, True)
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


def graph_size(step: Step) -> int | None:
    """The captured graph `step` replays in: the smallest that holds its busiest rank."""
    if not step.graph:
        return None
    busiest = max(1 if c == IDLE else c for c in step.counts)
    return min(g for g in GRAPH_SIZES if g >= busiest)


def step_tokens(step: Step) -> tuple[int, ...]:
    """Every rank's token count in `step`, in rank order: what the routed MoE's collectives take."""
    size = graph_size(step)
    if size is not None:
        return (size * TOKENS,) * WORLD_SIZE
    out = []
    for c in step.counts:
        if c == PREFILL:
            out.append(LOADER_TOKENS)
        elif isinstance(c, Decode):
            out.append(c.rows)
        elif c == IDLE:
            out.append(TOKENS)
        else:
            out.append(c * TOKENS)
    return tuple(out)


COMMIT_ROWS = 15
"""Rows per rank in the commit's graph verify: the probes and fillers, one padding row in the graph
of 16."""
SHORT_TOKENS, LONG_TOKENS = 256, 1024
"""Tokens dep4's MTP identity gate generates per prompt: the first from the prefill, the rest from
verify steps. The drift follows each probe for the verifies of that many tokens."""
TOP = 8
"""The decode reference's top tokens whose gap to its top-1 a verify must keep."""
WORST_KEYS = ("logit", "gap", "ssm", "conv", "kv")
"""The compared quantities a run records the worst case of."""

# The bands. A verify token differs from its decode step through every kernel whose choice moves
# with the step: cuBLAS at 3B rows against B, thop_attention's generation kernels at three tokens a
# row against one, the GDP and conv verify entries against the decode ones, and the routed MoE's
# window and reduce-scatter over every rank's tokens, which the other ranks' load changes. Each
# rounds within a bf16 step of the other, and every later layer carries the difference into the
# token's state, its K/V and its logits. How far is measured over every run below, every rank: each
# probe's three tokens alone, in eager batches beside idle, verifying, decoding or prefilling ranks,
# in padded graphs, at the length limit and through the commit. The kernels are deterministic, so
# the worst cases repeat run to run: |logit diff| 0.7644 alone and 0.9792 in batches and graphs;
# top-gap shift 0.2601 alone, 0.3949 in batches; recurrent state, conv state and K/V 0.04898 /
# 0.06904 / 0.08275 of max |decode| (the commit's slots 0.0229 / 0.0447); no top-1 changed. Each
# band is its worst case plus a quarter, rounded up: room for a batch, history, graph size or load
# on the other ranks the runs did not take, where cuBLAS or thop may pick yet another kernel.
LOGIT_BAND = 1.23
"""max |verify logit - decode logit| over the vocabulary, fp32."""
GAP_BAND = 0.50
"""The largest change, verify against decode, of the gap between the decode step's top-1 and any of
its top `TOP` tokens: how close to a tie the verify can tip the decode step's choice."""
SSM_BAND = 0.062
CONV_BAND = 0.087
KV_BAND = 0.11
"""max |verify - decode| / max |decode| per tensor, over every GDP layer (recurrent state fp32,
conv state bf16) and every attention layer (K/V)."""
DRIFT_LOGIT_BAND = 5.03
DRIFT_GAP_BAND = 1.17
"""The same two quantities over the drift's 6132 matched positions, where the verify's state and K/V
carry each earlier verify's deviation: worst cases 4.0192 (|logit diff|, on the 2-shot chat
`clips`; 0.61-1.41 on every other probe) and 0.9300 (top-gap shift, on `haiku`), plus a quarter,
rounded up; ten top-1 changes, at decode margins of at most 0.0384. dep4's MTP identity gate's
near-tie band is built on the measured top-gap shift; its MTP acceptance gate's verify term is the
measured |logit diff|."""


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
    pid_dir = tempfile.mkdtemp(prefix="dep4_verify_invariance_")
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
    """One sequence on one rank: its prompt, the tokens its verify step feeds -- the prompt's
    greedy next token t0, then two more -- and how many tokens dep4's MTP identity gate generates
    for it."""

    rid: int
    name: str
    prompt: tuple
    tokens: tuple = ()
    generated: int = SHORT_TOKENS

    @property
    def drift_steps(self) -> int:
        """The verifies of the drift: every generated token but the prefill's, three a step."""
        return (self.generated - 1) // TOKENS


@dataclass
class Run:
    """One lockstep run: this rank's part of it."""

    step: str
    group: str
    graph_size: int | None
    kind: str = "idle"  # "verify", "decode", "prefill" or "idle"
    rows: list = field(default_factory=list)  # this rank's Requests, or [] when idle / prefilling
    probes: list = field(default_factory=list)  # row positions of this rank's probes


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
    from _validating import validating

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
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    started = time.monotonic()

    def say(message: str) -> None:
        if rank == 0:
            print(f"[job {time.monotonic() - started:7.1f}s] {message}", flush=True)

    def bits(x: torch.Tensor) -> torch.Tensor:
        return x.view({4: torch.int32, 2: torch.int16}[x.element_size()])

    def equal(a: torch.Tensor, b: torch.Tensor) -> bool:
        return torch.equal(bits(a), bits(b))

    records: dict[str, list] = dict(
        reference=[],
        attention=[],
        runs=[],
        commit=[],
        limit=[],
        teeth=[],
        drift=[],
        draft=[],
    )

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
    config = spec = None
    for turn in range(WORLD_SIZE):
        if turn == rank:
            # `MTPDecodingConfig(max_draft_len=2)` resolved against the checkpoint as the model
            # loader resolves it: one-model MTP-Eagle over the one repeated MTP layer.
            spec = MTPDecodingConfig(max_draft_len=DRAFT)
            update_spec_config_from_model_config(
                spec,
                ModelConfig.from_pretrained(
                    checkpoint_path, trust_remote_code=True
                ).pretrained_config,
            )
            config = ModelConfig.from_pretrained(
                checkpoint_path, trust_remote_code=True, mapping=mapping, spec_config=spec
            )
            validate_and_set_mamba_ssm_cache_dtype(config, "auto")
        comm.Barrier()
    assert spec.spec_dec_mode.is_mtp_eagle_one_model(), spec.spec_dec_mode
    assert spec.max_draft_len == DRAFT and spec.tokens_per_gen_step == TOKENS
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
    lm_head = model.lm_head.weight
    mtp_layer = model.draft_model.mtp_layers[0]
    assert (core.expert_offset, core.local_experts) == (rank * 128, 128)
    assert core.mtp_layer == 44 and core.tokens_per_gen_step == TOKENS
    weights_gib = torch.cuda.memory_allocated() / 2**30
    say(f"target built on every rank, MTP layer included: {weights_gib:.2f} GiB on rank 0")

    extra_kv = get_num_extra_kv_tokens(spec)
    # The cache managers' bound on a sequence, as `create_py_executor` sets it under one-model
    # speculative decoding with the overlap scheduler on: the drafts of the step in flight, the
    # extra KV tokens of the one-engine drafter, and the drafts of the next step.
    cache_max_seq_len = MAX_SEQ_LEN + 2 * (TOKENS - 1) + extra_kv
    kv_cache_config = KvCacheConfig(
        max_tokens=KV_TOKENS, enable_block_reuse=False, free_gpu_memory_fraction=0.5
    )
    manager = _create_kv_cache_manager(
        None,
        get_kv_cache_manager_cls(config, kv_cache_config),
        mapping,
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=cache_max_seq_len,
        max_batch_size=MAX_BATCH,
        spec_config=spec,
        sparse_attention_config=None,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=config,
        dtype=torch.bfloat16,
        max_cuda_graph_batch_size=max(GRAPH_SIZES),
    )
    # Under attention DP the MTP layer's KV sits in this one manager, beside the trunk's.
    assert manager.layer_offsets[core.mtp_layer] == 40, manager.layer_offsets
    assert manager.get_intermediate_ssm_states(0).shape[:2] == (MAX_BATCH, TOKENS)
    md = build_attention_metadata(
        config,
        max_batch_size=MAX_BATCH,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        attention_backend=get_attention_backend("TRTLLM"),
        attention_runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=False,
            cache_reuse=False,
            has_speculative_draft_tokens=True,
            chunk_size=MAX_NUM_TOKENS,
        ),
        mapping=mapping,
        cache_indirection=None,
        kv_cache_manager=manager,
    )
    # The shell binds its draft length into the state metadata it hands the engine.
    md.mamba_metadata_cls = model.mamba_metadata_cls
    assert md.mamba_metadata_cls.max_draft_len == DRAFT
    gdp_layers, attention_layers = core.gdp_layers, core.attention_layers

    # The requests: this rank's probes (the tp1 test's twelve, dealt round-robin), its fillers,
    # its loader (a 2048-token prompt it prefills beside other ranks' verify steps) and a request
    # at the length limit, its history zeroed rather than prefilled.
    prompts = P.prompt_set(tokenizer)
    by_name = {p.name: p for p in prompts}
    specs = [(p.name, p.token_ids, LONG_TOKENS if p.long else SHORT_TOKENS) for p in prompts]
    specs.append(("gpqa_explain[:511]", by_name["gpqa_explain"].token_ids[:511], SHORT_TOKENS))
    specs.append(
        ("gsm8k_multiturn[:4096]", by_name["gsm8k_multiturn"].token_ids[:4096], SHORT_TOKENS)
    )
    assert len(specs) == 3 * WORLD_SIZE, len(specs)
    probes = [
        Request(0, n, tuple(t), generated=g)
        for k, (n, t, g) in enumerate(specs)
        if k % WORLD_SIZE == rank
    ]
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
    limit = Request(0, "at the length limit", (0,) * (MAX_SEQ_LEN - 2))
    requests = probes + fillers
    resident = requests + [loader, limit]
    for rid, r in enumerate(resident, start=1):
        r.rid = rid
    for r in resident:
        # The prompt and the step's three tokens, a probe's drift besides; the manager adds the
        # drafter's extra KV token.
        drift = TOKENS * r.drift_steps if r in probes else 0
        added = manager.add_dummy_requests([r.rid], token_nums=[len(r.prompt) + drift + TOKENS])
        assert added, f"rank {rank}: no room for {r.name} ({len(r.prompt)} tokens)"
    # The executor's attention-DP dummy for an idle rank, and the runner's MTP padding dummy: each a
    # generation request carrying the step's drafts over a history of as many tokens.
    padding_id = CUDA_GRAPH_DUMMY_REQUEST_ID - DRAFT
    for dummy in (ATTENTION_DP_DUMMY_REQUEST_ID, padding_id):
        added = manager.add_dummy_requests([dummy], is_gen=True, max_num_draft_tokens=DRAFT)
        assert added, f"rank {rank}: no room for dummy {dummy}"
    ids = [r.rid for r in resident]
    slot = dict(zip(ids, manager.get_state_indices(ids)))
    assert len(set(slot.values())) == len(ids)

    # ------------------------------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------------------------------

    pad_tokens = torch.Generator(device="cuda").manual_seed(9 + rank)
    vocab = config.pretrained_config.text_config.vocab_size

    def fill(
        meta,
        ctx: list,
        gen: list,
        tokens: int,
        counts: list,
        cached=None,
        adp_dummy=False,
        padding=0,
    ) -> None:
        """The step's per-request fields, set as the runner's `_prepare_tp_inputs` and the
        speculative step's `update_spec_dec_param` set them, then `prepare()`, then every rank's
        token count. Context rows feed their whole prompt; generation rows feed `tokens` tokens
        over `cached` (default: the prompt); the attention-DP dummy and `padding` rows feed
        `tokens` tokens over their own history of DRAFT tokens."""
        # The cache manager stages a step's block-table gather index in one pinned host buffer
        # that the device reads only when the step's copy runs (`IndexMapper::getCopyIndex`): a
        # step filled while an earlier one is still queued would hand the earlier one its rows,
        # and that step's K/V appends would land in another request's pages. So the device
        # drains first.
        torch.cuda.synchronize()
        placeholders = int(adp_dummy) + padding
        cached = [len(r.prompt) for r in gen] if cached is None else list(cached)
        if not meta.is_cuda_graph:
            meta.seq_lens = torch.tensor(
                [len(r.prompt) for r in ctx] + [tokens] * (len(gen) + placeholders),
                dtype=torch.int,
            )
        meta.beam_width = 1
        meta.request_ids = (
            [r.rid for r in ctx + gen]
            + [ATTENTION_DP_DUMMY_REQUEST_ID] * int(adp_dummy)
            + [padding_id] * padding
        )
        meta.prompt_lens = [len(r.prompt) for r in ctx + gen] + [DRAFT] * placeholders
        meta.num_contexts = len(ctx)
        meta.num_chunked_ctx_requests = 0
        meta.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * len(ctx) + cached + [DRAFT] * placeholders,
            num_extra_kv_tokens=extra_kv,
        )
        meta.kv_cache_manager = manager
        meta.padded_num_tokens = None
        meta.update_spec_dec_param(
            batch_size=len(ctx) + len(gen) + placeholders,
            is_spec_decoding_enabled=True,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=DRAFT,
            max_total_draft_tokens=DRAFT,
            num_contexts=len(ctx),
        )
        meta.prepare()
        # The runner sets this after `prepare()`, every rank's count in rank order.
        meta.all_rank_num_tokens = list(counts)

    def noise(n: int) -> torch.Tensor:
        """Arbitrary tokens, as the runner leaves whatever its static buffer held on placeholder
        rows."""
        return torch.randint(0, vocab, (n,), generator=pad_tokens, dtype=torch.int, device="cuda")

    def as_ids(tokens: list) -> torch.Tensor:
        return torch.tensor(tokens, dtype=torch.int, device="cuda")

    def trunk(meta, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The trunk's fp32 logits for every token and its hidden states: the core's forward, then
        the lm_head the shell runs (`cublas_mm` with an fp32 output) on every row, as it does for
        the tokens a decode step or a verify step gathers."""
        hidden = core.forward(attn_metadata=meta, input_ids=input_ids)
        return M.cublas_mm(hidden, lm_head.t(), None, torch.float32), hidden

    # --- prefill: every request through PrefillTarget, then the MTP layer's first draft step ---
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
            tokens = sum(len(b.prompt) for b in batch) if batch else TOKENS
            counts = comm.allgather(tokens)
            if batch:
                fill(md, batch, [], TOKENS, counts)
                ids = as_ids([t for b in batch for t in b.prompt])
            else:
                fill(md, [], [], TOKENS, counts, adp_dummy=True)
                ids = noise(TOKENS)
            hidden = core.forward(attn_metadata=md, input_ids=ids)
            if batch:
                last = torch.cumsum(md.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                logits = M.cublas_mm(hidden[last], lm_head.t(), None, torch.float32)
                for b, token in zip(batch, logits.argmax(-1).tolist()):
                    b.tokens = (token,)
                shifted = as_ids([t for b in batch for t in b.prompt[1:] + b.tokens])
            else:
                shifted = noise(TOKENS)
            # The worker's first draft step over the same rows: each token's hidden state with the
            # next token's embedding, which writes the MTP layer's KV of the prompt.
            mtp_layer(core.w["embed"], counts, shifted, hidden, md)
        torch.cuda.synchronize()
    say(
        f"prefilled {len(requests)} requests per rank in {num_steps} lockstep steps, MTP KV included"
    )

    # The request at the limit: zero history, a fresh slot, and t0 a token of the text.
    kv_buffers = {i: manager.get_buffers(i) for i in attention_layers}
    pages = {
        r.rid: {
            i: manager.get_batch_cache_indices([r.rid], layer_idx=i)[0] for i in attention_layers
        }
        for r in probes + [limit]
    }
    for i in attention_layers:
        kv_buffers[i][torch.tensor(pages[limit.rid][i], dtype=torch.long, device="cuda")] = 0
    for i in gdp_layers:
        manager.get_ssm_states(i)[slot[limit.rid]] = 0
        manager.get_conv_states(i)[slot[limit.rid]] = 0
    limit.tokens = probes[0].tokens
    torch.cuda.synchronize()

    snapshot = [
        (manager.get_ssm_states(i).clone(), manager.get_conv_states(i).clone()) for i in gdp_layers
    ]

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

    def kv_at(r: Request, layer: int, j: int) -> torch.Tensor:
        pos = len(r.prompt) + j
        page = pages[r.rid][layer][pos // TOKENS_PER_BLOCK]
        return kv_buffers[layer][page, :, pos % TOKENS_PER_BLOCK, 0].clone()

    def slot_token(r: Request, j: int, logits: torch.Tensor) -> dict:
        """Token j of `r` as a decode step left it: its logits, its slot's state, its K/V."""
        s = slot[r.rid]
        return dict(
            logits=logits.clone(),
            ssm=[manager.get_ssm_states(i)[s].clone() for i in gdp_layers],
            conv=[manager.get_conv_states(i)[s].clone() for i in gdp_layers],
            kv=torch.stack([kv_at(r, i, j) for i in attention_layers]),
        )

    def verify_token(r: Request, row: int, j: int, logits: torch.Tensor) -> dict:
        """Token j of `r` as generation row `row` of a verify left it: its logits, the state the
        verify stored for it in the per-position buffers, its K/V."""
        return dict(
            logits=logits[row * TOKENS + j].clone(),
            ssm=[manager.get_intermediate_ssm_states(i)[row, j].clone() for i in gdp_layers],
            conv=[manager.get_intermediate_conv_states(i)[row, j].clone() for i in gdp_layers],
            kv=torch.stack([kv_at(r, i, j) for i in attention_layers]),
        )

    def relative(got: torch.Tensor, ref: torch.Tensor) -> float:
        """max |got - ref| / max |ref|."""
        ref = ref.float()
        return float((got.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))

    def logit_deviation(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
        """`(max |got - ref|, the largest change of a gap between ref's top-1 and one of its top
        TOP)`: the second bounds how far `got` can tip a choice `ref` made among its top tokens."""
        diff = got - ref
        top = ref.topk(TOP).indices
        shift = diff[top] - diff[top[0]]
        return float(diff.abs().max()), float(shift.abs().max())

    def margin_of(logits: torch.Tensor) -> float:
        """The top-1 minus top-2 logit: how close a greedy choice is to a tie."""
        two = logits.topk(2).values
        return float(two[0] - two[1])

    def deviation(got: dict, ref: dict) -> dict:
        """A verify token against its decode step, in every layer."""
        logit, gap = logit_deviation(got["logits"], ref["logits"])
        return dict(
            logit=logit,
            gap=gap,
            ssm=max(relative(a, b) for a, b in zip(got["ssm"], ref["ssm"])),
            conv=max(relative(a, b) for a, b in zip(got["conv"], ref["conv"])),
            kv=max(relative(a, b) for a, b in zip(got["kv"], ref["kv"])),
            flip=int(got["logits"].argmax()) != int(ref["logits"].argmax()),
            margin=margin_of(ref["logits"]),
        )

    def outside(d: dict) -> list:
        """What of a deviation lies outside the bands."""
        out = []
        for name, key, band in (
            ("logits", "logit", LOGIT_BAND),
            ("top-gap shift", "gap", GAP_BAND),
            ("recurrent state", "ssm", SSM_BAND),
            ("conv state", "conv", CONV_BAND),
            ("K/V", "kv", KV_BAND),
        ):
            if d[key] > band:
                out.append(f"{name} {d[key]:.4g} outside its band {band:.4g}")
        return out

    def worst(deviations: list) -> dict:
        """The worst case of each compared quantity, for the records."""
        if not deviations:
            return {}
        return {key: max(d[key] for d in deviations) for key in WORST_KEYS} | {
            "flip_margins": [d["margin"] for d in deviations if d["flip"]]
        }

    graphs: dict[int, tuple] = {}
    graph_pool = torch.cuda.graph_pool_handle()
    static_ids = torch.zeros(max(GRAPH_SIZES) * TOKENS, dtype=torch.int, device="cuda")

    def verify_ids(rows: list, placeholders: int) -> torch.Tensor:
        """A verify step's input ids, request-major: each row's three tokens, then arbitrary
        tokens on the placeholder rows."""
        real = as_ids([t for r in rows for t in r.tokens]) if rows else as_ids([])
        return torch.cat([real, noise(placeholders * TOKENS)])

    def run_eager(run: Run, counts: list) -> torch.Tensor:
        """One eager step from the snapshot: this rank's verify or decode rows, its loader's
        prefill, or the attention-DP dummy; every token's logits."""
        restore()
        if run.kind == "prefill":
            fill(md, [loader], [], TOKENS, counts)
            return trunk(md, as_ids(list(loader.prompt)))[0]
        if run.kind == "decode":
            fill(md, [], run.rows, 1, counts)
            return trunk(md, as_ids([r.tokens[0] for r in run.rows]))[0]
        if run.kind == "verify":
            fill(md, [], run.rows, TOKENS, counts)
            return trunk(md, verify_ids(run.rows, 0))[0]
        fill(md, [], [], TOKENS, counts, adp_dummy=True)
        return trunk(md, verify_ids([], 1))[0]

    workspace = {}

    def capture_graphs() -> None:
        """Capture the verify step at every size of GRAPH_SIZES on every rank, in lockstep, as
        the runner does: first one eager forward on every size's own CUDA-graph metadata (draft
        length 2), which grows the attention workspace all of them share to what the largest call
        needs -- thop_attention resizes it in place when a call needs more -- then at each size
        two warmup forwards and the capture, which resize nothing. A resize after a capture would
        move the workspace under the graph, which keeps its old address."""
        rows = probes + fillers
        metadata = {}
        for size in sorted(GRAPH_SIZES, reverse=True):
            gmd = md.create_cuda_graph_metadata(size, False, DRAFT, get_memory_buffers())
            fill(gmd, [], rows[:size], TOKENS, [size * TOKENS] * WORLD_SIZE)
            static_ids[: size * TOKENS].copy_(verify_ids(rows[:size], 0))
            trunk(gmd, static_ids[: size * TOKENS])
            metadata[size] = gmd
        torch.cuda.synchronize()
        comm.Barrier()
        workspace["address"] = metadata[max(GRAPH_SIZES)].cuda_graph_workspace.data_ptr()
        for size in GRAPH_SIZES:
            gmd = metadata[size]
            fill(gmd, [], rows[:size], TOKENS, [size * TOKENS] * WORLD_SIZE)
            static_ids[: size * TOKENS].copy_(verify_ids(rows[:size], 0))
            ids = static_ids[: size * TOKENS]
            for _ in range(2):
                trunk(gmd, ids)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=graph_pool):
                out, _ = trunk(gmd, ids)
            torch.cuda.synchronize()
            comm.Barrier()
            graphs[size] = (gmd, graph, out)
        assert gmd.cuda_graph_workspace.data_ptr() == workspace["address"], (
            "a capture resized the attention workspace"
        )
        restore()

    def run_graph(run: Run, ids=None, cached=None, from_snapshot: bool = True) -> torch.Tensor:
        """One verify step replayed in the captured graph of `run.graph_size`: this rank's rows
        (or the attention-DP dummy) padded with the padding dummy's rows; the real rows' logits.
        Each row feeds its three tokens over its prompt from the snapshot, or `ids` over `cached`
        tokens from the slots as they are when `from_snapshot` is off."""
        size = run.graph_size
        idle = not run.rows
        padding = size - len(run.rows) - int(idle)
        counts = [size * TOKENS] * WORLD_SIZE
        gmd, graph, out = graphs[size]
        assert gmd.cuda_graph_workspace.data_ptr() == workspace["address"], (
            "the attention workspace moved under the captured graphs"
        )
        if from_snapshot:
            restore()
        fill(gmd, [], run.rows, TOKENS, counts, cached=cached, adp_dummy=idle, padding=padding)
        if ids is None:
            ids = verify_ids(run.rows, int(idle) + padding)
        else:
            ids = torch.cat([ids, noise(int(idle) * TOKENS + padding * TOKENS)])
        static_ids[: size * TOKENS].copy_(ids)
        graph.replay()
        return out[: len(run.rows) * TOKENS]

    def commit_rows(meta, accepted: list) -> None:
        """The worker's commit after a verify step: `update_mamba_states` with the engine's own
        slots, placeholder rows included, each row's accepted count (its last token's, 1 to 3)."""
        manager.update_mamba_states(
            attn_metadata=meta,
            num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device="cuda"),
            state_indices=meta.mamba_metadata.state_indices,
        )

    def execute(run: Run, counts: list) -> torch.Tensor:
        return run_graph(run) if run.graph_size else run_eager(run, counts)

    # --- the plan: every rank expands every step the same way ---

    offsets = {}

    def batches_of(size: int) -> list[tuple[list, list]]:
        """Batches of `size` rows on this rank holding each of its probes at least once, with the
        probes' row positions: spread from the first row to the last when the batch holds all
        three, else taken in turn (reversed every other batch). Fillers fill the rest, from an
        offset of their own per size."""
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
        """`step`'s lockstep runs: (every rank's token count, this rank's part). A rank that
        decodes compares nothing; its probes are compared where it verifies."""
        verifying = [
            c for c in step.counts if c not in (IDLE, PREFILL) and not isinstance(c, Decode)
        ]
        num_runs = max(len(batches_of(c)) for c in verifying) if verifying else 1
        mine = step.counts[rank]
        counts = list(step_tokens(step))
        out = []
        for k in range(num_runs):
            run = Run(step=step.name, group=step.group, graph_size=graph_size(step))
            if mine == PREFILL:
                run.kind = "prefill"
            elif isinstance(mine, Decode):
                # Any of the rank's requests, its probes included: a decoding rank compares none.
                run.kind = "decode"
                offset = (mine.rows * 5 + 3 * rank) % len(requests)
                run.rows = (requests[offset:] + requests[:offset])[: mine.rows]
                assert len(run.rows) == mine.rows, (mine, len(requests))
            elif mine != IDLE:
                run.kind = "verify"
                options = batches_of(mine)
                run.rows, run.probes = options[k % len(options)]
            out.append((counts, run))
        return out

    # --- the reference: each probe's three decode steps alone on its rank, the others idle ---
    def decode_alone(owner: int, r: Request | None, j: int) -> tuple:
        """One lockstep MTP-off decode step: `owner` decodes `r`'s token j over its prompt and
        tokens 0..j-1, every other rank runs its attention-DP dummy; `owner`'s logits and hidden
        state."""
        counts = [1] * WORLD_SIZE
        if rank == owner:
            fill(md, [], [r], 1, counts, cached=[len(r.prompt) + j])
            logits, hidden = trunk(md, as_ids([r.tokens[j]]))
            return logits[0], hidden
        fill(md, [], [], 1, counts, adp_dummy=True)
        trunk(md, noise(1))
        return None, None

    def reference_steps(owner: int, r: Request | None) -> tuple[list, object]:
        """`r`'s three decode steps from the snapshot, on `owner` alone: what each token left and
        the trunk's hidden state of t0. Fixes `r.tokens` to t0 and the greedy tokens after it."""
        restore()
        steps, hidden0 = [], None
        for j in range(TOKENS):
            logits, hidden = decode_alone(owner, r, j)
            if rank == owner:
                steps.append(slot_token(r, j, logits))
                if j == 0:
                    hidden0 = hidden.clone()
                if j + 1 < TOKENS:
                    r.tokens = r.tokens[: j + 1] + (int(logits.argmax()),)
        restore()
        return steps, hidden0

    reference, hidden0 = {}, {}
    with torch.inference_mode():
        for owner in range(WORLD_SIZE):
            for k in range(len(probes)):
                r = probes[k] if rank == owner else None
                steps, h0 = reference_steps(owner, r)
                if rank == owner:
                    reference[r.rid], hidden0[r.rid] = steps, h0
        # Fillers verify their t0 and two arbitrary tokens.
        g = torch.Generator().manual_seed(17 + rank)
        for r in fillers:
            r.tokens = r.tokens + tuple(torch.randint(0, vocab, (DRAFT,), generator=g).tolist())
        loader.tokens = (0,)
        capture_graphs()
    say(f"graphs captured at {sorted(graphs)} on every rank, after a warmup at every size")
    for r in probes:
        steps = reference[r.rid]
        records["reference"].append(
            dict(
                rank=rank,
                probe=r.name,
                prompt_tokens=len(r.prompt),
                tokens=list(r.tokens),
                finite=all(bool(torch.isfinite(t["logits"]).all()) for t in steps),
                kv_nonzero=all(bool(kv.any()) for t in steps for kv in t["kv"]),
                states_nonzero=all(bool(s.any()) for t in steps for s in t["ssm"] + t["conv"]),
                # The slot's state after tokens 1, 2 and 3 differs pairwise in every GDP layer,
                # so a state taken from the wrong position fails the comparison.
                positions_distinct=all(
                    not equal(steps[a][name][layer], steps[b][name][layer])
                    for a in range(TOKENS)
                    for b in range(a + 1, TOKENS)
                    for name in ("ssm", "conv")
                    for layer in range(len(gdp_layers))
                ),
            )
        )
    say(f"reference: {len(probes)} probes' three decode steps alone on every rank")

    # --- which entry attention calls: q rows per call, a decode step's and a verify's ---
    original = M.thop_attention
    calls = []

    def counted(**kwargs):
        calls.append(int(kwargs["q"].shape[0]))
        return original(**kwargs)

    M.thop_attention = counted
    try:
        with torch.inference_mode():
            r = probes[0]
            restore()
            fill(md, [], [r], 1, [1] * WORLD_SIZE)
            trunk(md, as_ids([r.tokens[0]]))
            decode_calls, calls[:] = list(calls), []
            restore()
            fill(md, [], [r], TOKENS, [TOKENS] * WORLD_SIZE)
            trunk(md, verify_ids([r], 0))
            verify_calls = list(calls)
            restore()
    finally:
        M.thop_attention = original
    records["attention"].append(
        dict(rank=rank, layers=len(attention_layers), decode=decode_calls, verify=verify_calls)
    )
    say("attention: a decode step and a verify of every rank's first probe")

    alone = {}

    def run_and_compare(run: Run, counts: list, ref: dict, what: str) -> dict:
        """Run `run`, compare its probes' tokens with `ref` (`deviation`), report how many of its
        probes' verify logits move from their verify alone, and list the slots it wrote that its
        kind does not allow: a verify writes none, a decode only its rows', a prefill only the
        loader's, the attention-DP dummy none. A probe's verify alone (eager, the other ranks
        idle) is the first verify of it a run makes alone."""
        logits = execute(run, counts)
        failures, deviations = [], []
        for pos in run.probes:
            r = run.rows[pos]
            for j in range(TOKENS):
                d = deviation(verify_token(r, pos, j, logits), ref[r.rid][j])
                deviations.append(d)
                failures += [
                    f"{what}, rank {rank}, {r.name} at row {pos}, token {j}: {f}"
                    for f in outside(d)
                ]
        moved, moved_logit = 0, 0.0
        if run.kind == "verify":
            for pos in run.probes:
                r = run.rows[pos]
                mine = logits[pos * TOKENS : (pos + 1) * TOKENS]
                if len(run.rows) == 1 and not run.graph_size and r.rid not in alone:
                    alone[r.rid] = mine.clone()
                elif r.rid in alone and not equal(mine, alone[r.rid]):
                    moved += 1
                    moved_logit = max(moved_logit, float((mine - alone[r.rid]).abs().max()))
        allowed = set()
        if run.kind == "decode":
            allowed = {slot[r.rid] for r in run.rows}
        elif run.kind == "prefill":
            allowed = {slot[loader.rid]}
        return dict(
            compared=len(run.probes),
            failures=failures,
            worst=worst(deviations),
            moved=moved,
            moved_logit=moved_logit,
            illegal_writes=sorted(changed_slots() - allowed),
        )

    # --- every step; a probe's verify alone runs first, with every entry's guard on ---
    with torch.inference_mode():
        for step in STEPS:
            for counts, run in expand(step):
                guards = (
                    validating(
                        M.thop_attention,
                        M.cublas_mm,
                        M.gdp_verify,
                        M.causal_conv1d_verify,
                    )
                    if step.group == "eager alone, others idle"
                    else contextlib.nullcontext()
                )
                with guards:
                    result = run_and_compare(run, counts, reference, step.name)
                records["runs"].append(
                    dict(
                        step=step.name,
                        group=step.group,
                        kind=run.kind,
                        counts=counts,
                        graph_size=run.graph_size,
                        rank=rank,
                        rows=len(run.rows),
                        **result,
                    )
                )
    say(f"runs: {len(records['runs'])} on rank 0, graphs captured at {sorted(graphs)}")

    # --- the commit: a padded graph verify, then the worker's `update_mamba_states` ---
    def commit(idle: bool, accepted: list) -> dict:
        """A graph verify of this rank's commit rows (or, `idle`, the attention-DP dummy) padded to
        16, then the commit with the engine's own slots and the accepted counts; each probe's slot
        must hold exactly the state the verify stored for its last accepted token -- in every GDP
        layer within the state bands of its decode steps' -- and only the step's rows' engine slots
        may change."""
        rows = [] if idle else (probes + fillers)[:COMMIT_ROWS]
        run = Run(step="commit", group="commit", graph_size=16, kind="idle" if idle else "verify")
        run.rows = rows
        execute(run, [16 * TOKENS] * WORLD_SIZE)
        gmd = graphs[16][0]
        num_rows = 16
        stored = {
            r.rid: (
                [manager.get_intermediate_ssm_states(i)[row, n - 1].clone() for i in gdp_layers],
                [manager.get_intermediate_conv_states(i)[row, n - 1].clone() for i in gdp_layers],
            )
            for row, (r, n) in enumerate(zip(probes if not idle else [], accepted))
        }
        commit_rows(gmd, accepted[: len(rows)] + [1] * (num_rows - len(rows)))
        torch.cuda.synchronize()
        engine_slots = set(gmd.mamba_metadata.state_indices[:num_rows].tolist())
        written = changed_slots()
        failures, ssm_dev, conv_dev = [], 0.0, 0.0
        for r, n in zip(probes if not idle else [], accepted):
            s, ref = slot[r.rid], reference[r.rid][n - 1]
            for k, layer in enumerate(gdp_layers):
                ssm, conv = manager.get_ssm_states(layer)[s], manager.get_conv_states(layer)[s]
                if not (equal(ssm, stored[r.rid][0][k]) and equal(conv, stored[r.rid][1][k])):
                    failures.append(
                        f"rank {rank} {r.name}, {n} accepted: the slot is not the stored state, "
                        f"layer {layer}"
                    )
                ssm_dev = max(ssm_dev, relative(ssm, ref["ssm"][k]))
                conv_dev = max(conv_dev, relative(conv, ref["conv"][k]))
        if ssm_dev > SSM_BAND or conv_dev > CONV_BAND:
            failures.append(
                f"rank {rank}: committed state {ssm_dev:.4g} / {conv_dev:.4g} outside the bands"
            )
        restore()
        return dict(
            rank=rank,
            idle=idle,
            rows=len(rows),
            accepted=accepted[: len(probes)] if not idle else [],
            failures=failures,
            ssm=ssm_dev,
            conv=conv_dev,
            outside=sorted(written - engine_slots),
            real_written=sorted(written & {slot[r.rid] for r in rows}),
        )

    with torch.inference_mode():
        accepted = [1 + k % TOKENS for k in range(COMMIT_ROWS)]
        records["commit"].append(commit(False, accepted))
        records["commit"].append(
            commit(rank == WORLD_SIZE - 1, [1 + (k + 1) % TOKENS for k in range(COMMIT_ROWS)])
        )
    say("commit: every rank, then with the last rank idle")

    # --- a request at the length limit: its verify reaches past MAX_SEQ_LEN by the drafts ---
    with torch.inference_mode():
        # Its three decode steps, every rank its own request at the limit at once.
        restore()
        steps = []
        for j in range(TOKENS):
            counts = [1] * WORLD_SIZE
            fill(md, [], [limit], 1, counts, cached=[len(limit.prompt) + j])
            logits, _ = trunk(md, as_ids([limit.tokens[j]]))
            steps.append(slot_token(limit, j, logits[0]))
            if j + 1 < TOKENS:
                limit.tokens = limit.tokens[: j + 1] + (int(logits[0].argmax()),)
        restore()
        reference[limit.rid] = steps
        calls = []
        original = M.thop_attention

        def recorded(**kwargs):
            calls.append(dict(kwargs))
            return original(**kwargs)

        M.thop_attention = recorded
        try:
            with validating(original):
                run = Run(step="limit", group="limit", graph_size=None, kind="verify")
                run.rows, run.probes = [limit], [0]
                at_limit = run_and_compare(
                    run, [TOKENS] * WORLD_SIZE, reference, "alone at the limit"
                )
        finally:
            M.thop_attention = original
        longest = int(calls[0]["sequence_length"][0])
        bound, window = calls[0]["max_seq_len"], calls[0]["attention_window_size"]
        run = Run(step="limit graph", group="limit", graph_size=4, kind="verify")
        run.rows, run.probes = [limit, probes[0], probes[1]], [0, 1, 2]
        in_graph = run_and_compare(
            run, [4 * TOKENS] * WORLD_SIZE, reference, "graph B=3 at the limit"
        )
        restore()
        records["limit"].append(
            dict(
                rank=rank,
                longest=longest,
                bound=bound,
                window=window,
                finite=all(bool(torch.isfinite(t["logits"]).all()) for t in steps),
                failures=at_limit["failures"] + in_graph["failures"],
                compared=at_limit["compared"] + in_graph["compared"],
            )
        )
    say(f"limit: lengths up to {longest}, max_seq_len {bound}, window {window} (net {MAX_SEQ_LEN})")

    # --- the bands' teeth: a verify from the state one token too far, each probe alone ---
    with torch.inference_mode():
        smallest, smallest_gap, lines = float("inf"), float("inf"), []
        for owner in range(WORLD_SIZE):
            for k in range(len(probes)):
                restore()
                if rank == owner:
                    r = probes[k]
                    after_t0 = reference[r.rid][0]
                    for i, layer in enumerate(gdp_layers):
                        manager.get_ssm_states(layer)[slot[r.rid]] = after_t0["ssm"][i]
                        manager.get_conv_states(layer)[slot[r.rid]] = after_t0["conv"][i]
                    fill(md, [], [r], TOKENS, [TOKENS] * WORLD_SIZE)
                    logits, _ = trunk(md, verify_ids([r], 0))
                    moved = [
                        logit_deviation(logits[j], reference[r.rid][j]["logits"])
                        for j in range(TOKENS)
                    ]
                    worst_logit = max(m[0] for m in moved)
                    worst_gap = max(m[1] for m in moved)
                    smallest = min(smallest, worst_logit)
                    smallest_gap = min(smallest_gap, worst_gap)
                    lines.append(
                        f"rank {rank} {r.name}: one token off moves the logits by up to "
                        f"{worst_logit:.3f}, a top gap by up to {worst_gap:.3f}"
                    )
                else:
                    fill(md, [], [], TOKENS, [TOKENS] * WORLD_SIZE, adp_dummy=True)
                    trunk(md, verify_ids([], 1))
        restore()
        records["teeth"].append(dict(rank=rank, logit=smallest, gap=smallest_gap, lines=lines))
    say("teeth: one token off, every probe alone")

    # --- the drift: MTP on along MTP off's own greedy tokens, each probe alone on its rank ---
    with torch.inference_mode():
        # MTP off: probe k of every rank at once, each alone on its rank (a rank past its probe's
        # horizon runs the attention-DP dummy), decoded one token a step, eagerly, from the
        # snapshot. Position n feeds token n; its logits pick token n + 1.
        drift_tokens = {r.rid: [r.tokens[0]] for r in probes}
        decoded = {r.rid: [] for r in probes}
        failures = []
        for k in range(len(probes)):
            r = probes[k]
            horizon = comm.allreduce(r.drift_steps * TOKENS, op=MPI.MAX)
            restore()
            for n in range(horizon):
                if n < r.drift_steps * TOKENS:
                    fill(md, [], [r], 1, [1] * WORLD_SIZE, cached=[len(r.prompt) + n])
                    logits, _ = trunk(md, as_ids([drift_tokens[r.rid][n]]))
                    decoded[r.rid].append(logits[0].clone())
                    drift_tokens[r.rid].append(int(logits[0].argmax()))
                else:
                    fill(md, [], [], 1, [1] * WORLD_SIZE, adp_dummy=True)
                    trunk(md, noise(1))
            first = tuple(drift_tokens[r.rid][:TOKENS])
            if first != r.tokens:
                margins = [round(margin_of(t["logits"]), 4) for t in reference[r.rid]]
                failures.append(
                    f"rank {rank} {r.name}: the decode leg's first tokens {list(first)} are not "
                    f"the reference's {list(r.tokens)} (its top-1 margins {margins})"
                )
        # MTP on: probe k of every rank at once, each alone on its rank in the graph of one (a rank
        # past its probe's horizon runs the attention-DP dummy), every draft accepted and
        # committed.
        restore()
        worst_logit, worst_gap, flips, lines = 0.0, 0.0, [], []
        for k in range(len(probes)):
            r = probes[k]
            steps_k = comm.allreduce(r.drift_steps, op=MPI.MAX)
            r_logit, r_gap = 0.0, 0.0
            for s in range(steps_k):
                n = s * TOKENS
                run = Run(step="drift", group="drift", graph_size=1)
                if s < r.drift_steps:
                    run.kind, run.rows = "verify", [r]
                    ids = as_ids(drift_tokens[r.rid][n : n + TOKENS])
                    logits = run_graph(run, ids, cached=[len(r.prompt) + n], from_snapshot=False)
                    commit_rows(graphs[1][0], [TOKENS])
                    for j in range(TOKENS):
                        ref = decoded[r.rid][n + j]
                        logit, gap = logit_deviation(logits[j], ref)
                        r_logit, r_gap = max(r_logit, logit), max(r_gap, gap)
                        if int(logits[j].argmax()) != int(ref.argmax()):
                            flips.append((r.name, n + j, margin_of(ref)))
                else:
                    run_graph(run, from_snapshot=False)
            worst_logit, worst_gap = max(worst_logit, r_logit), max(worst_gap, r_gap)
            lines.append(
                f"rank {rank} {r.name}: {r.drift_steps * TOKENS} tokens, {r_logit:.4f} / "
                f"{r_gap:.4f}"
            )
        restore()
        del decoded
        if worst_logit > DRIFT_LOGIT_BAND or worst_gap > DRIFT_GAP_BAND:
            failures.append(
                f"rank {rank}: drift {worst_logit:.4f} / {worst_gap:.4f} outside the drift bands"
            )
        wide = [f for f in flips if f[2] > DRIFT_GAP_BAND]
        if wide:
            failures.append(f"rank {rank}: top-1 changes off a near-tie: {wide}")
        records["drift"].append(
            dict(
                rank=rank,
                logit=worst_logit,
                gap=worst_gap,
                positions=sum(r.drift_steps for r in probes) * TOKENS,
                flips=flips,
                lines=lines,
                failures=failures,
            )
        )
    say("drift: every probe alone on its rank, MTP on along MTP off's tokens")

    # --- the MTP layer's draft step through both targets and MTPLayer ---
    with torch.inference_mode():
        rows = list(probes)
        counts = [len(rows)] * WORLD_SIZE
        draft_ids = as_ids([r.tokens[1] for r in rows])
        hidden = torch.cat([hidden0[r.rid] for r in rows])

        def draft_step(target):
            fill(md, [], rows, 1, counts, cached=[len(r.prompt) for r in rows])
            return target(core.w["embed"], counts, draft_ids, hidden, md)

        with validating(M.cublas_mm):
            routed = draft_step(mtp_layer)
            decode = draft_step(core._decode.mtp_forward)
            prefill = draft_step(core._prefill.mtp_forward)
        heads = [mtp_layer.shared_head(x, model.lm_head, md, False) for x in (decode, prefill)]
        drafts = heads[1].argmax(-1).tolist()
        records["draft"].append(
            dict(
                rank=rank,
                finite=bool(torch.isfinite(prefill).all()),
                logit=max(logit_deviation(a, b)[0] for a, b in zip(*heads)),
                routed_equals_decode=equal(routed, decode),
                drafts_equal_t2=sum(d == r.tokens[2] for d, r in zip(drafts, rows)),
                rows=len(rows),
            )
        )
    say("draft: the MTP layer's step through both targets")

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
