# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 4-rank job behind `test_modeling_v2_tp4_blocks.py`: the tp4 target's tensor-parallel
blocks, run by the target's own forward on every rank of one node and measured there against the
plain-torch reference on the checkpoint's own weights.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _tp4_blocks_job.py --out <json> [--checkpoint <dir>] [--quick]

Run plainly, this script is its own launcher: it starts one MPI rank per device under `mpirun`,
with a deadline it enforces by killing the job, because a collective that breaks hangs rather
than raising. The ranks are an MPI world since that is what a tp4 serving engine starts, and the
target's all-reduces run on the TP group's MNNVL workspace built over it.

What runs, on every rank r: the tp4 target built as the engine builds it -- `ModelConfig` off the
checkpoint with rank r's tp4 `Mapping` (tp 4, moe_tp 4, no attention DP), the target class,
`load_weights` over the checkpoint's lazy safetensors slices (each rank reads its own part),
`post_load_weights` (the MNNVL workspace, built collectively) -- with the engine's hybrid cache
manager for that mapping (`_create_kv_cache_manager`) and the engine's TRTLLM attention metadata
(`build_attention_metadata`), filled per step as the executor's runner fills it. Every rank runs
the same steps, rows and tokens. The forward is never edited: its blocks are observed through the
module globals it calls, each call attributed to its layer by its weight operand, and every entry
it calls runs under `validating`, so each entry's `is_valid` holds on tp4's real calls.

The blocks under test (`LAYERS`), each with the stream write and reads around it:

* GDP (0, 6, 42): `in_proj` column-parallel over the rank's 6 of 24 heads, the conv and the
  recurrence on its heads' channels and state, the gated norm, `out_proj` row-parallel, all-reduced;
* attention (all eight): the rank's 4 of 16 query heads over the whole KV head (`k_proj` / `v_proj`
  replicated), `o_proj` row-parallel, all-reduced;
* dense MLP (1, 3, 41, 43): up column-parallel, the soft-clamped relu2 on the rank's quarter, down
  row-parallel, all-reduced;
* latent MoE (5, 6, 24, 39): router, `noaux_tc_op` and `fc1_latent_proj` replicated, every expert
  on the rank's quarter of its intermediate width, the shared expert column / row parallel; the
  routed latent partial and the shared partial all-reduced in one packed call before `fc2_norm`;
* lm_head: whole on every rank, and vocab-parallel up to `modeling._VOCAB_PARALLEL_ROWS` rows
  (every step here): each rank's GEMM over its quarter of the rows is compared against the
  reference over the same rows, and the forward's logits are every rank's slice side by side,
  gathered by the catalog all-gather. Above that the whole GEMM runs and is compared as it is.

The steps (`_Campaign.run`), the shapes a served engine runs: the six short prompts as one
varlen context batch; two ~8K prompts alone, unchunked; a ~8K prompt in 2048-token chunks over its
own cached KV and state; a long prompt's 3008-token chunks beside the short prompts' generation
rows; 54 prompt prefixes in context batches of up to 8192 tokens; then decode at batch 1, 4, 16
and 64 over histories up to ~8K. The GDP reference carries the states of the ten prompt-set
requests (its per-token fp32 loop is the slow part), so GDP's reference comparisons cover those
rows; every other comparison covers every row.

Every all-reduce in this target sums bf16 partials. What each rank records, per step, as one
record per comparison and output (`_Campaign.note`):

* `op`: each op of the rank on the rank's own input to it, against the reference op on the
  rank's slice of the checkpoint weights (sliced here, independently of the target's table);
* `module`: the target's block output, teacher-forced at the streams the block read (`S_in`),
  against the reference block on `S_in` -- GDP also against a reference fed the target's own
  in_proj output (`module@in_proj`), which leaves the path after in_proj alone to compare;
* `twin`: the same tp4 block with its all-reduce replaced by the single full-K (or full-width)
  call on the ranks' gathered inputs -- a single GPU's computation of what the ranks split -- run
  through the same entries, against the reference: the composition of tp4's per-rank ops alone;
* `tp`: the target against its twin, at the all-reduce's output and at every output after it:
  the TP rounding term;
* `colpar`: a column-parallel GEMM's output against the full GEMM's columns (bitwise share);
* `rival_*`: one planted tensor-parallel mistake each, against the reference block;
* flags: `ar_bitwise` -- the all-reduce's output is `allreduce.reference` of the four gathered
  partials, bit for bit; `same_on_every_rank` -- every rank holds the same bits of each
  replicated output; `tp_within_rounding` -- every element of the TP term at a row-parallel
  GEMM's all-reduce lies within its partials' rounding (`_Campaign.within_rounding`); the MoE's
  packing, lm_head reading the final hidden, and the returned logits being its GEMM, or every
  rank's lm_head slice side by side.

The MoE block is also measured with the family helper (`_nemotron4_moe_block`): routing against
HF's router, and the block against the fp32-GEMM and HF-order references, as on tp1.

Rank 0 writes every rank's records to `--out` as JSON. Measured: about 1-2 minutes to load and
5 minutes for the steps, at most 106 GiB per rank. Not a pytest module: the ranks run one
fixed sequence of collectives in lockstep, so every gather below is issued by every rank in the
same order and never under a condition that only one rank's data decides. The launcher must not
import `tensorrt_llm` -- that calls `MPI_Init`, and an MPI-initialized process cannot start
`mpirun` -- so everything that does is imported inside the rank body.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "NEMOTRON4_TP4_BLOCKS_PID_DIR"
DEADLINE_S = 5400
"""Ceiling on the whole job; a wedged collective is killed here rather than hanging the run."""

WORLD_SIZE = 4
"""tp4: tensor parallel over the four GPUs of one node."""

# The engine sizes every nemotron4 run uses (max_seq_len for 8K + 1K, a step of at most 8192
# tokens: tp4's per-rank cells end there).
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
TOKENS_PER_BLOCK = 32
CHUNK = 2048
#: The mixed steps' chunk: a whole number of KV blocks, as the scheduler cuts chunks.
MIXED_CHUNK = 3008
#: The largest decode batch below; also the state slots and attention rows the metadata holds.
MAX_BATCH = 64
#: Generated tokens a request may take here (KV capacity is reserved for them up front).
MAX_NEW = 12
#: KV tokens of each rank's paged pool: every request of the campaign stays resident.
KV_TOKENS = 262_144
KV_FRACTION = 0.1

LAYERS = dict(
    gdp=(0, 6, 42),
    attention=(5, 9, 14, 19, 24, 29, 34, 38),
    mlp=(1, 3, 41, 43),
    moe=(5, 6, 24, 39),
)
"""The blocks measured: the first linear_attention GDP block, the first shortcut GDP mixer and the
last GDP block; every attention block; every dense MLP (43 ends in the readout); the first
attention-side and GDP-side MoE and two deep ones."""
QUICK_LAYERS = dict(gdp=(0, 6), attention=(5,), mlp=(1, 43), moe=(5,))

PHASES = ("context", "chunked", "mixed", "decode_b1", "decode_b4", "decode_b16", "decode_b64")
QUICK_PHASES = ("context", "chunked", "decode_b6")

#: Prompt prefixes that give the batch-64 decode its other histories (54 of them, beside the ten
#: prompts): around the 32-token KV block (31, 32, 33), and odd lengths up to ~2K.
EXTRA_LENGTHS = [31, 32, 33, 64, 65, 100, 127, 129, 255, 257, 300, 511, 513, 700, 1000, 1023, 1025]
EXTRA_LENGTHS += [1100 + 23 * k for k in range(37)]

#: Rivals: one tensor-parallel mistake each, built from what the ranks really computed.
RIVALS = dict(
    gdp=("no_allreduce", "heads_rotated", "bf16_running_sum"),
    attention=("no_allreduce", "heads_rotated", "bf16_running_sum"),
    mlp=("no_allreduce", "slices_rotated", "bf16_running_sum"),
    moe=("no_allreduce", "norm_before_reduce", "bf16_running_sum"),
)

_BF16_ULP = 2.0**-8
_FP32_ULP = 2.0**-23

#: Modules and entries the rank body imports, bound by `_bind`.
X = SimpleNamespace()


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


def spawn_ranks(out: str, checkpoint: str, quick: bool) -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the four devices this run owns"
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    assert len(devices) >= WORLD_SIZE, (
        f"tp4 runs on {WORLD_SIZE} devices; CUDA_VISIBLE_DEVICES names {len(devices)}"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; the ranks are an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="tp4_blocks_")
    env = dict(
        _standalone_env(),
        CUDA_VISIBLE_DEVICES=",".join(devices[:WORLD_SIZE]),
        **{_PID_DIR_ENV: pid_dir},
    )
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(WORLD_SIZE)]
    if os.geteuid() == 0:
        command.insert(1, "--allow-run-as-root")
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    command += ["--out", out, "--checkpoint", checkpoint] + (["--quick"] if quick else [])
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
# Rank side: imports, measurements and the reference pieces this job adds
# ----------------------------------------------------------------------------------------------


def _bind() -> None:
    """Import everything the rank body uses into `X` (each rank, after MPI is up)."""
    here = Path(__file__).resolve().parent
    # Spawned by mpirun rather than collected by pytest, so the tree's `__extra_import_path__`
    # is not in play: the family's helpers and `_validating` are reached the plain way.
    sys.path[:0] = [str(here.parents[1]), str(here.parents[3])]
    import _nemotron4_moe_block as K
    import _nemotron4_prompts as P
    import _nemotron4_reference as R
    import torch
    from _validating import validating

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.softclamp_relu2 import (
        softclamp_relu2,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
        thop_attention,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import allreduce
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_update import (
        causal_conv1d_update,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.conv.causal_conv1d_varlen import (
        causal_conv1d_varlen,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_chunk import gdp_chunk
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_gates import gdp_gates
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_recurrent import (
        gdp_recurrent,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.latent_moe_experts import (
        latent_moe_experts,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
        flashinfer_rmsnorm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.gated_rmsnorm import (
        gated_rmsnorm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.wide_residual import (
        wide_residual,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4 import (  # noqa: E501
        modeling as M,
    )

    X.__dict__.update(
        torch=torch,
        K=K,
        P=P,
        R=R,
        M=M,
        validating=validating,
        allgather=allgather,
        allreduce=allreduce,
        cublas_mm=cublas_mm,
        latent_moe_experts=latent_moe_experts,
        noaux_tc_op=noaux_tc_op,
        flashinfer_rmsnorm=flashinfer_rmsnorm,
        softclamp_relu2=softclamp_relu2,
        wide_residual=wide_residual,
        # Every entry tp4's forward calls, each guarded by its is_valid during the forward.
        entries=(
            wide_residual,
            cublas_mm,
            allgather,
            thop_attention,
            allreduce,
            gdp_gates,
            causal_conv1d_varlen,
            causal_conv1d_update,
            gdp_chunk,
            gdp_recurrent,
            gated_rmsnorm,
            softclamp_relu2,
            noaux_tc_op,
            latent_moe_experts,
            flashinfer_rmsnorm,
        ),
    )


def measure(out, ref, scale=None, ulp: float = _BF16_ULP) -> dict:
    """`out` against `ref`, rows = last dim, scored as `assert_within_ulp` scores it.

    `element_ulp`: the largest |out - ref| over its row's scale -- the row's largest |ref| unless
    `scale` gives one per row (a stream write's operand magnitude, a recurrent state's operand
    scale) -- in ulps of `ulp`; `rms_ulp`: the relative RMS in the same unit; `max_abs`;
    `max_rel`: |out - ref| / |ref| over the elements at least 1/16 of their row's largest;
    `identical`: the bit-identical share; `worst`: the element behind `element_ulp`.
    """
    torch = X.torch
    assert out.shape == ref.shape, (tuple(out.shape), tuple(ref.shape))
    o = out.detach().float().reshape(-1, out.shape[-1])
    r = ref.detach().float().reshape(-1, ref.shape[-1])
    tiny = torch.finfo(torch.float32).tiny
    d = (o - r).abs()
    ref_max = r.abs().amax(-1)
    row_scale = (ref_max if scale is None else scale.detach().float().reshape(-1)).clamp_min(tiny)
    element = d / row_scale[:, None]
    row, col = divmod(int(element.argmax()), o.shape[-1])
    big = r.abs() >= ref_max[:, None] / 16
    rel = torch.where(big, d / r.abs().clamp_min(tiny), torch.zeros_like(d))
    return dict(
        rows=int(o.shape[0]),
        element_ulp=float(element.max()) / ulp,
        rms_ulp=float((o - r).norm() / r.norm().clamp_min(tiny)) / ulp,
        max_abs=float(d.max()),
        max_rel=float(rel.max()),
        identical=float((o == r).float().mean()),
        worst=dict(row=row, col=col, out=float(o[row, col]), ref=float(r[row, col])),
    )


def stream_scale(s_in, s_out):
    """Per token, the largest stream magnitude a write reads or writes: a write that cancels its
    operands is judged against them, not against its (small) result."""
    return X.torch.maximum(s_in.float().abs().amax(-1), s_out.float().abs().amax(-1))


def normed_read(streams, logit, norm, config):
    """A connection's read and the RMSNorm after it, in HF's order (`Nemotron4Layer._read`)."""
    R = X.R
    x = R.stream_read(streams, logit, config.wide_residual_num_streams)
    return R.rms_norm(x, norm, config.rms_norm_eps)


def stream_write(streams, update, write_logit, retention_logit, config):
    return X.R.stream_write(
        streams,
        update,
        write_logit,
        retention_logit,
        config.wide_residual_retention_max_forget,
        config.wide_residual_num_streams,
    )


def bf16_running_sum(parts):
    """The rival reduction: the partials added in rank order, each addition rounded to bf16."""
    acc = parts[0]
    for p in parts[1:]:
        acc = (acc.float() + p.float()).to(acc.dtype)
    return acc


@dataclasses.dataclass
class GdpRefState:
    """The reference GDP mixer's state of one sequence after its last token (24 heads)."""

    conv: object  # [conv_dim, kernel - 1] bf16, the last conv inputs, oldest first
    recurrent: object  # [heads, key_dim, value_dim] fp32, the checkpoint's layout
    scale: object  # [heads, value_dim] fp32, each state column's operand scale


def gdp_recurrence(q, k, v, g, beta, initial_state=None, initial_scale=None):
    """`_nemotron4_reference.gated_delta_rule_recurrent` (the checkpoint's per-token loop, fp32, q
    and k l2-normalized) op for op, plus the operand scale of each state column: the largest
    magnitude the column has held, decayed since by the decays the state went through -- the size
    of the values the recurrence rounded on its way to the column's value. A Householder step can
    overwrite most of what a token's earlier sub-steps wrote, so a column's final value can sit far
    below the values its rounding errors were made on; that scale is what an element error of a
    state is judged against.

    Returns `(out [L, H, V], state [H, K, V], scale [H, V])`.
    """
    torch, R = X.torch, X.R
    dtype = torch.float32
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))
    q = R.l2norm(q)
    k = R.l2norm(k)
    q = q / (q.shape[-1] ** 0.5)
    num_steps, heads, key_dim = k.shape
    if initial_state is None:
        state = torch.zeros(heads, key_dim, v.shape[-1], dtype=dtype, device=v.device)
    else:
        state = initial_state.to(dtype).clone()
    scale = state.abs().amax(dim=-2)
    if initial_scale is not None:
        scale = torch.maximum(scale, initial_scale)
    decay = g.exp()[:, :, None, None]
    out = torch.zeros_like(v)
    for i in range(num_steps):
        state = state * decay[i]
        scale = scale * decay[i, :, :, 0]
        kv_mem = (state * k[i, :, :, None]).sum(dim=-2)
        delta = (v[i] - kv_mem) * beta[i, :, None]
        state = state + k[i, :, :, None] * delta[:, None, :]
        scale = torch.maximum(scale, state.abs().amax(dim=-2))
        out[i] = (state * q[i, :, :, None]).sum(dim=-2)
    return out, state, scale


def gdp_reference_mixer(x, w, config, state: GdpRefState | None, proj=None):
    """`Nemotron4GDPMixer` over one sequence's tokens, continuing from `state` (all 24 heads).

    `_nemotron4_reference.gdp_mixer` op for op, except at the short conv: SiLU of the fp32 tap
    sum, rounded once -- where the conv entries round (the checkpoint's code rounds the conv
    output and then SiLU's; that one rounding is a reference choice, measured on tp1 and not
    folded into any band here). `proj` replaces the reference's own fp32 in_proj with a given
    in_proj output (the target's, gathered from the ranks).

    Returns `(u, h, state, proj)`: the mixer output, the gated norm's output (out_proj's input),
    the `GdpRefState` after the last token and the in_proj output used.
    """
    torch, R = X.torch, X.R
    F = torch.nn.functional
    if proj is None:
        proj = R.linear(x, w["in_proj.weight"])
    num_tokens = proj.shape[0]
    heads, groups = config.gdp_num_heads, config.gdp_num_groups
    head_v, head_k, n_h = config.gdp_head_dim, config.gdp_key_head_dim, config.gdp_num_householder
    value_dim, key_dim, conv_dim = config.gdp_value_dim, config.gdp_key_dim, config.gdp_conv_dim
    gate, mixed, b, a = proj.split([value_dim, conv_dim, n_h * heads, heads], dim=-1)
    taps = w["conv1d.weight"].reshape(conv_dim, -1).float()
    width = taps.shape[-1]
    prior = mixed.new_zeros(conv_dim, width - 1) if state is None else state.conv
    window = torch.cat([prior.t().to(mixed.dtype), mixed], dim=0)
    acc = window[0:num_tokens].float() * taps[:, 0]
    for i in range(1, width):
        acc = acc + window[i : i + num_tokens].float() * taps[:, i]
    mixed = F.silu(acc).to(proj.dtype)
    conv_state = window[num_tokens:].t().contiguous()
    value, key, query = mixed.split([n_h * value_dim, n_h * key_dim, key_dim], dim=-1)
    value = value.reshape(num_tokens, n_h, heads, head_v)
    key = key.reshape(num_tokens, n_h, groups, head_k).repeat_interleave(heads // groups, dim=-2)
    query = query.reshape(num_tokens, groups, head_k).repeat_interleave(heads // groups, dim=-2)
    beta = b.sigmoid().reshape(num_tokens, n_h, heads)
    g = -w["A_log"].float().exp() * F.softplus(a.float() + w["dt_bias"].float())
    expanded_query = query.new_zeros(num_tokens, n_h, heads, head_k)
    expanded_query[:, n_h - 1] = query
    expanded_g = g.new_zeros(num_tokens, n_h, heads)
    expanded_g[:, 0] = g
    out, recurrent, scale = gdp_recurrence(
        expanded_query.flatten(0, 1),
        key.flatten(0, 1),
        value.flatten(0, 1),
        expanded_g.flatten(0, 1),
        beta.flatten(0, 1),
        None if state is None else state.recurrent,
        None if state is None else state.scale,
    )
    out = out.to(proj.dtype).reshape(num_tokens, n_h, heads, head_v)[:, n_h - 1]
    h = R.gated_rms_norm(
        out.reshape(num_tokens, value_dim),
        gate,
        w["norm.weight"],
        value_dim // groups,
        config.gdp_norm_eps,
    )
    return R.linear(h, w["out_proj.weight"]), h, GdpRefState(conv_state, recurrent, scale), proj


# ----------------------------------------------------------------------------------------------
# The target, its cache manager and its attention metadata, built as the engine builds them
# ----------------------------------------------------------------------------------------------


def checkpoint_tensors(path: str) -> dict:
    """Every checkpoint tensor as a lazy safetensors slice, as the engine's lazy loader hands them
    to `load_weights` (indexing one reads only those bytes)."""
    from safetensors import safe_open

    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {
        shard: safe_open(os.path.join(path, shard), framework="pt", device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    return {key: handles[shard].get_slice(key) for key, shard in weight_map.items()}


@dataclasses.dataclass
class Request:
    """One sequence: its prompt, the tokens whose KV and state are written, its greedy output."""

    rid: int
    name: str
    prompt: list
    tracked: bool  # whether the GDP reference carries this request's state
    done: int = 0
    generated: list = dataclasses.field(default_factory=list)

    @property
    def in_context(self) -> bool:
        return self.done < len(self.prompt)


@dataclasses.dataclass
class Row:
    """One row of a step: a request and the tokens it feeds (a prompt chunk, or one token)."""

    request: Request
    tokens: list
    start: int = 0  # first token of the row in the step
    cached: int = 0  # tokens of the request already in KV and state


def run_step(model, md, manager, rows: list) -> object:
    """One forward over `rows` (context rows first), with the per-step fields set as the
    executor's runner sets them; returns each row's last-token fp32 logits."""
    torch = X.torch
    from tensorrt_llm._torch.metadata import KVCacheParams

    num_contexts = sum(r.request.in_context for r in rows)
    assert all(r.request.in_context for r in rows[:num_contexts]), "context rows go first"
    start = 0
    for r in rows:
        r.start, r.cached = start, r.request.done
        start += len(r.tokens)
    md.seq_lens = torch.tensor([len(r.tokens) for r in rows], dtype=torch.int)
    md.beam_width = 1
    md.request_ids = [r.request.rid for r in rows]
    # As the runner: a context row's "prompt length" is its chunk's length (the attention op
    # takes it as that row's query length), a generation row's is the request's prompt length.
    md.prompt_lens = [
        len(r.tokens) if r.request.in_context else len(r.request.prompt) for r in rows
    ]
    md.num_contexts = num_contexts
    md.num_chunked_ctx_requests = 0
    md.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=[r.request.done for r in rows],
        num_extra_kv_tokens=0,
    )
    md.kv_cache_manager = manager
    md.padded_num_tokens = None
    md.prepare()
    ids = torch.tensor([t for r in rows for t in r.tokens], dtype=torch.int, device="cuda")
    logits = model.forward(attn_metadata=md, input_ids=ids)
    for r, best in zip(rows, logits.argmax(-1).tolist()):
        r.request.done += len(r.tokens)
        if not r.request.in_context:
            r.request.generated.append(best)
    return logits


class Observer:
    """Wraps the module globals tp4's forward calls and keeps, per layer under test, what its
    block read, computed and wrote in the last forward (every kept tensor a copy taken at the
    call: `wide_residual` writes the streams in place, and the decode conv overwrites in_proj's
    output).

    Attribution: a GEMM by the target parameter it multiplies (`cublas_mm`'s `mat_b`, a `.t()`
    view of it; `invariant_mm`'s `weight`, the parameter as stored, which the decode target
    passes for the GDP in_proj, the router and `fc1_latent_proj`); an all-reduce by the tagged
    GEMM that produced its input -- the last GEMM call before it, so an untagged GEMM clears the
    tag; `noaux_tc_op` and `latent_moe_experts` by the router and `fc1_latent_proj` GEMMs that
    feed them; `flashinfer_rmsnorm` by its weight; a `wide_residual` call by the retention it
    writes with and the read factors it reads with; an attention call (`thop_attention`, or
    `invariant_gen_attention` in decode) by its KV layer.
    """

    def __init__(self, module, model, manager, layers: dict, config):
        self.module = module
        self.orig = {
            name: getattr(module, name)
            for name in (
                "wide_residual",
                "cublas_mm",
                "invariant_mm",
                "thop_attention",
                "invariant_gen_attention",
                "allreduce",
                "noaux_tc_op",
                "latent_moe_experts",
                "flashinfer_rmsnorm",
            )
        }
        w = model.model.w
        kinds = config.layer_types
        self.gemm, self.writes, self.reads, self.norms = {}, {}, {}, {}
        observed = sorted({i for group in layers.values() for i in group})
        for i in layers["gdp"]:
            self.gemm[w[f"l{i}_gdp_in"].data_ptr()] = (i, "in_proj")
            self.gemm[w[f"l{i}_gdp_out"].data_ptr()] = (i, "out_proj")
        for i in layers["attention"]:
            self.gemm[w[f"l{i}_qkv"].data_ptr()] = (i, "qkv")
            self.gemm[w[f"l{i}_o"].data_ptr()] = (i, "o")
        for i in layers["mlp"]:
            self.gemm[w[f"l{i}_mlp_up"].data_ptr()] = (i, "mlp_up")
            self.gemm[w[f"l{i}_mlp_down"].data_ptr()] = (i, "mlp_down")
        for i in layers["moe"]:
            for role, name in (
                ("router", "router"),
                ("fc1_latent", "fc1_latent"),
                ("shared_up", "shared_up"),
                ("shared_down", "shared_down"),
                ("fc2_latent", "fc2_latent"),
            ):
                self.gemm[w[f"l{i}_{name}"].data_ptr()] = (i, role)
            self.norms[w[f"l{i}_fc2_norm"].data_ptr()] = i
        # lm_head: the whole weight, and this rank's quarter of its rows, a view of it.
        lm_head = model.lm_head.weight
        quarter = lm_head.shape[0] // model.model.tp_size
        for ptr in (lm_head.data_ptr(), lm_head[model.model.tp_rank * quarter :].data_ptr()):
            self.gemm[ptr] = ("lm_head", "logits")
        for i in observed:
            if kinds[i] in ("linear_attention", "mlp"):
                self.writes[w[f"l{i}_res_retain"].data_ptr()] = (i, "plain_call")
                self.reads[w[f"l{i}_res_read"].data_ptr()] = (i, "x")
            else:
                self.writes[w[f"l{i}_mix_retain"].data_ptr()] = (i, "mixer_call")
                self.writes[w[f"l{i}_moe_retain"].data_ptr()] = (i, "moe_call")
                self.reads[w[f"l{i}_mix_read"].data_ptr()] = (i, "x")
                self.reads[w[f"l{i}_shortcut_read"].data_ptr()] = (i, "routed")
                self.reads[w[f"l{i}_moe_read"].data_ptr()] = (i, "shared")
        self.layer_of_kv = {manager.layer_offsets[i]: i for i in layers["attention"]}
        self.record = defaultdict(dict)
        self.last = None

    def install(self, monkeypatch) -> None:
        for name in self.orig:
            monkeypatch.setattr(self.module, name, getattr(self, name))

    def take(self) -> dict:
        record, self.record = self.record, defaultdict(dict)
        self.last = None
        return record

    def wide_residual(self, streams, update, retention, write, f1, g1, eps, f2=None, g2=None):
        owner = None if retention is None else self.writes.get(retention.data_ptr())
        before = streams.clone() if owner is not None else None
        out = self.orig["wide_residual"](streams, update, retention, write, f1, g1, eps, f2, g2)
        if owner is not None:
            i, which = owner
            self.record[i][which] = dict(
                s_in=before,
                update=update.clone(),
                s_out=streams.clone(),
                reads=[out[1].clone()] + ([] if f2 is None else [out[2].clone()]),
                args=(retention, write, f1, g1, eps, f2, g2),
            )
        for factors, normed in ((f1, out[1]), (f2, out[2])):
            if factors is not None and factors.data_ptr() in self.reads:
                i, role = self.reads[factors.data_ptr()]
                self.record[i][role] = normed.clone()
        return out

    def _gemm(self, a, weight, out) -> None:
        tag = self.gemm.get(weight.data_ptr())
        self.last = tag
        if tag is not None:
            i, role = tag
            self.record[i][role] = dict(a=a.clone(), out=out.clone())

    def cublas_mm(self, mat_a, mat_b, *args, **kwargs):
        out = self.orig["cublas_mm"](mat_a, mat_b, *args, **kwargs)
        self._gemm(mat_a, mat_b, out)
        return out

    def invariant_mm(self, x, weight, *args, **kwargs):
        out = self.orig["invariant_mm"](x, weight, *args, **kwargs)
        self._gemm(x, weight, out)
        return out

    def allreduce(self, input, comm_buffer, buffer_flags):
        out = self.orig["allreduce"](input, comm_buffer, buffer_flags)
        if self.last is not None:
            i, role = self.last
            self.record[i][f"ar_{role}"] = dict(input=input.clone(), out=out.clone())
        self.last = None
        return out

    def _attention(self, name, kwargs):
        out = self.orig[name](**kwargs)
        i = self.layer_of_kv.get(kwargs["local_layer_idx"])
        if i is not None:
            self.record[i]["attention"] = out.clone()
        return out

    def thop_attention(self, **kwargs):
        return self._attention("thop_attention", kwargs)

    def invariant_gen_attention(self, **kwargs):
        return self._attention("invariant_gen_attention", kwargs)

    def noaux_tc_op(self, router_logits, bias, *args):
        weights, ids = self.orig["noaux_tc_op"](router_logits, bias, *args)
        if self.last is not None and self.last[1] == "router":
            self.record[self.last[0]]["noaux"] = dict(
                logits=router_logits.clone(), bias=bias, weights=weights.clone(), ids=ids.clone()
            )
        return weights, ids

    def latent_moe_experts(self, x, topk_ids, topk_weights, up_proj, down_proj, *args):
        out = self.orig["latent_moe_experts"](x, topk_ids, topk_weights, up_proj, down_proj, *args)
        if self.last is not None and self.last[1] == "fc1_latent":
            self.record[self.last[0]]["experts"] = dict(out=out.clone(), args=args)
        return out

    def flashinfer_rmsnorm(self, x, weight, eps):
        out = self.orig["flashinfer_rmsnorm"](x, weight, eps)
        i = self.norms.get(weight.data_ptr())
        if i is not None:
            self.record[i]["fc2_norm"] = dict(x=x.clone(), out=out.clone())
        return out


# ----------------------------------------------------------------------------------------------
# The campaign: steps, and per step every block under test on every rank
# ----------------------------------------------------------------------------------------------


class _Campaign:
    def __init__(self, comm, checkpoint: str, quick: bool):
        torch, R = X.torch, X.R
        from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
        from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.pyexecutor._util import (
            _create_kv_cache_manager,
            get_kv_cache_manager_cls,
        )
        from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
        from tensorrt_llm._torch.pyexecutor.model_loader import (
            validate_and_set_mamba_ssm_cache_dtype,
        )
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig
        from tensorrt_llm.mapping import Mapping

        self.comm = comm
        self.rank = comm.Get_rank()
        self.started = time.monotonic()
        self.quick = quick
        self.layers = QUICK_LAYERS if quick else LAYERS
        self.ckpt = R.Checkpoint(checkpoint)
        self.cfg = self.ckpt.config
        cfg = self.cfg

        mapping = Mapping(
            world_size=WORLD_SIZE,
            rank=self.rank,
            gpus_per_node=WORLD_SIZE,
            tp_size=WORLD_SIZE,
            moe_tp_size=WORLD_SIZE,
            moe_ep_size=1,
        )
        self.group = list(mapping.tp_group)
        # One rank at a time: the remote-code config is copied into the shared modules cache.
        config = None
        for turn in range(WORLD_SIZE):
            if turn == self.rank:
                # max_num_tokens sizes the target's all-reduce workspace (its rows per call).
                config = ModelConfig.from_pretrained(
                    checkpoint,
                    trust_remote_code=True,
                    mapping=mapping,
                    max_num_tokens=MAX_NUM_TOKENS,
                )
            comm.Barrier()
        validate_and_set_mamba_ssm_cache_dtype(config, "auto")
        with torch.device("cuda"):
            model = X.M.ModelingV2Nemotron4NanoPreviewSm103Tp4(config)
        model.load_weights(checkpoint_tensors(checkpoint))
        model.post_load_weights()
        torch.cuda.synchronize()
        self.model = model.eval()
        self.say(f"target loaded on every rank; free {self.free_gib():.1f} GiB")

        kv_cache_config = KvCacheConfig(
            max_tokens=KV_TOKENS, enable_block_reuse=False, free_gpu_memory_fraction=KV_FRACTION
        )
        manager_cls = get_kv_cache_manager_cls(config, kv_cache_config)
        self.manager = _create_kv_cache_manager(
            None,
            manager_cls,
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
            max_cuda_graph_batch_size=MAX_BATCH,
        )
        self.say(f"{type(self.manager).__name__}; free {self.free_gib():.1f} GiB")

        def metadata(chunked_prefill: bool):
            md = build_attention_metadata(
                config,
                max_batch_size=MAX_BATCH,
                max_num_tokens=MAX_NUM_TOKENS,
                max_beam_width=1,
                attention_backend=get_attention_backend("TRTLLM"),
                attention_runtime_features=AttentionRuntimeFeatures(
                    chunked_prefill=chunked_prefill,
                    cache_reuse=False,
                    has_speculative_draft_tokens=False,
                    chunk_size=MAX_NUM_TOKENS,
                ),
                mapping=mapping,
                cache_indirection=None,
                kv_cache_manager=self.manager,
            )
            md.mamba_metadata_cls = X.M.ModelingV2Nemotron4NanoPreviewSm103Tp4.mamba_metadata_cls
            return md

        self.md_unchunked = metadata(False)
        self.md_chunked = metadata(True)
        self.observer = Observer(X.M, self.model, self.manager, self.layers, cfg)

        # The reference's weights: the checkpoint's tensors, read independently of the target's
        # relayout, only those the blocks under test read.
        self.w = {}
        for i in sorted({i for group in self.layers.values() for i in group}):
            self.w[i] = self.layer_weights(i)
        self.lm_head_w = self.ckpt.tensor(R.LM_HEAD_KEY)
        self.say(f"reference weights loaded; free {self.free_gib():.1f} GiB")

        # Rank r's part of every split weight, sliced here from the checkpoint's own layout.
        r, tp = self.rank, WORLD_SIZE
        q_width = cfg.num_attention_heads * cfg.head_dim // tp
        self.q_cols = slice(r * q_width, (r + 1) * q_width)
        mlp_width = cfg.intermediate_size // tp
        self.mlp_cols = slice(r * mlp_width, (r + 1) * mlp_width)
        moe_width = cfg.moe_intermediate_size // tp
        self.moe_cols = slice(r * moe_width, (r + 1) * moe_width)
        shared_width = cfg.moe_shared_expert_intermediate_size // tp
        self.shared_cols = slice(r * shared_width, (r + 1) * shared_width)
        gdp_width = cfg.gdp_value_dim // tp
        self.gdp_cols = slice(r * gdp_width, (r + 1) * gdp_width)
        gdp_heads = cfg.gdp_num_heads // tp
        self.gdp_heads = slice(r * gdp_heads, (r + 1) * gdp_heads)
        self.gdp_in_rows = [self.segment_index(self.gdp_in_widths(), rr) for rr in range(tp)]
        self.gdp_conv_rows = [self.segment_index(self.gdp_conv_widths(), rr) for rr in range(tp)]
        self.moe_rank_experts = {
            i: (
                self.w[i]["moe.experts.up_proj"][:, self.moe_cols].contiguous(),
                self.w[i]["moe.experts.down_proj"][:, :, self.moe_cols].contiguous(),
            )
            for i in self.layers["moe"]
        }

        self.records: list[dict] = []
        self.steps: list[dict] = []
        # Per (layer, request id): K and V of every token so far, as each side computed them.
        self.history = {"target": {}, "ref": {}}
        # Per (layer, request id): the GDP reference's state, fed its own in_proj ("own") or the
        # target's gathered in_proj output ("ip").
        self.gdp_state = {"own": {}, "ip": {}}
        self.next_rid = 1
        self.step_index = 0

    # --- utilities --------------------------------------------------------------------------------

    def say(self, message: str) -> None:
        if self.rank == 0:
            print(f"[job {time.monotonic() - self.started:7.1f}s] {message}", flush=True)

    @staticmethod
    def free_gib() -> float:
        return X.torch.cuda.mem_get_info()[0] / 2**30

    def layer_weights(self, i: int) -> dict:
        """The checkpoint tensors block `i`'s reference reads, keyed by their name in the layer,
        plus `next.*`: the reads after the block's last write (the next block's, or the readout)."""
        ckpt, cfg, layers = self.ckpt, self.cfg, self.layers
        kind = cfg.layer_types[i]
        names = ["input_layernorm.weight"]
        if kind in ("linear_attention", "mlp"):
            names += [f"residual_connection.{n}_logit" for n in ("read", "write", "retention")]
        else:
            names += ["shortcut_read.logit", "shortcut_input_layernorm.weight"]
            names += ["pre_moe_layernorm.weight"]
            for conn in ("mixer_residual_connection", "moe_residual_connection"):
                names += [f"{conn}.{n}_logit" for n in ("read", "write", "retention")]
        if i in layers["gdp"]:
            names += [
                f"mixer.{n}"
                for n in (
                    "in_proj.weight",
                    "conv1d.weight",
                    "A_log",
                    "dt_bias",
                    "norm.weight",
                    "out_proj.weight",
                )
            ]
        if i in layers["attention"]:
            names += [f"self_attn.{p}_proj.weight" for p in "qkvo"]
        if i in layers["mlp"]:
            names += ["mlp.up_proj.weight", "mlp.down_proj.weight"]
        prefix = f"{X.R.TEXT_PREFIX}layers.{i}."
        w = {n: ckpt.tensor(prefix + n) for n in names}
        if i in layers["moe"]:
            w.update({f"moe.{k}": v for k, v in ckpt.tensors(prefix + "moe.").items()})
        if i in layers["attention"]:
            w["qkv"] = X.torch.cat(
                [
                    w["self_attn.q_proj.weight"],
                    w["self_attn.k_proj.weight"],
                    w["self_attn.v_proj.weight"],
                ]
            )
        w["next"] = self.next_reads(i + 1)
        return w

    def next_reads(self, j: int) -> list:
        """`[(read logit, norm weight), ...]` of what follows block j - 1's last write: block j's
        reads (a shortcut block's routed input first, then its mixer input), or the readout."""
        ckpt, kinds, T = self.ckpt, self.cfg.layer_types, X.R.TEXT_PREFIX
        if j == len(kinds):
            return [(ckpt.tensor(T + "residual_readout.logit"), ckpt.tensor(T + "norm.weight"))]
        n = lambda name: ckpt.tensor(f"{T}layers.{j}.{name}")  # noqa: E731
        if kinds[j] in ("shortcut_moe_attn", "shortcut_moe_gdp"):
            return [
                (n("shortcut_read.logit"), n("shortcut_input_layernorm.weight")),
                (n("mixer_residual_connection.read_logit"), n("input_layernorm.weight")),
            ]
        return [(n("residual_connection.read_logit"), n("input_layernorm.weight"))]

    def gdp_in_widths(self) -> list[int]:
        """`in_proj`'s packed rows: `[z | V0..Vm | K0..Km | Q | b0..bm | a]`."""
        c = self.cfg
        n_h = c.gdp_num_householder
        return (
            [c.gdp_value_dim] * (1 + n_h)
            + [c.gdp_key_dim] * (n_h + 1)
            + [c.gdp_num_heads] * (n_h + 1)
        )

    def gdp_conv_widths(self) -> list[int]:
        """The conv's channels: `[V0..Vm | K0..Km | Q]`."""
        c = self.cfg
        n_h = c.gdp_num_householder
        return [c.gdp_value_dim] * n_h + [c.gdp_key_dim] * (n_h + 1)

    @staticmethod
    def segment_index(widths: list[int], rank: int):
        """Rank `rank`'s block of each packed segment, side by side in segment order: the indices
        into the full packed dimension of the rank's local one."""
        torch = X.torch
        parts, start = [], 0
        for width in widths:
            n = width // WORLD_SIZE
            parts.append(torch.arange(start + rank * n, start + (rank + 1) * n))
            start += width
        return torch.cat(parts).to("cuda")

    def gather_rows(self, t):
        """`[n, w]` from every rank, `[WORLD_SIZE, n, w]` in rank order (catalog all-gather)."""
        t = t.contiguous()
        return X.allgather(t, None, self.group).view(WORLD_SIZE, *t.shape)

    def gather_cols(self, t):
        """`[n, w]` from every rank side by side: `[n, WORLD_SIZE * w]`, rank r's at `[r w, (r+1) w)`."""
        g = self.gather_rows(t)
        return g.permute(1, 0, *range(2, g.dim())).reshape(t.shape[0], -1)

    def same_on_every_rank(self, t) -> bool:
        """Whether every rank holds the same bits of `t` (a 64-bit checksum, compared over MPI)."""
        torch = X.torch
        view = {2: torch.int16, 4: torch.int32}[t.element_size()]
        bits = t.contiguous().view(view).reshape(-1).to(torch.int64)
        weights = torch.arange(bits.numel(), device=bits.device, dtype=torch.int64)
        value = int((bits * (weights * 2654435761 + 97)).sum())
        return len(set(self.comm.allgather(value))) == 1

    def note(self, phase, kind, layer, comparison, output, metrics) -> None:
        self.records.append(
            dict(
                rank=self.rank,
                step=self.step_index,
                phase=phase,
                kind=kind,
                layer=layer,
                comparison=comparison,
                output=output,
                **metrics,
            )
        )

    def flag(self, phase, kind, layer, check, ok, **extra) -> None:
        self.note(phase, kind, layer, "flag", check, dict(ok=bool(ok), **extra))

    def ar_checks(self, phase, kind, layer, ar: dict, site: str) -> list:
        """The all-reduce at `site`: its output against `allreduce.reference` of the four gathered
        partials, bit for bit; every rank's output bits; and how far the partials cancel (the RMS
        of the partials over the RMS of their sum, which sets the size of the TP rounding term).
        Returns the gathered partials."""
        torch = X.torch
        parts = list(self.gather_rows(ar["input"]))
        reference = X.allreduce.reference(parts)
        total = reference.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
        ratio = float(torch.stack([p.float().norm() for p in parts]).norm() / total)
        self.flag(phase, kind, layer, f"ar_bitwise:{site}", torch.equal(ar["out"], reference))
        self.flag(
            phase,
            kind,
            layer,
            f"same_on_every_rank:{site}",
            self.same_on_every_rank(ar["out"]),
            partials_over_sum=ratio,
        )
        return parts

    def within_rounding(self, phase, kind, layer, site: str, parts, out, twin) -> None:
        """The TP term at a row-parallel GEMM's all-reduce, element by element, against the
        rounding the arithmetic allows.

        Rank r's GEMM rounds its K-slice's fp32 sum P_r to the bf16 partial p_r; the all-reduce
        sums the p_r in fp32 and rounds once (`out`); one GPU rounds the full-K fp32 sum once
        (`twin`). So `out - twin` is the four partials' rounding errors plus the two final
        roundings: each at most half a bf16 spacing of its value (a value in [2^e, 2^(e+1)) has
        spacing 2^(e-7)), plus fp32 accumulation-order differences, far below a bf16 spacing
        (allowed for here as 2^-16 of the magnitudes summed). An element beyond that sum is not a
        rounding: a wrong partial, slice or reduction. Records the count beyond and the largest
        share of its bound any element uses.
        """
        torch = X.torch

        def half_spacing(t):
            a = t.float().abs()
            e = torch.floor(torch.log2(a.clamp_min(2.0**-126)))
            return torch.where(a > 0, torch.exp2(e - 8), torch.zeros_like(a))

        magnitude = sum(p.float().abs() for p in parts) + twin.float().abs()
        bound = sum(half_spacing(p) for p in parts) + half_spacing(out) + half_spacing(twin)
        bound = bound + 2.0**-16 * magnitude
        diff = (out.float() - twin.float()).abs()
        use = diff / bound.clamp_min(torch.finfo(torch.float32).tiny)
        beyond = int((diff > bound).sum())
        self.flag(
            phase,
            kind,
            layer,
            f"tp_within_rounding:{site}",
            beyond == 0,
            beyond=beyond,
            numel=diff.numel(),
            max_share_of_bound=float(use.max()),
        )

    def replay_write(self, call: dict, update):
        """The block's write call again, on a copy of the streams it wrote and with another
        update: `(streams, [reads])` through the same `wide_residual` entry and coefficients."""
        retention, write, f1, g1, eps, f2, g2 = call["args"]
        streams = call["s_in"].clone()
        out = X.wide_residual(streams, update, retention, write, f1, g1, eps, f2, g2)
        return streams, [out[1]] + ([] if f2 is None else [out[2]])

    def segments(self, rows):
        for r in rows:
            yield r, slice(r.start, r.start + len(r.tokens))

    # --- requests and steps -----------------------------------------------------------------------

    def request(self, name: str, prompt, tracked: bool = True) -> Request:
        req = Request(self.next_rid, name, list(prompt), tracked)
        self.next_rid += 1
        added = self.manager.add_dummy_requests([req.rid], token_nums=[len(prompt) + MAX_NEW])
        assert added, f"the cache manager has no room for {name} ({len(prompt)} tokens)"
        return req

    def step(self, md, rows: list, phase: str, rivals: bool = False) -> None:
        torch = X.torch
        self.observer.take()
        self.step_index += 1
        t0 = time.monotonic()
        with X.validating(*X.entries):
            logits = run_step(self.model, md, self.manager, rows)
        torch.cuda.synchronize()
        t1 = time.monotonic()
        record = self.observer.take()
        slots = md.mamba_metadata.state_slots[: len(rows)].tolist()
        info = dict(
            step=self.step_index,
            phase=phase,
            rows=len(rows),
            contexts=sum(r.cached < len(r.request.prompt) for r in rows),
            tokens=sum(len(r.tokens) for r in rows),
            max_history=max(r.cached for r in rows),
            rivals=rivals,
        )
        for i in self.layers["gdp"]:
            self.gdp(i, record[i], rows, slots, phase, rivals)
        for i in self.layers["attention"]:
            self.attention(i, record[i], rows, phase, rivals)
        for i in self.layers["mlp"]:
            self.mlp(i, record[i], rows, phase, rivals)
        for i in self.layers["moe"]:
            self.moe(i, record[i], rows, phase, rivals)
        self.lm_head(record, logits, rows, phase)
        del record
        torch.cuda.synchronize()
        info.update(forward_s=round(t1 - t0, 3), compare_s=round(time.monotonic() - t1, 3))
        self.steps.append(info)
        torch.cuda.empty_cache()
        self.say(f"step {info}; free {self.free_gib():.1f} GiB")

    # --- GDP ---------------------------------------------------------------------------------------

    def gdp(self, i, rec, rows, slots, phase, rivals) -> None:
        torch, cfg, w, r = X.torch, self.cfg, self.w[i], self.rank
        plain = cfg.layer_types[i] == "linear_attention"
        call = rec["plain_call" if plain else "mixer_call"]
        missing = {"x", "in_proj", "out_proj", "ar_out_proj"} - rec.keys()
        assert not missing, f"layer {i}: the observer saw no {sorted(missing)}"
        mixer = {k[len("mixer.") :]: v for k, v in w.items() if k.startswith("mixer.")}
        conn = "residual_connection" if plain else "mixer_residual_connection"
        read_logit, norm = w[f"{conn}.read_logit"], w["input_layernorm.weight"]
        write_logit, retention_logit = w[f"{conn}.write_logit"], w[f"{conn}.retention_logit"]
        if plain:
            next_reads = w["next"]
        else:
            next_reads = [(w["moe_residual_connection.read_logit"], w["pre_moe_layernorm.weight"])]
        x, proj = rec["x"], rec["in_proj"]["out"]
        y, partial = rec["out_proj"]["a"], rec["out_proj"]["out"]
        s_in, u, s_out, reads = call["s_in"], call["update"], call["s_out"], call["reads"]
        note = lambda c, o, out, ref, scale=None, ulp=_BF16_ULP: self.note(  # noqa: E731
            phase, "gdp", i, c, o, measure(out, ref, scale, ulp)
        )

        # This rank's ops, on its own input to each.
        note("op", "read", x, normed_read(s_in, read_logit, norm, cfg))
        note("op", "in_proj", proj, X.R.linear(x, mixer["in_proj.weight"][self.gdp_in_rows[r]]))
        full_in = X.cublas_mm(x, mixer["in_proj.weight"].t())
        note("colpar", "in_proj", proj, full_in[:, self.gdp_in_rows[r]])
        note("op", "out_proj", partial, X.R.linear(y, mixer["out_proj.weight"][:, self.gdp_cols]))
        note(
            "op",
            "write",
            s_out,
            stream_write(s_in, u, write_logit, retention_logit, cfg),
            stream_scale(s_in, s_out),
        )
        for k, (logit, nw) in enumerate(next_reads):
            note("op", f"next_read{k}", reads[k], normed_read(s_out, logit, nw, cfg))

        # The all-reduce, and the twin: out_proj as one full-K GEMM on the ranks' gathered
        # gated-norm outputs, then the same write.
        parts = self.ar_checks(phase, "gdp", i, rec["ar_out_proj"], "out_proj")
        for k, read in enumerate(reads):
            self.flag(
                phase, "gdp", i, f"same_on_every_rank:next_read{k}", self.same_on_every_rank(read)
            )
        y_full = self.gather_cols(y)
        u_twin = X.cublas_mm(y_full, mixer["out_proj.weight"].t())
        self.within_rounding(phase, "gdp", i, "out_proj", parts, u, u_twin)
        s_twin, reads_twin = self.replay_write(call, u_twin)
        note("tp", "update", u, u_twin)
        note("tp", "write", s_out, s_twin, stream_scale(s_in, s_twin))
        for k in range(len(reads)):
            note("tp", f"next_read{k}", reads[k], reads_twin[k])

        # The target's in_proj output on every rank, reassembled into the checkpoint's packed
        # layout, for the reference that is fed it.
        gathered = self.gather_cols(proj)
        local = proj.shape[1]
        proj_full = torch.empty(proj.shape[0], local * WORLD_SIZE, dtype=proj.dtype, device="cuda")
        for rr in range(WORLD_SIZE):
            proj_full[:, self.gdp_in_rows[rr]] = gathered[:, rr * local : (rr + 1) * local]

        # The reference mixers, carrying their own states; only for tracked requests.
        tracked = [
            (k, row, seg) for k, (row, seg) in enumerate(self.segments(rows)) if row.request.tracked
        ]
        if not tracked:
            return
        refs = {"own": [], "ip": []}
        for _, row, seg in tracked:
            key = (i, row.request.rid)
            for side, given in (("own", None), ("ip", proj_full[seg])):
                state = self.gdp_state[side].get(key)
                assert (state is None) == (row.cached == 0), (key, row.cached, side)
                result = gdp_reference_mixer(x[seg], mixer, cfg, state, proj=given)
                self.gdp_state[side][key] = result[2]
                refs[side].append(result)
        tok = torch.cat([torch.arange(seg.start, seg.stop) for _, _, seg in tracked]).to("cuda")
        for side, comparison in (("ip", "module@in_proj"), ("own", "module")):
            u_ref = torch.cat([res[0] for res in refs[side]])
            h_ref = torch.cat([res[1] for res in refs[side]])
            s_ref = stream_write(s_in[tok], u_ref, write_logit, retention_logit, cfg)
            note(comparison, "h", y[tok], h_ref[:, self.gdp_cols])
            note(comparison, "update", u[tok], u_ref)
            note(comparison, "write", s_out[tok], s_ref, stream_scale(s_in[tok], s_ref))
            for k, (logit, nw) in enumerate(next_reads):
                note(comparison, f"next_read{k}", reads[k][tok], normed_read(s_ref, logit, nw, cfg))
            twin = "twin@in_proj" if side == "ip" else "twin"
            note(twin, "update", u_twin[tok], u_ref)
            note(twin, "write", s_twin[tok], s_ref, stream_scale(s_in[tok], s_ref))
            for k, (logit, nw) in enumerate(next_reads):
                note(twin, f"next_read{k}", reads_twin[k][tok], normed_read(s_ref, logit, nw, cfg))
            if rivals and side == "ip":
                planted = dict(
                    no_allreduce=partial,
                    heads_rotated=X.cublas_mm(
                        torch.roll(y_full, y.shape[1], dims=1), mixer["out_proj.weight"].t()
                    ),
                    bf16_running_sum=bf16_running_sum(parts),
                )
                for name, u_rival in planted.items():
                    s_rival = stream_write(
                        s_in[tok], u_rival[tok], write_logit, retention_logit, cfg
                    )
                    note(f"rival_{name}", "update", u_rival[tok], u_ref)
                    note(f"rival_{name}", "write", s_rival, s_ref, stream_scale(s_in[tok], s_ref))
                    self.flag(
                        phase,
                        "gdp",
                        i,
                        f"rival_{name}:ar_bitwise",
                        torch.equal(u_rival, X.allreduce.reference(parts)),
                    )

        # The state each tracked request left in this rank's pools: its heads' recurrent state
        # ([6, V, K]) and its channels' conv state, against the references' (all 24 heads).
        ssm_pool = self.manager.get_ssm_states(i)
        conv_pool = self.manager.get_conv_states(i)
        heads = self.gdp_heads
        conv_rows = self.gdp_conv_rows[r]
        for side, comparison in (("ip", "module@in_proj"), ("own", "module")):
            states, refs_, scales, convs, conv_refs = [], [], [], [], []
            for k, row, _ in tracked:
                slot = slots[k]
                ref = self.gdp_state[side][(i, row.request.rid)]
                states.append(ssm_pool[slot].float())
                refs_.append(ref.recurrent[heads].transpose(-1, -2))
                scales.append(ref.scale[heads])
                convs.append(conv_pool[slot])
                conv_refs.append(ref.conv[conv_rows])
            state, ref_state = torch.stack(states), torch.stack(refs_)
            op_scale = torch.maximum(
                ref_state.abs().amax(-1), torch.stack(scales)
            )  # per (row, head, value) column of K keys
            note(comparison, "state", state, ref_state, op_scale)
            note(comparison, "conv_state", torch.stack(convs), torch.stack(conv_refs))

    # --- attention ---------------------------------------------------------------------------------

    def attention(self, i, rec, rows, phase, rivals) -> None:
        torch, R, cfg, w = X.torch, X.R, self.cfg, self.w[i]
        missing = {"x", "qkv", "attention", "o", "ar_o", "mixer_call"} - rec.keys()
        assert not missing, f"layer {i}: the observer saw no {sorted(missing)}"
        call = rec["mixer_call"]
        s_in, u, s_mid, read = call["s_in"], call["update"], call["s_out"], call["reads"][0]
        x, qkv, attn = rec["x"], rec["qkv"]["out"], rec["attention"]
        o_in, partial = rec["o"]["a"], rec["o"]["out"]
        conn = "mixer_residual_connection"
        read_logit, norm = w[f"{conn}.read_logit"], w["input_layernorm.weight"]
        write_logit, retention_logit = w[f"{conn}.write_logit"], w[f"{conn}.retention_logit"]
        shared_logit, pre_moe = (
            w["moe_residual_connection.read_logit"],
            w["pre_moe_layernorm.weight"],
        )
        o_w = w["self_attn.o_proj.weight"]
        heads, d = cfg.num_attention_heads, cfg.head_dim
        local_heads, kv = heads // WORLD_SIZE, cfg.num_key_value_heads
        scale = d**-0.5
        n = x.shape[0]
        note = lambda c, o, out, ref, sc=None: self.note(  # noqa: E731
            phase, "attention", i, c, o, measure(out, ref, sc)
        )

        # This rank's ops, on its own input to each: its query heads' rows of q_proj and the
        # whole k_proj / v_proj; attention on its own q over the K / V it wrote; its o_proj columns.
        qkv_rank = torch.cat(
            [
                w["self_attn.q_proj.weight"][self.q_cols],
                w["self_attn.k_proj.weight"],
                w["self_attn.v_proj.weight"],
            ]
        )
        note("op", "read", x, normed_read(s_in, read_logit, norm, cfg))
        note("op", "qkv", qkv, R.linear(x, qkv_rank))
        full_qkv = X.cublas_mm(x, w["qkv"].t())
        q_width = heads * d
        cols = torch.cat(
            [
                torch.arange(self.q_cols.start, self.q_cols.stop),
                torch.arange(q_width, q_width + 2 * kv * d),
            ]
        ).to("cuda")
        note("colpar", "qkv", qkv, full_qkv[:, cols])
        q_t, k_t, v_t = qkv.split([local_heads * d, kv * d, kv * d], dim=-1)
        q_t = q_t.reshape(n, local_heads, d)
        k_t, v_t = k_t.reshape(n, kv, d), v_t.reshape(n, kv, d)

        x_ref = normed_read(s_in, read_logit, norm, cfg)
        q_r, k_r, v_r = R.linear(x_ref, w["qkv"]).split([q_width, kv * d, kv * d], dim=-1)
        q_r = q_r.reshape(n, heads, d)
        k_r, v_r = k_r.reshape(n, kv, d), v_r.reshape(n, kv, d)
        o_iso = torch.empty(n, local_heads, d, dtype=attn.dtype, device="cuda")
        o_ref = torch.empty(n, heads, d, dtype=attn.dtype, device="cuda")
        for row, seg in self.segments(rows):
            key = (i, row.request.rid)
            for side, k_new, v_new in (("target", k_t, v_t), ("ref", k_r, v_r)):
                past = self.history[side].get(key)
                k_all = k_new[seg] if past is None else torch.cat([past[0], k_new[seg]])
                v_all = v_new[seg] if past is None else torch.cat([past[1], v_new[seg]])
                assert k_all.shape[0] == row.cached + len(row.tokens), (
                    key,
                    k_all.shape,
                    row.cached,
                )
                self.history[side][key] = (k_all, v_all)
            k_all, v_all = self.history["target"][key]
            o_iso[seg] = R.causal_attention(q_t[seg], k_all, v_all, scale)
            k_all, v_all = self.history["ref"][key]
            o_ref[seg] = R.causal_attention(q_r[seg], k_all, v_all, scale)
        o_iso, o_ref = o_iso.reshape(n, -1), o_ref.reshape(n, -1)
        note("op", "attention", attn, o_iso)
        note("op", "o_proj", partial, R.linear(o_in, o_w[:, self.q_cols]))
        note(
            "op",
            "write",
            s_mid,
            stream_write(s_in, u, write_logit, retention_logit, cfg),
            stream_scale(s_in, s_mid),
        )
        note("op", "next_read0", read, normed_read(s_mid, shared_logit, pre_moe, cfg))

        parts = self.ar_checks(phase, "attention", i, rec["ar_o"], "o_proj")
        self.flag(
            phase, "attention", i, "same_on_every_rank:next_read0", self.same_on_every_rank(read)
        )
        o_full = self.gather_cols(o_in)
        u_twin = X.cublas_mm(o_full, o_w.t())
        self.within_rounding(phase, "attention", i, "o_proj", parts, u, u_twin)
        s_twin, (read_twin,) = self.replay_write(call, u_twin)
        note("tp", "update", u, u_twin)
        note("tp", "write", s_mid, s_twin, stream_scale(s_in, s_twin))
        note("tp", "next_read0", read, read_twin)

        u_ref = R.linear(o_ref, o_w)
        s_ref = stream_write(s_in, u_ref, write_logit, retention_logit, cfg)
        read_ref = normed_read(s_ref, shared_logit, pre_moe, cfg)
        sc = stream_scale(s_in, s_ref)
        note("module", "read", x, x_ref)
        note("module", "attention", attn, o_ref[:, self.q_cols])
        note("module", "update", u, u_ref)
        note("module", "write", s_mid, s_ref, sc)
        note("module", "next_read0", read, read_ref)
        note("twin", "update", u_twin, u_ref)
        note("twin", "write", s_twin, s_ref, sc)
        note("twin", "next_read0", read_twin, read_ref)
        if rivals:
            planted = dict(
                no_allreduce=partial,
                heads_rotated=X.cublas_mm(torch.roll(o_full, o_in.shape[1], dims=1), o_w.t()),
                bf16_running_sum=bf16_running_sum(parts),
            )
            for name, u_rival in planted.items():
                s_rival = stream_write(s_in, u_rival, write_logit, retention_logit, cfg)
                note(f"rival_{name}", "update", u_rival, u_ref)
                note(f"rival_{name}", "write", s_rival, s_ref, sc)
                self.flag(
                    phase,
                    "attention",
                    i,
                    f"rival_{name}:ar_bitwise",
                    torch.equal(u_rival, X.allreduce.reference(parts)),
                )

    # --- dense MLP ---------------------------------------------------------------------------------

    def mlp(self, i, rec, rows, phase, rivals) -> None:
        torch, R, cfg, w = X.torch, X.R, self.cfg, self.w[i]
        missing = {"x", "mlp_up", "mlp_down", "ar_mlp_down", "plain_call"} - rec.keys()
        assert not missing, f"layer {i}: the observer saw no {sorted(missing)}"
        call = rec["plain_call"]
        s_in, u, s_out, reads = call["s_in"], call["update"], call["s_out"], call["reads"]
        x, up = rec["x"], rec["mlp_up"]["out"]
        act, partial = rec["mlp_down"]["a"], rec["mlp_down"]["out"]
        conn = "residual_connection"
        read_logit, norm = w[f"{conn}.read_logit"], w["input_layernorm.weight"]
        write_logit, retention_logit = w[f"{conn}.write_logit"], w[f"{conn}.retention_logit"]
        up_w, down_w = w["mlp.up_proj.weight"], w["mlp.down_proj.weight"]
        clamp = cfg.activation_clamp_value
        note = lambda c, o, out, ref, sc=None: self.note(  # noqa: E731
            phase, "mlp", i, c, o, measure(out, ref, sc)
        )

        note("op", "read", x, normed_read(s_in, read_logit, norm, cfg))
        note("op", "up", up, R.linear(x, up_w[self.mlp_cols]))
        note("colpar", "up", up, X.cublas_mm(x, up_w.t())[:, self.mlp_cols])
        note("op", "act", act, R.clamped_relu2(up, clamp))
        note("op", "down", partial, R.linear(act, down_w[:, self.mlp_cols]))
        note(
            "op",
            "write",
            s_out,
            stream_write(s_in, u, write_logit, retention_logit, cfg),
            stream_scale(s_in, s_out),
        )
        for k, (logit, nw) in enumerate(w["next"]):
            note("op", f"next_read{k}", reads[k], normed_read(s_out, logit, nw, cfg))

        parts = self.ar_checks(phase, "mlp", i, rec["ar_mlp_down"], "down")
        for k, read in enumerate(reads):
            self.flag(
                phase, "mlp", i, f"same_on_every_rank:next_read{k}", self.same_on_every_rank(read)
            )
        act_full = self.gather_cols(act)
        u_twin = X.cublas_mm(act_full, down_w.t())
        self.within_rounding(phase, "mlp", i, "down", parts, u, u_twin)
        s_twin, reads_twin = self.replay_write(call, u_twin)
        note("tp", "update", u, u_twin)
        note("tp", "write", s_out, s_twin, stream_scale(s_in, s_twin))
        for k in range(len(reads)):
            note("tp", f"next_read{k}", reads[k], reads_twin[k])

        x_ref = normed_read(s_in, read_logit, norm, cfg)
        h_ref = R.linear(x_ref, up_w)
        a_ref = R.clamped_relu2(h_ref, clamp)
        u_ref = R.linear(a_ref, down_w)
        s_ref = stream_write(s_in, u_ref, write_logit, retention_logit, cfg)
        reads_ref = [normed_read(s_ref, logit, nw, cfg) for logit, nw in w["next"]]
        sc = stream_scale(s_in, s_ref)
        note("module", "read", x, x_ref)
        note("module", "up", up, h_ref[:, self.mlp_cols])
        note("module", "act", act, a_ref[:, self.mlp_cols])
        note("module", "update", u, u_ref)
        note("module", "write", s_out, s_ref, sc)
        for k, read_ref in enumerate(reads_ref):
            note("module", f"next_read{k}", reads[k], read_ref)
        note("twin", "update", u_twin, u_ref)
        note("twin", "write", s_twin, s_ref, sc)
        for k, read_ref in enumerate(reads_ref):
            note("twin", f"next_read{k}", reads_twin[k], read_ref)
        if rivals:
            planted = dict(
                no_allreduce=partial,
                slices_rotated=X.cublas_mm(torch.roll(act_full, act.shape[1], dims=1), down_w.t()),
                bf16_running_sum=bf16_running_sum(parts),
            )
            for name, u_rival in planted.items():
                s_rival = stream_write(s_in, u_rival, write_logit, retention_logit, cfg)
                note(f"rival_{name}", "update", u_rival, u_ref)
                note(f"rival_{name}", "write", s_rival, s_ref, sc)
                self.flag(
                    phase,
                    "mlp",
                    i,
                    f"rival_{name}:ar_bitwise",
                    torch.equal(u_rival, X.allreduce.reference(parts)),
                )
        # The last block's reads are the readout and the final norm: the hidden lm_head reads.
        if i == len(cfg.layer_types) - 1:
            self.final_hidden = reads[0]

    # --- latent MoE --------------------------------------------------------------------------------

    def moe(self, i, rec, rows, phase, rivals) -> None:
        torch, R, K, cfg, w = X.torch, X.R, X.K, self.cfg, self.w[i]
        missing = {
            "routed",
            "shared",
            "router",
            "noaux",
            "fc1_latent",
            "experts",
            "shared_up",
            "shared_down",
            "ar_shared_down",
            "fc2_norm",
            "fc2_latent",
            "mixer_call",
            "moe_call",
        } - rec.keys()
        assert not missing, f"layer {i}: the observer saw no {sorted(missing)}"
        mw = {k[len("moe.") :]: v for k, v in w.items() if k.startswith("moe.")}
        clamp, eps = cfg.activation_clamp_value, cfg.rms_norm_eps
        mixer_call, call = rec["mixer_call"], rec["moe_call"]
        s_mid, out, s_out, reads = call["s_in"], call["update"], call["s_out"], call["reads"]
        routed, shared = rec["routed"], rec["shared"]
        latent, partial = rec["fc1_latent"]["out"], rec["experts"]["out"]
        weights, ids = rec["noaux"]["weights"], rec["noaux"]["ids"]
        sh_up, sh_act, sh_part = (
            rec["shared_up"]["out"],
            rec["shared_down"]["a"],
            rec["shared_down"]["out"],
        )
        ar = rec["ar_shared_down"]
        norm_in, norm_out = rec["fc2_norm"]["x"], rec["fc2_norm"]["out"]
        n, latent_width = latent.shape
        conn = "moe_residual_connection"
        write_logit, retention_logit = w[f"{conn}.write_logit"], w[f"{conn}.retention_logit"]
        note = lambda c, o, out_, ref, sc=None, ulp=_BF16_ULP: self.note(  # noqa: E731
            phase, "moe", i, c, o, measure(out_, ref, sc, ulp)
        )

        # The family's MoE measurement, as on tp1: routing against HF's router, the block against
        # the fp32-GEMM and HF-order references, each stage on the target's own input. Its
        # `experts` stage is the all-reduced latent sum against the full-width reference loop.
        route, agree = K.routing(
            rec["noaux"]["logits"], rec["noaux"]["bias"], ids, weights, routed, mw, cfg
        )
        seen = SimpleNamespace(
            ids=ids, weights=weights, latent=latent, experts=norm_in, fc2_norm=norm_out
        )
        block, stages = K.block_and_stages(out, routed, shared, mw, cfg, seen, agree)
        self.note(phase, "moe", i, "routing", "routing", route)
        for key, metrics in block.items():
            self.note(phase, "moe", i, "block", key, metrics)
        for key, metrics in stages.items():
            self.note(phase, "moe", i, "stage", key, metrics)

        # This rank's ops, on its own input to each.
        note(
            "op",
            "routed_read",
            routed,
            normed_read(
                mixer_call["s_in"],
                w["shortcut_read.logit"],
                w["shortcut_input_layernorm.weight"],
                cfg,
            ),
        )
        note(
            "op",
            "shared_read",
            shared,
            normed_read(
                mixer_call["s_out"], w[f"{conn}.read_logit"], w["pre_moe_layernorm.weight"], cfg
            ),
        )
        up_r, down_r = self.moe_rank_experts[i]
        note(
            "op",
            "experts_partial",
            partial,
            X.latent_moe_experts.reference(
                latent, ids, weights, up_r, down_r, clamp, 0, cfg.n_routed_experts
            ),
        )
        sh_up_w = mw["shared_experts.up_proj.weight"]
        sh_down_w = mw["shared_experts.down_proj.weight"]
        note("op", "shared_up", sh_up, R.linear(shared, sh_up_w[self.shared_cols]))
        note("colpar", "shared_up", sh_up, X.cublas_mm(shared, sh_up_w.t())[:, self.shared_cols])
        note("op", "shared_act", sh_act, R.clamped_relu2(sh_up, clamp))
        note("op", "shared_down", sh_part, R.linear(sh_act, sh_down_w[:, self.shared_cols]))
        note(
            "op",
            "write",
            s_out,
            stream_write(s_mid, out, write_logit, retention_logit, cfg),
            stream_scale(s_mid, s_out),
        )
        for k, (logit, nw) in enumerate(w["next"]):
            note("op", f"next_read{k}", reads[k], normed_read(s_out, logit, nw, cfg))

        # The packed all-reduce: the latent partial and the shared partial end to end in one
        # [T, latent + hidden] call; then fc2_norm on the latent sum, fc2 and the add.
        packed = torch.cat((partial.view(-1), sh_part.view(-1))).view(n, -1)
        self.flag(phase, "moe", i, "packing", torch.equal(ar["input"], packed))
        summed = ar["out"].view(-1)
        split = n * latent_width
        latent_sum = summed[:split].view(n, latent_width)
        shared_sum = summed[split:].view(n, -1)
        self.flag(
            phase, "moe", i, "fc2_norm_reads_the_latent_sum", torch.equal(norm_in, latent_sum)
        )
        self.flag(
            phase,
            "moe",
            i,
            "block_adds_the_shared_sum",
            torch.equal(out, torch.add(rec["fc2_latent"]["out"], shared_sum)),
        )
        parts = self.ar_checks(phase, "moe", i, ar, "packed")
        self.flag(phase, "moe", i, "same_on_every_rank:block", self.same_on_every_rank(out))
        for k, read in enumerate(reads):
            self.flag(
                phase, "moe", i, f"same_on_every_rank:next_read{k}", self.same_on_every_rank(read)
            )

        # The TP term of each half of the packed all-reduce: the routed experts at full width on
        # the same latent input and routing (one call), and the shared expert's down projection
        # as one full-K GEMM on the ranks' gathered activations.
        latent_twin = X.latent_moe_experts(
            latent,
            ids,
            weights,
            mw["experts.up_proj"],
            mw["experts.down_proj"],
            clamp,
            0,
            cfg.n_routed_experts,
        )
        shared_twin = X.cublas_mm(self.gather_cols(sh_act), sh_down_w.t())
        shared_parts = [p.view(-1)[split:].view(n, -1) for p in parts]
        self.within_rounding(phase, "moe", i, "shared_down", shared_parts, shared_sum, shared_twin)
        note("tp", "latent_sum", latent_sum, latent_twin)
        note("tp", "shared_sum", shared_sum, shared_twin)

        # The twin block: the target's own `_moe` with every expert at full width, the shared
        # expert at full width and the all-reduce taken out -- a single GPU's block on the same
        # entries -- then the same write.
        # The target that ran the step: prefill when any row fed prompt tokens (`cached` is the
        # request's progress before the step).
        has_context = any(r.cached < len(r.request.prompt) for r in rows)
        target = self.model.model._prefill if has_context else self.model.model._decode
        layer = target._layers[i].moe
        ptr_up, ptr_down = layer.shared_up.data_ptr(), layer.shared_down.data_ptr()
        full = {ptr_up: sh_up_w.t(), ptr_down: sh_down_w.t()}

        def mm(mat_a, mat_b, *args, **kwargs):
            return X.cublas_mm(mat_a, full.get(mat_b.data_ptr(), mat_b), *args, **kwargs)

        def experts(x_, ids_, w_, up_, down_, clamp_, offset_, count_, out_dtype=None):
            return X.latent_moe_experts(
                x_,
                ids_,
                w_,
                mw["experts.up_proj"],
                mw["experts.down_proj"],
                clamp_,
                offset_,
                count_,
                out_dtype,
            )

        import pytest

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(X.M, "cublas_mm", mm)
            mp.setattr(X.M, "latent_moe_experts", experts)
            mp.setattr(X.M, "allreduce", lambda input, comm_buffer, buffer_flags: input)
            mp.setattr(X.M, "noaux_tc_op", X.noaux_tc_op)
            mp.setattr(X.M, "flashinfer_rmsnorm", X.flashinfer_rmsnorm)
            out_twin = target._moe(routed, shared, layer)
        s_twin, reads_twin = self.replay_write(call, out_twin)
        note("tp", "block", out, out_twin)
        note("tp", "write", s_out, s_twin, stream_scale(s_mid, s_twin))
        for k in range(len(reads)):
            note("tp", f"next_read{k}", reads[k], reads_twin[k])

        # The twin and the target against the references that round where the target rounds
        # (`R_t`) and where HF rounds (`R_hf`), and the write of the block, in the module view.
        r_t = K.reference_chain(routed, shared, mw, cfg, ids, weights, K.rms_norm_once)
        r_hf = K.reference_chain(routed, shared, mw, cfg, ids, weights, R.rms_norm)
        note("twin", "block vs R_t", out_twin, r_t)
        note("twin", "block vs R_hf", out_twin, r_hf)
        s_ref = stream_write(s_mid, r_t, write_logit, retention_logit, cfg)
        reads_ref = [normed_read(s_ref, logit, nw, cfg) for logit, nw in w["next"]]
        sc = stream_scale(s_mid, s_ref)
        note("module", "write", s_out, s_ref, sc)
        note("twin", "write", s_twin, s_ref, sc)
        for k, read_ref in enumerate(reads_ref):
            note("module", f"next_read{k}", reads[k], read_ref)
            note("twin", f"next_read{k}", reads_twin[k], read_ref)

        if rivals:
            fc2_w = mw["fc2_latent_proj.weight"]
            fc2_norm_w = mw["fc2_norm.weight"]

            def tail(latent_, shared_):
                normed = X.flashinfer_rmsnorm(latent_, fc2_norm_w, eps)
                return torch.add(X.cublas_mm(normed, fc2_w.t()), shared_)

            gathered = [p.view(-1)[:split].view(n, latent_width) for p in parts]
            norm_first = sum(
                X.cublas_mm(X.flashinfer_rmsnorm(g, fc2_norm_w, eps), fc2_w.t()).float()
                for g in gathered
            )
            planted = dict(
                no_allreduce=tail(partial, sh_part),
                norm_before_reduce=(norm_first + shared_sum.float()).to(out.dtype),
                bf16_running_sum=tail(bf16_running_sum(gathered), bf16_running_sum(shared_parts)),
            )
            for name, rival in planted.items():
                s_rival = stream_write(s_mid, rival, write_logit, retention_logit, cfg)
                note(f"rival_{name}", "block vs R_t", rival, r_t)
                note(f"rival_{name}", "write", s_rival, s_ref, sc)
            # The reductions a rival stands for, against the certified sum of the packed call.
            certified = X.allreduce.reference(parts)
            for name, packed_sum in (
                ("no_allreduce", ar["input"]),
                ("bf16_running_sum", bf16_running_sum(parts)),
            ):
                self.flag(
                    phase, "moe", i, f"rival_{name}:ar_bitwise", torch.equal(packed_sum, certified)
                )

    # --- lm_head -----------------------------------------------------------------------------------

    def lm_head(self, record, logits, rows, phase) -> None:
        torch = X.torch
        rec = record["lm_head"].get("logits")
        assert rec is not None, "the observer saw no lm_head GEMM"
        hidden, out = rec["a"], rec["out"]
        if out.shape[1] == logits.shape[1]:
            # The whole GEMM, above the vocab-parallel rows.
            cols = slice(None)
            self.flag(
                phase, "lm_head", -1, "returned_logits_are_the_gemm", torch.equal(out, logits)
            )
        else:
            # Vocab-parallel: this rank's slice, the rank-th quarter of lm_head's rows.
            cols = slice(self.rank * out.shape[1], (self.rank + 1) * out.shape[1])
            self.flag(
                phase,
                "lm_head",
                -1,
                "returned_logits_are_every_ranks_gemm_side_by_side",
                torch.equal(logits, self.gather_cols(out)),
            )
        self.flag(
            phase, "lm_head", -1, "same_on_every_rank:logits", self.same_on_every_rank(logits)
        )
        self.note(
            phase,
            "lm_head",
            -1,
            "op",
            "logits",
            measure(out, X.R.linear_fp32(hidden, self.lm_head_w[cols]), ulp=_FP32_ULP),
        )
        final = getattr(self, "final_hidden", None)
        if final is not None:
            last = torch.tensor([r.start + len(r.tokens) - 1 for r in rows], device="cuda")
            self.flag(
                phase, "lm_head", -1, "reads_the_final_hidden", torch.equal(hidden, final[last])
            )
            self.final_hidden = None

    # --- the phases --------------------------------------------------------------------------------

    def run(self) -> None:
        P = X.P
        prompts = P.prompt_set(P.load_tokenizer(self.ckpt.path))
        by_name = {p.name: p for p in prompts}
        short = [self.request(p.name, p.token_ids) for p in prompts if not p.long]
        req = {r.name: r for r in short}
        md_u, md_c = self.md_unchunked, self.md_chunked

        # context: the six short prompts in one varlen batch.
        self.step(md_u, [Row(r, r.prompt) for r in short], "context", rivals=True)
        if self.quick:
            r = req["gpqa_explain"] = self.request(
                "gpqa_explain", by_name["gpqa_explain"].token_ids
            )
            for _ in range(2):
                chunk = r.prompt[r.done : r.done + CHUNK]
                self.step(md_c, [Row(r, chunk)], "chunked", rivals=r.done > 0)
            self.step(md_u, [Row(s, [s.generated[-1]]) for s in short], "decode_b6", rivals=True)
            return

        # context: two long prompts alone, unchunked (the 8K-row cells).
        for name in ("gsm8k_fewshot", "news_digest"):
            r = req[name] = self.request(name, by_name[name].token_ids)
            self.step(md_u, [Row(r, r.prompt)], "context")

        # chunked: a long prompt in 2048-token chunks over its own cached KV and state.
        r = req["gpqa_explain"] = self.request("gpqa_explain", by_name["gpqa_explain"].token_ids)
        while r.in_context:
            chunk = r.prompt[r.done : r.done + CHUNK]
            self.step(md_c, [Row(r, chunk)], "chunked", rivals=r.done >= 2 * CHUNK)

        # mixed: the next long prompt's chunks beside the short prompts' generation rows.
        r = req["gsm8k_multiturn"] = self.request(
            "gsm8k_multiturn", by_name["gsm8k_multiturn"].token_ids
        )
        while r.in_context:
            gen = [Row(s, [s.generated[-1]]) for s in short]
            chunk = r.prompt[r.done : r.done + MIXED_CHUNK]
            self.step(md_c, [Row(r, chunk)] + gen, "mixed", rivals=r.done > 0)

        # More histories for the batch-64 decode: prompt prefixes, whose GDP states the reference
        # does not carry (its per-token loop is the slow part); every other block checks them.
        sources = [by_name[n].token_ids for n in ("gsm8k_fewshot", "news_digest", "gpqa_explain")]
        extra = [
            self.request(f"prefix{k}", sources[k % len(sources)][:length], tracked=False)
            for k, length in enumerate(EXTRA_LENGTHS)
        ]
        batch: list = []
        for e in extra:
            if batch and sum(len(b.tokens) for b in batch) + len(e.prompt) > MAX_NUM_TOKENS:
                self.step(md_u, batch, "context")
                batch = []
            batch.append(Row(e, e.prompt))
        if batch:
            self.step(md_u, batch, "context")

        # decode: generation-only steps at the batch sizes the decode graphs capture at tp4's
        # latency end (1, 4, 16) and at 64, two steps each, histories up to ~8K.
        long_ones = ["gsm8k_fewshot", "news_digest", "gpqa_explain", "gsm8k_multiturn"]
        everyone = list(req.values()) + extra
        assert len(everyone) == MAX_BATCH, len(everyone)
        for phase, batch_rows in (
            ("decode_b1", [req["gsm8k_fewshot"]]),
            ("decode_b4", [req[n] for n in long_ones]),
            ("decode_b16", list(req.values()) + extra[:6]),
            ("decode_b64", everyone),
        ):
            for k in range(2):
                self.step(
                    md_u,
                    [Row(e, [e.generated[-1]]) for e in batch_rows],
                    phase,
                    rivals=k == 0 and phase in ("decode_b4", "decode_b64"),
                )

    def close(self) -> None:
        shutdown = getattr(self.manager, "shutdown", None)
        if shutdown is not None:
            shutdown()


# ----------------------------------------------------------------------------------------------
# Rank body
# ----------------------------------------------------------------------------------------------


def _rank_body(out_path: str, checkpoint_path: str, quick: bool) -> None:
    """Everything one rank does; every collective in lockstep with the other three."""
    from mpi4py import MPI

    _bind()
    torch = X.torch
    import pytest

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == WORLD_SIZE, f"tp4 needs {WORLD_SIZE} ranks, got {comm.Get_size()}"
    with pytest.MonkeyPatch.context() as mp:
        # Built as the engine builds it, outside inference mode; only the steps run inside it.
        campaign = _Campaign(comm, checkpoint_path, quick)
        campaign.observer.install(mp)
        with torch.inference_mode():
            campaign.run()
        torch.cuda.synchronize()
        peak = comm.gather(torch.cuda.max_memory_allocated() / 2**30, root=0)
        everything = comm.gather(campaign.records, root=0)
        if rank == 0:
            merged = dict(
                records=[r for per in everything for r in per],
                steps=campaign.steps,
                meta=dict(
                    world_size=WORLD_SIZE,
                    quick=quick,
                    layers={k: list(v) for k, v in campaign.layers.items()},
                    seconds=time.monotonic() - campaign.started,
                    device=torch.cuda.get_device_name(),
                    manager=type(campaign.manager).__name__,
                    peak_gib_per_rank=peak,
                ),
            )
            with open(out_path, "w") as f:
                json.dump(merged, f)
            campaign.say(f"wrote {out_path}: {len(merged['records'])} records; peak GiB {peak}")
        comm.Barrier()
        campaign.close()


def _run_one_rank(out: str, checkpoint: str, quick: bool) -> int:
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
        _rank_body(out, checkpoint, quick)
    except BaseException:
        print(f"[rank {MPI.COMM_WORLD.Get_rank()}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Exit hard rather than return: a rank that leaves a collective early wedges every
        # sibling in it, and mpirun ends the job the moment it sees this exit.
        os._exit(13)
    sys.stdout.flush()
    # Finalize so mpirun counts this as a clean exit, then leave without running teardown.
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
        "--quick", action="store_true", help="a few layers and steps: a harness shakeout"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse(sys.argv[1:])
    if args.rank_worker:
        sys.exit(_run_one_rank(args.out, args.checkpoint, args.quick))
    spawn_ranks(os.path.abspath(args.out), args.checkpoint, args.quick)
    print("OK", flush=True)
