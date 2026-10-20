# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4-rank job: an MTP verify step of the nemotron4 tp4 target's GDP mixer rolls back exactly, on
every rank's own cache manager.

Under one-model MTP-Eagle a generation row carries its last token and D drafts. Each rank's GDP
mixer runs the row's 1 + D tokens on its 6 of the 24 heads from the row's state slot without
writing it, and stores the state after every token in the hybrid cache manager's per-position
buffers; after acceptance the engine's worker commits the last accepted token's state into the
slot (`update_mamba_states`). A rejected draft must leave no trace: the committed slot has to be
the state MTP-off decode reaches over the accepted tokens alone.

This script is its own launcher, as the tp4 decode-invariance job is: run it plainly and it starts
one MPI rank per device named in CUDA_VISIBLE_DEVICES under ``mpirun``, with a deadline it enforces
by killing the job.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _tp4_mtp_rollback_ranks.py --report <json>

What every rank builds, as the engine builds it for its rank under ``MTPDecodingConfig(max_draft_len
=2)``: ``ModelConfig`` off the checkpoint with the rank's tp4 mapping (tp 4, MoE tp 4, no attention
DP); the TP group's all-reduce workspace (collective, as the target's ``post_load_weights`` builds
it); and the trunk's hybrid cache manager for that mapping (``_create_kv_cache_manager`` over the
44 trunk layers, the MTP layer's KV living in a separate draft manager on tp4), with the verify's
per-position buffers -- both managers the engine offers, ``MambaHybridCacheManagerV2`` (the
default) and ``CppMambaHybridCacheManager``. The mixer is layer 0's, the rank's slice of the
checkpoint's weights cut and laid out by tp4's own weight table, run through the targets' own code
(``DecodeTarget._gdp`` for a decode or verify step, ``PrefillTarget._gdp`` for a mixed one), whose
``out_proj`` partials meet in the real all-reduce. The commit is the manager's
``update_mamba_states`` with the engine's own slots, as the worker calls it. Every rank feeds the
same rows, as the engine's ranks do.

The cases, on both managers:

* ``padded``: 5 requests verifying 1 + 2 tokens, padded to a graph batch of 8 with the padding
  dummy's rows (``CUDA_GRAPH_DUMMY_REQUEST_ID - 2``), two verify -> commit rounds whose accepted
  counts cover 1..3. The reference is three MTP-off decode steps through the same mixer.
* ``mixed``: a fresh 37-token prompt (a context row) beside 3 verify rows, through PrefillTarget.
  Its in_proj runs on cuBLAS, whose kernel at the step's 46 rows rounds otherwise than at a decode
  step's 3 -- mixed steps carry no batch-invariance claim -- so the reference decodes the verify
  rows' own in_proj output, one token per step through the decode entries (``_gdp_rows``), and
  finishes each as ``_gdp`` does (the gated norm, the ``out_proj`` partial, the all-reduce).

Per case: every real row's committed recurrent and conv state equals its slot after `accepted`
MTP-off decode steps over the same tokens; the states after one, two and three tokens differ
pairwise, so a commit from the wrong position cannot pass; the verify writes no slot; the commit
writes only the step's rows' engine slots (the padding dummy's included, whose stale snapshot it
copies) and never a context row's; and the verify's output rows equal the decode steps' rows. Rank
0 writes every rank's records to `--report` as JSON. Not a pytest module: everything that imports
`tensorrt_llm` is imported inside the rank body (an MPI-initialized process cannot start `mpirun`).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "MODELING_V2_TP4_MTP_ROLLBACK_PID_DIR"
_REPORT_ENV = "MODELING_V2_TP4_MTP_ROLLBACK_REPORT"
#: Building the managers and the workspace takes ~1 min, the cases seconds.
DEADLINE_S = 1200
WORLD_SIZE = 4
DRAFT = 2
TOKENS = 1 + DRAFT
MAX_BATCH = 8
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
#: A `linear_attention` block; every GDP mixer of the model is the same mixer.
LAYER = 0
MANAGERS = ("v2", "v1")

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4"


def _rank_body(rank: int, comm, checkpoint: str, log) -> list[dict]:
    import importlib
    from types import SimpleNamespace

    import torch
    from safetensors import safe_open

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    M = importlib.import_module(f"{_PACKAGE}.modeling")
    W = importlib.import_module(f"{_PACKAGE}.weights")

    mapping = Mapping(
        world_size=WORLD_SIZE,
        rank=rank,
        gpus_per_node=WORLD_SIZE,
        tp_size=WORLD_SIZE,
        moe_tp_size=WORLD_SIZE,
        moe_ep_size=1,
    )
    spec = MTPDecodingConfig(max_draft_len=DRAFT)
    model_config = ModelConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        mapping=mapping,
        max_num_tokens=MAX_NUM_TOKENS,
        spec_config=spec,
    )
    update_spec_config_from_model_config(spec, model_config.pretrained_config)
    assert spec.spec_dec_mode.is_mtp_eagle_one_model() and spec.tokens_per_gen_step == TOKENS
    config = model_config.pretrained_config
    # The target's shell sets this on the composite config (the text config derives it).
    config.num_hidden_layers = config.text_config.num_hidden_layers
    text = config.text_config

    # Layer 0's GDP mixer at this rank's heads, cut and laid out by tp4's weight table; its input
    # norm and the embedding table, which feed the mixer realistic rows.
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {
        shard: safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    d = W.MODEL_WEIGHTS.dims(SimpleNamespace(model_config=model_config))

    def table(name: str) -> torch.Tensor:
        """The weight table's parameter `name` of layer 0 (or the global one), filled as `load`
        fills it: each source part cut, transformed on the device, copied to its place."""
        entry = next(e for e in W.MODEL_WEIGHTS.WEIGHTS if e.name == name)
        layer = None if entry.layers is None else LAYER
        param = torch.empty(entry.shape(d), dtype=entry.dtype or d.dtype, device="cuda")
        for key, dst, src in entry.sources(d, layer):
            part = handles[weight_map[key]].get_slice(key)[slice(None) if src is None else src]
            part = part.to("cuda")
            if entry.transform is not None:
                part = entry.transform(part, d)
            (param if dst is None else param[dst]).copy_(part)
        return param

    w = {
        name: table(name)
        for name in ("gdp_in", "gdp_conv", "gdp_neg_a", "gdp_dt_bias", "gdp_norm", "gdp_out")
    }
    input_norm, embed = table("input_norm"), table("embed")

    def mixer(row_invariant: bool):
        """The mixer as each target holds it (`_layer_views`): decode's in_proj is the stored
        [out, in] weight `invariant_mm` reads, prefill's the `.t()` view `cublas_mm` reads."""
        return M._Gdp(
            layer=LAYER,
            in_proj=w["gdp_in"] if row_invariant else w["gdp_in"].t(),
            conv=w["gdp_conv"],
            neg_a=w["gdp_neg_a"],
            dt_bias=w["gdp_dt_bias"],
            norm=w["gdp_norm"],
            out_proj=w["gdp_out"].t(),
        )

    decode_mixer, prefill_mixer = mixer(True), mixer(False)

    # Collective: every rank builds the TP group's workspace, as `post_load_weights` does.
    workspace = M._allreduce_workspace(
        model_config, (text.hidden_size, text.moe_latent_size + text.hidden_size)
    )
    dims = M._dims(text, WORLD_SIZE)
    # The two targets' mixer code, without the rest of the model: `_gdp` reads `_d` and `_ar`.
    decode, prefill = object.__new__(M.DecodeTarget), object.__new__(M.PrefillTarget)
    decode._d = prefill._d = dims
    decode._ar = prefill._ar = workspace

    def rows(generator, n: int) -> torch.Tensor:
        """Layer 0's input for random tokens: their embedding, RMS-normed with its input norm."""
        ids = torch.randint(0, embed.shape[0], (n,), generator=generator, device="cuda")
        e = embed[ids].float()
        e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + text.rms_norm_eps)
        return (e.to(torch.bfloat16) * input_norm).contiguous()

    records = []
    for kind in MANAGERS:
        kv_cache_config = KvCacheConfig(
            max_tokens=64 * 1024,
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.05,
            use_kv_cache_manager_v2=kind == "v2",
        )
        manager = _create_kv_cache_manager(
            None,
            get_kv_cache_manager_cls(model_config, kv_cache_config),
            mapping,
            kv_cache_config,
            tokens_per_block=32,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH,
            spec_config=spec,
            sparse_attention_config=None,
            max_num_tokens=MAX_NUM_TOKENS,
            max_beam_width=1,
            kv_connector_manager=None,
            model_config=model_config,
            dtype=torch.bfloat16,
            layer_mask=[True] * config.num_hidden_layers,
            max_cuda_graph_batch_size=MAX_BATCH,
        )
        ssm, conv = manager.get_ssm_states(LAYER), manager.get_conv_states(LAYER)
        inter_ssm = manager.get_intermediate_ssm_states(LAYER)
        inter_conv = manager.get_intermediate_conv_states(LAYER)
        shapes = (tuple(ssm.shape[1:]), tuple(conv.shape[1:]), tuple(inter_ssm.shape[:2]))
        assert shapes == ((6, 64, 128), (4224, 3), (MAX_BATCH, TOKENS)), shapes

        # The step's requests and the padding dummy, their slots handed out by the manager.
        request_ids = list(range(1, 7))
        for rid in request_ids:
            assert manager.add_dummy_requests([rid], token_nums=[64]), rid
        padding_id = CUDA_GRAPH_DUMMY_REQUEST_ID - DRAFT
        added = manager.add_dummy_requests([padding_id], is_gen=True, max_num_draft_tokens=DRAFT)
        assert added, padding_id
        slot = dict(zip(request_ids, manager.get_state_indices(request_ids, [False] * 6)))
        padding_slot = manager.get_state_indices([padding_id], [True])[0]
        assert len(set(slot.values()) | {padding_slot}) == 7, (slot, padding_slot)

        def snapshot():
            return ssm.clone(), conv.clone()

        def restore(saved) -> None:
            ssm.copy_(saved[0])
            conv.copy_(saved[1])

        def changed(saved) -> set:
            out = set()
            for pool, old in zip((ssm, conv), saved):
                diff = (pool.view(pool.shape[0], -1) != old.view(old.shape[0], -1)).any(dim=1)
                out |= set(torch.nonzero(diff).flatten().tolist())
            return out

        def states(slots: list) -> tuple:
            index = torch.tensor(slots, device="cuda")
            return ssm[index].clone(), conv[index].clone()

        def step(slots: list, **context) -> dict:
            """What `step_args` hands the mixer for these rows' slots."""
            return dict(
                state_slots=torch.tensor(slots, dtype=torch.int32, device="cuda"),
                ssm_states={LAYER: ssm},
                conv_states={LAYER: conv},
                intermediate_ssm={LAYER: inter_ssm},
                intermediate_conv={LAYER: inter_conv},
                **context,
            )

        def commit(num_contexts: int, accepted: list, engine_slots: list) -> None:
            """The worker's commit: `update_mamba_states` with the engine's own slots (the padding
            dummy's included) and every row's accepted-token count."""
            manager.update_mamba_states(
                SimpleNamespace(num_seqs=len(engine_slots), num_contexts=num_contexts),
                torch.tensor(accepted, dtype=torch.int32, device="cuda"),
                state_indices=torch.tensor(engine_slots, dtype=torch.int32, device="cuda"),
            )
            torch.cuda.synchronize()

        def prime(g, slots: list, steps: int) -> None:
            """Give the slots a history: zero, then `steps` decode steps of random tokens."""
            index = torch.tensor(slots, device="cuda")
            ssm[index] = 0
            conv[index] = 0
            for _ in range(steps):
                decode._gdp(rows(g, len(slots)), decode_mixer, step(slots))

        def distinct(after: list, num_real: int) -> bool:
            """The states after one, two and three tokens differ pairwise in every row."""
            return all(
                not torch.equal(after[i][k][b], after[j][k][b])
                for b in range(num_real)
                for i in range(TOKENS)
                for j in range(i + 1, TOKENS)
                for k in (0, 1)
            )

        def check_states(real_slots: list, expected, what: str) -> list:
            got = states(real_slots)
            failures = []
            for b, (want_ssm, want_conv) in enumerate(expected):
                if not torch.equal(got[0][b], want_ssm):
                    failures.append(f"{what}: row {b} recurrent state differs")
                if not torch.equal(got[1][b], want_conv):
                    failures.append(f"{what}: row {b} conv state differs")
            return failures

        # The same seed on every rank: the ranks feed the same rows, as the engine's ranks do.
        g = torch.Generator(device="cuda").manual_seed(4404 + (kind == "v1"))

        with torch.inference_mode():
            # --- padded: 5 requests verifying, padded to a graph batch of 8 ---
            real = [slot[r] for r in request_ids[:5]]
            padded = real + [PAD_SLOT_ID] * 3
            engine = real + [padding_slot] * 3
            prime(g, real, steps=6)
            failures, all_distinct = [], True
            for accepted in ([3, 1, 2, 3, 1], [1, 2, 3, 2, 3]):
                start = snapshot()
                x = rows(g, len(padded) * TOKENS).view(len(padded), TOKENS, -1)
                out = decode._gdp(x.reshape(-1, x.shape[-1]), decode_mixer, step(padded))
                torch.cuda.synchronize()
                if changed(start):
                    failures.append(f"padded {accepted}: the verify wrote slots {changed(start)}")
                commit(0, accepted + [1, 2, 3], engine)
                committed = snapshot()
                outside = changed(start) - (set(real) | {padding_slot})
                if outside:
                    failures.append(f"padded {accepted}: the commit wrote slots {outside}")
                # MTP-off decode over the verify's tokens, one step per position, from the start.
                restore(start)
                after, outs = [], []
                for j in range(TOKENS):
                    outs.append(decode._gdp(x[:, j].contiguous(), decode_mixer, step(padded)))
                    after.append(states(real))
                all_distinct &= distinct(after, len(real))
                expected = [
                    (after[accepted[b] - 1][0][b], after[accepted[b] - 1][1][b])
                    for b in range(len(real))
                ]
                restore(committed)
                failures += check_states(real, expected, f"padded {accepted}")
                verify_rows = out.view(len(padded), TOKENS, -1)[: len(real)]
                if not torch.equal(verify_rows, torch.stack(outs, dim=1)[: len(real)]):
                    failures.append(f"padded {accepted}: verify output rows differ from decode's")
            records.append(
                dict(
                    rank=rank,
                    manager=type(manager).__name__,
                    kind=kind,
                    case="padded",
                    failures=failures,
                    distinct=all_distinct,
                )
            )
            log(f"{kind} padded: {len(failures)} failures")

            # --- mixed: a fresh 37-token prompt beside 3 verify rows ---
            context_slot, real = slot[request_ids[5]], [slot[r] for r in request_ids[:3]]
            prime(g, real, steps=4)
            accepted = [2, 3, 1]
            start = snapshot()
            context = rows(g, 37)
            x = rows(g, len(real) * TOKENS)
            batch_rows = torch.cat([context, x])
            starts = torch.tensor([0, 37], dtype=torch.int32, device="cuda")
            out = prefill._gdp(
                batch_rows,
                prefill_mixer,
                step(
                    [context_slot] + real,
                    num_rows=1 + len(real),
                    num_contexts=1,
                    num_ctx_tokens=37,
                    query_start_loc=starts,
                    query_start_loc_long=starts.long(),
                    has_initial_state=torch.zeros(1, dtype=torch.bool, device="cuda"),
                ),
            )
            torch.cuda.synchronize()
            failures = []
            if changed(start) != {context_slot}:
                failures.append(f"mixed: the verify wrote slots {changed(start) - {context_slot}}")
            prefilled = states([context_slot])
            commit(1, [1] + accepted, [context_slot] + real)
            committed = snapshot()
            if changed(start) - ({context_slot} | set(real)):
                failures.append("mixed: the commit wrote a slot outside the step's rows")
            if not all(torch.equal(a, b) for a, b in zip(states([context_slot]), prefilled)):
                failures.append("mixed: the commit wrote the context row's slot")
            # The verify rows' own in_proj output -- cuBLAS at the mixed step's row count, as
            # PrefillTarget computes it -- decoded one token per step from the start.
            restore(start)
            proj = M.cublas_mm(batch_rows, prefill_mixer.in_proj)[37:].view(len(real), TOKENS, -1)
            after, outs = [], []
            for j in range(TOKENS):
                token = proj[:, j].contiguous()
                beta, decay = M.gdp_gates(
                    token[:, dims.beta],
                    token[:, dims.decay],
                    prefill_mixer.neg_a,
                    prefill_mixer.dt_bias,
                )
                y = M._gdp_rows(
                    token, beta, decay, prefill_mixer, step(real)["state_slots"], step(real), dims
                )
                y = M.gated_rmsnorm(
                    y.view(len(real), dims.gdp_value_dim),
                    prefill_mixer.norm,
                    None,
                    dims.gdp_eps,
                    z=token[:, : dims.gdp_value_dim],
                    group_size=dims.gdp_value,
                    norm_before_gate=False,
                    is_rms_norm=True,
                )
                outs.append(
                    M.allreduce(
                        M.cublas_mm(y, prefill_mixer.out_proj), workspace.buffer, workspace.flags
                    )
                )
                after.append(states(real))
            torch.cuda.synchronize()
            expected = [
                (after[accepted[b] - 1][0][b], after[accepted[b] - 1][1][b])
                for b in range(len(real))
            ]
            mixed_distinct = distinct(after, len(real))
            restore(committed)
            failures += check_states(real, expected, f"mixed {accepted}")
            if not torch.equal(out[37:].view(len(real), TOKENS, -1), torch.stack(outs, dim=1)):
                failures.append("mixed: verify output rows differ from the decode steps' rows")
            records.append(
                dict(
                    rank=rank,
                    manager=type(manager).__name__,
                    kind=kind,
                    case="mixed",
                    failures=failures,
                    distinct=mixed_distinct,
                )
            )
            log(f"{kind} mixed: {len(failures)} failures")
        manager.shutdown()
        comm.Barrier()
    return records


def _run_one_rank() -> int:
    """Body of one MPI rank: a rank whose code raises exits hard, and mpirun ends the job."""
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    import torch

    torch.cuda.set_device(local)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == WORLD_SIZE, f"tp4 runs {WORLD_SIZE} ranks, got {comm.Get_size()}"
    pid_dir = os.environ.get(_PID_DIR_ENV)
    if pid_dir:
        Path(pid_dir, f"rank{rank}.pid").write_text(str(os.getpid()))
    here = Path(__file__).resolve()
    sys.path[:0] = [str(here.parents[1]), str(here.parents[3])]
    import _nemotron4_reference as R

    def log(line: str) -> None:
        print(f"[rank {rank}] {line}", flush=True)

    try:
        start = time.time()
        records = _rank_body(rank, comm, R.default_checkpoint(), log)
        log(f"done in {time.time() - start:.0f}s")
    except BaseException:
        print(f"[rank {rank}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    gathered = comm.gather(records, root=0)
    if rank == 0:
        report = dict(ranks=WORLD_SIZE, records=[r for per in gathered for r in per])
        Path(os.environ[_REPORT_ENV]).write_text(json.dumps(report, indent=1))
    comm.Barrier()
    torch.cuda.synchronize()
    sys.stdout.flush()
    # Finalize so mpirun counts this as a clean exit, then leave without running teardown: the
    # all-reduce workspace goes with the process.
    MPI.Finalize()
    os._exit(0)


def _standalone_env() -> dict:
    """This process's environment, minus any enclosing launcher's identity (SLURM_*, PMIX_*,
    PMI_*, OMPI_*): a nested mpirun that inherited it would join that job."""
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    return env


def _spawn_ranks(report: Path) -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 4 devices this run owns, e.g. 0,1,2,3"
    devices = [d for d in visible.split(",") if d.strip()]
    assert len(devices) == WORLD_SIZE, (
        f"CUDA_VISIBLE_DEVICES names {len(devices)} device(s); tp4 runs {WORLD_SIZE}"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; this job runs its ranks as an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="tp4_mtp_rollback_")
    env = dict(_standalone_env(), **{_PID_DIR_ENV: pid_dir, _REPORT_ENV: str(report)})
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(WORLD_SIZE)]
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
            raise AssertionError(f"the {WORLD_SIZE}-rank run did not finish in {DEADLINE_S}s")
        time.sleep(0.5)
    shutil.rmtree(pid_dir, ignore_errors=True)
    if job.returncode != 0:
        raise AssertionError(f"the {WORLD_SIZE}-rank run failed: mpirun exited {job.returncode}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    assert "--report" in sys.argv, "usage: _tp4_mtp_rollback_ranks.py --report <json>"
    _spawn_ranks(Path(sys.argv[sys.argv.index("--report") + 1]).resolve())
    print("OK", flush=True)
