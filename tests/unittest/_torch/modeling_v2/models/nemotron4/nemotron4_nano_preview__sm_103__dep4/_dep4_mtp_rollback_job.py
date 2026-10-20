# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 4-rank job behind `test_modeling_v2_dep4_mtp_rollback.py`: an MTP verify step of the dep4
target's GDP mixer, committed by the engine's own promotion, on every rank's own cache manager.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _dep4_mtp_rollback_job.py --out <json> [--draft D] [--checkpoint <dir>]

Run plainly, this script is its own launcher: it starts one MPI rank per device under `mpirun`, as
a dep4 serving engine starts its ranks, with a deadline it enforces by killing the job.

What every rank builds, as the engine builds it for its rank under ``MTPDecodingConfig(max_draft_len
=D)`` (``--draft``, one of the draft lengths dep4 runs, `DRAFTS`): ``ModelConfig`` off the checkpoint
with the rank's dep4 ``Mapping``, and the hybrid cache
manager for that mapping (``_create_kv_cache_manager``; under attention DP one manager holds the
trunk's state and KV and the MTP layer's KV, with the verify's per-position buffers) -- both
managers the engine offers, ``MambaHybridCacheManagerV2`` (the default) and
``CppMambaHybridCacheManager``. Requests, the attention-DP dummy (``ATTENTION_DP_DUMMY_REQUEST_ID``,
a generation request carrying the drafts, as the executor pads an idle rank) and the runner's MTP
padding dummy (``CUDA_GRAPH_DUMMY_REQUEST_ID - D``) are registered with the manager, which hands out
their slots. The mixer is layer 0's, on the checkpoint's own weights laid out by dep4's weight table
transforms, run through the targets' own code (``DecodeTarget._gdp`` for a decode or verify step,
``PrefillTarget._gdp`` for a mixed one); the commit is the manager's ``update_mamba_states`` with the
engine's own slots, as the worker calls it.

The cases, each on every rank with its own seed, on both managers (`CASES`):

* ``padded``: 5 requests verifying 1 + D tokens, padded to a graph batch of 8 with the padding dummy's
  rows, two verify -> commit rounds whose accepted counts cover 1..1 + D;
* ``idle``: a rank with no request: the attention-DP dummy padded to a graph batch of 4;
* ``dummy alone``: the attention-DP dummy alone, eagerly;
* ``mixed``: a fresh 37-token prompt (a context row) beside 3 verify rows.

Per case, every real row against its slot after `accepted` MTP-off decode steps (one token each)
over the same tokens: its committed conv state bit for bit -- a window of the in_proj output's conv
columns, which both conv entries copy rather than compute -- and its committed recurrent state
within `STATE_ULP`; the verify's GDP entry outputs within `OUT_ULP` of the decode steps', observed
where `_gdp_generation` calls `gdp_verify` and `gdp_recurrent`; and the states after each of the
1 + D tokens differ pairwise -- the conv states in some bit, the recurrent ones by more than their
band -- so a commit from the wrong position cannot pass. The verify writes no slot, and the
commit writes only the step's rows' engine slots (the dummies' included, whose stale snapshot it
copies) and never a context row's. Rank 0 writes every rank's records to `--out` as JSON.

Not a pytest module: everything that imports `tensorrt_llm` is imported inside the rank body (an
MPI-initialized process cannot start `mpirun`).
"""

from __future__ import annotations

import argparse
import json
import math
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
_PID_DIR_ENV = "NEMOTRON4_DEP4_MTP_ROLLBACK_PID_DIR"
DEADLINE_S = 1200
WORLD_SIZE = 4
DRAFTS = (2, 3)
"""The draft lengths dep4 runs: the checkpoint's trained depth, and the repeated layer's next."""
DRAFT = 2
"""The draft length of this run (`--draft`); a verify row carries `TOKENS` = 1 + DRAFT tokens."""
TOKENS = 1 + DRAFT
#: Per 1 + D, the accepted counts of the 5 real rows in each of the two padded rounds: every count
#: 1..1 + D in each round up to 1 + D = 5; at 6, one more count than rows, in the two together.
_ROUNDS = {
    3: ([3, 1, 2, 3, 1], [1, 2, 3, 2, 3]),
    4: ([4, 1, 2, 3, 4], [1, 4, 3, 2, 4]),
    5: ([5, 1, 3, 4, 2], [2, 5, 4, 1, 3]),
    6: ([6, 1, 3, 5, 2], [4, 6, 2, 1, 5]),
}
#: Per 1 + D, the accepted counts of the mixed step's 3 verify rows, the last position among them.
_MIXED = {3: [2, 3, 1], 4: [4, 1, 3], 5: [5, 2, 4], 6: [6, 4, 1]}
MAX_BATCH = 8
LAYER = 0
"""A `linear_attention` block; every GDP mixer of the model is the same mixer."""
MANAGERS = ("v2", "v1")
CASES = ("padded", "idle", "dummy alone", "mixed")
ULP_FP32, ULP_BF16 = 2.0**-23, 2.0**-8

STATE_ULP = 16.0
"""The committed recurrent state against the decode steps', element-wise in fp32 ulps of its row's
scale (a row: the 128 key entries of one head and value channel, scaled by their max |decode|),
times sqrt(t + 1) for the state after token t. gdp_verify's snapshot after token t is within
8 sqrt(t + 1) of the fp32 reference (`_compare_states` in its test): gdp_recurrent's band for one
token, 8, widened as the roundings of t + 1 tokens add. t + 1 gdp_recurrent calls carry the same
t + 1 tokens of roundings, so they sit as far from the same reference, and the two within
16 sqrt(t + 1) of each other. That holds where both are handed the same q, k, v, decay and beta:
where the in_proj GEMM gives a token the same bits at the verify's 3B rows as at a decode step's
B, as the bitwise conv state needs too, and the two conv entries round a token's output alike. A
commit from the wrong position moves the state by O(1) of its scale, orders beyond."""
OUT_ULP = 4.0
"""The verify's GDP entry outputs against the decode steps', element-wise in bf16 ulps of their
row's scale (a row: one head's 64 value channels for a token, scaled by their max |decode|): each
entry is within 2 of the fp32 reference on the same inputs (`_compare_out` in both entries' tests),
so 4 of each other."""


def _standalone_env() -> dict[str, str]:
    """This process's environment, minus any enclosing launcher's identity (SLURM_*, PMIX_*,
    PMI_*, OMPI_*): a nested mpirun that inherited it would join that job."""
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), f"expected the repo root at {repo_root}"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH")) if p)
    return env


def spawn_ranks(out: str, checkpoint: str, draft: int) -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the four devices this run owns"
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    assert len(devices) >= WORLD_SIZE, f"dep4 runs on {WORLD_SIZE} devices; got {devices}"
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; the ranks are an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="dep4_mtp_rollback_")
    env = dict(
        _standalone_env(),
        CUDA_VISIBLE_DEVICES=",".join(devices[:WORLD_SIZE]),
        **{_PID_DIR_ENV: pid_dir},
    )
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(WORLD_SIZE)]
    if os.geteuid() == 0:
        command.insert(1, "--allow-run-as-root")
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    command += ["--out", out, "--checkpoint", checkpoint, "--draft", str(draft)]
    print(f"[launcher] {' '.join(command)}", flush=True)
    job = subprocess.Popen(command, env=env, start_new_session=True)
    deadline = time.monotonic() + DEADLINE_S
    while job.poll() is None:
        if time.monotonic() > deadline:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(job.pid, sig)
                except ProcessLookupError:
                    pass
                time.sleep(5)
            for pid_file in Path(pid_dir).glob("rank*.pid"):
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            raise AssertionError(f"the {WORLD_SIZE}-rank job did not finish in {DEADLINE_S}s")
        time.sleep(0.5)
    shutil.rmtree(pid_dir, ignore_errors=True)
    if job.returncode != 0:
        raise AssertionError(f"the {WORLD_SIZE}-rank job failed: mpirun exited {job.returncode}")


def _rank_body(out_path: str, checkpoint_path: str) -> None:
    import torch
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == WORLD_SIZE

    from types import SimpleNamespace

    from safetensors import safe_open

    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
        weights as W,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=WORLD_SIZE,
        rank=rank,
        gpus_per_node=WORLD_SIZE,
        tp_size=WORLD_SIZE,
        moe_ep_size=WORLD_SIZE,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    model_config = spec = None
    for turn in range(WORLD_SIZE):
        if turn == rank:
            model_config = ModelConfig.from_pretrained(
                checkpoint_path, trust_remote_code=True, mapping=mapping
            )
            spec = MTPDecodingConfig(max_draft_len=DRAFT)
            update_spec_config_from_model_config(spec, model_config.pretrained_config)
        comm.Barrier()
    assert spec.spec_dec_mode.is_mtp_eagle_one_model() and spec.tokens_per_gen_step == TOKENS
    config = model_config.pretrained_config
    # The target's shell sets this on the composite config (the text config derives it).
    config.num_hidden_layers = config.text_config.num_hidden_layers
    text = config.text_config

    # Layer 0's GDP mixer, laid out by dep4's weight-table transforms; its input norm; the
    # embedding table, which feeds the mixer realistic rows.
    with open(os.path.join(checkpoint_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{LAYER}."

    def tensor(key: str) -> torch.Tensor:
        with safe_open(os.path.join(checkpoint_path, weight_map[key]), "pt", device="cpu") as f:
            return f.get_tensor(key).cuda()

    mixer = M._Gdp(
        layer=LAYER,
        in_proj=tensor(prefix + "mixer.in_proj.weight").t(),
        conv=W._conv_taps(tensor(prefix + "mixer.conv1d.weight"), None),
        neg_a=W._neg_a(tensor(prefix + "mixer.A_log"), None),
        dt_bias=tensor(prefix + "mixer.dt_bias"),
        norm=tensor(prefix + "mixer.norm.weight"),
        out_proj=tensor(prefix + "mixer.out_proj.weight").t(),
    )
    input_norm = tensor(prefix + "input_layernorm.weight")
    embed = tensor("model.language_model.embed_tokens.weight")

    # The two targets' mixer code, without the rest of the model: `_gdp` reads only `_d`.
    decode, prefill = object.__new__(M.DecodeTarget), object.__new__(M.PrefillTarget)
    decode._d = prefill._d = M._dims(text)

    # The GDP entries' outputs, as `_gdp_generation` calls them: the last verify's
    # `[rows, 1 + D, heads, value]`, the last decode step's `[rows, heads, value]`.
    entries = dict(verify=M.gdp_verify, decode=M.gdp_recurrent)
    seen = {}

    def observed(name: str):
        def call(*args, **kwargs):
            out = entries[name](*args, **kwargs)
            seen[name] = out.clone()
            return out

        return call

    M.gdp_verify, M.gdp_recurrent = observed("verify"), observed("decode")

    def ulps(got: torch.Tensor, want: torch.Tensor, ulp: float) -> float:
        """max |got - want| over its row's max |want|, a row being the last dimension, in `ulp`s:
        the element score of the entries' own tests (`assert_within_ulp`)."""
        g = got.float().reshape(-1, got.shape[-1])
        w = want.float().reshape(-1, want.shape[-1])
        scale = w.abs().amax(-1, keepdim=True).clamp_min(1e-9)
        return float(((g - w).abs() / scale).max()) / ulp

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
            max_seq_len=9344,
            max_batch_size=MAX_BATCH,
            spec_config=spec,
            sparse_attention_config=None,
            max_num_tokens=8192,
            max_beam_width=1,
            kv_connector_manager=None,
            model_config=model_config,
            dtype=torch.bfloat16,
            max_cuda_graph_batch_size=MAX_BATCH,
        )
        assert manager.get_intermediate_ssm_states(LAYER).shape[:2] == (MAX_BATCH, TOKENS)
        assert manager.layer_offsets[config.num_hidden_layers] == 40, manager.layer_offsets
        ssm, conv = manager.get_ssm_states(LAYER), manager.get_conv_states(LAYER)

        # The step's requests, the attention-DP dummy and the padding dummy, slots from the manager.
        request_ids = list(range(1, 7))
        for rid in request_ids:
            assert manager.add_dummy_requests([rid], token_nums=[64]), rid
        padding_id = CUDA_GRAPH_DUMMY_REQUEST_ID - DRAFT
        for dummy in (ATTENTION_DP_DUMMY_REQUEST_ID, padding_id):
            added = manager.add_dummy_requests([dummy], is_gen=True, max_num_draft_tokens=DRAFT)
            assert added, dummy
        slot = dict(zip(request_ids, manager.get_state_indices(request_ids, [False] * 6)))
        adp_slot = manager.get_state_indices([ATTENTION_DP_DUMMY_REQUEST_ID], [False])[0]
        padding_slot = manager.get_state_indices([padding_id], [True])[0]
        assert len(set(slot.values()) | {adp_slot, padding_slot}) == 8, (
            slot,
            adp_slot,
            padding_slot,
        )

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
            """What `step_args` hands the mixer for these rows' slots: an eager step's, so the
            GEMMs are cuBLAS's."""
            return dict(
                graph=False,
                state_slots=torch.tensor(slots, dtype=torch.int32, device="cuda"),
                ssm_states={LAYER: ssm},
                conv_states={LAYER: conv},
                intermediate_ssm={LAYER: manager.get_intermediate_ssm_states(LAYER)},
                intermediate_conv={LAYER: manager.get_intermediate_conv_states(LAYER)},
                **context,
            )

        def commit(num_contexts: int, accepted: list, engine_slots: list) -> None:
            """The worker's commit: `update_mamba_states` with the engine's own slots (the dummies'
            included) and every row's accepted-token count."""
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
                decode._gdp(rows(g, len(slots)), mixer, step(slots))

        def decode_reference(x, real_slots: list, padded_slots: list, accepted: list):
            """MTP-off decode over the verify's tokens, one step per position, from the slots as
            they are: each real row's state after its accepted tokens, every step's GDP entry
            output `[rows, 1 + D, heads, value]`, and how far apart the states after 1, 2 and 3
            tokens sit in every row: whether the conv states all differ, and the least distance
            between two recurrent states in units of the band at the later position."""
            after, outs = [], []
            for j in range(TOKENS):
                decode._gdp(x[:, j].contiguous(), mixer, step(padded_slots))
                outs.append(seen["decode"])
                after.append(states(real_slots))
            pairs = [(i, j) for i in range(TOKENS) for j in range(i + 1, TOKENS)]
            conv_distinct = all(
                not torch.equal(after[i][1][b], after[j][1][b])
                for b in range(len(real_slots))
                for i, j in pairs
            )
            separation = min(
                ulps(after[i][0][b], after[j][0][b], ULP_FP32) / (STATE_ULP * math.sqrt(j + 1))
                for b in range(len(real_slots))
                for i, j in pairs
            )
            taken = [
                (after[accepted[b] - 1][0][b], after[accepted[b] - 1][1][b])
                for b in range(len(real_slots))
            ]
            return taken, torch.stack(outs, dim=1), conv_distinct, separation

        def check_states(real_slots: list, expected, accepted: list, what: str) -> tuple:
            """Each real row's slot against `expected`: the conv state bitwise, the recurrent
            state within its band; the failures, and the recurrent state's largest distance in
            units of its band."""
            got = states(real_slots)
            failures, worst = [], 0.0
            for b, (want_ssm, want_conv) in enumerate(expected):
                band = STATE_ULP * math.sqrt(accepted[b])
                found = ulps(got[0][b], want_ssm, ULP_FP32)
                worst = max(worst, found / band)
                if found > band:
                    failures.append(
                        f"{what}: row {b} recurrent state {found:.3g} fp32 ulp off, band {band:.3g}"
                    )
                if not torch.equal(got[1][b], want_conv):
                    failures.append(f"{what}: row {b} conv state differs")
            return failures, worst

        g = torch.Generator(device="cuda").manual_seed(4404 + 97 * rank + (kind == "v1"))

        with torch.inference_mode():
            # --- padded: 5 requests verifying, padded to a graph batch of 8 ---
            real = [slot[r] for r in request_ids[:5]]
            padded = real + [PAD_SLOT_ID] * 3
            engine = real + [padding_slot] * 3
            prime(g, real, steps=6)
            failures, distinct_all = [], True
            state, out_ulp, separation = 0.0, 0.0, float("inf")
            for accepted in _ROUNDS[TOKENS]:
                start = snapshot()
                x = rows(g, len(padded) * TOKENS).view(len(padded), TOKENS, -1)
                decode._gdp(x.reshape(-1, x.shape[-1]), mixer, step(padded))
                out = seen["verify"]
                if changed(start):
                    failures.append(f"padded {accepted}: the verify wrote slots {changed(start)}")
                commit(0, accepted + [1, 2, 3], engine)
                committed = snapshot()
                outside = changed(start) - (set(real) | {padding_slot})
                if outside:
                    failures.append(f"padded {accepted}: the commit wrote slots {outside}")
                restore(start)
                expected, ref_out, conv_distinct, apart = decode_reference(
                    x, real, padded, accepted
                )
                distinct_all &= conv_distinct and apart > 1
                separation = min(separation, apart)
                restore(committed)
                found, worst = check_states(real, expected, accepted, f"padded {accepted}")
                failures += found
                state = max(state, worst)
                off = ulps(out[:5], ref_out[:5], ULP_BF16)
                out_ulp = max(out_ulp, off)
                if off > OUT_ULP:
                    failures.append(
                        f"padded {accepted}: the verify's GDP outputs {off:.3g} bf16 ulp off the "
                        "decode steps'"
                    )
            records.append(
                dict(
                    rank=rank,
                    manager=type(manager).__name__,
                    kind=kind,
                    case="padded",
                    draft=DRAFT,
                    failures=failures,
                    distinct=distinct_all,
                    accepted=sorted({n for counts in _ROUNDS[TOKENS] for n in counts}),
                    state=state,
                    out=out_ulp,
                    separation=separation,
                )
            )

            # --- idle: the attention-DP dummy padded to a graph batch of 4; and the dummy alone ---
            for case, engine in (
                ("idle", [adp_slot] + [padding_slot] * 3),
                ("dummy alone", [adp_slot]),
            ):
                start = snapshot()
                placeholders = [PAD_SLOT_ID] * len(engine)
                x = rows(g, len(engine) * TOKENS)
                decode._gdp(x, mixer, step(placeholders))
                failures = []
                if changed(start):
                    failures.append(f"{case}: the verify wrote slots {changed(start)}")
                commit(0, [3, 2, 1, 2][: len(engine)], engine)
                outside = changed(start) - set(engine)
                if outside:
                    failures.append(f"{case}: the commit wrote slots {outside}")
                restore(start)
                records.append(
                    dict(
                        rank=rank,
                        manager=type(manager).__name__,
                        kind=kind,
                        case=case,
                        draft=DRAFT,
                        failures=failures,
                        distinct=True,
                        accepted=[],
                    )
                )

            # --- mixed: a fresh 37-token prompt beside 3 verify rows ---
            context_slot, real = slot[request_ids[5]], [slot[r] for r in request_ids[:3]]
            prime(g, real, steps=4)
            accepted = _MIXED[TOKENS]
            start = snapshot()
            context = rows(g, 37)
            x = rows(g, len(real) * TOKENS).view(len(real), TOKENS, -1)
            starts = torch.tensor([0, 37], dtype=torch.int32, device="cuda")
            prefill._gdp(
                torch.cat([context, x.reshape(-1, x.shape[-1])]),
                mixer,
                step(
                    [context_slot] + real,
                    num_rows=1 + len(real),
                    num_contexts=1,
                    num_ctx_tokens=37,
                    max_ctx_len=37,
                    query_start_loc=starts,
                    has_initial_state=torch.zeros(1, dtype=torch.bool, device="cuda"),
                ),
            )
            out = seen["verify"]
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
            restore(start)
            expected, ref_out, conv_distinct, apart = decode_reference(x, real, real, accepted)
            restore(committed)
            found, state = check_states(real, expected, accepted, f"mixed {accepted}")
            failures += found
            out_ulp = ulps(out, ref_out, ULP_BF16)
            if out_ulp > OUT_ULP:
                failures.append(
                    f"mixed: the verify's GDP outputs {out_ulp:.3g} bf16 ulp off the decode steps'"
                )
            records.append(
                dict(
                    rank=rank,
                    manager=type(manager).__name__,
                    kind=kind,
                    case="mixed",
                    draft=DRAFT,
                    failures=failures,
                    distinct=conv_distinct and apart > 1,
                    accepted=accepted,
                    state=state,
                    out=out_ulp,
                    separation=apart,
                )
            )
        manager.shutdown()
        comm.Barrier()
    M.gdp_verify, M.gdp_recurrent = entries["verify"], entries["decode"]

    gathered = comm.gather(records, root=0)
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump([r for per in gathered for r in per], f)
        print(f"[job] wrote {out_path}", flush=True)
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
        os._exit(13)
    sys.stdout.flush()
    MPI.Finalize()
    os._exit(0)


def _parse(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(_WORKER_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", required=True, help="where rank 0 writes the records (JSON)")
    parser.add_argument(
        "--draft", type=int, default=DRAFT, choices=DRAFTS, help="the draft length (default: 2)"
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.environ.get("LLM_MODELS_ROOT", ""), "nemotron4-nano-preview"),
        help="the checkpoint directory (default: $LLM_MODELS_ROOT/nemotron4-nano-preview)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse(sys.argv[1:])
    # Every function reads the run's draft length off these module globals.
    DRAFT, TOKENS = args.draft, 1 + args.draft
    if args.rank_worker:
        sys.exit(_run_one_rank(args.out, args.checkpoint))
    spawn_ranks(args.out, args.checkpoint, args.draft)
