# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4-rank job: an MTP verify step of the nemotron4 tp4 target's GDP mixer rolls back to the state
MTP-off decode reaches, on every rank's own cache manager.

Under one-model MTP-Eagle a generation row carries its last token and D drafts. Each rank's GDP
mixer runs the row's 1 + D tokens on its 6 of the 24 heads from the row's state slot without
writing it, and stores the state after every token in the hybrid cache manager's per-position
buffers; after acceptance the engine's worker commits the last accepted token's state into the
slot (`update_mamba_states`). A rejected draft must leave no trace: the committed slot has to be
the state MTP-off decode reaches over the accepted tokens alone -- the conv state bit for bit, the
recurrent state within the band its two entries are certified to (`state_band`).

This script is its own launcher, as the tp4 decode-invariance job is: run it plainly and it starts
one MPI rank per device named in CUDA_VISIBLE_DEVICES under ``mpirun``, with a deadline it enforces
by killing the job.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _tp4_mtp_rollback_ranks.py --report <json> [--draft 2|5]

What every rank builds, as the engine builds it for its rank under ``MTPDecodingConfig(max_draft_len
=D)`` at a draft length D tp4 runs (`DRAFTS`): ``ModelConfig`` off the checkpoint with the rank's
tp4 mapping (tp 4, MoE tp 4, no attention DP); the TP group's all-reduce workspace (collective, as
the target's ``post_load_weights`` builds it); and the trunk's hybrid cache manager for that mapping
(``_create_kv_cache_manager`` over the 44 trunk layers, the MTP layer's KV living in a separate
draft manager on tp4), with the verify's per-position buffers -- both managers the engine offers,
``MambaHybridCacheManagerV2`` (the default) and ``CppMambaHybridCacheManager``. The mixer is layer
0's, the rank's slice of the checkpoint's weights cut and laid out by tp4's own weight table, run
through the targets' own code (``DecodeTarget._gdp`` for a decode or verify step,
``PrefillTarget._gdp`` for a mixed one), which return the heads' ``out_proj`` partial for the
caller's fused all-reduce. The commit is the manager's ``update_mamba_states`` with the engine's own
slots, as the worker calls it. Every rank feeds the same rows, as the engine's ranks do.

The cases, on both managers:

* ``padded``: 5 requests verifying 1 + D tokens, padded to a graph batch of 8 with the padding
  dummy's rows (``CUDA_GRAPH_DUMMY_REQUEST_ID - D``), two verify -> commit rounds whose accepted
  counts cover 1..1 + D (`_ROUNDS`), through DecodeTarget.
* ``mixed``: a fresh 37-token prompt (a context row) beside 3 verify rows, through PrefillTarget.

In both the reference is MTP-off decode of the verify rows' own in_proj output -- the target's own
GEMM at the verify's row count (`_mm` for DecodeTarget, `torch_mm` for PrefillTarget) -- one token
per step through the decode entries (``_gdp_rows``) from the slots the verify read. The kernel the
in_proj runs follows the row count, so a decode step's own in_proj can round otherwise than the
verify's; fed the verify's rows, the reference differs from the verify in the GDP entries alone.

Per case: every real row's committed conv state equals its slot after `accepted` MTP-off decode
steps over the same tokens, bit for bit, and its recurrent state lies within `state_band` of it;
the states after each of the 1 + D tokens lie farther apart than that band, so a commit from the
wrong position cannot pass; the verify writes no slot; the commit writes only the step's rows'
engine slots (the padding dummy's included, whose stale snapshot it copies) and never a context
row's; and the GDP verify entry's output rows lie within `OUT_BAND` of the decode entry's. Rank 0
writes every rank's records to `--report` as JSON. Not a pytest module: everything that imports
`tensorrt_llm` is imported inside the rank body (an MPI-initialized process cannot start `mpirun`).
"""

from __future__ import annotations

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
_PID_DIR_ENV = "MODELING_V2_TP4_MTP_ROLLBACK_PID_DIR"
_REPORT_ENV = "MODELING_V2_TP4_MTP_ROLLBACK_REPORT"
_DRAFT_ENV = "MODELING_V2_TP4_MTP_ROLLBACK_DRAFT"
#: Building the managers and the workspace takes ~1 min, the cases seconds.
DEADLINE_S = 1200
WORLD_SIZE = 4
#: The draft lengths tp4 runs: the checkpoint's trained depth, and five. One run checks one,
#: `--draft` (2 by default); a verify row carries 1 + DRAFT tokens.
DRAFTS = (2, 5)
DRAFT = 2
TOKENS = 1 + DRAFT
#: Per T = 1 + D, the accepted counts of the padded case's 5 real rows in each of its two verify
#: rounds -- every count 1..T in each round at T = 3, and at T = 6 one more count than rows in the
#: two together -- and of the mixed case's 3 verify rows.
_ROUNDS = {3: ([3, 1, 2, 3, 1], [1, 2, 3, 2, 3]), 6: ([6, 1, 3, 5, 2], [4, 6, 2, 1, 5])}
_MIXED = {3: [2, 3, 1], 6: [6, 4, 1]}
MAX_BATCH = 8
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192
#: A `linear_attention` block; every GDP mixer of the model is the same mixer.
LAYER = 0
MANAGERS = ("v2", "v1")

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4"

_BF16_ULP = 2.0**-8
_FP32_ULP = 2.0**-23

# The bands. The verify and its reference read the same q, k, v, beta and decay: the same in_proj
# rows and gates, and the conv verify entry takes the conv decode entry's arithmetic token by token
# (its note). A conv state is the window of the last inputs, copied rather than computed, so the
# committed conv state is the decode steps' bit for bit. The recurrent state and the output rows
# are computed, by gdp_verify on one side and by t + 1 decode steps of gdp_replay (no draft caches)
# on the other, each certified against the recurrence's reference: gdp_verify's state after token t
# within 8 sqrt(t + 1) fp32 ulp of its row's scale element-wise and sqrt(t + 1) of relative RMS
# (`test_modeling_v2_gdp_verify.py`), gdp_replay's within 8 sqrt(t + 1) and 1.5 sqrt(t + 1)
# (`test_modeling_v2_gdp_replay.py`), a token's output within 2 bf16 ulp and 0.25 on both. So the
# two sides lie within the sum of their bands of each other. A commit from the wrong position
# moves the state by O(1) of its scale, of the order of 2**23 fp32 ulp, orders beyond; each case
# measures how far (`apart`).
#: The output rows' band: (element, relative RMS) in bf16 ulp of each row's largest |decode|.
OUT_BAND = (4.0, 0.5)


def state_band(accepted: int) -> tuple[float, float]:
    """The committed recurrent state's band after `accepted` tokens: (element, relative RMS) in
    fp32 ulp of each row's largest |decode|, the sum of the two sides' bands at t = accepted - 1:
    8 sqrt(t + 1) + 8 sqrt(t + 1) and sqrt(t + 1) + 1.5 sqrt(t + 1)."""
    depth = math.sqrt(accepted)
    return 16.0 * depth, 2.5 * depth


def ulps(got, want, ulp: float) -> tuple[float, float]:
    """`got` against `want` as `assert_within_ulp` scores it, in units of `ulp`: the largest
    |got - want| over its row's largest |want| (a row is the last dim), and the relative RMS."""
    got = got.float().reshape(-1, got.shape[-1])
    want = want.float().reshape(-1, want.shape[-1])
    diff = got - want
    scale = want.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = float((diff.abs() / scale).max()) / ulp
    rms = float(diff.pow(2).mean().sqrt() / want.pow(2).mean().sqrt().clamp_min(1e-9)) / ulp
    return element, rms


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

    # The mixer as both targets hold it (`_layer_views`): every GEMM operand the `.t()` view
    # `cublas_mm` reads.
    mixer = M._Gdp(
        layer=LAYER,
        in_proj=w["gdp_in"].t(),
        conv=w["gdp_conv"],
        neg_a=w["gdp_neg_a"],
        dt_bias=w["gdp_dt_bias"],
        norm=w["gdp_norm"],
        out_proj=w["gdp_out"].t(),
    )

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

    def verified(gdp, *args) -> torch.Tensor:
        """Run `gdp(*args)`, a target's `_gdp` over a step holding verify rows, and return what
        `gdp_verify` returned inside it: the verify rows' GDP output, `[rows, T, heads, value]`."""
        seen, entry = [], M.gdp_verify

        def recorded(*inputs, **kwargs):
            out = entry(*inputs, **kwargs)
            seen.append(out.clone())
            return out

        M.gdp_verify = recorded
        try:
            gdp(*args)
        finally:
            M.gdp_verify = entry
        assert len(seen) == 1, f"gdp_verify ran {len(seen)} times in one step"
        return seen[0]

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
                decode._gdp(rows(g, len(slots)), mixer, step(slots))

        def decode_tokens(proj: torch.Tensor, slots: list, real_slots: list) -> tuple:
            """MTP-off decode of the verify rows' own in_proj output `proj`, `[rows, T, 4632]`,
            one token per step through the decode entries (`_gdp_rows`) from the slots as they
            are: each token's GDP output rows, and the real rows' states after it."""
            outs, after = [], []
            for j in range(TOKENS):
                token = proj[:, j].contiguous()
                beta, decay = M.gdp_gates(
                    token[:, dims.beta], token[:, dims.decay], mixer.neg_a, mixer.dt_bias
                )
                args = step(slots)
                outs.append(M._gdp_rows(token, beta, decay, mixer, args["state_slots"], args, dims))
                after.append(states(real_slots))
            return outs, after

        def apart(after: list, num_real: int) -> float:
            """How far apart the states after each of the 1 + D tokens lie, the least over the
            rows and pairs: the recurrent states in fp32 ulp of the row scale (`ulps`), 0 where a
            pair's conv states are bitwise equal. A commit from the wrong position lands that far
            from the right state."""
            least = math.inf
            for b in range(num_real):
                for i in range(TOKENS):
                    for j in range(i + 1, TOKENS):
                        if torch.equal(after[i][1][b], after[j][1][b]):
                            return 0.0
                        least = min(least, ulps(after[i][0][b], after[j][0][b], _FP32_ULP)[0])
            return least

        def check_states(real_slots: list, accepted: list, after: list, what: str) -> tuple:
            """Each real row's slot against its state after its `accepted` decode steps: the conv
            state bit for bit, the recurrent state within `state_band`. Returns the failures and
            the largest recurrent-state deviation, (element, relative RMS) in fp32 ulp."""
            got = states(real_slots)
            failures, worst = [], (0.0, 0.0)
            for b, n in enumerate(accepted[: len(real_slots)]):
                if not torch.equal(got[1][b], after[n - 1][1][b]):
                    failures.append(f"{what}: row {b} conv state differs")
                element, rms = ulps(got[0][b], after[n - 1][0][b], _FP32_ULP)
                band = state_band(n)
                if element > band[0] or rms > band[1]:
                    failures.append(
                        f"{what}: row {b} recurrent state {element:.3g} / {rms:.3g} fp32 ulp off, "
                        f"past {band[0]:.3g} / {band[1]:.3g}"
                    )
                worst = (max(worst[0], element), max(worst[1], rms))
            return failures, worst

        def check_out(got: torch.Tensor, want: torch.Tensor, what: str) -> tuple:
            """The GDP verify entry's output rows against the decode entry's, within `OUT_BAND`.
            Returns the failures and the deviation, (element, relative RMS) in bf16 ulp."""
            element, rms = ulps(got, want, _BF16_ULP)
            failures = []
            if element > OUT_BAND[0] or rms > OUT_BAND[1]:
                failures.append(
                    f"{what}: output rows {element:.3g} / {rms:.3g} bf16 ulp off, past "
                    f"{OUT_BAND[0]} / {OUT_BAND[1]}"
                )
            return failures, (element, rms)

        def record(case: str, failures: list, separation: float, state_ulp, out_ulp) -> None:
            records.append(
                dict(
                    rank=rank,
                    manager=type(manager).__name__,
                    kind=kind,
                    case=case,
                    failures=failures,
                    distinct=separation > state_band(TOKENS)[0],
                    separation=separation,
                    state_ulp=state_ulp,
                    out_ulp=out_ulp,
                )
            )
            log(
                f"{kind} {case}: {len(failures)} failures; recurrent state {state_ulp[0]:.3g} / "
                f"{state_ulp[1]:.3g} fp32 ulp from decode's, output rows {out_ulp[0]:.3g} / "
                f"{out_ulp[1]:.3g} bf16 ulp; the states after 1, 2 and 3 tokens at least "
                f"{separation:.3g} fp32 ulp apart"
            )

        # The same seed on every rank: the ranks feed the same rows, as the engine's ranks do.
        g = torch.Generator(device="cuda").manual_seed(4404 + (kind == "v1"))

        with torch.inference_mode():
            # --- padded: 5 requests verifying, padded to a graph batch of 8 ---
            real = [slot[r] for r in request_ids[:5]]
            padded = real + [PAD_SLOT_ID] * 3
            engine = real + [padding_slot] * 3
            prime(g, real, steps=6)
            failures, separation, state_ulp, out_ulp = [], math.inf, (0.0, 0.0), (0.0, 0.0)
            for accepted in _ROUNDS[TOKENS]:
                start = snapshot()
                x = rows(g, len(padded) * TOKENS)
                out = verified(decode._gdp, x, mixer, step(padded))
                torch.cuda.synchronize()
                if changed(start):
                    failures.append(f"padded {accepted}: the verify wrote slots {changed(start)}")
                commit(0, accepted + [1, 2, 3], engine)
                committed = snapshot()
                outside = changed(start) - (set(real) | {padding_slot})
                if outside:
                    failures.append(f"padded {accepted}: the commit wrote slots {outside}")
                # MTP-off decode over the verify rows' own in_proj output -- the same cuBLAS call
                # on the same rows as DecodeTarget's -- one token per step from the start.
                restore(start)
                proj = M._mm(x, mixer.in_proj).view(len(padded), TOKENS, -1)
                outs, after = decode_tokens(proj, padded, real)
                separation = min(separation, apart(after, len(real)))
                restore(committed)
                found, s = check_states(real, accepted, after, f"padded {accepted}")
                failures += found
                decoded = torch.stack(outs, dim=1)[: len(real)]
                found, o = check_out(out[: len(real)], decoded, f"padded {accepted}")
                failures += found
                state_ulp = (max(state_ulp[0], s[0]), max(state_ulp[1], s[1]))
                out_ulp = (max(out_ulp[0], o[0]), max(out_ulp[1], o[1]))
            record("padded", failures, separation, state_ulp, out_ulp)

            # --- mixed: a fresh 37-token prompt beside 3 verify rows ---
            context_slot, real = slot[request_ids[5]], [slot[r] for r in request_ids[:3]]
            prime(g, real, steps=4)
            accepted = _MIXED[TOKENS]
            start = snapshot()
            context = rows(g, 37)
            x = rows(g, len(real) * TOKENS)
            batch_rows = torch.cat([context, x])
            starts = torch.tensor([0, 37], dtype=torch.int32, device="cuda")
            out = verified(
                prefill._gdp,
                batch_rows,
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
            # The verify rows' own in_proj output -- `torch_mm` at the mixed step's row count, as
            # PrefillTarget computes it -- decoded one token per step from the start.
            restore(start)
            proj = M.torch_mm(batch_rows, mixer.in_proj)[37:].view(len(real), TOKENS, -1)
            outs, after = decode_tokens(proj, real, real)
            torch.cuda.synchronize()
            separation = apart(after, len(real))
            restore(committed)
            found, state_ulp = check_states(real, accepted, after, f"mixed {accepted}")
            failures += found
            found, out_ulp = check_out(out, torch.stack(outs, dim=1), f"mixed {accepted}")
            failures += found
            record("mixed", failures, separation, state_ulp, out_ulp)
        manager.shutdown()
        comm.Barrier()
    return records


def _run_one_rank() -> int:
    """Body of one MPI rank: a rank whose code raises exits hard, and mpirun ends the job."""
    global DRAFT, TOKENS
    DRAFT = int(os.environ.get(_DRAFT_ENV, DRAFT))
    TOKENS = 1 + DRAFT
    assert DRAFT in DRAFTS, f"tp4 runs {DRAFTS} drafts, not {DRAFT}"
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
        report = dict(ranks=WORLD_SIZE, draft=DRAFT, records=[r for per in gathered for r in per])
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


def _spawn_ranks(report: Path, draft: int) -> None:
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
    env = dict(
        _standalone_env(),
        **{_PID_DIR_ENV: pid_dir, _REPORT_ENV: str(report), _DRAFT_ENV: str(draft)},
    )
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
    assert "--report" in sys.argv, (
        "usage: _tp4_mtp_rollback_ranks.py --report <json> [--draft <max_draft_len>]"
    )
    draft = int(sys.argv[sys.argv.index("--draft") + 1]) if "--draft" in sys.argv else DRAFT
    _spawn_ranks(Path(sys.argv[sys.argv.index("--report") + 1]).resolve(), draft)
    print("OK", flush=True)
