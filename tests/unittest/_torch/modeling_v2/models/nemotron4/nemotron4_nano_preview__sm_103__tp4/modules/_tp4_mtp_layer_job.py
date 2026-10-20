# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 4-rank job behind `test_modeling_v2_tp4_mtp_layer.py`: the module accuracy of the tp4
target's MTP layer on every rank of one node, against the plain-torch reference.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _tp4_mtp_layer_job.py --out <json> [--checkpoint <dir>]
        [--trunk-cache <file>]

Run plainly, this script is its own launcher: it starts one MPI rank per device under `mpirun`, as
a tp4 serving engine starts its ranks, with a deadline it enforces by killing the job, because a
collective that breaks hangs rather than raising.

What every rank builds, as the engine builds it for its rank under ``MTPDecodingConfig(max_draft_len
=2)`` (the spec config resolved against the checkpoint: one-model MTP-Eagle over the one repeated
MTP layer): the tp4 shell by its own constructor on the meta device -- so `DraftModel`, `MTPLayer`,
`shared_head` and the spec worker are the served engine's -- with the MTP layer's rows at the
rank's slices (its 4 of the 16 query heads, a quarter of its shared expert's and of every routed
expert's intermediate width), the embedding and the whole lm_head filled from the checkpoint
through tp4's own weight table (`weights.MODEL_WEIGHTS.manifest`, cuts and transforms as `load`
applies them), and `post_load_weights`, which builds the TP group's all-reduce workspace; the
separate draft KV manager tp4 keeps the MTP layer's KV in (global layer 44 at index 0); and the
engine's TRTLLM attention metadata over it, filled per step as the runner fills it (no
recurrent-state metadata: the MTP layer keeps none).

What runs is `_nemotron4_mtp_layer`'s campaign (its docstring says which cells, references and
gates), every rank the same steps in lockstep, as a tp4 engine's ranks run every request: each
draft step's two all-reduces (the attention's `o_proj` partials; the routed experts' latent and
the shared expert's partials) meet across the ranks. Three things are tp4's own:

* the per-op references of the ops a rank runs on its slice -- the qkv GEMM over its query heads'
  rows, the attention of those heads, the `o_proj` partial over their columns -- and the residual
  add, which reads the all-reduced `o_proj`;
* the TP term: on every MTP-layer call, the layer's output against a twin of the same call with
  each all-reduce replaced by the single full-K call (the ranks' attention outputs gathered and
  projected by the whole `o_proj`, the MoE run with every expert and the shared expert whole --
  tp1's composition, on this call's own inputs and routing): what rounding each rank's partial to
  bf16 before the fp32 sum changes;
* the twin against `R_t`, which holds the twin to what the tp1 target's MTP layer is held to.

Rank 0 writes every rank's records to `--out` as JSON. Not a pytest module: everything that
imports `tensorrt_llm` is imported inside the rank body (an MPI-initialized process cannot start
`mpirun`).
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
from pathlib import Path

_WORKER_FLAG = "--rank-worker"
_PID_DIR_ENV = "NEMOTRON4_TP4_MTP_LAYER_PID_DIR"
DEADLINE_S = 5400
"""Ceiling on the whole job: the trunk's hidden states (when no cache holds them), the references,
the campaign, the rivals."""
WORLD_SIZE = 4
MAX_BATCH = 520
"""Requests resident at once: the campaign's pool of 512, and room to spare."""


def _standalone_env() -> dict[str, str]:
    """This process's environment, minus any enclosing launcher's identity (SLURM_*, PMIX_*,
    PMI_*, OMPI_*): a nested mpirun that inherited it would join that job."""
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env["TLLM_DISABLE_MPI"] = "0"
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), f"expected the repo root at {repo_root}"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH")) if p)
    return env


def spawn_ranks(out: str, checkpoint: str, trunk_cache: str) -> None:
    """Start one MPI rank per claimed device under mpirun; kill the job on failure or deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the four devices this run owns"
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    assert len(devices) >= WORLD_SIZE, f"tp4 runs on {WORLD_SIZE} devices; got {devices}"
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; the ranks are an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="tp4_mtp_layer_")
    env = dict(
        _standalone_env(),
        CUDA_VISIBLE_DEVICES=",".join(devices[:WORLD_SIZE]),
        **{_PID_DIR_ENV: pid_dir},
    )
    command = [mpirun, "--oversubscribe", "--bind-to", "none", "-n", str(WORLD_SIZE)]
    if os.geteuid() == 0:
        command.insert(1, "--allow-run-as-root")
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    command += ["--out", out, "--checkpoint", checkpoint, "--trunk-cache", trunk_cache]
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


def _rank_body(out_path: str, checkpoint_path: str, trunk_cache: str) -> None:
    import torch
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == WORLD_SIZE

    # Spawned by mpirun rather than collected by pytest: the family's helpers and `_validating`
    # are reached the plain way.
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here.parents[1]), str(here.parents[3])]
    os.environ["NEMOTRON4_MTP_TRUNK_CACHE"] = trunk_cache

    from types import SimpleNamespace

    import _nemotron4_mtp_layer as H
    import _nemotron4_prompts as P
    import _nemotron4_reference as R
    from torch import nn

    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4 import (  # noqa: E501
        weights as W,
    )
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm._torch.speculative.utils import resolve_mtp_checkpoint_source
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    started = time.monotonic()

    def say(message: str) -> None:
        if rank == 0:
            print(f"[job {time.monotonic() - started:7.1f}s] {message}", flush=True)

    mapping = Mapping(
        world_size=WORLD_SIZE,
        rank=rank,
        gpus_per_node=WORLD_SIZE,
        tp_size=WORLD_SIZE,
        moe_tp_size=WORLD_SIZE,
        moe_ep_size=1,
    )
    # One rank at a time: the remote-code config is copied into the shared modules cache.
    config = prompts = None
    for turn in range(WORLD_SIZE):
        if turn == rank:
            base = ModelConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
            validate_and_set_mamba_ssm_cache_dtype(base, "auto")
            composite = base.pretrained_config
            # The shell sets this on the composite config (the text config derives it).
            composite.num_hidden_layers = composite.text_config.num_hidden_layers
            spec = MTPDecodingConfig(max_draft_len=H.DRAFT)
            resolve_mtp_checkpoint_source(spec, checkpoint_path)
            update_spec_config_from_model_config(spec, composite)
            config = dataclasses.replace(
                base, mapping=mapping, spec_config=spec, max_num_tokens=H.MAX_NUM_TOKENS
            )
            prompts = P.prompt_set(P.load_tokenizer(checkpoint_path))
        comm.Barrier()
    spec = config.spec_config
    assert spec.spec_dec_mode.is_mtp_eagle_one_model() and spec.max_draft_len == H.DRAFT

    checkpoint = R.Checkpoint(checkpoint_path)
    text_config = checkpoint.config
    # The trunk's hidden states: rank 0 computes them (or reads the cache), the others read them.
    hidden = H.trunk_hidden(checkpoint, prompts, log=say) if rank == 0 else None
    comm.Barrier()
    if rank != 0:
        hidden = H.trunk_hidden(checkpoint, prompts, log=say)
    # Every rank the same requests in the same order: tp4 runs every request on every rank.
    sequences = [
        H.Sequence(p.name, list(p.token_ids), torch.tensor(p.token_ids, device="cuda"), h.cuda())
        for p, h in zip(prompts, hidden)
    ]
    say(f"trunk hidden states of {sum(len(s.ids) for s in sequences)} tokens on every rank")

    # The tp4 shell under the MTP config, every parameter declared on meta by its own
    # constructor; the MTP layer's rows, the embedding and the lm_head filled through the weight
    # table's manifest, each source's cut (the rank's slice) and transform applied.
    with torch.device("meta"):
        model = M.ModelingV2Nemotron4NanoPreviewSm103Tp4(config)
    core = model.model
    table = W.MODEL_WEIGHTS
    d = table.dims(core)
    assert d.mtp_layers == 1 and core.mtp_layer == d.num_layers, (d.mtp_layers, core.mtp_layer)
    assert (d.tp, d.tp_rank) == (WORLD_SIZE, rank), (d.tp, d.tp_rank)
    mtp_rows = f"l{d.num_layers}_"
    filled = 0
    for key, sources in table.manifest(core).items():
        if not (key.startswith(mtp_rows) or key == "embed"):
            continue
        param = torch.empty(core.w[key].shape, dtype=core.w[key].dtype, device="cuda")
        for ckpt_key, index, rows, transform in sources:
            src = checkpoint.tensor(ckpt_key)
            if rows is not None:
                src = src[rows]
            if transform is not None:
                src = transform(src, d)
            dst = param if index is None else param[index]
            assert (dst.shape, dst.dtype) == (src.shape, src.dtype), (
                ckpt_key,
                dst.shape,
                src.shape,
            )
            dst.copy_(src)
            filled += 1
        core.w[key] = nn.Parameter(param, requires_grad=False)
    model.lm_head.weight = nn.Parameter(checkpoint.tensor(R.LM_HEAD_KEY), requires_grad=False)
    # Collective: the TP group's all-reduce workspace, then both targets.
    core.post_load_weights()
    assert model.spec_worker is not None and model.draft_model is not None
    say(f"tp4 MTP layer built from {filled + 1} checkpoint tensor cuts on every rank")

    # The separate draft KV manager tp4 keeps the MTP layer's KV in, as the engine builds it.
    num_layers = config.pretrained_config.num_hidden_layers
    kv_cache_config = KvCacheConfig(
        max_tokens=H.KV_TOKENS, enable_block_reuse=False, free_gpu_memory_fraction=0.9
    )
    manager = _create_kv_cache_manager(
        None,
        get_kv_cache_manager_cls(config, kv_cache_config),
        mapping,
        kv_cache_config,
        tokens_per_block=H.TOKENS_PER_BLOCK,
        max_seq_len=H.MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH,
        spec_config=spec,
        sparse_attention_config=None,
        max_num_tokens=H.MAX_NUM_TOKENS,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=config,
        dtype=torch.bfloat16,
        max_cuda_graph_batch_size=MAX_BATCH,
        is_draft=True,
        layer_mask=[False] * num_layers + [True],
        num_layers=1,
    )
    assert manager.layer_offsets[core.mtp_layer] == 0, manager.layer_offsets
    md = build_attention_metadata(
        config,
        max_batch_size=MAX_BATCH,
        max_num_tokens=H.MAX_NUM_TOKENS,
        max_beam_width=1,
        attention_backend=get_attention_backend("TRTLLM"),
        attention_runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=True,
            cache_reuse=False,
            has_speculative_draft_tokens=True,
            chunk_size=H.MAX_NUM_TOKENS,
        ),
        mapping=mapping,
        cache_indirection=None,
        kv_cache_manager=manager,
    )
    # The MTP layer keeps no recurrent state, so no state metadata is prepared.
    md.mamba_metadata = False

    embed = checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")
    weights = checkpoint.mtp_layer()
    lm_head = checkpoint.tensor(R.LM_HEAD_KEY)
    seqrefs = [H.sequence_reference(s, embed, weights, text_config) for s in sequences]
    say("references of every sequence on every rank")

    # --- what a rank runs on its slice, and the twin of every call ---
    heads = text_config.num_attention_heads // WORLD_SIZE
    q_width = heads * text_config.head_dim
    attn_w = R.sub(weights, "block.self_attn.")
    block_w = R.sub(weights, "block.")
    moe_w = R.sub(weights, "block.moe.")
    qkv_rank = torch.cat(
        [
            attn_w["q_proj.weight"][rank * q_width : (rank + 1) * q_width],
            attn_w["k_proj.weight"],
            attn_w["v_proj.weight"],
        ]
    )
    o_rank = attn_w["o_proj.weight"][:, rank * q_width : (rank + 1) * q_width]
    eps = text_config.rms_norm_eps
    clamp = text_config.activation_clamp_value
    tp: dict = {}

    class Observer(H.MtpLayerObserver):
        """The harness's observer, plus the outputs of the MTP layer's two all-reduces: the
        `o_proj` partials' sum (``o_sum``) and the MoE's latent and shared partials' (``moe_sum``).

        A decode step's GEMMs of a few rows run the low-M entry `lowm_entry` names rather than
        `cublas_mm`, and a prompt step's run `torch_mm`; each is recorded by its weight operand
        as the harness records a `cublas_mm` call."""

        def install(self, mp) -> None:
            super().install(mp)
            allreduce, stage = self.M.allreduce, self._stage
            lowm_entry, torch_mm = self.M.lowm_entry, self.M.torch_mm

            def record(mat_a, mat_b, out) -> None:
                role = self.gemms.get(mat_b.data_ptr())
                if role is not None:
                    stage[role] = out.clone()
                elif mat_b.data_ptr() == self.o_proj:
                    stage["attn"], stage["o"] = mat_a.clone(), out.clone()

            def observed_torch_mm(mat_a, mat_b, *args, **kwargs):
                out = torch_mm(mat_a, mat_b, *args, **kwargs)
                record(mat_a, mat_b, out)
                return out

            def observed_allreduce(input, comm_buffer, buffer_flags):
                out = allreduce(input, comm_buffer, buffer_flags)
                stage["o_sum" if "o_sum" not in stage else "moe_sum"] = out.clone()
                return out

            def observed_lowm_entry(m, k, n):
                entry = lowm_entry(m, k, n)
                if entry is None:
                    return None

                def call(mat_a, mat_b, *args, **kwargs):
                    out = entry(mat_a, mat_b, *args, **kwargs)
                    record(mat_a, mat_b, out)
                    return out

                return call

            mp.setattr(self.M, "allreduce", observed_allreduce)
            mp.setattr(self.M, "lowm_entry", observed_lowm_entry)
            mp.setattr(self.M, "torch_mm", observed_torch_mm)

    def twin(call) -> dict:
        """This call's MTP layer with each all-reduce replaced by the single full-K call: the four
        ranks' attention outputs gathered in rank order and projected by the whole `o_proj`, the
        MoE run with every routed expert and the shared expert whole -- tp1's composition, through
        the same entries, on this call's inputs and routing. Collective (the gather)."""
        attn = torch.cat([a.cuda() for a in comm.allgather(call["attn"].cpu())], dim=-1)
        o = M.cublas_mm(attn, attn_w["o_proj.weight"].t())
        shared, mid = M.flashinfer_fused_add_rmsnorm(
            o.clone(), call["x"].clone(), block_w["pre_moe_layernorm.weight"], eps
        )
        latent = M.cublas_mm(call["routed"], moe_w["fc1_latent_proj.weight"].t())
        experts = M.latent_moe_experts_fused_act(
            latent,
            call["ids"],
            call["weights"],
            moe_w["experts.up_proj"],
            moe_w["experts.down_proj"],
            clamp,
            0,
            moe_w["experts.up_proj"].shape[0],
        )
        experts = M.cublas_mm(
            M.flashinfer_rmsnorm(experts, moe_w["fc2_norm.weight"], eps),
            moe_w["fc2_latent_proj.weight"].t(),
        )
        shared = M.cublas_mm(
            M.softclamp_relu2(
                M.cublas_mm(shared, moe_w["shared_experts.up_proj.weight"].t()), clamp
            ),
            moe_w["shared_experts.down_proj.weight"].t(),
        )
        moe = torch.add(experts, shared)
        out, _ = M.flashinfer_fused_add_rmsnorm(
            moe.clone(), mid, weights["final_layernorm.weight"], eps
        )
        return dict(o=o, moe=moe, out=out)

    class Comparator(H.Comparator):
        """The harness's comparator with the per-op references of a tp4 rank's slices, plus the
        TP term of every call and its twin against `R_t`."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.qkv = qkv_rank

        @torch.inference_mode()
        def compare(self, record, cells, stride=1):
            for depth, call in ((1, record.calls[0]), (2, record.calls[1])):
                index, where, segments = self._rows(record, depth, stride)
                idx = torch.tensor(index, device="cuda")
                cell = cells.setdefault((record.kind, depth), H.CellStats())
                cell.steps += 1
                cell.rows += len(index)
                cell.max_batch = max(cell.max_batch, len(record.rows))
                cell.max_tokens = max(cell.max_tokens, call["out"].shape[0])
                cell.max_position = max(cell.max_position, max(p for _, p in where))
                outputs, routed = self._outputs(call, depth, idx, where)
                for a, b in H.OUTPUT_PAIRS:
                    H._note(cell.output, f"{a}|{b}", H.errors(outputs[a], outputs[b]))
                same_routing = self._routing(cell, call, idx, routed)
                self._logits(cell, outputs, same_routing)
                self._stages(cell, call, depth, idx, segments)
                for name, rows in self._by_sequence(where).items():
                    for a, b in (("T", "R_t"), ("H_t", "R_hf_t")):
                        table_ = cell.by_sequence.setdefault(name, {})
                        H._note(table_, f"{a}|{b}", H.errors(outputs[a][rows], outputs[b][rows]))
                # The TP term, and the twin held to what the tp1 target is held to.
                w = twin(call)
                t = tp.setdefault(f"{record.kind}|{depth}", dict(logits=0.0, twin_logits=0.0))
                H._note(t, "T|W", H.errors(call["out"][idx], w["out"][idx]))
                H._note(t, "o|W", H.errors(call["o_sum"], w["o"]))
                H._note(t, "moe|W", H.errors(call["moe"], w["moe"]))
                H._note(t, "W|R_t", H.errors(w["out"][idx], outputs["R_t"]))
                head = SimpleNamespace(weight=self.lm_head)
                for start in range(0, len(index), 256):
                    rows = idx[start : start + 256]
                    logit_t = self.layer.shared_head(call["out"][rows], head, None, True)
                    logit_w = self.layer.shared_head(w["out"][rows], head, None, True)
                    logit_r = R.linear_fp32(outputs["R_t"][start : start + 256], self.lm_head_fp32)
                    t["logits"] = max(t["logits"], float((logit_t - logit_w).abs().max()))
                    t["twin_logits"] = max(t["twin_logits"], float((logit_w - logit_r).abs().max()))

        def _stages(self, cell, call, depth, idx, segments) -> None:
            """Each op against its entry's reference fed the target's own input to that op, at
            this rank's slices: the qkv GEMM over its query heads' rows (and the whole KV head),
            the attention of those heads, the `o_proj` partial over their columns; the residual
            add reads the all-reduced `o_proj`. HF's own op on the same input against the
            reference at HF's rounding points beside each."""
            cfg, w = self.config, self.weights
            kv_heads, dim = cfg.num_key_value_heads, cfg.head_dim
            kv_width = kv_heads * dim
            c = {k: v[idx] for k, v in call.items() if isinstance(v, torch.Tensor)}

            def note(stage, out, ref, table_=cell.stages):
                H._note(table_, stage, H.errors(out, ref))

            norm = H.flashinfer_rmsnorm.reference
            note("enorm", c["e"], norm(self.embed[c["input_ids"].long()], w["enorm.weight"], eps))
            note("hnorm", c["h"], norm(c["hidden"], w["hnorm.weight"], eps))
            for stage, out, x, weight in (
                ("eh_proj", c["x"], torch.cat([c["e"], c["h"]], -1), w["eh_proj.weight"]),
                ("qkv", c["qkv"], c["a"], self.qkv),
                ("o_proj", c["o"], c["attn"], o_rank),
            ):
                ref = R.linear(x, weight)
                note(stage, out, ref)
                note(stage, H.hf_linear(x, weight), ref, cell.stages_hf)
            note(
                "routed_norm",
                c["routed"],
                norm(c["x"], block_w["shortcut_input_layernorm.weight"], eps),
            )
            note("input_norm", c["a"], norm(c["x"], block_w["input_layernorm.weight"], eps))
            # Attention: each request's rows over the keys and values its cache holds before them
            # (the target's own, kept from its earlier depth-1 rows) and the rows' own.
            qkv = call["qkv"]
            attn = torch.empty_like(call["attn"])
            attn_hf = torch.empty_like(call["attn"])
            for req, rows, first in segments:
                q = qkv[rows, :q_width].reshape(-1, heads, dim)
                k = qkv[rows, q_width : q_width + kv_width].reshape(-1, kv_heads, dim)
                v = qkv[rows, q_width + kv_width :].reshape(-1, kv_heads, dim)
                kc, vc = self._cache(req, k)
                keys, values = torch.cat([kc[:first], k]), torch.cat([vc[:first], v])
                attn[rows] = R.causal_attention(q, keys, values, dim**-0.5).reshape(-1, q_width)
                attn_hf[rows] = H.hf_attention(q, keys, values, dim**-0.5).reshape(-1, q_width)
                if depth == 1:
                    kc[first : first + k.shape[0]] = k
                    vc[first : first + v.shape[0]] = v
            note("attention", c["attn"], attn[idx])
            note("attention", attn_hf[idx], attn[idx], cell.stages_hf)
            pre_moe, residual = H.target_add_norm(
                c["x"], c["o_sum"], block_w["pre_moe_layernorm.weight"], eps
            )
            note("residual", c["mid"], residual)
            note("pre_moe_norm", c["shared"], pre_moe)
            routing = (c["ids"], c["weights"])
            moe_in = (c["routed"], c["shared"])
            note("moe", c["moe"], H.moe(*moe_in, w, cfg, H.REF_T, routing))
            note(
                "moe",
                H.moe(*moe_in, w, cfg, H.HF, routing),
                H.moe(*moe_in, w, cfg, H.REF, routing),
                cell.stages_hf,
            )
            final, _ = H.target_add_norm(c["mid"], c["moe"], w["final_layernorm.weight"], eps)
            note("final", c["out"], final)

    def comparator(layer) -> Comparator:
        return Comparator(sequences, seqrefs, embed, weights, lm_head, layer, text_config)

    def driver() -> H.Driver:
        made = H.Driver(model, M, manager, md, sequences, MAX_BATCH)
        made.observer = Observer(model, M)
        return made

    records: dict = dict(rank=rank, order=[s.name for s in sequences])

    # `R_own` in pieces is `mtp_draft_hidden(mode="eagle")` bit for bit at both depths.
    with torch.inference_mode():
        seq = next(s for s in sequences if s.name == "clips")
        whole = R.mtp_draft_hidden(
            seq.ids_cuda, seq.hidden, embed, weights, text_config, num_depths=H.DRAFT, mode="eagle"
        )
        pieces = H.sequence_reference(seq, embed, weights, text_config)
        records["reference_pieces"] = [
            bool(torch.equal(pieces.own1, whole[0])),
            bool(torch.equal(pieces.own2, whole[1])),
        ]

    # --- the campaign, every rank the same steps, in lockstep ---
    campaign_driver = driver()
    with torch.inference_mode():
        cells = H.run_campaign(campaign_driver, comparator(campaign_driver.layer), log=say)
    records["cells"] = {
        f"{kind}|{depth}": dict(
            rows=cell.rows,
            steps=cell.steps,
            max_batch=cell.max_batch,
            max_tokens=cell.max_tokens,
            max_position=cell.max_position,
            failures=H.failures(cell, depth),
            output={k: v["worst_row"] / H.BF16_ULP for k, v in cell.output.items()},
            output_rms={k: v["rel_rms"] / H.BF16_ULP for k, v in cell.output.items()},
            logits=dict(cell.logits),
            router_flips=len(cell.router_flips),
            stages={
                k: [v["worst_row"] / H.BF16_ULP, v["rel_rms"] / H.BF16_ULP]
                for k, v in cell.stages.items()
            },
        )
        for (kind, depth), cell in cells.items()
    }
    totals = H.by_depth(cells)
    for t in totals.values():
        flips = t.pop("draft_flips")
        t["draft_flip_margins"] = {side: [m for m, _ in rows] for side, rows in flips.items()}
    records["by_depth"] = {str(k): v for k, v in totals.items()}
    records["report"] = H.report(cells)
    records["tp"] = {
        key: {
            name: (
                dict(
                    element=v["worst_row"] / H.BF16_ULP,
                    rms=v["rel_rms"] / H.BF16_ULP,
                    identical=v["identical"],
                    max_abs=v["max_abs"],
                    rows=v["rows"],
                )
                if isinstance(v, dict)
                else v
            )
            for name, v in value.items()
        }
        for key, value in tp.items()
    }
    say("campaign done on every rank")

    # --- rivals: one planted mistake each, every rank the same small campaign ---
    rivals = {}
    for name in sorted(H.RIVALS):
        with torch.inference_mode():
            rival_cells = H.run_rival(name, M, model, driver, comparator, log=say)
        broken = set()
        for (_, depth), cell in rival_cells.items():
            broken |= {f.split(":")[0] for f in H.failures(cell, depth)}
        rivals[name] = sorted(broken)
    records["rivals"] = rivals

    free, total = torch.cuda.mem_get_info()
    records["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    records["device_in_use_gib"] = (total - free) / 2**30
    gathered = comm.gather(records, root=0)
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(dict(ranks=gathered, seconds=time.monotonic() - started), f)
        say(f"wrote {out_path}")
    manager.shutdown()
    comm.Barrier()


def _run_one_rank(out: str, checkpoint: str, trunk_cache: str) -> int:
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
        _rank_body(out, checkpoint, trunk_cache)
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
        "--checkpoint",
        default=os.path.join(os.environ.get("LLM_MODELS_ROOT", ""), "nemotron4-nano-preview"),
        help="the checkpoint directory (default: $LLM_MODELS_ROOT/nemotron4-nano-preview)",
    )
    parser.add_argument(
        "--trunk-cache",
        required=True,
        help="file keeping the trunk's hidden states (rank 0 computes it when absent or stale)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse(sys.argv[1:])
    if args.rank_worker:
        sys.exit(_run_one_rank(args.out, args.checkpoint, args.trunk_cache))
    spawn_ranks(args.out, args.checkpoint, args.trunk_cache)
