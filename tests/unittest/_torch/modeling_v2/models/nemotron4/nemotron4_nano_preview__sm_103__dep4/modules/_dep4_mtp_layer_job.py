# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 4-rank job behind `test_modeling_v2_dep4_mtp_layer.py`: the module accuracy of the dep4
target's MTP layer on every rank of one node, against the plain-torch reference.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _dep4_mtp_layer_job.py --out <json> [--checkpoint <dir>]
        [--trunk-cache <file>]

Run plainly, this script is its own launcher: it starts one MPI rank per device under `mpirun`, as
a dep4 serving engine starts its ranks, with a deadline it enforces by killing the job, because a
collective that breaks hangs rather than raising.

What every rank builds, as the engine builds it for its rank under ``MTPDecodingConfig(max_draft_len
=2)`` (the spec config resolved against the checkpoint: one-model MTP-Eagle over the one repeated
MTP layer): the dep4 shell by its own constructor on the meta device -- so `DraftModel`, `MTPLayer`,
`shared_head` and the spec worker are the served engine's -- with the MTP layer's rows (its window
of 128 of the layer's 512 routed experts included), the embedding and the lm_head filled from the
checkpoint through dep4's own weight table (`weights.MODEL_WEIGHTS.manifest`, rows and transforms
as `load` applies them); the hybrid cache manager for the rank's dep4 mapping, which under attention
DP holds the MTP layer's KV beside the trunk's (global layer 44 at its own index; no separate draft
manager); and the engine's TRTLLM attention metadata over it, filled per step as the runner fills
it (no recurrent-state metadata: the MTP layer keeps none).

What runs is `_nemotron4_mtp_layer`'s campaign (its docstring says which cells, references and
gates), every rank its own: each rank takes the prompt set in its own rotation, so its cells see
other prompts and its steps other batches than the other ranks'. The draft loop is the worker's
own, and the MTP layer's MoE crosses the ranks, so every draft step is a collective: each rank
announces its step's tokens and rows (`rank_counts`, the counts the worker hands the draft steps),
and a rank with no step left -- its campaign ahead of the others', whose walks run as long as their
drafts are accepted -- runs the attention-DP dummy's draft step (a generation row of 1 + D tokens
over its own history), as an idle serving rank does, until every rank is done.

Beside the gates, the EP term: on every MTP-layer call, the layer's MoE output against the same
composition with the routed experts in one full-window call (`latent_moe_experts_fused_act` --
`latent_moe_experts`' bits -- over all 512 experts, rounding once, HF's order of the ten partial
sums) -- what dep4's four fp32 window partials, added in the combine's (or the reduce-scatter's)
order and then rounded, change. Rank 0 writes every rank's records to `--out` as JSON.

Not a pytest module: everything that imports `tensorrt_llm` is imported inside the rank body (an
MPI-initialized process cannot start `mpirun`).
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
_PID_DIR_ENV = "NEMOTRON4_DEP4_MTP_LAYER_PID_DIR"
DEADLINE_S = 5400
"""Ceiling on the whole job: the trunk's hidden states (when no cache holds them), the references,
the campaign, the rivals."""
WORLD_SIZE = 4
MAX_BATCH = 520
"""Requests a rank keeps resident at once: the campaign's pool of 512, and the attention-DP dummy."""


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
    assert len(devices) >= WORLD_SIZE, f"dep4 runs on {WORLD_SIZE} devices; got {devices}"
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH; the ranks are an MPI world"
    pid_dir = tempfile.mkdtemp(prefix="dep4_mtp_layer_")
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

    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
        weights as W,
    )
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
    from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
    from tensorrt_llm._torch.speculative import (
        get_num_extra_kv_tokens,
        update_spec_config_from_model_config,
    )
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
        moe_ep_size=WORLD_SIZE,
        moe_tp_size=1,
        enable_attention_dp=True,
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
            config = dataclasses.replace(base, mapping=mapping, spec_config=spec)
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
    everything = [
        H.Sequence(p.name, list(p.token_ids), torch.tensor(p.token_ids, device="cuda"), h.cuda())
        for p, h in zip(prompts, hidden)
    ]
    # Each rank's own rotation of the long and the short prompts.
    long_ = [i for i, s in enumerate(everything) if len(s.ids) > 4096]
    short = [i for i in range(len(everything)) if i not in long_]
    turn_long, turn_short = rank % len(long_), rank % len(short)
    order = long_[turn_long:] + long_[:turn_long] + short[turn_short:] + short[:turn_short]
    sequences = [everything[i] for i in order]
    say(f"trunk hidden states of {sum(len(s.ids) for s in everything)} tokens on every rank")

    # The dep4 shell under the MTP config, every parameter declared on meta by its own
    # constructor; the MTP layer's rows, the embedding and the lm_head filled through the weight
    # table's manifest, each source's rows (the rank's expert window) and transform applied.
    with torch.device("meta"):
        model = M.ModelingV2Nemotron4NanoPreviewSm103Dep4(config)
    core = model.model
    table = W.MODEL_WEIGHTS
    d = table.dims(core)
    assert d.mtp_layers == 1 and core.mtp_layer == d.num_layers, (d.mtp_layers, core.mtp_layer)
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
    core.post_load_weights()
    assert model.spec_worker is not None and model.draft_model is not None
    assert (core.expert_offset, core.local_experts) == (rank * 128, 128)
    say(f"dep4 MTP layer built from {filled + 1} checkpoint tensors on every rank")

    # The rank's one hybrid cache manager, as the engine builds it under attention DP.
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
    )
    assert manager.layer_offsets[core.mtp_layer] == 40, manager.layer_offsets
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
    assert manager.add_dummy_requests(
        [ATTENTION_DP_DUMMY_REQUEST_ID], is_gen=True, max_num_draft_tokens=H.DRAFT
    ), "no room for the attention-DP dummy"

    embed = checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")
    weights = checkpoint.mtp_layer()
    lm_head = checkpoint.tensor(R.LM_HEAD_KEY)
    seqrefs = [H.sequence_reference(s, embed, weights, text_config) for s in sequences]
    say("references of every sequence on every rank")

    # --- every draft step a collective: the ranks meet once per step ---
    worker, draft_model = model.spec_worker, model.draft_model
    tokens_per_dummy = 1 + H.DRAFT

    def exchange(active: bool, tokens: int, rows: int) -> tuple[list, list] | None:
        """Every rank's (tokens, rows) of this draft loop, the inactive ranks' being the
        attention-DP dummy's; None when no rank is active."""
        got = comm.allgather((active, tokens, rows))
        if not any(a for a, _, _ in got):
            return None
        return (
            [t if a else tokens_per_dummy for a, t, _ in got],
            [n if a else 1 for a, _, n in got],
        )

    def rank_counts(tokens: int, rows: int) -> tuple[list, list]:
        return exchange(True, tokens, rows)

    dummy_ids = torch.Generator(device="cuda").manual_seed(77 + rank)

    def dummy_step(all_rank_num_tokens: list, subseq: list) -> None:
        """The attention-DP dummy's draft loop: one generation row of 1 + D tokens over its own
        history of D tokens, through the worker's draft loop, as an idle serving rank runs it."""
        md.seq_lens = torch.tensor([tokens_per_dummy], dtype=torch.int)
        md.beam_width = 1
        md.request_ids = [ATTENTION_DP_DUMMY_REQUEST_ID]
        md.prompt_lens = [H.DRAFT]
        md.num_contexts = 0
        md.num_chunked_ctx_requests = 0
        md.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[H.DRAFT],
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec),
        )
        md.kv_cache_manager = manager
        md.padded_num_tokens = None
        md.update_spec_dec_param(
            batch_size=1,
            is_spec_decoding_enabled=spec.spec_dec_mode.attention_need_spec_dec_mode(
                TrtllmAttention
            ),
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=H.DRAFT,
            max_total_draft_tokens=H.DRAFT,
            spec_metadata=None,
            num_contexts=0,
        )
        md.prepare()
        spec_metadata = SimpleNamespace(
            runtime_draft_len=H.DRAFT,
            batch_indices_cuda=torch.arange(MAX_BATCH, dtype=torch.int32, device="cuda"),
            wants_advanced_draft_sampling=False,
            all_rank_num_tokens=all_rank_num_tokens,
            subseq_all_rank_num_tokens=subseq,
            max_num_requests=MAX_BATCH,
        )
        inputs = dict(
            input_ids=torch.randint(
                0,
                text_config.vocab_size,
                (tokens_per_dummy,),
                generator=dummy_ids,
                dtype=torch.int32,
                device="cuda",
            ),
            position_ids=torch.arange(
                H.DRAFT, H.DRAFT + tokens_per_dummy, dtype=torch.int32, device="cuda"
            ),
            hidden_states=torch.zeros(
                tokens_per_dummy, text_config.hidden_size, dtype=torch.bfloat16, device="cuda"
            ),
            attn_metadata=md,
            spec_metadata=spec_metadata,
        )
        worker._prepare_attn_metadata_for_spec_dec(md)
        try:
            worker._forward_linear_draft_loop(
                inputs,
                md,
                spec_metadata,
                draft_model,
                None,
                0,
                1,
                torch.ones(1, dtype=torch.int32, device="cuda"),
                None,
            )
        finally:
            worker._restore_attn_metadata_from_spec_dec(md)

    def drain() -> int:
        """Run the attention-DP dummy's draft steps until every rank has finished its own."""
        steps = 0
        while (counts := exchange(False, 0, 0)) is not None:
            dummy_step(*counts)
            steps += 1
        return steps

    # --- the EP term: dep4's four fp32 window partials against one full-window call ---
    m = core._decode._mtp
    moe_weights = R.sub(weights, "block.moe.")
    experts_up, experts_down = moe_weights["experts.up_proj"], moe_weights["experts.down_proj"]
    dims = core._decode._d
    ep: dict = {}

    def single_window(call) -> torch.Tensor:
        """The MTP layer's MoE on this call's own inputs and routing, the routed experts in one
        call over all 512 (tp1's composition); everything else as dep4 runs it."""
        latent = M.cublas_mm(call["routed"], m.moe.fc1_latent)
        experts = M.latent_moe_experts_fused_act(
            latent,
            call["ids"],
            call["weights"],
            experts_up,
            experts_down,
            dims.clamp,
            0,
            experts_up.shape[0],
        )
        experts = M.cublas_mm(
            M.flashinfer_rmsnorm(experts, m.moe.fc2_norm, dims.eps), m.moe.fc2_latent
        )
        shared = M.cublas_mm(
            M.softclamp_relu2(M.cublas_mm(call["shared"], m.moe.shared_up), dims.clamp),
            m.moe.shared_down,
        )
        return torch.add(experts, shared)

    class Comparator(H.Comparator):
        """The harness's comparator, plus the EP term of every call."""

        @torch.inference_mode()
        def compare(self, record, cells, stride=1):
            super().compare(record, cells, stride)
            for depth, call in ((1, record.calls[0]), (2, record.calls[1])):
                got, single = call["moe"], single_window(call)
                differ = (got.view(torch.int16) != single.view(torch.int16)).any(-1)
                diff = (got.float() - single.float()).abs()
                scale = single.float().abs().amax(-1).clamp_min(1e-30)
                e = ep.setdefault(
                    f"{record.kind}|{depth}",
                    dict(rows=0, rows_differing=0, elements_differing=0, max_abs=0.0, max_ulp=0.0),
                )
                e["rows"] += int(got.shape[0])
                e["rows_differing"] += int(differ.sum())
                e["elements_differing"] += int((diff > 0).sum())
                e["max_abs"] = max(e["max_abs"], float(diff.max()))
                e["max_ulp"] = max(e["max_ulp"], float((diff.amax(-1) / scale).max()) / H.BF16_ULP)

    def comparator(layer) -> Comparator:
        return Comparator(sequences, seqrefs, embed, weights, lm_head, layer, text_config)

    def driver() -> H.Driver:
        return H.Driver(model, M, manager, md, sequences, MAX_BATCH, rank_counts=rank_counts)

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

    # --- the campaign, every rank its own, in lockstep ---
    campaign_driver = driver()
    with torch.inference_mode():
        cells = H.run_campaign(campaign_driver, comparator(campaign_driver.layer), log=say)
        drained = drain()
    records["drained_steps"] = drained
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
        )
        for (kind, depth), cell in cells.items()
    }
    totals = H.by_depth(cells)
    for t in totals.values():
        flips = t.pop("draft_flips")
        t["draft_flip_margins"] = {side: [m for m, _ in rows] for side, rows in flips.items()}
    records["by_depth"] = {str(k): v for k, v in totals.items()}
    records["report"] = H.report(cells)
    records["ep"] = dict(ep)
    say(f"campaign done on every rank; rank 0 drained {drained} steps")

    # --- rivals: one planted mistake each, every rank its own small campaign ---
    rivals = {}
    for name in sorted(H.RIVALS):
        with torch.inference_mode():
            rival_cells = H.run_rival(name, M, model, driver, comparator, log=say)
            drain()
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
