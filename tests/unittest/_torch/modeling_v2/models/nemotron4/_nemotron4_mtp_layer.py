# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of a nemotron4 target's MTP layer, on the checkpoint's own weights: the harness.

Target-independent: a test hands it the target's modeling module (`M`), its shell built under
one-model MTP-Eagle (`max_draft_len` 2) with the MTP layer, the embedding and the lm_head real, the
cache manager holding the MTP layer's KV and the engine's TRTLLM attention metadata over it. What
runs is the target's MTP layer as the engine runs it: the worker's own draft loop
(`Eagle3OneModelWorker._forward_linear_draft_loop`) -- step 0 over every row of the step through
`draft_model.mtp_layers[0]` (context chunks, and the 1 + D tokens of each verify row), the gather of
one row per request, the worker's in-place rewrite of the metadata, and step 1 (depth 2) over one
row per request against the depth-1 KV. Under attention DP the draft steps' collectives take the
counts the worker hands them; a `Driver` given `rank_counts` asks for them every step.

The inputs are teacher forced on real text: the fixed prompt set, and the trunk's final normed
hidden state at every position from the plain-torch reference (`_nemotron4_reference`, the
transcription of `modeling_nemotron4.py`). Depth k at position i reads token i + k: at step 0
every row reads the token after it, and step 1 reads the true token after the depth-1 draft's
position (the worker's step-0 sample is recorded as the depth-1 draft and replaced by the true
token, as the reference's teacher forcing does; greedy acceptance never sees the difference,
because a depth-2 draft only counts when the depth-1 draft was the true token). Which positions a
request visits follows the engine: a verify at q accepts what the target's own drafts got right,
the gather row is q + accepted - 1, and the next verify starts after it.

The cells (`run_campaign`), each compared at depth 1 (step 0) and depth 2 (step 1):

* ``prefill_8k``: a long prompt's 7680-token context, fresh, in one step;
* ``prefill_chunked``: another's in 2048-token chunks, each over the earlier chunks' KV;
* ``prefill_varlen``: the six short prompts' first halves in one step;
* ``mixed``: a third long prompt's 2048-token chunks beside the others' verify rows;
* ``decode_walk``: every request verifying to the end of its text, 1 to 7 at a time;
* ``prefill_pack``: 512 requests' histories (64 to 7.4K tokens) packed into steps of up to 8192
  tokens (one depth-1 row in 32 compared);
* ``decode_b1`` .. ``decode_b512``: the CUDA-graph decode batch sizes, three verify steps each
  over the first B requests of that pool (histories of 6-7.4K tokens for the first 16).

References (`Numerics`; IEEE fp32 GEMMs and fp32 attention, rounded to bf16 once):

* `R_t`: the reference on exactly the target row's inputs with the target's routing (expert ids
  and fp32 weights), rounding where the target's entries round -- each RMSNorm once, the fused
  residual add normed from its fp32 sum (the entries' own `reference`s). The gated comparison:
  what is left is the kernels' accumulation order. Depth 2 reads the hidden state the target's
  depth 1 produced, over the reference's own depth-1 keys and values.
* `R_hf_t` / `H_t`: the reference at HF's rounding points, and HF's own implementation (bf16
  cuBLAS, bf16 SDPA, HF's expert loop), on the same inputs with the same routing. `H_t`
  against `R_hf_t` calibrates every band, which must admit the code the model was trained with.
* `R_own` / `H_own`: the reference's own drafting chain (`mtp_draft_hidden(mode="eagle")`) and
  HF's, each routing itself -- what `mtp-acceptance` compares the engine's drafts with.

Gates (`failures`), per cell and depth: the output against `R_t` within `OUTPUT_BAND`, the MTP
logits within `LOGIT_BAND`, the router exact on its own input up to near-ties, and each op of
`STAGES` against its entry's reference on the target's own input within `STAGE_BANDS`; per depth,
the MTP logits no further from `R_own` than `EPSILON_VS_HF` times HF's. Six planted mistakes
(`RIVALS`) must each break the output gate and the gate of the op they sit in. The bands were
derived on the tp1 target (twice the worst measured for the target and for HF's own code).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import time
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

__extra_import_path__ = [".", "../.."]
import _nemotron4_prompts as P  # noqa: E402 — needs the path declared above
import _nemotron4_reference as R  # noqa: E402
from _validating import ULP  # noqa: E402

DRAFT = 2
"""`max_draft_len`: the checkpoint's trained depth, the one the target is gated at."""

TOKENS_PER_BLOCK = 32
MAX_SEQ_LEN = 9344
MAX_NUM_TOKENS = 8192

BF16_ULP = ULP[torch.bfloat16]
TRUNK_CACHE_ENV = "NEMOTRON4_MTP_TRUNK_CACHE"
"""Set to a file path to keep the trunk's hidden states between runs (they take the whole text
model to compute). The file is keyed by the prompt ids, the checkpoint index and the reference's
source, and recomputed when any of them changes."""


# ----------------------------------------------------------------------------------------------
# Requests and steps
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Request:
    """One sequence being drafted for.

    ``cached`` positions (0 .. cached - 1) hold their final depth-1 keys and values in the draft
    KV; ``drafts`` are the depth-1 and depth-2 drafts proposed at position ``cached - 1`` (for the
    tokens at ``cached + 1`` and ``cached + 2``), which the next verify accepts or rejects.
    """

    rid: int
    seq: int
    prompt_len: int
    stop: int
    cached: int = 0
    drafts: tuple[int, int] | None = None
    handle: object = None

    @property
    def in_context(self) -> bool:
        return self.cached < self.prompt_len

    @property
    def done(self) -> bool:
        """No further verify fits: one at q reads tokens up to q + D + 1 and, accepting all, sends
        depth 2 the token at q + D + 2."""
        return not self.in_context and self.cached + DRAFT + 3 > self.stop


@dataclasses.dataclass
class Row:
    """One row of a step: a context chunk (``length`` tokens from ``start``), or a verify of the
    1 + D tokens from ``start`` that accepts ``accepted`` of them."""

    request: Request
    start: int
    length: int
    context: bool
    accepted: int = 1

    @property
    def gather(self) -> int:
        """The position whose depth-1 output feeds depth 2 (the worker's gather row)."""
        return self.start + (self.length if self.context else self.accepted) - 1


@dataclasses.dataclass
class StepRecord:
    """What one step of the worker's draft loop ran and produced: the step's rows, and per call of
    the MTP layer (step 0, step 1) what it read, returned and routed and each op's output
    (`MtpLayerObserver`)."""

    kind: str
    rows: list[Row]
    calls: list[dict]
    drafts: torch.Tensor  # [requests, 2]: the recorded argmax drafts of depth 1 and depth 2


class MtpLayerObserver:
    """Records every call of the target's MTP layer: what it read and returned, its MoE routing,
    and the output of each op on the way, which localizes a miss to one op.

    Installed on the `MTPLayer` instance the worker calls (its `forward`, looked up on the
    instance by `MTPLayer.__call__`) and on the globals of the modeling module `M` the MTP layer
    calls. Only the MTP layer runs here, so every call is its own; an op call is attributed to its
    role by its weight operand. Every recorded tensor is a copy taken at the call: the fused
    add-norm writes its operands in place.
    """

    def __init__(self, model, M):
        core = model.model
        self.M = M
        self.layer = model.draft_model.mtp_layers[0]
        m = core._decode._mtp
        self.norms = {
            m.enorm.data_ptr(): "e",
            m.hnorm.data_ptr(): "h",
            m.shortcut_norm.data_ptr(): "routed",
            m.input_norm.data_ptr(): "a",
        }
        self.gemms = {m.eh.data_ptr(): "x", m.attention.qkv.data_ptr(): "qkv"}
        self.o_proj = m.attention.o.data_ptr()
        self.pre_moe, self.final = m.pre_moe_norm.data_ptr(), m.final_norm.data_ptr()
        self.calls: list[dict] = []
        self._stage: dict = {}

    def install(self, mp: pytest.MonkeyPatch) -> None:
        M = self.M
        layer_forward = self.layer.forward
        rmsnorm, mm = M.flashinfer_rmsnorm, M.cublas_mm
        fused, noaux = M.flashinfer_fused_add_rmsnorm, M.noaux_tc_op
        stage = self._stage

        def forward(
            embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
        ):
            stage.clear()
            num_contexts = attn_metadata.num_contexts
            out = layer_forward(
                embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
            )
            expected = {"e", "h", "x", "routed", "a", "qkv", "attn", "o", "shared", "mid", "moe"}
            assert expected | {"router_logits", "weights", "ids"} <= set(stage), sorted(stage)
            self.calls.append(
                dict(
                    num_contexts=num_contexts,
                    input_ids=input_ids.clone(),
                    hidden=hidden_states.clone(),
                    out=out.clone(),
                    **stage,
                )
            )
            stage.clear()
            return out

        def observed_rmsnorm(x, weight, eps):
            out = rmsnorm(x, weight, eps)
            role = self.norms.get(weight.data_ptr())
            if role is not None:
                stage[role] = out.clone()
            return out

        def observed_mm(mat_a, mat_b, *args, **kwargs):
            out = mm(mat_a, mat_b, *args, **kwargs)
            role = self.gemms.get(mat_b.data_ptr())
            if role is not None:
                stage[role] = out.clone()
            elif mat_b.data_ptr() == self.o_proj:
                stage["attn"], stage["o"] = mat_a.clone(), out.clone()
            return out

        def observed_fused(x, residual, weight, eps):
            if weight.data_ptr() == self.final:
                stage["moe"] = x.clone()
            out, res = fused(x, residual, weight, eps)
            if weight.data_ptr() == self.pre_moe:
                stage["shared"], stage["mid"] = out.clone(), res.clone()
            return out, res

        def observed_noaux(router_logits, *args, **kwargs):
            weights, ids = noaux(router_logits, *args, **kwargs)
            stage.update(
                router_logits=router_logits.clone(), weights=weights.clone(), ids=ids.clone()
            )
            return weights, ids

        mp.setattr(self.layer, "forward", forward)
        mp.setattr(M, "flashinfer_rmsnorm", observed_rmsnorm)
        mp.setattr(M, "cublas_mm", observed_mm)
        mp.setattr(M, "flashinfer_fused_add_rmsnorm", observed_fused)
        mp.setattr(M, "noaux_tc_op", observed_noaux)

    def take(self) -> list[dict]:
        calls, self.calls = self.calls, []
        return calls


class Driver:
    """Runs engine steps of the MTP layer through the worker's draft loop.

    Each step: the metadata is prepared over the manager holding the MTP layer's KV as the runner
    prepares a step (contexts first; a verify row carries 1 + D tokens), the worker's draft loop
    runs both draft steps, and the record keeps what each MTP-layer call read and returned.

    ``rank_counts(tokens, rows)``, when given, returns the counts the worker hands the draft steps
    under attention DP -- every rank's tokens of step 0 and rows of step 1, in rank order
    (`spec_metadata.all_rank_num_tokens` / `subseq_all_rank_num_tokens`) -- for this rank's step
    of ``tokens`` tokens in ``rows`` rows; it is called once per step, where the ranks meet.
    """

    def __init__(self, model, M, manager, md, sequences, max_batch: int, rank_counts=None):
        self.model = model
        self.worker = model.spec_worker
        self.draft_model = model.draft_model
        self.layer = model.draft_model.mtp_layers[0]
        self.manager = manager
        self.md = md
        self.sequences = sequences
        self.max_batch = max_batch
        self.spec = model.model_config.spec_config
        self.next_rid = 1
        self.observer = MtpLayerObserver(model, M)
        self.rank_counts = rank_counts
        # Called with the requests `free` releases (the comparator drops their cached keys).
        self.on_free = None

    # --- requests --------------------------------------------------------------------------------

    def new_request(self, seq: int, prompt_len: int, stop: int | None = None) -> Request:
        n = len(self.sequences[seq].ids)
        stop = n if stop is None else min(stop, n)
        assert 0 < prompt_len and prompt_len + DRAFT + 2 <= stop, (prompt_len, stop)
        req = Request(self.next_rid, seq, prompt_len, stop)
        self.next_rid += 1
        handles = self.manager.add_dummy_requests([req.rid], token_nums=[stop + DRAFT + 2])
        assert handles, f"the KV manager has no room for {stop} tokens"
        req.handle = handles[0]
        return req

    def free(self, requests) -> None:
        for req in requests:
            if req.handle is not None:
                self.manager.free_resources(req.handle)
                req.handle = None
        if self.on_free is not None:
            self.on_free(requests)

    # --- one engine step -------------------------------------------------------------------------

    def _prepare(self, rows: list[Row]) -> None:
        from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
        from tensorrt_llm._torch.metadata import KVCacheParams
        from tensorrt_llm._torch.speculative import get_num_extra_kv_tokens

        md = self.md
        num_contexts = sum(r.context for r in rows)
        assert all(r.context for r in rows[:num_contexts]), "context rows go first"
        md.seq_lens = torch.tensor([r.length for r in rows], dtype=torch.int)
        md.beam_width = 1
        md.request_ids = [r.request.rid for r in rows]
        # The runner's prompt length: a context row's chunk, a generation row's whole prompt.
        md.prompt_lens = [r.length if r.context else r.request.prompt_len for r in rows]
        md.num_contexts = num_contexts
        md.num_chunked_ctx_requests = 0
        md.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[r.start for r in rows],
            num_extra_kv_tokens=get_num_extra_kv_tokens(self.spec),
        )
        md.kv_cache_manager = self.manager
        md.padded_num_tokens = None
        md.update_spec_dec_param(
            batch_size=len(rows),
            is_spec_decoding_enabled=self.spec.spec_dec_mode.attention_need_spec_dec_mode(
                TrtllmAttention
            ),
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=DRAFT,
            max_total_draft_tokens=DRAFT,
            spec_metadata=None,
            num_contexts=num_contexts,
        )
        md.prepare()

    def step(self, rows: list[Row], kind: str) -> StepRecord:
        """One engine step over ``rows`` (contexts first), both draft steps; advances the requests."""
        md = self.md
        num_contexts = sum(r.context for r in rows)
        batch = len(rows)
        for r in rows:
            seq = self.sequences[r.request.seq]
            assert r.start + r.length + 1 <= len(seq.ids), (r.start, r.length, len(seq.ids))
            assert r.gather + 2 < len(seq.ids)
        self._prepare(rows)
        seqs = [self.sequences[r.request.seq] for r in rows]
        positions = [torch.arange(r.start, r.start + r.length) for r in rows]
        input_ids = torch.cat([s.ids_cuda[p.to("cuda") + 1] for s, p in zip(seqs, positions)]).to(
            torch.int32
        )
        hidden = torch.cat([s.hidden[p.to("cuda")] for s, p in zip(seqs, positions)])
        position_ids = torch.cat(positions).to(device="cuda", dtype=torch.int32)
        num_accepted = torch.tensor(
            [1 if r.context else r.accepted for r in rows], dtype=torch.int32, device="cuda"
        )
        # The true token at gather + 2: what teacher forcing feeds depth 2 in place of the
        # depth-1 draft.
        true_next = torch.tensor(
            [self.sequences[r.request.seq].ids[r.gather + 2] for r in rows],
            dtype=torch.int32,
            device="cuda",
        )
        all_rank_num_tokens = subseq_all_rank_num_tokens = None
        if self.rank_counts is not None:
            all_rank_num_tokens, subseq_all_rank_num_tokens = self.rank_counts(
                int(input_ids.shape[0]), batch
            )
        spec_metadata = SimpleNamespace(
            runtime_draft_len=DRAFT,
            batch_indices_cuda=torch.arange(self.max_batch, dtype=torch.int32, device="cuda"),
            wants_advanced_draft_sampling=False,
            all_rank_num_tokens=all_rank_num_tokens,
            subseq_all_rank_num_tokens=subseq_all_rank_num_tokens,
            max_num_requests=self.max_batch,
        )
        drafts = []
        sample = self.worker.sample_draft_tokens

        def teacher_forced_sample(logits, spec_md, batch_size, *, draft_step=None, **kwargs):
            tokens = sample(logits, spec_md, batch_size, draft_step=draft_step, **kwargs)
            drafts.append(tokens.clone())
            return true_next.to(tokens.dtype) if draft_step == 0 else tokens

        inputs = dict(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            attn_metadata=md,
            spec_metadata=spec_metadata,
        )
        with pytest.MonkeyPatch.context() as mp:
            self.observer.install(mp)
            mp.setattr(self.worker, "sample_draft_tokens", teacher_forced_sample)
            self.worker._prepare_attn_metadata_for_spec_dec(md)
            try:
                self.worker._forward_linear_draft_loop(
                    inputs,
                    md,
                    spec_metadata,
                    self.draft_model,
                    None,
                    num_contexts,
                    batch,
                    num_accepted,
                    None,
                )
            finally:
                self.worker._restore_attn_metadata_from_spec_dec(md)
        torch.cuda.synchronize()
        calls = self.observer.take()
        assert len(calls) == DRAFT, len(calls)
        assert calls[0]["num_contexts"] == num_contexts and calls[1]["num_contexts"] == 0
        record = StepRecord(kind=kind, rows=list(rows), calls=calls, drafts=torch.stack(drafts, 1))
        # Advance: positions up to the gather row are final; the drafts made there are next.
        for r, (d1, d2) in zip(rows, record.drafts.tolist()):
            req = r.request
            req.cached = r.gather + 1
            req.drafts = (d1, d2)
        return record

    def verify_row(self, req: Request) -> Row:
        """The next verify of ``req``: what its last drafts get right is accepted."""
        assert not req.in_context and req.drafts is not None
        ids = self.sequences[req.seq].ids
        q = req.cached
        d1, d2 = req.drafts
        accepted = 1
        if d1 == ids[q + 1]:
            accepted += 1
            if d2 == ids[q + 2]:
                accepted += 1
        return Row(req, q, 1 + DRAFT, context=False, accepted=accepted)

    def context_row(self, req: Request, chunk: int) -> Row:
        length = min(chunk, req.prompt_len - req.cached)
        return Row(req, req.cached, length, context=True)


# ----------------------------------------------------------------------------------------------
# The sequences and the trunk's hidden states
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Sequence:
    """One text of the prompt set: its ids and the trunk's final normed hidden state at each
    position (``norm(residual_readout(streams))``, what the MTP layer reads)."""

    name: str
    ids: list[int]
    ids_cuda: torch.Tensor
    hidden: torch.Tensor  # [N, hidden] bf16


def _trunk_key(checkpoint: R.Checkpoint, prompts) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.name.encode())
        h.update(P.token_sha256(p.token_ids).encode())
    with open(os.path.join(checkpoint.path, "model.safetensors.index.json"), "rb") as f:
        h.update(f.read())
    with open(R.__file__, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


@torch.inference_mode()
def trunk_hidden(checkpoint: R.Checkpoint, prompts, log=print) -> list[torch.Tensor]:
    """The reference text model's final normed hidden state at every position of each prompt.

    Runs the reference's decoder layers one at a time over every prompt, each layer's weights
    loaded, used and freed in turn (the whole model never sits on the GPU at once). The GDP mixers
    run the reference's chunked (WY) form of the recurrence -- its independent fp32 formulation of
    HF's per-token loop, which is slow over 8K-token prompts; the inputs only have to be the
    model's own hidden states, the same on both sides of every comparison here.
    """
    cache = os.environ.get(TRUNK_CACHE_ENV)
    key = _trunk_key(checkpoint, prompts)
    if cache and os.path.exists(cache):
        saved = torch.load(cache, map_location="cuda")
        if saved.get("key") == key:
            log(f"trunk hidden states from {cache}")
            return saved["hidden"]
    config = checkpoint.config
    t0 = time.time()
    embed = checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")
    ids = [torch.tensor(p.token_ids, device="cuda") for p in prompts]
    streams = [embed[i].repeat(1, config.wide_residual_num_streams) for i in ids]
    del embed
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            R, "gdp_mixer", functools.partial(R.gdp_mixer, recurrence=R.gated_delta_rule_chunked)
        )
        for i, kind in enumerate(config.layer_types):
            weights = checkpoint.layer(i)
            streams = [R.decoder_layer(kind, s, weights, config)[0] for s in streams]
            del weights
            if i % 4 == 3:
                log(f"trunk: layer {i} done at {time.time() - t0:.0f}s")
    readout = checkpoint.tensor(R.TEXT_PREFIX + "residual_readout.logit")
    norm = checkpoint.tensor(R.TEXT_PREFIX + "norm.weight")
    hidden = [R.final_hidden(s, readout, norm, config) for s in streams]
    log(f"trunk hidden states of {sum(len(i) for i in ids)} tokens in {time.time() - t0:.0f}s")
    if cache:
        torch.save(dict(key=key, hidden=[h.cpu() for h in hidden]), cache)
    return hidden


# ----------------------------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Numerics:
    """How one implementation computes the MTP layer's ops: its bias-free bf16 GEMM, its attention
    (`R.causal_attention`'s signature), its routed-expert loop, its dense MLP, and where it rounds
    its RMSNorms (``norm(x, weight, eps)``) and its residual adds (``add_norm(x, update, weight,
    eps) -> (normed, residual)``)."""

    name: str
    linear: Callable
    attention: Callable
    experts: Callable
    mlp: Callable
    norm: Callable
    add_norm: Callable


def hf_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """A bias-free bf16 `nn.Linear` as HF runs it: torch's bf16 GEMM (cuBLAS, fp32 accumulation,
    rounded once)."""
    return F.linear(x, weight)


SDPA_BACKENDS = (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION)
"""The SDPA kernels `hf_attention` may run. Not cuDNN's: on GB300 it returned values off by 4e4
for a real 358-query step over a 6,421-key history of this layer (a boolean mask, and inf with
the lower-right causal bias), where the flash, memory-efficient and math kernels agree with fp32
within 0.43 bf16 ulps RMS."""


def hf_attention(q, k, v, scale) -> torch.Tensor:
    """HF's `sdpa_attention_forward` on bf16 operands: K/V repeated over the query heads, the bf16
    SDPA kernel, the T queries being the last T of the S keys -- causal over a sequence without
    history (HF's own call), the lower-right causal bias after one (a chunk, a verify, a draft
    step, which HF itself never runs: it recomputes the whole sequence)."""
    num_q, heads, dim = q.shape
    num_k, kv_heads, _ = k.shape
    group = heads // kv_heads
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
    vh = v.transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
    with sdpa_kernel(list(SDPA_BACKENDS)):
        if num_k == num_q:
            o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale)
        else:
            mask = causal_lower_right(num_q, num_k)
            o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    return o[0].transpose(0, 1).contiguous()


def hf_experts(x, ids, weights, up, down, clamp) -> torch.Tensor:
    """`Nemotron4Experts.forward`: the experts hit in ascending id, each one's (slot, token) pairs
    from `torch.where` over the one-hot mask, bf16 `F.linear` up, the clamped relu^2, the fp32
    routing weight multiplied and rounded before `down_proj`, `index_add_` into fp32."""
    acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    mask = F.one_hot(ids.long(), num_classes=up.shape[0]).permute(2, 1, 0)
    for expert in (mask.sum(dim=(-1, -2)) > 0).nonzero().flatten().tolist():
        slot, token = torch.where(mask[expert])
        h = R.clamped_relu2(F.linear(x[token], up[expert]), clamp)
        h = (h * weights[token, slot, None]).to(x.dtype)
        acc.index_add_(0, token, F.linear(h, down[expert]).to(acc.dtype))
    return acc.to(x.dtype)


def hf_mlp(x, up, down, clamp) -> torch.Tensor:
    """`Nemotron4MLP`: bf16 `F.linear` up, the clamped relu^2, bf16 `F.linear` down."""
    return F.linear(R.clamped_relu2(F.linear(x, up), clamp), down)


def hf_add_norm(x, update, weight, eps):
    """HF's residual add (`Nemotron4ResidualConnection.write`, bf16) and the RMSNorm of the rounded
    sum (`Nemotron4RMSNorm`: normalized in fp32, rounded, times the weight in bf16)."""
    residual = x + update.to(x.dtype)
    return R.rms_norm(residual, weight, eps), residual


def target_add_norm(x, update, weight, eps):
    """The target's residual add + RMSNorm (`flashinfer_fused_add_rmsnorm`'s reference): the sum
    in fp32, normalized and scaled in fp32 and rounded once; the residual is the sum rounded once."""
    return flashinfer_fused_add_rmsnorm.reference(update, x, weight, eps)


REF = Numerics("R", R.linear, R.causal_attention, R.latent_experts, R.mlp, R.rms_norm, hf_add_norm)
"""The reference at HF's rounding points: IEEE fp32 GEMMs and attention rounded once to bf16,
HF's two-rounding RMSNorm, bf16 residual adds. `R_own` is this chain with its own routing."""

REF_T = dataclasses.replace(
    REF, name="R_t", norm=flashinfer_rmsnorm.reference, add_norm=target_add_norm
)
"""The reference at the target's rounding points: the same fp32 GEMMs and attention, with the
RMSNorms and the fused residual-add norms rounded where the target's entries round them (once, the
norm of the fp32 sum). What is left between it and the target is the kernels' accumulation order."""

HF = Numerics("H", hf_linear, hf_attention, hf_experts, hf_mlp, R.rms_norm, hf_add_norm)
"""The checkpoint's own implementation as it runs: bf16 cuBLAS GEMMs, bf16 SDPA, HF's expert
loop, at HF's rounding points. A band must admit it: the model was trained against this code."""


@dataclasses.dataclass
class Front:
    """The MTP layer up to its MoE (routing never reaches it): the block input ``x`` (eh_proj's
    output), the routed input, the residual after attention ``mid``, the shared expert's input,
    and the rows' keys and values."""

    x: torch.Tensor
    routed: torch.Tensor
    mid: torch.Tensor
    shared: torch.Tensor
    k: torch.Tensor  # [N, kv_heads, head_dim]
    v: torch.Tensor

    def rows(self, index) -> Front:
        return Front(*(getattr(self, f.name)[index] for f in dataclasses.fields(self)))

    @staticmethod
    def cat(fronts) -> Front:
        return Front(
            *(torch.cat([getattr(f, x.name) for f in fronts]) for x in dataclasses.fields(Front))
        )


def _front_qkv(hidden, embeds, weights, config: R.TextConfig, nm: Numerics):
    eps = config.rms_norm_eps
    block = R.sub(weights, "block.")
    attn = R.sub(block, "self_attn.")
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    num = hidden.shape[0]
    e = nm.norm(embeds, weights["enorm.weight"], eps)
    h = nm.norm(hidden, weights["hnorm.weight"], eps)
    x = nm.linear(torch.cat([e, h], -1), weights["eh_proj.weight"])
    routed = nm.norm(x, block["shortcut_input_layernorm.weight"], eps)
    a = nm.norm(x, block["input_layernorm.weight"], eps)
    q = nm.linear(a, attn["q_proj.weight"]).view(num, heads, dim)
    k = nm.linear(a, attn["k_proj.weight"]).view(num, kv_heads, dim)
    v = nm.linear(a, attn["v_proj.weight"]).view(num, kv_heads, dim)
    return x, routed, q, k, v


def _front_rest(x, routed, o, k, v, weights, config: R.TextConfig, nm: Numerics) -> Front:
    block = R.sub(weights, "block.")
    heads_dim = config.num_attention_heads * config.head_dim
    update = nm.linear(o.reshape(x.shape[0], heads_dim), block["self_attn.o_proj.weight"])
    shared, mid = nm.add_norm(x, update, block["pre_moe_layernorm.weight"], config.rms_norm_eps)
    return Front(x=x, routed=routed, mid=mid, shared=shared, k=k, v=v)


def front(hidden, embeds, weights, config: R.TextConfig, nm: Numerics = REF, base_kv=None) -> Front:
    """``Nemotron4MultiTokenPredictionLayer`` up to its MoE at every position of one sequence.

    Depth 1 (``base_kv`` None): ordinary causal attention over the sequence (`R.mtp_layer`).
    Depth 2 (MTP-Eagle, `R._mtp_eagle_depth` with no earlier private keys): the query at i attends
    over the depth-1 keys of positions 0..i and its own key.
    """
    x, routed, q, k, v = _front_qkv(hidden, embeds, weights, config, nm)
    scale = config.head_dim**-0.5
    if base_kv is None:
        o = nm.attention(q, k, v, scale)
    else:
        assert nm.attention is R.causal_attention, "depth 2 over a whole sequence: reference only"
        o = nm.attention(q, base_kv[0], base_kv[1], scale, extra_k=k[:, None], extra_v=v[:, None])
    return _front_rest(x, routed, o, k, v, weights, config, nm)


def front_rows(hidden, embeds, weights, config: R.TextConfig, base_kv_rows, nm: Numerics = REF):
    """Depth 2 of single rows: row j's query attends over ``base_kv_rows[j]`` (its sequence's
    depth-1 keys and values of positions 0..g) and its own key -- `front`'s depth-2 arrangement
    for one position at a time."""
    x, routed, q, k, v = _front_qkv(hidden, embeds, weights, config, nm)
    scale = config.head_dim**-0.5
    o = torch.empty_like(q)
    for j, (bk, bv) in enumerate(base_kv_rows):
        keys, values = torch.cat([bk, k[j : j + 1]]), torch.cat([bv, v[j : j + 1]])
        o[j] = nm.attention(q[j : j + 1], keys, values, scale)[0]
    return _front_rest(x, routed, o, k, v, weights, config, nm)


def route(routed, weights, config: R.TextConfig):
    """The reference router (`R.route`: `Nemotron4TopkRouter` on IEEE fp32 logits) on the rows'
    routed input: ``(fp32 logits, ids, fp32 weights)``."""
    mw = R.sub(weights, "block.moe.")
    return R.route(
        routed,
        mw["gate.weight"],
        mw["gate.e_score_correction_bias"],
        config.num_experts_per_tok,
        config.routed_scaling_factor,
    )


def moe(routed, shared, weights, config: R.TextConfig, nm: Numerics = REF, routing=None):
    """`Nemotron4MoE` on the rows' routed and shared inputs: [N, hidden] bf16.

    ``routing`` None routes with the reference router (`route`); otherwise ``(ids, weights)`` per
    row. With the reference router and `REF` this is `R.latent_moe` op for op.
    """
    eps = config.rms_norm_eps
    mw = R.sub(weights, "block.moe.")
    if routing is None:
        _, ids, topk_weights = route(routed, weights, config)
    else:
        ids, topk_weights = routing
    clamp = config.activation_clamp_value
    h = nm.linear(routed, mw["fc1_latent_proj.weight"])
    h = nm.experts(
        h, ids.long(), topk_weights.float(), mw["experts.up_proj"], mw["experts.down_proj"], clamp
    )
    if config.moe_use_norm_before_up_proj:
        h = nm.norm(h, mw["fc2_norm.weight"], eps)
    h = nm.linear(h, mw["fc2_latent_proj.weight"])
    return h + nm.mlp(
        shared,
        mw["shared_experts.up_proj.weight"],
        mw["shared_experts.down_proj.weight"],
        clamp,
    )


def back(front_rows_: Front, weights, config: R.TextConfig, nm: Numerics = REF, routing=None):
    """The MoE, the residual add and ``final_layernorm`` of rows whose front is given:
    [N, hidden] bf16, the MTP layer's output."""
    out = moe(front_rows_.routed, front_rows_.shared, weights, config, nm, routing)
    out, _ = nm.add_norm(
        front_rows_.mid, out, weights["final_layernorm.weight"], config.rms_norm_eps
    )
    return out


@dataclasses.dataclass
class SequenceReference:
    """The references at every position of one sequence, teacher forced on the trunk's hidden
    state and the next tokens: depth 1's front at HF's rounding points, at the target's and in
    HF's own numerics; `R_own`'s output at both depths and its depth-2 routed input."""

    front1: Front
    front1_t: Front
    hf_front1: Front
    own1: torch.Tensor
    own2: torch.Tensor
    routed2: torch.Tensor


@torch.inference_mode()
def sequence_reference(seq: Sequence, embed, weights, config: R.TextConfig) -> SequenceReference:
    """`R_own` over one sequence is `mtp_draft_hidden(mode="eagle", num_depths=2)` in pieces -- the
    same ops in the same order, so the same bits (`test_reference_pieces_are_mtp_draft_hidden`)."""
    ids = seq.ids_cuda
    next1 = F.pad(ids[1:], (0, 1))
    next2 = F.pad(ids[2:], (0, 2))
    front1 = front(seq.hidden, embed[next1], weights, config)
    own1 = back(front1, weights, config)
    front2 = front(own1, embed[next2], weights, config, base_kv=(front1.k, front1.v))
    return SequenceReference(
        front1=front1,
        front1_t=front(seq.hidden, embed[next1], weights, config, nm=REF_T),
        hf_front1=front(seq.hidden, embed[next1], weights, config, nm=HF),
        own1=own1,
        own2=back(front2, weights, config),
        routed2=front2.routed,
    )


# ----------------------------------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------------------------------


def errors(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """``out`` against ``ref``, rows = last dim.

    ``worst_row`` is the largest element error over its row's largest ``|ref|``: in units of
    ``BF16_ULP`` it is ``assert_within_ulp``'s element gate. ``rel_rms`` is its RMS gate.
    ``max_rel`` is the largest element error over the element's own ``|ref|``, among elements at
    least 1/16 of their row's largest (smaller ones are cancellation, where a relative error is
    meaningless). ``worst`` is the element behind ``worst_row``.
    """
    assert out.shape == ref.shape, (out.shape, ref.shape)
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    d = (o - r).abs()
    tiny = torch.finfo(torch.float32).tiny
    scale = r.abs().amax(-1).clamp_min(tiny)
    rows = d.amax(-1) / scale
    row = int(rows.argmax())
    col = int(d[row].argmax())
    big = r.abs() >= scale[:, None] / 16
    rel = torch.where(big, d / r.abs().clamp_min(tiny), 0.0)
    return dict(
        rows=int(o.shape[0]),
        worst_row=float(rows.max()),
        median_row=float(rows.median()),
        rel_rms=float((o - r).norm() / r.norm().clamp_min(tiny)),
        identical=float((out == ref).float().mean()),
        max_abs=float(d.max()),
        max_rel=float(rel.max()),
        worst=dict(row=row, col=col, out=float(o[row, col]), ref=float(r[row, col])),
    )


def merge_errors(a: dict | None, b: dict) -> dict:
    """The worst of two error records (``worst`` follows ``worst_row``; ``identical`` is the
    smaller fraction; ``rows`` add up)."""
    if a is None:
        return dict(b)
    out = dict(a)
    for name in ("rel_rms", "median_row", "max_abs", "max_rel"):
        out[name] = max(a[name], b[name])
    out["identical"] = min(a["identical"], b["identical"])
    out["rows"] = a["rows"] + b["rows"]
    if b["worst_row"] > a["worst_row"]:
        out.update(worst_row=b["worst_row"], worst=b["worst"])
    return out


#: The comparisons of the MTP layer's output, (side, against). ``T`` is the target; ``R_t`` the
#: reference at the target's rounding points and ``R_hf_t`` / ``H_t`` the reference at HF's / HF's
#: own implementation, all three on the target row's inputs with its routing; ``R_own`` /
#: ``H_own`` the reference's / HF's own drafting chain with its own routing.
OUTPUT_PAIRS = (
    ("T", "R_t"),
    ("H_t", "R_hf_t"),
    ("T", "R_hf_t"),
    ("T", "R_own"),
    ("H_own", "R_own"),
)

#: Each op of the MTP layer against its entry's reference fed the target's own input to that op.
STAGES = (
    "enorm",  # flashinfer_rmsnorm(embedding, enorm)
    "hnorm",  # flashinfer_rmsnorm(hidden, hnorm)
    "eh_proj",  # cublas_mm(cat(e, h), eh_proj)
    "routed_norm",  # flashinfer_rmsnorm(x, shortcut_input_layernorm)
    "input_norm",  # flashinfer_rmsnorm(x, input_layernorm)
    "qkv",  # cublas_mm(a, qkv)
    "attention",  # thop_attention: the rows' queries over the cached keys and their own
    "o_proj",  # cublas_mm(attention, o_proj)
    "residual",  # flashinfer_fused_add_rmsnorm's residual: x + o
    "pre_moe_norm",  # its normed output: pre_moe_layernorm(x + o)
    "moe",  # _moe(routed, shared) with the target's routing
    "final",  # flashinfer_fused_add_rmsnorm(moe, x + o, final_layernorm)
)


@dataclasses.dataclass
class CellStats:
    """One cell's comparisons at one depth, merged over its steps."""

    rows: int = 0
    steps: int = 0
    max_batch: int = 0
    max_tokens: int = 0
    max_position: int = 0
    output: dict = dataclasses.field(default_factory=dict)  # "T|R_t" -> errors
    stages: dict = dataclasses.field(default_factory=dict)  # stage -> errors
    # The same ops in HF's own numerics on the same inputs (bf16 cuBLAS GEMMs, bf16 SDPA, HF's
    # expert loop), against the reference at HF's rounding points: what a band must admit.
    stages_hf: dict = dataclasses.field(default_factory=dict)
    # T|R_t and H_t|R_hf_t per sequence of the prompt set.
    by_sequence: dict = dataclasses.field(default_factory=dict)
    # The fp32 MTP logits: max |difference| per output pair over every row, and how far any
    # token gains on R_own's top token (max over rows of max_b (dlogit_b - dlogit_top)).
    logits: dict = dataclasses.field(default_factory=dict)
    gain_on_own_top: dict = dataclasses.field(default_factory=dict)  # "T" / "H_own" -> float
    # Every row's gain, and the largest |T - R_own| logit difference split by whether T routed
    # the row to the same experts as R_own.
    gains: dict = dataclasses.field(default_factory=dict)  # "T" / "H_own" -> [per row]
    logit_t_own_same_routing: float = 0.0
    logit_t_own_other_routing: float = 0.0
    # Draft (argmax) agreement per output pair: rows that agree.
    drafts: dict = dataclasses.field(default_factory=dict)
    # Rows where T's / H_own's draft differs from R_own's: (R_own's top-2 margin, the row's gain).
    draft_flips: dict = dataclasses.field(default_factory=dict)
    # The router on the target's own routed input: rows whose top-k differs from the reference
    # router's on that same input, as (the reference's score margin between the swapped experts,
    # twice the row's largest score difference); a legal flip has margin <= 2 * delta.
    router_flips: list = dataclasses.field(default_factory=list)
    # End to end: rows whose top-k differs from R_own's (each side routes its own routed input).
    routing_vs_own: dict = dataclasses.field(default_factory=dict)  # "T" / "H_own" -> count

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _note(table: dict, key: str, err: dict) -> None:
    table[key] = merge_errors(table.get(key), err)


class Comparator:
    """Compares each MTP-layer call of a step with the references, row by row.

    Every row is (sequence, depth, position): step 0's rows are depth 1 at each of the step's
    positions, step 1's rows depth 2 at each request's gather position. The target's own keys and
    values of every depth-1 row are kept per request: they are what its draft KV cache holds, the
    history the attention stage reads.
    """

    def __init__(self, sequences, seqrefs, embed, weights, lm_head, layer, config: R.TextConfig):
        self.sequences = sequences
        self.seqrefs = seqrefs
        self.embed = embed
        self.weights = weights
        self.lm_head = lm_head
        self.lm_head_fp32 = lm_head.float()
        self.layer = layer
        self.config = config
        attn = R.sub(weights, "block.self_attn.")
        self.qkv = torch.cat([attn["q_proj.weight"], attn["k_proj.weight"], attn["v_proj.weight"]])
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def forget(self, requests) -> None:
        for req in requests:
            self.kv.pop(req.rid, None)

    def _rows(self, record: StepRecord, depth: int, stride: int):
        """``(row index into the call, [(sequence, position)], [(request, call rows, first
        position)])``: the rows compared, and each request's contiguous rows."""
        index, where, segments = [], [], []
        if depth == 2:
            for j, r in enumerate(record.rows):
                index.append(j)
                where.append((r.request.seq, r.gather))
                segments.append((r.request, slice(j, j + 1), r.gather + 1))
            return index, where, segments
        offset = 0
        for r in record.rows:
            segments.append((r.request, slice(offset, offset + r.length), r.start))
            for j in range(0, r.length, stride if r.context else 1):
                index.append(offset + j)
                where.append((r.request.seq, r.start + j))
            offset += r.length
        return index, where, segments

    @torch.inference_mode()
    def compare(self, record: StepRecord, cells: dict, stride: int = 1) -> None:
        """Merge the step's comparisons into ``cells[(record.kind, depth)]``."""
        for depth, call in ((1, record.calls[0]), (2, record.calls[1])):
            index, where, segments = self._rows(record, depth, stride)
            idx = torch.tensor(index, device="cuda")
            cell = cells.setdefault((record.kind, depth), CellStats())
            cell.steps += 1
            cell.rows += len(index)
            cell.max_batch = max(cell.max_batch, len(record.rows))
            cell.max_tokens = max(cell.max_tokens, call["out"].shape[0])
            cell.max_position = max(cell.max_position, max(p for _, p in where))
            outputs, routed = self._outputs(call, depth, idx, where)
            for a, b in OUTPUT_PAIRS:
                _note(cell.output, f"{a}|{b}", errors(outputs[a], outputs[b]))
            same_routing = self._routing(cell, call, idx, routed)
            self._logits(cell, outputs, same_routing)
            self._stages(cell, call, depth, idx, segments)
            for name, rows in self._by_sequence(where).items():
                for a, b in (("T", "R_t"), ("H_t", "R_hf_t")):
                    table = cell.by_sequence.setdefault(name, {})
                    _note(table, f"{a}|{b}", errors(outputs[a][rows], outputs[b][rows]))

    def _by_sequence(self, where) -> dict:
        rows: dict = {}
        for j, (s, _) in enumerate(where):
            rows.setdefault(self.sequences[s].name, []).append(j)
        return {name: torch.tensor(r, device="cuda") for name, r in rows.items()}

    def _take(self, where, pick) -> list:
        """``pick(sequence reference, positions)`` over runs of rows on the same sequence."""
        out, i = [], 0
        while i < len(where):
            j = i
            while j < len(where) and where[j][0] == where[i][0]:
                j += 1
            pos = torch.tensor([p for _, p in where[i:j]], device="cuda")
            out.append(pick(self.seqrefs[where[i][0]], pos))
            i = j
        return out

    def _outputs(self, call, depth, idx, where):
        """Every side's output on the compared rows, and R_own's / H_own's routed inputs."""
        cfg, w = self.config, self.weights
        routing = (call["ids"][idx], call["weights"][idx])
        if depth == 1:
            r_front = Front.cat(self._take(where, lambda r, p: r.front1.rows(p)))
            t_front = Front.cat(self._take(where, lambda r, p: r.front1_t.rows(p)))
            h_front = Front.cat(self._take(where, lambda r, p: r.hf_front1.rows(p)))
            own = torch.cat(self._take(where, lambda r, p: r.own1[p]))
            h_own, h_routed = back(h_front, w, cfg, HF), h_front.routed
            own_routed = r_front.routed
        else:
            hidden = call["hidden"][idx]
            embeds = self.embed[call["input_ids"][idx].long()]
            refs = [self.seqrefs[s] for s, _ in where]

            def base(field):
                return [
                    (getattr(r, field).k[: g + 1], getattr(r, field).v[: g + 1])
                    for r, (_, g) in zip(refs, where)
                ]

            r_front = front_rows(hidden, embeds, w, cfg, base("front1"))
            t_front = front_rows(hidden, embeds, w, cfg, base("front1_t"), nm=REF_T)
            h_kv = base("hf_front1")
            h_front = front_rows(hidden, embeds, w, cfg, h_kv, nm=HF)
            own = torch.cat(self._take(where, lambda r, p: r.own2[p]))
            own_routed = torch.cat(self._take(where, lambda r, p: r.routed2[p]))
            # H's own chain: its own depth 1 at the gather position, then its own depth 2.
            h1 = back(Front.cat(self._take(where, lambda r, p: r.hf_front1.rows(p))), w, cfg, HF)
            h2_front = front_rows(h1, embeds, w, cfg, h_kv, nm=HF)
            h_own, h_routed = back(h2_front, w, cfg, HF), h2_front.routed
        outputs = {
            "T": call["out"][idx],
            "R_t": back(t_front, w, cfg, REF_T, routing),
            "R_hf_t": back(r_front, w, cfg, REF, routing),
            "H_t": back(h_front, w, cfg, HF, routing),
            "R_own": own,
            "H_own": h_own,
        }
        return outputs, dict(R_own=own_routed, H_own=h_routed)

    def _logits(self, cell: CellStats, outputs: dict, same_routing, chunk: int = 256) -> None:
        rows = outputs["T"].shape[0]
        head = SimpleNamespace(weight=self.lm_head)
        for start in range(0, rows, chunk):
            sl = slice(start, start + chunk)
            logits = {"T": self.layer.shared_head(outputs["T"][sl], head, None, True)}
            for side in ("R_t", "R_hf_t", "H_t", "R_own", "H_own"):
                logits[side] = R.linear_fp32(outputs[side][sl], self.lm_head_fp32)
            top = {side: v.argmax(-1) for side, v in logits.items()}
            for a, b in OUTPUT_PAIRS:
                key = f"{a}|{b}"
                d = float((logits[a] - logits[b]).abs().max())
                cell.logits[key] = max(cell.logits.get(key, 0.0), d)
                cell.drafts[key] = cell.drafts.get(key, 0) + int((top[a] == top[b]).sum())
            own = logits["R_own"].topk(2, dim=-1)
            margin = own.values[:, 0] - own.values[:, 1]
            for side in ("T", "H_own"):
                d = logits[side] - logits["R_own"]
                gain = (d - d.gather(1, own.indices[:, :1])).amax(-1)
                cell.gain_on_own_top[side] = max(
                    cell.gain_on_own_top.get(side, 0.0), float(gain.max())
                )
                cell.gains.setdefault(side, []).extend(gain.tolist())
                flips = cell.draft_flips.setdefault(side, [])
                for j in torch.nonzero(top[side] != own.indices[:, 0]).flatten().tolist():
                    flips.append((float(margin[j]), float(gain[j])))
            d_own = (logits["T"] - logits["R_own"]).abs().amax(-1)
            same = same_routing[sl]
            if bool(same.any()):
                cell.logit_t_own_same_routing = max(
                    cell.logit_t_own_same_routing, float(d_own[same].max())
                )
            if bool((~same).any()):
                cell.logit_t_own_other_routing = max(
                    cell.logit_t_own_other_routing, float(d_own[~same].max())
                )

    def _routing(self, cell: CellStats, call, idx, routed) -> None:
        cfg = self.config
        bias = R.sub(self.weights, "block.moe.")["gate.e_score_correction_bias"]
        ids_t = call["ids"][idx].long().sort(-1).values
        # The router on the target's own routed input: only the router GEMM's summation order
        # and noaux_tc_op's arithmetic separate the two.
        logits_t = call["router_logits"][idx].float()
        logits_r, ids_r, _ = route(call["routed"][idx], self.weights, cfg)
        same = ids_t == ids_r.sort(-1).values
        for j in torch.nonzero(~same.all(-1)).flatten().tolist():
            t, r = set(ids_t[j].tolist()), set(ids_r[j].tolist())
            choice = torch.sigmoid(logits_r[j]) + bias
            delta = float((torch.sigmoid(logits_t[j]) - torch.sigmoid(logits_r[j])).abs().max())
            margin = max(float(choice[a] - choice[b]) for a in r - t for b in t - r)
            cell.router_flips.append((margin, 2 * delta))
        # End to end: each side's own routing against R_own's.
        _, ids_own, _ = route(routed["R_own"], self.weights, cfg)
        ids_own = ids_own.sort(-1).values
        _, ids_h, _ = route(routed["H_own"], self.weights, cfg)
        for side, ids in (("T", ids_t), ("H_own", ids_h.sort(-1).values)):
            differ = int((ids != ids_own).any(-1).sum())
            cell.routing_vs_own[side] = cell.routing_vs_own.get(side, 0) + differ
        return (ids_t == ids_own).all(-1)

    def _stages(self, cell: CellStats, call, depth, idx, segments) -> None:
        """Each op against its entry's reference fed the target's own input to that op, and HF's
        own op on the same input against the reference at HF's rounding points."""
        cfg, w = self.config, self.weights
        eps = cfg.rms_norm_eps
        block = R.sub(w, "block.")
        heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q_width, kv_width = heads * dim, kv_heads * dim
        # Row-local ops on the compared rows only; the attention reads every row's keys.
        c = {k: v[idx] for k, v in call.items() if isinstance(v, torch.Tensor)}

        def note(stage, out, ref, table=cell.stages):
            _note(table, stage, errors(out, ref))

        norm = flashinfer_rmsnorm.reference
        note("enorm", c["e"], norm(self.embed[c["input_ids"].long()], w["enorm.weight"], eps))
        note("hnorm", c["h"], norm(c["hidden"], w["hnorm.weight"], eps))
        for stage, out, x, weight in (
            ("eh_proj", c["x"], torch.cat([c["e"], c["h"]], -1), w["eh_proj.weight"]),
            ("qkv", c["qkv"], c["a"], self.qkv),
            ("o_proj", c["o"], c["attn"], block["self_attn.o_proj.weight"]),
        ):
            ref = R.linear(x, weight)
            note(stage, out, ref)
            note(stage, hf_linear(x, weight), ref, cell.stages_hf)
        note(
            "routed_norm", c["routed"], norm(c["x"], block["shortcut_input_layernorm.weight"], eps)
        )
        note("input_norm", c["a"], norm(c["x"], block["input_layernorm.weight"], eps))
        # Attention: each request's rows over the keys and values its cache holds before them (the
        # target's own, kept from its earlier depth-1 rows) and the rows' own.
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
            attn_hf[rows] = hf_attention(q, keys, values, dim**-0.5).reshape(-1, q_width)
            if depth == 1:
                kc[first : first + k.shape[0]] = k
                vc[first : first + v.shape[0]] = v
        note("attention", c["attn"], attn[idx])
        note("attention", attn_hf[idx], attn[idx], cell.stages_hf)
        pre_moe, residual = target_add_norm(c["x"], c["o"], block["pre_moe_layernorm.weight"], eps)
        note("residual", c["mid"], residual)
        note("pre_moe_norm", c["shared"], pre_moe)
        routing = (c["ids"], c["weights"])
        moe_in = (c["routed"], c["shared"])
        note("moe", c["moe"], moe(*moe_in, w, cfg, REF_T, routing))
        note(
            "moe",
            moe(*moe_in, w, cfg, HF, routing),
            moe(*moe_in, w, cfg, REF, routing),
            cell.stages_hf,
        )
        final, _ = target_add_norm(c["mid"], c["moe"], w["final_layernorm.weight"], eps)
        note("final", c["out"], final)

    def _cache(self, req: Request, like: torch.Tensor):
        if req.rid not in self.kv:
            shape = (req.stop + DRAFT + 2, *like.shape[1:])
            self.kv[req.rid] = (like.new_zeros(shape), like.new_zeros(shape))
        return self.kv[req.rid]


# ----------------------------------------------------------------------------------------------
# The campaign: the cells the engine runs, each compared after every step
# ----------------------------------------------------------------------------------------------

LONG_PREFILL = 7680
"""A long prompt's context: 7680 tokens, the 8K workload's prompt as one step or as 2048-token
chunks over the earlier chunks' KV; the rest of the text is walked."""

CHUNK = 2048
DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
"""The CUDA-graph decode batch sizes: each runs ``ROUNDS`` verify steps (step 0 over B rows of
1 + D tokens, step 1 over B rows)."""

ROUNDS = 3


def _half(length: int) -> int:
    """A short prompt's context: its first half, leaving room for one draft window after it."""
    return max(1, min(length // 2, length - DRAFT - 2))


def _long(sequences) -> list[int]:
    return [i for i, s in enumerate(sequences) if len(s.ids) > 4096]


def run_campaign(
    driver: Driver,
    comparator: Comparator,
    log=print,
    parts=("prefill", "walk", "ladder"),
    cells: dict | None = None,
) -> dict:
    """Drive every cell and compare each step; returns ``{(kind, depth): CellStats}``.

    ``parts``: "prefill" and "walk" (the long and short prompts' prefills, mixed steps and walks
    to the end of their texts; "walk" needs "prefill"), "ladder" (the CUDA-graph batch sizes).
    """
    cells = {} if cells is None else cells
    seqs = driver.sequences
    long_ = _long(seqs)
    short = [i for i in range(len(seqs)) if i not in long_]

    def run(rows, kind, stride=1):
        record = driver.step(rows, kind)
        comparator.compare(record, cells, stride)
        return record

    driver.on_free = comparator.forget

    if "prefill" in parts:
        _prefills_and_walks(driver, run, seqs, long_, short, "walk" in parts, log)
    if "ladder" in parts:
        _ladder(driver, run, long_, log)
    return cells


def _prefills_and_walks(driver, run, seqs, long_, short, walk, log) -> None:
    # Prefill, fresh: one 7680-token context of a long prompt (the 8K workload's shape).
    a = driver.new_request(long_[0], LONG_PREFILL)
    run([driver.context_row(a, LONG_PREFILL)], "prefill_8k")
    # Prefill, chunked: another long prompt in 2048-token chunks over the earlier chunks' KV.
    b = driver.new_request(long_[1], LONG_PREFILL)
    while b.in_context:
        run([driver.context_row(b, CHUNK)], "prefill_chunked")
    # Prefill, a varlen batch of fresh contexts: the short prompts' first halves.
    shorts = [driver.new_request(i, _half(len(seqs[i].ids))) for i in short]
    run([driver.context_row(r, CHUNK) for r in shorts], "prefill_varlen")
    log("prefills done")
    if not walk:
        driver.free([a, b, *shorts])
        return

    # Mixed: a third long prompt's 2048-token chunks beside the others' verifies.
    c = driver.new_request(long_[2], 2 * CHUNK, stop=2 * CHUNK + 384)
    walking = [r for r in [a, b, *shorts] if not r.done]
    while c.in_context:
        run([driver.context_row(c, CHUNK)] + [driver.verify_row(r) for r in walking], "mixed")
        walking = [r for r in walking if not r.done]
    # Decode: every request walks its text to the end, all verifying together.
    walking = [r for r in [a, b, c, *shorts] if not r.done]
    while walking:
        run([driver.verify_row(r) for r in walking], "decode_walk")
        walking = [r for r in walking if not r.done]
    driver.free([a, b, c, *shorts])
    log("walks done")


def _ladder(driver, run, long_, log) -> None:
    # Decode at the CUDA-graph batch sizes: requests 0..B-1 of one pool verify together, so the
    # first requests take part in every batch size. Their histories: 6-7.4K tokens for the first
    # 16, 1-4K up to 128, 64-1K beyond; each over one of the long prompts.
    g = torch.Generator().manual_seed(0)
    pool = []
    for i in range(max(DECODE_BATCHES)):
        seq = long_[i % len(long_)]
        lo, hi = (6000, 7400) if i < 16 else (1024, 4096) if i < 128 else (64, 1024)
        prefix = int(torch.randint(lo, hi, (1,), generator=g))
        rounds = sum(ROUNDS for n in DECODE_BATCHES if n > i)
        pool.append(driver.new_request(seq, prefix, stop=prefix + 3 * rounds + 8))
    pending = list(pool)
    while pending:
        rows, budget = [], MAX_NUM_TOKENS
        while pending and budget > 0:
            row = driver.context_row(pending[0], min(budget, CHUNK * 4))
            rows.append(row)
            budget -= row.length
            if row.start + row.length == pending[0].prompt_len:
                pending.pop(0)
            else:
                break
        run(rows, "prefill_pack", stride=32)
    log(f"pool of {len(pool)} prefilled")
    for n in DECODE_BATCHES:
        for _ in range(ROUNDS):
            run([driver.verify_row(r) for r in pool[:n]], f"decode_b{n}")
    driver.free(pool)


# ----------------------------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------------------------

# Every band is twice the worst the campaign measured for the target and for HF's own
# implementation, each against the reference that rounds where it rounds. In bf16 ulps: element of
# the row's largest value / relative RMS.

OUTPUT_BAND = (32.0, 7.2)
"""The MTP layer's output, the target against `R_t` and HF's implementation against `R_hf_t`.
Measured: target 10.97 / 2.29, HF 15.80 / 3.58 (both worst at depth 1). The element gate is wide
on purpose: output column 2953 carries the routed experts' and the shared expert's outputs at
|~270| and nearly cancels them against the residual, so one bf16 spacing there (2.0) moves the
normed output by up to ~1 where the row's largest value is ~10-25; the RMS gate is the sharp one."""

EPSILON_VS_HF = 2.0
"""End to end, each side drafting with its own routing: the target's largest MTP-logit difference
from `R_own` may be at most this multiple of HF's own implementation's (`H_own`). Measured: depth
1 2.68 against 2.12, depth 2 2.48 against 2.41."""

LOGIT_BAND = 0.85
"""The fp32 MTP logits (the shared lm_head over the output): max |difference| of the target
against `R_t`, and of HF's own implementation against `R_hf_t`. Measured: target 0.419, HF
0.379."""

_NORM_BAND = (2.5, 0.12)  # measured 1.25 / 0.059: one rounding of the norm's fp32 result
STAGE_BANDS = {
    "enorm": _NORM_BAND,
    "hnorm": _NORM_BAND,
    "eh_proj": (4.0, 1.6),  # measured 1.98 / 0.78, and torch's bf16 GEMM gives the same bits
    "routed_norm": _NORM_BAND,
    "input_norm": _NORM_BAND,
    "qkv": (4.0, 1.6),
    "attention": (4.0, 1.5),  # target 2.00 / 0.73 (generation kernels), HF's SDPA 2.00 / 0.34
    "o_proj": (4.0, 1.6),
    "residual": (0.0, 0.0),  # the fused add's residual is the fp32 sum rounded once: exact
    "pre_moe_norm": _NORM_BAND,
    "moe": (25.0, 2.3),  # target 12.34 / 0.96, HF's expert loop 8.53 / 1.14
    "final": _NORM_BAND,
}
"""Each op against its entry's reference on the target's own input (HF's op against the reference
at HF's rounding points, where HF's differs): element / relative RMS, bf16 ulps."""


def _over(err: dict, band) -> bool:
    return err["worst_row"] > band[0] * BF16_ULP or err["rel_rms"] > band[1] * BF16_ULP


def failures(cell: CellStats, depth: int) -> list[str]:
    """The gates a cell breaks, by name: ``output``, ``logits``, ``router`` or an op of `STAGES`."""
    out = []
    for pair in ("T|R_t", "H_t|R_hf_t"):
        if _over(cell.output[pair], OUTPUT_BAND):
            out.append(f"output: {pair} {_ulps(cell.output[pair])} ulps")
        if cell.logits[pair] > LOGIT_BAND:
            out.append(f"logits: {pair} {cell.logits[pair]:.4f}")
    for margin, bound in cell.router_flips:
        if margin > bound:
            out.append(f"router: a top-k flip at margin {margin:.3e} > 2 delta {bound:.3e}")
    for stage, err in cell.stages.items():
        if _over(err, STAGE_BANDS[stage]):
            out.append(f"{stage}: {_ulps(err)} ulps")
    return out


# ----------------------------------------------------------------------------------------------
# Rivals: one plausible mistake each, planted in the target
# ----------------------------------------------------------------------------------------------


def _replace_mtp(mp: pytest.MonkeyPatch, model, **fields) -> None:
    """Swap fields of both targets' MTP-layer views (`_Mtp`) for the rival's."""
    for target in (model.model._prefill, model.model._decode):
        mp.setattr(target, "_mtp", target._mtp._replace(**fields))


def _plant_eh_halves_swapped(mp, model, M) -> None:
    """eh_proj reading the hidden state's half first (the embedding's half comes first)."""
    eh = model.model._decode._mtp.eh.t()  # [hidden, 2 * hidden] as the checkpoint stores it
    hidden = eh.shape[0]
    _replace_mtp(mp, model, eh=torch.cat([eh[:, hidden:], eh[:, :hidden]], 1).contiguous().t())


def _plant_norms_swapped(mp, model, M) -> None:
    """enorm on the hidden state and hnorm on the embedding."""
    m = model.model._decode._mtp
    _replace_mtp(mp, model, enorm=m.hnorm, hnorm=m.enorm)


def _plant_hidden_not_normed(mp, model, M) -> None:
    """hnorm skipped: the trunk's hidden state is already normed, so one might not norm it again."""
    hnorm = model.model._decode._mtp.hnorm.data_ptr()
    rmsnorm = M.flashinfer_rmsnorm

    def planted(x, weight, eps):
        return x if weight.data_ptr() == hnorm else rmsnorm(x, weight, eps)

    mp.setattr(M, "flashinfer_rmsnorm", planted)


def _plant_routed_after_attention(mp, model, M) -> None:
    """The MoE routed on the shared expert's input -- the block's input after attention, as a
    sequential `MoE(mixer(x))` block would route -- instead of the block's input. Planted in the
    MoE each target's MTP layer calls, `_moe`; any further arguments (the collectives' sizes under
    attention DP) pass through."""
    for target in (model.model._prefill, model.model._decode):
        orig = target._moe
        mp.setattr(
            target,
            "_moe",
            lambda routed, shared, m, *rest, _orig=orig: _orig(shared, shared, m, *rest),
        )


def _plant_output_not_normed(mp, model, M) -> None:
    """The layer returning its residual stream un-normed, leaving ``final_layernorm`` to the head
    (deepseek's MTP layer's convention, not this checkpoint's)."""
    final = model.model._decode._mtp.final_norm.data_ptr()
    fused = M.flashinfer_fused_add_rmsnorm

    def planted(x, residual, weight, eps):
        out, res = fused(x, residual, weight, eps)
        return (res.clone(), res) if weight.data_ptr() == final else (out, res)

    mp.setattr(M, "flashinfer_fused_add_rmsnorm", planted)


def _plant_attention_scale_of_head_128(mp, model, M) -> None:
    """The softmax scale of a 128-wide head (1/sqrt(128)) where the heads are 256 wide (1/16):
    thop_attention scales by 1 / (q_scaling * sqrt(head_size))."""
    mp.setitem(M._CALL_CONSTANTS, "q_scaling", 0.5**0.5)


RIVALS = {
    "attention_scale_of_head_128": _plant_attention_scale_of_head_128,
    "eh_halves_swapped": _plant_eh_halves_swapped,
    "norms_swapped": _plant_norms_swapped,
    "hidden_not_normed": _plant_hidden_not_normed,
    "routed_after_attention": _plant_routed_after_attention,
    "output_not_normed": _plant_output_not_normed,
}

RIVALS_BREAK = {
    "attention_scale_of_head_128": ("output", "attention"),
    "eh_halves_swapped": ("output", "eh_proj"),
    "norms_swapped": ("output", "enorm", "hnorm"),
    "hidden_not_normed": ("output", "hnorm"),
    "routed_after_attention": ("output", "moe"),
    "output_not_normed": ("output", "final"),
}
"""The gates each rival must break: the output's, and that of the op it sits in."""


def run_rival(name: str, M, model, driver_factory, comparator_factory, log=print) -> dict:
    """A small campaign with one mistake planted: a varlen prefill of every short prompt's first
    half and a long prompt's 2048-token chunk, then two verify steps of all of them.
    ``driver_factory()`` returns the `Driver` to run it with."""
    with pytest.MonkeyPatch.context() as mp:
        RIVALS[name](mp, model, M)
        driver = driver_factory()
        sequences = driver.sequences
        comparator = comparator_factory(driver.layer)
        driver.on_free = comparator.forget
        cells: dict = {}
        long_ = _long(sequences)
        short = [i for i in range(len(sequences)) if i not in long_]
        requests = [driver.new_request(i, _half(len(sequences[i].ids))) for i in short]
        requests.append(driver.new_request(long_[0], CHUNK, stop=CHUNK + 64))
        rows = [driver.context_row(r, CHUNK) for r in requests]
        comparator.compare(driver.step(rows, name), cells)
        for _ in range(2):
            walking = [r for r in requests if not r.done]
            comparator.compare(driver.step([driver.verify_row(r) for r in walking], name), cells)
        driver.free(requests)
    log(f"rival {name} done")
    return cells


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------


def _ulps(err: dict) -> str:
    return f"{err['worst_row'] / BF16_ULP:.2f} / {err['rel_rms'] / BF16_ULP:.3f}"


def _quantiles(values) -> dict:
    t = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.5, 0.99, 0.999], dtype=torch.float64)).tolist()
    return dict(p50=q[0], p99=q[1], p999=q[2], max=float(t.max()))


def by_depth(cells: dict) -> dict:
    """Every cell of one depth merged: the drafts' agreement with R_own, the MTP logits' distances
    (the epsilon `mtp-acceptance` needs) and the distribution of the gain on R_own's top token."""
    out = {}
    for depth in (1, 2):
        mine = [c for (_, d), c in cells.items() if d == depth]
        if not mine:
            continue
        rows = sum(c.rows for c in mine)
        out[depth] = dict(
            rows=rows,
            drafts_agree={
                side: sum(c.drafts[f"{side}|R_own"] for c in mine) / rows for side in ("T", "H_own")
            },
            draft_flips={
                side: sorted(f for c in mine for f in c.draft_flips.get(side, []))
                for side in ("T", "H_own")
            },
            logits={
                pair: max(c.logits[pair] for c in mine)
                for pair in (f"{a}|{b}" for a, b in OUTPUT_PAIRS)
            },
            logit_t_own_same_routing=max(c.logit_t_own_same_routing for c in mine),
            logit_t_own_other_routing=max(c.logit_t_own_other_routing for c in mine),
            routing_vs_own={
                side: sum(c.routing_vs_own[side] for c in mine) for side in ("T", "H_own")
            },
            gain={
                side: _quantiles([g for c in mine for g in c.gains[side]])
                for side in ("T", "H_own")
            },
        )
    return out


def report(cells: dict) -> str:
    """One line per (cell, depth): the output's distances in bf16 ulps (element of the row's max /
    relative RMS), the MTP logits' largest differences, the drafts and the routing; then each
    op's distance, HF's own op's beside it; then the depths' totals."""
    lines = []
    for (kind, depth), c in cells.items():
        o, lg, dr = c.output, c.logits, c.drafts
        legal = sum(margin <= bound for margin, bound in c.router_flips)
        lines.append(
            f"{kind} depth {depth}: rows {c.rows} steps {c.steps} batch<={c.max_batch} "
            f"pos<={c.max_position} | T-R_t {_ulps(o['T|R_t'])}, H_t-R_hf_t "
            f"{_ulps(o['H_t|R_hf_t'])}, T-R_hf_t {_ulps(o['T|R_hf_t'])}, T-R_own "
            f"{_ulps(o['T|R_own'])}, H_own-R_own {_ulps(o['H_own|R_own'])} | logits T-R_t "
            f"{lg['T|R_t']:.4f} H_t-R_hf_t {lg['H_t|R_hf_t']:.4f} T-R_own {lg['T|R_own']:.4f} "
            f"H_own-R_own {lg['H_own|R_own']:.4f} | drafts T=R_own {dr['T|R_own']}/{c.rows} "
            f"H_own=R_own {dr['H_own|R_own']}/{c.rows} | router flips {len(c.router_flips)} "
            f"({legal} near-ties), routing vs R_own T {c.routing_vs_own['T']} H_own "
            f"{c.routing_vs_own['H_own']}"
        )
        lines.append(
            "    "
            + ", ".join(
                f"{k} {_ulps(v)}" + (f" (H {_ulps(c.stages_hf[k])})" if k in c.stages_hf else "")
                for k, v in c.stages.items()
            )
        )
    for depth, t in by_depth(cells).items():
        lines.append(
            f"depth {depth} total: {json.dumps({k: v for k, v in t.items() if k != 'draft_flips'})}"
        )
        for side, flips in t["draft_flips"].items():
            margins = sorted(m for m, _ in flips)
            if margins:
                lines.append(
                    f"    {side} drafts differing from R_own: {len(margins)}, R_own margin there "
                    f"median {margins[len(margins) // 2]:.4f} max {margins[-1]:.4f}"
                )
    return "\n".join(lines)


KV_TOKENS = 1 << 20
"""KV pool: every request of the campaign stays resident (~1 KiB a token per layer for the one KV
head of 256 in bf16)."""

CELL_KINDS = (
    "prefill_8k",
    "prefill_chunked",
    "prefill_varlen",
    "mixed",
    "decode_walk",
    "prefill_pack",
) + tuple(f"decode_b{n}" for n in DECODE_BATCHES)
