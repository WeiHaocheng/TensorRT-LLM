# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module accuracy of the tp1 target's MTP layer, on the checkpoint's own weights.

What runs is the target's MTP layer as the engine runs it under one-model MTP-Eagle, at
`max_draft_len` 2 (the trained depth) and 5 (tp1's longest), one campaign each (`DRAFTS`):

* the shell `ModelingV2Nemotron4NanoPreviewSm103Tp1`, built by its own constructor with the
  resolved MTP-Eagle config, so `DraftModel`, `MTPLayer` and the spec worker are the served
  engine's. Only the MTP layer (global layer 44), the embedding and the lm_head are real; every
  trunk weight stays on the meta device. The MTP rows are filled from the checkpoint through the
  target's own weight table (`weights.MODEL_WEIGHTS.manifest`), the way `load` fills them;
* the engine's own separate draft KV manager for layer 44 (`_create_kv_cache_manager` with
  `is_draft`, as `_create_one_model_draft_kv_cache_manager` builds it on tp1) and the engine's
  TRTLLM attention metadata over it, filled per step as the executor's runner fills it;
* the worker's own draft loop (`Eagle3OneModelWorker._forward_linear_draft_loop`): step 0 over
  every row of the step through `draft_model.mtp_layers[0]` (context chunks, and the 1 + D tokens of
  each verify row), the gather of one row per request, the worker's in-place rewrite of the
  metadata, and steps 1 .. D - 1 (depths 2 .. D) over one row per request against the depth-1 KV
  and what the step's earlier depths wrote after it.

The inputs are teacher forced on real text: the fixed prompt set, and the trunk's final normed
hidden state at every position from the plain-torch reference (`_nemotron4_reference`, the
transcription of `modeling_nemotron4.py`). Depth k at position i reads token i + k: at step 0
every row reads the token after it, and step k - 1 reads the true token k positions after the
depth-1 draft's position (the worker's sample at every step but the last is recorded as that
depth's draft and replaced by the true token, as the reference's teacher forcing does; greedy
acceptance never sees the difference, because a depth-k draft only counts when the shallower
drafts were the true tokens). Which positions a request visits follows the engine: a verify at q
accepts what the target's own drafts got right, the gather row is q + accepted - 1, and the next
verify starts after it.

The cells (`run_campaign`), each compared at every depth (depth k is step k - 1):

* ``prefill_8k``: a long prompt's 7680-token context, fresh, in one step;
* ``prefill_chunked``: another's in 2048-token chunks, each over the earlier chunks' KV;
* ``prefill_varlen``: the short prompts' first halves in one step (at five drafts, the five that
  leave room for a draft window);
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
  what is left is the kernels' accumulation order. Depth k >= 2 reads the hidden state the
  target's depth k - 1 produced, over the reference's own depth-1 keys and values and the ones the
  same reference computed at depths 2 .. k - 1 of the step.
* `R_hf_t` / `H_t`: the reference at HF's rounding points, and HF's own implementation (bf16
  cuBLAS GEMMs, bf16 SDPA, HF's expert loop), on the same inputs with the same routing. `H_t`
  against `R_hf_t` calibrates every band, which must admit the code the model was trained with.
* `R_own` / `H_own`: the reference's own drafting chain (`mtp_draft_hidden(mode="eagle")`) and
  HF's, each routing itself -- what `mtp-acceptance` compares the engine's drafts with.

The campaign, the references, the bands and the rivals are the family's harness,
`_nemotron4_mtp_layer.py`, which the dep4 and tp4 targets' MTP-layer tests run too; this file
builds the tp1 shell, its draft KV manager and its metadata, and runs the harness over them.

Gates: per cell and depth (`failures`), the output against `R_t` within `OUTPUT_BAND`, the MTP
logits within `LOGIT_BAND`, the router exact on its own input up to near-ties, and each op of
`STAGES` against its entry's reference on the target's own input within `STAGE_BANDS`; per depth,
the MTP logits no further from `R_own` than `EPSILON_VS_HF` times HF's. Six planted mistakes must
each break the output gate and the gate of the op they sit in.

Measured (in bf16 ulps: element of the row's largest value / relative RMS, the largest cell;
logits in fp32 units). Two drafts, 49,877 depth-1 and 4,098 depth-2 rows:

==========================================  =====================  =====================
                                            depth 1                depth 2
==========================================  =====================  =====================
output, target vs `R_t`                     10.97 / 2.14           4.17 / 2.15
output, HF vs `R_hf_t`                      15.80 / 3.59           3.63 / 2.77
max |MTP logit diff|, target vs `R_t`       0.419                  0.262
max |MTP logit diff|, target vs `R_own`     2.681                  2.485
  ... on rows routed to `R_own`'s experts   0.862                  0.845
max |MTP logit diff|, HF vs `R_own`         2.124                  2.409
drafts equal to `R_own`'s: target / HF      99.64% / 99.88%        99.17% / 99.32%
==========================================  =====================  =====================

Five drafts, 59,992 depth-1 rows and 4,034 at each deeper depth:

==========================================  ============  ============  ============  ============
                                            depth 2       depth 3       depth 4       depth 5
==========================================  ============  ============  ============  ============
output, target vs `R_t`                     5.50 / 2.01   3.87 / 1.95   2.64 / 2.30   3.37 / 2.53
output, HF vs `R_hf_t`                      5.77 / 1.91   4.69 / 2.98   5.16 / 2.42   5.58 / 1.97
max |MTP logit diff|, target vs `R_t`       0.262         0.230         0.308         0.408
max |MTP logit diff|, target vs `R_own`     2.426         3.468         5.451         7.710
  ... on rows routed to `R_own`'s experts   1.454         3.359         4.945         7.710
max |MTP logit diff|, HF vs `R_own`         1.788         3.802         4.271         11.523
drafts equal to `R_own`'s: target / HF      99.18/99.36%  98.91/99.18%  98.49/98.93%  98.44/99.01%
==========================================  ============  ============  ============  ============

(Depth 1 at five drafts measures as at two: 10.97 / 1.82, 15.80 / 2.72, 0.419, 2.681, 0.862, 2.124.)

The router picks the reference router's experts on its own input on every row (one near-tie flip at
depth 5). The target's routing differs from `R_own`'s on 2.2% of depth-1 rows (HF's on 0.07%): all
near-ties, which the target's once-rounded norms upstream of the router resolve differently from
HF's twice-rounded ones; that, not the layer's arithmetic, is where its logits part from `R_own`'s
by more than 0.86. The deeper depths read the chain's own earlier outputs, so those differences
compound: the target's and HF's routing parts from `R_own`'s on 5.2% / 3.7% of depth-3 rows and
6.6% / 5.1% of depth-5 rows, and the distance from `R_own` grows with depth for both, while the
target stays within 0.41 of `R_t` -- the reference on its own inputs -- at every depth.
"""

from __future__ import annotations

import dataclasses
import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1 import (  # noqa: E501
    weights as W,
)

__extra_import_path__ = ["../..", "../../../.."]
import _nemotron4_mtp_layer as H  # noqa: E402 — needs the path declared above
import _nemotron4_prompts as P  # noqa: E402
import _nemotron4_reference as R  # noqa: E402

assert torch.cuda.is_available(), "the nemotron4 MTP layer test requires a CUDA device"


def _checkpoint_or_skip() -> str:
    root = os.environ.get("LLM_MODELS_ROOT")
    path = os.path.join(root, "nemotron4-nano-preview") if root else None
    if not path or not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
        pytest.skip("needs the checkpoint at $LLM_MODELS_ROOT/nemotron4-nano-preview")
    return path


# ----------------------------------------------------------------------------------------------
# The target, its draft KV manager and its attention metadata, built as the engine builds them
# ----------------------------------------------------------------------------------------------


def _mapping():
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)


def engine_config(path: str, draft: int = H.DRAFT):
    """The `ModelConfig` the engine hands the target under `MTPDecodingConfig(max_draft_len=draft)`:
    the checkpoint's own config, the speculative config resolved as the model loader resolves it
    (one-model MTP-Eagle, one MTP layer serving every depth), tp1."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm._torch.speculative.utils import resolve_mtp_checkpoint_source
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    config = ModelConfig.from_pretrained(path, trust_remote_code=True)
    validate_and_set_mamba_ssm_cache_dtype(config, "auto")
    composite = config.pretrained_config
    # The shell sets this on the composite config (the text config derives it).
    composite.num_hidden_layers = composite.text_config.num_hidden_layers
    spec = MTPDecodingConfig(max_draft_len=draft)
    resolve_mtp_checkpoint_source(spec, path)
    update_spec_config_from_model_config(spec, composite)
    assert spec.spec_dec_mode.is_mtp_eagle_one_model(), spec.spec_dec_mode
    assert spec.max_draft_len == draft
    return dataclasses.replace(config, mapping=_mapping(), spec_config=spec)


def build_mtp_target(config, checkpoint: R.Checkpoint):
    """The tp1 shell under one-model MTP-Eagle, its MTP layer, embedding and lm_head real.

    Every parameter is declared on the meta device by the shell's own constructor; the MTP
    layer's rows (`l44_*`) and the embedding are filled from the checkpoint through the weight
    table's manifest -- each row's checkpoint keys, index and transform, asserting shape and dtype
    as `load`'s `fill` does -- and the lm_head from `lm_head.weight`. The targets are then built by
    the core's own `post_load_weights`.

    Returns:
        ``(the shell, the number of checkpoint tensors filled)``.
    """
    with torch.device("meta"):
        model = M.ModelingV2Nemotron4NanoPreviewSm103Tp1(config)
    core = model.model
    table = W.MODEL_WEIGHTS
    d = table.dims(core)
    assert d.mtp_layers == 1, d.mtp_layers
    mtp_rows = f"l{d.num_layers}_"
    filled = 0
    for key, sources in table.manifest(core).items():
        if not (key.startswith(mtp_rows) or key == "embed"):
            continue
        param = torch.empty(core.w[key].shape, dtype=core.w[key].dtype, device="cuda")
        for ckpt_key, index, transform in sources:
            src = checkpoint.tensor(ckpt_key)
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
    filled += 1
    core.post_load_weights()
    assert model.spec_worker is not None and model.draft_model is not None
    return model, filled


def build_draft_manager(config, max_batch: int, max_tokens: int):
    """The engine's separate one-model draft KV manager on tp1: KV for the MTP layer (global
    layer 44) alone, built as `_create_one_model_draft_kv_cache_manager` builds it."""
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.speculative import get_num_spec_layers
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    kv_cache_config = KvCacheConfig(
        max_tokens=max_tokens, enable_block_reuse=False, free_gpu_memory_fraction=0.9
    )
    num_layers = config.pretrained_config.num_hidden_layers
    num_draft = get_num_spec_layers(config.spec_config)
    manager = _create_kv_cache_manager(
        None,
        get_kv_cache_manager_cls(config, kv_cache_config),
        _mapping(),
        kv_cache_config,
        tokens_per_block=H.TOKENS_PER_BLOCK,
        max_seq_len=H.MAX_SEQ_LEN,
        max_batch_size=max_batch,
        spec_config=config.spec_config,
        sparse_attention_config=None,
        max_num_tokens=H.MAX_NUM_TOKENS,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=config,
        dtype=torch.bfloat16,
        max_cuda_graph_batch_size=max_batch,
        is_draft=True,
        layer_mask=[False] * num_layers + [True] * num_draft,
        num_layers=num_draft,
    )
    assert manager.layer_offsets[num_layers] == 0, manager.layer_offsets
    return manager


def build_metadata(config, manager, max_batch: int):
    """The engine's TRTLLM attention metadata, over the draft manager.

    In the engine the worker swaps the draft manager, its block offsets and host block offsets into
    the trunk's prepared metadata for the draft steps (`prepare_attn_metadata_for_draft_replay`);
    the pool pointers are read off whichever manager the metadata carries. Metadata prepared over
    the draft manager is that swapped state. The MTP layer keeps no recurrent state, so no state
    metadata is prepared (`mamba_metadata` False): in the engine the trunk's manager prepares it.
    """
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata

    md = build_attention_metadata(
        config,
        max_batch_size=max_batch,
        max_num_tokens=H.MAX_NUM_TOKENS,
        max_beam_width=1,
        attention_backend=get_attention_backend("TRTLLM"),
        attention_runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=True,
            cache_reuse=False,
            has_speculative_draft_tokens=True,
            chunk_size=H.MAX_NUM_TOKENS,
        ),
        mapping=_mapping(),
        cache_indirection=None,
        kv_cache_manager=manager,
    )
    md.mamba_metadata = False
    return md


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------

DRAFTS = (2, 5)
"""The draft lengths a campaign runs at: the checkpoint's trained depth, and tp1's longest, whose
campaign compares depths 1 to 5. Draft lengths 3 and 4 run the same draft steps; their step 0 reads
four and five tokens a verify row (thop_attention's verify cells)."""

DEPTHS = tuple(range(1, max(DRAFTS) + 1))

MIN_FREE_GIB = 48
"""The trunk's layers one at a time (up to ~7 GB of weights each), the MTP layer twice (the
target's and the reference's copy, ~7 GB each), the lm_head in fp32, the references of 32K
positions and the draft KV pool: measured 28.4 GiB at torch's peak, 31.7 GiB of the device."""


@pytest.fixture(scope="module", params=DRAFTS, ids=[f"draft{d}" for d in DRAFTS])
def harness(request):
    draft = request.param
    path = _checkpoint_or_skip()
    free, _ = torch.cuda.mem_get_info()
    if free < MIN_FREE_GIB * 2**30:
        pytest.skip(f"needs {MIN_FREE_GIB} GiB of free GPU memory, {free / 2**30:.0f} GiB is")
    checkpoint = R.Checkpoint(path)
    config = checkpoint.config
    prompts = P.prompt_set(P.load_tokenizer(path))
    hidden = H.trunk_hidden(checkpoint, prompts)
    sequences = [
        H.Sequence(p.name, list(p.token_ids), torch.tensor(p.token_ids, device="cuda"), h.cuda())
        for p, h in zip(prompts, hidden)
    ]
    engine = engine_config(path, draft)
    model, _ = build_mtp_target(engine, checkpoint)
    embed = checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")
    weights = checkpoint.mtp_layer()
    seqrefs = [H.sequence_reference(s, embed, weights, config, draft) for s in sequences]
    max_batch = max(H.DECODE_BATCHES)
    manager = build_draft_manager(engine, max_batch, H.KV_TOKENS)
    md = build_metadata(engine, manager, max_batch)
    yield SimpleNamespace(
        draft=draft,
        checkpoint=checkpoint,
        config=config,
        model=model,
        manager=manager,
        md=md,
        sequences=sequences,
        seqrefs=seqrefs,
        embed=embed,
        weights=weights,
        lm_head=checkpoint.tensor(R.LM_HEAD_KEY),
        max_batch=max_batch,
    )
    manager.shutdown()


def _comparator(harness, layer) -> H.Comparator:
    return H.Comparator(
        harness.sequences,
        harness.seqrefs,
        harness.embed,
        harness.weights,
        harness.lm_head,
        layer,
        harness.config,
    )


@pytest.fixture(scope="module")
def campaign(harness):
    driver = H.Driver(
        harness.model, M, harness.manager, harness.md, harness.sequences, harness.max_batch
    )
    cells = H.run_campaign(driver, _comparator(harness, driver.layer))
    free, total = torch.cuda.mem_get_info()
    print(
        "\n" + H.report(cells) + f"\nGPU memory: torch peak "
        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, device in use "
        f"{(total - free) / 2**30:.1f} GiB"
    )
    return cells


@pytest.mark.high_cuda_memory
@torch.inference_mode()
def test_reference_pieces_are_mtp_draft_hidden(harness):
    """`sequence_reference`'s `R_own` -- the MTP layer split at its MoE so the routing can be swapped
    -- is `R.mtp_draft_hidden(mode="eagle")` bit for bit at every depth, on a short prompt of the
    set with the trunk's own hidden states."""
    seq = next(s for s in harness.sequences if s.name == "clips")
    whole = R.mtp_draft_hidden(
        seq.ids_cuda,
        seq.hidden,
        harness.embed,
        harness.weights,
        harness.config,
        num_depths=harness.draft,
        mode="eagle",
    )
    pieces = H.sequence_reference(
        seq, harness.embed, harness.weights, harness.config, harness.draft
    )
    assert len(pieces.own) == len(whole) == harness.draft
    for depth, (piece, ref) in enumerate(zip(pieces.own, whole), start=1):
        assert torch.equal(piece, ref), depth


def _depth_or_skip(harness, depth: int) -> None:
    if depth > harness.draft:
        pytest.skip(f"a campaign at {harness.draft} drafts has no depth {depth}")


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("kind", H.CELL_KINDS)
def test_mtp_layer(harness, campaign, kind, depth):
    """Every step of the cell: the MTP layer's output against `R_t`, its logits, its router and
    each of its ops within the bands (`failures`)."""
    _depth_or_skip(harness, depth)
    cell = campaign[(kind, depth)]
    assert cell.rows > 0
    assert not H.failures(cell, depth), H.failures(cell, depth)


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize("depth", DEPTHS)
def test_drafts_against_the_reference_chain(harness, campaign, depth):
    """Every cell of one depth together, each side drafting with its own routing: the target's MTP
    logits are no further from the reference's drafting chain (`R_own`, what `mtp-acceptance`
    compares the engine's drafts with) than `EPSILON_VS_HF` times HF's own implementation's.

    Prints what that gate needs: the largest MTP-logit difference (split by whether the target
    routed the row to the reference's experts), the distribution of how far any token gains on the
    reference's top token, and the draft agreement.
    """
    _depth_or_skip(harness, depth)
    totals = H.by_depth(campaign)[depth]
    flips = totals.pop("draft_flips")
    print(f"\ndraft {harness.draft} depth {depth}: {json.dumps(totals)}")
    for side, rows in flips.items():
        margins = sorted(m for m, _ in rows)
        if margins:
            print(
                f"  {side}: {len(margins)} drafts differ from R_own's, at R_own margins up to "
                f"{margins[-1]:.4f} (median {margins[len(margins) // 2]:.4f})"
            )
    eps_t, eps_h = totals["logits"]["T|R_own"], totals["logits"]["H_own|R_own"]
    assert eps_t <= H.EPSILON_VS_HF * eps_h, (eps_t, eps_h)


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize("rival", sorted(H.RIVALS))
def test_the_gates_refuse_plausible_wrong_layers(harness, rival):
    """Each planted mistake breaks the output band and the gate of the op it sits in (at the
    trained depth; every depth runs the same layer)."""
    if harness.draft != H.DRAFT:
        pytest.skip(f"the rivals run at {H.DRAFT} drafts")
    cells = H.run_rival(
        rival,
        M,
        harness.model,
        lambda: H.Driver(
            harness.model,
            M,
            harness.manager,
            harness.md,
            harness.sequences,
            harness.md.max_num_requests,
        ),
        lambda layer: _comparator(harness, layer),
    )
    broken = set()
    for (_, depth), cell in cells.items():
        broken |= {name.split(":")[0] for name in H.failures(cell, depth)}
    missing = set(H.RIVALS_BREAK[rival]) - broken
    assert not missing, (rival, sorted(broken), sorted(missing))
