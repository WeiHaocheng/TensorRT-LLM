# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: an MTP verify step of the nemotron4 tp1 target's GDP mixer rolls back exactly.

Under one-model MTP-Eagle a generation row carries its last token and D drafts. The target's GDP
mixer runs the row's 1 + D tokens from its state slot without writing it, and stores the state
after every token in the hybrid cache manager's per-position buffers; after acceptance the
engine's worker commits the last accepted token's state into the slot
(`update_mamba_states`). A rejected draft must leave no trace: the committed slot has to be the
state that MTP-off decode reaches over the accepted tokens alone.

Checked here bit for bit, on layer 0's real weights, through the target's own mixer code
(`DecodeTarget._gdp` for a decode step, `PrefillTarget._gdp` for a mixed one) and the engine's
own hybrid managers built with an MTP-Eagle config (`MambaHybridCacheManagerV2`, the default,
and `CppMambaHybridCacheManager`), committing with the manager's `update_mamba_states`, at
both draft lengths tp1 runs (D = 2, the checkpoint's trained depth, and D = 3):

- every real row's committed recurrent and conv state equals the slot after `accepted` decode
  steps (T = 1 calls of the same mixer) over the same tokens, for every accepted count;
- the verify writes no slot; the commit writes the real rows' slots and the padding dummy's
  (whose stale snapshot it copies, harmlessly), and nothing else;
- a second verify from the committed slots still matches decode continued from them;
- the mixer's output rows of the verify equal the decode steps' rows: in_proj, out_proj and
  the gates are row-invariant at these row counts, and the verify entries are bitwise with
  the decode entries.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1.modeling import (  # noqa: E501
    DecodeTarget,
    PrefillTarget,
    _dims,
    _Gdp,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

assert torch.cuda.is_available(), "the GDP mixer and the hybrid cache managers run on a CUDA device"

_LAYER = 0  # a `linear_attention` block; every GDP mixer of the model is the same mixer
#: The draft lengths tp1 runs: the checkpoint's trained depth, and the repeated layer's next one.
_DRAFTS = (2, 3)
_MAX_BATCH = 8
_PREFIX = f"model.language_model.layers.{_LAYER}."


def _checkpoint_or_skip() -> str:
    root = os.environ.get("LLM_MODELS_ROOT")
    path = os.path.join(root, "nemotron4-nano-preview") if root else None
    if not path or not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
        pytest.skip("needs the checkpoint at $LLM_MODELS_ROOT/nemotron4-nano-preview")
    return path


@pytest.fixture(scope="module")
def checkpoint() -> str:
    return _checkpoint_or_skip()


@pytest.fixture(scope="module")
def text_config(checkpoint) -> SimpleNamespace:
    with open(os.path.join(checkpoint, "config.json")) as f:
        return SimpleNamespace(**json.load(f)["text_config"])


@pytest.fixture(scope="module")
def layer(checkpoint):
    """Layer 0's GDP mixer, laid out as `weights.py` lays it out at load; its input norm; and the
    embedding table, which feeds the mixer realistic rows."""
    from safetensors import safe_open

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    def tensor(key: str) -> torch.Tensor:
        with safe_open(os.path.join(checkpoint, weight_map[key]), "pt", device="cpu") as f:
            return f.get_tensor(key).cuda()

    mixer = _Gdp(
        layer=_LAYER,
        in_proj=tensor(_PREFIX + "mixer.in_proj.weight").t(),
        conv=W._conv_taps(tensor(_PREFIX + "mixer.conv1d.weight"), None),
        neg_a=W._neg_a(tensor(_PREFIX + "mixer.A_log"), None),
        dt_bias=tensor(_PREFIX + "mixer.dt_bias"),
        norm=tensor(_PREFIX + "mixer.norm.weight"),
        out_proj=tensor(_PREFIX + "mixer.out_proj.weight").t(),
    )
    return SimpleNamespace(
        mixer=mixer,
        input_norm=tensor(_PREFIX + "input_layernorm.weight"),
        embed=tensor("model.language_model.embed_tokens.weight"),
    )


def _targets(text_config):
    """The two targets' mixer code, without the rest of the model: `_gdp` reads only `_d`."""
    decode, prefill = object.__new__(DecodeTarget), object.__new__(PrefillTarget)
    decode._d = prefill._d = _dims(text_config)
    return decode, prefill


@pytest.fixture(
    scope="module",
    params=[(pool, draft) for draft in _DRAFTS for pool in ("v2", "v1")],
    ids=lambda p: f"{p[0]}-draft{p[1]}",
)
def manager(request, checkpoint):
    """The engine's hybrid cache manager for this checkpoint, built as the engine builds the
    trunk's under one-model MTP-Eagle at draft length D: per-position buffers for 1 + D tokens."""
    pool, draft = request.param
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    model_config = ModelConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config = model_config.pretrained_config
    # The target's shell sets this on the composite config (the text config derives it).
    config.num_hidden_layers = config.text_config.num_hidden_layers
    kv_cache_config = KvCacheConfig(
        max_tokens=64 * 1024,
        enable_block_reuse=False,
        free_gpu_memory_fraction=0.05,
        use_kv_cache_manager_v2=pool == "v2",
    )
    manager = _create_kv_cache_manager(
        None,
        get_kv_cache_manager_cls(model_config, kv_cache_config),
        Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        kv_cache_config,
        tokens_per_block=32,
        max_seq_len=9344,
        max_batch_size=_MAX_BATCH,
        spec_config=MTPDecodingConfig(max_draft_len=draft),
        sparse_attention_config=None,
        max_num_tokens=8192,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        layer_mask=[True] * config.num_hidden_layers,
        max_cuda_graph_batch_size=_MAX_BATCH,
    )
    assert manager.get_intermediate_ssm_states(_LAYER).shape[:2] == (_MAX_BATCH, 1 + draft)
    assert manager.get_intermediate_conv_states(_LAYER).shape[:2] == (_MAX_BATCH, 1 + draft)
    yield manager
    manager.shutdown()


def _pools(manager):
    return manager.get_ssm_states(_LAYER), manager.get_conv_states(_LAYER)


def _tokens(manager) -> int:
    """1 + D: the tokens a verify row carries, the positions the manager keeps per row."""
    return manager.get_intermediate_ssm_states(_LAYER).shape[1]


def _snapshot(manager) -> tuple[torch.Tensor, torch.Tensor]:
    ssm, conv = _pools(manager)
    return ssm.clone(), conv.clone()


def _restore(manager, snapshot) -> None:
    for pool, saved in zip(_pools(manager), snapshot):
        pool.copy_(saved)


def _changed_slots(manager, snapshot) -> set[int]:
    changed = set()
    for pool, saved in zip(_pools(manager), snapshot):
        diff = (pool.view(pool.shape[0], -1) != saved.view(saved.shape[0], -1)).any(dim=1)
        changed |= set(torch.nonzero(diff).flatten().tolist())
    return changed


def _step(manager, slots: list[int], **context) -> dict:
    """What `step_args` hands the mixer for these rows' slots."""
    return dict(
        state_slots=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        ssm_states={_LAYER: manager.get_ssm_states(_LAYER)},
        conv_states={_LAYER: manager.get_conv_states(_LAYER)},
        intermediate_ssm={_LAYER: manager.get_intermediate_ssm_states(_LAYER)},
        intermediate_conv={_LAYER: manager.get_intermediate_conv_states(_LAYER)},
        **context,
    )


def _rows(layer, text_config, generator, num_rows: int) -> torch.Tensor:
    """Layer 0's input for random tokens: their embedding, RMS-normed with its input norm."""
    ids = torch.randint(0, layer.embed.shape[0], (num_rows,), generator=generator, device="cuda")
    e = layer.embed[ids].float()
    e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + text_config.rms_norm_eps)
    return (e.to(torch.bfloat16) * layer.input_norm).contiguous()


def _states(manager, slots: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    ssm, conv = _pools(manager)
    index = torch.tensor(slots, device="cuda")
    return ssm[index].clone(), conv[index].clone()


def _prime(manager, decode, layer, text_config, generator, slots: list[int], steps: int) -> None:
    """Give the slots a history: zero, then `steps` decode steps of random tokens."""
    ssm, conv = _pools(manager)
    index = torch.tensor(slots, device="cuda")
    ssm[index] = 0
    conv[index] = 0
    for _ in range(steps):
        decode._gdp(
            _rows(layer, text_config, generator, len(slots)), layer.mixer, _step(manager, slots)
        )
    primed_ssm, primed_conv = _states(manager, slots)
    assert all(bool(s.any()) and bool(c.any()) for s, c in zip(primed_ssm, primed_conv))


def _decode_reference(manager, decode, layer, x, slots, padded_slots, accepted):
    """MTP-off decode over the verify's tokens, one step per position, from the slots as they
    are now: each real row's state after its accepted tokens, and every step's output rows.

    The states after each number of tokens differ pairwise in every row, recurrent and conv
    state alike, so a commit from any position but the accepted one fails the comparison."""
    num_real, tokens = len(slots), x.shape[1]
    after = []
    outs = []
    for j in range(tokens):
        outs.append(decode._gdp(x[:, j].contiguous(), layer.mixer, _step(manager, padded_slots)))
        after.append(_states(manager, slots))
    for b in range(num_real):
        for i in range(tokens):
            for j in range(i + 1, tokens):
                assert not torch.equal(after[i][0][b], after[j][0][b]), (b, i, j)
                assert not torch.equal(after[i][1][b], after[j][1][b]), (b, i, j)
    taken = [(after[accepted[b] - 1][0][b], after[accepted[b] - 1][1][b]) for b in range(num_real)]
    return taken, torch.stack(outs, dim=1)


def _commit(manager, num_contexts: int, accepted: list[int], engine_slots: list[int]) -> None:
    """The worker's commit: `update_mamba_states` with the engine's own slots (the padding
    dummy's included) and every row's accepted-token count."""
    manager.update_mamba_states(
        SimpleNamespace(num_seqs=len(engine_slots), num_contexts=num_contexts),
        torch.tensor(accepted, dtype=torch.int32, device="cuda"),
        state_indices=torch.tensor(engine_slots, dtype=torch.int32, device="cuda"),
    )


def _assert_states_equal(manager, slots, expected, what: str) -> None:
    ssm, conv = _states(manager, slots)
    for b, (want_ssm, want_conv) in enumerate(expected):
        assert torch.equal(ssm[b], want_ssm), f"{what}: row {b} recurrent state differs"
        assert torch.equal(conv[b], want_conv), f"{what}: row {b} conv state differs"


#: Per T = 1 + D, the accepted counts of the 5 real rows in each of the two verify rounds: every
#: count 1..T in each round.
_ROUNDS = {3: ([3, 1, 2, 3, 1], [1, 2, 3, 2, 3]), 4: ([4, 1, 2, 3, 4], [1, 4, 3, 2, 4])}


@torch.inference_mode()
def test_verify_then_commit_equals_decode_over_the_accepted_tokens(manager, layer, text_config):
    """A padded CUDA-graph-shaped decode step, two verify rounds: 5 requests and 3 padding rows
    (graph batch 8), T = 1 + D, every accepted count 1..T among the rows."""
    decode, _ = _targets(text_config)
    tokens = _tokens(manager)
    g = torch.Generator(device="cuda").manual_seed(4404)
    num_slots = manager.get_ssm_states(_LAYER).shape[0]
    order = torch.randperm(num_slots, generator=torch.Generator().manual_seed(7)).tolist()
    slots, dummy = order[:5], order[5]
    padded = slots + [PAD_SLOT_ID] * 3
    engine_slots = slots + [dummy] * 3
    _prime(manager, decode, layer, text_config, g, slots, steps=6)

    for accepted in _ROUNDS[tokens]:
        start = _snapshot(manager)
        x = _rows(layer, text_config, g, len(padded) * tokens).view(len(padded), tokens, -1)

        out = decode._gdp(x.reshape(-1, x.shape[-1]), layer.mixer, _step(manager, padded))
        assert _changed_slots(manager, start) == set(), "the verify wrote a state slot"
        _commit(manager, 0, accepted + [1, 2, 3], engine_slots)
        committed = _snapshot(manager)
        assert _changed_slots(manager, start) <= set(slots) | {dummy}, (
            "the commit wrote a slot outside the step's rows"
        )

        _restore(manager, start)
        expected, ref_out = _decode_reference(manager, decode, layer, x, slots, padded, accepted)
        _restore(manager, committed)
        _assert_states_equal(manager, slots, expected, f"accepted {accepted}")
        assert torch.equal(out.view(len(padded), tokens, -1)[:5], ref_out[:5]), (
            "the verify's output rows differ from the decode steps' rows"
        )


@torch.inference_mode()
def test_mixed_step_commits_only_its_generation_rows(manager, layer, text_config):
    """A mixed step: a fresh 37-token prompt (context row 0) beside 3 verify rows. The context
    row's tokens go through the prefill entries into its own slot; the generation rows verify
    from theirs, and the commit, offset past the context row, rolls them back exactly."""
    decode, prefill = _targets(text_config)
    tokens = _tokens(manager)
    g = torch.Generator(device="cuda").manual_seed(3707)
    num_slots = manager.get_ssm_states(_LAYER).shape[0]
    order = torch.randperm(num_slots, generator=torch.Generator().manual_seed(11)).tolist()
    context_slot, slots = order[0], order[1:4]
    _prime(manager, decode, layer, text_config, g, slots, steps=4)
    accepted = {3: [2, 3, 1], 4: [4, 1, 3]}[tokens]

    start = _snapshot(manager)
    context = _rows(layer, text_config, g, 37)
    x = _rows(layer, text_config, g, len(slots) * tokens).view(len(slots), tokens, -1)
    starts = torch.tensor([0, 37], dtype=torch.int32, device="cuda")
    step = _step(
        manager,
        [context_slot] + slots,
        num_rows=1 + len(slots),
        num_contexts=1,
        num_ctx_tokens=37,
        query_start_loc=starts,
        query_start_loc_long=starts.long(),
        has_initial_state=torch.zeros(1, dtype=torch.bool, device="cuda"),
    )
    out = prefill._gdp(torch.cat([context, x.reshape(-1, x.shape[-1])]), layer.mixer, step)
    assert _changed_slots(manager, start) == {context_slot}, "the verify wrote a generation slot"
    prefilled = _states(manager, [context_slot])
    _commit(manager, 1, [1] + accepted, [context_slot] + slots)
    committed = _snapshot(manager)
    assert _changed_slots(manager, start) <= {context_slot} | set(slots)
    assert all(torch.equal(a, b) for a, b in zip(_states(manager, [context_slot]), prefilled)), (
        "the commit wrote the context row's slot"
    )

    _restore(manager, start)
    expected, ref_out = _decode_reference(manager, decode, layer, x, slots, slots, accepted)
    _restore(manager, committed)
    _assert_states_equal(manager, slots, expected, f"accepted {accepted} beside a context row")
    assert torch.equal(out[37:].view(len(slots), tokens, -1), ref_out), (
        "the verify's output rows differ from the decode steps' rows"
    )
