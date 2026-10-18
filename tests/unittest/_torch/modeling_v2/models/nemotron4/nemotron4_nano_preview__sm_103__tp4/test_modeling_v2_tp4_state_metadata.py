# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nemotron4 tp4 target's state metadata and state views, on one rank.

`Nemotron4StateMetadata.prepare` runs on a stand-in for one tensor-parallel
rank's hybrid cache manager: the slot lookup and the per-layer state pools at a
tp4 rank's state shapes -- 6 of the 24 GDP heads, 4224 of the 16896 conv
channels -- with a slot stride wider than one state (the managers pack state in
blocks). `state_views` is held to the rank's shapes. The engine path itself (the
LLM API on 4 ranks, CUDA graphs, the overlap scheduler, every rank's slots in
the same step) is exercised through the LLM API on a 4-GPU node, not here.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4.modeling import (  # noqa: E501
    Nemotron4NanoPreviewTp4ModelingV2,
    Nemotron4StateMetadata,
    _rank_state_shapes,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm.mapping import Mapping

assert torch.cuda.is_available(), "Nemotron4StateMetadata keeps its buffers on a CUDA device"

_GDP_LAYERS = (0, 2, 6)
_ATTENTION_LAYERS = (5, 9)
#: A tp4 rank's state per slot and GDP layer (tp4-scaffold's pool probe measured these).
_SSM = (6, 64, 128)
_CONV = (4224, 3)
_NUM_SLOTS = 12
_MAX_BATCH = 8
#: The padding dummy's own slot; every padding row is that one request.
_PADDING_SLOT = 11
_PAD = CUDA_GRAPH_DUMMY_REQUEST_ID

# The GDP geometry of nemotron4-nano-preview's text_config, as its config.json declares it.
_GDP_CONFIG = dict(
    gdp_num_heads=24,
    gdp_num_groups=24,
    gdp_head_dim=64,
    gdp_key_head_dim=128,
    gdp_num_householder=3,
    gdp_conv_kernel=4,
)


class _HybridManager:
    """What the metadata and the state views read off a hybrid cache manager, and
    nothing else.

    Each pool is the first half of a [slots, 2, *state] block, so its slot stride
    is twice one state, and a write that strays past a state lands in the other
    half, where `changed_slots` sees it.
    """

    def __init__(self, slot_of: dict[int, int], draft_len: int = 0, ssm=_SSM, conv=_CONV) -> None:
        self.slot_of = {**slot_of, _PAD - draft_len: _PADDING_SLOT}
        self.mamba_layer_offsets = {layer: i for i, layer in enumerate(_GDP_LAYERS)}
        # Every stateful layer, numbered in layer order, as the hybrid managers do.
        self.layer_offsets = {
            layer: i for i, layer in enumerate(sorted(_GDP_LAYERS + _ATTENTION_LAYERS))
        }
        g = torch.Generator(device="cuda").manual_seed(1234)
        # Every slot non-zero, so a reset is visible.
        self.blocks = [
            torch.randn(_NUM_SLOTS, 2, *ssm, generator=g, device="cuda") + 3.0 for _ in _GDP_LAYERS
        ] + [
            (torch.randn(_NUM_SLOTS, 2, *conv, generator=g, device="cuda") + 3.0).to(torch.bfloat16)
            for _ in _GDP_LAYERS
        ]
        self.ssm = [block[:, 0] for block in self.blocks[: len(_GDP_LAYERS)]]
        self.conv = [block[:, 0] for block in self.blocks[len(_GDP_LAYERS) :]]
        assert not self.ssm[0].is_contiguous() and not self.conv[0].is_contiguous()

    def get_state_indices(self, request_ids, is_padding):
        return [self.slot_of[rid] for rid in request_ids]

    def get_ssm_states(self, layer: int) -> torch.Tensor:
        return self.ssm[self.mamba_layer_offsets[layer]]

    def get_conv_states(self, layer: int) -> torch.Tensor:
        return self.conv[self.mamba_layer_offsets[layer]]

    def snapshot(self) -> list[torch.Tensor]:
        return [block.clone() for block in self.blocks]

    def changed_slots(self, before: list[torch.Tensor]) -> set[int]:
        """Slots whose block differs from `before` anywhere, in either half, in any layer."""
        changed = set()
        for old, new in zip(before, self.blocks):
            diff = (old != new).flatten(1).any(dim=1)
            changed |= set(torch.nonzero(diff).flatten().tolist())
        return changed

    def is_zero(self, slot: int) -> bool:
        return all(bool((t[slot] == 0).all()) for t in self.ssm + self.conv)


def _step(manager, request_ids, seq_lens, num_contexts, cached):
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    return SimpleNamespace(
        seq_lens=lens,
        seq_lens_cuda=lens.cuda(),
        num_contexts=num_contexts,
        num_ctx_tokens=int(lens[:num_contexts].sum()),
        kv_cache_manager=manager,
        request_ids=list(request_ids),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=list(cached)),
    )


def _model_config(tp: int, rank: int):
    """The configuration surface `_rank_state_shapes` reads."""
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(text_config=SimpleNamespace(**_GDP_CONFIG)),
        mapping=Mapping(world_size=tp, rank=rank, tp_size=tp, moe_tp_size=tp, moe_ep_size=1),
    )


def _core(rank: int):
    """The surface `state_views` reads off the core, for TP rank `rank` of 4."""
    return SimpleNamespace(
        gdp_layers=_GDP_LAYERS,
        attention_layers=_ATTENTION_LAYERS,
        state_shapes=_rank_state_shapes(_model_config(4, rank)),
        tp_rank=rank,
        tp_size=4,
        _checked_manager=None,
    )


@pytest.mark.parametrize("rank", range(4))
def test_a_rank_holds_a_quarter_of_the_heads_and_of_the_conv_channels(rank):
    """Every tp4 rank runs 6 of the 24 GDP heads, [6, 64, 128], and their 4224 of
    the 16896 conv channels; on one rank the same formula gives the whole state
    tp1's managers hold."""
    assert _rank_state_shapes(_model_config(4, rank)) == (_SSM, _CONV)
    assert _rank_state_shapes(_model_config(1, 0)) == ((24, 64, 128), (16896, 3))


def test_state_views_take_the_ranks_pools_and_refuse_the_whole_state():
    """The views map each GDP layer to the manager's pool and each attention layer
    to its index among every stateful layer. A manager that holds the whole
    model's state per slot -- the engine not splitting it over the ranks -- is
    refused before any entry reads it with a 6-head layout."""
    manager = _HybridManager({})
    core = _core(rank=1)
    views = Nemotron4NanoPreviewTp4ModelingV2.state_views(core, manager)
    # Stateful layers 0, 2, 5, 6, 9 in order: attention layer 5 is the third, 9 the fifth.
    assert views.kv_layer == {5: 2, 9: 4}
    assert all(views.ssm_states[i] is manager.get_ssm_states(i) for i in _GDP_LAYERS)
    assert all(views.conv_states[i] is manager.get_conv_states(i) for i in _GDP_LAYERS)
    assert core._checked_manager() is manager

    whole = _HybridManager({}, ssm=(24, 64, 128), conv=(16896, 3))
    with pytest.raises(RuntimeError, match="must split the state over the tensor-parallel ranks"):
        Nemotron4NanoPreviewTp4ModelingV2.state_views(_core(rank=1), whole)


def test_state_views_refuse_a_manager_without_the_gdp_layers():
    """A manager the engine built without recognizing the hybrid keeps no
    recurrent state for this target's GDP layers."""
    manager = _HybridManager({})
    manager.mamba_layer_offsets = {}
    with pytest.raises(RuntimeError, match="must recognize the checkpoint as a Nemotron 4 hybrid"):
        Nemotron4NanoPreviewTp4ModelingV2.state_views(_core(rank=0), manager)


def test_padding_rows_read_pad_slot_id_and_keep_the_padding_slot():
    """A decode step padded up to a graph batch: padding rows get PAD_SLOT_ID, the
    padding dummy's slot is left alone although its rows have no cached token,
    and the buffer the entries read keeps its address."""
    manager = _HybridManager({17: 4, 13: 1, 15: 9})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    before = manager.snapshot()
    md.prepare(_step(manager, [17, 13, 15, _PAD, _PAD], [1] * 5, 0, [40, 41, 42, 0, 0]))
    ptr = md.state_slots.data_ptr()
    torch.cuda.synchronize()
    assert md.state_slots[:5].tolist() == [4, 1, 9, PAD_SLOT_ID, PAD_SLOT_ID]
    assert md.state_indices[:5].tolist() == [4, 1, 9, _PADDING_SLOT, _PADDING_SLOT]
    assert manager.changed_slots(before) == set()

    md.prepare(_step(manager, [13, 15, _PAD, _PAD], [1] * 4, 0, [42, 43, 0, 0]))
    torch.cuda.synchronize()
    assert md.state_slots.data_ptr() == ptr
    assert md.state_slots[:4].tolist() == [1, 9, PAD_SLOT_ID, PAD_SLOT_ID]


def test_padding_ids_shift_by_the_bound_draft_length():
    """With draft tokens the padding dummy's id is CUDA_GRAPH_DUMMY_REQUEST_ID less
    the draft length, which neither the engine nor the V2 manager hands the
    metadata: the class bound to the target's draft length marks those rows, and
    the unbound one, for a target without drafts, does not."""
    bound = Nemotron4StateMetadata.for_draft_len(2)
    assert bound.max_draft_len == 2 and Nemotron4StateMetadata.max_draft_len == 0
    assert issubclass(bound, Nemotron4StateMetadata)
    assert bound.__name__ == Nemotron4StateMetadata.__name__

    manager = _HybridManager({17: 4}, draft_len=2)
    md = bound(_MAX_BATCH, 128, max_num_tokens=8192)
    md.prepare(_step(manager, [17, _PAD - 2, _PAD - 2], [1, 1, 1], 0, [40, 0, 0]))
    torch.cuda.synchronize()
    assert md.state_slots[:3].tolist() == [4, PAD_SLOT_ID, PAD_SLOT_ID]

    unbound = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    unbound.prepare(_step(manager, [17, _PAD - 2], [1, 1], 0, [40, 41]))
    torch.cuda.synchronize()
    assert unbound.state_slots[:2].tolist() == [4, _PADDING_SLOT]


def test_a_padding_row_between_real_rows_is_marked_where_it_sits():
    """Padding rows are marked row by row, not as a trailing block: a padding row
    ordered before a real one keeps the real rows' slots on either side."""
    manager = _HybridManager({17: 4, 18: 5})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    before = manager.snapshot()
    md.prepare(_step(manager, [17, _PAD, 18, _PAD], [1] * 4, 0, [40, 0, 41, 0]))
    torch.cuda.synchronize()
    assert md.state_slots[:4].tolist() == [4, PAD_SLOT_ID, 5, PAD_SLOT_ID]
    assert manager.changed_slots(before) == set()


def test_a_fresh_one_token_prompt_in_a_decode_batch_gets_a_zeroed_slot():
    """The runner promotes a fresh one-token prompt into a generation row with
    no cached token. Its recycled slot holds another request's state; prepare
    zeroes it in every GDP layer and touches no other slot -- not the padding
    slot, whose rows have no cached token either."""
    manager = _HybridManager({17: 4, 19: 6})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    before = manager.snapshot()
    assert not manager.is_zero(6)
    md.prepare(_step(manager, [17, 19, _PAD, _PAD], [1] * 4, 0, [12, 0, 0, 0]))
    torch.cuda.synchronize()
    assert md.state_slots[:4].tolist() == [4, 6, PAD_SLOT_ID, PAD_SLOT_ID]
    assert manager.changed_slots(before) == {6}
    assert manager.is_zero(6)
    # Only the state half of slot 6's blocks: the other half is untouched.
    assert all(
        bool((block[6, 1] == before[i][6, 1]).all()) for i, block in enumerate(manager.blocks)
    )


def test_context_rows_leave_their_slots_to_has_initial_state():
    """A mixed step: a fresh prompt, a chunk continuing 6144 cached tokens and a
    generation row. Context slots are never reset here -- has_initial_state
    tells the entries whether to read them -- and the starts of every row's
    tokens are in query_start_loc."""
    manager = _HybridManager({21: 2, 22: 3, 17: 4})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    before = manager.snapshot()
    md.prepare(_step(manager, [21, 22, 17], [37, 2048, 1], 2, [0, 6144, 90]))
    torch.cuda.synchronize()
    assert md.state_slots[:3].tolist() == [2, 3, 4]
    assert md.has_initial_states[:2].tolist() == [False, True]
    assert md.query_start_loc.tolist() == [0, 37, 2085, 2086]
    assert manager.changed_slots(before) == set()


def test_a_captured_graph_reads_the_slots_and_the_reset_of_the_step_being_replayed():
    """`prepare` refills state_slots in place and resets fresh slots before every
    replay, so a graph captured on one batch reads each later batch's slots,
    its padding rows' PAD_SLOT_ID and a fresh row's zeroed state."""
    manager = _HybridManager({11: 0, 12: 5, 13: 7, 14: 8, 19: 6})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    md.prepare(_step(manager, [11, 12, 13, 14], [1] * 4, 0, [5, 5, 5, 5]))
    counter = manager.get_ssm_states(_GDP_LAYERS[0])[:, 0, 0, 0]
    stale = counter.clone()
    seen_slots = torch.zeros(4, dtype=torch.int32, device="cuda")
    seen_state = torch.zeros(4, dtype=torch.float32, device="cuda")

    def read_step():
        slots = md.state_slots[:4]
        seen_slots.copy_(slots)
        seen_state.copy_(torch.where(slots >= 0, counter[slots.clamp(min=0).long()], -1.0))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        read_step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        read_step()

    # Request 19 is a fresh one-token prompt promoted into the batch, on slot 6.
    md.prepare(_step(manager, [14, 19, _PAD, _PAD], [1] * 4, 0, [6, 0, 0, 0]))
    graph.replay()
    torch.cuda.synchronize()
    assert seen_slots.tolist() == [8, 6, PAD_SLOT_ID, PAD_SLOT_ID]
    assert seen_state.tolist() == [stale[8].item(), 0.0, -1.0, -1.0]


@pytest.mark.parametrize("cached_as_tensor", [False, True], ids=["list", "tensor"])
def test_cached_counts_as_a_tensor_or_a_list(cached_as_tensor):
    """The engine hands num_cached_tokens_per_seq over as a list or a host tensor."""
    manager = _HybridManager({17: 4, 19: 6})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    step = _step(manager, [17, 19], [1, 1], 0, [3, 0])
    if cached_as_tensor:
        step.kv_cache_params.num_cached_tokens_per_seq = torch.tensor([3, 0], dtype=torch.int32)
    md.prepare(step)
    torch.cuda.synchronize()
    assert manager.is_zero(6) and not manager.is_zero(4)
