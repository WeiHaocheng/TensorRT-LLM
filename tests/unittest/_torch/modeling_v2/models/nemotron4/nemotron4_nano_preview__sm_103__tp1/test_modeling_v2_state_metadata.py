# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nemotron4 tp1 target's state metadata: slots, padding rows, fresh slots.

`Nemotron4StateMetadata.prepare` runs on a stand-in for the hybrid cache
manager: the slot lookup and the per-layer state pools, at this model's state
shapes, with a slot stride wider than one state (the managers pack state in
blocks). The engine path itself (LLM API, both managers, CUDA graphs, overlap
scheduler) is exercised by the target's boot probe, not here.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1.modeling import (  # noqa: E501
    Nemotron4StateMetadata,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

assert torch.cuda.is_available(), "Nemotron4StateMetadata keeps its buffers on a CUDA device"

_GDP_LAYERS = (0, 2, 6)
_SSM = (24, 64, 128)
_CONV = (16896, 3)
_NUM_SLOTS = 12
_MAX_BATCH = 8
#: The padding dummy's own slot; every padding row is that one request.
_DUMMY_SLOT = 11
_PAD = CUDA_GRAPH_DUMMY_REQUEST_ID


class _HybridManager:
    """What the metadata reads off a hybrid cache manager, and nothing else.

    Each pool is the first half of a [slots, 2, *state] block, so its slot stride
    is twice one state, and a write that strays past a state lands in the other
    half, where `changed_slots` sees it.

    Like `MambaHybridCacheManagerV2`, it does not state the draft length
    (`speculative_num_draft_tokens`): the padding dummy of a step whose rows carry
    `draft_len` draft tokens has id `CUDA_GRAPH_DUMMY_REQUEST_ID - draft_len`.
    """

    def __init__(self, slot_of: dict[int, int], draft_len: int = 0) -> None:
        self.slot_of = {**slot_of, _PAD - draft_len: _DUMMY_SLOT}
        self.mamba_layer_offsets = {layer: i for i, layer in enumerate(_GDP_LAYERS)}
        g = torch.Generator(device="cuda").manual_seed(1234)
        # Every slot non-zero, so a reset is visible.
        self.blocks = [
            torch.randn(_NUM_SLOTS, 2, *_SSM, generator=g, device="cuda") + 3.0 for _ in _GDP_LAYERS
        ] + [
            (torch.randn(_NUM_SLOTS, 2, *_CONV, generator=g, device="cuda") + 3.0).to(
                torch.bfloat16
            )
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


def test_padding_rows_read_pad_slot_id_and_keep_the_dummy_slot():
    """A padded decode step: padding rows get PAD_SLOT_ID, the dummy's slot is
    left alone although its rows have no cached token, and the buffer the
    entries read keeps its address from step to step."""
    manager = _HybridManager({7: 4, 3: 1, 5: 9})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    before = manager.snapshot()
    md.prepare(_step(manager, [7, 3, 5, _PAD, _PAD], [1] * 5, 0, [40, 41, 42, 0, 0]))
    ptr = md.state_slots.data_ptr()
    assert md.state_slots[:5].tolist() == [4, 1, 9, PAD_SLOT_ID, PAD_SLOT_ID]
    assert md.state_indices[:5].tolist() == [4, 1, 9, _DUMMY_SLOT, _DUMMY_SLOT]
    assert manager.changed_slots(before) == set()

    md.prepare(_step(manager, [3, 5, _PAD, _PAD], [1] * 4, 0, [42, 43, 0, 0]))
    assert md.state_slots.data_ptr() == ptr
    assert md.state_slots[:4].tolist() == [1, 9, PAD_SLOT_ID, PAD_SLOT_ID]


@pytest.mark.parametrize("runtime_draft_len", [2, 1], ids=["max_draft_len", "shorter_draft"])
def test_padding_ids_shift_by_the_draft_length(runtime_draft_len):
    """With draft tokens the padding dummy's id is CUDA_GRAPH_DUMMY_REQUEST_ID less
    the step's draft length, at most the target's max_draft_len, which the shell
    binds (`for_draft_len`). A verify row carries 1 + D tokens and one slot."""
    manager = _HybridManager({7: 4, 3: 1}, draft_len=runtime_draft_len)
    md = Nemotron4StateMetadata.for_draft_len(2)(_MAX_BATCH, 128, max_num_tokens=8192)
    tokens = 1 + runtime_draft_len
    pad = _PAD - runtime_draft_len
    md.prepare(_step(manager, [7, 3, pad, pad], [tokens] * 4, 0, [40, 41, 0, 0]))
    torch.cuda.synchronize()
    assert md.state_slots[:4].tolist() == [4, 1, PAD_SLOT_ID, PAD_SLOT_ID]
    # The engine's own slots stay with the engine's commit, which writes through them.
    assert md.state_indices[:4].tolist() == [4, 1, _DUMMY_SLOT, _DUMMY_SLOT]


def test_without_the_draft_length_an_mtp_padding_row_writes_the_dummy_slot():
    """The hazard the binding removes: with the draft length unknown (the V2
    manager does not state it), the padding dummy of an MTP step reads as a real
    row, so every padding row would carry the dummy's slot and write it."""
    manager = _HybridManager({7: 4}, draft_len=2)
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    md.prepare(_step(manager, [7, _PAD - 2, _PAD - 2], [3, 3, 3], 0, [40, 0, 0]))
    assert md.state_slots[:3].tolist() == [4, _DUMMY_SLOT, _DUMMY_SLOT]


def test_a_padding_row_between_real_rows_is_refused():
    """Only trailing padding rows are marked; a padding row ordered before a
    real one, a layout the runner does not produce (it appends padding after
    every real row), raises."""
    manager = _HybridManager({7: 4, 8: 5}, draft_len=2)
    md = Nemotron4StateMetadata.for_draft_len(2)(_MAX_BATCH, 128, max_num_tokens=8192)
    with pytest.raises(NotImplementedError, match="padding row precedes a real row"):
        md.prepare(_step(manager, [7, _PAD - 2, 8], [3, 3, 3], 0, [40, 0, 41]))


def test_the_bound_class_is_the_metadata_class():
    """The engine builds the metadata from the class the shell names and checks it
    against `Nemotron4StateMetadata`; the binding keeps the warmup's class reads."""
    bound = Nemotron4StateMetadata.for_draft_len(2)
    assert issubclass(bound, Nemotron4StateMetadata)
    assert bound.max_draft_len == 2 and Nemotron4StateMetadata.max_draft_len == 0
    assert bound.prefill_chunk_alignment == Nemotron4StateMetadata.prefill_chunk_alignment


def test_a_fresh_one_token_prompt_in_a_decode_batch_gets_a_zeroed_slot():
    """The runner promotes a fresh one-token prompt into a generation row with
    no cached token. Its recycled slot holds another request's state; prepare
    zeroes it in every GDP layer and touches no other slot."""
    manager = _HybridManager({7: 4, 9: 6})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    before = manager.snapshot()
    assert not manager.is_zero(6)
    md.prepare(_step(manager, [7, 9, _PAD, _PAD], [1] * 4, 0, [12, 0, 0, 0]))
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
    manager = _HybridManager({11: 2, 12: 3, 7: 4})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    before = manager.snapshot()
    md.prepare(_step(manager, [11, 12, 7], [37, 2048, 1], 2, [0, 6144, 90]))
    torch.cuda.synchronize()
    assert md.state_slots[:3].tolist() == [2, 3, 4]
    assert md.has_initial_states[:2].tolist() == [False, True]
    assert md.query_start_loc.tolist() == [0, 37, 2085, 2086]
    assert manager.changed_slots(before) == set()


def test_a_captured_graph_reads_the_slots_and_the_reset_of_the_step_being_replayed():
    """`prepare` refills state_slots in place and resets fresh slots before every
    replay, so a graph captured on one batch reads each later batch's slots and
    a fresh row's zeroed state."""
    manager = _HybridManager({1: 0, 2: 5, 3: 7, 4: 8, 9: 6})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    md.prepare(_step(manager, [1, 2, 3, 4], [1] * 4, 0, [5, 5, 5, 5]))
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

    # Request 9 is a fresh one-token prompt promoted into the batch, on slot 6.
    md.prepare(_step(manager, [4, 9, _PAD, _PAD], [1] * 4, 0, [6, 0, 0, 0]))
    graph.replay()
    torch.cuda.synchronize()
    assert seen_slots.tolist() == [8, 6, PAD_SLOT_ID, PAD_SLOT_ID]
    assert seen_state.tolist() == [stale[8].item(), 0.0, -1.0, -1.0]


@pytest.mark.parametrize("cached_as_tensor", [False, True], ids=["list", "tensor"])
def test_cached_counts_as_a_tensor_or_a_list(cached_as_tensor):
    """The engine hands num_cached_tokens_per_seq over as a list or a host tensor."""
    manager = _HybridManager({7: 4, 9: 6})
    md = Nemotron4StateMetadata(_MAX_BATCH, 128, max_num_tokens=8192)
    step = _step(manager, [7, 9], [1, 1], 0, [3, 0])
    if cached_as_tensor:
        step.kv_cache_params.num_cached_tokens_per_seq = torch.tensor([3, 0], dtype=torch.int32)
    md.prepare(step)
    torch.cuda.synchronize()
    assert manager.is_zero(6) and not manager.is_zero(4)
