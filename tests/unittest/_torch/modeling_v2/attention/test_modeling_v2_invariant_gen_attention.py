# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the invariant_gen_attention entry of nemotron4-nano-preview's tp1 target.

Every call runs over a real paged pool laid out as the C++ KV cache manager
lays one out: page p of layer l holds K then V, each `[kv heads, 32, 256]`;
`kv_cache_block_offsets` hold `p * layers * 2 + {0, 1}` and the layer's base
adds `l * 2` pages through `host_kv_cache_pool_mapping`. The pool holds two
layers and the calls read the second, so the intra-pool offset is exercised.

Every byte of the pool a call must not read before writing it is NaN: the
poison page every unused page-table entry points at, the tail of each
request's last page, and the slots the call appends into. A read of any of
them puts a NaN into the output.
"""

import importlib
import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.invariant_gen_attention import (
    invariant_gen_attention as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)

__extra_import_path__ = [".."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "invariant_gen_attention requires a CUDA device"

#: The entry's module, whose `_VERIFY_MIN_SEQS` picks the kernel a verify call runs.
_entry = importlib.import_module(type(op).__module__)

#: Layers sharing the pool, and the one the calls read.
_LAYERS, _LAYER = 2, 1
#: The page every unused page-table entry points at: all NaN, never written.
_POISON_PAGE = 0


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Kernel and `reference` differ in how they round, not in what they compute.

    The kernel rounds P to bf16 for its second dot and merges an fp32 online
    softmax over 512-key splits; the reference is one fp32 softmax. Against an
    fp64 transcription the kernel sits within 1.31 ulp of the row scale (0.58
    of relative RMS) and the fp32 reference within 1.00 (0.44), so the two land
    at most one bf16 step apart at the top of a row's range. Measured over the
    27 cells x 3 seeds: 2.00 element-wise, 0.64 of relative RMS.

    So: 4 ulp element-wise (two such steps) and 1.0 of relative RMS. The
    mistakes this gate exists for land outside it in every cell -- the next
    token's key leaking into a verify row at >= 1.18 of relative RMS (92 to 196
    wherever a history is short), the row's own key dropped at >= 1.35, the
    scale of a 128-wide head at >= 3.4. A 0.2% scale error (1/sqrt(255)) is
    below what bf16 outputs resolve here and is not claimed.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=1.0, ulp=ULP[torch.bfloat16])


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16)


def _lengths(spec, real: int, gen: torch.Generator) -> list:
    """History lengths, this call's tokens included: one request at max_seq_len,
    one with no history beyond its own tokens, the rest uniform in between."""
    tokens, top = spec["tokens"], spec["max_seq_len"]
    if real == 1:
        return [top]
    rest = torch.randint(tokens, top + 1, (real - 2,), generator=gen).tolist()
    return [top, tokens] + rest


class _Call:
    """One generation call's whole state: pool, page table, lengths, qkv, output.

    Rows are the real requests, then `pad_rows` padding rows that share one
    dummy request's pages and length, as the CUDA-graph runner pads a batch.
    `num_contexts` context entries come first in `sequence_length` and the
    page table and point at nothing the call may read. Each request owns pages
    for `headroom` more tokens than its length, for calls that follow.
    """

    def __init__(self, spec, seed: int, lens=None, num_contexts: int = 0, headroom: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.dev = torch.Generator(device="cuda").manual_seed(seed)
        self.spec = spec
        tokens, dim, tpb = spec["tokens"], spec["head_size"], spec["tokens_per_block"]
        self.real = spec["batch"] - spec["pad_rows"]
        self.lens = list(lens) if lens is not None else _lengths(spec, self.real, gen)
        assert len(self.lens) == self.real
        seqs = self.lens + ([tokens] if spec["pad_rows"] else [])
        self.row_seq = list(range(self.real)) + [len(seqs) - 1] * spec["pad_rows"]
        self.num_contexts = num_contexts

        # Pages: every sequence's own, scattered through the pool by a permutation.
        counts = [math.ceil((n + headroom) / tpb) for n in seqs]
        perm = (1 + torch.randperm(sum(counts), generator=gen)).tolist()
        self.pages, at = [], 0
        for c in counts:
            self.pages.append(perm[at : at + c])
            at += c
        self.pool = torch.full(
            (1 + sum(counts), _LAYERS, 2, 1, tpb, dim),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        )
        width = max(max(counts), math.ceil(spec["max_seq_len"] / tpb)) + 1
        offsets = torch.empty((1, num_contexts + spec["batch"], 2, width), dtype=torch.int32)
        offsets[..., 0, :] = _POISON_PAGE * _LAYERS * 2
        offsets[..., 1, :] = _POISON_PAGE * _LAYERS * 2 + 1
        for row, s in enumerate(self.row_seq):
            pages = torch.tensor(self.pages[s], dtype=torch.int32)
            offsets[0, num_contexts + row, 0, : len(pages)] = pages * _LAYERS * 2
            offsets[0, num_contexts + row, 1, : len(pages)] = pages * _LAYERS * 2 + 1
        self.block_offsets = offsets.cuda()
        lengths = [9000 + i for i in range(num_contexts)] + [seqs[s] for s in self.row_seq]
        self.sequence_length = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        self.pool_pointers = torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64)
        self.pool_mapping = torch.tensor(
            [[0, layer] for layer in range(_LAYERS)], dtype=torch.int32
        )

        # The real requests' histories before this call, written into their pages.
        self.history = [
            torch.randn(n - tokens, 2, dim, generator=self.dev, device="cuda").to(torch.bfloat16)
            for n in self.lens
        ]
        for s, kv in enumerate(self.history):
            self._scatter(self.pool, s, 0, kv)
        rows = spec["batch"] * tokens
        width = (spec["num_heads"] + 2 * spec["num_kv_heads"]) * dim
        self.qkv = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        self.output = torch.empty(
            rows, spec["num_heads"] * dim, dtype=torch.bfloat16, device="cuda"
        )
        self.draw()

    def draw(self) -> None:
        """New qkv for every row, in place (a captured graph reads this buffer)."""
        self.qkv.copy_(torch.randn(self.qkv.shape, generator=self.dev, device="cuda"))
        self.output.fill_(float("nan"))

    def _scatter(self, pool: torch.Tensor, seq: int, start: int, kv: torch.Tensor) -> None:
        """K/V rows `[n, 2, D]` of one sequence into `pool` at positions start, start + 1, ..."""
        tpb = self.spec["tokens_per_block"]
        pos = start + torch.arange(kv.shape[0], device="cuda")
        table = torch.tensor(self.pages[seq], dtype=torch.long, device="cuda")
        page, slot = table[pos // tpb], pos % tpb
        pool[page, _LAYER, 0, 0, slot] = kv[:, 0]
        pool[page, _LAYER, 1, 0, slot] = kv[:, 1]

    def new_kv(self, row: int) -> torch.Tensor:
        """This call's K/V rows of one batch row, `[T, 2, D]`, as qkv carries them."""
        tokens, width = self.spec["tokens"], self.spec["num_heads"] * self.spec["head_size"]
        return self.qkv[row * tokens : (row + 1) * tokens, width:].view(tokens, 2, -1)

    def expected_pool(self, before: torch.Tensor) -> torch.Tensor:
        """`before` with every real row's new K/V appended, and the padding rows'
        dummy pages (they race on one slot) taken from the pool as it is."""
        expected = before.clone()
        tokens = self.spec["tokens"]
        for r in range(self.real):
            self._scatter(expected, r, self.lens[r] - tokens, self.new_kv(r))
        if self.spec["pad_rows"]:
            dummy = torch.tensor(self.pages[-1], device="cuda")
            expected[dummy] = self.pool[dummy]
        return expected

    def owned(self, pool: torch.Tensor) -> torch.Tensor:
        """`pool` without the padding rows' dummy pages, which they race on."""
        keep = torch.ones(pool.shape[0], dtype=torch.bool, device="cuda")
        if self.spec["pad_rows"]:
            keep[torch.tensor(self.pages[-1], device="cuda")] = False
        return pool[keep]

    def advance(self) -> None:
        """The next step: every real request's history gains this call's tokens,
        its length grows on the device, and new tokens arrive."""
        tokens = self.spec["tokens"]
        self.history = [torch.cat([h, self.new_kv(r)]) for r, h in enumerate(self.history)]
        self.lens = [n + tokens for n in self.lens]
        start = self.num_contexts
        self.sequence_length[start : start + self.real].add_(tokens)
        self.draw()

    def args(self, **over) -> dict:
        spec = self.spec
        args = dict(
            q=self.qkv,
            output=self.output,
            sequence_length=self.sequence_length,
            kv_cache_block_offsets=self.block_offsets,
            host_kv_cache_pool_pointers=self.pool_pointers,
            host_kv_cache_pool_mapping=self.pool_mapping,
            local_layer_idx=_LAYER,
            num_contexts=self.num_contexts,
            tokens_per_seq=spec["tokens"],
            num_heads=spec["num_heads"],
            num_kv_heads=spec["num_kv_heads"],
            head_size=spec["head_size"],
            tokens_per_block=spec["tokens_per_block"],
            max_seq_len=spec["max_seq_len"],
            q_scaling=spec["q_scaling"],
        )
        args.update(over)
        return args

    def reference(self) -> torch.Tensor:
        """`op.reference` over the real rows' histories plus this call's tokens."""
        spec = self.spec
        full = torch.cat([torch.cat([h, self.new_kv(r)]) for r, h in enumerate(self.history)])
        return op.reference(
            q=self.qkv[: self.real * spec["tokens"]],
            k=full[:, 0],
            v=full[:, 1],
            kv_lens=self.lens,
            tokens_per_seq=spec["tokens"],
            num_heads=spec["num_heads"],
            num_kv_heads=spec["num_kv_heads"],
            head_size=spec["head_size"],
            q_scaling=spec["q_scaling"],
        )

    def real_rows(self, out: torch.Tensor) -> torch.Tensor:
        return out[: self.real * self.spec["tokens"]]


def _check(call: _Call, out: torch.Tensor, before: torch.Tensor) -> None:
    """Real rows against `reference`; the pool bit-identical to `before` plus the
    real rows' appends."""
    _compare(call.real_rows(out), call.reference())
    assert torch.equal(_bits(call.pool), _bits(call.expected_pool(before))), (
        "the pool differs from its state before the call plus this call's appends"
    )


def _cell(tokens: int, batch: int):
    return next(c for c in op.CELLS if (c.spec["tokens"], c.spec["batch"]) == (tokens, batch))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    call = _Call(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = call.pool.clone()
    out = op(**call.args())
    assert out is call.output, "the op writes output in place and returns it"
    _check(call, out, before)


@pytest.mark.parametrize("tokens,batch", [(1, 1024), (2, 512), (3, 512)])
def test_a_request_does_not_depend_on_its_batch(tokens, batch) -> None:
    """Bit-identical rows and appends for requests run alone, in an eager batch
    of three and inside the padded graph batch: a row's arithmetic is fixed by
    the kernels' split and tile, never by the batch around it."""
    call = _Call(_cell(tokens, batch).spec, seed=batch + tokens)
    full = op(**call.args()).clone()
    pool_after = call.pool.clone()
    for rows in (slice(0, 1), slice(1, 4), slice(7, 8), slice(batch - 3, batch - 2)):
        part = slice(rows.start * tokens, rows.stop * tokens)
        got = op(
            **call.args(
                q=call.qkv[part],
                output=torch.full_like(call.output[part], float("nan")),
                sequence_length=call.sequence_length[rows],
                kv_cache_block_offsets=call.block_offsets[:, rows],
            )
        )
        assert torch.equal(_bits(got), _bits(full[part])), (
            f"T={tokens}, rows {rows}: the output depends on the batch"
        )
        assert torch.equal(_bits(call.pool), _bits(pool_after)), (
            f"T={tokens}, rows {rows}: the appended bytes depend on the batch"
        )


@pytest.mark.parametrize("tokens", [2, 3])
def test_verify_row_equals_decode_of_the_same_token(tokens) -> None:
    """Row j of a T-token group gives the bits the same token gives as a decode
    row (T = 1) over the same keys: a verified token's attention is the one
    MTP-off decode computes. The groups straddle page (32), tile (64) and split
    (512) boundaries."""
    spec = _cell(tokens, 64).spec
    gen = torch.Generator().manual_seed(tokens)
    ends = [31, 32, 33, 65, 66, 511, 512, 513, 577, 578, 1023, 1024, 1025, 4097, 8704, 9344, tokens]
    lens = ends + [min(n + 1000, 9344) for n in ends]
    lens += torch.randint(
        tokens, 9345, (spec["batch"] - spec["pad_rows"] - len(lens),), generator=gen
    ).tolist()
    call = _Call(spec, seed=31 + tokens, lens=lens)
    group = op(**call.args()).clone().view(spec["batch"], tokens, -1)
    pool_after = call.pool.clone()
    real = call.real
    for j in range(tokens):
        got = op(
            **call.args(
                q=call.qkv.view(spec["batch"], tokens, -1)[:real, j],
                output=torch.full_like(call.output[:real], float("nan")),
                sequence_length=call.sequence_length[:real] - (tokens - 1 - j),
                kv_cache_block_offsets=call.block_offsets[:, :real],
                tokens_per_seq=1,
            )
        )
        assert torch.equal(_bits(got), _bits(group[:real, j])), (
            f"T={tokens}: row {j} of the group differs from the same token decoded alone"
        )
        assert torch.equal(_bits(call.pool), _bits(pool_after)), (
            f"T={tokens}: decoding token {j} appended other bytes than the group"
        )


@pytest.mark.parametrize("tokens,batch", [(2, 64), (3, 64), (3, 512)])
def test_grouped_verify_kernel_equals_the_per_token_kernel(tokens, batch, monkeypatch) -> None:
    """A verify call of `_VERIFY_MIN_SEQS` or more requests runs one program per
    (split, request) holding the whole group; a smaller one, and decode, run one
    program per (token, split). The two compute every token's partial with the
    same arithmetic, so the same call through either kernel gives the same bits,
    outputs and appended bytes alike. Group ends sit where a token's keys end a
    page (32), tile (64) or split (512) before its group's last token's do."""
    spec = _cell(tokens, batch).spec
    gen = torch.Generator().manual_seed(100 + tokens)
    ends = [31, 32, 33, 64, 65, 66, 511, 512, 513, 514, 576, 577, 578, 1024, 1025, 1026, 4097, 9344]
    lens = ends + [tokens] + [min(n + 1000, 9344) for n in ends]
    real = spec["batch"] - spec["pad_rows"]
    lens += torch.randint(tokens, 9345, (real - len(lens),), generator=gen).tolist()
    call = _Call(spec, seed=batch + tokens, lens=lens)
    before = call.pool.clone()
    outputs, pools = [], []
    for min_seqs in (1, 2**31):
        monkeypatch.setattr(_entry, "_VERIFY_MIN_SEQS", min_seqs)
        call.pool.copy_(before)
        out = op(**call.args(output=torch.full_like(call.output, float("nan"))))
        outputs.append(call.real_rows(out).clone())
        pools.append(call.owned(call.pool).clone())
    assert torch.equal(_bits(outputs[0]), _bits(outputs[1])), (
        f"T={tokens} B={batch}: the grouped and per-token kernels give different bits"
    )
    assert torch.equal(_bits(pools[0]), _bits(pools[1])), (
        f"T={tokens} B={batch}: the two kernels appended different bytes"
    )


def test_cuda_graph_replays_on_new_lengths_and_inputs() -> None:
    """One capture at B = 64, T = 3; two replays after every length grew on the
    device (as the overlap scheduler corrects it) and new tokens arrived. Each
    replay is checked against `reference` and against the eager call on the
    same state, bit for bit."""
    spec = _cell(3, 64).spec
    gen = torch.Generator().manual_seed(64)
    lens = torch.randint(
        3, spec["max_seq_len"] - 6 + 1, (spec["batch"] - spec["pad_rows"],), generator=gen
    )
    call = _Call(spec, seed=64, lens=lens.tolist(), headroom=6)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**call.args())  # compiles outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**call.args())

    for step in range(2):
        call.advance()
        before = call.pool.clone()
        graph.replay()
        torch.cuda.synchronize()
        _check(call, out, before)
        replayed, pool_replayed = out.clone(), call.pool.clone()
        call.pool.copy_(before)
        eager = op(**call.args(output=torch.full_like(out, float("nan"))))
        assert torch.equal(_bits(call.real_rows(replayed)), _bits(call.real_rows(eager))), (
            f"replay {step}: the graph's output differs from the eager call's"
        )
        assert torch.equal(_bits(call.owned(pool_replayed)), _bits(call.owned(call.pool))), (
            f"replay {step}: the graph appended other bytes than the eager call"
        )


@pytest.mark.parametrize("tokens", [1, 3])
def test_appends_match_thop_attention(tokens) -> None:
    """The pool after this entry's call is, byte for byte, the pool after
    `thop_attention`'s generation call on the same state -- so the two share
    one pool -- and both outputs sit within the band of `reference`.

    The pool's NaN poison becomes zeros here: with NaN in the slots past a
    request's length in its last page, thop's generation kernels return whole
    requests as NaN (measured at T = 1 and T = 3, only requests whose last page
    is partly filled), so they read those slots."""
    spec = dict(_cell(tokens, 8).spec, pad_rows=0)
    call = _Call(spec, seed=80 + tokens, lens=[9344, 4097, 513, 512, 33, 32, tokens + 1, 2000])
    call.pool.nan_to_num_(0.0)
    thop_out, thop_pool = _thop_generation(call, call.pool.clone())
    before = call.pool.clone()
    out = op(**call.args())
    _check(call, out, before)
    assert torch.equal(_bits(call.pool), _bits(thop_pool)), (
        "the appends differ from thop_attention's"
    )
    _compare(thop_out, call.reference())


@pytest.mark.parametrize("min_seqs", [1, 2**31], ids=["grouped", "per_token"])
def test_generation_rows_after_context_rows(min_seqs, monkeypatch) -> None:
    """With context entries first in `sequence_length` and the page table, the
    call reads its rows from `num_contexts` on, as `thop_attention` does --
    through either split kernel."""
    monkeypatch.setattr(_entry, "_VERIFY_MIN_SEQS", min_seqs)
    call = _Call(_cell(3, 8).spec, seed=5, num_contexts=3)
    before = call.pool.clone()
    _check(call, op(**call.args()), before)


def _thop_generation(call: _Call, pool: torch.Tensor):
    """`thop_attention`'s generation-only call over `call`'s state, on `pool`."""
    args = _thop_args(call, pool)
    thop_attention(**args)
    torch.cuda.synchronize()
    return args["output"], pool


def _thop_args(call: _Call, pool: torch.Tensor) -> dict:
    """The arguments of `thop_attention`'s generation-only call over `call`'s
    state on `pool`, as the target passes them (`thop_attention`'s nemotron4 cells)."""
    spec = call.spec
    tokens = spec["tokens"]
    lens = call.sequence_length.tolist()
    prompts = [n - tokens for n in lens]
    return dict(
        q=call.qkv.clone(),
        k=None,
        v=None,
        output=torch.empty_like(call.output),
        output_sf=None,
        workspace_=torch.empty(0, dtype=torch.int8, device="cuda"),
        sequence_length=call.sequence_length,
        host_past_key_value_lengths=torch.tensor(lens, dtype=torch.int32),
        host_total_kv_lens=torch.tensor([0, sum(lens)], dtype=torch.int32),
        context_lengths=torch.tensor(prompts, dtype=torch.int32, device="cuda"),
        host_context_lengths=torch.tensor(prompts, dtype=torch.int32),
        host_request_types=torch.ones(len(lens), dtype=torch.int32),
        max_context_q_len_override=None,
        kv_cache_block_offsets=call.block_offsets,
        host_kv_cache_pool_pointers=torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64),
        host_kv_cache_pool_mapping=call.pool_mapping,
        cache_indirection=None,
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
        out_scale=None,
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        latent_cache=None,
        q_pe=None,
        block_ids_per_seq=None,
        attention_sinks=None,
        is_fused_qkv=True,
        update_kv_cache=True,
        predicted_tokens_per_seq=1,
        local_layer_idx=_LAYER,
        num_heads=spec["num_heads"],
        num_kv_heads=spec["num_kv_heads"],
        head_size=spec["head_size"],
        tokens_per_block=spec["tokens_per_block"],
        max_num_requests=spec["batch"],
        max_context_length=spec["max_seq_len"],
        max_seq_len=spec["max_seq_len"],
        attention_window_size=spec["max_seq_len"],
        beam_width=1,
        mask_type=1,
        quant_mode=0,
        q_scaling=spec["q_scaling"],
        position_embedding_type=0,
        rope_dim=0,
        rope_base=10000.0,
        rope_scale_type=0,
        rope_scale=1.0,
        rope_short_m_scale=1.0,
        rope_long_m_scale=1.0,
        rope_max_positions=spec["max_seq_len"],
        rope_original_max_positions=spec["max_seq_len"],
        use_paged_context_fmha=True,
        attention_input_type=0,
        is_mla_enable=False,
        chunked_prefill_buffer_batch_size=1,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_nope_head_dim=None,
        qk_rope_head_dim=None,
        v_head_dim=None,
        rope_append=None,
        mrope_rotary_cos_sin=None,
        mrope_position_deltas=None,
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        attention_chunk_size=None,
        softmax_stats_tensor=None,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets_for_cpp=None,
        spec_decoding_packed_mask=None,
        spec_decoding_bl_tree_mask_offset=None,
        spec_decoding_bl_tree_mask=None,
        spec_bl_tree_first_sparse_mask_offset_kv=None,
        sparse_kv_indices=None,
        sparse_kv_offsets=None,
        sparse_attn_indices=None,
        sparse_attn_offsets=None,
        sparse_attn_indices_block_size=0,
        num_contexts=0,
        num_ctx_tokens=0,
    )


def _ok(**over):
    """The T = 3, B = 8 cell's call and its arguments, some replaced."""
    call = _Call(_cell(3, 8).spec, seed=3)
    return call, call.args(**over)


def test_guard_refuses_the_q_slice_for_the_fused_qkv() -> None:
    """The q columns alone: the kernels read K and V past each row, into the
    next one, and return a plausible wrong answer."""
    call, args = _ok()
    ref = call.reference()
    # One spare row past the end, so the kernels' over-read stays in owned memory.
    spare = torch.zeros(1, 16 * 256, dtype=torch.bfloat16, device="cuda")
    q_only = torch.cat([call.qkv[:, : 16 * 256], spare])[:-1]
    unguarded = call.real_rows(op(**dict(args, q=q_only)))
    assert unguarded.isfinite().all() and not torch.allclose(unguarded, ref, atol=0.05), (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="fused qkv"):
        with validating(op):
            op(**dict(args, q=q_only))


def test_guard_refuses_a_length_past_max_seq_len() -> None:
    """The grid covers max_seq_len keys: a longer history silently loses its tail."""
    call, args = _ok()
    with pytest.raises(AssertionError, match="exceeds max_seq_len"):
        with validating(op):
            op(**dict(args, max_seq_len=4096))


def test_guard_refuses_two_kv_heads() -> None:
    call, args = _ok()
    with pytest.raises(AssertionError, match="num_kv_heads must be 1"):
        with validating(op):
            op(**dict(args, num_kv_heads=2))


def test_guard_refuses_a_length_shorter_than_its_tokens() -> None:
    call, args = _ok()
    lengths = call.sequence_length.clone()
    lengths[1] = 2
    with pytest.raises(AssertionError, match="must be >= tokens_per_seq"):
        with validating(op):
            op(**dict(args, sequence_length=lengths))


def test_guard_refuses_a_page_table_too_narrow() -> None:
    call, args = _ok()
    with pytest.raises(AssertionError, match="kv_cache_block_offsets rows hold"):
        with validating(op):
            op(**dict(args, kv_cache_block_offsets=call.block_offsets[..., :8]))


def test_guard_refuses_an_fp16_q() -> None:
    call, args = _ok()
    with pytest.raises(AssertionError, match="are bfloat16"):
        with validating(op):
            op(**dict(args, q=call.qkv.half(), output=call.output.half()))
