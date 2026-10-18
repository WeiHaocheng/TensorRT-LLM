# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the plain-torch nemotron4 reference (``_nemotron4_reference.py``) and the prompt set.

The reference is the oracle every nemotron4 target is compared with, so each part a target check
leans on is held against an independent form of itself, on the checkpoint's real weights:

- the GDP recurrence (HF's per-token loop) against a chunked (WY) formulation, both against fp64;
- the GDP core and attention with their state carried across a split against one unsplit run;
- the MTP-Eagle draft depths (computed for every position at once) against the sequential
  KV-cache form TensorRT-LLM's one-model draft loop runs;
- the greedy acceptance walk on hand-built drafts;
- with the whole text model on one GPU, a greedy continuation, chunked prefill and the MTP
  acceptance of that continuation.

The comparison with the checkpoint's own HF classes needs transformers >= 5.17 and runs outside
this tree.
"""

import os

import pytest
import torch

__extra_import_path__ = ["."]
import _nemotron4_prompts as P  # noqa: E402 — needs the path declared above
import _nemotron4_reference as R  # noqa: E402

BF16_REL_SPACING = 2.0**-7
"""Distance between adjacent bf16 values relative to the larger one, at most."""


def _checkpoint_or_skip() -> str:
    root = os.environ.get("LLM_MODELS_ROOT")
    path = os.path.join(root, "nemotron4-nano-preview") if root else None
    if not path or not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
        pytest.skip("needs the checkpoint at $LLM_MODELS_ROOT/nemotron4-nano-preview")
    return path


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    """max |out - ref| / max |ref|, in fp64."""
    return float((out.double() - ref.double()).abs().max() / ref.double().abs().max())


@pytest.fixture(scope="module")
def checkpoint() -> R.Checkpoint:
    return R.Checkpoint(_checkpoint_or_skip())


@pytest.fixture(scope="module")
def tokenizer(checkpoint):
    return P.load_tokenizer(checkpoint.path)


@pytest.fixture(scope="module")
def text_ids(tokenizer) -> list[int]:
    """1024 token ids of real text: GSM8K train items, which no prompt of the set ends with."""
    if not os.environ.get("LLM_MODELS_ROOT"):
        pytest.skip("needs $LLM_MODELS_ROOT/datasets")
    rows = P._gsm8k("train")[:40]
    text = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in rows)
    ids = P.encode(tokenizer, text)
    assert len(ids) >= 1024
    return list(ids[:1024])


@pytest.fixture(scope="module")
def embed(checkpoint) -> torch.Tensor:
    return checkpoint.tensor(R.TEXT_PREFIX + "embed_tokens.weight")


@pytest.fixture(scope="module")
def gdp_layer0(checkpoint, embed, text_ids):
    """Layer 0 (``linear_attention``): its weights and its real input on ``text_ids``."""
    weights = checkpoint.layer(0)
    config = checkpoint.config
    ids = torch.tensor(text_ids, device="cuda")
    streams = embed[ids].repeat(1, config.wide_residual_num_streams)
    x = R._connection_read(
        streams, weights, "residual_connection", weights["input_layernorm.weight"], config
    )
    return R.sub(weights, "mixer."), x


# ----------------------------------------------------------------------------------------------
# CPU
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hit, expected",
    [
        # Both drafts right: two accepted per step, the walk advances by three.
        (lambda d, q: True, [2, 2, 2, 2, 2]),
        # Both wrong: nothing accepted, the walk advances by one.
        (lambda d, q: False, [0] * 13),
        # Depth 1 right, depth 2 wrong.
        (lambda d, q: d == 0, [1] * 7),
        # Depth 2 right but depth 1 wrong: acceptance is a prefix, so nothing.
        (lambda d, q: d == 1, [0] * 13),
        # Depth 1 wrong only at q = 6: steps at q = 4, 7 (accept 2), 10, 13, 16.
        (lambda d, q: not (d == 0 and q == 6), [2, 2, 2, 2, 2]),
        # Depth 1 wrong only at q = 7: q = 4 (2), 7 (0), 8 (2), 11 (2), 14 (2), 17 stops.
        (lambda d, q: not (d == 0 and q == 7), [2, 0, 2, 2, 2]),
    ],
)
def test_greedy_acceptance_walk(hit, expected):
    """``greedy_acceptance`` on a 20-token sequence with a 5-token prompt and two draft depths:
    drafting starts at position 4 and stops once a verify window would pass the end."""
    tokens = list(range(100, 120))
    drafts = [
        [tokens[q + 2 + d] if q + 2 + d < len(tokens) and hit(d, q) else -1 for q in range(20)]
        for d in range(2)
    ]
    assert R.greedy_acceptance(tokens, drafts, first_draft_position=4) == expected


def test_prompt_set_is_pinned(tokenizer):
    """The prompt set builds offline and matches ``PROMPT_SHA256``: >= 4 short prompts and
    >= 4 long ones of 7K-8K tokens."""
    prompts = P.prompt_set(tokenizer, check=True)
    short = [p for p in prompts if not p.long]
    long_ = [p for p in prompts if p.long]
    assert len(short) >= 4 and len(long_) >= 4
    assert len({p.name for p in prompts}) == len(prompts)
    for p in long_:
        assert 7000 <= len(p.token_ids) <= P.LONG_PROMPT_MAX_TOKENS, (p.name, len(p.token_ids))


# ----------------------------------------------------------------------------------------------
# GPU, one layer's weights
# ----------------------------------------------------------------------------------------------


@torch.inference_mode()
def test_gdp_recurrence_chunked_matches_loop(checkpoint, gdp_layer0):
    """The per-token loop (HF's ``torch_recurrent_gated_delta_rule``) and the chunked WY form
    agree, and each agrees with fp64, on layer 0's real q/k/v/g/beta over 1024 tokens (3072
    sub-steps).

    Bound: the fp32 forms within 1e-4 of fp64 relative to the largest value. The recurrence
    output is rounded to bf16 next (spacing up to 2^-7 relative), so an fp32 error 1e-4 is
    far below what that rounding can resolve. In fp64 the two forms agree to 1e-9.
    """
    weights, x = gdp_layer0
    captured = {}

    def capture(q, k, v, g, beta, initial_state=None, use_qk_l2norm=True):
        captured.update(q=q, k=k, v=v, g=g, beta=beta)
        return R.gated_delta_rule_recurrent(q, k, v, g, beta, initial_state, use_qk_l2norm)

    R.gdp_mixer(x, weights, checkpoint.config, recurrence=capture)
    args = [captured[n] for n in ("q", "k", "v", "g", "beta")]
    assert args[0].shape[0] == 3 * x.shape[0]
    out64, state64 = R.gated_delta_rule_recurrent(*args, dtype=torch.float64)
    out64c, state64c = R.gated_delta_rule_chunked(*args, dtype=torch.float64)
    measured = {"fp64 chunked vs fp64 loop": (_rel(out64c, out64), _rel(state64c, state64))}
    outs = {}
    for form in (R.gated_delta_rule_recurrent, R.gated_delta_rule_chunked):
        out, state = form(*args)
        assert out.dtype == torch.float32 and state.dtype == torch.float32
        outs[form.__name__] = (out, state)
        measured[f"fp32 {form.__name__} vs fp64"] = (_rel(out, out64), _rel(state, state64))
    (loop_out, loop_state), (chunk_out, chunk_state) = outs.values()
    measured["fp32 loop vs fp32 chunked"] = (
        _rel(loop_out, chunk_out),
        _rel(loop_state, chunk_state),
    )
    print(f"(output, final state) max error / max value: {measured}")
    assert max(measured["fp64 chunked vs fp64 loop"]) < 1e-9, measured
    for form in (R.gated_delta_rule_recurrent, R.gated_delta_rule_chunked):
        assert max(measured[f"fp32 {form.__name__} vs fp64"]) < 1e-4, measured


@torch.inference_mode()
@pytest.mark.parametrize("split", [1, 511, 1021])
def test_gdp_core_state_carry(checkpoint, gdp_layer0, split):
    """``gdp_core`` over tokens [0, split) and then the rest one token at a time, carrying the conv
    and recurrent state, reproduces one run over all 1024 tokens.

    The inputs of both are the same ``in_proj`` rows, so the conv state is the same bytes and the
    recurrent state can differ only by fp32 rounding (1e-5 relative); a carried-state bug
    (taps in the wrong order, a lost or stale state) moves the outputs by O(1).
    """
    weights, x = gdp_layer0
    config = checkpoint.config
    projected = R.linear(x, weights["in_proj.weight"])
    tail = list(range(split, min(split + 3, projected.shape[0])))
    whole, whole_state = R.gdp_core(projected[: tail[-1] + 1], weights, config)
    out, state = R.gdp_core(projected[:split], weights, config)
    parts = [out]
    for t in tail:
        out, state = R.gdp_core(projected[t : t + 1], weights, config, state)
        parts.append(out)
    carried = torch.cat(parts)
    diff = (carried.float() - whole.float()).abs()
    scale = whole.float().abs().amax(dim=-1, keepdim=True)
    measured = {
        "conv state equal": torch.equal(state.conv, whole_state.conv),
        "recurrent state rel": _rel(state.recurrent, whole_state.recurrent),
        "output worst row rel": float((diff / scale).max()),
        "output identical": float((carried == whole).float().mean()),
    }
    print(f"split {split}: {measured}")
    assert measured["conv state equal"], measured
    assert measured["recurrent state rel"] < 1e-5, measured
    assert measured["output worst row rel"] <= BF16_REL_SPACING, measured
    assert measured["output identical"] > 0.99, measured


@torch.inference_mode()
def test_attention_kv_offsets(checkpoint, embed, text_ids):
    """``causal_attention`` for the last queries of a sequence over the cached keys equals the
    matching rows of one causal run over the whole sequence (fp32 in, fp32 out).

    Layer 5's real projections of 257 tokens; the last 3 queries each see keys 0..own position.
    Only the reduction order differs between the two runs, so they agree to fp32 rounding; an
    off-by-one in what a query sees moves its row by O(1).
    """
    config = checkpoint.config
    attn = checkpoint.tensors(R.TEXT_PREFIX + "layers.5.self_attn.")
    num = 257
    x = R.rms_norm(
        embed[torch.tensor(text_ids[:num], device="cuda")],
        torch.ones(config.hidden_size, dtype=torch.bfloat16, device="cuda"),
        config.rms_norm_eps,
    )
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q = R.linear(x, attn["q_proj.weight"]).view(num, heads, dim).float()
    k = R.linear(x, attn["k_proj.weight"]).view(num, kv_heads, dim).float()
    v = R.linear(x, attn["v_proj.weight"]).view(num, kv_heads, dim).float()
    scale = dim**-0.5
    whole = R.causal_attention(q, k, v, scale)
    measured = {}
    for t in range(num - 3, num):
        row = R.causal_attention(q[t : t + 1], k[: t + 1], v[: t + 1], scale)
        measured[f"query {t} over keys 0..{t}"] = _rel(row, whole[t : t + 1])
    # The same rows with the query chunking active (64 queries per chunk) and with 2 queries
    # over their keys.
    chunked = R.causal_attention(q, k, v, scale, query_chunk=64)
    measured["64-query chunks"] = _rel(chunked, whole)
    measured["last 2 queries"] = _rel(R.causal_attention(q[-2:], k, v, scale), whole[-2:])
    print(f"max error / max value: {measured}")
    assert max(measured.values()) < 1e-5, measured


def _linear_fp64(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``R.linear`` with an fp64 GEMM: the bf16 result no longer depends on the row count."""
    return torch.matmul(x.double(), weight.double().t()).to(x.dtype)


def _linear_fp32_via_fp64(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.double(), weight.double().t()).float()


@torch.inference_mode()
def test_mtp_eagle_depths_match_the_sequential_cache(checkpoint, embed, text_ids, monkeypatch):
    """The MTP-Eagle depths (``mtp_draft_hidden(mode="eagle")``, every position at once) equal the
    sequential form of TensorRT-LLM's one-model draft loop, where one KV cache serves every depth:
    depth k >= 2 at position i appends its key to the depth-1 keys of positions 0..i and the keys
    depths 2..k-1 wrote at i, and attends over them.

    The sequential form runs ``mtp_layer`` on one position at a time over that cache, each depth
    starting from the vectorized form's previous depth, so both forms see the same inputs and
    differ only in how the attention is arranged. The GEMMs run in fp64 for this test, so one row
    and 48 rows round to the same bf16 values. What is left is fp32 association: with two
    private keys (depth 3) the vectorized form sums their two value products before adding the
    base keys' sum, the cache adds them one after the other, which moves some outputs by a bf16
    spacing (on GB300: depth 2 bit-identical, depth 3 within 1.3e-3 of the row's largest value
    with 90% of the values identical). The bound is 2 bf16 spacings. The HF semantics (each depth
    over its own sequence) is the alternative the test must tell apart; it differs by 0.19 and
    0.28 of the row's largest value.
    """
    monkeypatch.setattr(R, "linear", _linear_fp64)
    monkeypatch.setattr(R, "linear_fp32", _linear_fp32_via_fp64)
    config = checkpoint.config
    mtp = checkpoint.mtp_layer()
    num = 48
    ids = torch.tensor(text_ids[:num], device="cuda")
    # Any bf16 hidden exercises the wiring; hnorm normalizes it first.
    hidden = embed[torch.tensor(text_ids[1 : num + 1], device="cuda")]
    eagle = R.mtp_draft_hidden(ids, hidden, embed, mtp, config, num_depths=3, mode="eagle")
    hf = R.mtp_draft_hidden(ids, hidden, embed, mtp, config, num_depths=3, mode="hf")
    assert torch.equal(eagle[0], hf[0]), "depth 1 is the same computation in both modes"

    # Sequential: depth 1 over the whole sequence gives the cache; then per position i, depth 2
    # and depth 3 each append one entry at i.
    shifted = [torch.nn.functional.pad(ids[d:], (0, d)) for d in range(1, 4)]
    depth1, (k1, v1) = R.mtp_layer(hidden, embed[shifted[0]], mtp, config)
    assert torch.equal(depth1, eagle[0])
    worst = {2: 0.0, 3: 0.0}
    gap = {2: 0.0, 3: 0.0}
    identical = {2: [], 3: []}
    for i in range(num):
        cache = (k1[: i + 1], v1[: i + 1])
        for depth in (2, 3):
            emb = embed[shifted[depth - 1][i : i + 1]]
            h, cache = R.mtp_layer(eagle[depth - 2][i : i + 1], emb, mtp, config, kv=cache)
            ref = eagle[depth - 1][i : i + 1].float()
            scale = float(ref.abs().max())
            worst[depth] = max(worst[depth], float((h.float() - ref).abs().max()) / scale)
            other = hf[depth - 1][i : i + 1].float()
            gap[depth] = max(gap[depth], float((other - ref).abs().max()) / scale)
            identical[depth].append(float((h.float() == ref).float().mean()))
    measured = {d: (worst[d], gap[d], sum(identical[d]) / num) for d in (2, 3)}
    print(f"depth: (worst rel diff, rel gap to the hf semantics, fraction identical) {measured}")
    for depth, (worst_rel, gap_rel, _) in measured.items():
        assert worst_rel <= 2 * BF16_REL_SPACING, (depth, measured)
        assert gap_rel > 10 * BF16_REL_SPACING, (depth, measured)


# ----------------------------------------------------------------------------------------------
# GPU, the whole text model (~186 GiB with the fp32 lm_head)
# ----------------------------------------------------------------------------------------------

CAPITAL_GREEDY_24 = [
    13125, 1046, 3339, 1460, 1327, 24996, 11205, 1320, 10188, 1413, 1327, 4633,
    10415, 3409, 11205, 1320, 1327, 17448, 21681, 1046, 13125, 1460, 10540, 1476,
]  # fmt: skip
"""The reference's 24-token greedy continuation of "The capital of France is" on GB300
(" Paris. It is the largest city in France and the most populous city in the European Union.
Paris is known"); its smallest top-2 margin is 0.077, at the 11th token."""

DECODE_PREFILL_LOGIT_NOISE = 1.0
"""Bound on the logit differences between decoding with the states carried and one forward over
the same tokens, for the 24 tokens of "capital" checked below (measured 0.26 on GB300). The two
differ only in the GEMMs' row counts, which move bf16 roundings, and 44 layers amplify them:
over 128-256 tokens of each prompt of the set the worst position reaches 0.20-1.15 (mean
0.05-0.08), so a longer or different check needs a wider bound."""


@pytest.fixture(scope="module")
def model():
    path = _checkpoint_or_skip()
    free, _ = torch.cuda.mem_get_info()
    if free < 200 * 2**30:
        pytest.skip(f"the whole text model needs ~190 GiB free, {free / 2**30:.0f} GiB is")
    return R.Nemotron4Reference(path)


@pytest.mark.high_cuda_memory
@torch.inference_mode()
def test_whole_model_greedy_prefill_and_mtp(model, tokenizer):
    """The whole text model: greedy continuation, decode against teacher forcing, chunked prefill,
    and the MTP acceptance of the continuation.

    - "The capital of France is" continues " Paris." and, token for token, as pinned.
    - Decoding one token at a time with the states carried picks the same token as one forward
      over prompt + continuation wherever that forward's top-2 margin exceeds
      ``DECODE_PREFILL_LOGIT_NOISE``, and the chosen tokens' logits agree within it.
    - A 2-chunk prefill with the states carried ends with logits within that noise of one
      prefill. A state carried wrongly moves them by more.
    - The teacher-forced MTP acceptance length of the continuation, both modes, lies in [1, 3]
      and the depth-1 drafts match the next token at least half the time.
    """
    ids = P.encode(tokenizer, "The capital of France is")
    generated, step_logits = model.greedy(ids, 24)
    assert tokenizer.decode(generated).startswith(" Paris."), tokenizer.decode(generated)
    assert generated == CAPITAL_GREEDY_24, generated

    full = torch.tensor(list(ids) + generated, device="cuda")
    tf = model.forward(full, logits="all")
    tf_top = tf.logits[len(ids) - 1 : -1].topk(2, dim=-1)
    margins = tf_top.values[:, 0] - tf_top.values[:, 1]
    agree = tf_top.indices[:, 0] == torch.tensor(generated, device="cuda")
    assert bool((agree | (margins < DECODE_PREFILL_LOGIT_NOISE)).all()), (agree, margins)
    step_top1 = step_logits.max(dim=-1).values
    tf_same_token = tf.logits[len(ids) - 1 : -1].gather(1, full[len(ids) :, None])[:, 0]
    assert float((step_top1 - tf_same_token).abs().max()) < DECODE_PREFILL_LOGIT_NOISE

    whole = model.forward(full)
    first = model.forward(full[:13])
    chunked = model.forward(full[13:], first.states)
    chunk_diff = float((chunked.logits[-1] - whole.logits[-1]).abs().max())
    assert chunk_diff < DECODE_PREFILL_LOGIT_NOISE

    acceptance = {}
    for mode in R.MTP_MODES:
        acc = model.teacher_forced_mtp_acceptance(
            full.tolist(), len(ids), hidden=tf.hidden, mode=mode
        )
        acceptance[mode] = (acc["acceptance_length"], acc["verify_steps"], acc["depth_match"])
        assert 1.0 <= acc["acceptance_length"] <= 3.0, acc["acceptance_length"]
        assert acc["depth_match"][0] >= 0.5, acc["depth_match"]
    print(
        f"greedy {tokenizer.decode(generated)!r}; decode vs prefill top-1 logit max diff "
        f"{float((step_top1 - tf_same_token).abs().max()):.4f}; 2-chunk vs 1-chunk prefill last "
        f"logits max diff {chunk_diff:.4f}; MTP (AL, verify steps, depth match) {acceptance}"
    )
