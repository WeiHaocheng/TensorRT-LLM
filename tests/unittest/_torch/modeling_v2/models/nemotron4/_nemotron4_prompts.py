# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fixed prompt set of the nemotron4-nano-preview targets.

Every identity, MTP and acceptance check of the nemotron4 targets runs these prompts: six short
literal ones (``SHORT_PROMPTS``) and four of 7.5K-8K tokens built deterministically from offline
texts under ``$LLM_MODELS_ROOT/datasets`` (no network). Half of each group is plain text, half is
rendered with the checkpoint's chat template (non-thinking mode: the ``enable_thinking=False``
TensorRT-LLM passes by default).

This checkpoint does not answer a single-turn chat in non-thinking mode: it opens a new
``system`` turn or repeats the question (the checkpoint's HF implementation does the same). With
two worked example turns before the question it answers, and what follows the answer is no
longer an answer: often token 138922 ``'ĄĄ'`` where ``<|im_end|>`` would be, then repetitive
text. None of the continuations of this set contains an end id (2 or 11). So the short chat
prompts carry two example turns, and a continuation past the answer is still the model's
(deterministic) output but not coherent text.

A prompt is its token ids for the checkpoint's tokenizer. The tokenizer's post-processor adds no
special tokens, so the ids are ``tokenizer.encode(text)`` with or without ``add_special_tokens``.
``PROMPT_SHA256`` pins every prompt's ids; ``prompt_set`` fails if a tokenizer or dataset change
would silently change what the checks compare.

The module imports only ``tokenizers`` and ``pyarrow``, never TensorRT-LLM or transformers.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import os
from collections.abc import Sequence

LONG_PROMPT_MAX_TOKENS = 8000
"""Upper bound on a long prompt: with 1024 generated tokens and a few draft tokens it fits a
``max_seq_len`` of 9.3K."""

_HAIKU_SHOTS = (
    (
        "Write a haiku about the ocean.",
        "Waves fold into foam\nsalt wind carries gull voices\nthe tide keeps its time",
    ),
    (
        "Write a haiku about a library.",
        "Rows of quiet spines\ndust drifts through a slant of light\npages wait for hands",
    ),
)
_GSM8K_SHOTS = (
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of"
        " babysitting. How much did she earn?",
        "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned"
        " 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does"
        " he write a year?",
        "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages"
        " every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624",
    ),
)
_SCIENCE_SHOTS = (
    (
        "Explain in two sentences why ice floats on water.",
        "When water freezes, its molecules lock into a hexagonal lattice that holds them farther"
        " apart than in the liquid. Ice is therefore less dense than liquid water, so it floats.",
    ),
    (
        "Explain in two sentences why we see lightning before we hear thunder.",
        "Light travels about a million times faster than sound in air. The flash reaches us almost"
        " instantly, while the sound of the same discharge arrives seconds later.",
    ),
)

SHORT_PROMPTS: tuple[tuple[str, tuple[tuple[str, str], ...] | None, str], ...] = (
    ("capital", None, "The capital of France is"),
    ("fibonacci", None, 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'),
    ("forest", None, "Once upon a time, in a small village at the edge of a great forest,"),
    ("haiku", _HAIKU_SHOTS, "Write a haiku about a GPU running out of memory."),
    (
        "clips",
        _GSM8K_SHOTS,
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips"
        " in May. How many clips did Natalia sell altogether in April and May?",
    ),
    ("sky", _SCIENCE_SHOTS, "Explain in two sentences why the sky is blue."),
)
"""(name, None for plain text or the (user, assistant) example turns of a chat, text)."""

PROMPT_SHA256: dict[str, tuple[int, str]] = {
    "capital": (5, "c004097f7960d2379d8362142619168335ab269fdb3ad97f6a1e5d6a5b83ebd5"),
    "fibonacci": (14, "47d58d6d4d7feb38efdf6e6ccf3e6bcc4e4f697de1d78bf8be21982bdaa712e0"),
    "forest": (17, "8c2c8ea004a2f8bfee113468b36d04e4a94cdc52f9d086571d8879737ee68743"),
    "haiku": (105, "89b8ed3b9b8dc1f0a098c211d97a6cd127aa79b446b5416229956e88bf6221ac"),
    "clips": (251, "7ab70bba44814935bd99629fb9fd9571a88a8d0cac7817f04314988275190484"),
    "sky": (138, "b8ff99d20a0a17a4c68665236717495019e86195db943f06d1a12ab01d517017"),
    "gsm8k_fewshot": (7818, "8fa6831500174cbb4ba546da1cb38a43cf24fec437a675e834e98d95335ff74b"),
    "gsm8k_multiturn": (7735, "f1f431ca58cb1a29c71026cd1e6663edaa7b6179ade1118353e283f7dc66a009"),
    "news_digest": (7984, "10a12ce0925a680f3fedc90aec9446897ddc054845476573ad199fd72a0912f6"),
    "gpqa_explain": (7992, "c676459a1d2106b064e40d158f5bdd02fc70abe0cd4d188a8d64dae6045affe5"),
}
"""name -> (number of tokens, sha256 of the token ids); see ``token_sha256``."""


@dataclasses.dataclass(frozen=True)
class Prompt:
    """One prompt of the set.

    Attributes:
        name: stable identifier.
        long: one of the ~8K-token prompts.
        chat: rendered with the chat template (a generation prompt ends it).
        text: the prompt text.
        token_ids: its token ids.
    """

    name: str
    long: bool
    chat: bool
    text: str
    token_ids: tuple[int, ...]


def datasets_root() -> str:
    root = os.environ.get("LLM_MODELS_ROOT")
    if not root:
        raise RuntimeError("set LLM_MODELS_ROOT (the long prompts read $LLM_MODELS_ROOT/datasets)")
    return os.path.join(root, "datasets")


def load_tokenizer(checkpoint: str):
    """The checkpoint's ``tokenizer.json`` as a ``tokenizers.Tokenizer``."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(checkpoint, "tokenizer.json"))


def encode(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False).ids)


def token_sha256(token_ids: Sequence[int]) -> str:
    """sha256 of the ids written as comma-separated decimal."""
    return hashlib.sha256(",".join(str(t) for t in token_ids).encode()).hexdigest()


def chat(user_turns: Sequence[str], assistant_turns: Sequence[str] = ()) -> str:
    """The checkpoint's ``chat_template.jinja`` for alternating user / assistant turns (no system
    message, no tools) ending with a user turn, ``add_generation_prompt=True`` and
    ``enable_thinking=False``.

    The template always opens an empty system block; earlier assistant turns get an empty
    ``<think></think>`` and are right-trimmed; the generation prompt opens an assistant turn with an
    empty think block.
    """
    if len(user_turns) != len(assistant_turns) + 1:
        raise ValueError("chat() takes one more user turn than assistant turns")
    parts = ["<|im_start|>system\n<|im_end|>\n"]
    for i, user in enumerate(user_turns):
        parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
        if i < len(assistant_turns):
            answer = assistant_turns[i].rstrip()
            parts.append(f"<|im_start|>assistant\n<think></think>{answer}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think></think>")
    return "".join(parts)


def short_prompts(tokenizer) -> list[Prompt]:
    prompts = []
    for name, shots, text in SHORT_PROMPTS:
        if shots is None:
            rendered = text
        else:
            rendered = chat([u for u, _ in shots] + [text], [a for _, a in shots])
        prompts.append(
            Prompt(name, False, shots is not None, rendered, encode(tokenizer, rendered))
        )
    return prompts


def _read_parquet(path: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _gsm8k(split: str) -> list[dict]:
    return _read_parquet(
        os.path.join(datasets_root(), "openai", "gsm8k", "main", f"{split}-00000-of-00001.parquet")
    )


def _news_articles() -> list[str]:
    """The CNN/DailyMail articles of the MLPerf GPT-J OpenOrca calibration set (its "Summarize the
    following news article" prompts), without the instruction around them."""
    path = os.path.join(
        datasets_root(), "common", "mlperf_gptj_openorca_calibration_1k", "data.parquet"
    )
    articles = []
    for row in _read_parquet(path):
        text = row["text"]
        if "### Input:\n" not in text:
            continue
        article = text.split("### Input:\n", 1)[1].split("### Response:", 1)[0].strip()
        if article:
            articles.append(article)
    return articles


def _gpqa() -> list[dict]:
    csv.field_size_limit(1 << 30)
    path = os.path.join(datasets_root(), "gpqa", "gpqa_main.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fill(tokenizer, render, pieces: Sequence, budget: int = LONG_PROMPT_MAX_TOKENS) -> str:
    """The longest prefix of ``pieces`` whose rendering ``render(pieces[:n])`` fits ``budget``
    tokens, rendered. Pieces are whole examples, so no prompt ends mid-example."""
    text = None
    for n in range(1, len(pieces) + 1):
        candidate = render(pieces[:n])
        if len(encode(tokenizer, candidate)) > budget:
            break
        text = candidate
    if text is None:
        raise ValueError("the first piece alone exceeds the budget")
    return text


def _gsm8k_fewshot(tokenizer) -> tuple[str, bool]:
    shots = [f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in _gsm8k("train")]
    final = _gsm8k("test")[0]["question"]
    return _fill(tokenizer, lambda p: "".join(p) + f"Question: {final}\nAnswer:", shots), False


def _gsm8k_multiturn(tokenizer) -> tuple[str, bool]:
    rows = _gsm8k("train")[2000:]
    final = _gsm8k("test")[1]["question"]

    def render(history):
        return chat([r["question"] for r in history] + [final], [r["answer"] for r in history])

    return _fill(tokenizer, render, rows), True


def _news_digest(tokenizer) -> tuple[str, bool]:
    def render(articles):
        body = "".join(f"Article {i + 1}:\n{text}\n\n" for i, text in enumerate(articles))
        return (
            f"Below are {len(articles)} news articles.\n\n{body}"
            "One-sentence summaries of the articles above, in order:\n\n1."
        )

    return _fill(tokenizer, render, _news_articles()), False


def _gpqa_explain(tokenizer) -> tuple[str, bool]:
    rows = _gpqa()
    final = rows[0]
    choices = [
        final["Incorrect Answer 1"],
        final["Incorrect Answer 2"],
        final["Incorrect Answer 3"],
    ]
    choices.insert(2, final["Correct Answer"])
    options = "".join(f"{'ABCD'[i]}) {c.strip()}\n" for i, c in enumerate(choices))
    ask = (
        f"Now answer a new question in the same way.\n\nQuestion: {final['Question'].strip()}\n"
        f"{options}Explain your reasoning step by step, then finish with 'Answer: <letter>'."
    )

    def render(items):
        body = "".join(
            f"Question: {r['Question'].strip()}\nCorrect answer: {r['Correct Answer'].strip()}\n"
            f"Explanation: {r['Explanation'].strip()}\n\n"
            for r in items
        )
        return chat([f"Here are solved science questions with explanations.\n\n{body}{ask}"])

    return _fill(tokenizer, render, rows[1:]), True


_LONG_BUILDERS = (
    ("gsm8k_fewshot", _gsm8k_fewshot),
    ("gsm8k_multiturn", _gsm8k_multiturn),
    ("news_digest", _news_digest),
    ("gpqa_explain", _gpqa_explain),
)


def long_prompts(tokenizer) -> list[Prompt]:
    prompts = []
    for name, build in _LONG_BUILDERS:
        text, is_chat = build(tokenizer)
        prompts.append(Prompt(name, True, is_chat, text, encode(tokenizer, text)))
    return prompts


def prompt_set(tokenizer, check: bool = True) -> list[Prompt]:
    """The short prompts, then the long ones.

    Args:
        check: fail unless every prompt's ids match ``PROMPT_SHA256``.
    """
    prompts = short_prompts(tokenizer) + long_prompts(tokenizer)
    if check:
        drift = {
            p.name: (len(p.token_ids), token_sha256(p.token_ids))
            for p in prompts
            if PROMPT_SHA256.get(p.name) != (len(p.token_ids), token_sha256(p.token_ids))
        }
        if drift:
            raise AssertionError(f"prompt token ids differ from PROMPT_SHA256: {drift}")
    return prompts
