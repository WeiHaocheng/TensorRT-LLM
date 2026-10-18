# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nemotron4's dep4 weight table declares, and maps, exactly what the checkpoint stores, by rank.

dep4 is attention DP over 4 ranks with the routed experts split four ways, so
every rank's table is the whole text model except each MoE block's two fused
expert stacks, of which a rank reads its own window of 128 experts. Two
transcriptions of the HF modeling code (`modeling_nemotron4.py`) hold the table
to account, each written from the module tree rather than imported from the
table, so the table cannot agree with itself:

* `_reference_declaration` -- every parameter a rank allocates, with the shape
  and dtype the HF module that owns it stores, or the layout the target
  re-lays it into at load (fused qkv, bf16 residual coefficients, conv taps,
  `-exp(A_log)`), the expert stacks cut to the rank's window;
* `_reference_text_keys` -- every text-model key of the checkpoint's state
  dict, the MTP layer's included.

Against those, for every rank with MTP off and on: every declared parameter has
a source, every text key is consumed, and the only keys left are the predicted
non-load; the four ranks' windows tile each stack's 512 experts exactly once and
nothing else differs between ranks. When the checkpoint itself is on disk, its
index and safetensors headers are read too (headers only, no tensor data): every
manifest row's stored shape and dtype, cut to the row's window and run through
the row's transform on the meta device, must land exactly on the declared
slice, and a rank's weight bytes are held to the stored ones.

No GPU: parameters are declared on the meta device, and `dims` reads
`model_config` alone.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__dep4 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_BLOCKS = dict(L="linear_attention", M="mlp", A="shortcut_moe_attn", G="shortcut_moe_gdp")

# nemotron4-nano-preview's text_config, as its config.json declares it.
_TEXT_CONFIG = dict(
    layer_types=[_BLOCKS[b] for b in "LMLMLAGGLAGGGLAGGGLAGGGLAGGGLAGGGLAGGLAGLMLM"],
    hidden_size=3072,
    vocab_size=262272,
    num_attention_heads=16,
    num_key_value_heads=1,
    head_dim=256,
    intermediate_size=12288,
    gdp_num_heads=24,
    gdp_num_groups=24,
    gdp_key_head_dim=128,
    gdp_head_dim=64,
    gdp_num_householder=3,
    gdp_conv_kernel=4,
    n_routed_experts=512,
    moe_latent_size=768,
    moe_intermediate_size=4096,
    moe_shared_expert_intermediate_size=4096,
    wide_residual_num_streams=3,
    wide_residual_retention_max_forget=0.1,
    num_nextn_predict_layers=2,
    mtp_use_repeated_layer=True,
)
_NUM_LAYERS = len(_TEXT_CONFIG["layer_types"])
_EP = 4
_LOCAL_EXPERTS = _TEXT_CONFIG["n_routed_experts"] // _EP
_RANKS = range(_EP)
_EXPERT_STACKS = ("experts_up", "experts_down")

_CHECKPOINT = Path(os.environ.get("LLM_MODELS_ROOT", "/nonexistent")) / "nemotron4-nano-preview"

# Keys of the vision tower and projector: what the checkpoint stores beside the
# text model, and what the target never loads. A handful stands for the family.
_VISION_KEYS = {
    "model.vision_tower.ln_pre.weight",
    "model.vision_tower.patch_conv.weight",
    "model.vision_tower.transformer.layers.0.attention.q_proj.weight",
    "model.multi_modal_projector.linear_1.weight",
    "model.multi_modal_projector.patch_merger.merging_layer.weight",
}


def _core(mtp: bool, rank: int):
    """The configuration surface `dims` reads: a composite config whose text_config
    carries this checkpoint's numbers, rank `rank`'s dep4 mapping, and a speculative
    config when MTP is live."""
    pretrained = SimpleNamespace(text_config=SimpleNamespace(**_TEXT_CONFIG))
    mapping = Mapping(
        world_size=_EP,
        rank=rank,
        tp_size=_EP,
        moe_ep_size=_EP,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=pretrained,
            mapping=mapping,
            torch_dtype=torch.bfloat16,
            spec_config=object() if mtp else None,
        )
    )


def _window(rank: int) -> slice:
    return slice(rank * _LOCAL_EXPERTS, (rank + 1) * _LOCAL_EXPERTS)


def _reference_declaration(mtp: bool) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Every parameter a rank allocates, transcribed from the HF module tree.

    The fp32 entries are the router's correction bias, which HF keeps fp32
    through load (`_keep_in_fp32_modules_strict`), and `-exp(A_log)`, which HF
    forms in fp32 from its fp32 `A_log`. The wide-residual logits HF also keeps
    fp32 become the bf16 coefficients HF rounds them into on every forward
    (`.to(streams.dtype)`), and the attention's q, k and v one fused operand.
    The two expert stacks hold the rank's window of `Nemotron4Experts`'
    `[num_experts, out, in]` parameters.
    """
    c = SimpleNamespace(**_TEXT_CONFIG)
    bf16, f32 = torch.bfloat16, torch.float32
    hidden, streams = c.hidden_size, c.wide_residual_num_streams
    # Nemotron4GDPMixer: value_dim, key_dim, conv_dim and the in_proj width.
    value_dim = c.gdp_num_heads * c.gdp_head_dim
    key_dim = c.gdp_num_groups * c.gdp_key_head_dim
    m = c.gdp_num_householder
    conv_dim = m * value_dim + (m + 1) * key_dim
    projection = value_dim + conv_dim + (m + 1) * c.gdp_num_heads
    w = {}

    def connection(i, name):  # Nemotron4WideResidualConnection: f, w and r
        for coefficient in ("read", "write", "retain"):
            w[f"l{i}_{name}_{coefficient}"] = ((streams,), bf16)

    def gdp(i):  # Nemotron4GDPMixer
        w[f"l{i}_gdp_in"] = ((projection, hidden), bf16)
        w[f"l{i}_gdp_conv"] = ((conv_dim, c.gdp_conv_kernel), bf16)
        w[f"l{i}_gdp_neg_a"] = ((c.gdp_num_heads,), f32)
        w[f"l{i}_gdp_dt_bias"] = ((c.gdp_num_heads,), bf16)
        w[f"l{i}_gdp_norm"] = ((value_dim,), bf16)
        w[f"l{i}_gdp_out"] = ((hidden, value_dim), bf16)

    def attention(i):  # Nemotron4Attention, q, k and v fused
        q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        w[f"l{i}_qkv"] = ((q + 2 * kv, hidden), bf16)
        w[f"l{i}_o"] = ((hidden, q), bf16)

    def moe(i):  # the shortcut block's norms and Nemotron4MoE
        e, latent, inter = c.n_routed_experts, c.moe_latent_size, c.moe_intermediate_size
        shared = c.moe_shared_expert_intermediate_size
        w[f"l{i}_shortcut_norm"] = ((hidden,), bf16)
        w[f"l{i}_pre_moe_norm"] = ((hidden,), bf16)
        w[f"l{i}_router"] = ((e, hidden), bf16)
        w[f"l{i}_router_bias"] = ((e,), f32)
        w[f"l{i}_fc1_latent"] = ((latent, hidden), bf16)
        w[f"l{i}_experts_up"] = ((e // _EP, inter, latent), bf16)
        w[f"l{i}_experts_down"] = ((e // _EP, latent, inter), bf16)
        w[f"l{i}_fc2_norm"] = ((latent,), bf16)
        w[f"l{i}_fc2_latent"] = ((hidden, latent), bf16)
        w[f"l{i}_shared_up"] = ((shared, hidden), bf16)
        w[f"l{i}_shared_down"] = ((hidden, shared), bf16)

    for i, layer_type in enumerate(c.layer_types):
        w[f"l{i}_input_norm"] = ((hidden,), bf16)
        if layer_type == "linear_attention":
            gdp(i)
            connection(i, "res")
        elif layer_type == "mlp":
            w[f"l{i}_mlp_up"] = ((c.intermediate_size, hidden), bf16)
            w[f"l{i}_mlp_down"] = ((hidden, c.intermediate_size), bf16)
            connection(i, "res")
        else:
            w[f"l{i}_shortcut_read"] = ((streams,), bf16)
            connection(i, "mix")
            connection(i, "moe")
            moe(i)
            if layer_type == "shortcut_moe_attn":
                attention(i)
            else:
                gdp(i)
    w["embed"] = ((c.vocab_size, hidden), bf16)
    w["final_norm"] = ((hidden,), bf16)
    w["readout"] = ((streams,), bf16)
    if mtp:
        # One repeated MTP layer, at the index HF gives its attention block. Its
        # block is a shortcut attention + MoE block on a plain residual.
        j = _NUM_LAYERS
        w[f"l{j}_input_norm"] = ((hidden,), bf16)
        moe(j)
        attention(j)
        w[f"l{j}_enorm"] = ((hidden,), bf16)
        w[f"l{j}_hnorm"] = ((hidden,), bf16)
        w[f"l{j}_eh"] = ((hidden, 2 * hidden), bf16)
        w[f"l{j}_mtp_norm"] = ((hidden,), bf16)
    return w


def _reference_text_keys(mtp: bool) -> set[str]:
    """Every text-model key of the HF state dict, transcribed from the module tree."""
    c = SimpleNamespace(**_TEXT_CONFIG)
    keys = {"lm_head.weight"}
    text = "model.language_model"

    def connection(p, name):
        keys.update(f"{p}.{name}.{logit}_logit" for logit in ("read", "write", "retention"))

    def gdp(p):
        keys.update(
            f"{p}.mixer.{n}"
            for n in (
                "in_proj.weight",
                "conv1d.weight",
                "A_log",
                "dt_bias",
                "norm.weight",
                "out_proj.weight",
            )
        )

    def attention(p):
        keys.update(f"{p}.self_attn.{n}_proj.weight" for n in ("q", "k", "v", "o"))

    def moe(p):
        keys.update(
            f"{p}.{n}"
            for n in (
                "shortcut_input_layernorm.weight",
                "pre_moe_layernorm.weight",
                "moe.gate.weight",
                "moe.gate.e_score_correction_bias",
                "moe.fc1_latent_proj.weight",
                "moe.experts.up_proj",
                "moe.experts.down_proj",
                "moe.fc2_norm.weight",
                "moe.fc2_latent_proj.weight",
                "moe.shared_experts.up_proj.weight",
                "moe.shared_experts.down_proj.weight",
            )
        )

    for i, layer_type in enumerate(c.layer_types):
        p = f"{text}.layers.{i}"
        keys.add(f"{p}.input_layernorm.weight")
        if layer_type == "linear_attention":
            gdp(p)
            connection(p, "residual_connection")
        elif layer_type == "mlp":
            keys.update({f"{p}.mlp.up_proj.weight", f"{p}.mlp.down_proj.weight"})
            connection(p, "residual_connection")
        else:
            keys.add(f"{p}.shortcut_read.logit")
            connection(p, "mixer_residual_connection")
            connection(p, "moe_residual_connection")
            moe(p)
            if layer_type == "shortcut_moe_attn":
                attention(p)
            else:
                gdp(p)
    keys.update(
        {
            f"{text}.embed_tokens.weight",
            f"{text}.norm.weight",
            f"{text}.residual_readout.logit",
        }
    )
    if mtp:
        p = f"{text}.mtp.layers.0"
        keys.update(f"{p}.{n}.weight" for n in ("enorm", "hnorm", "eh_proj", "final_layernorm"))
        keys.add(f"{p}.block.input_layernorm.weight")
        moe(f"{p}.block")
        attention(f"{p}.block")
    return keys


def _declared(mtp: bool, rank: int) -> dict:
    core = _core(mtp, rank)
    with torch.device("meta"):
        return W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))


def _consumed(manifest: dict) -> set[str]:
    return {ckpt for rows in manifest.values() for ckpt, _, _, _ in rows}


_MTP = pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
_RANK = pytest.mark.parametrize("rank", _RANKS, ids=[f"rank{r}" for r in _RANKS])


@_MTP
@_RANK
def test_the_table_declares_what_the_hf_modules_store(mtp, rank):
    """Shape and dtype per key, so a failure names the weight that drifted."""
    got, want = _declared(mtp, rank), _reference_declaration(mtp)
    assert set(got.keys()) == set(want.keys()), set(got.keys()) ^ set(want.keys())
    for key, (shape, dtype) in sorted(want.items()):
        assert tuple(got[key].shape) == shape, key
        assert got[key].dtype == dtype, key


@_MTP
@_RANK
def test_every_declared_weight_has_exactly_its_sources(mtp, rank):
    """A declared weight with no source allocates memory nothing ever fills; a key
    mapped twice is filled twice. Neither may happen. Only the expert stacks are
    read in part, at the rank's window; everything else is read whole, and only
    the fused qkv assembles one parameter from several sources."""
    core = _core(mtp, rank)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    assert set(manifest) == set(_declared(mtp, rank).keys())
    assert all(manifest.values())
    sources = [ckpt for rows in manifest.values() for ckpt, _, _, _ in rows]
    assert len(sources) == len(set(sources))
    for key, rows in manifest.items():
        for ckpt, index, src_rows, _ in rows:
            assert (index is None) == (not key.endswith("_qkv")), (key, index)
            if key.endswith(_EXPERT_STACKS):
                assert ckpt.endswith((".moe.experts.up_proj", ".moe.experts.down_proj")), key
                assert src_rows == _window(rank), (key, src_rows)
            else:
                assert src_rows is None, (key, src_rows)


@_MTP
@_RANK
def test_coverage_holds_in_both_directions(mtp, rank):
    """Every text key is consumed on every rank (lm_head by the shell, the expert
    stacks for the rank's own rows), and what is left is exactly the predicted
    non-load: the vision tower and projector, and the MTP module when no
    speculative config makes it live."""
    core = _core(mtp, rank)
    checkpoint = _reference_text_keys(mtp=True) | _VISION_KEYS
    consumed = _consumed(W.MODEL_WEIGHTS.manifest(core)) | {"lm_head.weight"}
    assert consumed <= checkpoint, sorted(consumed - checkpoint)[:8]
    assert consumed == _reference_text_keys(mtp)
    mtp_keys = _reference_text_keys(mtp=True) - _reference_text_keys(mtp=False)
    expected = _VISION_KEYS | (set() if mtp else mtp_keys)
    assert W.MODEL_WEIGHTS.expected_unconsumed(core, dict.fromkeys(checkpoint)) == expected
    assert checkpoint - consumed == expected


@_MTP
def test_the_ranks_split_the_experts_and_nothing_else(mtp):
    """The four windows of every MoE block's stacks tile its 512 experts once, in
    rank order; every other row reads the same whole tensor on every rank."""
    manifests = [W.MODEL_WEIGHTS.manifest(_core(mtp, rank)) for rank in _RANKS]
    assert all(set(m) == set(manifests[0]) for m in manifests)
    num_experts = _TEXT_CONFIG["n_routed_experts"]
    stacks = 0
    for key in manifests[0]:
        rows = [m[key] for m in manifests]
        if not key.endswith(_EXPERT_STACKS):
            assert all(r == rows[0] for r in rows), key
            continue
        stacks += 1
        windows = [src_rows for r in rows for _, _, src_rows, _ in r]
        covered = [e for window in windows for e in range(num_experts)[window]]
        assert covered == list(range(num_experts)), key
        assert len({ckpt for r in rows for ckpt, _, _, _ in r}) == 1, key
    moe_blocks = sum(t.startswith("shortcut_moe") for t in _TEXT_CONFIG["layer_types"])
    assert stacks == 2 * (moe_blocks + (1 if mtp else 0))


def test_mtp_rows_appear_only_when_mtp_is_live():
    """A non-speculative engine must not carry the MTP layer's weights."""
    off = set(W.MODEL_WEIGHTS.manifest(_core(False, 0)))
    on = set(W.MODEL_WEIGHTS.manifest(_core(True, 0)))
    assert not {k for k in off if k.startswith(f"l{_NUM_LAYERS}_")}
    assert {k for k in on if k.startswith(f"l{_NUM_LAYERS}_")}
    assert off < on


def test_fused_qkv_rows_tile_the_operand():
    """q, k and v land side by side in the rows of one operand, in that order,
    covering it exactly: the fused-qkv GEMM's columns are thop_attention's
    `[q heads | k head | v head]`."""
    manifest = W.MODEL_WEIGHTS.manifest(_core(True, 0))
    q = _TEXT_CONFIG["num_attention_heads"] * _TEXT_CONFIG["head_dim"]
    kv = _TEXT_CONFIG["num_key_value_heads"] * _TEXT_CONFIG["head_dim"]
    fused = 0
    for key, rows in manifest.items():
        if not key.endswith("_qkv"):
            continue
        fused += 1
        assert [(ckpt.rsplit(".", 2)[-2], index) for ckpt, index, _, _ in rows] == [
            ("q_proj", (slice(0, q),)),
            ("k_proj", (slice(q, q + kv),)),
            ("v_proj", (slice(q + kv, q + 2 * kv),)),
        ], key
    assert fused == sum(t == "shortcut_moe_attn" for t in _TEXT_CONFIG["layer_types"]) + 1


def _checkpoint_headers() -> dict[str, tuple[str, tuple[int, ...]]]:
    """Every tensor's (dtype, shape) from the safetensors headers, no data read."""
    index = json.loads((_CHECKPOINT / "model.safetensors.index.json").read_text())
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(_CHECKPOINT / shard, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        header.pop("__metadata__", None)
        headers.update({k: (v["dtype"], tuple(v["shape"])) for k, v in header.items()})
    assert set(headers) == set(index["weight_map"]), "index and shard headers disagree"
    return headers


_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


def _nbytes(dtype: torch.dtype, shape) -> int:
    return torch.Size(shape).numel() * dtype.itemsize


@pytest.mark.skipif(
    not (_CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason="nemotron4-nano-preview is not under LLM_MODELS_ROOT",
)
@_MTP
def test_the_table_matches_the_checkpoint_on_disk(mtp):
    """The transcriptions above, held to the checkpoint itself: its text keys are
    exactly the transcribed ones, and on every rank every manifest row names a
    tensor whose rows, cut to the row's window and run through the row's
    transform on a meta tensor of the stored shape and dtype, land exactly on
    the declared slice. A rank's weights are then the stored text model's bytes
    less three quarters of the routed experts' -- the replicated weights whole
    and one window of every stack -- and less half of every wide-residual logit,
    stored fp32 and kept as its bf16 coefficient."""
    headers = _checkpoint_headers()
    vision = {k for k in headers if k.startswith(W._VISION_PREFIXES)}
    assert set(headers) - vision == _reference_text_keys(mtp=True)

    text_keys = _reference_text_keys(mtp)
    stored = sum(_nbytes(_DTYPES[headers[k][0]], headers[k][1]) for k in text_keys)
    stacks = sum(
        _nbytes(_DTYPES[headers[k][0]], headers[k][1])
        for k in text_keys
        if k.endswith((".moe.experts.up_proj", ".moe.experts.down_proj"))
    )
    logits = [k for k in text_keys if k.endswith("logit")]
    assert all(headers[k][0] == "F32" for k in logits)
    halved = sum(_nbytes(torch.float32, headers[k][1]) // 2 for k in logits)
    lm_dtype, lm_shape = headers["lm_head.weight"]
    vocab, hidden = _TEXT_CONFIG["vocab_size"], _TEXT_CONFIG["hidden_size"]
    assert (_DTYPES[lm_dtype], lm_shape) == (torch.bfloat16, (vocab, hidden))

    for rank in _RANKS:
        core = _core(mtp, rank)
        d = W.MODEL_WEIGHTS.dims(core)
        declared = _declared(mtp, rank)
        for param_key, rows in W.MODEL_WEIGHTS.manifest(core).items():
            for ckpt, index, src_rows, transform in rows:
                dtype, shape = headers[ckpt]
                if src_rows is not None:
                    assert 0 <= src_rows.start < src_rows.stop <= shape[0], (ckpt, src_rows)
                    shape = (src_rows.stop - src_rows.start, *shape[1:])
                src = torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
                if transform is not None:
                    src = transform(src, d)
                dst = declared[param_key] if index is None else declared[param_key][index]
                assert tuple(dst.shape) == tuple(src.shape), (param_key, ckpt, shape)
                assert dst.dtype == src.dtype, (param_key, ckpt, dtype)
        leftover = set(headers) - _consumed(W.MODEL_WEIGHTS.manifest(core))
        leftover.discard("lm_head.weight")
        assert leftover == W.MODEL_WEIGHTS.expected_unconsumed(core, headers)

        rank_bytes = sum(p.numel() * p.dtype.itemsize for p in declared.values())
        rank_bytes += _nbytes(torch.bfloat16, lm_shape)
        assert rank_bytes == stored - stacks + stacks // _EP - halved, rank
