# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nemotron4's tp4 weight table declares, and maps, exactly its rank's part of the checkpoint.

tp4 is tensor parallel over 4 ranks without attention DP: a rank holds a
quarter of the GDP heads, of the query heads and of every MLP's intermediate
width (the routed experts' included), and everything else whole, lm_head
included. Transcriptions of the HF modeling code (`modeling_nemotron4.py`) hold
the table to account, each written from the module tree or the forward rather
than imported from the table, so the table cannot agree with itself:

* `_reference_declaration` -- every parameter a rank allocates, at the shape the
  HF module that owns it stores, cut to the rank's part, or the layout the
  target re-lays it into at load (fused qkv, bf16 residual coefficients, conv
  taps, `-exp(A_log)`);
* `_reference_text_keys` -- every text-model key of the checkpoint's state dict;
* `_hf_gdp_rows` and friends -- which rows (or columns) of each checkpoint
  tensor serve which head, read off the reshapes `Nemotron4GDPMixer.forward`,
  `Nemotron4Attention.forward` and `Nemotron4RMSNormGated` apply.

Against those, for every rank with MTP off and on: every declared parameter has
a source, every text key is consumed, the only keys left are the predicted
non-load, each rank's parts are exactly its heads' and its width's, in the order
a mixer of that many heads reads them, and the four ranks' parts tile every split
tensor once. When the checkpoint itself is on disk, its safetensors headers are
read too (headers only, no tensor data): every manifest row's stored part, run
through the row's transform on the meta device, must land exactly on the
declared slice, and a rank's weight bytes are compared against the stored ones.

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

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp4 import (  # noqa: E501
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
_C = SimpleNamespace(**_TEXT_CONFIG)
_NUM_LAYERS = len(_C.layer_types)
_TP = 4
_RANKS = range(_TP)

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

# The checkpoint tensors a tp4 rank reads only part of -- the plan's slicing table --
# by the suffix of their key, with the dimension each is cut along. Everything
# else is read whole, lm_head included.
_SPLIT = {
    "mixer.in_proj.weight": 0,
    "mixer.conv1d.weight": 0,
    "mixer.A_log": 0,
    "mixer.dt_bias": 0,
    "mixer.norm.weight": 0,
    "mixer.out_proj.weight": 1,
    "self_attn.q_proj.weight": 0,
    "self_attn.o_proj.weight": 1,
    "mlp.up_proj.weight": 0,
    "mlp.down_proj.weight": 1,
    "moe.shared_experts.up_proj.weight": 0,
    "moe.shared_experts.down_proj.weight": 1,
    "moe.experts.up_proj": 1,
    "moe.experts.down_proj": 2,
}


def _split_dim(ckpt_key: str) -> int | None:
    return next((dim for suffix, dim in _SPLIT.items() if ckpt_key.endswith(suffix)), None)


def _core(mtp: bool, rank: int):
    """The configuration surface `dims` reads: a composite config whose text_config
    carries this checkpoint's numbers, rank `rank`'s tp4 mapping, and a speculative
    config when MTP is live."""
    pretrained = SimpleNamespace(text_config=SimpleNamespace(**_TEXT_CONFIG))
    mapping = Mapping(world_size=_TP, rank=rank, tp_size=_TP, moe_tp_size=_TP, moe_ep_size=1)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=pretrained,
            mapping=mapping,
            torch_dtype=torch.bfloat16,
            spec_config=object() if mtp else None,
        )
    )


def _reference_declaration(mtp: bool) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Every parameter a rank allocates, transcribed from the HF module tree.

    The fp32 entries are the router's correction bias, which HF keeps fp32
    through load (`_keep_in_fp32_modules_strict`), and `-exp(A_log)`, which HF
    forms in fp32 from its fp32 `A_log`. The wide-residual logits HF also keeps
    fp32 become the bf16 coefficients HF rounds them into on every forward
    (`.to(streams.dtype)`), and the attention's q, k and v one fused operand. A
    split parameter is the HF one with the split dimension divided by 4.
    """
    c = _C
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

    def gdp(i):  # Nemotron4GDPMixer, the rank's heads
        w[f"l{i}_gdp_in"] = ((projection // _TP, hidden), bf16)
        w[f"l{i}_gdp_conv"] = ((conv_dim // _TP, c.gdp_conv_kernel), bf16)
        w[f"l{i}_gdp_neg_a"] = ((c.gdp_num_heads // _TP,), f32)
        w[f"l{i}_gdp_dt_bias"] = ((c.gdp_num_heads // _TP,), bf16)
        w[f"l{i}_gdp_norm"] = ((value_dim // _TP,), bf16)
        w[f"l{i}_gdp_out"] = ((hidden, value_dim // _TP), bf16)

    def attention(i):  # Nemotron4Attention, the rank's query heads over the whole KV head, fused
        q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        w[f"l{i}_qkv"] = ((q // _TP + 2 * kv, hidden), bf16)
        w[f"l{i}_o"] = ((hidden, q // _TP), bf16)

    def moe(i):  # the shortcut block's norms and Nemotron4MoE
        e, latent, inter = c.n_routed_experts, c.moe_latent_size, c.moe_intermediate_size
        shared = c.moe_shared_expert_intermediate_size
        w[f"l{i}_shortcut_norm"] = ((hidden,), bf16)
        w[f"l{i}_pre_moe_norm"] = ((hidden,), bf16)
        w[f"l{i}_router"] = ((e, hidden), bf16)
        w[f"l{i}_router_bias"] = ((e,), f32)
        w[f"l{i}_fc1_latent"] = ((latent, hidden), bf16)
        w[f"l{i}_experts_up"] = ((e, inter // _TP, latent), bf16)
        w[f"l{i}_experts_down"] = ((e, latent, inter // _TP), bf16)
        w[f"l{i}_fc2_norm"] = ((latent,), bf16)
        w[f"l{i}_fc2_latent"] = ((hidden, latent), bf16)
        w[f"l{i}_shared_up"] = ((shared // _TP, hidden), bf16)
        w[f"l{i}_shared_down"] = ((hidden, shared // _TP), bf16)

    for i, layer_type in enumerate(c.layer_types):
        w[f"l{i}_input_norm"] = ((hidden,), bf16)
        if layer_type == "linear_attention":
            gdp(i)
            connection(i, "res")
        elif layer_type == "mlp":
            w[f"l{i}_mlp_up"] = ((c.intermediate_size // _TP, hidden), bf16)
            w[f"l{i}_mlp_down"] = ((hidden, c.intermediate_size // _TP), bf16)
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
    c = _C
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


# --- which rows serve which head, read off the HF forward ------------------------


def _rank_heads(total: int, rank: int) -> range:
    n = total // _TP
    return range(rank * n, (rank + 1) * n)


def _hf_gdp_rows(rank: int) -> list[int]:
    """`in_proj` rows a GDP mixer of rank `rank`'s heads reads, in the order its forward
    splits them -- `Nemotron4GDPMixer.forward` at H/tp heads.

    `projected.split([value_dim, conv_dim, m * H, H])` gives the gate z, the conv
    input, beta and the decay input a. The conv input splits into
    `[m * value_dim | m * key_dim | key_dim]` and reshapes as
    `value.reshape(B, T * m, H, head_dim)`, `key.reshape(B, T * m, G, key_head_dim)`
    and `query.reshape(B, T, G, key_head_dim)`: sub-step j's head h reads value row
    `j * value_dim + h * head_dim + v`. beta reshapes as `(B, T, m, H)`. z is gated per
    group of `value_dim // num_groups` = head_dim channels (`Nemotron4RMSNormGated`), so
    head h owns z rows `h * head_dim + v`. G equals H, so group g serves head g.
    """
    c = _C
    hd, kd, h_all, m = c.gdp_head_dim, c.gdp_key_head_dim, c.gdp_num_heads, c.gdp_num_householder
    value_dim, key_dim = h_all * hd, c.gdp_num_groups * kd
    heads = _rank_heads(h_all, rank)
    conv0 = value_dim
    beta0 = value_dim + m * value_dim + (m + 1) * key_dim
    rows = [h * hd + v for h in heads for v in range(hd)]
    for j in range(m):
        rows += [conv0 + j * value_dim + h * hd + v for h in heads for v in range(hd)]
    for j in range(m):
        rows += [conv0 + m * value_dim + j * key_dim + g * kd + k for g in heads for k in range(kd)]
    rows += [conv0 + m * value_dim + m * key_dim + g * kd + k for g in heads for k in range(kd)]
    for j in range(m):
        rows += [beta0 + j * h_all + h for h in heads]
    rows += [beta0 + m * h_all + h for h in heads]
    return rows


def _hf_gdp_channels(rank: int) -> list[int]:
    """Channels of the recurrent output, the gated norm and `out_proj`'s input that rank
    `rank`'s heads produce: `recurrent_output.reshape(B, T, m, H, head_dim)[:, :, -1]`
    flattened to value_dim, head h at `h * head_dim + v`."""
    hd = _C.gdp_head_dim
    return [h * hd + v for h in _rank_heads(_C.gdp_num_heads, rank) for v in range(hd)]


def _hf_query_rows(rank: int) -> list[int]:
    """`q_proj` rows (and `o_proj` columns) of rank `rank`'s query heads:
    `q_proj(x).view(..., -1, head_dim)` puts head h at rows `h * head_dim + d`, and
    the output reshapes back the same way."""
    d = _C.head_dim
    return [h * d + i for h in _rank_heads(_C.num_attention_heads, rank) for i in range(d)]


def _positions(rows, full: int, dim: int, local: int) -> list[int]:
    """The source positions along `dim` that a manifest row list copies, in the
    parameter's order. Every other dimension must be read whole, and the parts
    must fill the parameter's `local` positions along `dim` once."""
    placed = {}
    for _, dst, src, _ in rows:
        index = src if isinstance(src, tuple) else (src,)
        assert len(index) == dim + 1, index
        assert all(i == slice(None) for i in index[:dim]), index
        ids = list(range(full))[index[dim]]
        start = 0 if dst is None else dst.start
        assert dst is None or (dim == 0 and dst.stop - dst.start == len(ids)), (dst, len(ids))
        placed[start] = ids
    out = [i for start in sorted(placed) for i in placed[start]]
    assert len(out) == local, (len(out), local)
    return out


def _declared(mtp: bool, rank: int) -> dict:
    core = _core(mtp, rank)
    with torch.device("meta"):
        return W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))


def _consumed(manifest: dict) -> set[str]:
    return {ckpt for rows in manifest.values() for ckpt, _, _, _ in rows}


def _query_rows(rows) -> list:
    """A fused qkv operand's `q_proj` row: the part a rank's query heads cut out."""
    return [row for row in rows if row[0].endswith("self_attn.q_proj.weight")]


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
    feeding two parameters is read twice. Neither may happen. Exactly the tensors of
    the slicing table are read in part, along their split dimension; everything else
    is read whole, into the whole parameter -- or, for the KV head's `k_proj` and
    `v_proj`, into their rows of the fused qkv operand."""
    core = _core(mtp, rank)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    assert set(manifest) == set(_declared(mtp, rank).keys())
    assert all(manifest.values())
    owners = {}
    for key, rows in manifest.items():
        for ckpt, dst, src, _ in rows:
            assert owners.setdefault(ckpt, key) == key, (ckpt, owners[ckpt], key)
            dim = _split_dim(ckpt)
            if dim is None:
                fused = key.endswith("_qkv")
                assert src is None and (dst is None) != fused, (key, dst, src)
            else:
                index = src if isinstance(src, tuple) else (src,)
                assert len(index) == dim + 1 and isinstance(index[dim], slice), (key, src)


@_MTP
@_RANK
def test_coverage_holds_in_both_directions(mtp, rank):
    """Every text key is consumed on every rank (lm_head by the shell), and what is
    left is exactly the predicted non-load: the vision tower and projector, and the
    MTP module when no speculative config makes it live."""
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
@_RANK
def test_a_rank_reads_its_own_heads_in_the_order_a_mixer_of_that_many_reads_them(mtp, rank):
    """The GDP `in_proj` rows, conv channels, `A_log`, `dt_bias`, gated-norm weight and
    `out_proj` columns a rank reads are exactly those of its 6 heads, in the layout of a
    6-head mixer; the conv's channels are the rows of `in_proj`'s middle the forward
    feeds it. The attention's `q_proj` rows and `o_proj` columns are its 4 query heads'."""
    c = _C
    manifest = W.MODEL_WEIGHTS.manifest(_core(mtp, rank))
    m = c.gdp_num_householder
    value_dim = c.gdp_num_heads * c.gdp_head_dim
    conv_dim = m * value_dim + (m + 1) * c.gdp_num_groups * c.gdp_key_head_dim
    projection = value_dim + conv_dim + (m + 1) * c.gdp_num_heads
    heads = list(_rank_heads(c.gdp_num_heads, rank))
    gdp = [j for j, t in enumerate(c.layer_types) if t in ("linear_attention", "shortcut_moe_gdp")]
    for i in gdp:
        rows = _positions(manifest[f"l{i}_gdp_in"], projection, 0, projection // _TP)
        assert rows == _hf_gdp_rows(rank), i
        z_local, conv_local = value_dim // _TP, conv_dim // _TP
        conv = _positions(manifest[f"l{i}_gdp_conv"], conv_dim, 0, conv_local)
        assert conv == [r - value_dim for r in rows[z_local : z_local + conv_local]], i
        for name in ("gdp_neg_a", "gdp_dt_bias"):
            assert _positions(manifest[f"l{i}_{name}"], c.gdp_num_heads, 0, len(heads)) == heads
        channels = _hf_gdp_channels(rank)
        assert rows[:z_local] == channels
        assert _positions(manifest[f"l{i}_gdp_norm"], value_dim, 0, z_local) == channels
        assert _positions(manifest[f"l{i}_gdp_out"], value_dim, 1, z_local) == channels
    q = c.num_attention_heads * c.head_dim
    attention = [j for j, t in enumerate(c.layer_types) if t == "shortcut_moe_attn"]
    for i in attention + ([_NUM_LAYERS] if mtp else []):
        query = _query_rows(manifest[f"l{i}_qkv"])
        assert _positions(query, q, 0, q // _TP) == _hf_query_rows(rank), i
        assert _positions(manifest[f"l{i}_o"], q, 1, q // _TP) == _hf_query_rows(rank), i


@_MTP
@_RANK
def test_fused_qkv_rows_tile_the_operand(mtp, rank):
    """The rank's query heads, the KV head's k and its v land side by side in the rows of
    one operand, in that order, covering it exactly: the fused-qkv GEMM's columns are
    thop_attention's `[q heads | k head | v head]` at the rank's 4 query heads."""
    c = _C
    q = c.num_attention_heads * c.head_dim // _TP
    kv = c.num_key_value_heads * c.head_dim
    manifest = W.MODEL_WEIGHTS.manifest(_core(mtp, rank))
    fused = [k for k in manifest if k.endswith("_qkv")]
    attention = sum(t == "shortcut_moe_attn" for t in c.layer_types) + (1 if mtp else 0)
    assert len(fused) == attention
    for key in fused:
        assert [(ckpt.rsplit(".", 2)[-2], dst, src) for ckpt, dst, src, _ in manifest[key]] == [
            ("q_proj", slice(0, q), slice(rank * q, (rank + 1) * q)),
            ("k_proj", slice(q, q + kv), None),
            ("v_proj", slice(q + kv, q + 2 * kv), None),
        ], key


@_MTP
def test_the_ranks_tile_every_split_tensor_and_share_the_rest(mtp):
    """Along its split dimension every split tensor is read once over the four ranks,
    and its up and down halves (into and out of a head, or of an intermediate slice)
    agree on which positions a rank holds; every other tensor is read whole by every
    rank, the fused qkv operand's k and v rows included."""
    manifests = [W.MODEL_WEIGHTS.manifest(_core(mtp, rank)) for rank in _RANKS]
    declared = [_declared(mtp, rank) for rank in _RANKS]
    assert all(set(m) == set(manifests[0]) for m in manifests)
    split = 0
    for key in manifests[0]:
        (ckpt, _, _, _), *_ = manifests[0][key]
        dim = _split_dim(ckpt)
        if dim is None:
            assert all(m[key] == manifests[0][key] for m in manifests), key
            continue
        split += 1
        if key.endswith("_qkv"):
            rows = [_query_rows(m[key]) for m in manifests]
            kv_rows = [[r for r in m[key] if r not in q] for m, q in zip(manifests, rows)]
            assert all(r == kv_rows[0] for r in kv_rows), key
            local = _C.num_attention_heads * _C.head_dim // _TP
        else:
            rows = [m[key] for m in manifests]
            local = declared[0][key].shape[dim]
        full = local * _TP
        held = [_positions(r, full, dim, local) for r in rows]
        assert sorted(p for ranks in held for p in ranks) == list(range(full)), key
    pairs = (
        ("mlp_up", 0, "mlp_down", 1),
        ("shared_up", 0, "shared_down", 1),
        ("experts_up", 1, "experts_down", 2),
    )
    for rank, manifest in enumerate(manifests):
        for up, up_dim, down, down_dim in pairs:
            for key in (k for k in manifest if k.endswith(f"_{up}")):
                twin = key[: -len(up)] + down
                local = declared[rank][key].shape[up_dim]
                ups = _positions(manifest[key], local * _TP, up_dim, local)
                downs = _positions(manifest[twin], local * _TP, down_dim, local)
                assert ups == downs == list(range(rank * local, (rank + 1) * local)), key
    gdp = sum(t in ("linear_attention", "shortcut_moe_gdp") for t in _C.layer_types)
    attention = sum(t == "shortcut_moe_attn" for t in _C.layer_types) + (1 if mtp else 0)
    mlps = sum(t == "mlp" for t in _C.layer_types)
    moe = sum(t.startswith("shortcut_moe") for t in _C.layer_types) + (1 if mtp else 0)
    assert split == 6 * gdp + 2 * attention + 2 * mlps + 4 * moe


def test_mtp_rows_appear_only_when_mtp_is_live():
    """A non-speculative engine must not carry the MTP layer's weights."""
    off = set(W.MODEL_WEIGHTS.manifest(_core(False, 0)))
    on = set(W.MODEL_WEIGHTS.manifest(_core(True, 0)))
    assert not {k for k in off if k.startswith(f"l{_NUM_LAYERS}_")}
    assert {k for k in on if k.startswith(f"l{_NUM_LAYERS}_")}
    assert off < on


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
    exactly the transcribed ones, and on every rank every manifest row cuts a stored
    tensor to a part that its transform -- run on a meta tensor of the stored shape
    and dtype -- turns into exactly the declared slice of its parameter, filling the
    parameter once. A rank's weights are then the stored text model's bytes with
    every split tensor at a quarter, lm_head whole, and each fp32 wide-residual
    logit at the bf16 width of the coefficient it becomes."""
    headers = _checkpoint_headers()
    vision = {k for k in headers if k.startswith(W._VISION_PREFIXES)}
    assert set(headers) - vision == _reference_text_keys(mtp=True)

    def held_bytes(key: str) -> int:
        dtype, shape = headers[key]
        coefficient = key.endswith("_logit") or key.endswith(".logit")
        return _nbytes(torch.bfloat16 if coefficient else _DTYPES[dtype], shape)

    text_keys = _reference_text_keys(mtp)
    whole = sum(held_bytes(k) for k in text_keys if _split_dim(k) is None)
    split = sum(held_bytes(k) for k in text_keys if _split_dim(k) is not None)
    lm_dtype, lm_shape = headers["lm_head.weight"]
    assert (_DTYPES[lm_dtype], lm_shape) == (torch.bfloat16, (_C.vocab_size, _C.hidden_size))

    for rank in _RANKS:
        core = _core(mtp, rank)
        d = W.MODEL_WEIGHTS.dims(core)
        declared = _declared(mtp, rank)
        for param_key, rows in W.MODEL_WEIGHTS.manifest(core).items():
            filled = 0
            for ckpt, dst, src, transform in rows:
                dtype, shape = headers[ckpt]
                stored_part = torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
                stored_part = stored_part if src is None else stored_part[src]
                if transform is not None:
                    stored_part = transform(stored_part, d)
                param = declared[param_key] if dst is None else declared[param_key][dst]
                assert tuple(param.shape) == tuple(stored_part.shape), (param_key, ckpt, src)
                assert param.dtype == stored_part.dtype, (param_key, ckpt, dtype)
                filled += param.numel()
            assert filled == declared[param_key].numel(), param_key
        leftover = set(headers) - _consumed(W.MODEL_WEIGHTS.manifest(core))
        leftover.discard("lm_head.weight")
        assert leftover == W.MODEL_WEIGHTS.expected_unconsumed(core, headers)

        rank_bytes = sum(p.numel() * p.dtype.itemsize for p in declared.values())
        rank_bytes += _nbytes(torch.bfloat16, lm_shape)
        assert rank_bytes == whole + split // _TP, rank
