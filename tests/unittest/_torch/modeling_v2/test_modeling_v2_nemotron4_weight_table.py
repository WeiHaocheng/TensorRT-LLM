# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nemotron4's weight table declares, and maps, exactly what the checkpoint stores.

Two transcriptions of the HF modeling code (`modeling_nemotron4.py`) hold the
table to account, each written from the module tree rather than imported from
the table, so the table cannot agree with itself:

* `_reference_declaration` -- every parameter the target allocates, with the
  shape and dtype the HF module that owns it stores;
* `_reference_text_keys` -- every text-model key of the checkpoint's state
  dict, the MTP layer's included.

Against those, coverage is proved in both directions with MTP off and on:
every declared parameter has a source, every text key is consumed, and the
only keys left are the predicted non-load. When the checkpoint itself is on
disk, its index and safetensors headers are read too (headers only, no tensor
data) and every manifest row is compared against the stored shape and dtype.

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

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron4.nemotron4_nano_preview__sm_103__tp1 import (  # noqa: E501
    weights as W,
)

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
    num_nextn_predict_layers=2,
    mtp_use_repeated_layer=True,
)
_NUM_LAYERS = len(_TEXT_CONFIG["layer_types"])

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


def _core(mtp: bool):
    """The configuration surface `dims` reads: a composite config whose text_config
    carries this checkpoint's numbers, and a speculative config when MTP is live."""
    pretrained = SimpleNamespace(text_config=SimpleNamespace(**_TEXT_CONFIG))
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=pretrained,
            torch_dtype=torch.bfloat16,
            spec_config=object() if mtp else None,
        )
    )


def _reference_declaration(mtp: bool) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Every parameter the target allocates, transcribed from the HF module tree.

    The fp32 entries are the ones HF keeps fp32 through load
    (`_keep_in_fp32_modules_strict`): the router's correction bias, every
    wide-residual logit and `A_log`.
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

    def connection(i, name):  # Nemotron4WideResidualConnection
        for logit in ("read", "write", "retain"):
            w[f"l{i}_{name}_{logit}"] = ((streams,), f32)

    def gdp(i):  # Nemotron4GDPMixer
        w[f"l{i}_gdp_in"] = ((projection, hidden), bf16)
        w[f"l{i}_gdp_conv"] = ((conv_dim, 1, c.gdp_conv_kernel), bf16)
        w[f"l{i}_gdp_a_log"] = ((c.gdp_num_heads,), f32)
        w[f"l{i}_gdp_dt_bias"] = ((c.gdp_num_heads,), bf16)
        w[f"l{i}_gdp_norm"] = ((value_dim,), bf16)
        w[f"l{i}_gdp_out"] = ((hidden, value_dim), bf16)

    def attention(i):  # Nemotron4Attention
        q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        w[f"l{i}_q"] = ((q, hidden), bf16)
        w[f"l{i}_k"] = ((kv, hidden), bf16)
        w[f"l{i}_v"] = ((kv, hidden), bf16)
        w[f"l{i}_o"] = ((hidden, q), bf16)

    def moe(i):  # the shortcut block's norms and Nemotron4MoE
        e, latent, inter = c.n_routed_experts, c.moe_latent_size, c.moe_intermediate_size
        shared = c.moe_shared_expert_intermediate_size
        w[f"l{i}_shortcut_norm"] = ((hidden,), bf16)
        w[f"l{i}_pre_moe_norm"] = ((hidden,), bf16)
        w[f"l{i}_router"] = ((e, hidden), bf16)
        w[f"l{i}_router_bias"] = ((e,), f32)
        w[f"l{i}_fc1_latent"] = ((latent, hidden), bf16)
        w[f"l{i}_experts_up"] = ((e, inter, latent), bf16)
        w[f"l{i}_experts_down"] = ((e, latent, inter), bf16)
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
            w[f"l{i}_shortcut_read"] = ((streams,), f32)
            connection(i, "mix")
            connection(i, "moe")
            moe(i)
            if layer_type == "shortcut_moe_attn":
                attention(i)
            else:
                gdp(i)
    w["embed"] = ((c.vocab_size, hidden), bf16)
    w["final_norm"] = ((hidden,), bf16)
    w["readout"] = ((streams,), f32)
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


def _declared(mtp: bool) -> dict:
    core = _core(mtp)
    with torch.device("meta"):
        return W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))


@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
def test_the_table_declares_what_the_hf_modules_store(mtp):
    """Shape and dtype per key, so a failure names the weight that drifted."""
    got, want = _declared(mtp), _reference_declaration(mtp)
    assert set(got.keys()) == set(want.keys()), set(got.keys()) ^ set(want.keys())
    for key, (shape, dtype) in sorted(want.items()):
        assert tuple(got[key].shape) == shape, key
        assert got[key].dtype == dtype, key


@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
def test_every_declared_weight_has_exactly_its_sources(mtp):
    """A declared weight with no source allocates memory nothing ever fills; a
    key mapped twice is filled twice. Neither may happen."""
    core = _core(mtp)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    assert set(manifest) == set(_declared(mtp).keys())
    sources = [ckpt for rows in manifest.values() for ckpt, _ in rows]
    assert all(manifest.values())
    assert len(sources) == len(set(sources))


@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
def test_coverage_holds_in_both_directions(mtp):
    """Every text key is consumed (lm_head by the shell), and what is left is
    exactly the predicted non-load: the vision tower and projector, and the MTP
    module when no speculative config makes it live."""
    core = _core(mtp)
    checkpoint = _reference_text_keys(mtp=True) | _VISION_KEYS
    consumed = {ckpt for rows in W.MODEL_WEIGHTS.manifest(core).values() for ckpt, _ in rows}
    consumed.add("lm_head.weight")
    assert consumed <= checkpoint, sorted(consumed - checkpoint)[:8]
    assert consumed == _reference_text_keys(mtp)
    mtp_keys = _reference_text_keys(mtp=True) - _reference_text_keys(mtp=False)
    expected = _VISION_KEYS | (set() if mtp else mtp_keys)
    assert W.MODEL_WEIGHTS.expected_unconsumed(core, dict.fromkeys(checkpoint)) == expected
    assert checkpoint - consumed == expected


def test_mtp_rows_appear_only_when_mtp_is_live():
    """A non-speculative engine must not carry the MTP layer's 6.6 GB."""
    off = set(W.MODEL_WEIGHTS.manifest(_core(False)))
    on = set(W.MODEL_WEIGHTS.manifest(_core(True)))
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


@pytest.mark.skipif(
    not (_CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason="nemotron4-nano-preview is not under LLM_MODELS_ROOT",
)
@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
def test_the_table_matches_the_checkpoint_on_disk(mtp):
    """The transcriptions above, held to the checkpoint itself: its text keys are
    exactly the transcribed ones, its other keys are the vision tower and
    projector, and every manifest row names a tensor of exactly the declared
    shape and dtype."""
    headers = _checkpoint_headers()
    vision = {k for k in headers if k.startswith(W._VISION_PREFIXES)}
    assert set(headers) - vision == _reference_text_keys(mtp=True)

    core = _core(mtp)
    declared = _declared(mtp)
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    for param_key, rows in W.MODEL_WEIGHTS.manifest(core).items():
        for ckpt, index in rows:
            assert index is None, (param_key, index)
            dtype, shape = headers[ckpt]
            assert tuple(declared[param_key].shape) == shape, (param_key, ckpt, shape)
            assert declared[param_key].dtype == dtypes[dtype], (param_key, ckpt, dtype)
    dtype, shape = headers["lm_head.weight"]
    vocab, hidden = _TEXT_CONFIG["vocab_size"], _TEXT_CONFIG["hidden_size"]
    assert (dtypes[dtype], shape) == (torch.bfloat16, (vocab, hidden))
    leftover = set(headers) - {
        ckpt for rows in W.MODEL_WEIGHTS.manifest(core).values() for ckpt, _ in rows
    }
    leftover.discard("lm_head.weight")
    assert leftover == W.MODEL_WEIGHTS.expected_unconsumed(core, headers)
