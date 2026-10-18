# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: nemotron4-nano-preview (text model) / sm_103 / dep4.

One `W` per weight role, carrying its shape, checkpoint source, dtype and the
rows of the source a rank reads. Every rank is handed the whole checkpoint
dict, so the split is entirely this table's, and under attention DP there is
only one:

* **expert windows** -- the checkpoint stores each MoE block's routed experts
  fused, one `[E, out, in]` tensor per stack (`moe.experts.up_proj` and
  `moe.experts.down_proj`), so expert-parallel rank `r` reads rows
  `[r E/ep, (r + 1) E/ep)` of each: 128 of the 512 experts on 4 ranks, the
  28 trunk blocks' and, when live, the MTP layer's. Local expert `j` is global
  expert `offset + j`, and the stack keeps the checkpoint's layout, which is
  the layout `latent_moe_experts` takes for its window. Each rank consumes every
  stack key, for its own rows;
* **replicated** -- everything else, whole on every rank: the topology divides
  the requests, not the weights. The embedding, the norms and wide-residual
  logits, the GDP mixers, the attention projections (one KV head is not split),
  the dense MLPs, the router and its correction bias over all 512 experts, the
  latent projections, the shared experts, the MTP layer's own weights, and the
  shell's `lm_head`, which the base class builds whole under attention DP.

Every tensor keeps the checkpoint's own layout and dtype: HF `[out, in]`
row-major for every projection, `[E, out, in]` for the routed-expert stacks,
`[channels, 1, kernel]` for the GDP short conv, and fp32 for the tensors the
checkpoint keeps fp32: `A_log`, the router's correction bias and every
wide-residual logit. `dt_bias` is bf16 in the checkpoint and stays bf16 here.
No row carries a transform: a kernel that wants another layout adds it to its
row.

Which weights a block has is decided by `text_config.layer_types`:

* every block: `input_layernorm`;
* `linear_attention` (a GDP mixer) and `mlp` (a dense relu2 MLP): one
  wide-residual connection, `residual_connection.{read,write,retention}_logit`;
* `shortcut_moe_attn` / `shortcut_moe_gdp`: an attention or GDP mixer beside a
  latent MoE, with three connections -- `shortcut_read.logit` (the routed
  input's read), `mixer_residual_connection.*` and `moe_residual_connection.*`;
* the GDP mixer: `in_proj` packing `[z | V0..V2 K0..K2 Q | b0..b2 | a]`, the
  conv over the middle (V, K, Q) block, `A_log`, `dt_bias`, the gated norm and
  `out_proj`;
* the latent MoE: router and correction bias at model width, `fc1_latent_proj`
  down to the latent width, this rank's window of the relu2 experts there,
  `fc2_norm` and `fc2_latent_proj` back up, and a shared expert at model width.

The checkpoint is a vision-language composite, so its keys fall in three
families:

* the text model, `model.language_model.*` and `lm_head.weight`: consumed;
* the vision tower and projector, `model.vision_tower.*` and
  `model.multi_modal_projector.*`: a predicted non-load, since this target
  serves the text model alone;
* the multi-token-prediction module, `model.language_model.mtp.layers.0.*`:
  loaded only under a `speculative_config` and a predicted non-load otherwise.
  With `mtp_use_repeated_layer` the checkpoint stores one layer that serves
  every draft depth. Its block is the trunk's shortcut attention + MoE block
  built on a plain residual -- no wide-residual logits -- so it shares those
  rows, its expert window included, addressed as layer `len(layer_types)`, the
  index the HF code gives it; its four own weights (`enorm`, `hnorm`,
  `eh_proj`, `final_layernorm`) sit outside the block.

Loading contract: `load(model, weights)` consumes the engine-provided
checkpoint dict, fills every declared parameter exactly once, and asserts
coverage in both directions -- every declared parameter written, and every
checkpoint key either consumed or in the predicted non-load families above,
named by prefix off the checkpoint rather than enumerated.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger

_TEXT = "model.language_model"

#: The non-text halves of the composite checkpoint.
_VISION_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")
_MTP_PREFIX = f"{_TEXT}.mtp."


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, and which of them this rank reads.

    `shape` takes the dimension bundle and returns the declared shape; every
    shape is derived from configuration rather than constant.

    `src` says where the bytes come from: a checkpoint key, for a weight that
    is not per-layer, or a callable `(d, i) -> [(key, index), ...]` taking the
    dimension bundle and the layer index -- the checkpoint prefix differs
    between the trunk's blocks and the MTP layer's, which is why it is
    computed rather than templated. `index` is an index into the destination,
    or None for the whole parameter.

    `layers` says which layers carry this weight: None for a weight that is
    not per-layer at all, or a callable `d -> iterable[int]`.

    `rows` says which rows of each source tensor this rank reads, as a callable
    `d -> slice` over the source's first dimension -- the rank's expert window
    for the routed-expert stacks -- or None for the whole tensor.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    layers: Callable | None = None
    rows: Callable | None = None

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        return None if self.layers is None else list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None) -> list[tuple[str, Any, slice | None]]:
        """`[(checkpoint key, index into the param | None, source rows | None), ...]`."""
        rows = None if self.rows is None else self.rows(d)
        if i is None:
            return [(self.src, None, rows)]
        return [(key, index, rows) for key, index in self.src(d, i)]


def _block_prefix(d: SimpleNamespace, i: int) -> str:
    """Checkpoint prefix of layer `i`'s block: a trunk block, or the MTP layer's `block`."""
    if i < d.num_layers:
        return f"{_TEXT}.layers.{i}"
    return f"{_TEXT}.mtp.layers.{i - d.num_layers}.block"


def _block(suffix: str) -> Callable:
    """One tensor of layer `i`'s block, whole."""
    return lambda d, i, s=suffix: [(f"{_block_prefix(d, i)}.{s}", None)]


def _mtp(suffix: str) -> Callable:
    """One tensor of an MTP layer outside its block, whole."""
    return lambda d, i, s=suffix: [(f"{_TEXT}.mtp.layers.{i - d.num_layers}.{s}", None)]


def _expert_window(d: SimpleNamespace) -> slice:
    """This rank's routed experts, as rows of a fused `[E, out, in]` stack."""
    return slice(d.expert_offset, d.expert_offset + d.local_experts)


def _of_type(d: SimpleNamespace, *layer_types: str) -> list[int]:
    return [i for i, t in enumerate(d.layer_types) if t in layer_types]


def _mtp_layers(d: SimpleNamespace) -> list[int]:
    """The MTP layers when they are live, after the trunk; none otherwise."""
    return list(range(d.num_layers, d.num_layers + d.mtp_layers))


def _every_block(d: SimpleNamespace) -> list[int]:
    return list(range(d.num_layers)) + _mtp_layers(d)


def _plain_blocks(d: SimpleNamespace) -> list[int]:
    """Blocks with one wide-residual connection: a GDP mixer or a dense MLP alone."""
    return _of_type(d, "linear_attention", "mlp")


def _shortcut_blocks(d: SimpleNamespace) -> list[int]:
    """Trunk blocks with a mixer beside a shortcut MoE, and three wide-residual connections."""
    return _of_type(d, "shortcut_moe_attn", "shortcut_moe_gdp")


def _gdp_mixers(d: SimpleNamespace) -> list[int]:
    return _of_type(d, "linear_attention", "shortcut_moe_gdp")


def _attention_mixers(d: SimpleNamespace) -> list[int]:
    """The trunk's attention blocks, plus the MTP layer's when live."""
    return _of_type(d, "shortcut_moe_attn") + _mtp_layers(d)


def _dense_mlps(d: SimpleNamespace) -> list[int]:
    return _of_type(d, "mlp")


def _moe_blocks(d: SimpleNamespace) -> list[int]:
    """The trunk's shortcut blocks, plus the MTP layer's when live."""
    return _shortcut_blocks(d) + _mtp_layers(d)


def _logits(prefix: str, suffix: str, layers: Callable) -> tuple[W, ...]:
    """A wide-residual connection's fp32 logits: read, write and learned retention."""
    return tuple(
        W(
            f"{prefix}_{name}",
            shape=lambda d: (d.streams,),
            src=_block(f"{suffix}.{key}_logit"),
            dtype=torch.float32,
            layers=layers,
        )
        for name, key in (("read", "read"), ("write", "write"), ("retain", "retention"))
    )


class Nemotron4NanoPreviewDep4Weights:
    """nemotron4-nano-preview (text model) / sm_103 / dep4.

    The table, and the passes over it: `declare` allocates every weight it names,
    `manifest` maps checkpoint keys onto them, `load` fills them. `declare` and
    `manifest` share one "which parameters exist" decision (`_targets`), so they
    cannot disagree about a weight.
    """

    WEIGHTS: tuple[W, ...] = (
        # --- every block, the MTP layer's included ---
        W(
            "input_norm",
            shape=lambda d: (d.hidden,),
            src=_block("input_layernorm.weight"),
            layers=_every_block,
        ),
        # --- the wide residual: one connection per plain block, three per shortcut block ---
        *_logits("res", "residual_connection", _plain_blocks),
        W(
            "shortcut_read",
            shape=lambda d: (d.streams,),
            src=_block("shortcut_read.logit"),
            dtype=torch.float32,
            layers=_shortcut_blocks,
        ),
        *_logits("mix", "mixer_residual_connection", _shortcut_blocks),
        *_logits("moe", "moe_residual_connection", _shortcut_blocks),
        # --- the GDP mixer ---
        W(
            "gdp_in",
            shape=lambda d: (d.gdp_in, d.hidden),
            src=_block("mixer.in_proj.weight"),
            layers=_gdp_mixers,
        ),
        W(
            "gdp_conv",
            shape=lambda d: (d.gdp_conv, 1, d.gdp_conv_kernel),
            src=_block("mixer.conv1d.weight"),
            layers=_gdp_mixers,
        ),
        W(
            "gdp_a_log",
            shape=lambda d: (d.gdp_heads,),
            src=_block("mixer.A_log"),
            dtype=torch.float32,
            layers=_gdp_mixers,
        ),
        W(
            "gdp_dt_bias",
            shape=lambda d: (d.gdp_heads,),
            src=_block("mixer.dt_bias"),
            layers=_gdp_mixers,
        ),
        W(
            "gdp_norm",
            shape=lambda d: (d.gdp_value,),
            src=_block("mixer.norm.weight"),
            layers=_gdp_mixers,
        ),
        W(
            "gdp_out",
            shape=lambda d: (d.hidden, d.gdp_value),
            src=_block("mixer.out_proj.weight"),
            layers=_gdp_mixers,
        ),
        # --- the attention mixer: 16 query heads, one KV head ---
        W(
            "q",
            shape=lambda d: (d.q_width, d.hidden),
            src=_block("self_attn.q_proj.weight"),
            layers=_attention_mixers,
        ),
        W(
            "k",
            shape=lambda d: (d.kv_width, d.hidden),
            src=_block("self_attn.k_proj.weight"),
            layers=_attention_mixers,
        ),
        W(
            "v",
            shape=lambda d: (d.kv_width, d.hidden),
            src=_block("self_attn.v_proj.weight"),
            layers=_attention_mixers,
        ),
        W(
            "o",
            shape=lambda d: (d.hidden, d.q_width),
            src=_block("self_attn.o_proj.weight"),
            layers=_attention_mixers,
        ),
        # --- the dense MLP ---
        W(
            "mlp_up",
            shape=lambda d: (d.mlp_inter, d.hidden),
            src=_block("mlp.up_proj.weight"),
            layers=_dense_mlps,
        ),
        W(
            "mlp_down",
            shape=lambda d: (d.hidden, d.mlp_inter),
            src=_block("mlp.down_proj.weight"),
            layers=_dense_mlps,
        ),
        # --- the shortcut latent MoE ---
        W(
            "shortcut_norm",
            shape=lambda d: (d.hidden,),
            src=_block("shortcut_input_layernorm.weight"),
            layers=_moe_blocks,
        ),
        W(
            "pre_moe_norm",
            shape=lambda d: (d.hidden,),
            src=_block("pre_moe_layernorm.weight"),
            layers=_moe_blocks,
        ),
        W(
            "router",
            shape=lambda d: (d.num_experts, d.hidden),
            src=_block("moe.gate.weight"),
            layers=_moe_blocks,
        ),
        W(
            "router_bias",
            shape=lambda d: (d.num_experts,),
            src=_block("moe.gate.e_score_correction_bias"),
            dtype=torch.float32,
            layers=_moe_blocks,
        ),
        W(
            "fc1_latent",
            shape=lambda d: (d.latent, d.hidden),
            src=_block("moe.fc1_latent_proj.weight"),
            layers=_moe_blocks,
        ),
        # --- this rank's routed experts: its window of each fused stack ---
        W(
            "experts_up",
            shape=lambda d: (d.local_experts, d.moe_inter, d.latent),
            src=_block("moe.experts.up_proj"),
            layers=_moe_blocks,
            rows=_expert_window,
        ),
        W(
            "experts_down",
            shape=lambda d: (d.local_experts, d.latent, d.moe_inter),
            src=_block("moe.experts.down_proj"),
            layers=_moe_blocks,
            rows=_expert_window,
        ),
        W(
            "fc2_norm",
            shape=lambda d: (d.latent,),
            src=_block("moe.fc2_norm.weight"),
            layers=_moe_blocks,
        ),
        W(
            "fc2_latent",
            shape=lambda d: (d.hidden, d.latent),
            src=_block("moe.fc2_latent_proj.weight"),
            layers=_moe_blocks,
        ),
        W(
            "shared_up",
            shape=lambda d: (d.shared_inter, d.hidden),
            src=_block("moe.shared_experts.up_proj.weight"),
            layers=_moe_blocks,
        ),
        W(
            "shared_down",
            shape=lambda d: (d.hidden, d.shared_inter),
            src=_block("moe.shared_experts.down_proj.weight"),
            layers=_moe_blocks,
        ),
        # --- the MTP layer's own weights, outside its block ---
        W("enorm", shape=lambda d: (d.hidden,), src=_mtp("enorm.weight"), layers=_mtp_layers),
        W("hnorm", shape=lambda d: (d.hidden,), src=_mtp("hnorm.weight"), layers=_mtp_layers),
        W(
            "eh",
            shape=lambda d: (d.hidden, 2 * d.hidden),
            src=_mtp("eh_proj.weight"),
            layers=_mtp_layers,
        ),
        W(
            "mtp_norm",
            shape=lambda d: (d.hidden,),
            src=_mtp("final_layernorm.weight"),
            layers=_mtp_layers,
        ),
        # --- not per-layer ---
        W("embed", shape=lambda d: (d.vocab, d.hidden), src=f"{_TEXT}.embed_tokens.weight"),
        W("final_norm", shape=lambda d: (d.hidden,), src=f"{_TEXT}.norm.weight"),
        W(
            "readout",
            shape=lambda d: (d.streams,),
            src=f"{_TEXT}.residual_readout.logit",
            dtype=torch.float32,
        ),
    )

    def dims(self, core) -> SimpleNamespace:
        """The dimension bundle, read off the text config and this rank's place in the mapping."""
        cfg = core.model_config.pretrained_config.text_config
        mapping = core.model_config.mapping
        householder = cfg.gdp_num_householder
        gdp_value = cfg.gdp_num_heads * cfg.gdp_head_dim
        gdp_key = cfg.gdp_num_groups * cfg.gdp_key_head_dim
        # `in_proj` packs [z | V0..Vm K0..Km Q | b0..bm | a]; the conv runs over the middle.
        gdp_conv = householder * gdp_value + (householder + 1) * gdp_key
        mtp_enabled = getattr(core.model_config, "spec_config", None) is not None
        mtp_layers = 1 if cfg.mtp_use_repeated_layer else cfg.num_nextn_predict_layers
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        return SimpleNamespace(
            layer_types=tuple(cfg.layer_types),
            num_layers=len(cfg.layer_types),
            hidden=cfg.hidden_size,
            vocab=cfg.vocab_size,
            streams=cfg.wide_residual_num_streams,
            q_width=cfg.num_attention_heads * cfg.head_dim,
            kv_width=cfg.num_key_value_heads * cfg.head_dim,
            mlp_inter=cfg.intermediate_size,
            gdp_heads=cfg.gdp_num_heads,
            gdp_value=gdp_value,
            gdp_conv=gdp_conv,
            gdp_conv_kernel=cfg.gdp_conv_kernel,
            gdp_in=gdp_value + gdp_conv + (householder + 1) * cfg.gdp_num_heads,
            num_experts=cfg.n_routed_experts,
            ep_rank=mapping.moe_ep_rank,
            local_experts=local_experts,
            expert_offset=local_experts * mapping.moe_ep_rank,
            latent=cfg.moe_latent_size,
            moe_inter=cfg.moe_intermediate_size,
            shared_inter=cfg.moe_shared_expert_intermediate_size,
            mtp_enabled=mtp_enabled,
            mtp_layers=mtp_layers if mtp_enabled else 0,
            dtype=core.model_config.torch_dtype,
        )

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        """`[(param key, layer index | None), ...]`.

        One place decides which parameters an entry declares, so `declare` and
        `manifest` cannot disagree about whether a weight exists.
        """
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None)]
        return [(f"l{i}_{entry.name}", i) for i in layers]

    def manifest(self, core) -> dict[str, list[tuple[str, Any, slice | None]]]:
        """param key -> [(checkpoint key, index into the param | None, source rows | None)]."""
        d = self.dims(core)
        return {
            key: entry.sources(d, i) for entry in self.WEIGHTS for key, i in self._targets(entry, d)
        }

    def expected_unconsumed(self, core, weights) -> set:
        """The predicted non-load, read off the checkpoint by prefix rather than
        enumerated, so a key belonging to the text trunk can never land here:
        the vision tower and projector always, and the MTP module when no
        speculative config makes it live. The routed-expert stacks are not in
        it: every rank consumes each of them, for its own window of rows."""
        prefixes = _VISION_PREFIXES
        if not self.dims(core).mtp_enabled:
            prefixes += (_MTP_PREFIX,)
        return {k for k in weights.keys() if k.startswith(prefixes)}

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint, and assert coverage in
        both directions rather than only checkpoint-side."""
        core = model.model
        d = self.dims(core)
        manifest = self.manifest(core)
        consumed: set = set()
        windowed: set = set()

        def fill(param: nn.Parameter, ckpt_key: str, index, rows) -> None:
            assert ckpt_key in weights, f"checkpoint key missing: {ckpt_key}"
            # Index before materializing: a lazy safetensors slice then reads only these
            # rows, and an in-memory tensor copies only this view of itself.
            src = weights[ckpt_key][slice(None) if rows is None else rows]
            dst = param.data if index is None else param.data[index]
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert dst.dtype == src.dtype, (ckpt_key, dst.dtype, src.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.add(ckpt_key)
            if rows is not None:
                windowed.add(ckpt_key)

        assert set(manifest) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            assert sources, f"declared weight with no checkpoint source: {param_key}"
            for ckpt_key, index, rows in sources:
                fill(core.w[param_key], ckpt_key, index, rows)

        # Shell-registered exception: the base class owns lm_head (untied). Under attention
        # DP it is whole on every rank, so the fill's shape check refuses a vocab-parallel
        # shard. Name every registered parameter outside the target's own ParameterDict
        # rather than assuming lm_head is the only one.
        fill(model.lm_head.weight, "lm_head.weight", None, None)
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly the vision tower, the "
            "projector and (without a speculative config) the MTP module; unexpected "
            f"{sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )
        window = _expert_window(d)
        logger.info(
            f"nemotron4 dep4 weights, EP rank {d.ep_rank}: {len(consumed)} checkpoint tensors "
            f"consumed into {len(manifest)} parameters and lm_head, {len(windowed)} of them "
            f"routed-expert stacks read at experts [{window.start}, {window.stop}) of "
            f"{d.num_experts}; {len(leftover)} deliberately unconsumed "
            f"({sum(k.startswith(_VISION_PREFIXES) for k in leftover)} vision "
            f"tower/projector, {sum(k.startswith(_MTP_PREFIX) for k in leftover)} MTP)"
        )


MODEL_WEIGHTS = Nemotron4NanoPreviewDep4Weights()
