# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: nemotron4-nano-preview (text model) / sm_103 / tp1.

One `W` per weight role, carrying its shape, checkpoint source, dtype and
load-time transform. tp1: no sharding -- every checkpoint tensor reaches
exactly one parameter whole. Most weights keep the checkpoint's own layout and
dtype: HF `[out, in]` row-major for every projection (the forward hands
`cublas_mm` their `.t()` views), `[E, out, in]` for the two routed-expert
stacks (stored fused, one tensor per stack, as `latent_moe_experts` takes
them), fp32 for the router's correction bias. Four kinds are laid out at load
for the entries that read them, so the forward does no per-step arithmetic on
weights:

* `qkv`: the attention's q, k and v projections side by side, `[q | k | v]`
  rows, one `[4608, hidden]` operand for the fused-qkv GEMM `thop_attention`
  reads;
* the wide-residual coefficients, each formed from its fp32 logit exactly as
  HF forms it on every forward and rounded to bf16 as HF's
  `.to(streams.dtype)` rounds it: `f = sigmoid(read logit)` for every read,
  `w = 2 sigmoid(write_logit)` and `r = 1 - max_forget * sigmoid(-retention_logit)`
  for every write -- the `[n]` vectors `wide_residual` takes;
* `gdp_conv`: the short conv's `[channels, 1, kernel]` as the `[channels,
  kernel]` the conv entries take;
* `gdp_neg_a`: HF's `-A_log.float().exp()`, the fp32 factor of the decay
  `gdp_gates` computes. `dt_bias` stays the checkpoint's bf16.

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
  down to the latent width, the 512 relu2 experts there, `fc2_norm` and
  `fc2_latent_proj` back up, and a shared expert at model width.

The checkpoint is a vision-language composite, so its keys fall in three
families:

* the text model, `model.language_model.*` and `lm_head.weight`: consumed;
* the vision tower and projector, `model.vision_tower.*` and
  `model.multi_modal_projector.*`: a predicted non-load, since this target
  serves the text model alone;
* the multi-token-prediction module, `model.language_model.mtp.layers.0.*`:
  loaded only under a `speculative_config` and a predicted non-load otherwise,
  so a non-speculative engine does not carry its 6.6 GB. With
  `mtp_use_repeated_layer` the checkpoint stores one layer that serves every
  draft depth. Its block is the trunk's shortcut attention + MoE block built
  on a plain residual -- no wide-residual logits -- so it shares those rows,
  addressed as layer `len(layer_types)`, the index the HF code gives it; its
  four own weights (`enorm`, `hnorm`, `eh_proj`, `final_layernorm`) sit
  outside the block.

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
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape; every
    shape is derived from configuration rather than constant.

    `src` says where the bytes come from: a checkpoint key, for a weight that
    is not per-layer, or a callable `(d, i) -> [(key, index), ...]` taking the
    dimension bundle and the layer index -- the checkpoint prefix differs
    between the trunk's blocks and the MTP layer's, which is why it is
    computed rather than templated. `index` is an index into the destination,
    or None for the whole parameter; several sources with disjoint indices
    assemble one parameter side by side.

    `layers` says which layers carry this weight: None for a weight that is
    not per-layer at all, or a callable `d -> iterable[int]`.

    `transform` runs on each source tensor before the copy, on the
    destination's device, as `transform(tensor, d)`. None means the checkpoint
    tensor is already the declared layout.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    layers: Callable | None = None
    transform: Callable | None = None

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        return None if self.layers is None else list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None) -> list[tuple[str, Any]]:
        """`[(checkpoint key, index into the param | None), ...]`."""
        if i is None:
            return [(self.src, None)]
        return list(self.src(d, i))


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


def _read_factors(logit: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """`f = sigmoid(logit)`: a read's stream factors (`Nemotron4WideResidualRead.forward`,
    `Nemotron4WideResidualConnection.read`), formed in fp32 and rounded to the streams' dtype."""
    return torch.sigmoid(logit.float()).to(d.dtype)


def _write_factors(logit: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """`w = 2 sigmoid(write_logit)` (`Nemotron4WideResidualConnection.write`), fp32, rounded."""
    return (2.0 * torch.sigmoid(logit.float())).to(d.dtype)


def _retention_factors(logit: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """`r = 1 - max_forget * sigmoid(-retention_logit)` (`Nemotron4WideResidualConnection.write`),
    fp32, rounded."""
    return (1.0 - d.retention_max_forget * torch.sigmoid(-logit.float())).to(d.dtype)


def _conv_taps(weight: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """`nn.Conv1d`'s depthwise `[channels, 1, kernel]` as the conv entries' `[channels, kernel]`."""
    return weight.reshape(weight.shape[0], weight.shape[-1])


def _neg_a(a_log: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """`-A_log.float().exp()`, the decay's factor as `Nemotron4GDPMixer.forward` forms it."""
    return -a_log.float().exp()


def _connection(prefix: str, suffix: str, layers: Callable) -> tuple[W, ...]:
    """A wide-residual connection's coefficients, each from its fp32 logit: the read
    factors `f`, the write factors `w` and the retention `r`."""
    return tuple(
        W(
            f"{prefix}_{name}",
            shape=lambda d: (d.streams,),
            src=_block(f"{suffix}.{key}_logit"),
            layers=layers,
            transform=transform,
        )
        for name, key, transform in (
            ("read", "read", _read_factors),
            ("write", "write", _write_factors),
            ("retain", "retention", _retention_factors),
        )
    )


def _qkv(d: SimpleNamespace, i: int) -> list[tuple[str, tuple]]:
    """Layer `i`'s q, k and v projections, side by side in the rows of one operand."""
    p, q, kv = _block_prefix(d, i), d.q_width, d.kv_width
    return [
        (f"{p}.self_attn.q_proj.weight", (slice(0, q),)),
        (f"{p}.self_attn.k_proj.weight", (slice(q, q + kv),)),
        (f"{p}.self_attn.v_proj.weight", (slice(q + kv, q + 2 * kv),)),
    ]


class Nemotron4NanoPreviewTp1Weights:
    """nemotron4-nano-preview (text model) / sm_103 / tp1.

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
        *_connection("res", "residual_connection", _plain_blocks),
        W(
            "shortcut_read",
            shape=lambda d: (d.streams,),
            src=_block("shortcut_read.logit"),
            layers=_shortcut_blocks,
            transform=_read_factors,
        ),
        *_connection("mix", "mixer_residual_connection", _shortcut_blocks),
        *_connection("moe", "moe_residual_connection", _shortcut_blocks),
        # --- the GDP mixer ---
        W(
            "gdp_in",
            shape=lambda d: (d.gdp_in, d.hidden),
            src=_block("mixer.in_proj.weight"),
            layers=_gdp_mixers,
        ),
        W(
            "gdp_conv",
            shape=lambda d: (d.gdp_conv, d.gdp_conv_kernel),
            src=_block("mixer.conv1d.weight"),
            layers=_gdp_mixers,
            transform=_conv_taps,
        ),
        W(
            "gdp_neg_a",
            shape=lambda d: (d.gdp_heads,),
            src=_block("mixer.A_log"),
            dtype=torch.float32,
            layers=_gdp_mixers,
            transform=_neg_a,
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
            "qkv",
            shape=lambda d: (d.q_width + 2 * d.kv_width, d.hidden),
            src=_qkv,
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
        W(
            "experts_up",
            shape=lambda d: (d.num_experts, d.moe_inter, d.latent),
            src=_block("moe.experts.up_proj"),
            layers=_moe_blocks,
        ),
        W(
            "experts_down",
            shape=lambda d: (d.num_experts, d.latent, d.moe_inter),
            src=_block("moe.experts.down_proj"),
            layers=_moe_blocks,
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
            transform=_read_factors,
        ),
    )

    def dims(self, core) -> SimpleNamespace:
        """The dimension bundle, read off the text config alone."""
        cfg = core.model_config.pretrained_config.text_config
        householder = cfg.gdp_num_householder
        gdp_value = cfg.gdp_num_heads * cfg.gdp_head_dim
        gdp_key = cfg.gdp_num_groups * cfg.gdp_key_head_dim
        # `in_proj` packs [z | V0..Vm K0..Km Q | b0..bm | a]; the conv runs over the middle.
        gdp_conv = householder * gdp_value + (householder + 1) * gdp_key
        mtp_enabled = getattr(core.model_config, "spec_config", None) is not None
        mtp_layers = 1 if cfg.mtp_use_repeated_layer else cfg.num_nextn_predict_layers
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
            latent=cfg.moe_latent_size,
            moe_inter=cfg.moe_intermediate_size,
            shared_inter=cfg.moe_shared_expert_intermediate_size,
            retention_max_forget=cfg.wide_residual_retention_max_forget,
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

    def manifest(self, core) -> dict[str, list[tuple[str, Any, Callable | None]]]:
        """param key -> [(checkpoint key, index into the param | None, transform | None)]."""
        d = self.dims(core)
        return {
            key: [(ckpt, index, entry.transform) for ckpt, index in entry.sources(d, i)]
            for entry in self.WEIGHTS
            for key, i in self._targets(entry, d)
        }

    def expected_unconsumed(self, core, weights) -> set:
        """The predicted non-load, read off the checkpoint by prefix rather than
        enumerated, so a key belonging to the text trunk can never land here:
        the vision tower and projector always, and the MTP module when no
        speculative config makes it live."""
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

        def fill(param: nn.Parameter, ckpt_key: str, index, transform) -> None:
            assert ckpt_key in weights, f"checkpoint key missing: {ckpt_key}"
            src = weights[ckpt_key][:]  # materialize a lazy slice
            dst = param.data if index is None else param.data[index]
            if transform is not None:
                src = transform(src.to(dst.device, non_blocking=True), d)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert dst.dtype == src.dtype, (ckpt_key, dst.dtype, src.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.add(ckpt_key)

        assert set(manifest) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            assert sources, f"declared weight with no checkpoint source: {param_key}"
            for ckpt_key, index, transform in sources:
                fill(core.w[param_key], ckpt_key, index, transform)

        # Shell-registered exception: the base class owns lm_head (untied). Name
        # every registered parameter outside the target's own ParameterDict
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
        logger.info(
            f"nemotron4 tp1 weights: {len(consumed)} checkpoint tensors consumed into "
            f"{len(manifest)} parameters and lm_head; {len(leftover)} deliberately "
            f"unconsumed ({sum(k.startswith(_VISION_PREFIXES) for k in leftover)} vision "
            f"tower/projector, {sum(k.startswith(_MTP_PREFIX) for k in leftover)} MTP)"
        )


MODEL_WEIGHTS = Nemotron4NanoPreviewTp1Weights()
