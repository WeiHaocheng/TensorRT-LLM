# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch reference of the nemotron4-nano-preview text model and its MTP predictor.

The math is transcribed from the checkpoint's ``modeling_nemotron4.py`` (each function names the HF
class or function it transcribes) and is the ground truth every nemotron4 target is checked
against. This module imports neither transformers' modeling code nor TensorRT-LLM, so it runs in
the TensorRT-LLM environment and in the HuggingFace one alike.

Numerics: values are rounded to bf16 exactly where the HF code rounds them, and every reduction
runs in IEEE fp32.

- A bf16 ``nn.Linear`` is an fp32 GEMM rounded once to bf16 (``linear``). HF's bf16 GEMM also
  accumulates in fp32; the two differ in summation order only, which flips the bf16 rounding of a
  small fraction of outputs by one ulp.
- The router and the lm_head are fp32 GEMMs with fp32 outputs, as in the HF code (``linear_fp32``).
- Attention is computed in fp32 and rounded once (``causal_attention``). HF's eager path rounds
  the scores and the probabilities to bf16 and its SDPA kernels round the probabilities; this side
  is the more accurate one.
- Every fp32 GEMM runs under ``ieee_fp32()``: this container's torch makes fp32 matmul TF32 by
  default (~1e-3 relative), which would make the reference the inaccurate side of a comparison.
- The GDP recurrence is HF's per-token loop in fp32 (``gated_delta_rule_recurrent``);
  ``gated_delta_rule_chunked`` is an independent chunked (WY) formulation of the same recurrence.

Layouts: one sequence at a time, token-major (activations ``[T, D]``, bf16). The three residual
streams sit side by side, ``[T, 3 * hidden]``, stream k in columns ``k * hidden:(k + 1) * hidden``
(HF's ``unflatten(-1, (3, -1))``). Per-layer state of one sequence (``LayerState``): attention K
and V ``[S, kv_heads, head_dim]`` bf16; GDP conv state ``[conv_dim, kernel - 1]`` bf16 (the last
``kernel - 1`` conv inputs, oldest first); GDP recurrent state ``[heads, key_dim, value_dim]`` fp32
(HF's ``[H, K, V]``; the TensorRT-LLM GDP entries keep ``[H, V, K]``).

Weights are the checkpoint's tensors, unmodified, keyed by their name below a module prefix: the
GDP mixer of layer 0 reads ``in_proj.weight`` from ``model.language_model.layers.0.mixer.``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F

TEXT_PREFIX = "model.language_model."
MTP_PREFIX = TEXT_PREFIX + "mtp.layers.0."
LM_HEAD_KEY = "lm_head.weight"
LAYER_KINDS = ("linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp")
MTP_MODES = ("hf", "eagle")


def default_checkpoint() -> str:
    """The checkpoint under ``$LLM_MODELS_ROOT/nemotron4-nano-preview``."""
    root = os.environ.get("LLM_MODELS_ROOT")
    if not root:
        raise RuntimeError(
            "set LLM_MODELS_ROOT (the checkpoint is $LLM_MODELS_ROOT/nemotron4-nano-preview)"
        )
    return os.path.join(root, "nemotron4-nano-preview")


@contextlib.contextmanager
def ieee_fp32() -> Iterator[None]:
    """fp32 matmuls in IEEE fp32, not TF32, for the duration.

    The same two switches as ``catalog/_op.py`` ``true_fp32_matmul`` (both have to move), restated
    here so that this module imports without TensorRT-LLM.
    """
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


@dataclasses.dataclass(frozen=True)
class TextConfig:
    """The fields of ``config.json`` ``text_config`` the text model and its MTP layer read."""

    hidden_size: int
    vocab_size: int
    layer_types: tuple[str, ...]
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    activation_clamp_value: float | None
    mlp_hidden_act: str
    gdp_num_heads: int
    gdp_num_groups: int
    gdp_head_dim: int
    gdp_key_head_dim: int
    gdp_num_householder: int
    gdp_conv_kernel: int
    gdp_norm_eps: float
    n_routed_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    moe_latent_size: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int | None
    moe_use_norm_before_up_proj: bool
    moe_shortcut_post_norm: bool
    wide_residual_num_streams: int
    wide_residual_learned_retention: bool
    wide_residual_retention_max_forget: float
    residual_in_fp32: bool
    num_nextn_predict_layers: int
    mtp_use_repeated_layer: bool

    @classmethod
    def from_checkpoint(cls, path: str) -> TextConfig:
        with open(os.path.join(path, "config.json")) as f:
            text = json.load(f)["text_config"]
        values = {field.name: text[field.name] for field in dataclasses.fields(cls)}
        values["layer_types"] = tuple(values["layer_types"])
        config = cls(**values)
        # The paths this reference transcribes; the HF code has others this checkpoint never takes.
        unsupported = {
            "mlp_hidden_act": config.mlp_hidden_act != "relu2",
            "residual_in_fp32": config.residual_in_fp32,
            "moe_shortcut_post_norm": config.moe_shortcut_post_norm,
            "mtp_use_repeated_layer": not config.mtp_use_repeated_layer,
            "layer_types": any(kind not in LAYER_KINDS for kind in config.layer_types),
        }
        bad = [name for name, is_bad in unsupported.items() if is_bad]
        if bad:
            raise NotImplementedError(
                f"the reference does not transcribe these config values: {bad}"
            )
        return config

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    @property
    def gdp_value_dim(self) -> int:
        return self.gdp_num_heads * self.gdp_head_dim

    @property
    def gdp_key_dim(self) -> int:
        return self.gdp_num_groups * self.gdp_key_head_dim

    @property
    def gdp_conv_dim(self) -> int:
        n = self.gdp_num_householder
        return n * self.gdp_value_dim + (n + 1) * self.gdp_key_dim


class Checkpoint:
    """Reads tensors of the safetensors checkpoint by key, one at a time.

    Args:
        path: checkpoint directory (``config.json``, ``model.safetensors.index.json``, shards).
    """

    def __init__(self, path: str | None = None):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.path = path or default_checkpoint()
        self.config = TextConfig.from_checkpoint(self.path)
        with open(os.path.join(self.path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(key for key in self.weight_map if key.startswith(prefix))

    def tensor(self, key: str, device: torch.device | str = "cuda") -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard].get_tensor(key).to(device)

    def tensors(self, prefix: str, device: torch.device | str = "cuda") -> dict[str, torch.Tensor]:
        """Every tensor under ``prefix``, keyed by the rest of its name."""
        return {key[len(prefix) :]: self.tensor(key, device) for key in self.keys(prefix)}

    def layer(self, index: int, device: torch.device | str = "cuda") -> dict[str, torch.Tensor]:
        """The weights of decoder layer ``index`` (0..43)."""
        return self.tensors(f"{TEXT_PREFIX}layers.{index}.", device)

    def mtp_layer(self, device: torch.device | str = "cuda") -> dict[str, torch.Tensor]:
        """The weights of the MTP layer (``mtp.layers.0``, which serves every draft depth)."""
        return self.tensors(MTP_PREFIX, device)


def sub(weights: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """The entries of ``weights`` under ``prefix``, keyed by the rest of their name."""
    return {key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)}


# ----------------------------------------------------------------------------------------------
# Element-wise ops, norms and GEMMs
# ----------------------------------------------------------------------------------------------


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """A bias-free bf16 ``nn.Linear``: fp32 GEMM, rounded once to ``x.dtype``.

    Args:
        x: ``[N, in]`` bf16.
        weight: ``[out, in]`` (the checkpoint's layout).
    """
    with ieee_fp32():
        return torch.matmul(x.float(), weight.float().t()).to(x.dtype)


def linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """HF's fp32-output GEMM ``F.linear(x.float(), weight.float())`` (router, lm_head): ``[N, out]`` fp32."""
    with ieee_fp32():
        return torch.matmul(x.float(), weight.float().t())


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Nemotron4RMSNorm``: normalize in fp32, round to ``x.dtype``, then multiply by the weight in
    that dtype (two roundings)."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


def clamped_relu2(x: torch.Tensor, clamp: float | None) -> torch.Tensor:
    """``Nemotron4ClampedActivation`` with ``relu2``: ``relu(bf16(c * tanh(x / c)))**2`` in bf16."""
    if clamp is not None:
        x = (clamp * torch.tanh(x.float() / clamp)).to(x.dtype)
    return torch.square(F.relu(x))


def mlp(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor, clamp: float | None) -> torch.Tensor:
    """``Nemotron4MLP`` (dense MLP and shared expert): ``down(act(up(x)))``."""
    return linear(clamped_relu2(linear(x, up), clamp), down)


def gated_rms_norm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    """``Nemotron4RMSNormGated``: ``x * silu(gate)`` in fp32, RMS-normalized per group of
    ``group_size`` channels, times the weight, rounded once to ``x.dtype``."""
    input_dtype = x.dtype
    h = x.float() * F.silu(gate.float())
    grouped = h.reshape(*h.shape[:-1], -1, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    grouped = grouped * torch.rsqrt(variance + eps)
    h = grouped.reshape_as(h) * weight.float()
    return h.to(input_dtype)


# ----------------------------------------------------------------------------------------------
# Wide residual streams
# ----------------------------------------------------------------------------------------------


def stream_read(streams: torch.Tensor, logit: torch.Tensor, num_streams: int) -> torch.Tensor:
    """``Nemotron4WideResidualRead`` / ``Nemotron4WideResidualConnection.read``.

    ``sum_k f_k * S_k`` with ``f = bf16(sigmoid(logit))``: HF's bf16 matmul over the streams. The
    products of two bf16 values are exact in fp32, so the fp32 sum rounded once is that matmul.

    Args:
        streams: ``[T, num_streams * hidden]`` bf16.
        logit: ``[num_streams]`` fp32 (``read_logit``, ``shortcut_read.logit``, ``residual_readout.logit``).

    Returns:
        ``[T, hidden]`` in the streams' dtype.
    """
    s = streams.unflatten(-1, (num_streams, -1))
    factors = torch.sigmoid(logit.float()).to(s.dtype)
    return (factors.float()[:, None] * s.float()).sum(-2).to(s.dtype)


def stream_write(
    streams: torch.Tensor,
    update: torch.Tensor,
    write_logit: torch.Tensor,
    retention_logit: torch.Tensor | None,
    max_forget: float,
    num_streams: int,
) -> torch.Tensor:
    """``Nemotron4WideResidualConnection.write``: ``S_k <- bf16(bf16(r_k * S_k) + w_k * U)``.

    ``r = bf16(1 - max_forget * sigmoid(-retention_logit))`` (r = 1 without learned retention) and
    ``w = bf16(2 * sigmoid(write_logit))``, both formed in fp32. HF's ``addcmul_`` on bf16 computes
    ``S + U * w`` in fp32 (the product of two bf16 values is exact) and rounds once.

    Args:
        streams: ``[T, num_streams * hidden]`` bf16.
        update: ``[T, hidden]``, the branch output.

    Returns:
        The new streams (the input is not modified).
    """
    s = streams.unflatten(-1, (num_streams, -1))
    u = update.to(s.dtype).unsqueeze(-2)
    write = (2.0 * torch.sigmoid(write_logit.float())).to(s.dtype).unsqueeze(-1)
    if retention_logit is not None:
        retention = 1.0 - max_forget * torch.sigmoid(-retention_logit.float())
        s = (s.float() * retention.to(s.dtype).float().unsqueeze(-1)).to(s.dtype)
    return (s.float() + u.float() * write.float()).to(s.dtype).flatten(-2)


def _connection_read(streams, weights, name, norm_weight, config: TextConfig):
    """``Nemotron4Layer._read``: a connection's read followed by its RMSNorm."""
    x = stream_read(streams, weights[f"{name}.read_logit"], config.wide_residual_num_streams)
    return rms_norm(x, norm_weight, config.rms_norm_eps)


def _connection_write(streams, update, weights, name, config: TextConfig):
    return stream_write(
        streams,
        update,
        weights[f"{name}.write_logit"],
        weights.get(f"{name}.retention_logit"),
        config.wide_residual_retention_max_forget,
        config.wide_residual_num_streams,
    )


# ----------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    extra_k: torch.Tensor | None = None,
    extra_v: torch.Tensor | None = None,
    query_chunk: int = 1024,
) -> torch.Tensor:
    """Softmax attention in fp32 for T queries that are the last T of the S + T keys, rounded once.

    Query t (0-based among the T) sees keys ``0 .. S + t``. ``extra_k`` / ``extra_v``
    ``[T, E, kv_heads, head_dim]`` add E keys private to each query (MTP-Eagle draft depths >= 2,
    see ``mtp_draft_hidden``). Query head h uses KV head ``h // (heads // kv_heads)``, as HF's
    ``repeat_kv``.

    Args:
        q: ``[T, heads, head_dim]``.
        k, v: ``[S + T, kv_heads, head_dim]``.

    Returns:
        ``[T, heads, head_dim]`` in ``q.dtype``.
    """
    num_q, heads, dim = q.shape
    num_k, kv_heads, _ = k.shape
    past = num_k - num_q
    group = heads // kv_heads
    kf = k.float()
    vf = v.float()
    out = torch.empty(num_q, heads, dim, dtype=torch.float32, device=q.device)
    with ieee_fp32():
        for start in range(0, num_q, query_chunk):
            stop = min(start + query_chunk, num_q)
            qc = q[start:stop].float().view(stop - start, kv_heads, group, dim)
            scores = torch.einsum("tkgd,nkd->tkgn", qc, kf) * scale
            rows = torch.arange(start, stop, device=q.device)[:, None]
            cols = torch.arange(num_k, device=q.device)[None, :]
            mask = (cols <= rows + past)[:, None, None, :]
            scores = scores.masked_fill(~mask, float("-inf"))
            if extra_k is not None:
                ek = extra_k[start:stop].float()
                extra_scores = torch.einsum("tkgd,tekd->tkge", qc, ek) * scale
                scores = torch.cat([scores, extra_scores], dim=-1)
            probs = torch.softmax(scores, dim=-1)
            o = torch.einsum("tkgn,nkd->tkgd", probs[..., :num_k], vf)
            if extra_k is not None:
                o = o + torch.einsum(
                    "tkge,tekd->tkgd", probs[..., num_k:], extra_v[start:stop].float()
                )
            out[start:stop] = o.reshape(stop - start, heads, dim)
    return out.to(q.dtype)


def attention(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: TextConfig,
    kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """``Nemotron4Attention``: q/k/v projections, NoPE causal attention with scale
    ``head_dim**-0.5`` (no rotary, no q/k norm, no bias), ``o_proj``.

    Args:
        x: ``[T, hidden]`` bf16, the tokens that follow the cached ones.
        weights: ``q_proj.weight``, ``k_proj.weight``, ``v_proj.weight``, ``o_proj.weight``.
        kv: K and V ``[S, kv_heads, head_dim]`` of the S earlier tokens, or None.

    Returns:
        ``([T, hidden], (K, V) of the S + T tokens)``.
    """
    num_tokens = x.shape[0]
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q = linear(x, weights["q_proj.weight"]).view(num_tokens, heads, dim)
    k = linear(x, weights["k_proj.weight"]).view(num_tokens, kv_heads, dim)
    v = linear(x, weights["v_proj.weight"]).view(num_tokens, kv_heads, dim)
    if kv is not None:
        k = torch.cat([kv[0], k])
        v = torch.cat([kv[1], v])
    o = causal_attention(q, k, v, dim**-0.5)
    return linear(o.reshape(num_tokens, heads * dim), weights["o_proj.weight"]), (k, v)


# ----------------------------------------------------------------------------------------------
# Gated delta product (GDP) mixer
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class GdpState:
    """One sequence's GDP state after its last token.

    Attributes:
        conv: ``[conv_dim, kernel - 1]`` bf16, the last ``kernel - 1`` conv inputs, oldest first.
        recurrent: ``[heads, key_dim, value_dim]`` fp32 (HF's layout).
    """

    conv: torch.Tensor
    recurrent: torch.Tensor


def causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The GDP short conv: depthwise ``nn.Conv1d`` (``groups=channels``, ``padding=kernel - 1``,
    no bias, first T outputs kept) continued from a saved state; fp32 sum rounded once.

    Args:
        x: ``[T, channels]`` bf16, the conv input (before the SiLU).
        weight: ``[channels, 1, kernel]``.
        state: ``[channels, kernel - 1]``, the ``kernel - 1`` inputs before ``x`` (zeros if None).

    Returns:
        ``(y [T, channels] in x.dtype, new state [channels, kernel - 1])``.
    """
    num_tokens, channels = x.shape
    kernel = weight.shape[-1]
    if state is None:
        state = x.new_zeros(channels, kernel - 1)
    xe = torch.cat([state.t().to(x.dtype), x], dim=0)
    w = weight.reshape(channels, kernel).float()
    y = xe[0:num_tokens].float() * w[:, 0]
    for i in range(1, kernel):
        y = y + xe[i : i + num_tokens].float() * w[:, i]
    return y.to(x.dtype), xe[num_tokens:].t().contiguous()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """HF's ``l2norm``: ``x * rsqrt(sum(x * x) + eps)`` over the last dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gated_delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF's ``torch_recurrent_gated_delta_rule`` for one sequence: the per-token loop, all fp32.

    Per step: ``S *= exp(g)``; ``delta = beta * (v - S^T k)``; ``S += k delta^T``; ``o = S^T q``,
    with q and k L2-normalized (eps 1e-6) when ``use_qk_l2norm`` and q scaled by ``key_dim**-0.5``.

    Args:
        q, k: ``[L, heads, key_dim]``.
        v: ``[L, heads, value_dim]``.
        g: ``[L, heads]``, the log decay.
        beta: ``[L, heads]``.
        initial_state: ``[heads, key_dim, value_dim]`` or None (zeros).
        dtype: the compute dtype (fp32 as HF; fp64 measures fp32's error).

    Returns:
        ``(o [L, heads, value_dim], final state [heads, key_dim, value_dim])`` in ``dtype``.
    """
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))
    if use_qk_l2norm:
        q = l2norm(q)
        k = l2norm(k)
    q = q / (q.shape[-1] ** 0.5)
    num_steps, heads, key_dim = k.shape
    if initial_state is None:
        state = torch.zeros(heads, key_dim, v.shape[-1], dtype=dtype, device=v.device)
    else:
        state = initial_state.to(dtype).clone()
    decay = g.exp()[:, :, None, None]
    out = torch.zeros_like(v)
    for i in range(num_steps):
        state = state * decay[i]
        kv_mem = (state * k[i, :, :, None]).sum(dim=-2)
        delta = (v[i] - kv_mem) * beta[i, :, None]
        state = state + k[i, :, :, None] * delta[:, None, :]
        out[i] = (state * q[i, :, :, None]).sum(dim=-2)
    return out, state


def gated_delta_rule_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrence of ``gated_delta_rule_recurrent`` in chunked (WY) form, fp32 throughout.

    An independent formulation, used to check the per-token loop. Within a chunk that starts from
    state ``S_0``, with cumulative log decay ``G_i = g_1 + ... + g_i``, ``gamma_i = exp(G_i)`` and
    ``D[i, j] = exp(G_i - G_j)`` for ``j <= i`` (0 above the diagonal), the state after step i is
    ``S_i = gamma_i S_0 + sum_{j <= i} D[i, j] k_j u_j^T``, where the per-step updates
    ``u_i = beta_i (v_i - (a_i S_{i-1})^T k_i)`` solve the unit lower-triangular system
    ``(I + diag(beta) strict_tril(D * K K^T)) U = diag(beta) (V - diag(gamma) K S_0)``. Then
    ``O = diag(gamma) Q S_0 + (D * Q K^T) U`` and the next chunk starts from
    ``S_n = gamma_n S_0 + K^T diag(D[n, :]) U``.

    Same arguments and returns as ``gated_delta_rule_recurrent``.
    """
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))
    if use_qk_l2norm:
        q = l2norm(q)
        k = l2norm(k)
    q = q / (q.shape[-1] ** 0.5)
    num_steps, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(heads, key_dim, value_dim, dtype=dtype, device=v.device)
    else:
        state = initial_state.to(dtype).clone()
    out = torch.empty(num_steps, heads, value_dim, dtype=dtype, device=v.device)
    with ieee_fp32():
        for start in range(0, num_steps, chunk_size):
            stop = min(start + chunk_size, num_steps)
            n = stop - start
            qc, kc, vc = (t[start:stop].transpose(0, 1) for t in (q, k, v))  # [H, n, D]
            bc = beta[start:stop].t()  # [H, n]
            cum = g[start:stop].t().cumsum(dim=-1)  # [H, n]
            lower = torch.ones(n, n, dtype=torch.bool, device=v.device).tril()
            diff = (cum[:, :, None] - cum[:, None, :]).masked_fill(~lower, float("-inf"))
            decay = diff.exp()  # [H, n, n], exp(G_i - G_j) for j <= i, else 0
            kk = decay * torch.matmul(kc, kc.transpose(1, 2))
            system = torch.eye(n, dtype=dtype, device=v.device) + bc[:, :, None] * kk.tril(-1)
            rhs = bc[:, :, None] * (vc - cum.exp()[:, :, None] * torch.matmul(kc, state))
            updates = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
            qk = decay * torch.matmul(qc, kc.transpose(1, 2))
            o = cum.exp()[:, :, None] * torch.matmul(qc, state) + torch.matmul(qk, updates)
            out[start:stop] = o.transpose(0, 1)
            tail = (cum[:, -1:] - cum).exp()  # [H, n]
            state = cum[:, -1, None, None].exp() * state + torch.matmul(
                kc.transpose(1, 2), tail[:, :, None] * updates
            )
    return out, state


def gdp_mixer(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: TextConfig,
    state: GdpState | None = None,
    recurrence=gated_delta_rule_recurrent,
) -> tuple[torch.Tensor, GdpState]:
    """``Nemotron4GDPMixer`` over one sequence's tokens, continuing from its saved state.

    ``in_proj`` -> [z | conv input (V_0..V_{n-1}, K_0..K_{n-1}, Q) | b_0..b_{n-1} | a]; depthwise
    causal conv + SiLU over the conv input; ``beta = sigmoid(b)`` (bf16); decay
    ``g = -exp(A_log) * softplus(a + dt_bias)`` (fp32); the gated delta rule over the sequence
    expanded n-fold (sub-step j of a token uses K_j, V_j, b_j; the decay acts on sub-step 0 only, the
    query reads the state after sub-step n-1); the output (bf16) through ``Nemotron4RMSNormGated``
    with z, then ``out_proj``. ``linear_attention`` and ``shortcut_moe_gdp`` layers run this same
    mixer.

    Args:
        x: ``[T, hidden]`` bf16.
        weights: ``in_proj.weight``, ``conv1d.weight``, ``dt_bias``, ``A_log``, ``norm.weight``,
            ``out_proj.weight``.
        state: the state after the sequence's earlier tokens, or None (zeros: a new sequence).
        recurrence: ``gated_delta_rule_recurrent`` (HF's loop) or ``gated_delta_rule_chunked``.

    Returns:
        ``([T, hidden], the state after the last token)``.
    """
    projected = linear(x, weights["in_proj.weight"])
    h, state = gdp_core(projected, weights, config, state, recurrence)
    return linear(h, weights["out_proj.weight"]), state


def gdp_core(
    projected: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: TextConfig,
    state: GdpState | None = None,
    recurrence=gated_delta_rule_recurrent,
) -> tuple[torch.Tensor, GdpState]:
    """``gdp_mixer`` between its two GEMMs: conv + SiLU, the gated delta rule, the gated RMSNorm.

    Every step is per token or runs along the sequence, so splitting a sequence anywhere and
    carrying the state reproduces the unsplit run (the GEMMs, whose rounding can depend on the
    row count, are outside).

    Args:
        projected: ``[T, in_proj rows]`` bf16, the ``in_proj`` output.
        weights: ``conv1d.weight``, ``dt_bias``, ``A_log``, ``norm.weight``.

    Returns:
        ``([T, value_dim] bf16, the out_proj input; the state after the last token)``.
    """
    num_tokens = projected.shape[0]
    heads, groups = config.gdp_num_heads, config.gdp_num_groups
    value_head, key_head, n_h = (
        config.gdp_head_dim,
        config.gdp_key_head_dim,
        config.gdp_num_householder,
    )
    value_dim, key_dim, conv_dim = config.gdp_value_dim, config.gdp_key_dim, config.gdp_conv_dim
    gate, mixed, b, a = projected.split([value_dim, conv_dim, n_h * heads, heads], dim=-1)
    conv_out, conv_state = causal_conv1d(
        mixed, weights["conv1d.weight"], None if state is None else state.conv
    )
    mixed = F.silu(conv_out)
    value, key, query = mixed.split([n_h * value_dim, n_h * key_dim, key_dim], dim=-1)
    value = value.reshape(num_tokens * n_h, heads, value_head)
    key = key.reshape(num_tokens * n_h, groups, key_head).repeat_interleave(heads // groups, dim=-2)
    query = query.reshape(num_tokens, groups, key_head).repeat_interleave(heads // groups, dim=-2)
    beta = b.sigmoid().reshape(num_tokens, n_h, heads)
    g = -weights["A_log"].float().exp() * F.softplus(a.float() + weights["dt_bias"].float())
    expanded_query = query.new_zeros(num_tokens, n_h, heads, key_head)
    expanded_query[:, -1] = query
    expanded_g = g.new_zeros(num_tokens, n_h, heads)
    expanded_g[:, 0] = g
    out, recurrent = recurrence(
        expanded_query.flatten(0, 1),
        key,
        value,
        expanded_g.flatten(0, 1),
        beta.flatten(0, 1),
        None if state is None else state.recurrent,
        use_qk_l2norm=True,
    )
    out = out.to(projected.dtype).reshape(num_tokens, n_h, heads, value_head)[:, -1]
    h = gated_rms_norm(
        out.reshape(num_tokens, value_dim),
        gate,
        weights["norm.weight"],
        value_dim // groups,
        config.gdp_norm_eps,
    )
    return h, GdpState(conv_state, recurrent)


# ----------------------------------------------------------------------------------------------
# Latent MoE
# ----------------------------------------------------------------------------------------------


def route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``Nemotron4TopkRouter``: fp32 logits, sigmoid scores, top-k of ``scores + bias`` (one expert
    group, so HF's group mask keeps every expert), the chosen sigmoid scores normalized by their sum
    (+1e-20) and scaled.

    Args:
        x: ``[N, hidden]`` bf16, the routed input.

    Returns:
        ``(logits [N, experts] fp32, top-k ids [N, k] int64 in torch.topk(sorted=False) order,
        weights [N, k] fp32)``.
    """
    logits = linear_fp32(x, gate_weight)
    scores = logits.sigmoid()
    choice = scores + correction_bias
    ids = torch.topk(choice, k=top_k, dim=-1, sorted=False)[1]
    weights = scores.gather(1, ids)
    if top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, ids, weights * scaling


def latent_experts(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    clamp: float | None,
) -> torch.Tensor:
    """``Nemotron4Experts`` in HF's order.

    Per expert, ascending id: ``up`` GEMM (bf16), clamped relu^2, times the routing weight (fp32
    product rounded to bf16) before ``down``, ``down`` GEMM (bf16); the expert outputs accumulated
    in fp32 in ascending expert order; the sum rounded to bf16.

    Args:
        x: ``[N, latent]`` bf16.
        ids, weights: ``[N, k]`` from ``route``.
        up: ``[experts, intermediate, latent]``; down: ``[experts, latent, intermediate]``.

    Returns:
        ``[N, latent]`` bf16.
    """
    acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    by_slot = ids.t()  # [k, N]: torch.where walks (slot, token) as HF's expert mask does
    for expert in torch.unique(ids).tolist():
        slot, token = torch.where(by_slot == expert)
        h = clamped_relu2(linear(x[token], up[expert]), clamp)
        h = (h * weights[token, slot, None]).to(x.dtype)
        acc.index_add_(0, token, linear(h, down[expert]).to(torch.float32))
    return acc.to(x.dtype)


def latent_moe(
    routed_input: torch.Tensor,
    shared_input: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: TextConfig,
) -> torch.Tensor:
    """``Nemotron4MoE``: route at model width, ``fc1_latent_proj``, routed experts, ``fc2_norm``,
    ``fc2_latent_proj``, plus the shared expert on ``shared_input``.

    Args:
        routed_input, shared_input: ``[N, hidden]`` bf16.
        weights: the ``moe.`` tensors of a layer.
    """
    _, ids, topk_weights = route(
        routed_input,
        weights["gate.weight"],
        weights["gate.e_score_correction_bias"],
        config.num_experts_per_tok,
        config.routed_scaling_factor,
    )
    h = linear(routed_input, weights["fc1_latent_proj.weight"])
    h = latent_experts(
        h,
        ids,
        topk_weights,
        weights["experts.up_proj"],
        weights["experts.down_proj"],
        config.activation_clamp_value,
    )
    if config.moe_use_norm_before_up_proj:
        h = rms_norm(h, weights["fc2_norm.weight"], config.rms_norm_eps)
    h = linear(h, weights["fc2_latent_proj.weight"])
    if config.moe_shared_expert_intermediate_size is not None:
        h = h + mlp(
            shared_input,
            weights["shared_experts.up_proj.weight"],
            weights["shared_experts.down_proj.weight"],
            config.activation_clamp_value,
        )
    return h


# ----------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------

LayerState = GdpState | tuple[torch.Tensor, torch.Tensor] | None
"""Per-layer state of one sequence: GDP state, attention (K, V), or None (MLP layers)."""


def gdp_block(streams, weights, config: TextConfig, state: GdpState | None = None):
    """``Nemotron4GDPLayer`` (``linear_attention``): read, ``input_layernorm``, GDP mixer, write.

    Returns ``(streams, GdpState)``.
    """
    x = _connection_read(
        streams, weights, "residual_connection", weights["input_layernorm.weight"], config
    )
    y, state = gdp_mixer(x, sub(weights, "mixer."), config, state)
    return _connection_write(streams, y, weights, "residual_connection", config), state


def mlp_block(streams, weights, config: TextConfig):
    """``Nemotron4MLPLayer`` (``mlp``): read, ``input_layernorm``, dense MLP, write."""
    x = _connection_read(
        streams, weights, "residual_connection", weights["input_layernorm.weight"], config
    )
    y = mlp(
        x,
        weights["mlp.up_proj.weight"],
        weights["mlp.down_proj.weight"],
        config.activation_clamp_value,
    )
    return _connection_write(streams, y, weights, "residual_connection", config)


def shortcut_moe_block(
    streams, weights, config: TextConfig, state: LayerState = None, wide: bool = True
):
    """``Nemotron4ShortcutMoELayer`` with its attention or GDP mixer.

    The routed input ``shortcut_input_layernorm(shortcut_read(S_in))`` is read before the mixer; the
    mixer reads ``input_layernorm(mixer read(S_in))`` and writes S_mid; the shared expert reads
    ``pre_moe_layernorm(moe read(S_mid))``; the MoE output is written to S_mid. With ``wide=False``
    (the MTP layer's block) the stream is one ordinary residual: reads are the identity and writes
    add (bf16).

    Args:
        streams: ``[T, 3 * hidden]`` (``[T, hidden]`` when not wide) bf16.
        weights: the layer's tensors (attention mixers carry ``self_attn.*``, GDP ones ``mixer.*``).
        state: the mixer's state (attention (K, V) or ``GdpState``) or None.

    Returns:
        ``(streams, the mixer's new state)``.
    """
    eps = config.rms_norm_eps
    if wide:
        routed = stream_read(
            streams, weights["shortcut_read.logit"], config.wide_residual_num_streams
        )
        routed = rms_norm(routed, weights["shortcut_input_layernorm.weight"], eps)
        x = _connection_read(
            streams, weights, "mixer_residual_connection", weights["input_layernorm.weight"], config
        )
    else:
        routed = rms_norm(streams, weights["shortcut_input_layernorm.weight"], eps)
        x = rms_norm(streams, weights["input_layernorm.weight"], eps)
    if "self_attn.q_proj.weight" in weights:
        y, state = attention(x, sub(weights, "self_attn."), config, state)
    else:
        y, state = gdp_mixer(x, sub(weights, "mixer."), config, state)
    if wide:
        streams = _connection_write(streams, y, weights, "mixer_residual_connection", config)
        shared = _connection_read(
            streams, weights, "moe_residual_connection", weights["pre_moe_layernorm.weight"], config
        )
    else:
        streams = streams + y.to(streams.dtype)
        shared = rms_norm(streams, weights["pre_moe_layernorm.weight"], eps)
    out = latent_moe(routed, shared, sub(weights, "moe."), config)
    if wide:
        return _connection_write(streams, out, weights, "moe_residual_connection", config), state
    return streams + out.to(streams.dtype), state


def decoder_layer(kind: str, streams, weights, config: TextConfig, state: LayerState = None):
    """One decoder layer of type ``kind`` (``config.layer_types``): ``(streams, new state)``."""
    if kind == "linear_attention":
        return gdp_block(streams, weights, config, state)
    if kind == "mlp":
        return mlp_block(streams, weights, config), None
    return shortcut_moe_block(streams, weights, config, state)


def final_hidden(streams, readout_logit, norm_weight, config: TextConfig) -> torch.Tensor:
    """``Nemotron4TextModel`` end: ``norm(residual_readout(streams))``, the final normed hidden."""
    h = stream_read(streams, readout_logit, config.wide_residual_num_streams)
    return rms_norm(h, norm_weight, config.rms_norm_eps)


# ----------------------------------------------------------------------------------------------
# Multi-token prediction
# ----------------------------------------------------------------------------------------------


def mtp_layer(
    hidden: torch.Tensor,
    embeds: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: TextConfig,
    kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """``Nemotron4MultiTokenPredictionLayer`` over T positions with ordinary causal attention.

    ``final_layernorm(block(eh_proj(cat(enorm(embeds), hnorm(hidden)))))`` where the block is one
    shortcut attention + MoE block on an ordinary residual.

    Args:
        hidden: ``[T, hidden]`` the previous depth's output (depth 1: the main model's final normed
            hidden).
        embeds: ``[T, hidden]`` the embedding of the token each position reads.
        weights: the ``mtp.layers.0.`` tensors.
        kv: the MTP attention's K and V of earlier positions, or None.

    Returns:
        ``([T, hidden], (K, V) of all positions)``.
    """
    eps = config.rms_norm_eps
    x = torch.cat(
        [
            rms_norm(embeds, weights["enorm.weight"], eps),
            rms_norm(hidden, weights["hnorm.weight"], eps),
        ],
        -1,
    )
    x = linear(x, weights["eh_proj.weight"])
    x, kv = shortcut_moe_block(x, sub(weights, "block."), config, kv, wide=False)
    return rms_norm(x, weights["final_layernorm.weight"], eps), kv


def _mtp_eagle_depth(hidden, embeds, weights, config: TextConfig, base_kv, extra_kv):
    """One MTP-Eagle draft depth >= 2, teacher-forced at every position i at once.

    The query at i attends over the depth-1 keys of positions 0..i (``base_kv``), the keys the
    earlier depths >= 2 wrote at i (``extra_kv``, ``[N, E, kv_heads, head_dim]`` each), and its own.
    Returns ``(out [N, hidden], (k, v) [N, kv_heads, head_dim] of this depth)``.
    """
    eps = config.rms_norm_eps
    block = sub(weights, "block.")
    attn = sub(block, "self_attn.")
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    num = hidden.shape[0]
    x = torch.cat(
        [
            rms_norm(embeds, weights["enorm.weight"], eps),
            rms_norm(hidden, weights["hnorm.weight"], eps),
        ],
        -1,
    )
    x = linear(x, weights["eh_proj.weight"])
    routed = rms_norm(x, block["shortcut_input_layernorm.weight"], eps)
    a = rms_norm(x, block["input_layernorm.weight"], eps)
    q = linear(a, attn["q_proj.weight"]).view(num, heads, dim)
    k = linear(a, attn["k_proj.weight"]).view(num, kv_heads, dim)
    v = linear(a, attn["v_proj.weight"]).view(num, kv_heads, dim)
    ek = torch.stack([kk for kk, _ in extra_kv] + [k], dim=1)
    ev = torch.stack([vv for _, vv in extra_kv] + [v], dim=1)
    # Each query sees the base keys 0..i: the same causal mask as a T == S prefill (past = 0).
    o = causal_attention(q, base_kv[0], base_kv[1], dim**-0.5, extra_k=ek, extra_v=ev)
    x = x + linear(o.reshape(num, heads * dim), attn["o_proj.weight"]).to(x.dtype)
    shared = rms_norm(x, block["pre_moe_layernorm.weight"], eps)
    x = x + latent_moe(routed, shared, sub(block, "moe."), config).to(x.dtype)
    return rms_norm(x, weights["final_layernorm.weight"], eps), (k, v)


def mtp_draft_hidden(
    token_ids: torch.Tensor,
    hidden: torch.Tensor,
    embed: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: TextConfig,
    num_depths: int | None = None,
    mode: str = "eagle",
) -> list[torch.Tensor]:
    """Every MTP depth's output at every position of one sequence, teacher-forced.

    Depth k (1-based) at position i reads token ``i + k`` (id 0 past the end, as HF pads) and the
    previous depth's output at i (depth 1: the main model's final normed hidden ``hidden[i]``); its
    logits predict token ``i + k + 1``. One layer serves every depth (``mtp_use_repeated_layer``).
    The depths differ in what their attention sees:

    - ``"hf"``: ``Nemotron4MultiTokenPredictor.forward``: depth k attends causally over its own
      depth's positions 0..i.
    - ``"eagle"``: TensorRT-LLM's MTP-Eagle one-model drafting, where one KV cache serves every
      depth: depth 1 is HF's; depth k >= 2 at i attends over depth 1's entries at 0..i, the entries
      depths 2..k-1 wrote at i, and its own.

    Teacher forcing feeds depth k the true token ``i + k``. Greedy acceptance never sees the
    difference: depth k's draft counts only when depths 1..k-1 matched the true tokens, and then
    they fed exactly those tokens.

    Args:
        token_ids: ``[N]`` int64, the sequence (prompt + continuation).
        hidden: ``[N, hidden]`` bf16, the main model's final normed hidden at each position.
        embed: ``[vocab, hidden]``, ``embed_tokens.weight``.
        weights: the ``mtp.layers.0.`` tensors.
        num_depths: draft depths (default ``num_nextn_predict_layers``).

    Returns:
        One ``[N, hidden]`` per depth.
    """
    if mode not in MTP_MODES:
        raise ValueError(f"mode must be one of {MTP_MODES}")
    num_depths = num_depths or config.num_nextn_predict_layers
    ids = token_ids
    outs = []
    depth1_kv = None
    extra_kv = []
    for depth in range(num_depths):
        ids = F.pad(ids[1:], (0, 1))
        if mode == "hf" or depth == 0:
            hidden, kv = mtp_layer(hidden, embed[ids], weights, config)
            if depth == 0:
                depth1_kv = kv
        else:
            hidden, kv = _mtp_eagle_depth(hidden, embed[ids], weights, config, depth1_kv, extra_kv)
            extra_kv.append(kv)
        outs.append(hidden)
    return outs


def greedy_acceptance(
    token_ids: Sequence[int], drafts: Sequence[Sequence[int]], first_draft_position: int
) -> list[int]:
    """The accepted draft count of every verify step of greedy MTP decoding of ``token_ids``.

    Drafting at position q (the last position whose main-model hidden state is known; token q + 1 is
    the newest greedy token) proposes ``drafts[d][q]`` for token ``q + 2 + d``. The verify step
    accepts the longest matching prefix (a drafts) and the next drafting position is q + 1 + a. The
    walk starts at the prompt's last position and stops when a window would pass the end of the
    sequence, so every counted step had all its drafts checked.

    Returns:
        The accepted count per verify step; the acceptance length is ``mean(1 + a)``, the quantity
        ``accuracy_core.acceptance_length_from_iteration_stats`` reports.
    """
    depth = len(drafts)
    steps = []
    q = first_draft_position
    while q + 1 + depth < len(token_ids):
        accepted = 0
        while accepted < depth and drafts[accepted][q] == token_ids[q + 2 + accepted]:
            accepted += 1
        steps.append(accepted)
        q += 1 + accepted
    return steps


# ----------------------------------------------------------------------------------------------
# Whole model
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class ForwardResult:
    """What ``Nemotron4Reference.forward`` returns for one sequence.

    Attributes:
        hidden: ``[T, hidden]`` bf16, the final normed hidden (what the lm_head and MTP read).
        logits: ``[T or 1, vocab]`` fp32, or None.
        states: the per-layer state after the last token (feed it back to continue the sequence).
        streams: one ``[T, 3 * hidden]`` bf16 per layer (each layer's output), when captured.
    """

    hidden: torch.Tensor
    logits: torch.Tensor | None
    states: list[LayerState]
    streams: list[torch.Tensor] | None


class Nemotron4Reference:
    """The whole text model (and its MTP layer) on one device, weights as in the checkpoint.

    Args:
        checkpoint: checkpoint directory (default ``default_checkpoint()``).
        device: where the weights live (all text weights: ~197 GB in bf16, one GB300).
        layers: which decoder layers to load (default all; a partial model can only run blocks).
        load_mtp: also load the MTP layer.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        device: torch.device | str = "cuda",
        layers: Sequence[int] | None = None,
        load_mtp: bool = True,
    ):
        self.checkpoint = Checkpoint(checkpoint)
        self.config = self.checkpoint.config
        self.device = torch.device(device)
        read = self.checkpoint.tensor
        self.embed = read(TEXT_PREFIX + "embed_tokens.weight", self.device)
        self.readout_logit = read(TEXT_PREFIX + "residual_readout.logit", self.device)
        self.norm_weight = read(TEXT_PREFIX + "norm.weight", self.device)
        # HF upcasts the bf16 lm_head on every call; keep the upcast once.
        self.lm_head_fp32 = read(LM_HEAD_KEY, self.device).float()
        indices = range(self.config.num_layers) if layers is None else layers
        self.layers: dict[int, dict[str, torch.Tensor]] = {
            i: self.checkpoint.layer(i, self.device) for i in indices
        }
        self.mtp = self.checkpoint.mtp_layer(self.device) if load_mtp else None

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The fp32 lm_head: ``[N, vocab]`` fp32."""
        return linear_fp32(hidden, self.lm_head_fp32)

    def top_logits(
        self, hidden: torch.Tensor, k: int, rows_per_chunk: int = 2048
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``k`` largest fp32 logits at each row of ``hidden`` and their ids: ``([N, k], [N, k])``.

        The lm_head runs ``rows_per_chunk`` rows at a time, so a long sequence never holds all its
        ``[N, vocab]`` logits at once.
        """
        values, ids = [], []
        for start in range(0, hidden.shape[0], rows_per_chunk):
            top = torch.topk(self.logits(hidden[start : start + rows_per_chunk]), k, dim=-1)
            values.append(top.values)
            ids.append(top.indices)
        return torch.cat(values), torch.cat(ids)

    def greedy_ids(self, hidden: torch.Tensor, rows_per_chunk: int = 2048) -> torch.Tensor:
        """The argmax of the fp32 logits at each row of ``hidden``: ``[N]`` int64."""
        return torch.cat(
            [
                self.logits(hidden[start : start + rows_per_chunk]).argmax(dim=-1)
                for start in range(0, hidden.shape[0], rows_per_chunk)
            ]
        )

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        states: list[LayerState] | None = None,
        logits: str | None = "last",
        capture_streams: bool = False,
    ) -> ForwardResult:
        """The text model over one sequence's next tokens (``Nemotron4TextModel.forward`` + head).

        Args:
            token_ids: ``[T]`` int64 on the device.
            states: the per-layer states after the sequence's earlier tokens, or None (a new one).
            logits: ``"last"``, ``"all"`` or None.
            capture_streams: keep every layer's output streams.
        """
        config = self.config
        if len(self.layers) != config.num_layers:
            raise RuntimeError("the whole-model forward needs every layer loaded")
        states = states or [None] * config.num_layers
        streams = self.embed[token_ids].repeat(1, config.wide_residual_num_streams)
        captured = [] if capture_streams else None
        new_states = []
        for i, kind in enumerate(config.layer_types):
            streams, state = decoder_layer(kind, streams, self.layers[i], config, states[i])
            new_states.append(state)
            if captured is not None:
                captured.append(streams)
        hidden = final_hidden(streams, self.readout_logit, self.norm_weight, config)
        out_logits = None
        if logits == "all":
            out_logits = self.logits(hidden)
        elif logits == "last":
            out_logits = self.logits(hidden[-1:])
        return ForwardResult(hidden, out_logits, new_states, captured)

    @torch.inference_mode()
    def greedy(
        self, prompt_ids: Sequence[int], max_new_tokens: int, stop_ids: Sequence[int] = ()
    ) -> tuple[list[int], torch.Tensor]:
        """Greedy continuation: a prefill, then one token per step with the states carried.

        Returns:
            ``(generated ids, [len(generated), vocab] fp32 logits that chose them)``.
        """
        ids = torch.tensor(list(prompt_ids), dtype=torch.long, device=self.device)
        result = self.forward(ids)
        generated, step_logits = [], []
        while True:
            step_logits.append(result.logits[-1])
            token = int(result.logits[-1].argmax())
            generated.append(token)
            if len(generated) >= max_new_tokens or token in stop_ids:
                break
            next_ids = torch.tensor([token], dtype=torch.long, device=self.device)
            result = self.forward(next_ids, result.states)
        return generated, torch.stack(step_logits)

    @torch.inference_mode()
    def mtp_drafts(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor,
        mode: str = "eagle",
        num_depths: int | None = None,
    ) -> list[torch.Tensor]:
        """Every depth's greedy draft at every position: one ``[N]`` int64 per depth (see
        ``mtp_draft_hidden``)."""
        outs = mtp_draft_hidden(
            token_ids, hidden, self.embed, self.mtp, self.config, num_depths, mode
        )
        return [self.greedy_ids(h) for h in outs]

    @torch.inference_mode()
    def teacher_forced_mtp_acceptance(
        self,
        token_ids: Sequence[int],
        prompt_len: int,
        hidden: torch.Tensor | None = None,
        mode: str = "eagle",
        num_depths: int | None = None,
    ) -> dict:
        """Teacher-forced greedy MTP acceptance over one sequence (prompt + greedy continuation).

        Args:
            token_ids: the prompt followed by its greedy continuation (e.g. a target's output).
            prompt_len: number of prompt tokens; drafting starts at the prompt's last position.
            hidden: ``[N, hidden]`` the main model's final normed hidden at every position, or None
                to compute it with one forward over the whole sequence.
            mode: ``"eagle"`` (TensorRT-LLM's MTP-Eagle) or ``"hf"`` (see ``mtp_draft_hidden``).

        Returns:
            ``drafts`` (per depth, the draft at every position), ``accepted`` (per verify step),
            ``acceptance_length`` (mean of 1 + accepted), ``verify_steps``, and ``depth_match``: per
            depth, the fraction of positions in the continuation whose teacher-forced draft equals
            the true token.
        """
        ids = torch.tensor(list(token_ids), dtype=torch.long, device=self.device)
        if hidden is None:
            hidden = self.forward(ids, logits=None).hidden
        drafts = [d.tolist() for d in self.mtp_drafts(ids, hidden, mode, num_depths)]
        accepted = greedy_acceptance(list(token_ids), drafts, prompt_len - 1)
        depth_match = []
        for d, draft in enumerate(drafts):
            positions = range(prompt_len - 1, len(token_ids) - 2 - d)
            hits = [draft[i] == token_ids[i + 2 + d] for i in positions]
            depth_match.append(sum(hits) / max(len(hits), 1))
        return dict(
            drafts=drafts,
            accepted=accepted,
            verify_steps=len(accepted),
            acceptance_length=(sum(accepted) + len(accepted)) / max(len(accepted), 1),
            depth_match=depth_match,
        )
