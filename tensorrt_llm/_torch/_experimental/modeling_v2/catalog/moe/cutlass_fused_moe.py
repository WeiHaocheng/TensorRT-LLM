# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routed experts through TensorRT-LLM's CUTLASS fused MoE, with the tanh soft-capped relu^2."""

from __future__ import annotations

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.utils import ActivationType

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: The checkpoint's routed experts (`Nemotron4Experts`): 512 experts, top-10, the 768-wide latent
#: as the MoE's hidden size, intermediate 4096, and `activation_clamp_value` 16.
_LATENT_MOE = dict(num_experts=512, top_k=10, hidden=768, intermediate=4096, limit=16.0)


def _cell(
    why: str, tokens: int, ep: int = 1, ep_rank: int = 0, tp: int = 1, tp_rank: int = 0
) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            tokens=tokens,
            ep_size=ep,
            ep_rank=ep_rank,
            tp_size=tp,
            tp_rank=tp_rank,
            **dict(_LATENT_MOE, intermediate=_LATENT_MOE["intermediate"] // tp),
        ),
    )


class CutlassFusedMoe(OpWrapper):
    """`torch.ops.trtllm.fused_moe` on unquantized bf16: each token's top-k experts, weighted,
    summed. New tensor out (or `out_tensor`, written in place and returned).

    For every slot (token t, choice k) whose expert e = `token_selected_experts[t, k]` is one of
    this rank's -- [ep_rank * n, (ep_rank + 1) * n) for n = `fc1_expert_weights.shape[0]` local
    experts -- h = bf16(x_t @ fc1_e^T), a = bf16(act(h)), o = a @ fc2_e^T in fp32, and
    out_t = bf16(sum over the slots of w_tk * o). act is `activation_type`'s; for `Relu2` with a
    per-expert `swiglu_limit` c (and no alpha / beta) it is the tanh soft-capped relu^2,
    relu(c * tanh(h / c))^2, computed in fp32 from the bf16 h and rounded once.

    The routing weight multiplies each expert's fp32 down-projection in the finalize -- the
    generic MoE order. The checkpoint weights the activation before down_proj and rounds there;
    the two are the same sum (down_proj is linear), rounded at other points.

    Layouts as the checkpoint stores them: `fc1_expert_weights` [n, intermediate, hidden] is
    `experts.up_proj`, `fc2_expert_weights` [n, hidden, intermediate] is `experts.down_proj`,
    each sliced to the rank's experts (expert parallelism) or to its share of the intermediate
    dimension (tensor parallelism; up_proj's rows, down_proj's columns). `token_selected_experts`
    [T, k] int32 global ids, `token_final_scales` [T, k] fp32 (the router's normalized, scaled
    weights).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The latent MoE's three shapes, as the nemotron4 targets would call them: tp1's whole expert
    # range at its decode, MTP-verify and prefill sizes; a dep4 rank's window of 128 experts over
    # the four ranks' gathered tokens (its partial sum: bf16, see `note`); a tp4 rank's quarter
    # of every expert's intermediate width. One cell off every tile multiple.
    CELLS: tuple[Cell, ...] = (
        _cell("tp1 decode, one token: ten experts, one row each", 1),
        _cell("tp1, 37 tokens: group sizes off every tile multiple", 37),
        _cell("tp1 decode 512: ~285 experts touched, a few rows each", 512),
        _cell("tp1 MTP verify, 512 requests x 3 tokens", 1536),
        _cell("tp1 prefill 8192, the max_num_tokens baseline: ~160 rows per expert", 8192),
        _cell("tp1 prefill 16384, the larger token budget", 16384),
        _cell("dep4 rank 0, 4 gathered tokens: experts 0..127, ~10 slots in the window", 4, 4, 0),
        _cell("dep4 rank 3, 4096 gathered tokens: experts 384..511, ~80 rows each", 4096, 4, 3),
        _cell("dep4 rank 0, 32768 gathered tokens: four 8192-token prefills", 32768, 4, 0),
        _cell("tp4 rank 1, one token: every expert's intermediate 1024..2047", 1, tp=4, tp_rank=1),
        _cell("tp4 rank 0, 512 tokens at intermediate 1024", 512, tp=4, tp_rank=0),
        _cell("tp4 rank 3, prefill 8192 at intermediate 1024", 8192, tp=4, tp_rank=3),
    )

    note = """
    Certified: bf16 activations and weights, fp32 routing weights, int32 ids, top-10 of 512
    experts, hidden 768, intermediate 4096 and a tp4 rank's 1024, `Relu2` with `swiglu_limit` 16
    on every local expert (the soft cap), no biases, bf16 output, with the tactics the AutoTuner
    picks (`autotune()` around a first call, as the engine's warm-up does). The soft cap is selected
    by `Relu2` plus `swiglu_limit` alone; with `swiglu_alpha` or `swiglu_beta` also given the op
    promotes the layer to the gated SwigluBias, which the non-gated weights do not fit.

    Not certified: every quantized path, fp16 or fp32 activations, other activations, biases,
    LoRA, all-to-all, min-latency mode, unfused finalize, `out_tensor`.

    An fp32 `output_dtype` on bf16 activations is accepted, raises nothing and returns wrong
    values: measured, a relative RMS error of 1.2 to 1.7 against the reference at 1, 64 and 4096
    tokens, whole range or an expert-parallel window, tuned or not (values near 1e30 on the
    released library). So an expert-parallel rank's partial sums leave this op rounded to bf16,
    one rounding more per rank than an fp32 partial would take. `is_valid` refuses the fp32
    output.

    An id outside the rank's window adds nothing (expert parallelism needs exactly that); on a
    single rank the window is every expert, so an id out of [0, 512) is dropped silently. The
    router's contract keeps ids in range.

    The AutoTuner chooses each GEMM's tactic per power-of-two token bucket up to
    `tune_max_num_tokens`; an untuned bucket falls back to a default tactic. A tactic changes the
    accumulation order, not the rounding points; the cells are measured with the tuned ones.

    The tanh is tanh.approx.f32 (relative error below 2^-10.9). The fused finalize sums a
    token's experts without the reference's single final rounding, which is most of the band:
    measured, this op sits 0.53-0.81 ulp of relative RMS from an fp32 evaluation of the
    checkpoint's math on its real inputs and weights, where the checkpoint's own bf16 loop sits
    0.60-1.11.
    """

    def __call__(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        swizzled_input_sf: bool = True,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_int8_woq_per_channel: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        use_fused_finalize: bool = True,
        tune_max_num_tokens: int = 8192,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        activation_type: int = int(ActivationType.Swiglu),
        unpadded_hidden_size: Optional[int] = None,
        out_tensor: Optional[torch.Tensor] = None,
        use_dynamic_fc2_scale: bool = False,
        use_mxfp8_weight_scaling: bool = False,
        fc1_lora_ranks: Optional[torch.Tensor] = None,
        fc1_lora_weight_ptrs: Optional[torch.Tensor] = None,
        fc2_lora_ranks: Optional[torch.Tensor] = None,
        fc2_lora_weight_ptrs: Optional[torch.Tensor] = None,
        gated_lora_ranks: Optional[torch.Tensor] = None,
        gated_lora_weight_ptrs: Optional[torch.Tensor] = None,
        host_request_types: Optional[torch.Tensor] = None,
        host_context_lengths: Optional[torch.Tensor] = None,
        lora_max_low_rank: int = 0,
        fc1_slot_lora_ranks: Optional[torch.Tensor] = None,
        fc1_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
        fc2_slot_lora_ranks: Optional[torch.Tensor] = None,
        fc2_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
        gated_slot_lora_ranks: Optional[torch.Tensor] = None,
        gated_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
        token_to_slot: Optional[torch.Tensor] = None,
        swiglu_clamp_after_silu: bool = False,
    ) -> torch.Tensor:
        """The op returns `[out]`, or `[]` when it wrote `out_tensor`; this returns the tensor."""
        result = torch.ops.trtllm.fused_moe(
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            output_dtype,
            quant_scales=quant_scales,
            input_sf=input_sf,
            swizzled_input_sf=swizzled_input_sf,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_int8_woq_per_channel=use_int8_woq_per_channel,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            use_fused_finalize=use_fused_finalize,
            tune_max_num_tokens=tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=activation_type,
            unpadded_hidden_size=unpadded_hidden_size,
            out_tensor=out_tensor,
            use_dynamic_fc2_scale=use_dynamic_fc2_scale,
            use_mxfp8_weight_scaling=use_mxfp8_weight_scaling,
            fc1_lora_ranks=fc1_lora_ranks,
            fc1_lora_weight_ptrs=fc1_lora_weight_ptrs,
            fc2_lora_ranks=fc2_lora_ranks,
            fc2_lora_weight_ptrs=fc2_lora_weight_ptrs,
            gated_lora_ranks=gated_lora_ranks,
            gated_lora_weight_ptrs=gated_lora_weight_ptrs,
            host_request_types=host_request_types,
            host_context_lengths=host_context_lengths,
            lora_max_low_rank=lora_max_low_rank,
            fc1_slot_lora_ranks=fc1_slot_lora_ranks,
            fc1_slot_lora_weight_ptrs=fc1_slot_lora_weight_ptrs,
            fc2_slot_lora_ranks=fc2_slot_lora_ranks,
            fc2_slot_lora_weight_ptrs=fc2_slot_lora_weight_ptrs,
            gated_slot_lora_ranks=gated_slot_lora_ranks,
            gated_slot_lora_weight_ptrs=gated_slot_lora_weight_ptrs,
            token_to_slot=token_to_slot,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        )
        return out_tensor if out_tensor is not None else result[0]

    def reference(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        swizzled_input_sf: bool = True,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_int8_woq_per_channel: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        use_fused_finalize: bool = True,
        tune_max_num_tokens: int = 8192,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        activation_type: int = int(ActivationType.Swiglu),
        unpadded_hidden_size: Optional[int] = None,
        out_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The certified path in plain torch: unquantized, no biases, `Relu2` with or without the
        soft cap. Rounds where the kernel does: h and the activation to the activation dtype; the
        down-projection, the weighting and the sum in fp32; the output once.

        Takes `__call__`'s parameters through `out_tensor` by name; only the routing, the
        weights, `output_dtype`, `swiglu_limit`, `ep_size` / `ep_rank` and `activation_type` are
        read (tensor parallelism is in the weights' shapes, `tp_size` / `tp_rank` are not).
        """
        assert int(activation_type) == int(ActivationType.Relu2), "reference: Relu2 only"
        dt = input.dtype
        n_local = fc1_expert_weights.shape[0]
        lo = ep_rank * n_local
        local = token_selected_experts.long() - lo
        out = torch.zeros(
            input.shape[0], fc2_expert_weights.shape[1], dtype=torch.float32, device=input.device
        )
        for e in torch.unique(local).tolist():
            if e < 0 or e >= n_local:
                continue
            tok, pos = torch.where(local == e)
            with true_fp32_matmul():
                h = (input[tok].float() @ fc1_expert_weights[e].float().t()).to(dt).float()
            if swiglu_limit is not None:
                c = float(swiglu_limit[e])
                h = c * torch.tanh(h * (1.0 / c))
            a = torch.relu(h).square().to(dt)
            with true_fp32_matmul():
                o = a.float() @ fc2_expert_weights[e].float().t()
            out.index_add_(0, tok, o * token_final_scales[tok, pos, None].float())
        return out.to(output_dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        # Measured: with bf16 activations an fp32 output_dtype runs without an error and returns
        # values a relative RMS error of 1.2-1.7 from the reference.
        assert not (input.dtype == torch.bfloat16 and output_dtype == torch.float32), (
            "an fp32 output_dtype on bf16 activations is accepted and returns wrong values; "
            "take the output in bf16"
        )


cutlass_fused_moe = CutlassFusedMoe()
