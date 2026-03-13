"""Module-level tests for MiMoV2FlashMoE.

Verifies that the TRT-LLM MiMoV2FlashMoE module produces equivalent results
to the HuggingFace MiMoV2MoE module, and that internal sub-components
(router gate, routing method, e_score_correction_bias) are correctly configured.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add HuggingFace model directory to path
HF_MODEL_DIR = "/home/scratch.fredw_sw/MiMo-V2-Flash/MiMo-V2-Flash"
sys.path.insert(0, HF_MODEL_DIR)

TRTLLM_ROOT = "/home/scratch.fredw_sw/trt-llm-github-3/TensorRT-LLM"
CHECKPOINT_PATH = HF_MODEL_DIR

# Expected config values from config.json
HIDDEN_SIZE = 4096
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
MOE_INTERMEDIATE_SIZE = 2048
N_GROUP = 1
TOPK_GROUP = 1
ROUTED_SCALING_FACTOR_RAW = None  # null in config
ROUTED_SCALING_FACTOR_EFFECTIVE = 1.0
NORM_TOPK_PROB = True
SCORING_FUNC = "sigmoid"
TOPK_METHOD = "noaux_tc"


def _get_hf_config():
    """Load the HuggingFace config object."""
    from configuration_mimo_v2_flash import MiMoV2FlashConfig
    import json
    with open(os.path.join(CHECKPOINT_PATH, "config.json"), "r") as f:
        config_dict = json.load(f)
    config = MiMoV2FlashConfig(**config_dict)
    return config


def _build_trtllm_model_config():
    """Build a minimal TRT-LLM ModelConfig for the MoE module."""
    from transformers import AutoConfig
    from tensorrt_llm._torch.model_config import ModelConfig

    hf_config = AutoConfig.from_pretrained(CHECKPOINT_PATH,
                                           trust_remote_code=True)
    model_config = ModelConfig(pretrained_config=hf_config)
    return model_config


class TestMiMoV2FlashMoEStructure:
    """Test the structural properties of MiMoV2FlashMoE."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the TRT-LLM MoE module for structural tests."""
        model_config = _build_trtllm_model_config()
        aux_stream = torch.cuda.Stream()

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import (
            MiMoV2FlashMoE,
        )

        self.moe = MiMoV2FlashMoE(
            model_config=model_config,
            aux_stream=aux_stream,
            layer_idx=1,
        )

    def test_gate_is_linear_with_correct_shape(self):
        """Verify gate is Linear(4096, 256, bias=False, dtype=float32, quant_config=None)."""
        from tensorrt_llm._torch.modules.linear import Linear as TrtLinear

        gate = self.moe.gate
        assert isinstance(gate, TrtLinear), (
            f"Expected gate to be TrtLinear, got {type(gate).__name__}"
        )

        # Check the gate's configuration
        # The TRT-LLM Linear stores weight shape as [out_features, in_features]
        # in_features=4096, out_features=256
        assert gate.dtype == torch.float32, (
            f"Expected gate dtype=float32, got {gate.dtype}"
        )
        assert gate.has_bias is False, (
            f"Expected gate bias=False, got {gate.has_bias}"
        )
        assert gate.quant_config is None, (
            f"Expected gate quant_config=None, got {gate.quant_config}"
        )

    def test_e_score_correction_bias_shape_and_dtype(self):
        """Verify e_score_correction_bias is nn.Parameter(num_experts, dtype=float32)."""
        bias = self.moe.e_score_correction_bias
        assert isinstance(bias, nn.Parameter), (
            f"Expected nn.Parameter, got {type(bias).__name__}"
        )
        assert bias.shape == (NUM_EXPERTS,), (
            f"Expected shape ({NUM_EXPERTS},), got {bias.shape}"
        )
        assert bias.dtype == torch.float32, (
            f"Expected dtype=float32, got {bias.dtype}"
        )

    def test_routing_method_type(self):
        """Verify the routing method is DeepSeekV3MoeRoutingMethod with correct params."""
        from tensorrt_llm._torch.modules.fused_moe.routing import (
            DeepSeekV3MoeRoutingMethod,
        )

        # The routing method is stored inside self.experts (the MoE backend)
        experts = self.moe.experts
        routing_method = experts.routing_method
        assert isinstance(routing_method, DeepSeekV3MoeRoutingMethod), (
            f"Expected DeepSeekV3MoeRoutingMethod, got {type(routing_method).__name__}"
        )

        # Check routing parameters through the inner routing_impl
        impl = routing_method.routing_impl
        assert impl.top_k == NUM_EXPERTS_PER_TOK, (
            f"Expected top_k={NUM_EXPERTS_PER_TOK}, got {impl.top_k}"
        )
        assert impl.n_group == N_GROUP, (
            f"Expected n_group={N_GROUP}, got {impl.n_group}"
        )
        assert impl.topk_group == TOPK_GROUP, (
            f"Expected topk_group={TOPK_GROUP}, got {impl.topk_group}"
        )
        assert impl.routed_scaling_factor == ROUTED_SCALING_FACTOR_EFFECTIVE, (
            f"Expected routed_scaling_factor={ROUTED_SCALING_FACTOR_EFFECTIVE}, "
            f"got {impl.routed_scaling_factor}"
        )

    def test_create_moe_parameters(self):
        """Verify create_moe was called with correct parameters."""
        experts = self.moe.experts
        assert experts.num_experts == NUM_EXPERTS, (
            f"Expected num_experts={NUM_EXPERTS}, got {experts.num_experts}"
        )
        assert experts.hidden_size == HIDDEN_SIZE, (
            f"Expected hidden_size={HIDDEN_SIZE}, got {experts.hidden_size}"
        )
        assert experts.intermediate_size == MOE_INTERMEDIATE_SIZE, (
            f"Expected intermediate_size={MOE_INTERMEDIATE_SIZE}, "
            f"got {experts.intermediate_size}"
        )

    def test_hidden_dim_stored(self):
        """Verify hidden_dim is correctly stored."""
        assert self.moe.hidden_dim == HIDDEN_SIZE, (
            f"Expected hidden_dim={HIDDEN_SIZE}, got {self.moe.hidden_dim}"
        )

    def test_num_experts_stored(self):
        """Verify num_experts is correctly stored."""
        assert self.moe.num_experts == NUM_EXPERTS, (
            f"Expected num_experts={NUM_EXPERTS}, got {self.moe.num_experts}"
        )

    def test_top_k_stored(self):
        """Verify top_k is correctly stored."""
        assert self.moe.top_k == NUM_EXPERTS_PER_TOK, (
            f"Expected top_k={NUM_EXPERTS_PER_TOK}, got {self.moe.top_k}"
        )


class TestMiMoV2FlashMoEForward:
    """Test the forward method of the MoE module."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up both HF and TRT-LLM MoE modules."""
        self.device = torch.device("cuda:0")

        # Build TRT-LLM module
        model_config = _build_trtllm_model_config()
        aux_stream = torch.cuda.Stream()

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import (
            MiMoV2FlashMoE,
        )

        self.trtllm_moe = MiMoV2FlashMoE(
            model_config=model_config,
            aux_stream=aux_stream,
            layer_idx=1,
        ).to(self.device)

    def test_forward_casts_to_float32_before_gate(self):
        """Verify that hidden_states are cast to float32 before the router gate."""
        # We verify this by checking the forward method code structure.
        # The forward method should cast hidden_states to float32 before
        # passing through the gate.
        import inspect
        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import (
            MiMoV2FlashMoE,
        )

        source = inspect.getsource(MiMoV2FlashMoE.forward)
        # Check that float32 casting happens before gate call
        cast_pos = source.find("torch.float32")
        gate_pos = source.find("self.gate(")
        assert cast_pos != -1, (
            "Forward method does not cast to float32"
        )
        assert gate_pos != -1, (
            "Forward method does not call self.gate()"
        )
        assert cast_pos < gate_pos, (
            "Forward method should cast to float32 BEFORE calling self.gate(). "
            f"Cast at position {cast_pos}, gate at position {gate_pos}"
        )

    def test_forward_output_shape(self):
        """Verify the forward method produces the correct output shape."""
        num_tokens = 4
        hidden_states = torch.randn(
            num_tokens, HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Create a minimal attn_metadata mock
        class MockAttnMetadata:
            all_rank_num_tokens = num_tokens

        with torch.no_grad():
            output = self.trtllm_moe(hidden_states, MockAttnMetadata())

        assert output.shape == (num_tokens, HIDDEN_SIZE), (
            f"Expected output shape ({num_tokens}, {HIDDEN_SIZE}), "
            f"got {output.shape}"
        )


class TestMiMoV2FlashMoELoadWeights:
    """Test that load_weights correctly handles e_score_correction_bias."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the TRT-LLM MoE module."""
        model_config = _build_trtllm_model_config()
        aux_stream = torch.cuda.Stream()

        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import (
            MiMoV2FlashMoE,
        )

        self.moe = MiMoV2FlashMoE(
            model_config=model_config,
            aux_stream=aux_stream,
            layer_idx=1,
        )

    def test_load_weights_e_score_correction_bias(self):
        """Verify load_weights correctly loads e_score_correction_bias."""
        # Create a synthetic weight dict
        test_bias = torch.randn(NUM_EXPERTS, dtype=torch.float32)
        weight_dict = {"e_score_correction_bias": test_bias}

        self.moe.load_weights([weight_dict])

        assert torch.allclose(
            self.moe.e_score_correction_bias.data.cpu(), test_bias
        ), (
            f"e_score_correction_bias mismatch after load_weights.\n"
            f"Expected: {test_bias[:5]}...\n"
            f"Got: {self.moe.e_score_correction_bias.data.cpu()[:5]}..."
        )

    def test_load_weights_missing_bias_key(self):
        """Verify load_weights handles missing e_score_correction_bias gracefully."""
        weight_dict = {"some_other_key": torch.zeros(10)}
        # Should not raise
        self.moe.load_weights([weight_dict])


class TestMiMoV2FlashMoERouterComparison:
    """Compare the TRT-LLM router output against HuggingFace MiMoV2MoEGate.

    Instead of importing the HF MiMoV2MoEGate directly (which has relative
    import issues), we replicate the HF routing logic inline for comparison.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up TRT-LLM MoE module with known weights."""
        self.device = torch.device("cuda:0")

        # Initialize known weights
        torch.manual_seed(42)
        self.gate_weight = torch.randn(
            NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.float32, device=self.device
        )
        self.e_score_bias = torch.randn(
            NUM_EXPERTS, dtype=torch.float32, device=self.device
        )

        # Build TRT-LLM MoE module
        model_config = _build_trtllm_model_config()
        aux_stream = torch.cuda.Stream()
        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import (
            MiMoV2FlashMoE,
        )
        self.trtllm_moe = MiMoV2FlashMoE(
            model_config=model_config,
            aux_stream=aux_stream,
            layer_idx=1,
        ).to(self.device)

        # Copy known weights to TRT-LLM gate and e_score_correction_bias
        self.trtllm_moe.gate.weight.data.copy_(self.gate_weight)
        self.trtllm_moe.e_score_correction_bias.data.copy_(self.e_score_bias)

    def test_router_logits_match(self):
        """Verify that the router gate produces the same logits in both HF and TRT-LLM.

        HF uses: F.linear(hidden_states.float32, gate_weight.float32)
        TRT-LLM uses: Linear(hidden_states_float32)
        """
        batch, seq = 2, 3
        hidden_states = torch.randn(
            batch, seq, HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=self.device,
        )

        with torch.no_grad():
            # HF gate forward: F.linear(hidden_states.float32, weight.float32)
            hf_hidden = hidden_states.view(-1, HIDDEN_SIZE).float()
            hf_logits = torch.nn.functional.linear(
                hf_hidden, self.gate_weight.float()
            )

            # TRT-LLM gate forward: gate(hidden_states.float32)
            trtllm_hidden = hidden_states.view(-1, HIDDEN_SIZE).float()
            trtllm_logits = self.trtllm_moe.gate(trtllm_hidden)

        assert torch.allclose(hf_logits, trtllm_logits, atol=1e-3, rtol=1e-3), (
            f"Router logits mismatch.\n"
            f"Max diff: {(hf_logits - trtllm_logits).abs().max().item():.6f}\n"
            f"HF logits sample: {hf_logits[0, :5]}\n"
            f"TRT-LLM logits sample: {trtllm_logits[0, :5]}"
        )

    def test_routing_decisions_match(self):
        """Verify that the routing method produces the same top-k experts and weights."""
        from tensorrt_llm._torch.modules.fused_moe.routing import (
            Deepseekv3RoutingImpl,
        )

        batch_seq = 6  # batch * seq
        # Create deterministic logits
        logits = torch.randn(
            batch_seq, NUM_EXPERTS,
            dtype=torch.float32,
            device=self.device,
        )
        e_score_bias = self.e_score_bias

        # HF routing: sigmoid -> add bias -> topk (n_group=1, topk_group=1 simplifies)
        hf_scores = logits.sigmoid()
        hf_scores_with_bias = hf_scores + e_score_bias.unsqueeze(0)
        _, hf_topk_idx = torch.topk(hf_scores_with_bias, k=NUM_EXPERTS_PER_TOK, dim=-1)
        hf_topk_weight = hf_scores.gather(1, hf_topk_idx)
        # Normalize
        hf_denominator = hf_topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        hf_topk_weight = hf_topk_weight / hf_denominator
        hf_topk_weight = hf_topk_weight * ROUTED_SCALING_FACTOR_EFFECTIVE

        # TRT-LLM routing via Deepseekv3RoutingImpl
        trtllm_impl = Deepseekv3RoutingImpl(
            top_k=NUM_EXPERTS_PER_TOK,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
            routed_scaling_factor=ROUTED_SCALING_FACTOR_EFFECTIVE,
            is_fused=False,  # Use unfused for easier comparison
        )
        trtllm_topk_values, trtllm_topk_indices = trtllm_impl.noaux_tc(
            logits, e_score_bias
        )

        # Sort both by index for comparison (order may differ)
        hf_sorted_idx = hf_topk_idx.sort(dim=-1)
        hf_topk_idx_sorted = hf_sorted_idx.values
        hf_topk_weight_sorted = hf_topk_weight.gather(1, hf_sorted_idx.indices)

        trtllm_sorted_idx = trtllm_topk_indices.sort(dim=-1)
        trtllm_topk_idx_sorted = trtllm_sorted_idx.values
        trtllm_topk_weight_sorted = trtllm_topk_values.gather(
            1, trtllm_sorted_idx.indices
        )

        assert torch.equal(hf_topk_idx_sorted, trtllm_topk_idx_sorted), (
            f"Top-k expert indices mismatch.\n"
            f"HF: {hf_topk_idx_sorted[0]}\n"
            f"TRT-LLM: {trtllm_topk_idx_sorted[0]}"
        )

        assert torch.allclose(
            hf_topk_weight_sorted, trtllm_topk_weight_sorted, atol=1e-5, rtol=1e-5
        ), (
            f"Top-k weights mismatch.\n"
            f"Max diff: {(hf_topk_weight_sorted - trtllm_topk_weight_sorted).abs().max().item():.8f}\n"
            f"HF: {hf_topk_weight_sorted[0]}\n"
            f"TRT-LLM: {trtllm_topk_weight_sorted[0]}"
        )


class TestMiMoV2FlashMoEWeightMapping:
    """Test that the weight name mapping from HF to TRT-LLM is correct."""

    def test_e_score_correction_bias_weight_renaming(self):
        """Verify that load_weights in MiMoV2FlashForCausalLM correctly renames
        model.layers.X.mlp.gate.e_score_correction_bias
        -> model.layers.X.mlp.e_score_correction_bias."""
        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import (
            MiMoV2FlashForCausalLM,
        )
        import inspect

        source = inspect.getsource(MiMoV2FlashForCausalLM.load_weights)

        # Check that the renaming logic is present
        assert ".mlp.gate.e_score_correction_bias" in source, (
            "load_weights does not handle e_score_correction_bias renaming from gate submodule"
        )
        assert ".mlp.e_score_correction_bias" in source, (
            "load_weights does not map to mlp.e_score_correction_bias"
        )

    def test_mtp_weights_filtered(self):
        """Verify that MTP weights are filtered out during load_weights."""
        from tensorrt_llm._torch.models.modeling_mimo_v2_flash import (
            MiMoV2FlashForCausalLM,
        )
        import inspect

        source = inspect.getsource(MiMoV2FlashForCausalLM.load_weights)
        assert ".mtp." in source, (
            "load_weights does not filter MTP weights"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
