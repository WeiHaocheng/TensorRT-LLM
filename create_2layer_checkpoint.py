#!/usr/bin/env python3
"""Create a 2-layer MiMo-V2-Flash checkpoint for memory-efficient testing.

Takes original layers 0 and 1:
- Layer 0: full attention + dense MLP (pattern=0, moe_freq=0)
- Layer 1: SWA + MoE (pattern=1, moe_freq=1)

This tests both attention types (full/SWA) and both MLP types (dense/MoE).

Also handles missing/empty embedding file by creating a random one.
"""

import json
import os
import shutil
import sys

import torch
import safetensors.torch as st

SRC_DIR = "/home/scratch.fredw_sw/MiMo-V2-Flash/MiMo-V2-Flash"
DST_DIR = "/tmp/mimo_v2_flash_2layer"
ORIG_LAYERS = [0, 1]  # original layer indices to keep


def main():
    # Clean and create destination
    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR, exist_ok=True)

    # 1. Load and modify config.json
    with open(os.path.join(SRC_DIR, "config.json")) as f:
        config = json.load(f)

    # Original patterns for layers 0 and 1
    # Layer 0: hybrid_layer_pattern=0, moe_layer_freq=0 -> full attn + dense MLP
    # Layer 1: hybrid_layer_pattern=1, moe_layer_freq=1 -> SWA + MoE
    new_hybrid = [config["hybrid_layer_pattern"][i] for i in ORIG_LAYERS]
    new_moe_freq = [config["moe_layer_freq"][i] for i in ORIG_LAYERS]

    config["num_hidden_layers"] = 2
    config["hybrid_layer_pattern"] = new_hybrid  # [0, 1]
    config["moe_layer_freq"] = new_moe_freq      # [0, 1]

    # Update ignored_layers in quantization_config for 2 layers only
    config["quantization_config"]["ignored_layers"] = [
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.self_attn.o_proj",
        "model.decoder.self_attn.o_proj",
    ]

    with open(os.path.join(DST_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 2. Build new safetensors index
    with open(os.path.join(SRC_DIR, "model.safetensors.index.json")) as f:
        orig_index = json.load(f)

    new_weight_map = {}

    # Add layer 0 weights (unchanged, same layer index)
    for key, fname in orig_index["weight_map"].items():
        if key.startswith("model.layers.0."):
            new_weight_map[key] = fname

    # Add layer 1 weights (unchanged, same layer index)
    for key, fname in orig_index["weight_map"].items():
        if key.startswith("model.layers.1."):
            new_weight_map[key] = fname

    # Add global weights
    for key, fname in orig_index["weight_map"].items():
        if not key.startswith("model.layers.") and ".mtp." not in key:
            new_weight_map[key] = fname

    new_index = {
        "metadata": {"total_size": 0},
        "weight_map": dict(sorted(new_weight_map.items()))
    }
    with open(os.path.join(DST_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # 3. Symlink all required safetensors files
    required_files = set(new_weight_map.values())
    for fname in required_files:
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)
        if os.path.exists(dst_path):
            continue
        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            os.symlink(src_path, dst_path)
            print(f"Symlinked: {fname} ({os.path.getsize(src_path)} bytes)")
        elif os.path.exists(src_path) and os.path.getsize(src_path) == 0:
            print(f"WARNING: {fname} is empty (0 bytes), will create dummy")
        else:
            print(f"WARNING: {fname} not found")

    # 4. Handle empty embedding file - create random embeddings
    emb_dst = os.path.join(DST_DIR, "model_embedding.safetensors")
    if not os.path.exists(emb_dst) or os.path.getsize(emb_dst) == 0:
        print("Creating dummy embedding weights (model_embedding.safetensors)...")
        vocab_size = config["vocab_size"]    # 152576
        hidden_size = config["hidden_size"]  # 4096
        # Create random bfloat16 embedding
        emb_weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
        st.save_file({"model.embed_tokens.weight": emb_weight}, emb_dst)
        print(f"  Created: {emb_dst} ({os.path.getsize(emb_dst)} bytes)")

    # 5. Check model_final.safetensors for lm_head and norm
    final_dst = os.path.join(DST_DIR, "model_final.safetensors")
    if not os.path.exists(final_dst) or os.path.getsize(final_dst) == 0:
        print("Creating dummy final weights (model_final.safetensors)...")
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        final_tensors = {
            "lm_head.weight": torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        }
        st.save_file(final_tensors, final_dst)
        print(f"  Created: {final_dst} ({os.path.getsize(final_dst)} bytes)")

    # 6. Copy other needed files (tokenizer, python modules, etc.)
    for fname in os.listdir(SRC_DIR):
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)
        if os.path.exists(dst_path):
            continue
        if os.path.isfile(src_path) and (
            fname.endswith(".py") or
            fname in ("tokenizer.model", "tokenizer_config.json",
                      "special_tokens_map.json", "tokenizer.json") or
            "tokenizer" in fname.lower()
        ):
            os.symlink(src_path, dst_path)
            print(f"Symlinked: {fname}")

    print(f"\n2-layer checkpoint created at: {DST_DIR}")
    print(f"Config: num_hidden_layers=2")
    print(f"  hybrid_layer_pattern={new_hybrid}")
    print(f"  moe_layer_freq={new_moe_freq}")
    print(f"\nLayer 0: Full Attention + Dense MLP (original layer 0)")
    print(f"Layer 1: SWA + MoE (original layer 1)")
    print(f"\nNOTE: Embedding weights are dummy/random - output text will be meaningless.")
    print(f"       This test validates model construction, weight loading, and forward pass.")


if __name__ == "__main__":
    main()
