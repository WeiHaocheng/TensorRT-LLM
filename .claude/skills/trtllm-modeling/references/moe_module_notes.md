# MoE Module Notes

## MoE-Specific Verification Checks

When verifying MoE models, check the following:

- `weight_loading_mode` matches `plan.md` exactly (should be `FUSED_GATE_UP_PROJ` unless plan says otherwise)
- `create_moe` parameters (`bias`, `swiglu_alpha`/`swiglu_beta`/`swiglu_limit`, `reduce_results`) match `plan.md`
- `_transform_weights` produces keys matching the chosen loading mode's expected format (see "FUSED_GATE_UP_PROJ Weight Format" table below)

## MoE Analysis Checklist

When analyzing an MoE model for porting, document the following:

1. **MoE architecture**: `num_experts`, `experts_per_token`, routing method (top-k renormalize, etc.), whether the gate/router has bias.
2. **Gate/Router mapping**: How the HuggingFace router weight/bias names map to the TRT-LLM `Gate` module. Common mapping: `mlp.router` → `mlp.gate`. Ensure the router weight name is correctly mapped in `_transform_weights` so that TRT-LLM's `Gate` module loads it.
3. **MoE bias**: Whether the expert MLPs use bias terms (`bias=True` in `create_moe`). Check the HuggingFace source for `bias` arguments in expert linear layers.
4. **Custom activation parameters source**: Document where SwiGLU `alpha`/`beta`/`limit` values come from — whether they are read from a config field (e.g., `config.swiglu_alpha`) or hardcoded in the HuggingFace modeling source. This is important for correctly initializing `create_moe`.

## MoE Model Implementation Guide

When implementing MoE (Mixture of Experts) models, follow these rules:

### weight_loading_mode Selection

**Always use `MoEWeightLoadingMode.FUSED_GATE_UP_PROJ`** unless `plan.md` explicitly specifies otherwise. This mode has better performance than VANILLA because the MoE backend handles weight splitting internally, avoiding per-expert tensor creation overhead.

- `FUSED_GATE_UP_PROJ`: Weights are stored as stacked tensors indexed by expert_id. The MoE backend internally splits gate/up weights via `.chunk(2, dim=0)`. Expected key names: `gate_up_proj`, `down_proj`, `gate_up_proj.bias`, `down_proj.bias`, `gate_up_proj_weight_scale`, `down_proj_weight_scale`.
- `VANILLA`: Weights are stored as per-expert individual tensors with keys like `{expert_id}.w1.weight`, `{expert_id}.w3.weight`, `{expert_id}.w2.weight`. Only use this when FUSED_GATE_UP_PROJ cannot work (e.g., expert weights are not stackable).

### FUSED_GATE_UP_PROJ Weight Format

The FUSED_GATE_UP_PROJ loading code expects weights in **transposed** format. For each weight type, the per-expert tensor shape and the code's processing are:

| Key | Per-expert shape stored | Code processing | Result |
|-----|----------------------|-----------------|--------|
| `gate_up_proj[e]` | `[packed_hidden, 2*inter]` | `.transpose(0,1).chunk(2, dim=0)` | w1 `[inter, packed_hidden]`, w3 `[inter, packed_hidden]` |
| `down_proj[e]` | `[packed_inter, hidden]` | `.transpose(0,1)` | `[hidden, packed_inter]` |
| `gate_up_proj_weight_scale[e]` | `[num_blocks, 2*inter]` | `.transpose(0,1).chunk(2, dim=0)` | w1_scale `[inter, num_blocks]`, w3_scale `[inter, num_blocks]` |
| `down_proj_weight_scale[e]` | `[num_blocks, hidden]` | `.transpose(0,1)` | `[hidden, num_blocks]` |
| `gate_up_proj.bias[e]` | `[2*inter]` | `.chunk(2, dim=0)` | w1_bias `[inter]`, w3_bias `[inter]` |
| `down_proj.bias[e]` | `[hidden]` | (no processing) | `[hidden]` |

**CRITICAL**: If the HuggingFace checkpoint uses an interleaved layout (e.g., alternating gate/up rows), `_transform_weights` must de-interleave and re-concatenate into `[gate_rows; up_rows]` (contiguous halves) before storing, then transpose to match the expected format above.

### SwiGLU Parameters Device Placement

When passing `swiglu_alpha`, `swiglu_beta`, `swiglu_limit` tensors to `create_moe`, always create them on CUDA: `torch.full(..., device='cuda')`. These tensors are stored as plain attributes (not `nn.Parameter`), so they won't be moved automatically by `.cuda()`.

## MXFP4 Quantized Weights with FUSED_GATE_UP_PROJ

MXFP4 quantized expert weights **CAN and SHOULD** use `FUSED_GATE_UP_PROJ` mode. Do not fall back to VANILLA just because weights are quantized.

### MXFP4 Block Format

MXFP4 checkpoints store expert weights as blocks + scales:
- **Blocks**: `[E, out_dim, num_blocks, block_size]` dtype `uint8` — each block of `block_size` bytes contains `block_size*2` FP4 values (2 per byte)
- **Scales**: `[E, out_dim, num_blocks]` dtype `uint8` — one E8M0 scale per block
- **Bias**: `[E, out_dim]` dtype `bfloat16` — not quantized

where `num_blocks = in_dim / group_size` (typically group_size=32) and `block_size = group_size / 2`.

### Transform Strategy: Pre-Transpose for FUSED_GATE_UP_PROJ Protocol

The FUSED_GATE_UP_PROJ loader applies `.transpose(0,1)` to blocks and scales before `.chunk(2, dim=0)`. To make MXFP4 data compatible:

1. **Reshape** blocks from `[E, out_dim, num_blocks, block_size]` → `[E, out_dim, num_blocks * block_size]` (packed 2D)
2. **De-interleave** if needed (see below)
3. **Concatenate** gate and up halves as `[w1=up, w3=gate]` along dim 1
4. **Pre-transpose** the result: `.transpose(1, 2)` — this stores as `[E, packed_in_dim, 2*inter]`
5. The loader's `.transpose(0,1)` on per-expert tensor `[packed_in_dim, 2*inter]` recovers `[2*inter, packed_in_dim]`, and `.chunk(2, dim=0)` correctly splits w1/w3

For **scales**, same approach: concatenate → `.transpose(1, 2)` → `[E, num_blocks, 2*inter]`.

For **bias**, NO transpose — the loader only does `.chunk(2, dim=0)` on bias.

### De-Interleaving Gate/Up Rows

Some models (e.g., GptOss) store gate and up rows **interleaved** in the fused gate_up weight:
- Even rows (0, 2, 4, ...) = gate
- Odd rows (1, 3, 5, ...) = up

De-interleave before fusing:
```python
gate = value[:, 0::2, ...]   # gate rows
up = value[:, 1::2, ...]     # up rows
fused = torch.cat([up, gate], dim=1)  # [w1=up, w3=gate] order for FUSED_GATE_UP_PROJ
```

The interleaving is always along the **output dimension** (dim 1 in the `[E, out_dim, ...]` layout). Apply the same stride-2 indexing to blocks, scales, and bias consistently.

### Complete _transform_weights Pattern for MXFP4 + FUSED_GATE_UP_PROJ

```python
def _fuse_gate_up(self, key, value, transformed, target_suffix):
    prefix = key.rsplit('.mlp.experts.', 1)[0] + '.mlp.experts'
    new_key = f'{prefix}.{target_suffix}'

    if target_suffix == 'gate_up_proj':
        # blocks: [E, 2*inter, num_blocks, block_size]
        gate = value[:, 0::2, :, :]  # de-interleave if needed
        up = value[:, 1::2, :, :]
        gate = gate.reshape(gate.shape[0], gate.shape[1], -1)  # pack blocks
        up = up.reshape(up.shape[0], up.shape[1], -1)
        fused = torch.cat([up, gate], dim=1)  # [w1=up, w3=gate]
        transformed[new_key] = fused.transpose(1, 2).contiguous()  # pre-transpose
    elif target_suffix == 'gate_up_proj_weight_scale':
        # scales: [E, 2*inter, num_blocks]
        gate = value[:, 0::2, :]
        up = value[:, 1::2, :]
        fused = torch.cat([up, gate], dim=1)
        transformed[new_key] = fused.transpose(1, 2).contiguous()  # pre-transpose
    else:
        # bias: [E, 2*inter] — NO transpose
        gate = value[:, 0::2]
        up = value[:, 1::2]
        transformed[new_key] = torch.cat([up, gate], dim=1)

def _fuse_down(self, key, value, transformed, target_suffix):
    prefix = key.rsplit('.mlp.experts.', 1)[0] + '.mlp.experts'
    new_key = f'{prefix}.{target_suffix}'

    if target_suffix == 'down_proj':
        # blocks: [E, out_dim, num_blocks, block_size] → pack + pre-transpose
        value = value.reshape(value.shape[0], value.shape[1], -1)
        transformed[new_key] = value.transpose(1, 2).contiguous()
    elif target_suffix == 'down_proj_weight_scale':
        # scales: [E, out_dim, num_blocks] → pre-transpose
        transformed[new_key] = value.transpose(1, 2).contiguous()
    else:
        # bias: [E, out_dim] — no transformation
        transformed[new_key] = value
```

### Checkpoint Key → Transformed Key Mapping

| Checkpoint key suffix | Target key suffix | Transform |
|----------------------|-------------------|-----------|
| `gate_up_proj_blocks` | `gate_up_proj` | de-interleave + pack blocks + cat [up, gate] + transpose |
| `gate_up_proj_scales` | `gate_up_proj_weight_scale` | de-interleave + cat [up, gate] + transpose |
| `gate_up_proj_bias` | `gate_up_proj.bias` | de-interleave + cat [up, gate] (no transpose) |
| `down_proj_blocks` | `down_proj` | pack blocks + transpose |
| `down_proj_scales` | `down_proj_weight_scale` | transpose |
| `down_proj_bias` | `down_proj.bias` | passthrough |
