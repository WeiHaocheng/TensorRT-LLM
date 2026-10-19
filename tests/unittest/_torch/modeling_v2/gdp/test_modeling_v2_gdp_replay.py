# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdp_replay catalog entry, at nemotron4-nano-preview's tp1 and tp4-rank shapes."""

import math
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gdp.gdp_replay import gdp_replay as op
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

__extra_import_path__ = [".."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gdp_replay requires a CUDA device"

#: A v2 (`MambaHybridCacheManagerV2`) slot holds the ssm states of all 32 GDP
#: layers back to back; the view under test is this one's.
_V2_LAYERS, _V2_LAYER = 32, 13
#: The head counts the cells certify: tp1's 24 and a tp4 rank's 6.
HEADS = sorted({c.spec["heads"] for c in op.CELLS}, reverse=True)


def _conv_dim(spec) -> int:
    """The GDP conv's channels, [V0..V2 | K0..K2 | Q]: 16896 at 24 heads, 4224 at 6."""
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    return nh * h * dv + (nh + 1) * h * dk


def _conv_state_floats(spec) -> int:
    """What a v1 slot carries after a GDP layer's ssm state: its bf16 conv state, in floats."""
    return _conv_dim(spec) * 3 * 2 // 4


def _rows_view(proj: torch.Tensor, start: int, inner: tuple, tokens: int) -> torch.Tensor:
    """Columns ``start:`` of the in_proj-shaped ``proj`` as ``[rows / tokens, tokens, *inner]``, a token's
    row stride the in_proj width -- the views the targets hand the GDP entries."""
    strides, size = [], 1
    for n in reversed(inner):
        strides.insert(0, size)
        size *= n
    rows, row = proj.shape[0], proj.stride(0)
    shape, stride = (rows // tokens, tokens, *inner), (tokens * row, row, *strides)
    return proj.as_strided(shape, stride, proj.storage_offset() + start)


def _num_slots(spec) -> int:
    """v1: two slots per row plus three, so the rows' slots are scattered through a pool they do not
    fill. v2: one per row plus three -- a v2 slot is 25 MB."""
    return spec["batch"] + 3 if spec["pool"] == "v2" else 2 * spec["batch"] + 3


def _raw_pool(spec, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """The memory a GDP layer's pool view lives in, every float a non-zero state the op must leave
    alone outside its rows. v1: a guard block, then per slot this layer's ssm state and its conv
    state. v2: per slot the 32 layers' ssm states, the view taking layer 13's."""
    ssm = spec["heads"] * spec["value_dim"] * spec["key_dim"]
    if spec["pool"] == "v2":
        shape = (num_slots, _V2_LAYERS * ssm)
    else:
        shape = (num_slots + 1, ssm + _conv_state_floats(spec))
    return torch.randn(shape, generator=g, device="cuda") * 0.5


def _pool_view(raw: torch.Tensor, spec) -> torch.Tensor:
    h, dv, dk = spec["heads"], spec["value_dim"], spec["key_dim"]
    ssm = h * dv * dk
    if spec["pool"] == "v2":
        return raw[:, _V2_LAYER * ssm : (_V2_LAYER + 1) * ssm].view(-1, h, dv, dk)
    return raw[1:, :ssm].view(-1, h, dv, dk)


def _decay(shape, spec, g: torch.Generator) -> torch.Tensor:
    """HF's fp32 decay over the checkpoint's parameter ranges (exp(A_log) in [0.0061, 18.3], dt_bias
    in [-7.03, -1.94] across its 32 GDP layers): some heads forget within a token, others barely."""
    h = spec["heads"]
    a_log = torch.empty(h, device="cuda").uniform_(math.log(0.0061), math.log(18.3), generator=g)
    dt_bias = torch.empty(h, device="cuda").uniform_(-7.03, -1.94, generator=g).to(torch.bfloat16)
    a = torch.randn(*shape, h, generator=g, device="cuda").to(torch.bfloat16)
    return -a_log.exp() * F.softplus(a.float() + dt_bias.float())


def _inputs(spec, batch: int, g: torch.Generator):
    """(q, k, v, g, beta) as a target holds them after the conv: [B, T, ...] views of an in_proj
    output (NaN outside the conv columns, which a stray read would carry into the result)."""
    h, nh, dk, dv, tokens = (
        spec[n] for n in ("heads", "householder", "key_dim", "value_dim", "tokens")
    )
    conv_dim = _conv_dim(spec)
    mixed = F.silu(torch.randn(batch * tokens, conv_dim, generator=g, device="cuda"))
    proj = torch.full(
        (batch * tokens, spec["proj_width"]), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    lo = spec["conv_offset"]
    proj[:, lo : lo + conv_dim].copy_(mixed)
    v = _rows_view(proj, lo, (nh, h, dv), tokens)
    k = _rows_view(proj, lo + nh * h * dv, (nh, h, dk), tokens)
    q = _rows_view(proj, lo + nh * h * (dv + dk), (h, dk), tokens)
    b = torch.randn(batch, tokens, nh * h, generator=g, device="cuda").to(torch.bfloat16)
    beta = b.sigmoid().view(batch, tokens, nh, h)
    return q, k, v, _decay((batch, tokens), spec, g), beta


def _caches(spec, num_slots: int, g: torch.Generator):
    """(pending, parity, cache_k, cache_u, cache_g) of a pool whose slots each hold a verify's drafts in
    a random buffer, 0..D of them pending; None under MTP off. The keys are unit vectors and the u's of
    the size a verify computes, as the op caches them."""
    drafts = spec["drafts"]
    if drafts == 0:
        return None, None, None, None, None
    h, nh, dk, dv = spec["heads"], spec["householder"], spec["key_dim"], spec["value_dim"]
    cache_k = F.normalize(
        torch.randn(num_slots, 2, drafts, nh, h, dk, generator=g, device="cuda"), dim=-1
    )
    cache_u = torch.randn(num_slots, 2, drafts, nh, h, dv, generator=g, device="cuda") * 0.5
    cache_g = _decay((num_slots, 2, drafts), spec, g)
    pending = torch.randint(0, drafts + 1, (num_slots,), generator=g, device="cuda").to(torch.int32)
    parity = torch.randint(0, 2, (num_slots,), generator=g, device="cuda").to(torch.int32)
    return pending, parity, cache_k, cache_u, cache_g


def _slots(batch: int, pad_rows: int, num_slots: int, g: torch.Generator) -> torch.Tensor:
    """Distinct random slots for the real rows, PAD_SLOT_ID for the padding rows at the end."""
    real = torch.randperm(num_slots, generator=g, device="cuda")[: batch - pad_rows]
    pad = torch.full((pad_rows,), PAD_SLOT_ID, device="cuda")
    return torch.cat([real, pad]).to(torch.int32)


def _build(spec, seed: int):
    """One cell: (args, the pool's raw memory)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_slots = _num_slots(spec)
    pool_raw = _raw_pool(spec, num_slots, g)
    q, k, v, decay, beta = _inputs(spec, spec["batch"], g)
    pending, parity, cache_k, cache_u, cache_g = _caches(spec, num_slots, g)
    args = dict(
        q=q,
        k=k,
        v=v,
        g=decay,
        beta=beta,
        state_pool=_pool_view(pool_raw, spec),
        slots=_slots(spec["batch"], spec["pad_rows"], num_slots, g),
        pending=pending,
        parity=parity,
        cache_k=cache_k,
        cache_u=cache_u,
        cache_g=cache_g,
        scale=spec["scale"],
    )
    return args, pool_raw


def _real_rows(slots: torch.Tensor) -> torch.Tensor:
    return (slots >= 0).nonzero().squeeze(-1)


# Bands. Kernel and reference both compute in fp32 and differ only in the order
# the reductions over K and the rank-one updates round. Measured with
# `measure_bands` (32 seeds of every cell), as (element ulp of the row scale,
# relative-RMS ulp); each band is about twice its worst case. The mistakes the
# bands exist to catch sit orders of magnitude out: a reference whose fp32
# draft fold ran in TF32 read 4816 / 317 on the state.
#: out: one bf16 rounding of an fp32 value on each side, so up to one bf16 step,
#: 2 ulp of the row scale (`gdp_verify`'s band); measured 1.90 / 0.085.
OUT_ELEMENT_ULP, OUT_RMS_ULP = 2.0, 0.25
#: The committed state, fp32 ulp per sqrt(drafts replayed + 1 token): independent
#: roundings that add like a random walk; measured 4.01 / 0.87 per sqrt(depth).
STATE_ELEMENT_ULP, STATE_RMS_ULP = 8.0, 1.5
#: A cached u: one Householder step's reduction over K in another order, after
#: up to two drafts and a token of state drift; measured 6.06 / 1.66.
U_ELEMENT_ULP, U_RMS_ULP = 12.0, 3.0
#: A cached key: the l2 norm summed in another order; measured 2.04 / 0.23.
K_ELEMENT_ULP, K_RMS_ULP = 4.0, 0.5


def _compare_out(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert_within_ulp(out, ref, element_ulp=OUT_ELEMENT_ULP, rms_ulp=OUT_RMS_ULP)


def _compare_states(states: torch.Tensor, ref: torch.Tensor, depth: torch.Tensor) -> None:
    """[n, H, V, K] committed states, each banded by sqrt(its depth): the drafts it replayed plus the
    first token, independent roundings that add like a random walk."""
    for d in sorted(set(depth.tolist())):
        sel = depth == d
        assert_within_ulp(
            states[sel],
            ref[sel],
            element_ulp=STATE_ELEMENT_ULP * math.sqrt(d),
            rms_ulp=STATE_RMS_ULP * math.sqrt(d),
            ulp=ULP[torch.float32],
        )


def _check(out, ref, args, spec, pool_raw, pool_before, caches_before) -> None:
    """The outputs, the committed states of the real slots and the drafts they cached against
    `reference`; every other float of the pool's memory and of the caches as it was."""
    ref_out, ref_pool, ref_k, ref_u, ref_g = ref
    _compare_out(out, ref_out)
    slots = args["slots"]
    real_slots = slots[slots >= 0].long()
    pool = args["state_pool"]
    depth = torch.ones_like(real_slots)
    if args["pending"] is not None:
        depth = depth + args["pending"][real_slots].long()
    _compare_states(pool[real_slots], ref_pool[real_slots], depth)
    pool_owned = torch.zeros(pool.shape[0], dtype=torch.bool, device=pool.device)
    pool_owned[real_slots] = True
    view_before = _pool_view(pool_before, spec)
    assert torch.equal(pool[~pool_owned], view_before[~pool_owned]), (
        "the op wrote another slot's state"
    )
    raw_owned = torch.zeros_like(pool_raw, dtype=torch.bool)
    _pool_view(raw_owned, spec)[real_slots] = True
    assert torch.equal(pool_raw[~raw_owned], pool_before[~raw_owned]), (
        "the op wrote outside the pool view's slots"
    )
    if args["cache_k"] is None:
        return
    tokens = args["q"].shape[1]
    cached = torch.zeros(args["cache_k"].shape[:3], dtype=torch.bool, device=pool.device)
    if tokens > 1:
        write = 1 - args["parity"][real_slots].long()
        for t in range(1, tokens):
            cached[real_slots, write, t - 1] = True
        assert torch.equal(args["cache_g"][cached], ref_g[cached]), "a cached decay differs from g"
        assert_within_ulp(
            args["cache_u"][cached],
            ref_u[cached],
            element_ulp=U_ELEMENT_ULP,
            rms_ulp=U_RMS_ULP,
            ulp=ULP[torch.float32],
        )
        assert_within_ulp(
            args["cache_k"][cached],
            ref_k[cached],
            element_ulp=K_ELEMENT_ULP,
            rms_ulp=K_RMS_ULP,
            ulp=ULP[torch.float32],
        )
    for name, before in zip(("cache_k", "cache_u", "cache_g"), caches_before):
        assert torch.equal(args[name][~cached], before[~cached]), (
            f"the op wrote outside {name}'s new drafts"
        )


def _cell(spec, seed: int):
    """(args, the pool's raw memory, a copy of it, copies of the caches) for one cell and seed."""
    args, pool_raw = _build(spec, seed)
    caches = ("cache_k", "cache_u", "cache_g")
    caches_before = tuple(args[n].clone() for n in caches) if args["cache_k"] is not None else ()
    return args, pool_raw, pool_raw.clone(), caches_before


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args, pool_raw, pool_before, caches_before = _cell(
        spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    pending_before = (
        None if args["pending"] is None else (args["pending"].clone(), args["parity"].clone())
    )
    ref = op.reference(**args)
    out = op(**args)
    assert out.shape == (spec["batch"], spec["tokens"], spec["heads"], spec["value_dim"])
    assert out.dtype == args["q"].dtype
    _check(out, ref, args, spec, pool_raw, pool_before, caches_before)
    if pending_before is not None:
        assert torch.equal(args["pending"], pending_before[0]) and torch.equal(
            args["parity"], pending_before[1]
        ), "the op wrote pending or parity, which only the caller's commit does"


# ------------------------------------------------------------------------------------------------ chain
def _l2n64(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _fp64_tokens(state, q, k, v, g, beta, scale, tokens):
    """HF's recurrence in float64 over `tokens` tokens from `state` [n, H, K, V]: (outs [n, tokens,
    H, V], the state after each token [n, tokens, H, K, V])."""
    outs, states = [], []
    s = state.clone()
    for t in range(tokens):
        s = s * g[:, t].double().exp()[..., None, None]
        for j in range(v.shape[2]):
            kj = _l2n64(k[:, t, j])
            u = (v[:, t, j].double() - (s * kj[..., None]).sum(-2)) * beta[:, t, j].double()[
                ..., None
            ]
            s = s + kj[..., None] * u[..., None, :]
        outs.append((s * (_l2n64(q[:, t]) * scale)[..., None]).sum(-2))
        states.append(s.clone())
    return torch.stack(outs, 1), torch.stack(states, 1)


def _folded(args, slots) -> torch.Tensor:
    """The state each real slot stands for, [n, H, K, V] float64: the pool's, with its pending drafts
    folded in (what the next step will start from)."""
    real_slots = slots[slots >= 0].long()
    s = args["state_pool"][real_slots].double().transpose(-1, -2)
    if args["cache_k"] is None:
        return s
    pend = args["pending"][real_slots].long()
    par = args["parity"][real_slots].long()
    for d in range(args["cache_k"].shape[2]):
        keep = (d < pend)[:, None, None, None]
        upd = s * args["cache_g"][real_slots, par, d].double().exp()[
            ..., None, None
        ] + torch.einsum(
            "njhk,njhv->nhkv",
            args["cache_k"][real_slots, par, d].double(),
            args["cache_u"][real_slots, par, d].double(),
        )
        s = torch.where(keep, upd, s)
    return s


@pytest.mark.parametrize("heads", HEADS)
def test_steps_with_acceptance_follow_the_accepted_tokens(heads: int) -> None:
    """The design end to end: verify steps whose drafts the sampler partly accepts, the caller's commit
    (pending <- accepted - 1, parity flipped) between them, a decode step without drafts in the middle
    (it folds the last verify's drafts in and caches none), against HF's recurrence in float64 over
    exactly the accepted tokens. Every step's outputs for the tokens it ran, and the state each slot
    stands for at the end."""
    spec = next(
        c.spec
        for c in op.CELLS
        if c.spec["heads"] == heads and c.spec["tokens"] == 3 and c.spec["batch"] == 64
    )
    gen = torch.Generator(device="cuda").manual_seed(2026)
    args, _, _, _ = _cell(spec, seed=7)
    args["pending"].zero_()  # every slot fresh from a prefill
    slots = args["slots"]
    real = _real_rows(slots)
    real_slots = slots[real].long()
    ref_state = args["state_pool"][real_slots].double().transpose(-1, -2)
    worst_out = 0.0
    for step, tokens in enumerate((3, 3, 1, 3, 3, 2, 3)):
        sp = dict(spec, tokens=tokens)
        q, k, v, decay, beta = _inputs(sp, spec["batch"], gen)
        accepted = torch.randint(1, tokens + 1, (spec["batch"],), generator=gen, device="cuda")
        out = op(
            q,
            k,
            v,
            decay,
            beta,
            args["state_pool"],
            slots,
            args["pending"],
            args["parity"],
            args["cache_k"],
            args["cache_u"],
            args["cache_g"],
            spec["scale"],
        )
        ref_out, ref_states = _fp64_tokens(
            ref_state, q[real], k[real], v[real], decay[real], beta[real], spec["scale"], tokens
        )
        # Outputs within one bf16 rounding of float64 (plus the fp32 drift, far below it).
        _compare_out(out[real], ref_out.to(out.dtype))
        worst_out = max(worst_out, (out[real].double() - ref_out).abs().max().item())
        ref_state = ref_states[torch.arange(len(real), device="cuda"), accepted[real] - 1]
        # The caller's commit after acceptance.
        args["pending"][real_slots] = (accepted[real] - 1).to(torch.int32) if tokens > 1 else 0
        if tokens > 1:
            args["parity"][real_slots] = 1 - args["parity"][real_slots]
    assert_within_ulp(
        _folded(args, slots).float(),
        ref_state.float(),
        element_ulp=4 * STATE_ELEMENT_ULP,
        rms_ulp=4 * STATE_RMS_ULP,
        ulp=ULP[torch.float32],
    )


def test_cuda_graph_replays_with_the_commit_between() -> None:
    """One capture, three replays with the caller's commit between them: each replay reads whatever the
    input buffers, the slot list, pending and parity hold at replay time -- the same slots in another
    row order -- and continues from the drafts the previous one cached."""
    spec = next(
        c.spec
        for c in op.CELLS
        if c.spec["batch"] == 64 and c.spec["tokens"] == 3 and c.spec["heads"] == 24
    )
    args, pool_raw, _, _ = _cell(spec, seed=64)
    g = torch.Generator(device="cuda").manual_seed(65)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**args)  # compiles the kernel outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**args)

    n_real = spec["batch"] - spec["pad_rows"]
    real_slots = args["slots"][:n_real].clone()
    for _ in range(3):
        fresh = _inputs(spec, spec["batch"], g)
        for name, value in zip(("q", "k", "v", "g", "beta"), fresh):
            args[name].copy_(value)
        order = real_slots[torch.randperm(n_real, generator=g, device="cuda")]
        args["slots"][:n_real].copy_(order)
        pool_before = pool_raw.clone()
        caches_before = tuple(args[n].clone() for n in ("cache_k", "cache_u", "cache_g"))
        ref = op.reference(**args)
        graph.replay()
        _check(out, ref, args, spec, pool_raw, pool_before, caches_before)
        accepted = torch.randint(1, spec["tokens"] + 1, (n_real,), generator=g, device="cuda")
        args["pending"][order.long()] = (accepted - 1).to(torch.int32)
        args["parity"][order.long()] = 1 - args["parity"][order.long()]


def measure_bands(seeds: int = 32) -> dict:
    """Worst-case errors of the op against `reference` over every cell x `seeds`, in the units of the
    bands above: (element ulp, relative-RMS ulp). Run by hand; the band comments quote it."""

    def ulps(x, ref, ulp):
        x, ref = x.float().reshape(-1, x.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
        scale = ref.abs().amax(1, keepdim=True).clamp_min(1e-9)
        el = ((x - ref).abs() / scale).max().item() / ulp
        rms = (
            (x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-9)
        ).item() / ulp
        return el, rms

    worst: dict = {}

    def note(name, el, rms):
        prev = worst.get(name, (0.0, 0.0))
        worst[name] = (max(prev[0], el), max(prev[1], rms))

    for cell in op.CELLS:
        for seed in range(seeds):
            args, _, _, _ = _cell(cell.spec, seed)
            ref_out, ref_pool, ref_k, ref_u, _ = op.reference(**args)
            out = op(**args)
            note("out", *ulps(out, ref_out, ULP[torch.bfloat16]))
            slots = args["slots"]
            real_slots = slots[slots >= 0].long()
            depth = torch.ones_like(real_slots)
            if args["pending"] is not None:
                depth = depth + args["pending"][real_slots].long()
            for d in sorted(set(depth.tolist())):
                sel = depth == d
                el, rms = ulps(
                    args["state_pool"][real_slots][sel],
                    ref_pool[real_slots][sel],
                    ULP[torch.float32],
                )
                note("state per sqrt(depth)", el / math.sqrt(d), rms / math.sqrt(d))
            if args["cache_k"] is not None and cell.spec["tokens"] > 1:
                write = 1 - args["parity"][real_slots].long()
                for t in range(1, cell.spec["tokens"]):
                    note(
                        "u",
                        *ulps(
                            args["cache_u"][real_slots, write, t - 1],
                            ref_u[real_slots, write, t - 1],
                            ULP[torch.float32],
                        ),
                    )
                    note(
                        "k",
                        *ulps(
                            args["cache_k"][real_slots, write, t - 1],
                            ref_k[real_slots, write, t - 1],
                            ULP[torch.float32],
                        ),
                    )
    return worst


# ------------------------------------------------------------------------------------------------ guards
def _ok(**over):
    spec = dict(
        next(c.spec for c in op.CELLS if c.spec["tokens"] == 3 and c.spec["batch"] == 3), **over
    )
    args, _, _, _ = _cell(spec, seed=3)
    return args


def test_guard_refuses_a_bf16_pool() -> None:
    args = _ok()
    args["state_pool"] = args["state_pool"].to(torch.bfloat16)
    with pytest.raises(AssertionError, match="float32"), validating(op):
        op(**args)


def test_guard_refuses_a_step_with_drafts_and_no_caches() -> None:
    """Tokens 1..T-1 would update nothing anyone keeps."""
    args = _ok()
    for name in ("pending", "parity", "cache_k", "cache_u", "cache_g"):
        args[name] = None
    with pytest.raises(AssertionError, match="needs the draft caches"), validating(op):
        op(**args)


def test_guard_refuses_caches_shallower_than_the_drafts() -> None:
    args = _ok()
    for name in ("cache_k", "cache_u", "cache_g"):
        args[name] = args[name][:, :, :1].contiguous()
    args["pending"].clamp_(max=1)
    with pytest.raises(AssertionError, match="caches hold 1 drafts"), validating(op):
        op(**args)


def test_guard_refuses_more_pending_drafts_than_the_caches_hold() -> None:
    """The replay would read the next slot's lines."""
    args = _ok()
    used = args["slots"][args["slots"] >= 0].long()
    args["pending"][used[0]] = 3
    with pytest.raises(AssertionError, match="pending must be in"), validating(op):
        op(**args)


def test_guard_refuses_a_parity_outside_the_two_buffers() -> None:
    args = _ok()
    used = args["slots"][args["slots"] >= 0].long()
    args["parity"][used[0]] = 2
    with pytest.raises(AssertionError, match="parity must be 0 or 1"), validating(op):
        op(**args)


def test_guard_refuses_two_rows_on_one_slot() -> None:
    args = _ok()
    args["slots"][1] = args["slots"][0]
    with pytest.raises(AssertionError, match="share a slot"), validating(op):
        op(**args)


def test_guard_refuses_a_decay_that_grows_the_state() -> None:
    args = _ok()
    args["g"] = args["g"].abs()
    with pytest.raises(AssertionError, match="log decay"), validating(op):
        op(**args)


def test_guard_refuses_a_pre_sigmoid_beta() -> None:
    args = _ok()
    args["beta"] = (args["beta"].float() * 4 - 2).to(args["beta"].dtype)
    with pytest.raises(AssertionError, match="post-sigmoid"), validating(op):
        op(**args)
