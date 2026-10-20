# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for `catalog._launch.Launch`, the catalog's straight-to-the-compiled-kernel launches.

Its claim is that a call launches the compiled kernel `kernel[grid](...)` would launch, with the
same arguments. The way that claim fails is a cached kernel reused for a call it was not compiled
for: Triton specializes a kernel on its arguments -- an int equal to 1 becomes a constant, a
pointer's 16-byte alignment and an int's divisibility by 16 change the code -- so a kernel compiled
for a contiguous stride reads a strided tensor wrongly, and one compiled for an aligned pointer
misreads an offset view. So the calls below alternate between those specializations, and each is
checked for the kernel it ran, against the plain launch's output and against torch.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton import knobs

from tensorrt_llm._torch._experimental.modeling_v2.catalog._launch import Launch

assert torch.cuda.is_available(), "Launch requires a CUDA device"


@triton.jit
def _axpy_kernel(
    x_ptr, y_ptr, out_ptr, n, stride_x, alpha, BLOCK: tl.constexpr, SQUARE: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * stride_x, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    r = alpha * x + y
    if SQUARE:
        r = r * r
    tl.store(out_ptr + offs, r, mask=mask)


_AXPY = Launch(_axpy_kernel)

_CASES = [
    # (n, stride, offset, square): each case differs from the one before in one specialization.
    (4096, 1, 0, False),  # contiguous x (stride 1 is a constant), aligned, n % 16 == 0
    (4096, 2, 0, False),  # stride 2: a stride-1 kernel would read the wrong elements
    (4096, 1, 1, False),  # x one element into its buffer: not 16-byte aligned
    (4096, 1, 0, False),  # the first case again: served from the cache
    (4001, 1, 0, False),  # n not a multiple of 16
    (4096, 1, 0, True),  # another constexpr value
    (4096, 3, 1, True),  # stride 3 and misaligned together
    (1, 1, 0, False),  # n == 1, itself a constant
]


def _case(n, stride, offset, square, seed):
    """(args, kwargs, grid): x read with `stride` from `offset` elements into its buffer, y and
    out dense."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn(offset + n * stride + 16, generator=g, device="cuda")
    x = buf[offset:]
    y = torch.randn(n, generator=g, device="cuda")
    out = torch.empty_like(y)
    kwargs = dict(n=n, stride_x=stride, alpha=0.75, BLOCK=256, SQUARE=square, num_warps=4)
    return (x, y, out), kwargs, (triton.cdiv(n, 256),)


def _torch(args, kwargs):
    x, y, _ = args
    n, s = kwargs["n"], kwargs["stride_x"]
    r = kwargs["alpha"] * x[: n * s : s] + y
    return r * r if kwargs["SQUARE"] else r


def _cached(launcher, args, kwargs):
    """The compiled kernel `launcher` holds for these arguments, by its own key."""
    binder = launcher.kernel.device_caches[torch.cuda.current_device()][4]
    bind = dict(
        kwargs,
        debug=knobs.runtime.debug,
        instrumentation_mode=knobs.compilation.instrumentation_mode,
    )
    _, specialization, options = binder(*args, **bind)
    cached = launcher._compiled.get(
        (torch.cuda.current_device(), *specialization, *options.items())
    )
    return None if cached is None else cached[0]


def test_every_call_runs_the_kernel_run_would_run() -> None:
    """Each call's cached kernel is the very one the JIT's own launch picks for its arguments,
    and its output is the plain launch's, bit for bit, and torch's. A fresh launcher sees the
    cases in order, twice, so every case after a different one exercises the cache's key."""
    launcher = Launch(_axpy_kernel)
    for i, case in enumerate(_CASES * 2):
        args, kwargs, grid = _case(*case, seed=i)
        launcher[grid](*args, **kwargs)
        got = args[2].clone()
        assert _cached(launcher, args, kwargs) is _axpy_kernel.run(
            *args, grid=grid, warmup=True, **kwargs
        ), case
        _axpy_kernel[grid](*args, **kwargs)
        assert torch.equal(got, args[2]), case
        torch.testing.assert_close(got, _torch(args, kwargs), rtol=1e-6, atol=1e-6)


def test_each_specialization_compiles_once() -> None:
    """The second pass over the cases compiles nothing: every call is served from the cache.
    And the cases do reach different kernels -- stride 1 and stride 2 do not share one."""
    launcher = Launch(_axpy_kernel)
    for i, case in enumerate(_CASES):
        args, kwargs, grid = _case(*case, seed=i)
        launcher[grid](*args, **kwargs)
    size = len(launcher._compiled)
    for i, case in enumerate(_CASES):
        args, kwargs, grid = _case(*case, seed=i)
        launcher[grid](*args, **kwargs)
    assert len(launcher._compiled) == size
    one, two = (_case(*_CASES[j], seed=j)[:2] for j in (0, 1))
    assert _cached(launcher, *one) is not _cached(launcher, *two)


def test_installed_launch_hooks_still_run() -> None:
    """With a launch hook installed a call goes through the JIT's own launch, which calls it;
    the cached path would skip it."""
    args, kwargs, grid = _case(*_CASES[0], seed=0)
    _AXPY[grid](*args, **kwargs)  # cached
    seen = []

    def hook(metadata):
        seen.append(metadata)

    knobs.runtime.launch_enter_hook.add(hook)
    try:
        _AXPY[grid](*args, **kwargs)
    finally:
        knobs.runtime.launch_enter_hook.remove(hook)
    assert seen, "the launch hook did not run"
    torch.testing.assert_close(args[2], _torch(args, kwargs), rtol=1e-6, atol=1e-6)


def test_a_cuda_graph_replays_a_cached_launch() -> None:
    """Captured once its kernel is cached, a launch replays on new data in the captured buffers."""
    args, kwargs, grid = _case(4096, 2, 0, True, seed=3)
    _AXPY[grid](*args, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _AXPY[grid](*args, **kwargs)
    args[0].normal_()
    args[1].normal_()
    graph.replay()
    torch.cuda.synchronize()
    got = args[2].clone()
    _axpy_kernel[grid](*args, **kwargs)
    assert torch.equal(got, args[2])


def test_a_wrapped_kernel_is_refused() -> None:
    """A heuristics or autotune wrapper adds arguments the launcher would not compute."""
    wrapped = triton.heuristics({"SQUARE": lambda args: False})(_axpy_kernel)
    with pytest.raises(TypeError, match="bare triton.jit"):
        Launch(wrapped)
