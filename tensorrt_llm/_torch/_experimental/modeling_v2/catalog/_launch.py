# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton launches straight to the compiled kernel once its specialization is compiled.

`kernel[grid](...)` goes through `JITFunction.run` on every call: the knobs it reads from the
environment, the current device and stream, the kernel's binder, a cache key built by formatting
the options into a string, a check of every global the kernel reads, launch metadata built for
launch hooks whether or not any is installed, and the compiled kernel's launcher object, before
the C launch function runs. Measured here, the same launch repeated: 11-21 us of host per launch,
of which the binder is 1.4-4.3 us and the C launch a few. An eager step of a model whose entries
are Triton kernels issues hundreds of them, and the tp4 target's prompt step is host-bound.

`Launch(kernel)[grid](...)` takes the same arguments and launches the same compiled kernel. It
keeps the one step that decides which compiled kernel a call needs -- the kernel's own binder,
which derives the specialization (argument types, 16-byte alignment, divisibility by 16, the
constexpr values) exactly as `run` derives it -- and keys its cache on that specialization and
the launch options. A call whose key it has seen goes straight to that kernel's C launch
function with the binder's arguments, the arguments `run` passes (through the kernel's launcher
object when that object has anything of its own to add: scratch buffers, sanitizer state). Any
other call, and every call while a launch hook, a pre-run hook or a stages-inspection hook is
installed, goes through `run` itself, which compiles the kernel if it has to and returns it to
the cache.

So a call computes the same bits either way: the kernel and the arguments are the ones `run`
would launch. What the fast path skips is bookkeeping: hooks it would not call, metadata nobody
reads, and the check that the globals a kernel reads have not changed since it was compiled
(module constants here; `run` still makes it on the first call of every specialization).
"""

from __future__ import annotations

from typing import Any, Callable

from triton import knobs
from triton.compiler.compiler import CompiledKernel
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction


def _installed(hook) -> bool:
    """Whether a launch hook would run: a `HookChain` with any hook in it, or a hook assigned
    as a bare callable."""
    calls = getattr(hook, "calls", None)
    return bool(calls) if calls is not None else hook is not None


def _launcher(compiled: CompiledKernel) -> Callable[..., None]:
    """How a cached kernel is launched: `(grid_x, grid_y, grid_z, stream, bound_args)`.

    Straight into the backend's C launch function with the arguments the compiled kernel's
    launcher object passes it, when that object would pass nothing else -- no global or profile
    scratch to allocate, no sanitizer state; through the launcher object otherwise. Either way
    the C function receives what `run` would hand it, without launch metadata or hooks."""
    run = compiled.run
    launch = getattr(run, "launch", None)
    direct = (
        launch is not None
        and getattr(run, "global_scratch_size", None) == 0
        and getattr(run, "profile_scratch_size", None) == 0
        and getattr(run, "gsan_enabled", None) is False
    )
    if direct:
        head = (
            compiled.function,
            run.launch_cooperative_grid,
            run.launch_pdl,
            compiled.packed_metadata,
            None,
            None,
            None,
            None,
            None,
            run.arg_annotations,
            run.kernel_signature,
        )

        def go(gx, gy, gz, stream, bound):
            launch(gx, gy, gz, stream, *head, tuple(bound.values()))

        return go
    head = (compiled.function, compiled.packed_metadata, None, None, None)

    def go(gx, gy, gz, stream, bound):
        run(gx, gy, gz, stream, *head, *bound.values())

    return go


class Launch:
    """`Launch(kernel)[grid](*args, **kwargs)`: `kernel[grid](*args, **kwargs)`, the same compiled
    kernel and arguments, without `JITFunction.run` once the call's specialization is cached.

    `kernel` is the bare `triton.jit` function. A kernel under `triton.heuristics` or
    `triton.autotune` is launched through its `.fn` with the values those wrappers would add
    passed explicitly; this class does not evaluate them.
    """

    __slots__ = ("kernel", "_compiled")

    def __init__(self, kernel: JITFunction) -> None:
        if not isinstance(kernel, JITFunction):
            raise TypeError(
                f"Launch takes the bare triton.jit function, not {type(kernel).__name__}: "
                "pass a heuristics or autotune wrapper's values explicitly to its .fn"
            )
        self.kernel = kernel
        # specialization key -> (the compiled kernel, how to launch it: `_launcher`)
        self._compiled: dict[tuple, tuple[CompiledKernel, Callable[..., None]]] = {}

    def __getitem__(self, grid) -> Callable[..., None]:
        return lambda *args, **kwargs: self.launch(grid, *args, **kwargs)

    def launch(self, grid, *args: Any, **kwargs: Any) -> None:
        kernel = self.kernel
        runtime = knobs.runtime
        device = driver.active.get_current_device()
        # The options `run` adds before it binds, so the key is the one `run` would compute.
        bind_kwargs = dict(kwargs)
        bind_kwargs["debug"] = kwargs.get("debug", kernel.debug) or runtime.debug
        bind_kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
        bound, specialization, options = kernel.device_caches[device][4](*args, **bind_kwargs)
        key = (device, *specialization, *options.items())
        cached = self._compiled.get(key)
        if (
            cached is None
            or kernel.pre_run_hooks
            or _installed(runtime.launch_enter_hook)
            or _installed(runtime.launch_exit_hook)
            or runtime.add_stages_inspection_hook is not None
        ):
            compiled = kernel.run(*args, grid=grid, warmup=False, **kwargs)
            if isinstance(compiled, CompiledKernel):
                self._compiled.setdefault(key, (compiled, _launcher(compiled)))
            return
        if callable(grid):
            grid = grid(bound)
        size = len(grid)
        cached[1](
            grid[0],
            grid[1] if size > 1 else 1,
            grid[2] if size > 2 else 1,
            driver.active.get_current_stream(device),
            bound,
        )
