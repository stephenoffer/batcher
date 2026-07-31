"""Naming this process's work so an external device profiler can see the engine in it.

A Nsight Systems or `rocprof` capture of a Batcher run is, today, a wall of anonymous kernels.
The profiler sees `cudf::detail::hash_join` and a `memcpy`; it has no idea which operator issued
them, which stage they belong to, or that the 200 ms gap in the middle is the host waiting on a
Parquet read. That gap is the whole reason anyone opens the profiler, and it is exactly the part
the trace cannot label.

NVTX fixes it for free. A range pushed before an operator runs and popped after it appears in
the timeline as a labelled band above the kernels it contains, so the capture reads as the plan
rather than as the kernel list. The cost is a function call per range and nothing at all when no
profiler is attached — the driver's NVTX implementation is a no-op until a collector injects
itself.

**Four backends, in preference order, because none of them is reliably present.** The `nvtx`
package is the direct binding and the only one that supports domains and colors. `torch.cuda.nvtx`
is present in every environment that has torch with CUDA, which on an inference node is all of
them, and it maps to ROCTX on a ROCm build — so AMD gets the same annotation with no separate
code path. CuPy's binding covers a RAPIDS environment without torch. And when none is present
every entry point here is a no-op that costs one attribute lookup.

**Never raise, and never let a profiler bug become a query failure.** Annotation is diagnostic;
a range that fails to push must not take a stage with it. Every call is guarded, and a backend
that raises once is dropped for the life of the process rather than retried per range — a
mismatched push and pop under a broken backend corrupts the whole capture, which is worse than
having no capture.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator

__all__ = [
    "device_range",
    "nvtx_backend",
    "pop_range",
    "push_range",
    "range_decorator",
    "reset_nvtx_backend",
]

#: Backends in preference order, as `(module path, push attribute, pop attribute)`. The `nvtx`
#: package first because it is the only one that carries a domain, then torch because it is the
#: one that is actually installed, then CuPy for a RAPIDS environment without torch.
_BACKENDS = (
    ("nvtx", "push_range", "pop_range"),
    ("torch.cuda.nvtx", "range_push", "range_pop"),
    ("cupy.cuda.nvtx", "RangePush", "RangePop"),
)

#: Set once a backend raises, which disables annotation for the process. A backend that failed
#: to push cannot be trusted to pop, and an unbalanced stack turns a readable capture into a
#: nest of ranges that never close.
_DISABLED: list[bool] = []


def _resolve(path: str):
    """Import one dotted module path, or `None` when it is absent or fails to import.

    `torch.cuda.nvtx` imports torch, which on a CPU-only host is a multi-second import for a
    module that will then report no CUDA. That cost is paid once, inside the memoized resolver,
    and only when the `nvtx` package was not found first.
    """
    import importlib

    try:
        return importlib.import_module(path)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _backend() -> tuple[str, object, object] | None:
    """`(name, push, pop)` for the first usable backend, or `None` when none is present.

    Memoized because it imports; the answer cannot change within a process, and the resolution
    sits in front of a call that a per-operator path makes.
    """
    for path, push_attr, pop_attr in _BACKENDS:
        module = _resolve(path)
        if module is None:
            continue
        push = getattr(module, push_attr, None)
        pop = getattr(module, pop_attr, None)
        if callable(push) and callable(pop):
            return (path, push, pop)
    return None


def nvtx_backend() -> str:
    """Which profiler-annotation backend this process will use, `""` when none is available.

    Returns:
        The module path of the active backend, one of the entries in `_BACKENDS`, or `""` when
        annotation is a no-op — which is the normal case outside a profiling run and is not a
        condition worth warning about.
    """
    if _DISABLED:
        return ""
    resolved = _backend()
    return "" if resolved is None else resolved[0]


def reset_nvtx_backend() -> None:
    """Forget the resolved backend and re-enable annotation after a failure.

    The hook a test faking a backend needs, and the only way to recover a process that disabled
    itself — which is deliberate, because automatic recovery would re-enter the unbalanced-stack
    state that caused the disable.
    """
    _DISABLED.clear()
    _backend.cache_clear()


def push_range(label: str) -> None:
    """Open a named range on the device profiler's timeline.

    Must be paired with `pop_range` on every path including failure, which is why almost every
    caller should use `device_range` instead. A leaked push does not raise; it produces a range
    that swallows the rest of the capture.

    Args:
        label: What appears on the timeline. Conventionally `"Kind#id"` to match the stage
            identifiers the energy ledger and the feasibility verdicts already use, so a
            profiler capture joins to them by name.
    """
    if _DISABLED:
        return
    resolved = _backend()
    if resolved is None:
        return
    try:
        resolved[1](label)
    except Exception:
        _DISABLED.append(True)


def pop_range() -> None:
    """Close the innermost open range.

    Safe to call without a matching push: the backend either ignores it or raises, and a raise
    disables annotation rather than propagating.
    """
    if _DISABLED:
        return
    resolved = _backend()
    if resolved is None:
        return
    try:
        resolved[2]()
    except Exception:
        _DISABLED.append(True)


@contextlib.contextmanager
def device_range(label: str) -> Iterator[None]:
    """Bracket a block as one named range on the device profiler's timeline.

    The form every caller should reach for: the pop happens on the exception path too, so a
    stage that raises closes its range instead of swallowing everything after it.

    Args:
        label: What appears on the timeline.

    Yields:
        Nothing; the block runs inside the range.
    """
    push_range(label)
    try:
        yield
    finally:
        pop_range()


def range_decorator(label: str):
    """Wrap a function so every call appears as a named range.

    For the handful of call sites that are a function rather than a block — a UDF entry point, a
    model's forward pass — where wrapping the body in a `with` would mean editing user code.

    Args:
        label: What appears on the timeline.

    Returns:
        A decorator that preserves the wrapped function's name, docstring, and signature, so a
        wrapped callable stays introspectable by the inference pool's own dispatch.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with device_range(label):
                return fn(*args, **kwargs)

        return wrapper

    return decorate
