"""Carrying the caller's context onto a worker thread.

`threading.Thread` does not copy context variables, and a `ThreadPoolExecutor` task runs in
whichever context its worker thread happens to hold. The control plane keeps everything
that answers "what does this query think the machine looks like" in a `contextvars.
ContextVar` -- the active `Config`, the cancellation scope, the machine-scoping key learned
statistics are filed under, the shuffle fleet, the scheduling envelope. So work handed to a
thread reads every one of them at its *default*.

That failure is silent in the worst way. A `config_context` wrapped around a call governs
whatever the calling thread does and then stops applying at the thread boundary, so a
pinned `max_memory_bytes` reverts to the static fallback, an adjusted morsel size reverts
to 16,384 rows, and a spill directory reverts to the system tempdir -- with no error
anywhere and a result that is still correct, only produced under a machine model nobody
asked for.

Both helpers here snapshot the caller's context at the moment they are called, which is
also the right *lifetime*: a long-lived consumer (a streaming query, a prefetch producer)
outlives the `with` block that launched it, so the config it runs under has to be frozen at
launch rather than read live.
"""

from __future__ import annotations

import contextvars
import functools
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["bound_to_context", "start_context_thread"]


def bound_to_context(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Callable[[], Any]:
    """Bind `fn` to a snapshot of the caller's context, as a zero-argument callable.

    For submitting to a pool, or for any callable that will run somewhere else later.

    Each call takes its **own** copy, which is not an optimization detail: a
    `contextvars.Context` cannot be entered twice at once, so reusing one snapshot across
    concurrent tasks raises `RuntimeError` on the second. One snapshot per bind, one bind
    per task.

    Args:
        fn: The callable to run under the snapshot.
        *args: Positional arguments to pass to `fn`.
        **kwargs: Keyword arguments to pass to `fn`.

    Returns:
        A zero-argument callable that runs `fn(*args, **kwargs)` under the snapshot and
        returns its result.
    """
    ctx = contextvars.copy_context()
    return functools.partial(ctx.run, functools.partial(fn, *args, **kwargs))


def start_context_thread(
    fn: Callable[..., Any],
    /,
    *args: Any,
    name: str | None = None,
    daemon: bool = True,
    **kwargs: Any,
) -> threading.Thread:
    """Start a thread running `fn(*args, **kwargs)` under the caller's context.

    The drop-in for `threading.Thread(target=fn, ...).start()` wherever the work being
    handed off is part of the caller's query rather than independent of it.

    Args:
        fn: The thread body.
        *args: Positional arguments to pass to `fn`.
        name: The thread name, for stack traces and `ps`.
        daemon: Whether the thread should not hold the interpreter open at exit.
        **kwargs: Keyword arguments to pass to `fn`.

    Returns:
        The started `threading.Thread`.
    """
    thread = threading.Thread(
        target=bound_to_context(fn, *args, **kwargs), name=name, daemon=daemon
    )
    thread.start()
    return thread
