"""Process-wide runtime services for Core: the default MetadataHub, and query cancellation.

Both are process-wide state Core owns because Core is the layer that *executes*. The hub is
where measurements land; the cancellation registry is how a running execution is asked to
stop. Neither decides anything — Kyber decides, Carbonite protects — they are the bookkeeping
that executing requires.

The cancellation registry itself lives in Rust (`bc_resource::cancel`), because the thing
that has to observe the flag is the native executor holding the GIL open. What is here is the
Python-side scope: assigning a query its id, making that id reachable from the thread running
it, and turning Ctrl-C into a cancellation rather than a signal nobody can deliver.
"""

from __future__ import annotations

import signal
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from batcher._internal.logging import get_logger
from batcher._internal.native import engine
from batcher.config import active_config
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend, make_backend

__all__ = [
    "cancel_query",
    "current_query_id",
    "default_hub",
    "query_scope",
    "reset_default_hub",
    "running_queries",
]

_log = get_logger("metadata")

# The id of the query executing on this thread, or "" outside a terminal op. A `ContextVar`
# rather than a thread-local so it is also correct under asyncio and inherited by a task.
_current_query: ContextVar[str] = ContextVar("batcher_current_query", default="")
# Set by the SIGINT handler so `query_scope` can tell "the user pressed Ctrl-C" apart from
# "something called cancel_query", and raise the exception each one deserves.
_interrupted: ContextVar[bool] = ContextVar("batcher_query_interrupted", default=False)


def current_query_id() -> str:
    """The cancellable id of the query running on this thread, or `""` if none is."""
    return _current_query.get()


def cancel_query(query_id: str) -> bool:
    """Ask a running query to stop at its next morsel boundary.

    Cancellation is cooperative. The engine checks a flag between morsels, between
    operators, and between spill merge passes, so a query stops at the next such point
    rather than instantly. A pipeline breaker consumes its whole input inside one step, so a
    query in the middle of building a hash table notices when that build finishes.

    The cancelled query raises `QueryCancelledError` in the thread that started it. It never
    returns a short result, because rows that look complete and are not are worse than an
    error.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.cancel_query("q-not-running")
            False

    Args:
        query_id: The id to cancel, as reported by `running_queries`.

    Returns:
        Whether a query with that id was running. `False` means it already finished, which
        is information rather than an error.
    """
    return bool(engine().cancel_query(query_id))


def running_queries() -> list[str]:
    """List the ids of the queries executing in this process right now.

    Each terminal operation (`collect`, `to_pydict`, `write.parquet`, ...) registers one id
    for its duration. Pass one to `cancel_query` to stop it.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.running_queries()
            []

    Returns:
        The running query ids, in unspecified order. Empty when nothing is executing.
    """
    return list(engine().running_queries())


@contextmanager
def query_scope() -> Iterator[str]:
    """Make the enclosed execution cancellable, and route Ctrl-C into cancelling it.

    Yields the query id. The id is registered with the native executor for the duration and
    removed on the way out, however the block exits. Opening a scope inside an active one
    yields the outer id and changes nothing else, so a caller that brackets its own work and
    a terminal op that brackets itself agree on which query is running.

    Ctrl-C is the reason this exists. `execute_plan` runs inside `Python::allow_threads`,
    which releases the GIL, so Python's signal handler has no bytecode boundary to run at —
    a `SIGINT` during a ten-minute `collect()` is simply not delivered until the native call
    returns. Here the handler sets the cancellation flag instead, the executor sees it at its
    next morsel, and the resulting `QueryCancelledError` is re-raised as `KeyboardInterrupt`
    so the caller sees what pressing Ctrl-C is supposed to produce.

    The handler is installed for the duration and restored after, and only on the main
    thread: `signal.signal` raises off it, and a worker thread has no business owning the
    process's signal disposition. A second Ctrl-C reaches the *previous* handler, so the
    usual hard interrupt still escapes a query that will not stop.
    """
    from batcher._internal.errors import QueryCancelledError

    # Re-entrant: a scope opened inside one that is already active reuses its id rather
    # than minting a second. One terminal op is one cancellable query, and a nested scope
    # that renamed it would silently detach every handle the caller already holds — a
    # `cancel_query(id)` against the outer id would then cancel nothing while the query ran
    # on happily under the inner one.
    active = _current_query.get()
    if active:
        yield active
        return

    query_id = f"q-{uuid.uuid4().hex[:16]}"
    native = engine()
    # Registered here rather than inside `execute_plan`, so the id exists from the moment the
    # scope opens. Optimization runs before the native call, and a Ctrl-C during a slow
    # optimize would otherwise land on a token that did not exist yet and be dropped.
    native.register_query(query_id)
    id_token = _current_query.set(query_id)
    interrupt_token = _interrupted.set(False)
    previous = _install_interrupt_handler(query_id)
    try:
        yield query_id
    except QueryCancelledError:
        # A cancel the user asked for with Ctrl-C should read as Ctrl-C. One asked for by
        # `cancel_query` from another thread should not — nobody pressed anything.
        if _interrupted.get():
            raise KeyboardInterrupt from None
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
        _interrupted.reset(interrupt_token)
        _current_query.reset(id_token)
        native.unregister_query(query_id)


def _install_interrupt_handler(query_id: str):
    """Route SIGINT to cancelling `query_id`, returning the handler to restore, or `None`.

    `None` means no handler was installed, which happens off the main thread and in an
    embedding that has taken over SIGINT. Cancellation still works there through
    `cancel_query`; only the Ctrl-C shortcut is unavailable.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    try:
        previous = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):  # noqa: ARG001 - the signal module's signature
            _interrupted.set(True)
            cancel_query(query_id)
            # Hand SIGINT back to whoever had it, so a second Ctrl-C is the hard interrupt
            # the user expects when the first one appears not to have worked.
            signal.signal(signal.SIGINT, previous)

        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        # ValueError: not the main thread after all (a subinterpreter). OSError: the
        # platform refused. Neither is worth failing a query over.
        return None
    return previous


_hub: MetadataHub | None = None
_hub_backend_key: tuple[str, str | None] | None = None
# The hub is a process singleton and `execution.max_concurrent_queries` lets several queries
# run at once, so two threads reaching `default_hub()` before either has built one each build
# their own and one assignment wins. The loser is not merely wasted work: whichever caller
# already holds it keeps recording into a hub nothing will ever read again, so that query's
# whole feedback is lost — and against a durable backend it is a second connection to the same
# store left open. `carbonite.cache.result_cache` guards its singleton for the same reason;
# this one did not.
_hub_lock = threading.Lock()


def reset_default_hub() -> None:
    """Drop the cached process-wide hub so the next `default_hub()` rebuilds fresh.

    For test isolation: learned stats accumulate in the process-wide hub, so a test
    that asserts on cardinality/cost-driven plan shape can otherwise be perturbed by
    stats an earlier test recorded. Resetting between tests makes those assertions
    deterministic without changing production behavior.
    """
    global _hub, _hub_backend_key
    with _hub_lock:
        _hub = None
        _hub_backend_key = None


def default_hub() -> MetadataHub:
    """Return a process-wide MetadataHub built from the active config.

    Rebuilt if the configured backend changes, so `config_context` switching the
    metadata backend takes effect.
    """
    global _hub, _hub_backend_key
    meta = active_config().metadata
    key = (meta.backend, meta.uri)
    hub = _hub
    if hub is not None and key == _hub_backend_key:
        return hub  # the steady state, and it stays lock-free
    with _hub_lock:
        if _hub is None or key != _hub_backend_key:
            _hub = MetadataHub(_build_backend(meta.backend, meta.uri))
            _hub_backend_key = key
        return _hub


def _build_backend(backend: str, uri: str | None):
    """Construct the configured backend, degrading to in-process on failure.

    A durable backend (object storage / SQLite / Redis) can fail to construct — a
    missing optional dependency, an unreachable or misconfigured URI. Learned stats are
    an optimization, never a correctness input, so a broken store must not fail every
    query: fall back to the in-process store (this session still learns; only cross-run
    persistence is lost) and log once instead of raising into the hot path.
    """
    if backend == "in_process":
        return InProcessBackend()
    try:
        return make_backend(backend, uri)
    except Exception:
        _log.warning(
            "metadata backend %r (uri=%r) unavailable; using an in-process store "
            "(cross-run learning disabled this session)",
            backend,
            uri,
            exc_info=True,
        )
        return InProcessBackend()
