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

import contextlib
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TypeVar

from batcher._internal.logging import get_logger
from batcher._internal.native import engine
from batcher.config import active_config
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend, make_backend

__all__ = [
    "bounded_iteration",
    "cancel_query",
    "cancellable",
    "current_query_id",
    "default_hub",
    "expire_query",
    "note_stage",
    "query_scope",
    "query_time_remaining",
    "reset_default_hub",
    "running_queries",
    "timed_terminal",
]

_T = TypeVar("_T")
_A = TypeVar("_A")

_log = get_logger("metadata")

# The id of the query executing on this thread, or "" outside a terminal op. A `ContextVar`
# rather than a thread-local so it is also correct under asyncio and inherited by a task.
_current_query: ContextVar[str] = ContextVar("batcher_current_query", default="")
# Set by the SIGINT handler so `query_scope` can tell "the user pressed Ctrl-C" apart from
# "something called cancel_query", and raise the exception each one deserves.
_interrupted: ContextVar[bool] = ContextVar("batcher_query_interrupted", default=False)


@dataclass
class _QueryState:
    """The Python-side half of one cancellable query: its deadline and how it ended.

    The native flag stops the engine at its next morsel. It cannot stop the Python loops a
    query also runs -- `map_batches` calls a user function per batch on the driver -- so
    `cancel_query` and the timeout set `cancelled` here as well, and those loops poll it
    through `cancellable`.
    """

    query_id: str
    timeout_s: float | None
    started: float = field(default_factory=time.monotonic)
    stage: str = ""
    timed_out: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)


# Live queries by id. Written only by `query_scope`, on entry and exit; read from any thread.
_states: dict[str, _QueryState] = {}


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
    running = bool(engine().cancel_query(query_id))
    state = _states.get(query_id)
    if state is not None:
        state.cancelled.set()
    return running or state is not None


def note_stage(name: str) -> None:
    """Record the phase the current query has entered, so a timeout can name it.

    Args:
        name: The phase name, such as ``"kyber.optimize_full"`` or ``"core.execute"``.
    """
    state = _states.get(_current_query.get())
    if state is not None:
        state.stage = name


def cancellable(call: Callable[[_A], _T], stage: str) -> Callable[[_A], _T]:
    """Wrap a per-batch `call` so it raises once the current query is cancelled.

    For the Python-side loops a query runs (one call per `map_batches` batch), which the
    engine's own morsel-boundary check cannot reach. The query is captured when the wrapper
    is built, on the thread that owns it, so the check still works when the calls are
    fanned out to a thread pool -- a pool thread does not inherit the query's context.
    Outside a query scope it returns `call` itself.

    Args:
        call: The per-batch function.
        stage: Names the loop, for the timeout message.

    Returns:
        A function that checks for cancellation, then calls `call`.
    """
    state = _states.get(_current_query.get())
    if state is None:
        return call

    def checked(batch: _A) -> _T:
        if state.cancelled.is_set():
            state.stage = stage
            raise _cancelled_error(state)
        return call(batch)

    return checked


def query_time_remaining() -> float | None:
    """Seconds left before the current query's timeout, or None when it has no limit.

    For a step that blocks somewhere a cancellation flag cannot reach -- a process pool
    waiting on its children -- so it can bound its own wait by the query's deadline.

    Returns:
        The remaining seconds (never negative), or None outside a timed query.
    """
    state = _states.get(_current_query.get())
    if state is None or state.timeout_s is None:
        return None
    return max(0.0, state.timeout_s - (time.monotonic() - state.started))


def expire_query(stage: str) -> Exception:
    """Cancel the current query as timed out, and return the error to raise for it.

    Used by a step whose wait was bounded by `query_time_remaining` and ran out, so the
    rest of the query sees the same cancellation the timer would have delivered.

    Args:
        stage: Names the step that ran out of time.

    Returns:
        The timeout error naming the limit and `stage`.
    """
    state = _states.get(_current_query.get())
    if state is None:
        return _timeout_error(None, stage)
    state.timed_out = True
    state.stage = stage
    cancel_query(state.query_id)
    return _cancelled_error(state)


def _cancelled_error(state: _QueryState) -> Exception:
    """The error a cancelled query raises: a timeout names the limit and the phase."""
    from batcher._internal.errors import QueryCancelledError

    if not state.timed_out:
        return QueryCancelledError(f"query {state.query_id} was cancelled")
    return _timeout_error(state.timeout_s, state.stage or "executing", state.query_id)


def _timeout_error(timeout_s: float | None, stage: str, query_id: str = "") -> Exception:
    from batcher._internal.errors import QueryCancelledError

    who = f"query {query_id}" if query_id else "the query"
    return QueryCancelledError(
        f"{who} exceeded execution.query_timeout_s={timeout_s}s during {stage} and was cancelled",
        hint="Raise execution.query_timeout_s, or set it to None for no limit.",
    )


@contextmanager
def timed_terminal() -> Iterator[None]:
    """Open a query scope for a whole terminal operation when a query timeout is set.

    A no-op without `execution.query_timeout_s`, so a terminal op runs exactly as it
    always has. With one, the scope opens before optimization and admission rather than
    around execution alone, so the limit bounds the whole operation the user called --
    including a `map_batches` pipeline, which never opens the relational scope.
    """
    if active_config().execution.query_timeout_s is None:
        yield
        return
    with query_scope():
        yield


def bounded_iteration(batches: Iterable[_T], *, stage: str = "iter_batches") -> Iterator[_T]:
    """Yield `batches`, raising once producing them has taken longer than the query timeout.

    Counts only the time spent inside the producer, never the time the consumer holds a
    batch, so a training loop that spends minutes per step is not cut off by a limit meant
    for the query. Checked as each batch arrives. The source iterator is closed however the
    loop ends -- exhausted, abandoned, or timed out -- so a timed-out stream releases what
    it holds instead of waiting for the garbage collector.

    Args:
        batches: The batches to pass through.
        stage: Names the operation, for the timeout message.

    Yields:
        The batches, unchanged.

    Raises:
        QueryCancelledError: If production exceeded `execution.query_timeout_s`.
    """
    timeout = active_config().execution.query_timeout_s
    source = iter(batches)
    try:
        if timeout is None:
            yield from source
            return
        spent = 0.0
        while True:
            started = time.monotonic()
            try:
                batch = next(source)
            except StopIteration:
                return
            spent += time.monotonic() - started
            if spent > timeout:
                raise _timeout_error(timeout, stage)
            yield batch
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


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

    # Random, never sequential or caller-supplied. The engine's cleanup removes whatever is
    # registered under the id (`bc_py::unregister_query`), so an id that could be live twice
    # would let one query's exit silently deregister another's — leaving it uncancellable.
    # Sixteen hex digits of uuid4 is what makes that unreachable; changing this scheme means
    # threading the token through and using `unregister_token` on the Rust side.
    query_id = f"q-{uuid.uuid4().hex[:16]}"
    native = engine()
    # Registered here rather than inside `execute_plan`, so the id exists from the moment the
    # scope opens. Optimization runs before the native call, and a Ctrl-C during a slow
    # optimize would otherwise land on a token that did not exist yet and be dropped.
    native.register_query(query_id)
    state = _QueryState(query_id, active_config().execution.query_timeout_s)
    _states[query_id] = state
    id_token = _current_query.set(query_id)
    interrupt_token = _interrupted.set(False)
    previous = _install_interrupt_handler(query_id)
    timer = _arm_timeout(state)
    try:
        yield query_id
    except QueryCancelledError as exc:
        # A cancel the user asked for with Ctrl-C should read as Ctrl-C. One asked for by
        # `cancel_query` from another thread should not — nobody pressed anything. One the
        # timeout asked for names the limit and the phase it interrupted.
        if _interrupted.get():
            raise KeyboardInterrupt from None
        if state.timed_out:
            raise _cancelled_error(state) from exc
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
        _interrupted.reset(interrupt_token)
        _current_query.reset(id_token)
        _states.pop(query_id, None)
        native.unregister_query(query_id)


def _arm_timeout(state: _QueryState) -> threading.Timer | None:
    """Start the timer that cancels `state`'s query at its deadline, or None without one.

    It cancels through `cancel_query` -- the native flag and the Python-side event -- so a
    timeout is exactly a cancellation that knows why it happened.
    """
    if state.timeout_s is None:
        return None

    def expire() -> None:
        state.timed_out = True
        with contextlib.suppress(Exception):  # the query may have finished this instant
            cancel_query(state.query_id)

    timer = threading.Timer(state.timeout_s, expire)
    timer.daemon = True
    timer.start()
    return timer


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
_hub_backend_key: tuple[str, str | None, bool] | None = None
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
    key = (meta.backend, meta.uri, meta.require_durable)
    hub = _hub
    if hub is not None and key == _hub_backend_key:
        return hub  # the steady state, and it stays lock-free
    with _hub_lock:
        if _hub is None or key != _hub_backend_key:
            _hub = MetadataHub(_build_backend(meta.backend, meta.uri, meta.require_durable))
            _hub_backend_key = key
        return _hub


def _build_backend(backend: str, uri: str | None, require_durable: bool = False):
    """Construct the configured backend, degrading to in-process on failure.

    A durable backend (object storage / SQLite / Redis) can fail to construct — a
    missing optional dependency, an unreachable or misconfigured URI. Learned stats are
    an optimization, never a correctness input, so a broken store must not fail every
    query: fall back to the in-process store (this session still learns; only cross-run
    persistence is lost) and log once instead of raising into the hot path.

    `metadata.require_durable` turns that off. A deployment that depends on cross-run
    learning otherwise keeps running with none, and the one sign is a warning in a log.
    """
    if backend == "in_process":
        return InProcessBackend()
    try:
        return make_backend(backend, uri)
    except Exception as exc:
        if require_durable:
            from batcher._internal.errors import ConfigError

            raise ConfigError(
                f"metadata backend {backend!r} (uri={uri!r}) could not be built, and "
                f"metadata.require_durable is set, so Batcher will not fall back to an "
                f"in-process store: {exc}"
            ) from exc
        _log.warning(
            "metadata backend %r (uri=%r) unavailable; using an in-process store "
            "(cross-run learning disabled this session)",
            backend,
            uri,
            exc_info=True,
        )
        return InProcessBackend()
