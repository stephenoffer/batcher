"""The `StreamingQuery` handle users hold, and the registry of running queries.

A thin façade over `core.StreamingQueryEngine` (`stop` / `await_termination` / `status` /
`recent_progress` / `is_active`), plus the process-wide active-query registry exposed as
`bt.streams`. The launchers that construct the engine live in `_launch` (single-node) and
`_distributed` (the micro-batch fanned across the cluster); this module knows nothing about
either, so the handle is the same object whichever one produced it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from batcher.plan.streaming import StreamingQueryProgress, StreamingQueryStatus

if TYPE_CHECKING:
    from batcher.core.streaming_query import StreamingQueryEngine

__all__ = ["StreamingQuery", "active_streams", "await_any_termination"]


# Process-wide registry of running queries, surfaced as `bt.streams`.
_ACTIVE: dict[str, StreamingQuery] = {}
_LOCK = threading.Lock()
_COUNTER = 0


def _next_name() -> str:
    global _COUNTER
    with _LOCK:
        _COUNTER += 1
        return f"query-{_COUNTER}"


def _sweep_stopped() -> None:
    """Drop queries that have finished. **Caller must hold `_LOCK`.**

    A query is removed on `stop()` or a completed `await_termination()`, and a driver
    that does neither is ordinary: a scheduler firing an `available_now` backfill every
    few minutes gets its rows and moves on. Those entries stayed forever, each holding a
    handle, an engine, a processor, a sink and a bounded progress deque — invisible in
    `bt.streams()`, which filters on liveness, and unbounded in the dict behind it.
    """
    for name in [n for n, q in _ACTIVE.items() if not q.is_active]:
        _ACTIVE.pop(name, None)


def _register(name: str, query: StreamingQuery) -> None:
    """Add `query` to the active registry, rejecting a duplicate *active* name.

    Two active queries under one name would leave the first unreachable through
    `bt.streams` while it kept running — unstoppable and still writing to its sink. Spark
    rejects a duplicate active name; this is the one place that enforces it. A name whose
    prior query has already stopped is free to reuse.
    """
    from batcher._internal.errors import PlanError

    with _LOCK:
        existing = _ACTIVE.get(name)
        if existing is not None and existing.is_active:
            raise PlanError(
                f"a streaming query named {name!r} is already active; stop it first or "
                "pass a distinct name= to write(...)"
            )
        _sweep_stopped()
        _ACTIVE[name] = query


def _deregister(name: str) -> None:
    """Drop `name` from the active registry (a stopped or failed-to-start query)."""
    with _LOCK:
        _ACTIVE.pop(name, None)


def _warn_if_checkpoint_not_durable(location: str) -> None:
    """Under ``resilience="spot"``, warn when the checkpoint location looks node-local.

    A streaming query's exactly-once recovery lives in its `checkpoint_location`. On a
    spot/preemptible cluster a reclaimed node takes a node-local checkpoint with it, so
    a restart cannot resume — defeating the durability the checkpoint exists for. A
    durable location (object storage, or a shared mount) survives the node. Only a
    warning, not an error: a bare path may legitimately be a shared filesystem, which
    we cannot tell apart from node-local storage."""
    import re
    import warnings

    from batcher.config import active_config

    if active_config().distributed.resilience != "spot":
        return
    has_scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", location)
    node_local = has_scheme is None or location.lower().startswith("file://")
    if node_local:
        warnings.warn(
            f"streaming checkpoint_location {location!r} looks node-local; on a "
            "spot/preemptible cluster a reclaimed node loses the checkpoint and its "
            "exactly-once recovery. Use a durable location (s3://, gs://, hdfs://, or a "
            "shared mount that survives node loss).",
            stacklevel=3,
        )


def active_streams() -> list[StreamingQuery]:
    """All currently-active streaming queries (the `bt.streams` accessor).

    Sweeps the finished ones out on the way past, so the registry is bounded by the
    queries actually running rather than by every query the process ever started.
    """
    with _LOCK:
        _sweep_stopped()
        return list(_ACTIVE.values())


def await_any_termination(timeout: float | None = None) -> bool:
    """Block until any active streaming query stops (Spark ``awaitAnyTermination``).

    Waits for the first of the currently-running queries to terminate, re-raising its
    exception if it failed. With no active queries, returns immediately.

    Args:
        timeout: Maximum seconds to wait; ``None`` waits indefinitely.

    Returns:
        ``True`` if a query stopped (or none were active), ``False`` on timeout.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.await_any_termination(timeout=0.0)  # no active queries
            True
    """
    import time

    watching = active_streams()
    if not watching:
        return True
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        for q in watching:
            if not q.is_active:
                q.await_termination(0.0)  # re-raise if it failed; deregister
                return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        # Poll: the queries run on their own daemon threads, so a short sleep between
        # liveness checks keeps this cheap without a shared condition variable.
        time.sleep(0.05)


class StreamingQuery:
    """A handle to a running streaming query (Spark `StreamingQuery` parity).

    Returned by `ds.write(..., trigger=...)` (and `ds.write.console()/memory()/...`)
    when the write runs in streaming mode. Methods mirror Spark: `stop()`,
    `await_termination(timeout)`, `status`, `recent_progress`, `is_active`.
    """

    __slots__ = ("_engine", "_name")

    def __init__(self, name: str, engine: StreamingQueryEngine) -> None:
        self._name = name
        self._engine = engine

    def __repr__(self) -> str:
        """Show the query name, liveness, and the most recent micro-batch's progress."""
        state = "active" if self._engine.is_active else "stopped"
        last = self.last_progress
        if last is None:
            tail = "no batches yet"
        else:
            tail = (
                f"batch {last.batch_id}, {last.num_input_rows} rows in "
                f"@ {last.input_rows_per_second:.0f} rows/s"
            )
        return f"StreamingQuery(name={self._name!r}, {state}, {tail})"

    def __enter__(self) -> StreamingQuery:
        """Enter a ``with`` block; the query keeps running until the block exits."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the query on leaving the ``with`` block (even if the body raised)."""
        self.stop()

    @property
    def name(self) -> str:
        """The query's name (auto-generated if not supplied)."""
        return self._name

    @property
    def id(self) -> str:
        """A stable identifier for the query (its name); Spark `StreamingQuery.id` parity."""
        return self._name

    @property
    def is_active(self) -> bool:
        """Whether the micro-batch loop is still running."""
        return self._engine.is_active

    def stop(self) -> None:
        """Halt the query at the next micro-batch boundary and wait for it to finish."""
        self._engine.stop()
        with _LOCK:
            _ACTIVE.pop(self._name, None)

    def await_termination(self, timeout: float | None = None) -> bool:
        """Block until the query stops (or `timeout` seconds elapse).

        Returns whether the query has stopped. Re-raises any exception the query
        loop failed with.
        """
        stopped = self._engine.await_termination(timeout)
        if stopped:
            with _LOCK:
                _ACTIVE.pop(self._name, None)
        return stopped

    @property
    def status(self) -> StreamingQueryStatus:
        """A point-in-time snapshot of the query's state."""
        return self._engine.status()

    @property
    def recent_progress(self) -> list[StreamingQueryProgress]:
        """Per-micro-batch metrics for the most recent batches (rolling window).

        A *property*, matching `status` and `last_progress` beside it and Spark's
        `recentProgress`. It was the one accessor on this handle that had to be called,
        so a ported `query.recentProgress` read back as a bound method — truthy, with a
        `len()` that raises — rather than as the list of batches it names. The window is
        bounded by `streaming.progress_history`.
        """
        return self._engine.recent_progress()

    @property
    def last_progress(self) -> StreamingQueryProgress | None:
        """The most recent micro-batch's metrics, or None if none completed yet."""
        progress = self._engine.recent_progress()
        return progress[-1] if progress else None

    def exception(self) -> BaseException | None:
        """The exception that terminated the query, or None if it is healthy.

        Spark `StreamingQuery.exception()` parity: read the failure without letting
        it propagate (unlike `await_termination`, which re-raises). Returns None while
        the query is running normally or after a clean stop.

        Returns:
            The terminating exception, or None.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> stream = bt.read.rate(rows_per_second=5, num_rows=5, pace=False)
                >>> q = stream.write.memory(  # doctest: +SKIP
                ...     "m", trigger=bt.Trigger.available_now()
                ... )
                >>> q.await_termination()  # doctest: +SKIP
                True
                >>> q.exception() is None  # doctest: +SKIP
                True
        """
        return self._engine.exception

    # --- Spark Structured Streaming spellings ------------------------------
    # `ds.write(...)` returns this handle; a job ported from Spark reaches for the
    # camelCase names. They are thin aliases of the snake_case methods above.
    def awaitTermination(self, timeout: float | None = None) -> bool:
        """Spark spelling of `await_termination` — block until the query stops.

        Args:
            timeout: Maximum seconds to wait; ``None`` waits indefinitely.

        Returns:
            Whether the query has stopped.
        """
        return self.await_termination(timeout)

    def processAllAvailable(self) -> bool:
        """Block until all currently-available data is processed (Spark parity).

        Waits for the query to finish draining. This is meaningful for a draining
        trigger (`Trigger.once` / `Trigger.available_now`); on a never-ending
        continuous stream it blocks until the query is stopped elsewhere.

        Returns:
            Whether the query has stopped once all available data was processed.
        """
        return self.await_termination()

    @property
    def isActive(self) -> bool:
        """Spark spelling of `is_active` — whether the query is still running."""
        return self.is_active

    @property
    def lastProgress(self) -> StreamingQueryProgress | None:
        """Spark spelling of `last_progress` — the most recent micro-batch's metrics."""
        return self.last_progress

    @property
    def recentProgress(self) -> list[StreamingQueryProgress]:
        """Spark spelling of `recent_progress` — metrics for the most recent batches."""
        return self.recent_progress
