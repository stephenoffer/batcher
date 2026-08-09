"""`StreamingQueryListener` — a callback that sees every query start, batch, and stop.

Spark's `StreamingQueryListener` and `spark.streams.addListener`. A streaming query runs
for days on a background thread, and the only surface it offered was polling: read
`recent_progress` on a timer and diff it yourself. That misses batches (the window is
bounded), misses the start entirely, and cannot see a termination without a second thread.
A listener sees each event once, as it happens.

The registry lives here, in `plan`, because both sides need it and neither may import the
other: `core` fires the events from the micro-batch loop, `api` is where a user registers a
listener. It is the same arrangement as `metadata`'s hub — a neutral layer holding the
state two layers exchange.

**A listener may not break a query.** Every callback is invoked inside a guard that logs
and continues, exactly as the observability bus does. Monitoring that can fail the thing it
monitors is worse than no monitoring: the failure mode is a production stream dying because
a metrics endpoint was down.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.plan.streaming.progress import StreamingQueryProgress

__all__ = [
    "QueryProgressEvent",
    "QueryStartedEvent",
    "QueryTerminatedEvent",
    "StreamingQueryListener",
    "add_streaming_listener",
    "notify_query_progress",
    "notify_query_started",
    "notify_query_terminated",
    "remove_streaming_listener",
    "streaming_listeners",
]


@dataclass(frozen=True, slots=True)
class QueryStartedEvent:
    """A streaming query began running (Spark `QueryStartedEvent`).

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import QueryStartedEvent
            >>> QueryStartedEvent("nightly", "nightly", 0.0).name
            'nightly'
    """

    #: The query's name, unique among *active* queries.
    name: str
    #: A stable id for the query. The name, matching `StreamingQuery.id`.
    id: str
    #: Unix wall-clock seconds at which the query started.
    timestamp: float


@dataclass(frozen=True, slots=True)
class QueryProgressEvent:
    """A micro-batch completed (Spark `QueryProgressEvent`).

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import (
            ...     QueryProgressEvent,
            ...     StreamingQueryProgress,
            ... )
            >>> event = QueryProgressEvent("nightly", StreamingQueryProgress(2, 5, 5, 1.0, 0.0))
            >>> event.progress.batch_id
            2
    """

    #: The query's name.
    name: str
    #: The micro-batch's metrics.
    progress: StreamingQueryProgress


@dataclass(frozen=True, slots=True)
class QueryTerminatedEvent:
    """A streaming query stopped, cleanly or with an exception (Spark parity).

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import QueryTerminatedEvent
            >>> QueryTerminatedEvent("nightly", "nightly").exception is None
            True
    """

    #: The query's name.
    name: str
    #: A stable id for the query. The name, matching `StreamingQuery.id`.
    id: str
    #: The string form of the exception that ended it, or None for a clean stop. A
    #: string rather than the exception, because a listener that is shipping this to a
    #: metrics system needs it serializable and one that wants the object can read
    #: `StreamingQuery.exception()`.
    exception: str | None = None


class StreamingQueryListener:
    """Subclass and override the callbacks you care about (Spark parity).

    Register with `bt.add_streaming_listener`; every active and future query in the
    process reports to it. All three callbacks default to doing nothing, so a listener
    that only wants terminations overrides only that one.

    Callbacks run **on the query's own loop thread**, between micro-batches. Keep them
    cheap: work done here is latency the next batch pays. Ship to a queue rather than
    doing I/O inline if the destination can be slow.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> class Watcher(bt.StreamingQueryListener):
            ...     def on_query_progress(self, event):
            ...         if event.progress.num_late_rows:
            ...             print(f"{event.name}: {event.progress.num_late_rows} late rows")
            >>> watcher = Watcher()
            >>> bt.add_streaming_listener(watcher)
            >>> bt.remove_streaming_listener(watcher)
            True
    """

    def on_query_started(self, event: QueryStartedEvent) -> None:
        """Called once when a query starts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> class Log(bt.StreamingQueryListener):
                ...     def on_query_started(self, event):
                ...         print("started", event.name)

        Args:
            event: The start event, carrying the query's name and start time.
        """

    def on_query_progress(self, event: QueryProgressEvent) -> None:
        """Called after every completed micro-batch.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> class Log(bt.StreamingQueryListener):
                ...     def on_query_progress(self, event):
                ...         print(event.progress)

        Args:
            event: The progress event, carrying the micro-batch's metrics.
        """

    def on_query_terminated(self, event: QueryTerminatedEvent) -> None:
        """Called once when a query stops, cleanly or with an exception.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> class Alert(bt.StreamingQueryListener):
                ...     def on_query_terminated(self, event):
                ...         if event.exception:
                ...             print("failed:", event.exception)

        Args:
            event: The termination event, carrying the failure when there was one.
        """

    # Spark spellings, so a listener ported from a PySpark job keeps working. Thin
    # aliases: a subclass may override either spelling and both are dispatched.
    def onQueryStarted(self, event: QueryStartedEvent) -> None:
        """Spark spelling of `on_query_started`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> class Ported(bt.StreamingQueryListener):
                ...     def onQueryStarted(self, event):
                ...         pass

        Args:
            event: The start event.
        """

    def onQueryProgress(self, event: QueryProgressEvent) -> None:
        """Spark spelling of `on_query_progress`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> class Ported(bt.StreamingQueryListener):
                ...     def onQueryProgress(self, event):
                ...         pass

        Args:
            event: The progress event.
        """

    def onQueryTerminated(self, event: QueryTerminatedEvent) -> None:
        """Spark spelling of `on_query_terminated`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> class Ported(bt.StreamingQueryListener):
                ...     def onQueryTerminated(self, event):
                ...         pass

        Args:
            event: The termination event.
        """


# The process-wide registry. A tuple swapped under a lock, so the fire path is a
# truthiness check and needs no lock at all — the same shape as `_internal.events`,
# and for the same reason: this is called once per micro-batch.
_LISTENERS: tuple[StreamingQueryListener, ...] = ()
_LOCK = threading.Lock()


def add_streaming_listener(listener: StreamingQueryListener) -> None:
    """Register `listener` to receive every streaming query's events (Spark `addListener`).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> listener = bt.StreamingQueryListener()
            >>> bt.add_streaming_listener(listener)
            >>> listener in bt.streaming_listeners()
            True
            >>> bt.remove_streaming_listener(listener)
            True

    Args:
        listener: The listener to register. Registering one twice adds it once.
    """
    global _LISTENERS
    with _LOCK:
        if listener not in _LISTENERS:
            _LISTENERS = (*_LISTENERS, listener)


def remove_streaming_listener(listener: StreamingQueryListener) -> bool:
    """Unregister `listener` (Spark `removeListener`).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> listener = bt.StreamingQueryListener()
            >>> bt.add_streaming_listener(listener)
            >>> bt.remove_streaming_listener(listener)
            True
            >>> bt.remove_streaming_listener(listener)
            False

    Args:
        listener: The listener to remove.

    Returns:
        True if it was registered, False if it was not.
    """
    global _LISTENERS
    with _LOCK:
        remaining = tuple(x for x in _LISTENERS if x is not listener)
        removed = len(remaining) != len(_LISTENERS)
        _LISTENERS = remaining
    return removed


def streaming_listeners() -> list[StreamingQueryListener]:
    """Every registered streaming-query listener, in registration order.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> isinstance(bt.streaming_listeners(), list)
            True

    Returns:
        The registered listeners.
    """
    return list(_LISTENERS)


def _fire(method: str, spark_method: str, event: object) -> None:
    """Deliver `event` to every listener, never letting one break the query.

    Both spellings are dispatched so a subclass may override either. A listener that
    overrides neither gets two no-op calls, which is free.
    """
    listeners = _LISTENERS
    if not listeners:
        return
    for listener in listeners:
        for name in (method, spark_method):
            try:
                getattr(listener, name)(event)
            except Exception:
                from batcher._internal.logging import get_logger

                get_logger("streaming").warning(
                    "streaming listener %s.%s raised; the query is unaffected",
                    type(listener).__name__,
                    name,
                    exc_info=True,
                )


def notify_query_started(name: str, timestamp: float) -> None:
    """Deliver a `QueryStartedEvent` to every listener.

    Args:
        name: The query's name.
        timestamp: Unix wall-clock seconds at which it started.
    """
    _fire("on_query_started", "onQueryStarted", QueryStartedEvent(name, name, timestamp))


def notify_query_progress(name: str, progress: StreamingQueryProgress) -> None:
    """Deliver a `QueryProgressEvent` to every listener.

    Args:
        name: The query's name.
        progress: The completed micro-batch's metrics.
    """
    _fire("on_query_progress", "onQueryProgress", QueryProgressEvent(name, progress))


def notify_query_terminated(name: str, exception: BaseException | None) -> None:
    """Deliver a `QueryTerminatedEvent` to every listener.

    Args:
        name: The query's name.
        exception: The failure that ended it, or None for a clean stop.
    """
    detail = None if exception is None else f"{type(exception).__name__}: {exception}"
    _fire(
        "on_query_terminated",
        "onQueryTerminated",
        QueryTerminatedEvent(name, name, detail),
    )
