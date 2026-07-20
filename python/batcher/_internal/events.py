"""The engine's one observability event bus — every subsystem publishes here.

Batcher reports what it is doing through exactly one channel. `kyber`, `carbonite`,
`core`, `dist`, `io`, and `api` all call `publish`; the *sinks* (the terminal progress
renderer, the in-memory store behind the web UI, the JSON event log) all `subscribe`.
Neither side knows the other exists, which is what lets a new sink appear without
touching a single call site — and what keeps the bus in layer 0, importable by every
package without crossing a layer boundary.

The bus is **free when nobody is listening**. `_subscribers` is a plain tuple swapped
under a lock, so `publish` is a tuple-truthiness check on the hot path and returns
before building an `Event` when no sink is attached. That matters because progress is
published per morsel batch: a cost paid per batch must be zero by default.

Delivery is best-effort and never propagates: a sink that raises is logged once and
dropped from that emit, because observability must not be able to fail a query. Events
carry a monotonic `ts` for durations plus a wall-clock `wall` for display, since a UI
needs both and deriving one from the other after the fact is lossy.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DECISION",
    "LOG",
    "PROGRESS",
    "QUERY_END",
    "QUERY_START",
    "STAGE_END",
    "STAGE_START",
    "Event",
    "Subscriber",
    "listening",
    "publish",
    "subscribe",
]

# --- Event kinds -------------------------------------------------------------
# String constants (not an Enum) because they cross the JSON boundary to the web UI
# verbatim; an Enum would only add a `.value` at every serialization point.

#: A query began. Fields: ``sql``/``plan`` summary, ``n_ops``, ``est_rows``.
QUERY_START = "query_start"
#: A query finished. Fields: ``rows``, ``total_ms``, ``ok``, and the profile ``dict``.
QUERY_END = "query_end"
#: An operator/stage began. Fields: ``op_id``, ``kind``, ``est_rows``.
STAGE_START = "stage_start"
#: An operator/stage finished. Fields: ``op_id``, ``rows_out``, ``elapsed_ms``, ``spilled``.
STAGE_END = "stage_end"
#: Incremental progress within a stage. Fields: ``rows``, ``bytes``, ``total`` (may be None).
PROGRESS = "progress"
#: A subsystem hand-off worth explaining (see `plan.profile.Decision`).
DECISION = "decision"
#: A `batcher.*` log record, bridged onto the bus so the UI shows logs beside metrics.
LOG = "log"


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened, timestamped and addressed to a query.

    `kind` is one of the module-level constants; `fields` carries the kind-specific
    payload (documented on each constant). `query_id` is the id the event log assigns,
    or `""` for engine-level events that precede a query.
    """

    kind: str
    #: Monotonic seconds — safe to subtract for durations, meaningless as a date.
    ts: float
    #: Unix wall-clock seconds — meaningful as a date, unsafe to subtract.
    wall: float
    query_id: str = ""
    name: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The event as a JSON-encodable dict (the wire shape the web UI consumes)."""
        return {
            "kind": self.kind,
            "ts": self.ts,
            "wall": self.wall,
            "query_id": self.query_id,
            "name": self.name,
            "fields": self.fields,
        }


#: A sink: called with each `Event`. Must not raise; if it does, it is skipped.
Subscriber = Callable[[Event], None]

# Swapped wholesale under `_lock` rather than mutated, so `publish` can read it without
# taking the lock at all — the reader sees either the old tuple or the new one, never a
# half-mutated list. This is what keeps per-batch progress publishing lock-free.
_subscribers: tuple[Subscriber, ...] = ()
_lock = threading.Lock()

# Re-entrancy guard, per thread. Publishing can re-enter itself: a sink raises, `publish`
# logs that at DEBUG, the logging bridge turns the record into a LOG event, and that event
# goes back to the same failing sink. Left open, that is unbounded recursion — a broken sink
# at DEBUG verbosity blew the stack and took the query down with it, which is precisely the
# failure this module promises cannot happen. The guard makes the nested publish a no-op, so
# the first failure is still reported and the cycle cannot form.
_publishing = threading.local()


def listening() -> bool:
    """Whether any sink is attached — the guard for building an expensive payload.

    `publish` performs this check itself; call it directly only to skip *computing* the
    fields (a row count that costs something to obtain) rather than to skip the emit.

    Examples:
        .. doctest::

            >>> from batcher._internal.events import listening
            >>> listening()
            False
    """
    return bool(_subscribers)


def subscribe(sink: Subscriber) -> Callable[[], None]:
    """Attach `sink` to the bus and return a callable that detaches it.

    The returned unsubscribe is idempotent, so a sink shut down twice (a UI server
    stopped, then stopped again at interpreter exit) is not an error.

    Args:
        sink: Called with every subsequent `Event`.

    Returns:
        A zero-argument callable that removes `sink` from the bus.
    """
    global _subscribers
    with _lock:
        _subscribers = (*_subscribers, sink)

    def _unsubscribe() -> None:
        global _subscribers
        with _lock:
            _subscribers = tuple(s for s in _subscribers if s is not sink)

    return _unsubscribe


def publish(kind: str, *, query_id: str = "", name: str = "", **fields: Any) -> None:
    """Emit an event to every attached sink; a no-op when none are attached.

    Best-effort by contract: a sink that raises is reported at DEBUG and skipped, so a
    broken UI or a full disk can never fail the query that was being observed.

    Args:
        kind: One of the module-level kind constants.
        query_id: The owning query's id, or empty for engine-level events.
        name: A short human label (the operator kind, the stage name).
        **fields: The kind-specific payload; must be JSON-encodable.
    """
    sinks = _subscribers
    if not sinks:
        return
    if getattr(_publishing, "active", False):
        # Already delivering on this thread — this call is a sink's own side effect (almost
        # always the failure log). Dropping it keeps delivery acyclic; see `_publishing`.
        return
    event = Event(
        kind=kind,
        ts=time.monotonic(),
        wall=time.time(),
        query_id=query_id,
        name=name,
        fields=fields,
    )
    _publishing.active = True
    try:
        for sink in sinks:
            try:
                sink(event)
            except Exception:  # pragma: no cover - a sink must never fail a query
                _report_sink_failure()
    finally:
        _publishing.active = False


def _report_sink_failure() -> None:
    """Log a sink exception at DEBUG, without recursing back onto the bus.

    Imported lazily and called only on the failure path: `logging` bridges records *onto*
    the bus, so doing this eagerly at module scope would make the two modules mutually
    importable at load time for no benefit.
    """
    from batcher._internal.logging import get_logger

    get_logger("observe").debug("event sink raised; skipped", exc_info=True)
