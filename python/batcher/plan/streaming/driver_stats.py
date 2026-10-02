"""What a driver-produced stream reads and retains, for the micro-batch progress record.

A stream-stream join, a session window, a watermark dedup and a union of streams are run
by a *driver*: a generator that reads its own sources and yields finished output rows. The
engine only sees the output, so its progress record reported the rows the driver emitted as
the rows it read, and no state at all for operators that hold buffers, open sessions or a
seen-key set.

`DriverStats` is the side channel. The launcher activates one for the query
(`collecting`), the source read path adds every row it pulls from an unbounded source, and
each driver reports its retained state after every batch. The runner then takes both into
the progress record. Nothing is collected when no stats are active, which is every
`iter_batches()` consumer.

The active object is found through a context variable, so it follows the query's context
snapshot onto the engine loop thread and onto the per-input reader threads
(`api.terminal.stream.multiplex`). They share the one object, hence the lock.

Layer: plan (neutral contract).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

from batcher.plan.streaming.progress import StateOperatorProgress

__all__ = ["DriverStats", "active_driver_stats", "collecting", "report_state"]


class DriverStats:
    """Source rows read and per-operator state, reported by a stream's driver."""

    __slots__ = ("_lock", "_operators", "_source_rows")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source_rows = 0
        self._operators: dict[str, StateOperatorProgress] = {}

    def add_source_rows(self, rows: int) -> None:
        """Count `rows` read from an unbounded source."""
        with self._lock:
            self._source_rows += rows

    def take_source_rows(self) -> int:
        """The rows read since the last call, resetting the count."""
        with self._lock:
            rows, self._source_rows = self._source_rows, 0
            return rows

    def report(self, operator: StateOperatorProgress) -> None:
        """Record `operator`'s current state, replacing its previous report."""
        with self._lock:
            self._operators[operator.operator_name] = operator

    def operators(self) -> tuple[StateOperatorProgress, ...]:
        """The latest report of every operator, in the order they first reported."""
        with self._lock:
            return tuple(self._operators.values())


_ACTIVE: ContextVar[DriverStats | None] = ContextVar("batcher_driver_stats", default=None)


def active_driver_stats() -> DriverStats | None:
    """The stats the running driver reports into, or None when nothing collects them."""
    return _ACTIVE.get()


@contextlib.contextmanager
def collecting(stats: DriverStats) -> Iterator[DriverStats]:
    """Make `stats` the active collector for the block (and any context copied inside it)."""
    token = _ACTIVE.set(stats)
    try:
        yield stats
    finally:
        _ACTIVE.reset(token)


def report_state(
    operator_name: str,
    *retained: Any,
    watermark: int | None = None,
    late_dropped: int = 0,
    removed: int = 0,
) -> None:
    """Report a driver operator's retained state, when a query is collecting it.

    Args:
        operator_name: The operator, as `StateOperatorProgress.operator_name` names it.
        *retained: The Arrow tables the operator holds (``None`` for an empty buffer).
        watermark: The operator's event-time watermark in epoch microseconds.
        late_dropped: Rows dropped this batch for arriving below the watermark.
        removed: Rows this batch evicted from the state.
    """
    stats = _ACTIVE.get()
    if stats is None:
        return
    held = [t for t in retained if t is not None]
    stats.report(
        StateOperatorProgress(
            operator_name,
            num_rows_total=sum(t.num_rows for t in held),
            num_rows_removed=removed,
            memory_used_bytes=sum(t.nbytes for t in held),
            num_late_inputs_dropped=late_dropped,
            watermark_micros=watermark,
        )
    )
