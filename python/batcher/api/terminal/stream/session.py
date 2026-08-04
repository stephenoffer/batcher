"""The streaming session window — sessions whose end you only learn by waiting.

A tumbling window knows its bounds before a single row arrives, so a streaming
aggregation can close one the moment the watermark passes its end. A *session* window
knows nothing in advance: every event can extend the session it lands in, and an event
arriving between two sessions merges them into one. So the operator cannot emit a
session when it sees it; it has to hold the rows until nothing can change them.

The watermark is what says nothing can. A session whose last event is at ``t`` can only
be extended by an event in ``(t, t + gap]``, and the watermark is the engine's promise
that no event older than it will arrive. So the session is complete exactly once the
watermark passes ``t + gap`` — and a row that turns up older than the watermark is
dropped as late, which is what makes the promise keepable rather than merely hopeful.

That gives the memory bound: buffered rows belong only to sessions still open, which is
the live key space times the gap, not the length of the stream. It also gives the
correctness argument, and it is worth stating precisely because it is the whole design.
A closed session can never gain a row: a new row has event time at least the watermark,
which is already past ``t + gap``, so it starts a session of its own. The aggregation
over closed rows is therefore the same aggregation the bounded operator would compute,
and it *is* that operator — `sessionize` is called on each closed slice, so the two
paths cannot drift.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.io.source import Source
from batcher.plan.logical import StreamingSessionWindow

__all__ = ["stream_session_window"]

#: The internal columns `mark_sessions` adds, dropped before anything is emitted.
_INTERNAL = ("_t", "_prev", "_new", "_sid", "_session_end")


def stream_session_window(
    plan: StreamingSessionWindow,
    source: Source,
    batch_size: int | None,
) -> Iterator[pa.RecordBatch]:
    """Emit each session once the watermark guarantees no event can still extend it.

    Driver-executed today, including under `distributed=True`: the result is identical
    (the same rows through the same code), it simply does not fan out. Its mergeable form
    is a shuffle by `partition_by`, because a session belongs to exactly one key -- so each
    worker would own a disjoint set of keys, and `combine` is their union. Without
    `partition_by` there is one global session chain and no shuffle key, which is the shape
    that genuinely cannot distribute.

    Args:
        plan: The `StreamingSessionWindow` node.
        source: The single unbounded source beneath it.
        batch_size: Optional output rebatching.

    Yields:
        One row per closed session: the partition keys, `session_start`, `session_end`,
        and the aggregates.
    """
    from batcher.api.terminal.stream.dispatch import _iter_streaming

    state = _SessionBuffer(plan)
    for batch in _iter_streaming(plan.input, [source], None):
        if batch.num_rows == 0:
            continue
        for out in state.push(batch):
            yield from _emit(out, batch_size)
    # The stream ended, so every remaining session is complete by definition: nothing
    # more can arrive to extend it. Without this a bounded-but-unbounded-typed source
    # (a drained `available_now` trigger, a finite test feed) would silently lose its
    # last session per key — the one shape a watermark alone never closes.
    for out in state.drain():
        yield from _emit(out, batch_size)


def _emit(table: pa.Table, batch_size: int | None) -> Iterator[pa.RecordBatch]:
    if table.num_rows == 0:
        return
    yield from (table.to_batches() if batch_size is None else table.to_batches(batch_size))


class _SessionBuffer:
    """Rows whose session may still be open, plus the watermark that closes them.

    Held as one Arrow table rather than a per-key Python structure: the split into
    closed and open rows is a relational computation over the whole buffer, so it runs
    in the engine, and the driver never touches a row.
    """

    __slots__ = ("_buffer", "_max_seen", "_plan")

    def __init__(self, plan: StreamingSessionWindow) -> None:
        self._plan = plan
        self._buffer: pa.Table | None = None
        self._max_seen: int | None = None  # the largest event time seen, in epoch micros

    def push(self, batch: pa.RecordBatch) -> Iterator[pa.Table]:
        """Buffer `batch`, then yield the sessions the new watermark closes."""
        import pyarrow.compute as pc

        from batcher._internal.errors import PlanError

        column = batch.column(self._plan.time_col) if self._has_time(batch) else None
        if column is None:
            raise PlanError(
                f"session_window(): the event-time column {self._plan.time_col!r} is not in "
                f"the streamed rows ({batch.schema.names})"
            )
        table = pa.Table.from_batches([batch])
        # Drop rows the watermark has already passed -- the watermark as it stood *before*
        # this batch, which is the one whose promise the engine has already acted on. Using
        # the batch's own maximum instead would drop the earlier rows of the very session
        # that maximum belongs to, and Spark advances the watermark at the end of a
        # micro-batch for exactly this reason.
        before = self.watermark
        if before is not None:
            keep = pc.greater_equal(pc.cast(table.column(self._plan.time_col), pa.int64()), before)
            table = table.filter(pc.fill_null(keep, True))
        arrived = _to_micros(column)
        if arrived is not None:
            self._max_seen = arrived if self._max_seen is None else max(self._max_seen, arrived)

        self._buffer = table if self._buffer is None else pa.concat_tables([self._buffer, table])
        self._check_bounded()
        watermark = self.watermark
        if watermark is None or self._buffer.num_rows == 0:
            return
        yield from self._close(watermark)

    def drain(self) -> Iterator[pa.Table]:
        """Close every remaining session, because the stream has ended."""
        if self._buffer is None or self._buffer.num_rows == 0:
            return
        yield self._aggregate(self._buffer)
        self._buffer = None

    @property
    def watermark(self) -> int | None:
        """``max observed event time - lateness``, in epoch microseconds."""
        if self._max_seen is None:
            return None
        return self._max_seen - self._plan.lateness_micros

    def _has_time(self, batch: pa.RecordBatch) -> bool:
        return self._plan.time_col in batch.schema.names

    def _close(self, watermark: int) -> Iterator[pa.Table]:
        """Split the buffer on the watermark, emit the closed part, keep the rest."""
        import pyarrow.compute as pc

        from batcher.api.dataset._build import mark_sessions
        from batcher.api.session import from_arrow

        pk = list(self._plan.partition_by)
        marked = mark_sessions(
            from_arrow(self._buffer), self._plan.time_col, self._plan.gap_micros, pk
        )
        # A session is closed when the watermark is past its last event plus the gap.
        # `_session_end` is that last event, computed per session by the same engine.
        ends = marked.window(
            partition_by=[*pk, "_sid"], order_by=["_t"], functions={"_session_end": ("max", "_t")}
        ).collect()
        # Strictly less, not `<=`. At equality a row arriving exactly on the watermark
        # would be exactly `gap` after the session's last event, which is *not* a new
        # session (the boundary is `> gap`) -- so it would extend a session already
        # emitted. One comparison, and the difference between correct and silently wrong.
        closed_mask = pc.less(pc.add(ends.column("_session_end"), self._plan.gap_micros), watermark)
        closed = ends.filter(closed_mask)
        self._buffer = _strip(ends.filter(pc.invert(closed_mask)))
        if closed.num_rows:
            yield self._aggregate(_strip(closed))

    def _aggregate(self, rows: pa.Table) -> pa.Table:
        """Run the *bounded* session aggregation over rows whose sessions are complete."""
        from batcher.api.dataset._build import sessionize
        from batcher.api.session import from_arrow

        return sessionize(
            from_arrow(rows),
            self._plan.time_col,
            self._plan.gap_micros,
            list(self._plan.partition_by),
            dict(self._plan.aggs),
        ).collect()

    def _check_bounded(self) -> None:
        """Fail loudly when the buffer outgrows the streaming-state budget.

        A session window's state is bounded by the watermark advancing, and the
        watermark advances on event time. A source whose event time stalls — a clock
        that stopped, a key that never goes quiet — never closes a session, and the
        buffer grows without limit. A named error beats an OOM, because the remedy is
        to look at the stalled source rather than at the heap.
        """
        from batcher._internal.errors import ResourceError
        from batcher.config import active_config

        if self._buffer is None:
            return
        budget = active_config().memory.streaming_state_budget_bytes()
        if budget and self._buffer.nbytes > budget:
            raise ResourceError(
                f"session_window(): {self._buffer.nbytes / 1e6:.0f} MB of rows are buffered "
                f"for sessions that never close, over the {budget / 1e6:.0f} MB streaming "
                "state budget. A session closes when the watermark passes its last event "
                f"plus the {self._plan.gap_micros / 1e6:.0f}s gap, so this means event time "
                "is not advancing — check whether the source has stalled, and whether the "
                "gap is larger than the sessions you meant to measure."
            )


def _strip(table: pa.Table) -> pa.Table:
    """Drop the marker columns, so a re-marked buffer does not collide with itself."""
    present = [name for name in _INTERNAL if name in table.schema.names]
    return table.drop_columns(present) if present else table


def _to_micros(column: pa.ChunkedArray | pa.Array) -> int | None:
    """The largest value of a timestamp column, in epoch microseconds."""
    import pyarrow.compute as pc

    largest = pc.max(pc.cast(column, pa.int64()))
    return largest.as_py()
