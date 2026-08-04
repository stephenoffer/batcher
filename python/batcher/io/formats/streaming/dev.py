"""Development streaming sources — `rate`, `rate_micro_batch`, and `socket` (Spark parity).

The test/dev sources Spark ships. `rate` generates a steady stream of
``(timestamp, value)`` rows for benchmarking and demos; `rate_micro_batch` generates a
*fixed number of rows per micro-batch* rather than per second, which is what makes a
benchmark reproducible; `socket` reads newline-delimited text from a TCP connection. All
are unbounded (``bounded = False``); `rate` and `rate_micro_batch` accept a `num_rows` cap
so they can also drive a bounded ``available_now`` run or a test.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["RateMicroBatchSource", "RateSource", "SocketSource"]

#: Naive UTC epoch, for turning a wall-clock read into microseconds without a per-row object.
_EPOCH = datetime.datetime(1970, 1, 1)

#: The `timestamp` column is `timestamp[us]`, so a rate above one row per microsecond has no
#: representable spacing: `1_000_000 // rows_per_second` floors to zero and *every* row gets
#: the epoch. That is not a slow stream, it is a silently wrong one — a windowed aggregate
#: over it puts the whole run in one bucket and a watermark never advances.
_MAX_ROWS_PER_SECOND = 1_000_000


@SOURCES.register("rate")
class RateSource:
    """Generate ``(timestamp, value)`` rows at `rows_per_second` (Spark `rate`).

    `value` counts up from `start_value`; `timestamp` is spaced by
    ``1/rows_per_second`` seconds from the Unix epoch (deterministic, not a
    wall-clock read, so a generated stream is reproducible). `num_rows` bounds the
    total (``None`` = unbounded). `pace=True` sleeps one second between full batches
    so a `processing_time` trigger sees a realistic cadence; tests pass ``pace=False``.
    """

    bounded = False
    #: Deliberately *not* `continues_across_passes`: `iter_batches` restarts at
    #: `_start`, so a fresh generator replays the same values. The stream is expressed
    #: as one never-ending generator instead (or a `num_rows` cap that ends it), which is
    #: why re-opening it is never the right move. See `io.source.continues_across_passes`.

    def __init__(
        self,
        rows_per_second: int = 1,
        *,
        num_rows: int | None = None,
        start_value: int = 0,
        pace: bool = True,
        **_: Any,
    ) -> None:
        if rows_per_second < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(f"rate source rows_per_second must be >= 1, got {rows_per_second}")
        if rows_per_second > _MAX_ROWS_PER_SECOND:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"rate source rows_per_second must be <= {_MAX_ROWS_PER_SECOND}, got "
                f"{rows_per_second}: the timestamp column is microsecond-resolution, so a "
                "faster rate would give every row the same timestamp"
            )
        self._rps = rows_per_second
        self._num_rows = num_rows
        self._start = start_value
        self._pace = pace and num_rows is None
        # The next `value` to emit — advances as batches are produced, so a streaming
        # query can checkpoint it (`snapshot_position`) and resume (`seek`).
        self._cursor = start_value
        self._should_stop: Any = None

    def set_stop_signal(self, should_stop: Any) -> None:
        """Register a predicate that ends the generation loop between batches.

        A paced rate source sleeps a full second between batches, so a `stop()` on a query
        reading one waited out that sleep *and* then the rest of the loop before the driver
        thread could be joined. The predicate is checked before each batch and, when the
        source is pacing, in short slices *during* the sleep — so a stop is observed in
        milliseconds instead of at the next tick.

        Args:
            should_stop: A zero-argument predicate that becomes true when the stream should
                end. ``None`` clears it.
        """
        self._should_stop = should_stop

    def _nap(self, seconds: float) -> bool:
        """Sleep in slices, returning True if a stop was signalled part-way through."""
        import time

        if self._should_stop is None:
            time.sleep(seconds)
            return False
        deadline = time.monotonic() + seconds
        while True:
            if self._should_stop():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))

    def snapshot_position(self) -> dict:
        """The next `value` to emit (for exactly-once checkpoint/resume)."""
        return {"value": self._cursor}

    def seek(self, position: dict) -> None:
        """Resume generation from a previously snapshotted position."""
        self._start = int(position["value"])
        self._cursor = self._start

    def schema(self) -> pa.Schema:
        return pa.schema([("timestamp", pa.timestamp("us")), ("value", pa.int64())])

    def row_count(self) -> int | None:
        return self._num_rows

    def identity(self) -> str:
        return f"rate:{self._rps}:{self._num_rows}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]

    def _make_batch(self, first_value: int, n: int) -> pa.RecordBatch:
        import numpy as np
        import pyarrow.compute as pc

        # Vectorized, not a Python list comprehension: a rate source emits `rps` rows per
        # batch, and building `n` `datetime` objects per batch was O(rows) Python in the data
        # plane. `value` counts up from `first_value`; `timestamp` is `value * step_us`
        # microseconds since the epoch, computed as int64 and reinterpreted as `timestamp[us]`.
        step_us = 1_000_000 // self._rps
        values = pa.array(np.arange(first_value, first_value + n, dtype=np.int64))
        timestamps = pc.multiply(values, np.int64(step_us)).cast(pa.timestamp("us"))
        return pa.record_batch({"timestamp": timestamps, "value": values})

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        # `value` is the absolute row counter; `num_rows` caps it. Resuming after a
        # `seek` continues from the recorded value (no rows replayed or skipped).
        value = self._start
        while self._num_rows is None or value < self._num_rows:
            if self._should_stop is not None and self._should_stop():
                return
            n = self._rps
            if self._num_rows is not None:
                n = min(n, self._num_rows - value)
            batch = self._make_batch(value, n)
            value += n
            self._cursor = value
            yield batch.select(projection) if projection is not None else batch
            if self._pace and self._nap(1.0):
                return

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Materialize — only valid when `num_rows` bounds the stream."""
        if self._num_rows is None:
            from batcher._internal.errors import PlanError

            raise PlanError("rate source is unbounded; set num_rows to read(), or use iter_batches")
        return list(self.iter_batches(projection))


@SOURCES.register("socket")
class SocketSource:
    """Read newline-delimited text from a TCP socket (Spark `socket`).

    Connects to ``host:port`` and yields one ``value: string`` column (plus a
    `timestamp` of receipt). Unbounded; the connection closing ends the stream. For
    development only — there is no replay, so it is at-most-once.
    """

    bounded = False

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9999,
        *,
        batch_size: int = 1024,
        include_timestamp: bool = True,
        **_: Any,
    ) -> None:
        self._host = host
        self._port = port
        self._batch_size = batch_size
        # Spark's `includeTimestamp`, whose default is the opposite of this one. Ours stays
        # True because the column has always been here and dropping it silently would break
        # every existing reader; the option exists so a ported job can ask for Spark's
        # one-column shape rather than discovering an extra column at runtime.
        self._include_timestamp = include_timestamp

    def schema(self) -> pa.Schema:
        fields = [("value", pa.string())]
        if self._include_timestamp:
            fields.append(("timestamp", pa.timestamp("us")))
        return pa.schema(fields)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"socket:{self._host}:{self._port}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        import socket

        with (
            socket.create_connection((self._host, self._port)) as conn,
            conn.makefile("rb") as fh,
        ):
            lines: list[str] = []
            for raw in fh:
                # `rstrip("\r\n")` strips a CRLF terminator, not just `\n`: a Windows
                # or HTTP-style producer sends `\r\n`, and stripping only `\n` left a
                # trailing `\r` in every `value`.
                lines.append(raw.decode("utf-8", "replace").rstrip("\r\n"))
                if len(lines) >= self._batch_size:
                    yield self._batch(lines, projection)
                    lines = []
            if lines:
                yield self._batch(lines, projection)

    def _batch(self, lines: list[str], projection: list[str] | None) -> pa.RecordBatch:
        import numpy as np

        # UTC, not local wall-clock. The `timestamp` column carries no zone, and every other
        # streaming source stamps UTC into that convention — so a local-time socket stream
        # sat a whole UTC offset away from a Kafka stream it was joined or watermarked
        # against, which reads as "no matches" rather than as a timezone mistake.
        now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        micros = (now - _EPOCH).days * 86_400_000_000 + (now - _EPOCH).seconds * 1_000_000
        micros += (now - _EPOCH).microseconds
        # Broadcast the scalar through numpy rather than replicating a `datetime` object per
        # row: `[now] * len(lines)` is O(rows) Python object handling in the data plane, the
        # same shape `RateSource._make_batch` was vectorized out of.
        stamps = pa.array(np.full(len(lines), micros, dtype=np.int64)).cast(pa.timestamp("us"))
        columns: dict[str, Any] = {"value": pa.array(lines, type=pa.string())}
        if self._include_timestamp:
            columns["timestamp"] = stamps
        batch = pa.record_batch(columns)
        return batch.select(projection) if projection is not None else batch

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read until the connection closes (the bounded-test convenience)."""
        return list(self.iter_batches(projection))


@SOURCES.register("rate_micro_batch")
class RateMicroBatchSource(RateSource):
    """Generate exactly `rows_per_batch` rows per micro-batch (Spark ``rate-micro-batch``).

    The difference from `rate` is the unit, and it is the whole point. `rate` promises rows
    per *second*, so how many land in a micro-batch depends on how long the previous one
    took — which makes it useless as a benchmark input, because the thing being measured
    changes the input. `rate_micro_batch` promises rows per *batch*, so a run is
    reproducible and a comparison between two engine builds is a comparison.

    Spark added it in 3.5 for exactly that reason. `start_timestamp` and
    `advance_ms_per_batch` shape the event-time column so a windowed query over it is
    deterministic too: batch *k*'s rows are stamped ``start_timestamp + k *
    advance_ms_per_batch``, so a fixed number of batches always closes the same windows.

    Args:
        rows_per_batch: Rows in every micro-batch.
        num_rows: Total rows before the stream ends; None is unbounded.
        start_timestamp: Milliseconds since the epoch for the first batch.
        advance_ms_per_batch: How far event time moves between batches.
        pace: Sleep a second between batches, as `rate` does. Off by default here —
            a source whose point is a reproducible batch size is usually being read as
            fast as possible.
    """

    def __init__(
        self,
        rows_per_batch: int = 1,
        *,
        num_rows: int | None = None,
        start_value: int = 0,
        start_timestamp: int = 0,
        advance_ms_per_batch: int = 1000,
        pace: bool = False,
        **options: Any,
    ) -> None:
        if rows_per_batch < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(f"rate_micro_batch rows_per_batch must be >= 1, got {rows_per_batch}")
        if advance_ms_per_batch < 0:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"rate_micro_batch advance_ms_per_batch must be >= 0, got "
                f"{advance_ms_per_batch}: event time does not run backwards"
            )
        # `rows_per_second` is the parent's batch size, which is exactly the rows-per-batch
        # promise under a different name — so the generation loop is inherited unchanged and
        # only the timestamp derivation differs.
        super().__init__(
            rows_per_batch,
            num_rows=num_rows,
            start_value=start_value,
            pace=pace,
            **options,
        )
        self._start_timestamp = start_timestamp
        self._advance = advance_ms_per_batch

    def identity(self) -> str:
        return f"rate_micro_batch:{self._rps}:{self._num_rows}:{self._advance}"

    def _make_batch(self, first_value: int, n: int) -> pa.RecordBatch:
        """Rows stamped by *batch index*, so a fixed batch count closes fixed windows."""
        import numpy as np

        batch_index = first_value // self._rps
        stamp_us = (self._start_timestamp + batch_index * self._advance) * 1000
        values = pa.array(np.arange(first_value, first_value + n, dtype=np.int64))
        timestamps = pa.array(np.full(n, stamp_us, dtype=np.int64)).cast(pa.timestamp("us"))
        return pa.record_batch({"timestamp": timestamps, "value": values})

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Materialize — only valid when `num_rows` bounds the stream."""
        if self._num_rows is None:
            from batcher._internal.errors import PlanError

            raise PlanError(
                "rate_micro_batch is unbounded; set num_rows to read(), or use iter_batches"
            )
        return list(self.iter_batches(projection))
