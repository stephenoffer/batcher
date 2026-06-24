"""Development streaming sources — `rate` and `socket` (Spark parity).

These are the test/dev sources Spark ships: `rate` generates a steady stream of
``(timestamp, value)`` rows for benchmarking and demos, and `socket` reads
newline-delimited text from a TCP connection. Both are unbounded (``bounded =
False``); `rate` accepts an optional `num_rows` cap so it can also drive a bounded
``available_now`` run or a test.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["RateSource", "SocketSource"]

_EPOCH = datetime.datetime(1970, 1, 1)


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
        self._rps = rows_per_second
        self._num_rows = num_rows
        self._start = start_value
        self._pace = pace and num_rows is None
        # The next `value` to emit — advances as batches are produced, so a streaming
        # query can checkpoint it (`snapshot_position`) and resume (`seek`).
        self._cursor = start_value

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
        values = list(range(first_value, first_value + n))
        step_us = 1_000_000 // self._rps
        timestamps = [_EPOCH + datetime.timedelta(microseconds=v * step_us) for v in values]
        return pa.record_batch(
            {
                "timestamp": pa.array(timestamps, type=pa.timestamp("us")),
                "value": pa.array(values, type=pa.int64()),
            }
        )

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        import time

        # `value` is the absolute row counter; `num_rows` caps it. Resuming after a
        # `seek` continues from the recorded value (no rows replayed or skipped).
        value = self._start
        while self._num_rows is None or value < self._num_rows:
            n = self._rps
            if self._num_rows is not None:
                n = min(n, self._num_rows - value)
            batch = self._make_batch(value, n)
            value += n
            self._cursor = value
            yield batch.select(projection) if projection is not None else batch
            if self._pace:
                time.sleep(1.0)

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
        self, host: str = "localhost", port: int = 9999, *, batch_size: int = 1024, **_: Any
    ) -> None:
        self._host = host
        self._port = port
        self._batch_size = batch_size

    def schema(self) -> pa.Schema:
        return pa.schema([("value", pa.string()), ("timestamp", pa.timestamp("us"))])

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"socket:{self._host}:{self._port}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        import socket

        with socket.create_connection((self._host, self._port)) as conn:
            buf = b""
            with conn.makefile("rb") as fh:
                lines: list[str] = []
                for raw in fh:
                    lines.append((buf + raw).decode("utf-8", "replace").rstrip("\n"))
                    buf = b""
                    if len(lines) >= self._batch_size:
                        yield self._batch(lines, projection)
                        lines = []
                if lines:
                    yield self._batch(lines, projection)

    def _batch(self, lines: list[str], projection: list[str] | None) -> pa.RecordBatch:
        now = datetime.datetime.now()
        batch = pa.record_batch(
            {
                "value": pa.array(lines, type=pa.string()),
                "timestamp": pa.array([now] * len(lines), type=pa.timestamp("us")),
            }
        )
        return batch.select(projection) if projection is not None else batch

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read until the connection closes (the bounded-test convenience)."""
        return list(self.iter_batches(projection))
