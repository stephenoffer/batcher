"""A lagging stream partition must not have its rows dropped by a faster one's clock.

The windowed streaming aggregate advances event time from what it reads. When that was one
maximum over the batch, a topic whose partition 0 had reached 10:00 while partition 1 was
still replaying 09:00 declared the whole stream at 10:00 — and then correctly dropped every
partition-1 row as late against a watermark that was wrong. The total for the 09:00 window
came out short, no error was raised, and the only trace was `num_late_inputs_dropped`.

This drives that exact shape through the real driver over a source shaped like a broker
(rows carrying `topic` and `partition`), and asserts the rows survive.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.api.session._scan import _scan

pytestmark = pytest.mark.integration

BASE = dt.datetime(2024, 1, 1, 9, 0)
SCHEMA = pa.schema(
    [
        ("topic", pa.string()),
        ("partition", pa.int64()),
        ("ts", pa.timestamp("us")),
        ("v", pa.int64()),
    ]
)


def _rows(pairs: list[tuple[int, int]]) -> pa.RecordBatch:
    """One batch: `(partition, minutes past 09:00)` pairs, each contributing v=1."""
    return pa.record_batch(
        {
            "topic": pa.array(["events"] * len(pairs), type=pa.string()),
            "partition": pa.array([p for p, _ in pairs], type=pa.int64()),
            "ts": pa.array(
                [BASE + dt.timedelta(minutes=m) for _, m in pairs], type=pa.timestamp("us")
            ),
            "v": pa.array([1] * len(pairs), type=pa.int64()),
        },
        schema=SCHEMA,
    )


class _PartitionedStream:
    """A stream that says which partition each row came from, as a broker does.

    Deliberately declares exactly the contract `io.source.watermark_partition_columns` reads
    rather than subclassing `BrokerSource`: what is under test is the driver's use of the
    declaration, not a particular broker client.
    """

    format_name = "test-partitioned"
    bounded = False
    continues_across_passes = True
    watermark_partition_columns = ("topic", "partition")

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches

    def schema(self) -> pa.Schema:
        return SCHEMA

    def row_count(self) -> int | None:
        return None

    def iter_batches(self, projection: list[str] | None = None):
        for batch in self._batches:
            yield batch.select(projection) if projection is not None else batch

    def read(self, projection: list[str] | None = None):  # pragma: no cover - never materialized
        raise AssertionError("an unbounded stream must not be materialized")


def _windowed(ds):
    return (
        ds.with_watermark("ts", "1 minute")
        .group_by(w=bt.window(col("ts"), "1 hour"))
        .agg(total=col("v").sum())
    )


def _totals(ds) -> dict:
    out = {}
    for batch in ds.iter_batches():
        for row in batch.to_pylist():
            out[row["w"]] = out.get(row["w"], 0) + row["total"]
    return out


def test_a_lagging_partitions_rows_are_not_dropped_as_late() -> None:
    """Partition 1 delivers 09:xx rows after partition 0 has already reached 11:xx."""
    stream = _PartitionedStream(
        [
            # Both partitions start together in the 09:00 window.
            _rows([(0, 5), (1, 5)]),
            # Partition 0 races ahead by two hours; partition 1 has said nothing.
            _rows([(0, 125), (0, 130)]),
            # Partition 1 finally delivers the rest of its 09:00 window.
            _rows([(1, 10), (1, 20), (1, 30)]),
        ]
    )
    totals = _totals(_windowed(_scan(stream)))
    assert totals[BASE] == 5, (
        f"the 09:00 window totalled {totals.get(BASE)} of 5 rows — the lagging partition's "
        "rows were dropped against the fast partition's watermark"
    )
    assert totals[BASE + dt.timedelta(hours=2)] == 2


def test_the_late_counter_stays_at_zero_for_a_merely_lagging_partition() -> None:
    """`num_late_inputs_dropped` was the only symptom; it must now report nothing dropped."""
    from batcher.core.streaming import _window_key, _WindowedAggFold

    plan = _windowed(_scan(_PartitionedStream([])))._plan
    fold = _WindowedAggFold(plan, _window_key(plan), partition_cols=("topic", "partition"))
    fold.push(_rows([(0, 5), (1, 5)]))
    fold.push(_rows([(0, 125), (0, 130)]))
    fold.push(_rows([(1, 10), (1, 20), (1, 30)]))
    assert fold.metrics().num_late_inputs_dropped == 0


def test_a_genuinely_late_row_is_still_dropped() -> None:
    """The fix must not turn the watermark off: once every partition has passed a window,
    a row arriving inside it is late and stays late."""
    from batcher.core.streaming import _window_key, _WindowedAggFold

    plan = _windowed(_scan(_PartitionedStream([])))._plan
    fold = _WindowedAggFold(plan, _window_key(plan), partition_cols=("topic", "partition"))
    fold.push(_rows([(0, 200), (1, 200)]))  # both partitions well past the 09:00 window
    fold.push(_rows([(1, 5)]))
    assert fold.metrics().num_late_inputs_dropped == 1


def test_an_unpartitioned_stream_behaves_exactly_as_before() -> None:
    """A source that cannot attribute rows to partitions keeps the maximum it always had."""
    from batcher.core.streaming import _window_key, _WindowedAggFold

    plan = _windowed(_scan(_PartitionedStream([])))._plan
    fold = _WindowedAggFold(plan, _window_key(plan))  # no partition columns
    fold.push(_rows([(0, 5), (1, 5)]))
    fold.push(_rows([(0, 125), (0, 130)]))
    fold.push(_rows([(1, 10), (1, 20), (1, 30)]))
    assert fold.metrics().num_late_inputs_dropped == 3
