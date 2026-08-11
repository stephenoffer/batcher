"""A sliding windowed streaming aggregation is a windowed aggregation, not a running one.

`window(ts, width, slide)` puts a row in several overlapping windows, so `group_by` refuses
it and the caller explodes it first. That left the aggregate's group key an ordinary column
with the geometry two nodes below it, and the streaming driver — which recognized only a
`window_start` key — fell through to the *unwatermarked* running aggregate. Nothing was
dropped as late, nothing was evicted, and nothing was emitted until a source that never ends
ended. The watermark was configured and none of it ran.

These tests pin the three things that were missing: the result equals batch and DuckDB, the
state is actually released as the watermark advances, and the shape the driver still cannot
window is refused rather than silently degraded.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

BASE = dt.datetime(2024, 1, 1, 0, 0)
SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])


def _batches(count: int = 12, per: int = 5) -> list[pa.RecordBatch]:
    """Rows marching forward in event time, several micro-batches' worth."""
    out = []
    for c in range(count):
        stamps = [BASE + dt.timedelta(minutes=10 * (c * per + i)) for i in range(per)]
        out.append(
            pa.record_batch(
                {"ts": pa.array(stamps, type=pa.timestamp("us")), "v": pa.array(range(per))},
                schema=SCHEMA,
            )
        )
    return out


def _stream(batches: list[pa.RecordBatch], *, bounded: bool = True):
    """A genuine streaming source. `bounded=False` is one the engine may not materialize."""
    return bt.from_batches(lambda: iter(batches), SCHEMA, bounded=bounded)


def _sliding(ds, *, watermark: bool = True):
    """`window(ts, '1h', '30m')` exploded, then grouped — the documented spelling."""
    source = ds.with_watermark("ts", "10 minutes") if watermark else ds
    return (
        source.select(w=bt.window(col("ts"), "1 hour", "30 minutes"), v=col("v"))
        .explode("w")
        .group_by("w")
        .agg(s=col("v").sum(), n=col("v").count())
    )


def _rows(table: pa.Table) -> set:
    return {tuple(row.values()) for row in table.to_pylist()}


def _streamed(ds) -> pa.Table:
    parts = list(ds.iter_batches())
    return pa.Table.from_batches(parts) if parts else pa.table({})


def test_a_sliding_windowed_stream_matches_batch_and_duckdb(duck) -> None:
    """The rows are the same however the query is run; only the memory bound differs."""
    batches = _batches()
    full = pa.Table.from_batches(batches)
    duck.register("t", full)

    streamed = _streamed(_sliding(_stream(batches)))
    batched = _sliding(bt.from_arrow(full), watermark=False).collect()
    # DuckDB has no sliding-window helper, so the overlap is generated explicitly: each row
    # belongs to the window starting on its own hour boundary and to the one a half-hour
    # before, which is exactly what a 1h window hopping every 30m means.
    expected = duck.sql(
        """
        WITH exploded AS (
            SELECT time_bucket(INTERVAL '30 minutes', ts) - INTERVAL '30 minutes' * o AS w, v
            FROM t, (SELECT 0 AS o UNION ALL SELECT 1) hops
        )
        SELECT w, SUM(v) s, COUNT(v) n FROM exploded GROUP BY w
        """
    ).to_arrow_table()

    assert _rows(streamed) == _rows(batched)
    assert _rows(streamed) == _rows(expected)


def test_the_sliding_stream_actually_evicts_rather_than_accumulating() -> None:
    """The distinguishing behavior: a running aggregate emits once, at the very end.

    Under the fallback this query produced exactly one batch — the whole result, after the
    source ended. A windowed aggregation emits as the watermark closes each window, which is
    both what makes the state bounded and what makes it usable on a stream that never ends.
    """
    emitted = list(_sliding(_stream(_batches())).iter_batches())
    assert len(emitted) > 1, "the query emitted once at the end — it never windowed"


def test_the_state_stays_bounded_as_the_stream_runs() -> None:
    """Open windows, not every window the stream ever saw."""
    from batcher.core.streaming import _window_key, _WindowedAggFold

    plan = _sliding(_stream(_batches()))._plan
    fold = _WindowedAggFold(plan, _window_key(plan))
    peak = 0
    for batch in _batches(count=40):
        fold.push(batch)
        peak = max(peak, fold._fold.state().num_rows if fold._fold.state() is not None else 0)
    # A 1h window hopping every 30m with a 10m watermark holds a handful of open windows;
    # forty batches of five rows each span two hundred ten-minute steps, so an accumulating
    # fold would be holding hundreds of groups by now.
    assert peak <= 12, f"the fold retained {peak} windows — eviction is not running"


def test_an_unwindowed_watermarked_aggregate_over_a_stream_is_refused() -> None:
    """The other half of the silent fallback: a watermark with nothing to close.

    Over a bounded source the same plan is an ordinary aggregate and still runs; over an
    unbounded one it emits nothing, ever, while growing — so it names the problem instead.
    """
    ds = _stream(_batches(), bounded=False).with_watermark("ts", "10 minutes")
    with pytest.raises(PlanError, match="none of which is an event-time window"):
        list(ds.group_by("v").agg(s=col("v").sum()).iter_batches())


def test_the_same_plan_over_a_bounded_source_still_runs() -> None:
    """A bounded source ends, so the running aggregate terminates and is correct."""
    full = pa.Table.from_batches(_batches())
    ds = bt.from_arrow(full).with_watermark("ts", "10 minutes")
    result = _streamed(ds.group_by("v").agg(s=col("v").sum()))
    assert result.num_rows == 5
