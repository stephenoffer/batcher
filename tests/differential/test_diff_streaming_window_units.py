"""Windowed streaming aggregation is unit-safe across timestamp resolutions.

A watermarked tumbling-window aggregate streamed via `iter_batches` must equal the
batch aggregate (and DuckDB) whatever the event-time column's resolution — second,
millisecond, microsecond, or nanosecond. The streaming driver keeps its watermark
and window bounds in microseconds (the engine emits `window_start` as `timestamp[us]`),
so it must normalize a non-`us` event-time column before comparing against them.

Regression: reading the raw int64 ticks of a `timestamp[ns]` column as microseconds
scaled the watermark by 1000x (overflowing the literal → crash), and a `timestamp[ms]`
/ `timestamp[s]` column raised an engine unit-mismatch on the late-drop comparison —
so every windowed stream over anything but microsecond timestamps was broken.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal
from batcher import col, count, window


def _chunks(unit: str, n_chunks: int = 5, per: int = 40) -> list[pa.RecordBatch]:
    base = dt.datetime(2024, 1, 1)
    batches = []
    for c in range(n_chunks):
        start = c * per
        ts = [base + dt.timedelta(seconds=start + i) for i in range(per)]
        vs = [(start + i) % 7 for i in range(per)]
        batches.append(
            pa.record_batch({"ts": pa.array(ts, pa.timestamp(unit)), "v": pa.array(vs, pa.int64())})
        )
    return batches


def _stream(batches: list[pa.RecordBatch]):
    return bt.from_batches(lambda: iter(batches), batches[0].schema)


def _streamed(ds) -> pa.Table:
    parts = list(ds.iter_batches())
    return pa.Table.from_batches(parts) if parts else pa.table({})


def _windowed(ds):
    return (
        ds.with_watermark("ts", "5s")
        .group_by(w=window(col("ts"), "10s"))
        .agg(s=col("v").sum(), n=count())
    )


@pytest.mark.differential
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_windowed_stream_matches_batch_across_units(unit: str) -> None:
    batches = _chunks(unit)
    full = pa.Table.from_batches(batches)

    streamed = _streamed(_windowed(_stream(batches)))
    # Batch oracle: the same tumbling window with no watermark (event time is in order
    # and the 5s lateness never drops a row), so streaming must equal batch exactly.
    batch = (
        bt.from_arrow(full)
        .group_by(w=window(col("ts"), "10s"))
        .agg(s=col("v").sum(), n=count())
        .collect()
    )
    assert_tables_equal(streamed, batch)


@pytest.mark.differential
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_windowed_stream_matches_duckdb_across_units(unit: str, duck) -> None:
    batches = _chunks(unit)
    full = pa.Table.from_batches(batches)
    duck.register("t", full)

    streamed = _streamed(_windowed(_stream(batches)))
    expected = duck.sql(
        "SELECT time_bucket(INTERVAL '10 seconds', ts) w, SUM(v) s, COUNT(*) n FROM t GROUP BY 1"
    )
    assert_same(streamed, expected)
