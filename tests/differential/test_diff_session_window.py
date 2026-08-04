"""Workstream C — session windows match DuckDB.

A session groups consecutive per-key events whose gap is below the timeout; a larger
gap starts a new session. Batcher composes this from the window engine (lag + a
running-sum session id) + group-by — no new operator. DuckDB computes the same
session id with the identical SQL window formulation as the independent oracle.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

_BASE = dt.datetime(2024, 1, 1, 0, 0, 0)


def _data() -> pa.Table:
    # Two keys with several sessions each; 10-minute gap boundary.
    recs = [
        ("a", 0, 1),
        ("a", 2, 2),
        ("a", 5, 3),  # session 1 (gaps ≤ 10m)
        ("a", 40, 4),
        ("a", 45, 5),  # session 2
        ("b", 0, 6),  # session 1 (lone)
        ("b", 30, 7),
        ("b", 31, 8),  # session 2
    ]
    ts = [_BASE + dt.timedelta(minutes=m) for _, m, _ in recs]
    return pa.table(
        {
            "k": pa.array([k for k, _, _ in recs], pa.string()),
            "ts": pa.array(ts, pa.timestamp("us")),
            "v": pa.array([v for _, _, v in recs], pa.int64()),
        }
    )


@pytest.mark.parametrize("gap", ["10m", "1m"])
def test_session_window_matches_duckdb(duck, gap):
    tbl = _data()
    got = (
        bt.from_arrow(tbl)
        .session_window("ts", gap, partition_by=["k"], total=col("v").sum(), n=col("v").count())
        .collect()
    )

    gap_secs = {"10m": 600, "1m": 60}[gap]
    duck.register("t", tbl)
    # Independent oracle: the same session-id formulation in SQL (lag → new-session
    # marker → running sum), then group by (k, session_id).
    rel = duck.sql(
        f"""
        WITH marked AS (
            SELECT *,
                CASE WHEN epoch_us(ts) - lag(epoch_us(ts)) OVER w > {gap_secs} * 1000000
                          OR lag(epoch_us(ts)) OVER w IS NULL
                     THEN 1 ELSE 0 END AS new_session
            FROM t WINDOW w AS (PARTITION BY k ORDER BY ts)
        ),
        sessioned AS (
            SELECT *, sum(new_session) OVER (PARTITION BY k ORDER BY ts) AS sid FROM marked
        )
        SELECT k, min(ts) AS session_start, max(ts) AS session_end,
               sum(v) AS total, count(v) AS n
        FROM sessioned GROUP BY k, sid
        """
    )
    assert_same(got, rel)


def _feed(tbl: pa.Table, chunks: int):
    """The same rows as an unbounded source, delivered `chunks` micro-batches at a time."""

    def source():
        rows = tbl.num_rows
        size = max(1, -(-rows // chunks))
        for start in range(0, rows, size):
            yield tbl.slice(start, size).to_batches()[0]

    return bt.from_batches(source, tbl.schema, bounded=False)


@pytest.mark.parametrize("chunks", [1, 3, 8])
def test_a_streaming_session_window_matches_duckdb_too(duck, chunks):
    """The streaming operator holds rows until the watermark says a session cannot grow.
    Whether it held the right ones is only visible against an oracle that saw them all at
    once -- and `chunks` varies where the micro-batch boundaries fall, because a boundary
    inside a session is exactly what the operator exists to survive."""
    tbl = _data().sort_by([("ts", "ascending")])  # a stream arrives in event-time order
    got = []
    stream = _feed(tbl, chunks).session_window(
        "ts", "10m", partition_by=["k"], total=col("v").sum(), n=col("v").count()
    )
    for batch in stream.iter_batches():
        got.append(batch)
    result = pa.Table.from_batches(got)

    duck.register("t", tbl)
    rel = duck.sql(
        """
        WITH marked AS (
            SELECT *,
                CASE WHEN epoch_us(ts) - lag(epoch_us(ts)) OVER w > 600 * 1000000
                          OR lag(epoch_us(ts)) OVER w IS NULL
                     THEN 1 ELSE 0 END AS new_session
            FROM t WINDOW w AS (PARTITION BY k ORDER BY ts)
        ),
        sessioned AS (
            SELECT *, sum(new_session) OVER (PARTITION BY k ORDER BY ts) AS sid FROM marked
        )
        SELECT k, min(ts) AS session_start, max(ts) AS session_end,
               sum(v) AS total, count(v) AS n
        FROM sessioned GROUP BY k, sid
        """
    )
    assert_same(result, rel)


@pytest.mark.parametrize(
    ("arrow_type", "expected"),
    [
        (pa.timestamp("us", tz="UTC"), "timestamp[us, tz=UTC]"),
        (pa.timestamp("ms"), "timestamp[ms]"),
        (pa.timestamp("us"), "timestamp[us]"),
    ],
)
def test_the_session_bounds_keep_the_event_time_columns_own_type(arrow_type, expected):
    """The bounds used to be computed from an epoch-micros copy and cast back, which
    produced a naive `timestamp[us]` whatever went in. A UTC-aware column came back with
    the right instant and no timezone, and a millisecond column came back in microseconds
    -- right values, wrong type, which is the shape of bug an order-independent value
    comparison cannot see and anything rendering a local time downstream reads as wrong."""
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc if arrow_type.tz else None)
    table = pa.table(
        {
            "k": pa.array(["a", "a"], pa.string()),
            "ts": pa.array([base, base + dt.timedelta(minutes=1)], arrow_type),
            "v": pa.array([1, 2], pa.int64()),
        }
    )
    out = bt.from_arrow(table).session_window("ts", "30m", partition_by=["k"], total=col("v").sum())
    result = out.collect()
    assert str(result.schema.field("session_start").type) == expected
    assert str(result.schema.field("session_end").type) == expected
