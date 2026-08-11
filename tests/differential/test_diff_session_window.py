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
    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC if arrow_type.tz else None)
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


def test_a_null_event_time_lands_where_duckdb_puts_it(duck):
    """A row with no event time has no place in a gap-based session, so where it goes is
    decided by how nulls sort in the ordering both engines use. They agree -- the null row
    joins the session that follows it -- and this pins that rather than leaving the one
    input a session window has no answer for untested."""
    base = _BASE
    tbl = pa.table(
        {
            "k": pa.array(["a", "a", "a"], pa.string()),
            "ts": pa.array([base, None, base + dt.timedelta(hours=4)], pa.timestamp("us")),
            "v": pa.array([1, 7, 2], pa.int64()),
        }
    )
    got = (
        bt.from_arrow(tbl)
        .session_window("ts", "30m", partition_by=["k"], total=col("v").sum())
        .collect()
    )
    duck.register("t", tbl)
    rel = duck.sql(
        """
        WITH marked AS (
            SELECT *,
                CASE WHEN epoch_us(ts) - lag(epoch_us(ts)) OVER w > 1800 * 1000000
                          OR lag(epoch_us(ts)) OVER w IS NULL
                     THEN 1 ELSE 0 END AS new_session
            FROM t WINDOW w AS (PARTITION BY k ORDER BY ts)
        ),
        sessioned AS (
            SELECT *, sum(new_session) OVER (PARTITION BY k ORDER BY ts) AS sid FROM marked
        )
        SELECT k, min(ts) AS session_start, max(ts) AS session_end, sum(v) AS total
        FROM sessioned GROUP BY k, sid
        """
    )
    assert_same(got, rel)


def test_the_streaming_operator_puts_the_null_row_in_the_same_place():
    """Whatever the answer is, both paths must give it. A null event time cannot advance a
    watermark either, which is the part only the streaming path has to get right."""
    base = _BASE
    schema = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])

    def feed():
        yield pa.record_batch({"k": ["a", "a"], "ts": [base, None], "v": [1, 7]}, schema=schema)
        yield pa.record_batch(
            {"k": ["a"], "ts": [base + dt.timedelta(hours=4)], "v": [2]}, schema=schema
        )

    streamed = []
    plan = bt.from_batches(feed, schema, bounded=False).session_window(
        "ts", "30m", partition_by=["k"], total=col("v").sum()
    )
    for batch in plan.iter_batches():
        streamed.extend(batch.to_pylist())
    bounded = (
        bt.from_pydict(
            {"k": ["a", "a", "a"], "ts": [base, None, base + dt.timedelta(hours=4)], "v": [1, 7, 2]}
        )
        .session_window("ts", "30m", partition_by=["k"], total=col("v").sum())
        .to_pylist()
    )
    assert _by_start(streamed) == _by_start(bounded)


def _by_start(rows: list[dict]) -> list[tuple]:
    """Sessions keyed by start, so the comparison does not depend on emission order."""
    return sorted((str(row["session_start"]), row["total"]) for row in rows)


def test_a_date_event_time_column_sessionizes_by_the_gap_it_was_given(duck):
    """A `date32` column cast straight to int64 counts *days*, and the gap is expressed in
    microseconds -- so every gap looked smaller than every threshold and a key's whole
    history collapsed into one session. Silently: the answer was one plausible row per key.
    Sessionizing an order or visit date by "3 days" is an ordinary analytic query, so the
    column type had to stop deciding the answer."""
    tbl = pa.table(
        {
            "k": pa.array(["a", "a", "a", "b"], pa.string()),
            "d": pa.array(
                [
                    dt.date(2024, 1, 1),
                    dt.date(2024, 1, 2),
                    dt.date(2024, 6, 1),
                    dt.date(2024, 1, 1),
                ],
                pa.date32(),
            ),
            "v": pa.array([1, 2, 3, 4], pa.int64()),
        }
    )
    got = (
        bt.from_arrow(tbl)
        .session_window("d", "2d", partition_by=["k"], total=col("v").sum())
        .collect()
    )
    assert str(got.schema.field("session_start").type) == "date32[day]"

    duck.register("t", tbl)
    rel = duck.sql(
        """
        WITH marked AS (
            SELECT *,
                CASE WHEN epoch_us(d::TIMESTAMP) - lag(epoch_us(d::TIMESTAMP)) OVER w > 172800000000
                          OR lag(epoch_us(d::TIMESTAMP)) OVER w IS NULL
                     THEN 1 ELSE 0 END AS new_session
            FROM t WINDOW w AS (PARTITION BY k ORDER BY d)
        ),
        sessioned AS (
            SELECT *, sum(new_session) OVER (PARTITION BY k ORDER BY d) AS sid FROM marked
        )
        SELECT k, min(d) AS session_start, max(d) AS session_end, sum(v) AS total
        FROM sessioned GROUP BY k, sid
        """
    )
    assert_same(got, rel)


def test_the_streaming_operator_takes_a_date_event_time_column_too():
    """It raised `ArrowNotImplementedError: Unsupported cast from date32[day] to int64` --
    the same mistake as the bounded path's, arriving as a crash instead of a wrong
    answer."""
    schema = pa.schema([("k", pa.string()), ("d", pa.date32()), ("v", pa.int64())])

    def feed():
        yield pa.record_batch(
            {"k": ["a", "a"], "d": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)], "v": [1, 2]},
            schema=schema,
        )
        yield pa.record_batch({"k": ["a"], "d": [dt.date(2024, 6, 1)], "v": [3]}, schema=schema)

    streamed = []
    plan = bt.from_batches(feed, schema, bounded=False).session_window(
        "d", "2d", partition_by=["k"], total=col("v").sum()
    )
    for batch in plan.iter_batches():
        streamed.extend(batch.to_pylist())
    assert sorted((str(r["session_start"]), r["total"]) for r in streamed) == [
        ("2024-01-01", 3),
        ("2024-06-01", 3),
    ]


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_the_gap_means_the_same_thing_at_every_timestamp_resolution(unit):
    """The gap is microseconds and the column's raw ticks are not, so a resolution other
    than `us` would scale every comparison by up to a thousand in either direction -- the
    same failure the `date32` case makes obvious, in a form that stays plausible. A
    four-hour gap has to break a thirty-minute session at every resolution."""
    base = _BASE
    tbl = pa.table(
        {
            "k": pa.array(["a", "a", "a"], pa.string()),
            "ts": pa.array(
                [base, base + dt.timedelta(minutes=1), base + dt.timedelta(hours=4)],
                pa.timestamp(unit),
            ),
            "v": pa.array([1, 2, 3], pa.int64()),
        }
    )
    got = (
        bt.from_arrow(tbl)
        .session_window("ts", "30m", partition_by=["k"], total=col("v").sum())
        .collect()
    )
    assert str(got.schema.field("session_start").type) == f"timestamp[{unit}]"
    assert sorted(got.to_pydict()["total"]) == [3, 3]
