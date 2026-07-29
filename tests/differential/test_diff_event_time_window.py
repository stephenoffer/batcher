"""Event-time windows (Workstream B) match DuckDB.

Tumbling windows are a group-by on the Rust `window_start` expression; sliding
windows fan each row out to its overlapping windows via `window_buckets` + explode,
then group. Both cross-check against DuckDB — `time_bucket` for tumbling, an explicit
range join over generated bucket starts for sliding.
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
    # Minute offsets spread across a few hours, with a null instant and repeats.
    mins = [0, 7, 30, 59, 60, 65, 95, 120, 121, 185, 200, 240]
    ts = [_BASE + dt.timedelta(minutes=m) for m in mins]
    ts.append(None)  # a null event-time → its own (null) bucket
    vals = list(range(1, len(ts) + 1))
    return pa.table({"ts": pa.array(ts, type=pa.timestamp("us")), "v": pa.array(vals, pa.int64())})


# Batcher compact duration → the equivalent DuckDB INTERVAL literal.
_DUCK_INTERVAL = {"1h": "1 hour", "30m": "30 minutes", "15m": "15 minutes", "20m": "20 minutes"}


@pytest.mark.parametrize("width", ["1h", "30m", "15m"])
def test_tumbling_window_matches_duckdb(duck, width):
    tbl = _data()
    got = (
        bt.from_arrow(tbl)
        .group_by(w=bt.window(col("ts"), width))
        .agg(total=col("v").sum(), n=col("v").count())
        .collect()
    )
    duck.register("t", tbl)
    rel = duck.sql(
        f"SELECT time_bucket(INTERVAL '{_DUCK_INTERVAL[width]}', ts, TIMESTAMP '1970-01-01') AS w, "
        "sum(v) AS total, count(v) AS n FROM t GROUP BY w"
    )
    assert_same(got, rel)


@pytest.mark.parametrize("width", ["1h", "30m", "15m"])
def test_tumbling_window_before_the_epoch_matches_duckdb(duck, width):
    """Pre-epoch instants need a *floor* division, not a truncating one.

    Bucketing negative microseconds-since-epoch by truncation rounds toward zero, which
    puts the window start *after* the row it contains. Every other case here is in 2024,
    so nothing exercised the sign.
    """
    mins = [-1, -7, -30, -59, -60, -61, -120, -1440, 0, 5]
    ts = [_BASE.replace(year=1970) + dt.timedelta(minutes=m) for m in mins]
    tbl = pa.table(
        {
            "ts": pa.array(ts, type=pa.timestamp("us")),
            "v": pa.array(list(range(1, len(ts) + 1)), pa.int64()),
        }
    )
    got = (
        bt.from_arrow(tbl)
        .group_by(w=bt.window(col("ts"), width))
        .agg(total=col("v").sum(), n=col("v").count())
        .collect()
    )
    duck.register("t", tbl)
    rel = duck.sql(
        f"SELECT time_bucket(INTERVAL '{_DUCK_INTERVAL[width]}', ts, TIMESTAMP '1970-01-01') AS w, "
        "sum(v) AS total, count(v) AS n FROM t GROUP BY w"
    )
    assert_same(got, rel)
    # ...and the start must actually contain its rows, which an order-independent
    # multiset comparison cannot see.
    rows = bt.from_arrow(tbl).select(ts=col("ts"), w=bt.window(col("ts"), width)).to_pydict()
    for row_ts, start in zip(rows["ts"], rows["w"], strict=True):
        assert start <= row_ts, f"window start {start} is after the row {row_ts} it holds"


@pytest.mark.parametrize(("width", "slide"), [("1h", "30m"), ("1h", "20m")])
def test_sliding_window_matches_duckdb(duck, width, slide):
    tbl = _data()
    got = (
        bt.from_arrow(tbl)
        .select(w=bt.window(col("ts"), width, slide), v=col("v"))
        .explode("w")
        .group_by("w")
        .agg(total=col("v").sum(), n=col("v").count())
        .collect()
    )
    duck.register("t", tbl)
    width, slide = _DUCK_INTERVAL[width], _DUCK_INTERVAL[slide]
    # Independent oracle: generate every candidate window start at `slide` cadence,
    # then range-join rows in [start, start+width). Windows with no rows drop out of
    # the inner join, matching the explode path (which only emits a start for rows
    # that land in it).
    rel = duck.sql(
        f"""
        WITH b AS (SELECT min(ts) lo, max(ts) hi FROM t WHERE ts IS NOT NULL),
        starts AS (
            SELECT UNNEST(generate_series(
                lo - INTERVAL '{width}', hi, INTERVAL '{slide}'
            )) AS w FROM b
        )
        SELECT s.w AS w, sum(t.v) AS total, count(t.v) AS n
        FROM starts s JOIN t ON t.ts >= s.w AND t.ts < s.w + INTERVAL '{width}'
        GROUP BY s.w
        """
    )
    assert_same(got, rel)
