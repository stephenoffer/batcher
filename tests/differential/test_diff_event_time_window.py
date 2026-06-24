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
    from conftest import assert_same

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


@pytest.mark.parametrize(("width", "slide"), [("1h", "30m"), ("1h", "20m")])
def test_sliding_window_matches_duckdb(duck, width, slide):
    from conftest import assert_same

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
