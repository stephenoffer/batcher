"""`bt.window` origin/offset/label/closed and SQL `time_bucket`'s origin, held to DuckDB
``time_bucket`` and Polars ``group_by_dynamic``.

Before these existed a window grid was always anchored at the Unix epoch, so "fifteen-minute
bins starting at :07" could not be written, and SQL `time_bucket`'s third argument (an
origin or an offset) was silently ignored -- the default grid came back with no error.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

_TS = [
    dt.datetime(2024, 1, 1, 9, 52),  # exactly on a :07-anchored 15-minute boundary
    dt.datetime(2024, 1, 1, 10, 5),
    dt.datetime(2024, 1, 1, 10, 7, 0, 1),
    dt.datetime(2024, 1, 1, 11, 0),  # exactly on an hour boundary
    dt.datetime(1969, 12, 31, 23, 59, 59),
    None,
]


def _table():
    return pa.table({"ts": pa.array(_TS, pa.timestamp("us")), "v": list(range(len(_TS)))})


@pytest.mark.parametrize(
    ("width", "origin"),
    [
        ("15 MINUTE", "2000-01-01 00:07:00"),
        ("1 HOUR", "2024-01-01 00:30:00"),
        ("2 DAY", "2021-01-01 00:00:00"),
        ("7 MINUTE", "1999-12-31 23:59:00"),
    ],
)
def test_sql_time_bucket_origin_matches_duckdb(duck, width, origin):
    query = f"SELECT time_bucket(INTERVAL {width}, ts, TIMESTAMP '{origin}') AS r FROM t"
    duck.register("t", _table())
    assert_same(bt.sql(query, t=bt.from_arrow(_table())).collect(), duck.sql(query))


@pytest.mark.parametrize("width", ["15 MINUTE", "2 DAY", "1 HOUR"])
@pytest.mark.parametrize("offset", ["7 MINUTE", "1 HOUR", "-5 MINUTE"])
def test_sql_time_bucket_offset_matches_duckdb(duck, width, offset):
    query = f"SELECT time_bucket(INTERVAL {width}, ts, INTERVAL '{offset}') AS r FROM t"
    duck.register("t", _table())
    assert_same(bt.sql(query, t=bt.from_arrow(_table())).collect(), duck.sql(query))


@pytest.mark.parametrize(
    ("width", "duration"), [("15 MINUTE", "15m"), ("7 MINUTE", "7m"), ("2 DAY", "2d")]
)
def test_python_origin_matches_duckdb(duck, width, duration):
    origin = dt.datetime(2000, 1, 1, 0, 7)
    got = bt.from_arrow(_table()).select(r=bt.window(col("ts"), duration, origin=origin))
    duck.register("t", _table())
    expected = duck.sql(
        f"SELECT time_bucket(INTERVAL {width}, ts, TIMESTAMP '{origin.isoformat()}') AS r FROM t"
    )
    assert_same(got.collect(), expected)


@pytest.mark.parametrize("closed", ["left", "right"])
@pytest.mark.parametrize("label", ["left", "right"])
@pytest.mark.parametrize("offset", [None, "7m", "-20m"])
def test_window_matches_polars_group_by_dynamic(closed, label, offset):
    rows = [t for t in _TS if t is not None]
    frame = pl.DataFrame({"ts": rows, "v": list(range(len(rows)))}).sort("ts")
    polars_out = (
        frame.group_by_dynamic(
            "ts",
            every="15m",
            period="15m",
            offset=offset or "0m",
            closed=closed,
            label=label,
            start_by="window",
        )
        .agg(pl.col("v").sum())
        .sort("ts")
    )
    epoch = dt.datetime(1970, 1, 1)
    w = bt.window(col("ts"), "15m", origin=epoch, offset=offset, closed=closed, label=label)
    ours = (
        bt.from_pydict({"ts": rows, "v": list(range(len(rows)))})
        .group_by(ts=w)
        .agg(v=col("v").sum())
        .sort("ts")
        .to_pydict()
    )
    assert ours == {"ts": polars_out["ts"].to_list(), "v": polars_out["v"].to_list()}


def test_an_aware_column_keeps_its_zone_on_the_window_label():
    ny = pa.table(
        {"ts": pa.array([dt.datetime(2024, 3, 10, 7, 30)], pa.timestamp("us", "America/New_York"))}
    )
    q = bt.from_arrow(ny).select(w=bt.window(col("ts"), "1h", closed="right", label="right"))
    assert q.schema.field("w").type == q.to_arrow().schema.field("w").type
    assert q.to_arrow().schema.field("w").type == pa.timestamp("us", tz="America/New_York")


def test_tumbling_only_options_are_refused_with_slide():
    with pytest.raises(bt.PlanError, match="tumbling"):
        bt.window(col("ts"), "1h", "15m", origin=dt.datetime(2024, 1, 1))
    with pytest.raises(bt.PlanError, match="label"):
        bt.window(col("ts"), "1h", label="center")


def test_the_default_window_ir_is_unchanged():
    assert bt.window(col("ts"), "1h").to_ir() == {
        "e": "window_start",
        "input": col("ts").to_ir(),
        "width_micros": 3_600_000_000,
    }
