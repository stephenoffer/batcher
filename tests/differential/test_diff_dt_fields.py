"""Date/time field-extraction (`.dt`) differential tests vs DuckDB."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    base = dt.datetime(2021, 1, 1, 0, 0, 0)
    # Spread across months/days/hours/minutes/seconds and weekdays.
    ts = [
        base + dt.timedelta(days=i * 9, hours=i * 5, minutes=i * 7, seconds=i * 11)
        for i in range(60)
    ]
    tbl = pa.table({"ts": pa.array(ts, type=pa.timestamp("us"))})
    duck.register("t", tbl)
    return tbl


def test_all_dt_fields_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            y=col("ts").dt.year(),
            mo=col("ts").dt.month(),
            d=col("ts").dt.day(),
            h=col("ts").dt.hour(),
            mi=col("ts").dt.minute(),
            s=col("ts").dt.second(),
            q=col("ts").dt.quarter(),
            w=col("ts").dt.week(),
            dw=col("ts").dt.dayofweek(),
            dy=col("ts").dt.dayofyear(),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT year(ts) y, month(ts) mo, day(ts) d, hour(ts) h, minute(ts) mi, "
        "second(ts) s, quarter(ts) q, week(ts) w, dayofweek(ts) dw, dayofyear(ts) dy "
        "FROM t"
    )
    assert_same(out, expected)


def test_dt_fields_filter_vs_duckdb(duck, t):
    from conftest import assert_same

    # Field extractions used in a predicate (exercises them in filter position).
    out = (
        bt.from_arrow(t)
        .filter(col("ts").dt.month() == 6)
        .select(d=col("ts").dt.day(), h=col("ts").dt.hour())
        .collect()
    )
    expected = duck.sql("SELECT day(ts) d, hour(ts) h FROM t WHERE month(ts) = 6")
    assert_same(out, expected)
