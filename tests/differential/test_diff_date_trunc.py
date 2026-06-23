"""date_trunc (`.dt.truncate`) differential tests vs DuckDB."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    ts = [
        dt.datetime(2021, 3, 14, 9, 26, 53) + dt.timedelta(days=i * 23, minutes=i * 17)
        for i in range(30)
    ]
    tbl = pa.table({"ts": pa.array(ts, type=pa.timestamp("us"))})
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize("unit", ["year", "month", "day", "hour", "minute", "second"])
def test_date_trunc_vs_duckdb(duck, t, unit):
    from conftest import assert_same

    out = bt.from_arrow(t).select(x=col("ts").dt.truncate(unit)).collect()
    expected = duck.sql(f"SELECT date_trunc('{unit}', ts) x FROM t")
    assert_same(out, expected)
