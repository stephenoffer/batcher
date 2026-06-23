"""Differential `.dt` tests for isodow/century/decade vs DuckDB."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    # Span multiple decades and centuries, including the 1999/2000 century edge
    # and 2021, plus a few others across weekdays.
    ts = [
        dt.datetime(1999, 12, 31, 23, 59, 59),
        dt.datetime(2000, 1, 1, 0, 0, 0),
        dt.datetime(2000, 6, 15, 12, 30, 0),
        dt.datetime(2021, 3, 8, 9, 0, 0),
        dt.datetime(1985, 7, 4, 6, 0, 0),
        dt.datetime(1900, 1, 1, 0, 0, 0),
        dt.datetime(1901, 1, 1, 0, 0, 0),
        dt.datetime(2010, 10, 10, 10, 10, 10),
        dt.datetime(2099, 12, 31, 0, 0, 0),
        dt.datetime(2100, 1, 1, 0, 0, 0),
    ]
    tbl = pa.table({"ts": pa.array(ts, type=pa.timestamp("us"))})
    duck.register("t", tbl)
    return tbl


def test_dt_extra_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            d=col("ts").dt.isodow(),
            c=col("ts").dt.century(),
            de=col("ts").dt.decade(),
        )
        .collect()
    )
    expected = duck.sql("SELECT isodow(ts) d, century(ts) c, decade(ts) de FROM t")
    assert_same(out, expected)
