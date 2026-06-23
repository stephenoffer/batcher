"""Timestamp / date literal comparisons vs DuckDB."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "ts": pa.array(
                [dt.datetime(2020, 1, 1), dt.datetime(2021, 6, 15), dt.datetime(2022, 3, 1)],
                pa.timestamp("us"),
            ),
            "d": pa.array(
                [dt.date(2020, 1, 1), dt.date(2021, 6, 15), dt.date(2022, 3, 1)], pa.date32()
            ),
            "v": [1, 2, 3],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_timestamp_literal_filter(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(col("ts") > lit(dt.datetime(2021, 1, 1))).select("v").collect()
    assert_same(out, duck.sql("SELECT v FROM t WHERE ts > TIMESTAMP '2021-01-01'"))


def test_date_range_filter(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .filter((col("d") >= lit(dt.date(2021, 1, 1))) & (col("d") < lit(dt.date(2022, 1, 1))))
        .select("v")
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT v FROM t WHERE d >= DATE '2021-01-01' AND d < DATE '2022-01-01'")
    )


def test_date_equality(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(col("d") == lit(dt.date(2020, 1, 1))).select("v").collect()
    assert_same(out, duck.sql("SELECT v FROM t WHERE d = DATE '2020-01-01'"))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT v FROM t WHERE ts > TIMESTAMP '2021-01-01'",
        "SELECT v FROM t WHERE d >= DATE '2021-01-01' AND d < DATE '2022-01-01'",
        "SELECT v FROM t WHERE d = DATE '2020-01-01'",
        "SELECT v FROM t WHERE ts BETWEEN TIMESTAMP '2020-06-01' AND TIMESTAMP '2021-12-31'",
    ],
)
def test_sql_temporal_literals(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
