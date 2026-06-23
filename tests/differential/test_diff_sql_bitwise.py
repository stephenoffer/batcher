"""Bitwise operators and MONTH/YEAR interval arithmetic vs DuckDB."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt


@pytest.mark.parametrize(
    "q",
    [
        "SELECT a & b r FROM t",
        "SELECT a | b r FROM t",
        "SELECT xor(a, b) r FROM t",
        "SELECT a << 2 r FROM t",
        "SELECT a >> 1 r FROM t",
        "SELECT a FROM t WHERE (a & 1) = 0",
    ],
)
def test_bitwise(duck, q):
    from conftest import assert_same

    t = pa.table({"a": [12, 7, 255, 1024], "b": [10, 3, 15, 7]})
    duck.register("t", t)
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, CAST(d + INTERVAL 1 MONTH AS DATE) r FROM t",
        "SELECT id, CAST(d + INTERVAL 3 MONTH AS DATE) r FROM t",
        "SELECT id, CAST(d - INTERVAL 1 MONTH AS DATE) r FROM t",
        "SELECT id, CAST(d + INTERVAL 1 YEAR AS DATE) r FROM t",
        "SELECT id, CAST(d + INTERVAL 13 MONTH AS DATE) r FROM t",
        "SELECT id, CAST(date_add(d, INTERVAL 2 MONTH) AS DATE) r FROM t",
    ],
)
def test_month_year_interval(duck, q):
    from conftest import assert_same

    # Month-end clamping + leap year edges (Jan 31, Feb 29).
    t = pa.table(
        {
            "id": [1, 2, 3],
            "d": pa.array(
                [dt.date(2021, 1, 31), dt.date(2021, 6, 15), dt.date(2020, 2, 29)], pa.date32()
            ),
        }
    )
    duck.register("t", t)
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
