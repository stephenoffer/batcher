"""More `.dt` (dayname/monthname) and `.list` (arg_min/arg_max) tests.

`dayname`/`monthname` are checked DIFFERENTIALLY vs DuckDB; `arg_min`/`arg_max`
are STRUCTURAL (hand-computed expected values).
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    base = dt.datetime(2021, 1, 1, 0, 0, 0)
    # Spread across months/days/weekdays so every name appears.
    ts = [
        base + dt.timedelta(days=i * 9, hours=i * 5, minutes=i * 7, seconds=i * 11)
        for i in range(60)
    ]
    tbl = pa.table({"ts": pa.array(ts, type=pa.timestamp("us"))})
    duck.register("t", tbl)
    return tbl


def test_dayname_monthname_vs_duckdb(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(d=col("ts").dt.dayname(), m=col("ts").dt.monthname()).collect()
    expected = duck.sql("SELECT dayname(ts) d, monthname(ts) m FROM t")
    assert_same(out, expected)


def test_list_arg_min_arg_max_structural():
    tbl = pa.table(
        {"xs": pa.array([[3, 1, 2], [5], [], None, [2, 2, 9, 1]], type=pa.list_(pa.int64()))}
    )
    out = (
        bt.from_arrow(tbl)
        .select(amin=col("xs").list.arg_min(), amax=col("xs").list.arg_max())
        .collect()
        .to_pydict()
    )
    assert out["amin"] == [1, 0, None, None, 3]
    assert out["amax"] == [0, 0, None, None, 2]
