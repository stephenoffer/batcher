"""String and date function differential tests vs DuckDB."""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": ["Hello", "wOrld", "abcdef", None],
            "d": [
                datetime.date(2021, 3, 15),
                datetime.date(1999, 12, 31),
                datetime.date(2024, 2, 29),
                None,
            ],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "expr,sql",
    [
        (col("s").str.upper(), "upper(s)"),
        (col("s").str.lower(), "lower(s)"),
        (col("s").str.len(), "length(s)"),
        (col("s").str.contains("o"), "contains(s, 'o')"),
        (col("s").str.starts_with("a"), "starts_with(s, 'a')"),
        (col("s").str.ends_with("f"), "ends_with(s, 'f')"),
        (col("s").str.substr(2, 3), "substr(s, 2, 3)"),
        (col("s").str.substr(3), "substr(s, 3)"),
        (col("d").dt.year(), "year(d)"),
        (col("d").dt.month(), "month(d)"),
        (col("d").dt.day(), "day(d)"),
    ],
)
def test_str_date_fn_vs_duckdb(duck, t, expr, sql):
    from conftest import assert_same

    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


def test_str_in_filter_and_projection(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .filter(col("s").str.contains("o"))
        .select(up=col("s").str.upper(), n=col("s").str.len())
        .collect()
    )
    expected = duck.sql("SELECT upper(s) AS up, length(s) AS n FROM t WHERE contains(s, 'o')")
    assert_same(out, expected)
