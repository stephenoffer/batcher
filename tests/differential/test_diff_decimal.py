"""Decimal128 columns: exact sum/min/max and numeric-literal comparisons vs DuckDB."""

from __future__ import annotations

import decimal as D

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit


@pytest.fixture
def t(duck):
    prices = pa.array(
        [
            D.Decimal("1.50"),
            D.Decimal("2.25"),
            D.Decimal("3.75"),
            D.Decimal("1.50"),
            D.Decimal("9.99"),
        ],
        pa.decimal128(10, 2),
    )
    tbl = pa.table({"k": [1, 2, 1, 2, 1], "price": prices})
    duck.register("t", tbl)
    return tbl


def test_decimal_grouped_sum_min_max(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .group_by("k")
        .agg(s=col("price").sum(), mn=col("price").min(), mx=col("price").max())
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT k, SUM(price) s, MIN(price) mn, MAX(price) mx FROM t GROUP BY k")
    )


def test_decimal_global_sum(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).group_by().agg(s=col("price").sum()).collect()
    assert_same(out, duck.sql("SELECT SUM(price) s FROM t"))


def test_decimal_filter_against_numeric_literal(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(col("price") > lit(2.0)).select("price").collect()
    assert_same(out, duck.sql("SELECT price FROM t WHERE price > 2.0"))


def test_decimal_sort_and_distinct(duck, t):
    from conftest import assert_same

    assert_same(
        bt.from_arrow(t).select("price").distinct().collect(),
        duck.sql("SELECT DISTINCT price FROM t"),
    )
