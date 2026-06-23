"""Numeric coercion in COALESCE and grouping by a derived (expression) key."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import coalesce, col, count, lit


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "a": pa.array([1, None, 3, None], pa.int64()),
            "f": pa.array([None, 2.5, None, 4.5], pa.float64()),
            "v": [10, 20, 30, 40],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_coalesce_int_float_promotes(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(r=coalesce(col("a"), col("f"))).collect()
    assert_same(out, duck.sql("SELECT coalesce(a, f) AS r FROM t"))


def test_coalesce_int_float_literal(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(r=coalesce(col("a"), lit(2.5))).collect()
    assert_same(out, duck.sql("SELECT coalesce(a, 2.5) AS r FROM t"))


def test_fill_null_float(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(r=col("a").fill_null(0.5)).collect()
    assert_same(out, duck.sql("SELECT coalesce(a, 0.5) AS r FROM t"))


def test_group_by_expression_key(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).group_by(par=col("v") % 2).agg(n=count(), s=col("v").sum()).collect()
    assert_same(out, duck.sql("SELECT v % 2 AS par, COUNT(*) n, SUM(v) s FROM t GROUP BY v % 2"))


def test_group_by_column_and_expression(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).group_by("a", bucket=col("v") % 3).agg(n=count()).collect()
    assert_same(
        out,
        duck.sql("SELECT a, v % 3 AS bucket, COUNT(*) n FROM t GROUP BY a, v % 3"),
    )
