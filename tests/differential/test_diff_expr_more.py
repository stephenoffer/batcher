"""Differential tests for coalesce/fill_null, math, more string fns, set ops."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import coalesce, col, lit


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "a": [1, 4, 9, None],
            "f": [1.4, 2.6, -3.5, None],
            "s": ["  Hi ", "aXbXc", None, "end"],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "expr,sql",
    [
        (col("a").abs(), "abs(a)"),
        (col("f").abs(), "abs(f)"),
        (col("f").floor(), "floor(f)"),
        (col("f").ceil(), "ceil(f)"),
        (col("f").round(), "round(f)"),
        (col("a").cast("float64").sqrt(), "sqrt(CAST(a AS DOUBLE))"),
        (col("a").fill_null(0), "coalesce(a, 0)"),
        (coalesce(col("a"), lit(99)), "coalesce(a, 99)"),
        (col("s").str.trim(), "trim(s)"),
        (col("s").str.lstrip(), "ltrim(s)"),
        (col("s").str.rstrip(), "rtrim(s)"),
        (col("s").str.replace("X", "-"), "replace(s, 'X', '-')"),
    ],
)
def test_expr_vs_duckdb(duck, t, expr, sql):
    from conftest import assert_same

    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


def test_set_operations_vs_duckdb(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 3, 3, 4]})
    b = pa.table({"x": [3, 4, 5]})
    duck.register("a", a)
    duck.register("b", b)
    assert_same(
        bt.from_arrow(a).intersect(bt.from_arrow(b)).collect(),
        duck.sql("SELECT x FROM a INTERSECT SELECT x FROM b"),
    )
    assert_same(
        bt.from_arrow(a).except_(bt.from_arrow(b)).collect(),
        duck.sql("SELECT x FROM a EXCEPT SELECT x FROM b"),
    )


def test_sql_set_operations_vs_duckdb(duck):
    from conftest import assert_same

    u = pa.table({"x": [1, 2, 3, 3]})
    v = pa.table({"x": [3, 4]})
    duck.register("u", u)
    duck.register("v", v)
    assert_same(
        bt.sql("SELECT x FROM u INTERSECT SELECT x FROM v", u=u, v=v).collect(),
        duck.sql("SELECT x FROM u INTERSECT SELECT x FROM v"),
    )
    assert_same(
        bt.sql("SELECT x FROM u EXCEPT SELECT x FROM v", u=u, v=v).collect(),
        duck.sql("SELECT x FROM u EXCEPT SELECT x FROM v"),
    )
