"""Differential tests for sort/limit/top-N against DuckDB (order-sensitive)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_sort_single_key(duck):
    from conftest import assert_same_ordered

    t = pa.table({"x": [3, 1, 2, 5, 4], "y": [10, 20, 30, 40, 50]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x"))


def test_sort_desc(duck):
    from conftest import assert_same_ordered

    t = pa.table({"x": [3, 1, 2, 5, 4]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x", descending=True).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x DESC"))


def test_sort_multi_key_mixed(duck):
    from conftest import assert_same_ordered

    t = pa.table({"a": [1, 1, 2, 2, 1], "b": [2, 1, 4, 3, 3], "v": [10, 20, 30, 40, 50]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("a", "b", descending=[False, True]).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY a ASC, b DESC"))


def test_top_n(duck):
    from conftest import assert_same_ordered

    t = pa.table({"x": list(range(20, 0, -1))})  # 20..1
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").limit(5).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x LIMIT 5"))


def test_limit_offset(duck):
    from conftest import assert_same_ordered

    t = pa.table({"x": list(range(10))})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").limit(3, offset=4).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x LIMIT 3 OFFSET 4"))


def test_sql_limit_offset(duck):
    from conftest import assert_same_ordered

    t = pa.table({"x": list(range(10))})
    duck.register("t", t)
    q = "SELECT x FROM t ORDER BY x LIMIT 3 OFFSET 4"
    assert_same_ordered(bt.sql(q, t=t).collect(), duck.sql(q))


def test_sql_bare_offset(duck):
    from conftest import assert_same_ordered

    t = pa.table({"x": list(range(10))})
    duck.register("t", t)
    q = "SELECT x FROM t ORDER BY x OFFSET 7"
    assert_same_ordered(bt.sql(q, t=t).collect(), duck.sql(q))


def test_sort_by_expression(duck):
    from conftest import assert_same_ordered

    t = pa.table({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort(col("x") + col("y"), col("x")).collect()
    # x+y is constant (5) here, so tiebreak by x; mirror in DuckDB.
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x + y, x"))
