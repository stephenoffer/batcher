"""Differential tests for distinct/union against DuckDB."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def test_distinct_vs_duckdb(duck):
    from conftest import assert_same

    t = pa.table({"a": [1, 1, 2, 2, 3, 1], "b": ["x", "x", "y", "z", "z", "x"]})
    duck.register("t", t)
    out = bt.from_arrow(t).distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT * FROM t"))


def test_distinct_after_projection_vs_duckdb(duck):
    from conftest import assert_same

    t = pa.table({"a": [1, 2, 3, 4], "b": [10, 10, 20, 20]})
    duck.register("t", t)
    out = bt.from_arrow(t).select("b").distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT b FROM t"))


def test_union_all_vs_duckdb(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 3]})
    b = pa.table({"x": [3, 4, 5]})
    duck.register("a", a)
    duck.register("b", b)
    out = bt.from_arrow(a).union(bt.from_arrow(b)).collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION ALL SELECT * FROM b"))


def test_union_distinct_vs_duckdb(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2, 3]})
    b = pa.table({"x": [3, 4, 5]})
    duck.register("a", a)
    duck.register("b", b)
    out = bt.from_arrow(a).union(bt.from_arrow(b), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION SELECT * FROM b"))
