"""Differential tests for distinct/union against DuckDB."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same


def test_distinct_vs_duckdb(duck):
    t = pa.table({"a": [1, 1, 2, 2, 3, 1], "b": ["x", "x", "y", "z", "z", "x"]})
    duck.register("t", t)
    out = bt.from_arrow(t).distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT * FROM t"))


def test_distinct_after_projection_vs_duckdb(duck):
    t = pa.table({"a": [1, 2, 3, 4], "b": [10, 10, 20, 20]})
    duck.register("t", t)
    out = bt.from_arrow(t).select("b").distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT b FROM t"))


def test_union_all_vs_duckdb(duck):
    a = pa.table({"x": [1, 2, 3]})
    b = pa.table({"x": [3, 4, 5]})
    duck.register("a", a)
    duck.register("b", b)
    out = bt.from_arrow(a).union(bt.from_arrow(b)).collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION ALL SELECT * FROM b"))


def test_union_distinct_vs_duckdb(duck):
    a = pa.table({"x": [1, 2, 2, 3]})
    b = pa.table({"x": [3, 4, 5]})
    duck.register("a", a)
    duck.register("b", b)
    out = bt.from_arrow(a).union(bt.from_arrow(b), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION SELECT * FROM b"))


def test_streamed_union_all_vs_duckdb(duck):
    """UNION ALL streams branch by branch rather than materializing; the streamed rows
    must still be DuckDB's. Nulls and an empty branch included — the edges where a
    per-branch driver differs from a whole-relation one."""
    a = pa.table({"x": [1, None, 3], "s": ["p", "q", None]})
    b = pa.table({"x": [3, 4, None], "s": [None, "r", "t"]})
    empty = pa.table({"x": pa.array([], pa.int64()), "s": pa.array([], pa.string())})
    duck.register("a", a)
    duck.register("b", b)
    duck.register("e", empty)
    ds = bt.from_arrow(a).union(bt.from_arrow(empty)).union(bt.from_arrow(b))
    streamed = pa.Table.from_batches(list(ds.iter_batches()))
    assert_same(
        streamed,
        duck.sql("SELECT * FROM a UNION ALL SELECT * FROM e UNION ALL SELECT * FROM b"),
    )


def test_streamed_union_all_preserves_concatenation_order(duck):
    """`assert_same` is order-independent by design, so it cannot see the one thing this
    path promises: branch 0's rows, then branch 1's, in source order."""
    a = pa.table({"x": [1, 2, 3]})
    b = pa.table({"x": [4, 5, 6]})
    duck.register("a", a)
    duck.register("b", b)
    ds = bt.from_arrow(a).union(bt.from_arrow(b))
    streamed = pa.Table.from_batches(list(ds.iter_batches())).to_pydict()["x"]
    expected = [r[0] for r in duck.sql("SELECT * FROM a UNION ALL SELECT * FROM b").fetchall()]
    assert streamed == expected == [1, 2, 3, 4, 5, 6]
