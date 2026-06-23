"""Empty-relation propagation vs DuckDB — folding an empty subtree must not change
the result (an empty input stays empty; an empty union branch contributes nothing)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _t(duck):
    t = pa.table({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    duck.register("t", t)
    return bt.from_arrow(t)


def test_filter_over_empty(duck):
    from conftest import assert_same

    out = _t(duck).limit(0).filter(bt.col("x") > 2).collect()
    assert_same(out, duck.sql("SELECT * FROM (SELECT * FROM t LIMIT 0) s WHERE x > 2"))


def test_sort_filter_over_empty(duck):
    from conftest import assert_same

    out = _t(duck).limit(0).sort("x").filter(bt.col("x") > 2).collect()
    assert_same(out, duck.sql("SELECT * FROM (SELECT * FROM t LIMIT 0) s WHERE x > 2 ORDER BY x"))


def test_union_with_empty_branch(duck):
    from conftest import assert_same

    other = pa.table({"x": [6, 7], "y": [60, 70]})
    duck.register("o", other)
    out = _t(duck).limit(0).union(bt.from_arrow(other)).collect()
    assert_same(out, duck.sql("(SELECT * FROM t LIMIT 0) UNION ALL SELECT * FROM o"))


def test_union_distinct_with_empty_branch(duck):
    from conftest import assert_same

    other = pa.table({"x": [1, 1, 2], "y": [10, 10, 20]})
    duck.register("o2", other)
    out = _t(duck).limit(0).union(bt.from_arrow(other), distinct=True).collect()
    assert_same(out, duck.sql("(SELECT * FROM t LIMIT 0) UNION SELECT * FROM o2"))
