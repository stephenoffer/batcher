"""Differential tests for sort/limit/top-N against DuckDB (order-sensitive)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same_ordered
from batcher import col


def test_sort_single_key(duck):
    t = pa.table({"x": [3, 1, 2, 5, 4], "y": [10, 20, 30, 40, 50]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x"))


def test_sort_desc(duck):
    t = pa.table({"x": [3, 1, 2, 5, 4]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x", descending=True).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x DESC"))


def test_sort_multi_key_mixed(duck):
    t = pa.table({"a": [1, 1, 2, 2, 1], "b": [2, 1, 4, 3, 3], "v": [10, 20, 30, 40, 50]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("a", "b", descending=[False, True]).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY a ASC, b DESC"))


def test_top_n(duck):
    t = pa.table({"x": list(range(20, 0, -1))})  # 20..1
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").limit(5).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x LIMIT 5"))


def test_limit_offset(duck):
    t = pa.table({"x": list(range(10))})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").limit(3, offset=4).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x LIMIT 3 OFFSET 4"))


def test_sql_limit_offset(duck):
    t = pa.table({"x": list(range(10))})
    duck.register("t", t)
    q = "SELECT x FROM t ORDER BY x LIMIT 3 OFFSET 4"
    assert_same_ordered(bt.sql(q, t=t).collect(), duck.sql(q))


def test_sql_bare_offset(duck):
    t = pa.table({"x": list(range(10))})
    duck.register("t", t)
    q = "SELECT x FROM t ORDER BY x OFFSET 7"
    assert_same_ordered(bt.sql(q, t=t).collect(), duck.sql(q))


def test_sort_by_expression(duck):
    t = pa.table({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort(col("x") + col("y"), col("x")).collect()
    # x+y is constant (5) here, so tiebreak by x; mirror in DuckDB.
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x + y, x"))


def test_sort_by_null_typed_key_orders_by_remaining(duck):
    """ORDER BY an all-null (Null-typed) column must not error — it is all-equal, so
    the ordering falls to the following keys, exactly as DuckDB does.

    A `from_pydict` column that is entirely None is Arrow's `Null` type, which has no
    natural order; arrow's sort kernels reject it, so the engine used to raise
    "The data type type Null has no natural order" on a query DuckDB runs fine.
    """
    t = pa.table(
        {
            "n": pa.array([None, None, None, None, None], type=pa.null()),
            "y": [3, 1, 2, 1, 3],
            "p": [10, 11, 12, 13, 14],
        }
    )
    duck.register("t", t)
    # Leading null key contributes nothing; rows come out in y order (ties stable).
    out = bt.from_arrow(t).sort("n", "y").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY n, y, p"))


def test_sort_by_sole_null_typed_key(duck):
    """ORDER BY only an all-null column preserves input order (a stable no-op sort)."""
    t = pa.table({"n": pa.array([None, None, None], type=pa.null()), "p": [10, 20, 30]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("n").collect()
    # Every row ties on n; both engines keep input order (assert on p to pin it).
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY n, p"))
