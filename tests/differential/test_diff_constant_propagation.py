"""Constant propagation vs DuckDB — substituting a `col = literal` equality into the
rest of a conjunction must not change the result (incl. nulls and contradictions)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col


def _t(duck):
    t = pa.table({"x": [5, 5, 3, 5, None], "y": [10, 3, 7, 6, 5]})
    duck.register("t", t)
    return bt.from_arrow(t)


def test_propagate_into_comparison(duck):
    ds = _t(duck).filter((col("x") == 5) & (col("y") > col("x")))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE x = 5 AND y > x"))


def test_propagate_with_null_rows(duck):
    # The null-x row is dropped by `x = 5` either way.
    ds = _t(duck).filter((col("x") == 5) & (col("y") <= col("x")))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE x = 5 AND y <= x"))


def test_contradiction_is_empty(duck):
    ds = _t(duck).filter((col("x") == 5) & (col("x") == 3))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE x = 5 AND x = 3"))


def test_multiple_constants(duck):
    t = pa.table({"a": [1, 1, 2], "b": [2, 5, 2], "c": [3, 3, 9]})
    duck.register("m", t)
    pred = (col("a") == 1) & (col("b") == 2) & (col("c") > col("a") + col("b"))
    ds = bt.from_arrow(t).filter(pred)
    assert_same(ds.collect(), duck.sql("SELECT * FROM m WHERE a = 1 AND b = 2 AND c > a + b"))
