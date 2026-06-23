"""Sort elimination vs DuckDB — eliminating a redundant sort must leave the rows in
the same (correct) order. `assert_same` is order-independent, so these tests check
the observed order explicitly as well."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _t(duck):
    t = pa.table({"x": [3, 1, 2, 1, 4], "y": [1, 2, 3, 4, 5]})
    duck.register("t", t)
    return bt.from_arrow(t)


def test_redundant_resort_matches_single_sort_order(duck):
    # The doubly-sorted (rule fires) and singly-sorted results must be identical,
    # in order — proving elimination preserved the ordering.
    once = _t(duck).sort("x").collect().to_pydict()
    twice = _t(duck).sort("x").sort("x").collect().to_pydict()
    assert once == twice
    assert once["x"] == sorted(once["x"])  # genuinely ascending


def test_coarser_resort_order(duck):
    # Sort by (x, y) then by x: result stays ordered by (x, y).
    out = _t(duck).sort("x", "y").sort("x").collect().to_pydict()
    pairs = list(zip(out["x"], out["y"], strict=True))
    assert pairs == sorted(pairs)


def test_resort_multiset_matches_duckdb(duck):
    from conftest import assert_same

    out = _t(duck).sort("x").sort("x").collect()
    assert_same(out, duck.sql("SELECT * FROM t ORDER BY x"))
