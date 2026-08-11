"""Sort elimination vs DuckDB — eliminating a redundant sort must leave the rows in
the same (correct) order. `assert_same` is order-independent, so these tests check
the observed order explicitly as well."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same_ordered


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
    out = _t(duck).sort("x").sort("x").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x"))


# --- descending and null placement --------------------------------------------------------
#
# A descending ordering is now tracked and eliminated against, which is the common real
# shape (`ORDER BY ts DESC`). These are the cases that would silently reorder a user's rows
# if the direction were dropped or reinterpreted, so each one checks the order explicitly
# against DuckDB rather than relying on the order-independent multiset comparison.


def test_redundant_descending_resort_matches_duckdb_order(duck):
    once = _t(duck).sort("x", descending=True).collect().to_pydict()
    twice = _t(duck).sort("x", descending=True).sort("x", descending=True).collect().to_pydict()
    assert once == twice
    assert once["x"] == sorted(once["x"], reverse=True)
    assert_same_ordered(
        _t(duck).sort("x", descending=True).sort("x", descending=True).collect(),
        duck.sql("SELECT * FROM t ORDER BY x DESC"),
    )


def test_ascending_resort_over_descending_input_still_ascends(duck):
    """The rule must NOT fire here. If it did, the rows would come back descending and
    only an order-sensitive comparison would notice."""
    out = _t(duck).sort("x", descending=True).sort("x").collect()
    assert out.to_pydict()["x"] == sorted(out.to_pydict()["x"])
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x"))


def test_descending_resort_over_ascending_input_still_descends(duck):
    out = _t(duck).sort("x").sort("x", descending=True).collect()
    assert out.to_pydict()["x"] == sorted(out.to_pydict()["x"], reverse=True)
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x DESC"))


def test_coarser_descending_resort_keeps_the_finer_order(duck):
    out = _t(duck).sort("x", "y", descending=True).sort("x", descending=True).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x DESC, y DESC"))


def test_nulls_first_resort_over_nulls_last_input_is_not_eliminated(duck):
    """With nulls present the two placements are different orders, so the outer sort
    must survive and the nulls must end up first."""
    t = pa.table({"x": [3, None, 1, None, 2], "y": [1, 2, 3, 4, 5]})
    duck.register("n", t)
    out = bt.from_arrow(t).sort("x").sort("x", nulls_first=True).collect()
    assert out.to_pydict()["x"][:2] == [None, None]
    assert_same_ordered(out, duck.sql("SELECT * FROM n ORDER BY x NULLS FIRST"))
