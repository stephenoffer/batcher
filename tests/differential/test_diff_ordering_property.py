"""Eliminating a sort the input already delivers must not change the answer.

The ordering property now survives a projection, so a redundant `ORDER BY` across a `SELECT`
is removed. That is only sound if the surviving order really is the one the query asked for —
so these compare **row order** against DuckDB (`assert_same_ordered`), not just the multiset,
which is the only comparison that can see a sort bug at all. An order-independent assertion
cannot see a sort bug; the repo's own contract says so.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    table = pa.table(
        {
            "a": pa.array([3, 1, 2, 1, None], type=pa.int64()),
            "b": pa.array([1, 2, 3, 9, 4], type=pa.int64()),
        }
    )
    duck.register("t", table)
    return bt.from_arrow(table)


def test_redundant_sort_across_a_select(t, duck):
    got = t.sort("a").select(x=col("a"), y=col("b")).sort("x").collect()
    assert_same_ordered(
        got,
        duck.sql("SELECT a AS x, b AS y FROM (SELECT * FROM t ORDER BY a) ORDER BY x"),
    )


def test_a_different_sort_after_a_select_is_honored(t, duck):
    """The second sort changes the order, so it must survive."""
    got = t.sort("a").select(x=col("a"), y=col("b")).sort("y").collect()
    assert_same_ordered(
        got,
        duck.sql("SELECT a AS x, b AS y FROM (SELECT * FROM t ORDER BY a) ORDER BY y"),
    )


def test_a_renamed_key_still_orders_correctly(t, duck):
    got = t.sort("a").select(k=col("a")).sort("k").collect()
    assert_same_ordered(got, duck.sql("SELECT a AS k FROM t ORDER BY k"))


def test_a_computed_output_does_not_inherit_the_order(t, duck):
    """`a + 1` is not `a`; the sort on it must actually run."""
    got = t.sort("a").select(x=col("a") + 1).sort("x").collect()
    assert_same_ordered(got, duck.sql("SELECT a + 1 AS x FROM t ORDER BY x"))


def test_a_descending_sort_is_never_elided(t, duck):
    got = t.sort("a").select(x=col("a")).sort("x", descending=True).collect()
    assert_same_ordered(got, duck.sql("SELECT a AS x FROM t ORDER BY x DESC"))


def test_nulls_are_ordered_the_same_way(t, duck):
    """The canonical form is nulls-last; the null row is where the sort bug would show."""
    got = t.sort("a").select(x=col("a"), y=col("b")).sort("x").collect()
    assert got.to_pydict()["x"][-1] is None
    assert_same_ordered(got, duck.sql("SELECT a AS x, b AS y FROM t ORDER BY x NULLS LAST"))
