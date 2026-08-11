"""`row_number() ... = 1` rewritten to `DISTINCT ON`, held against DuckDB.

`rank1_window_to_distinct_on` swaps the operator that answers this shape — a window that
ranks every row for one that keeps a single survivor per key. The rows must not move.

The interesting cases are the ones where the two operators have *freedom*: a tie on the
order key, a null in the key or the order column, an empty partition set. Those are where a
"faster equivalent" stops being equivalent, so they are what this file spends its length on.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col


def _rank1(ds, *, order=("v",), keys=("k",), out=("k", "v", "p")):
    """The `QUALIFY row_number() = 1` shape the rewrite consumes."""
    ranked = ds.window(partition_by=list(keys), order_by=list(order), functions={"r": "row_number"})
    return ranked.filter(col("r") == 1).select(*out)


def _register(duck, table):
    duck.register("t", table)
    return bt.from_arrow(table)


def test_earliest_row_per_key_matches_duckdb(duck):
    """The canonical dedup: one row per key, the smallest order value, payload carried."""
    ds = _register(
        duck,
        pa.table(
            {
                "k": [1, 1, 1, 2, 2, 3],
                "v": [30, 10, 20, 5, 15, 7],
                "p": ["a", "b", "c", "d", "e", "f"],
            }
        ),
    )
    assert_same(
        _rank1(ds).collect(),
        duck.sql("SELECT k, v, p FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"),
    )


def test_descending_order_selects_the_maximum(duck):
    """`ORDER BY v DESC` makes the survivor the per-key maximum, not the minimum."""
    ds = _register(
        duck,
        pa.table({"k": [1, 1, 2, 2], "v": [30, 10, 5, 15], "p": ["a", "b", "c", "d"]}),
    )
    assert_same(
        _rank1(ds, order=[("v", True)]).collect(),
        duck.sql(
            "SELECT k, v, p FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v DESC) = 1"
        ),
    )


def test_a_tie_on_the_order_key_still_yields_one_row_per_key(duck):
    """A tie is where the freedom is, and the row *count* is the part that is pinned.

    SQL does not say which of two rows tied on the order key `row_number()` calls first, so
    neither engine owes the other a particular payload — but both owe exactly one row per
    key, and the key and order columns are then determined. Comparing those and the count is
    the strongest claim that holds; asserting the payload too would be pinning an arbitrary
    choice and would fail for a reason that is not a defect.
    """
    ds = _register(
        duck,
        pa.table(
            {
                "k": [1, 1, 1, 2, 2],
                "v": [10, 10, 20, 5, 5],
                "p": ["a", "b", "c", "d", "e"],
            }
        ),
    )
    assert_same(
        _rank1(ds, out=("k", "v")).collect(),
        duck.sql("SELECT k, v FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"),
    )


def test_nulls_in_the_order_key(duck):
    """A null orders last under SQL's default, so it survives only for an all-null key."""
    ds = _register(
        duck,
        pa.table(
            {
                "k": [1, 1, 2, 2, 3],
                "v": [None, 10, None, None, 4],
                "p": ["a", "b", "c", "d", "e"],
            }
        ),
    )
    assert_same(
        _rank1(ds, out=("k", "v")).collect(),
        duck.sql("SELECT k, v FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"),
    )


def test_a_null_partition_key_is_its_own_group(duck):
    """NULL is a single grouping value here, as it is for `GROUP BY` — not "no group"."""
    ds = _register(
        duck,
        pa.table({"k": [None, None, 1, 1], "v": [9, 3, 8, 2], "p": ["a", "b", "c", "d"]}),
    )
    assert_same(
        _rank1(ds).collect(),
        duck.sql("SELECT k, v, p FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"),
    )


def test_a_compound_key_and_a_compound_ordering(duck):
    """Several partition keys and several order keys, since both are carried across."""
    ds = _register(
        duck,
        pa.table(
            {
                "k": [1, 1, 1, 2, 2],
                "g": ["x", "x", "y", "x", "x"],
                "v": [5, 5, 3, 9, 1],
                "p": ["a", "b", "c", "d", "e"],
            }
        ),
    )
    ranked = ds.window(partition_by=["k", "g"], order_by=["v", "p"], functions={"r": "row_number"})
    assert_same(
        ranked.filter(col("r") == 1).select("k", "g", "v", "p").collect(),
        duck.sql(
            "SELECT k, g, v, p FROM t "
            "QUALIFY row_number() OVER (PARTITION BY k, g ORDER BY v, p) = 1"
        ),
    )


def test_the_rank_column_itself_is_readable_after_the_rewrite(duck):
    """The rewrite drops the window, so `r` has to be restored — and it is always 1."""
    ds = _register(duck, pa.table({"k": [1, 1, 2], "v": [3, 1, 2], "p": ["a", "b", "c"]}))
    ranked = ds.window(partition_by=["k"], order_by=["v"], functions={"r": "row_number"})
    assert_same(
        ranked.filter(col("r") == 1).select("k", "r").collect(),
        duck.sql(
            "SELECT k, r FROM ("
            "  SELECT k, row_number() OVER (PARTITION BY k ORDER BY v) AS r FROM t"
            ") WHERE r = 1"
        ),
    )


def test_an_empty_relation_stays_empty(duck):
    """The degenerate input, which a hash-based survivor and a sort answer differently."""
    ds = _register(
        duck,
        pa.table(
            {
                "k": pa.array([], type=pa.int64()),
                "v": pa.array([], type=pa.int64()),
                "p": pa.array([], type=pa.string()),
            }
        ),
    )
    assert_same(
        _rank1(ds).collect(),
        duck.sql("SELECT k, v, p FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"),
    )


def test_top_two_is_left_alone(duck):
    """`= 1` is the only bound the rewrite claims; `<= 2` must still take the window path."""
    ds = _register(
        duck,
        pa.table(
            {
                "k": [1, 1, 1, 2, 2],
                "v": [30, 10, 20, 5, 15],
                "p": ["a", "b", "c", "d", "e"],
            }
        ),
    )
    ranked = ds.window(partition_by=["k"], order_by=["v"], functions={"r": "row_number"})
    assert_same(
        ranked.filter(col("r") <= 2).select("k", "v", "p").collect(),
        duck.sql(
            "SELECT k, v, p FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) <= 2"
        ),
    )
