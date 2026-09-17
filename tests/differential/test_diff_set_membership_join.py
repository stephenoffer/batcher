"""INTERSECT / EXCEPT vs DuckDB on the inputs where Kyber turns them into semi and anti joins.

`set_membership_to_join` replaces the tagged-union aggregate with a join when every column is
proven null-free on at least one side. That makes NULL placement the axis that matters, so
the matrix is NULLs on the left only, the right only, split across columns, on both sides of
one column (where the rule must *not* fire), and none — crossed with the shapes that break a
join rather than an aggregate: empty sides, one row, duplicates, float keys carrying NaN and
-0.0, strings, and an input large enough to shard (past 65,536 rows) whose integer keys are
spread wide enough to take the join's key-range bitmap. Each result is checked under
`collect()`, `collect(spill=True)` and `iter_batches()`, and through SQL.

`assert_same` is order-independent, which is right here: neither operation defines an order.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_NAN = float("nan")

#: (left, right) column dicts. Every pair shares column names and types.
PAIRS: dict[str, tuple[dict, dict]] = {
    "no-nulls": ({"x": [1, 2, 2, 3, 4]}, {"x": [2, 4, 4, 9]}),
    "nulls-left-only": ({"x": [1, None, 2, None, 3]}, {"x": [2, 3, 7]}),
    "nulls-right-only": ({"x": [1, 2, 3, 3]}, {"x": [None, 3, None, 8]}),
    "nulls-split-across-columns": (
        {"x": [1, None, 2, 2], "s": ["a", "b", "c", "c"]},
        {"x": [1, 2, 5, 2], "s": ["a", None, "c", "c"]},
    ),
    "nulls-both-sides-one-column": (
        {"x": [None, 1, None, 2], "s": ["a", "b", "c", "d"]},
        {"x": [None, 1, 7, None], "s": ["a", "b", "z", "q"]},
    ),
    "empty-left": ({"x": pa.array([], pa.int64())}, {"x": [1, 2]}),
    "empty-right": ({"x": [1, 1, 2]}, {"x": pa.array([], pa.int64())}),
    "one-row-each": ({"x": [5]}, {"x": [5]}),
    "floats-nan-negzero": ({"f": [_NAN, -0.0, 1.5, _NAN, 2.5]}, {"f": [_NAN, 0.0, 3.5]}),
    "strings": ({"s": ["a", "b", "", "b", "c"]}, {"s": ["", "b", "d"]}),
    "sharded-wide-keys": (
        {"x": [i * 37 for i in range(90_000)]},
        {"x": [i * 37 * 11 for i in range(3_000)] + [5, 7]},
    ),
}


def _tables(name: str) -> tuple[pa.Table, pa.Table]:
    left, right = PAIRS[name]
    return pa.table(left), pa.table(right)


def _stream(ds: bt.Dataset) -> pa.Table:
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


@pytest.mark.parametrize("op", ["intersect", "except_"])
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_every_scheduling_matches_duckdb(duck, name, op):
    a, b = _tables(name)
    duck.register("a", a)
    duck.register("b", b)
    keyword = "INTERSECT" if op == "intersect" else "EXCEPT"
    oracle = duck.sql(f"SELECT * FROM a {keyword} SELECT * FROM b")
    ds = getattr(bt.from_arrow(a), op)(bt.from_arrow(b))
    assert_same(ds.collect(), oracle)
    assert_same(ds.collect(spill=True), oracle)
    assert_same(_stream(ds), oracle)
    assert_same(bt.sql(f"SELECT * FROM a {keyword} SELECT * FROM b", a=a, b=b).collect(), oracle)


def _uses_join(name: str) -> bool:
    a, b = _tables(name)
    return "hash_join" in bt.from_arrow(a).except_(bt.from_arrow(b)).explain()


def test_the_matrix_reaches_the_join():
    # A control in both directions: without it, every case above could be passing through
    # the aggregate (or all through the join) and this file would say nothing about the
    # rewrite it exists for.
    routed = {name: _uses_join(name) for name in PAIRS}
    assert routed["no-nulls"] and routed["nulls-left-only"] and routed["nulls-right-only"]
    assert routed["sharded-wide-keys"]
    assert routed["nulls-both-sides-one-column"] is False
