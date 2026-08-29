"""`rollup` / `cube` / `grouping_sets` on every execution path, including no rows at all.

The three grouping-set operators do not lower to one aggregate. They lower to a **union** of
aggregates, one per grouping set, and that shape is why they belong in a file of their own:
every other relational operator hands the streaming router a node with a single `input`, and
these hand it a `Union` with `inputs`.

The router dereferenced `plan.input` before checking what it had been given, so
`iter_batches()` on a `rollup` over an empty relation raised
``AttributeError: 'Union' object has no attribute 'input'`` — a Python internal, from a query
`collect()` answered correctly, on the one input shape (no rows) that folds the grouping sets
to a union of constants and so reaches the router with the union still on top.

The empty case is therefore not an edge case here, it is the case. It is kept beside a
single-row and a multi-morsel input so the file says what it means: these operators agree
across the three paths at every size, not just the one that broke.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_N = 40_000  # several morsels, so the three paths are genuinely different schedulings


def _table(shape: str) -> pa.Table:
    n = {"empty": 0, "one": 1}.get(shape, _N)
    idx = range(n)
    return pa.table(
        {
            "a": pa.array([i % 5 for i in idx], pa.int64()),
            "b": pa.array([f"g{i % 3}" for i in idx]),
            "v": pa.array([None if i % 7 == 0 else i % 29 for i in idx], pa.int64()),
        }
    )


_BUILDERS = {
    "rollup": lambda ds: ds.rollup("a", "b").agg(n=bt.col("v").count()),
    "cube": lambda ds: ds.cube("a", "b").agg(n=bt.col("v").count()),
    # `grouping_sets` takes one *argument* per level, not a list of levels; `()` is the
    # grand total.
    "grouping_sets": lambda ds: ds.grouping_sets(["a"], ["b"], ()).agg(
        n=bt.col("v").count(), s=bt.col("v").sum()
    ),
    # A grouping set feeding a further operator: the union is then no longer the root, which
    # is the arrangement the router was accidentally correct for.
    "rollup_then_filter": lambda ds: (
        ds.rollup("a", "b").agg(n=bt.col("v").count()).filter(bt.col("n") > 0)
    ),
}


def _canonical(table: pa.Table) -> tuple:
    """Column types, and the rows as a multiset — a grouping-set result has no row order."""
    names = sorted(table.column_names)
    types = [str(table.schema.field(name).type) for name in names]
    data = table.to_pydict()
    rows = sorted(
        (tuple(data[name][i] for name in names) for i in range(table.num_rows)),
        key=lambda row: tuple(repr(v) for v in row),
    )
    return (names, types, rows)


def _run(dataset, path: str) -> pa.Table:
    if path == "collect":
        return dataset.collect()
    if path == "spill":
        return dataset.collect(spill=True, num_partitions=5)
    batches = list(dataset.iter_batches())
    return pa.Table.from_batches(batches) if batches else dataset.collect().slice(0, 0)


@pytest.mark.parametrize("path", ["spill", "iter"])
@pytest.mark.parametrize("shape", ["multibatch", "empty", "one"])
@pytest.mark.parametrize("op", sorted(_BUILDERS))
def test_the_grouping_sets_agree_across_paths(op, shape, path):
    rows = _table(shape)
    build = _BUILDERS[op]
    expected = _canonical(_run(build(bt.from_arrow(rows)), "collect"))
    got = _canonical(_run(build(bt.from_arrow(rows)), path))
    assert got[0] == expected[0], f"columns {got[0]} vs {expected[0]}"
    assert got[1] == expected[1], f"types {got[1]} vs {expected[1]}"
    assert got[2] == expected[2]


def test_the_empty_case_is_not_trivially_empty():
    """Guard against a vacuous sweep on the shape this file was written for.

    A `rollup` over no rows is not a zero-row result: the grand total is still a group, so
    every path must produce exactly one row. If it ever became zero rows the comparisons
    above would compare two empties and pass while testing nothing.
    """
    empty = bt.from_arrow(_table("empty"))
    assert _BUILDERS["rollup"](empty).collect().num_rows == 1
    streamed = list(_BUILDERS["rollup"](bt.from_arrow(_table("empty"))).iter_batches())
    assert sum(batch.num_rows for batch in streamed) == 1
