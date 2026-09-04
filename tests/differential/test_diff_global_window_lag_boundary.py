"""A global `lag` split into ordered buckets must equal the single-node one, at any cut.

`lag` is the one ordered-bucket correction that reads a *neighbouring* bucket rather than a
running scalar. Every other function in the algebra is handed what the prior buckets
accumulated -- a row count, a fold, a moment triple -- and needs nothing else from them; a row
`k` places into a bucket has its source row `k` places back, which for the bucket's first `k`
rows is in the bucket before it. The window kernel run on a bucket alone returns NULL there,
so before the boundary exchange existed the whole relation had to be windowed on one node,
and on distributed data `_unsupported` **raised** rather than doing that silently.

What makes it worth its own file rather than a row in the fold table is that its correctness
depends on the *cut* in a way the running scalars' does not. A fold that is wrong by a bucket
boundary is wrong everywhere; a `lag` that is wrong by a boundary is wrong on exactly `k` rows
per bucket, so a two-bucket run over 4,000 rows can hide a defect that a thirteen-bucket run
exposes. Hence the parametrization over partition count, and the lag distances that straddle
the interesting cases: `1` (the common one, and the only one where the head and the tail are a
single row), a distance short of a bucket, and one **longer than a bucket**, where the value a
row needs is not in the previous bucket at all but two or more back.

The `descending` arm is not decoration either. Bucket 0 always holds the lowest keys, so a
descending window has to be *walked* highest-first; a boundary exchange that carried its tail
in the visit order but read it in the key order would pass every ascending case here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

_N = 4000


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    """Duplicated order keys, so peer groups straddle what would otherwise be a clean cut."""
    return pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "o": pa.array([(i * 37) % 401 for i in range(_N)], pa.int64()),
            # Nulls in the *value*: a lag reads them back like any other value, and a boundary
            # exchange that dropped them would silently shift every later row by one.
            "v": pa.array([None if i % 7 == 0 else float(i % 29) for i in range(_N)], pa.float64()),
        }
    )


def _by_rid(table: pa.Table) -> pa.Table:
    """`table` in `rid` order — the row identity both paths carry, sorted outside the engine."""
    return table.sort_by([("rid", "ascending")])


@pytest.mark.parametrize("partitions", [2, 3, 7, 13])
@pytest.mark.parametrize("distance", [1, 3, 50, 900])
@pytest.mark.parametrize("descending", [False, True])
def test_a_split_lag_equals_the_single_node_one(rows, distance, partitions, descending):
    order = [("o", True)] if descending else ["o"]
    ds = bt.from_arrow(rows).window(order_by=order, functions={"w": ("lag", bt.col("v"), distance)})
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=partitions)),
        _by_rid(ds.collect()),
        ordered=True,
    )


@pytest.mark.parametrize("size", [0, 1, 2])
def test_a_relation_shorter_than_the_lag_is_all_null(size):
    """The degenerate end: nothing has gone past, so every row's `lag` is NULL on both paths.

    This is where an off-by-one in the boundary index shows as an `IndexError` rather than a
    wrong number — it did, while this was being written — and where a split path that invented
    a value would be most obviously wrong.
    """
    table = pa.table(
        {
            "rid": pa.array(range(size), pa.int64()),
            "o": pa.array(range(size), pa.int64()),
            "v": pa.array([float(i) for i in range(size)], pa.float64()),
        }
    )
    ds = bt.from_arrow(table).window(order_by=["o"], functions={"w": ("lag", bt.col("v"), 5)})
    spilled = ds.collect(spill=True, num_partitions=3)
    assert_tables_equal(_by_rid(spilled), _by_rid(ds.collect()), ordered=True)
    assert spilled.column("w").to_pylist() == [None] * size


def test_the_split_actually_produces_several_buckets(rows):
    """Guard against a vacuous sweep.

    Every assertion above compares a split run with a whole one. If the split silently stopped
    splitting — a declined predicate, a bucket count collapsing to one — the comparison would
    hold trivially and the file would report that a boundary exchange it never exercised was
    correct. A `lag` whose distance exceeds the whole relation is the control: split or not, it
    is all NULL, so it cannot distinguish the two, and the assertion below is instead that the
    algebra *claims* this shape.
    """
    from batcher.dist.global_window import supports_ordered_bucket_offsets

    ds = bt.from_arrow(rows).window(order_by=["o"], functions={"w": ("lag", bt.col("v"), 2)})
    assert supports_ordered_bucket_offsets(ds._plan) is True
    # And one function it still declines, so the predicate is not uniformly permissive.
    lead = bt.from_arrow(rows).window(order_by=["o"], functions={"w": ("lead", bt.col("v"), 2)})
    assert supports_ordered_bucket_offsets(lead._plan, assembled=True) is False
