"""A computed shuffle key must spill and stream out-of-core, not materialize.

`collect(spill=True)` cuts a sort into ordered pieces and a window into per-partition ones,
using the same partitioners the cluster uses — and a partitioner reads a key's *values*, so
it can only read them from a column. `sort(col("a") + col("b"))` and
`partition_by=[col("v") % 4]` have no column to read.

The distributed dispatcher solved that by materializing the key as a hidden column below the
breaker and projecting it away above (`plan.logical.hoist_sort_key` / `hoist_window_keys`).
Neither `collect(spill=True)` nor `iter_batches()` did, because the rewrite was private to
`dist/executor.py` — so the identical query distributed, then declined to spill under the
very memory envelope the spill exists for, and materialized its whole result from the one
entry point whose promise is that it does not. All three now share one definition.

The sort needed a second fix to benefit, and it is the more interesting half:
`supports_spilling_sort` read the key's type out of the **source's** schema, and a hoisted
key is by construction not in it — so the predicate answered "unknown type, decline" for
every hoisted key and the rewrite could never take effect. It now asks the sort's own
`available_schema` first, which is the plan layer's static inference and what the global
window's sibling predicate already used.

Two things have to hold and only one of them is obvious:

* the answer is the single-node answer;
* the hidden column does **not** reach the caller. It is real data in the spilled result, so
  a path that hoists and then declines — or one that forgets to project — returns an extra
  `__sort_key_0` column, which is a schema divergence between `collect()` and
  `collect(spill=True)` on the same query.

Row order is not under test and the two paths legitimately differ on it (the spilled path
emits piece by piece), so each row carries a unique `rid` and both results are sorted by it
outside the engine before an ordered comparison.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

_N = 2000


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    rng = random.Random(20260826)
    return pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "a": pa.array([rng.randint(0, 60) for _ in range(_N)], pa.int64()),
            "b": pa.array([rng.randint(0, 60) for _ in range(_N)], pa.int64()),
        }
    )


def _by_rid(table: pa.Table) -> pa.Table:
    return table.sort_by([("rid", "ascending")])


#: Each builds a breaker whose shuffle key is an expression rather than a column, alongside
#: the plain-column spelling that already worked — so a regression that disables the hoist
#: entirely still fails, and one that breaks the ordinary path fails too.
_SHAPES = {
    "computed_sort_key": lambda ds: ds.sort(bt.col("a") + bt.col("b")),
    "plain_sort_key": lambda ds: ds.sort("a"),
    "computed_window_key": lambda ds: ds.window(
        partition_by=[bt.col("a") % 4], order_by=["b"], functions={"r": "row_number"}
    ),
    "plain_window_key": lambda ds: ds.window(
        partition_by=["a"], order_by=["b"], functions={"r": "row_number"}
    ),
    # A breaker keyed on a column the plan *derived*, with a projection above that drops it.
    # The out-of-core stage narrows its read to what the stage needs, and the narrowing is
    # computed from the **source's** columns — so subtracting it from the breaker's input
    # removes the derived key, and the breaker then raises `ColumnNotFoundError` on a plan
    # that answered `collect()` perfectly. A hard failure rather than a silent one, and only
    # on the out-of-core paths.
    "derived_sort_key_projected_away": lambda ds: (
        ds.with_columns(m=bt.col("a") * 2).sort("m").select("rid", "a")
    ),
    "derived_window_key_projected_away": lambda ds: (
        ds.with_columns(m=bt.col("a") % 4)
        .window(partition_by=["m"], order_by=["b"], functions={"r": "row_number"})
        .select("rid", "b", "r")
    ),
    # Both rewrites at once, which is where they could contradict each other: the stage first
    # narrows its read to the columns it can prove it needs, and *then* the computed key is
    # materialized into a hidden column out of what survived. A narrowing that dropped a
    # column the key expression reads would leave nothing to hoist from.
    "computed_key_and_a_projection_above": lambda ds: ds.sort(bt.col("a") + bt.col("b")).select(
        "rid", "a"
    ),
    "computed_window_key_and_a_projection_above": lambda ds: ds.window(
        partition_by=[bt.col("a") % 4], order_by=["b"], functions={"r": "row_number"}
    ).select("rid", "a", "r"),
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
@pytest.mark.parametrize("partitions", [2, 5])
def test_a_spilled_computed_key_equals_the_in_memory_answer(rows, shape, partitions):
    ds = _SHAPES[shape](bt.from_arrow(rows))
    assert_tables_equal(
        _by_rid(ds.collect(spill=True, num_partitions=partitions)),
        _by_rid(ds.collect()),
        ordered=True,
    )


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_a_streamed_computed_key_equals_the_materialized_answer(rows, shape):
    """`iter_batches()` is the entry point the hoist matters most for: without it a computed
    key fell out of the streaming branch entirely and materialized the whole result."""
    ds = _SHAPES[shape](bt.from_arrow(rows))
    batches = list(ds.iter_batches())
    streamed = pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)
    assert_tables_equal(_by_rid(streamed), _by_rid(ds.collect()), ordered=True)


@pytest.mark.parametrize("descending", [False, True])
def test_a_streamed_computed_sort_is_globally_ordered(rows, descending):
    """The rows must come out in the sort's own order, not merely carry the right values.

    Asserted on the key directly rather than through `_by_rid`: every other test here
    deliberately sorts by row identity first, which is exactly the comparison that would let
    a bucket emitted out of range order pass.
    """
    ds = bt.from_arrow(rows).sort(bt.col("a") + bt.col("b"), descending=descending)
    batches = list(ds.iter_batches())
    streamed = pa.Table.from_batches(batches)
    keys = [
        a + b
        for a, b in zip(
            streamed.column("a").to_pylist(),
            streamed.column("b").to_pylist(),
            strict=True,
        )
    ]
    ordered = sorted(keys, reverse=descending)
    assert keys == ordered
    assert_tables_equal(streamed, ds.collect(), ordered=True)


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_hidden_shuffle_key_never_reaches_the_caller(rows, shape):
    """The materialized key is real data in the spilled result; the schema must not show it."""
    ds = _SHAPES[shape](bt.from_arrow(rows))
    expected = ds.collect().column_names
    spilled = ds.collect(spill=True, num_partitions=5)
    assert spilled.column_names == expected
    assert not [c for c in spilled.column_names if c.startswith("__")]
    for batch in ds.iter_batches():
        assert batch.schema.names == expected
