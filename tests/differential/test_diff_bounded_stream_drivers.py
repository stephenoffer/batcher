"""`sample(n=)` and `distinct(keys=...)` must stream, and return the collected rows.

Both materialized under `iter_batches()`, for opposite reasons that come to the same thing.
The router's aggregate/dedup branch excludes `plan.keys` because `Distinct.as_aggregate`
refuses a keyed dedup — its survivor carries columns the key does not determine, so folding
them with per-column aggregates would build a row that was never in the input. And the peeling
loop excludes a fixed-count `Sample` because it is not partition-independent: run per
partition it would keep `n` rows from *every* partition.

Neither objection applies to an incremental fold over an ordered stream. Both operators are
mergeable in their own right, so re-applying each to (running + batch) lands on the collected
answer with peak memory at the key count and at `n` rather than at the relation.

**Two things this file is careful about, both learned the hard way today.**

*The seed.* `sample(n=None-seeded)` bakes a fresh seed at plan-build, so comparing two
default-seeded plans shows a disagreement that is not a defect — it is two different queries.
Every sampling assertion here holds the seed fixed, and compares **rows**, not counts. A
count-only comparison passes on any sample of the right size, which is exactly what a broken
fold would produce.

*The comparator.* `assert_same` is set-based on columns as well as rows, and a keyed dedup's
whole output is "one row per key" — which row, and in what column order, is precisely what a
set comparison discards. So each row carries a unique `rid`, both sides are sorted by it
outside the engine, and the comparison is ordered.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

_N = 4000
_SEED = 20260826


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    return pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "k": pa.array([i % 23 for i in range(_N)], pa.int64()),
            "o": pa.array([(i * 7) % (_N + 1) for i in range(_N)], pa.int64()),
            "v": pa.array([f"r{i}" for i in range(_N)]),
        }
    )


def _by_rid(table: pa.Table) -> pa.Table:
    return table.sort_by([("rid", "ascending")])


def _streamed(ds) -> pa.Table:
    batches = list(ds.iter_batches())
    return pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)


def _materializes(ds) -> bool:
    """Whether the router fell through to building the whole result in memory."""
    import batcher.api.terminal.core as core

    reached = []
    original = core._collect
    core._collect = lambda *a, **k: (reached.append(1), original(*a, **k))[1]
    try:
        list(ds.iter_batches())
    finally:
        core._collect = original
    return bool(reached)


_SHAPES = {
    "sample_n_small": lambda ds: ds.sample(n=10, seed=_SEED),
    "sample_n_large": lambda ds: ds.sample(n=900, seed=_SEED),
    "distinct_on_keep_first": lambda ds: ds.distinct(["k"], order_by="o", keep="first"),
    "distinct_on_keep_last": lambda ds: ds.distinct(["k"], order_by="o", keep="last"),
    "distinct_on_arbitrary": lambda ds: ds.distinct(["k"]),
    # The controls: these already streamed, and they fail if a change here breaks the
    # branches that were already right.
    "sample_fraction": lambda ds: ds.sample(fraction=0.3, seed=_SEED),
    "distinct_whole_row": lambda ds: ds.select("k").distinct(),
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_streamed_result_is_the_collected_result(rows, shape):
    ds = _SHAPES[shape](bt.from_arrow(rows))
    collected = ds.collect()
    streamed = _streamed(ds)
    if "rid" in collected.column_names:
        assert_tables_equal(_by_rid(streamed), _by_rid(collected), ordered=True)
    else:
        assert_tables_equal(
            streamed.sort_by([(c, "ascending") for c in collected.column_names]),
            collected.sort_by([(c, "ascending") for c in collected.column_names]),
            ordered=True,
        )


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_it_actually_streams(rows, shape):
    """The point of the change. Without this every assertion above passes on the
    materializing path, which is the state these shapes were already in."""
    assert not _materializes(_SHAPES[shape](bt.from_arrow(rows))), (
        f"{shape} materialized instead of streaming"
    )


def test_a_default_seeded_sample_is_a_different_query_each_time(rows):
    """The control for the seed discipline in this file's docstring.

    If this ever starts passing as equal, `seed=None` has stopped baking a fresh seed — and
    every other sampling assertion here would then be holding a variable that no longer
    varies, which is a test that has quietly stopped testing.
    """
    ds = bt.from_arrow(rows)
    first = ds.sample(n=10).collect().column("rid").to_pylist()
    second = ds.sample(n=10).collect().column("rid").to_pylist()
    assert sorted(first) != sorted(second)
    held = ds.sample(n=10, seed=_SEED).collect().column("rid").to_pylist()
    again = ds.sample(n=10, seed=_SEED).collect().column("rid").to_pylist()
    assert sorted(held) == sorted(again)
