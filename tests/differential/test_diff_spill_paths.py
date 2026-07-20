"""Differential tests for the out-of-core (`spill=True`) path, vs DuckDB.

The spilling operators are a *second consumer* of the mergeable primitives, not a second
implementation of them — so every shape that works in memory must work spilled, with the
identical result. These tests pin that.

They exist because it was not pinned: `collect(spill=True)` on a `descending` sort used to
return **unsorted** data (nulls emitted mid-result), because the out-of-core sort hand-rolled
its own range partitioner instead of calling the shared `range_partition_batches`, and got the
null-bucket end wrong for the reversed emission order. Nothing caught it — the ordinary
differential harness (`assert_same`) is order-*independent*, so it is structurally blind to a
sort bug, and no test combined `spill=True` with `descending`. Hence: ordered assertions, and
the ordering flags in the parameter matrix.
"""

from __future__ import annotations

import itertools

import pyarrow as pa
import pytest

from _harness import assert_same, assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

ORDERINGS = list(itertools.product([False, True], [False, True]))  # (descending, nulls_first)


@pytest.fixture
def spill_table() -> pa.Table:
    """Nulls, duplicates, negative values and a string column — one table, many shapes."""
    return pa.table(
        {
            "k": pa.array([3, 1, None, 10, 7, None, 0, 5, 9, None, 2, 8, 4, None, 6], pa.int64()),
            "g": pa.array(
                ["a", "b", "a", "c", None, "b", "a", "c", "b", None, "a", "c", "b", "a", "c"]
            ),
            "v": pa.array([5, 3, 9, 1, 4, 8, 2, 7, 6, 0, 5, 3, 8, 1, 2], pa.int64()),
        }
    )


def _register(duck, table: pa.Table, name: str = "t") -> None:
    duck.register(name, table)


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
def test_spilling_sort_matches_duckdb(duck, spill_table, descending, nulls_first):
    """A spilled sort is byte-for-byte the sort DuckDB produces — every ordering flag.

    The regression: `descending=True` used to interleave nulls into the middle of the
    result, because the null bucket was chosen from `nulls_first` alone while the buckets
    were *emitted* in reverse. Ordered assertion — an unordered one cannot see this.
    """
    _register(duck, spill_table)
    out = (
        bt.from_arrow(spill_table)
        .sort(bt.col("k"), descending=descending, nulls_first=nulls_first)
        .collect(spill=True)
    )
    direction = "DESC" if descending else "ASC"
    nulls = "FIRST" if nulls_first else "LAST"
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k {direction} NULLS {nulls}"))


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
def test_spilling_sort_equals_in_memory_sort(spill_table, descending, nulls_first):
    """The spilled sort equals the in-memory sort — one operator, two schedulings."""
    plan = bt.from_arrow(spill_table).sort(
        bt.col("k"), descending=descending, nulls_first=nulls_first
    )
    assert_tables_equal(plan.collect(spill=True), plan.collect(), ordered=True)


@pytest.mark.parametrize("descending", [False, True])
def test_spilling_top_n_matches_duckdb(duck, spill_table, descending):
    """A spilled top-N (sort + limit) stops early but still emits in key order."""
    _register(duck, spill_table)
    out = (
        bt.from_arrow(spill_table)
        .sort(bt.col("k"), descending=descending, nulls_first=False)
        .limit(5)
        .collect(spill=True)
    )
    direction = "DESC" if descending else "ASC"
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k {direction} NULLS LAST LIMIT 5"))


def test_spilling_sort_on_string_key_falls_back_not_crashes(spill_table):
    """A non-numeric sort key cannot be range-partitioned; it must fall back, not raise.

    The KLL sketch that supplies the range boundaries is numeric-only. The gate used to admit
    any plain column and then die inside the partitioner (`TypeError` on a string subtraction).
    """
    plan = bt.from_arrow(spill_table).sort(bt.col("g"))
    assert_tables_equal(plan.collect(spill=True), plan.collect(), ordered=True)


def test_spilling_aggregate_matches_duckdb(duck, spill_table):
    """A spilled group-by equals DuckDB's, nulls in the key included."""
    _register(duck, spill_table)
    out = (
        bt.from_arrow(spill_table)
        .group_by("g")
        .agg(s=bt.col("v").sum(), n=bt.col("v").count())
        .collect(spill=True)
    )
    assert_same(out, duck.sql("SELECT g, SUM(v) AS s, COUNT(v) AS n FROM t GROUP BY g"))


def test_spilling_distinct_matches_duckdb(duck, spill_table):
    """A spilled DISTINCT equals DuckDB's — nulls compare equal, so they collapse to one."""
    _register(duck, spill_table)
    out = bt.from_arrow(spill_table).select(bt.col("g")).distinct().collect(spill=True)
    assert_same(out, duck.sql("SELECT DISTINCT g FROM t"))


def test_streaming_distinct_under_a_budget_matches_duckdb(duck):
    """A high-cardinality DISTINCT run through the *normal* `collect()` (not `spill=True`) under
    a tight memory budget must never OOM and must match DuckDB.

    This exercises the streaming executor's deferred-breaker path: DISTINCT is handed to the
    oracle, but under a budget the streaming driver bounds it — over budget it yields to the
    spilling parallel executor. The regression it guards: that deferral used to run the oracle
    *unbounded*, so a DISTINCT whose key set exceeded RAM crashed where the parallel path spills.
    """
    import numpy as np

    from batcher.config import Config, MemoryConfig, config_context

    rng = np.random.default_rng(5)
    n = 300_000
    t = pa.table(
        {
            "k": rng.integers(0, 250_000, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )
    _register(duck, t)
    # A 2 MB envelope is far smaller than the ~175k-group DISTINCT state, so the streaming
    # breaker goes over budget and must spill rather than OOM.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=2_000_000))
    with config_context(cfg):
        distinct = bt.from_arrow(t).select(bt.col("k")).distinct().collect()
        union = (
            bt.from_arrow(t)
            .select(bt.col("k"))
            .union(bt.from_arrow(t).select(bt.col("k")), distinct=True)
            .collect()
        )
    assert_same(distinct, duck.sql("SELECT DISTINCT k FROM t"))
    assert_same(union, duck.sql("SELECT DISTINCT k FROM t"))  # union-distinct of t with itself


def test_spilling_join_matches_duckdb(duck, spill_table):
    """A spilled hash join equals DuckDB's, including the null keys that never match."""
    right = pa.table(
        {"k": pa.array([1, 3, 5, 7, 9, None], pa.int64()), "w": ["p", "q", "r", "s", "u", "z"]}
    )
    _register(duck, spill_table)
    duck.register("r", right)
    out = (
        bt.from_arrow(spill_table)
        .join(bt.from_arrow(right), left_on="k", right_on="k")
        .collect(spill=True)
    )
    assert_same(out, duck.sql("SELECT * FROM t JOIN r USING (k)"))
