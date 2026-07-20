"""Kyber must walk *every* `Expr` node type when pruning columns and inlining projections.

An optimizer that treats a node type as a leaf either prunes a column the node actually reads
(a crash: ``unknown column``) or inlines a projection's defining expression everywhere *except*
inside that node (a silent wrong result: the node keeps reading a stale/raw column). Both are the
same class of bug — an omission in a structural `Expr` visitor — and both are exercised here:

* ``referenced_columns`` (column-pruning input analysis) once omitted ``Strptime``, so
  ``col("s").str.to_datetime(...)`` reported reading no columns and the scan dropped ``s``.
* ``transform_expr_up`` / ``substitute_columns`` (projection inlining) once treated ``MakeStruct``
  and ``Sequence`` as leaves, so ``merge_projections`` folded two projections without rewriting
  the column references *inside* a struct/sequence, reading the pre-projection column instead.

Each test builds a plan Kyber will optimize (stacked projections / a prunable scan) and checks
the answer still matches DuckDB (or the known-correct value the un-inlined plan produces).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")


def test_strptime_column_not_pruned(duck):
    """`str.to_datetime` must keep its input column alive through projection pruning."""
    # ``a`` deliberately precedes ``s``: column pruning keeps the *first* column as its
    # cardinality-preserving fallback, so an analysis that reports ``s`` as read-by-nothing
    # prunes ``s`` and the scan fails with ``unknown column: s``.
    t = pa.table({"a": [1, 2, 3], "s": ["2021-01-01", "2021-06-02", "2022-03-04"]})
    got = bt.from_arrow(t).select(bt.col("s").str.to_datetime("%Y-%m-%d").alias("dt")).collect()
    duck.register("t", t)
    assert_same(got, duck.sql("SELECT strptime(s, '%Y-%m-%d') AS dt FROM t"))


def test_merge_projection_inlines_into_struct():
    """Stacked projections that build a struct from a recomputed column must inline correctly.

    `merge_projections` folds the two `select`s into one. If substitution does not descend into
    the `struct(...)`, the merged projection's struct reads the raw source ``a`` (1, 2, 3) instead
    of the recomputed ``a`` (10, 20, 30) — a silent wrong result.
    """
    t = pa.table({"a": [1, 2, 3]})
    got = (
        bt.from_arrow(t)
        .select((bt.col("a") * 10).alias("a"))
        .select(bt.struct(v=bt.col("a")).alias("s"))
        .select(bt.col("s").struct.field("v").alias("out"))
        .collect()
    )
    assert_tables_equal(got, pa.table({"out": pa.array([10, 20, 30], pa.int64())}))


def test_merge_projection_inlines_into_sequence():
    """The `Sequence` sibling of the struct case — inlining must reach a sequence's bounds."""
    t = pa.table({"a": [1, 2, 3]})
    got = (
        bt.from_arrow(t)
        .select((bt.col("a") * 10).alias("a"))
        .select(bt.sequence(bt.col("a"), bt.col("a")).alias("s"))
        .collect()
    )
    assert_tables_equal(got, pa.table({"s": pa.array([[10], [20], [30]], pa.list_(pa.int64()))}))
