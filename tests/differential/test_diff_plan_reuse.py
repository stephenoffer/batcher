"""A terminal operation must not rewrite the plan it was handed.

A `Dataset` is a handle to an immutable `LogicalPlan`, and every builder returns a new one.
The terminals are supposed to read that plan, not edit it — but `LogicalPlan.to_ir()`
**memoizes per node and returns the node's own dict**, so any caller that re-roots the IR
by assigning into it edits the caller's plan, permanently and invisibly.

The out-of-core window path did exactly that: it re-rooted the window's IR on a bare scan of
its bucket (right) by writing `ir["input"] = ...` into the memoized dict (wrong). Streaming a
plan with a top-level window therefore *deleted whatever produced the window's input* from
the caller's plan. The next use of the same handle then ran

    window ← scan

where the plan said

    window ← project(x = ...) ← scan

which is either the wrong rows, or `RuntimeError: unknown column: x` when the window's
function reads a column the projection was the only thing producing. Nothing raised at the
point of corruption, and a *freshly built* handle of the identical shape worked — which is
why this survived: every test builds its plan and uses it once.

These tests pin the invariant directly (the IR is unchanged across a terminal) rather than
only its consequence, because the consequence is data-dependent and the invariant is not.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")


def _windowed():
    """A plan whose window reads a column only the projection beneath it produces."""
    table = pa.table(
        {
            "k": pa.array([1, 2, 2, 3, 3, 3], pa.int64()),
            "v": pa.array([10, 20, 30, 40, 50, 60], pa.int64()),
        }
    )
    ds = bt.from_arrow(table)
    return (
        ds.with_columns(__one=bt.lit(1))
        .window(partition_by=["k"], order_by=[], functions={"n": ("count", "__one")})
        .filter(bt.col("n") > 1)
        .select("k", "v")
    )


def test_streaming_leaves_the_plan_it_was_given_unchanged():
    plan = _windowed()
    before = json.dumps(plan._plan.to_ir(), sort_keys=True)
    list(plan.iter_batches())
    assert json.dumps(plan._plan.to_ir(), sort_keys=True) == before


def test_collect_after_streaming_the_same_handle_agrees_with_collect_alone():
    """The consequence: the second terminal must see the plan, not a rewrite of it."""
    oracle = _windowed().collect()
    plan = _windowed()
    list(plan.iter_batches())
    assert_tables_equal(plan.collect(), oracle)


def test_the_streamed_and_collected_results_agree():
    plan = _windowed()
    streamed = pa.Table.from_batches(list(plan.iter_batches()))
    assert_tables_equal(streamed, _windowed().collect())


def test_repeated_terminals_on_one_handle_stay_stable():
    """Three terminals in a row, each of which must see the same plan."""
    plan = _windowed()
    first = plan.collect()
    list(plan.iter_batches())
    assert_tables_equal(plan.collect(spill=True), first)
    assert_tables_equal(plan.collect(), first)


def test_a_quality_split_survives_being_streamed_first():
    """The shape that found this: a `dq.unique` split, whose two halves share one plan.

    It was chosen because `unique` lowered to a window over a derived column; it now lowers to
    a keyed dedup. What the test is about — the same handle surviving being streamed and then
    collected — does not depend on which, so the case is kept and the description corrected."""
    table = pa.table({"id": pa.array([1, 1, 2, 3, 3], pa.int64()), "v": ["a", "b", "c", "d", "e"]})
    clean, rejected = bt.from_arrow(table).dq.unique("id").quarantine()
    streamed = list(rejected.iter_batches())
    assert sum(b.num_rows for b in streamed) == 4
    # The same handle again, after streaming: the plan must still be the one that was built.
    assert rejected.collect().num_rows == 4
    assert clean.collect().num_rows == 1
