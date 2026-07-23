"""Plan-shape, idempotence, and refusal tests for the `pushdown_gaps` rules.

Every rule gets a fires-test *and* a no-op test for each shape it must refuse — the
refusals are the load-bearing half here, because each one is a wrong answer, not a slow
plan: a filter below a `RowId` renumbers the rows, below a fixed-count `Sample` re-picks
them, and on an ASOF join's right side re-points the nearest match. Correctness vs DuckDB
lives in tests/differential/test_diff_pushdown_gaps.py.

Importing the rule module registers its `@rule` decorators into `DEFAULT_REGISTRY`.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.pushdown_gaps import (
    drop_dead_row_index,
    prefilter_unnest_by_list_contains,
    prune_asof_join_output,
    prune_union_columns_under_aggregate,
    prune_union_columns_under_join,
    prune_union_columns_under_unpivot,
    push_filter_into_asof_by_keys,
    push_filter_into_unpivot_columns,
    push_filter_through_asof_join,
    push_filter_through_sample,
    push_filter_through_unnest,
    push_filter_through_unpivot,
)
from batcher.plan.logical import (
    AsofJoin,
    Filter,
    Project,
    RowId,
    Sample,
    Scan,
    Unnest,
    Unpivot,
)

pytestmark = pytest.mark.unit

RULE_NAMES = {
    "drop_dead_row_index",
    "prefilter_unnest_by_list_contains",
    "prune_asof_join_output",
    "prune_union_columns_under_aggregate",
    "prune_union_columns_under_join",
    "prune_union_columns_under_unpivot",
    "push_filter_into_asof_by_keys",
    "push_filter_into_unpivot_columns",
    "push_filter_through_asof_join",
    "push_filter_through_sample",
    "push_filter_through_unnest",
    "push_filter_through_unpivot",
}


def _lists():
    return bt.from_pydict(
        {"id": [1, 2, 3], "xs": [[1, 2], [3], []], "g": ["a", "b", "a"]},
    )


def _wide():
    return bt.from_pydict({"id": [1, 2], "a": [10, 20], "b": [30, 40], "z": [7, 8]})


def _left():
    return bt.from_pydict({"t": [1, 5, 10], "v": ["a", "b", "c"], "g": ["x", "y", "x"]})


def _right():
    return bt.from_pydict({"t": [2, 6], "w": ["p", "q"], "g": ["x", "y"]})


def _branch(k):
    return bt.from_pydict({"k": k, "v": [i * 2 for i in k], "z": [i * 3 for i in k]})


def test_rules_registered():
    assert {r.name for r in DEFAULT_REGISTRY.rules()} >= RULE_NAMES


# --- push_filter_through_unnest -----------------------------------------------


def test_filter_on_passthrough_pushed_below_unnest():
    plan = _lists().explode("xs").filter(col("id") > 1)._plan
    out = push_filter_through_unnest(plan, None)
    assert isinstance(out, Unnest)
    assert isinstance(out.input, Filter)


def test_filter_on_exploded_column_not_pushed_below_unnest():
    # `xs` is per-element above the Unnest — it does not exist below it.
    plan = _lists().explode("xs").filter(col("xs") == 3)._plan
    assert push_filter_through_unnest(plan, None) is None


def test_mixed_filter_splits_across_unnest():
    plan = _lists().explode("xs").filter((col("id") > 1) & (col("xs") == 3))._plan
    out = push_filter_through_unnest(plan, None)
    assert isinstance(out, Filter)  # the `xs` conjunct stays above
    assert isinstance(out.input, Unnest)
    assert isinstance(out.input.input, Filter)  # the `id` conjunct went below


def test_unnest_rule_is_idempotent():
    plan = _lists().explode("xs").filter(col("id") > 1)._plan
    out = push_filter_through_unnest(plan, None)
    assert push_filter_through_unnest(out, None) is None  # no Filter on top any more


# --- prefilter_unnest_by_list_contains ----------------------------------------


def test_eq_on_exploded_column_prefilters_the_list():
    plan = _lists().explode("xs").filter(col("xs") == 3)._plan
    out = prefilter_unnest_by_list_contains(plan, None)
    assert isinstance(out, Filter)  # the original filter is KEPT above
    assert isinstance(out.input, Unnest)
    assert isinstance(out.input.input, Filter)  # the list_contains guard
    assert out.input.input.predicate.to_ir()["e"] == "list_contains"


def test_prefilter_is_idempotent():
    plan = _lists().explode("xs").filter(col("xs") == 3)._plan
    out = prefilter_unnest_by_list_contains(plan, None)
    assert prefilter_unnest_by_list_contains(out, None) is None


def test_prefilter_refuses_non_equality():
    plan = _lists().explode("xs").filter(col("xs") > 1)._plan
    assert prefilter_unnest_by_list_contains(plan, None) is None


def test_prefilter_refuses_float_elements():
    # A float `-0.0`/NaN could make `list_contains` disagree with the post-explode `=`.
    ds = bt.from_pydict({"id": [1], "xs": [[1.0, 2.0]]})
    plan = ds.explode("xs").filter(col("xs") == 1.0)._plan
    assert prefilter_unnest_by_list_contains(plan, None) is None


# --- push_filter_through_unpivot ----------------------------------------------


def test_filter_on_index_pushed_below_unpivot():
    plan = _wide().unpivot(index=["id"], on=["a", "b"]).filter(col("id") == 1)._plan
    out = push_filter_through_unpivot(plan, None)
    assert isinstance(out, Unpivot)
    assert isinstance(out.input, Filter)


def test_filter_on_value_not_pushed_below_unpivot():
    plan = _wide().unpivot(index=["id"], on=["a", "b"]).filter(col("value") > 15)._plan
    assert push_filter_through_unpivot(plan, None) is None


def test_filter_on_variable_not_pushed_below_unpivot():
    plan = _wide().unpivot(index=["id"], on=["a", "b"]).filter(col("variable") == "a")._plan
    assert push_filter_through_unpivot(plan, None) is None


# --- push_filter_into_unpivot_columns -----------------------------------------


def test_variable_equality_prunes_the_melted_columns():
    plan = _wide().unpivot(index=["id"], on=["a", "b"]).filter(col("variable") == "b")._plan
    out = push_filter_into_unpivot_columns(plan, None)
    assert isinstance(out, Filter)  # the (now always-true) filter is kept
    assert out.input.on == ("b",)
    assert out.available_columns() == plan.available_columns()  # schema unmoved
    assert push_filter_into_unpivot_columns(out, None) is None  # idempotent


def test_variable_equality_refuses_mixed_types():
    # `value`'s type is the promotion of every `on` column's; narrowing `on` here would
    # change it from Float64 to Int64 — a schema change, which is never allowed.
    ds = bt.from_pydict({"id": [1], "a": [1], "b": [2.5]})
    plan = ds.unpivot(index=["id"], on=["a", "b"]).filter(col("variable") == "a")._plan
    assert push_filter_into_unpivot_columns(plan, None) is None


def test_variable_inequality_refuses():
    plan = _wide().unpivot(index=["id"], on=["a", "b"]).filter(col("variable") != "b")._plan
    assert push_filter_into_unpivot_columns(plan, None) is None


# --- push_filter_through_sample -----------------------------------------------


def test_filter_pushed_through_fraction_sample():
    plan = _lists().sample(0.5, seed=7).filter(col("id") > 1)._plan
    out = push_filter_through_sample(plan, None)
    assert isinstance(out, Sample)
    assert isinstance(out.input, Filter)
    assert out.fraction == 0.5 and out.seed == 7


def test_filter_not_pushed_through_fixed_count_sample():
    # `sample(n=k)` keeps the k smallest-hash rows of the WHOLE input — filtering first
    # promotes rows that would have lost, so the sampled set changes.
    plan = _lists().sample(n=2, seed=7).filter(col("id") > 1)._plan
    assert push_filter_through_sample(plan, None) is None


# --- push_filter_through_asof_join / push_filter_into_asof_by_keys -------------


def test_filter_on_left_columns_pushed_into_asof():
    plan = _left().join_asof(_right(), on="t", by="g").filter(col("v") == "a")._plan
    out = push_filter_through_asof_join(plan, None)
    assert isinstance(out, AsofJoin)
    assert isinstance(out.left, Filter)
    assert isinstance(out.right, Scan)  # the right side is untouched


def test_filter_on_right_columns_not_pushed_into_asof():
    # THE load-bearing refusal: ASOF picks the nearest right row first and the predicate
    # sees it only afterwards. Pre-filtering the right side promotes a farther row.
    plan = _left().join_asof(_right(), on="t", by="g").filter(col("w") == "p")._plan
    assert push_filter_through_asof_join(plan, None) is None


def test_asof_by_key_predicate_mirrors_onto_the_right():
    plan = _left().filter(col("g") == "x").join_asof(_right(), on="t", by="g")._plan
    out = push_filter_into_asof_by_keys(plan, None)
    assert isinstance(out.right, Filter)  # the `by` constraint reached the build side
    assert push_filter_into_asof_by_keys(out, None) is None  # idempotent


def test_asof_non_by_left_predicate_does_not_mirror():
    # `v` is not a `by` key: it says nothing about which right rows are reachable.
    plan = _left().filter(col("v") == "a").join_asof(_right(), on="t", by="g")._plan
    assert push_filter_into_asof_by_keys(plan, None) is None


def test_asof_without_by_keys_does_not_mirror():
    plan = _left().filter(col("v") == "a").join_asof(_right(), on="t")._plan
    assert push_filter_into_asof_by_keys(plan, None) is None


# --- prune_asof_join_output ---------------------------------------------------


def test_asof_output_pruned_to_what_the_projection_reads():
    plan = _left().join_asof(_right(), on="t", by="g").select("v")._plan
    out = prune_asof_join_output(plan, None)
    assert [c.alias for c in out.input.output] == ["v"]
    assert out.available_columns() == ["v"]  # the projection's schema is unmoved
    assert prune_asof_join_output(out, None) is None  # idempotent


def test_asof_output_not_pruned_when_everything_is_read():
    ds = _left().join_asof(_right(), on="t", by="g")
    plan = ds.select(*ds.columns)._plan
    assert prune_asof_join_output(plan, None) is None


# --- drop_dead_row_index ------------------------------------------------------


def test_dead_row_index_dropped_under_project():
    plan = bt.from_pydict({"k": [1, 2]}).with_row_index().select("k")._plan
    out = drop_dead_row_index(plan, None)
    assert isinstance(out, Project)
    assert isinstance(out.input, Scan)


def test_dead_row_index_dropped_under_aggregate():
    ds = bt.from_pydict({"k": [1, 2], "v": [3, 4]}).with_row_index()
    plan = ds.group_by("k").agg(s=col("v").sum())._plan
    out = drop_dead_row_index(plan, None)
    assert isinstance(out.input, Scan)


def test_live_row_index_is_kept():
    plan = bt.from_pydict({"k": [1, 2]}).with_row_index().select("index", "k")._plan
    assert drop_dead_row_index(plan, None) is None


def test_no_filter_rule_descends_past_a_live_row_index():
    # Numbering is positional: pushing a filter below RowId would renumber the survivors.
    plan = bt.from_pydict({"k": [1, 2, 3]}).with_row_index().filter(col("k") > 1)._plan
    assert isinstance(plan, Filter) and isinstance(plan.input, RowId)
    for fire in (
        push_filter_through_unnest,
        push_filter_through_unpivot,
        push_filter_through_sample,
        push_filter_through_asof_join,
        prefilter_unnest_by_list_contains,
        push_filter_into_unpivot_columns,
    ):
        assert fire(plan, None) is None
    # And the full optimizer leaves the RowId above the Filter.
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "filter" and ir["input"]["op"] == "row_id"


# --- prune_union_columns_under_* ----------------------------------------------


def test_union_branches_pruned_under_aggregate():
    plan = _branch([1, 2]).union(_branch([3])).group_by("k").agg(s=col("v").sum())._plan
    out = prune_union_columns_under_aggregate(plan, None)
    assert out.input.available_columns() == ["k", "v"]  # `z` is gone, order preserved
    assert prune_union_columns_under_aggregate(out, None) is None  # idempotent


def test_distinct_union_branches_are_not_pruned():
    # `UNION` (distinct) dedups over EVERY column — dropping one merges distinct rows.
    plan = (
        _branch([1, 2]).union(_branch([3]), distinct=True).group_by("k").agg(s=col("v").sum())._plan
    )
    assert prune_union_columns_under_aggregate(plan, None) is None


def test_union_branches_pruned_under_join():
    right = bt.from_pydict({"k": [1], "c": [9]})
    plan = _branch([1, 2]).union(_branch([3])).join(right, on="k")._plan
    narrowed = dataclasses.replace(  # the column pruner would narrow the output like this
        plan, output=tuple(o for o in plan.output if o.alias in {"k", "c"})
    )
    out = prune_union_columns_under_join(narrowed, None)
    assert out.left.available_columns() == ["k"]
    assert prune_union_columns_under_join(out, None) is None  # idempotent


def test_union_branches_not_pruned_under_join_when_all_read():
    right = bt.from_pydict({"k": [1], "c": [9]})
    plan = _branch([1, 2]).union(_branch([3])).join(right, on="k")._plan
    assert prune_union_columns_under_join(plan, None) is None


def test_union_branches_pruned_under_unpivot():
    ds = _branch([1, 2]).union(_branch([3]))
    plan = ds.unpivot(index=["k"], on=["v"])._plan
    out = prune_union_columns_under_unpivot(plan, None)
    assert out.input.available_columns() == ["k", "v"]
    assert out.available_columns() == plan.available_columns()  # schema unmoved
    assert prune_union_columns_under_unpivot(out, None) is None  # idempotent


# --- the whole optimizer still produces the right shape ------------------------


def test_optimizer_pushes_filter_below_explode():
    plan = _lists().explode("xs").filter(col("id") > 1)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "unnest"
    assert ir["input"]["op"] == "filter"


def test_optimizer_leaves_a_ranking_window_filter_alone():
    # A predicate on a rank output may not descend past the Window that computes it.
    ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    ranked = col("v").rank().over(partition_by=["g"], order_by=["v"])
    plan = ds.with_columns(r=ranked).filter(col("r") == 1)._plan
    ir = Optimizer().optimize(plan).ir
    # It fuses into the window's rank_limit, or stays above it — never below it.
    assert ir["op"] in {"window", "filter"}
    if ir["op"] == "filter":
        assert ir["input"]["op"] == "window"
