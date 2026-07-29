"""Plan-shape, idempotence, and does-not-fire tests for the `nullability` rules.

The family reasons from *declared* nullability (a source field's Arrow `nullable` flag), so
the tests that matter are the negatives: the same shape over a **nullable** column, over a
column whose nullability is **unknown** (below an aggregate, or a `map_batches` black box),
and in a **projection**, where a folded null check is a *value* the user sees rather than a
row that survives a filter. Result-correctness vs DuckDB lives in
`tests/differential/test_diff_nullability_rules.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.null_shapes as ns  # importing registers the rules
import batcher.kyber.rules.extra.nullability as nb  # importing registers the rules
from batcher import col, lit
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Col
from batcher.plan.expr_ir.constructors import coalesce
from batcher.plan.logical import Filter, Project, Sort, SortKeySpec
from batcher.plan.visitor import walk

RULE_NAMES = {
    "canonicalize_not_null_check",
    "count_of_non_nullable_column_to_count_star",
    "drop_coalesce_args_after_non_nullable",
    "drop_coalesce_of_non_nullable_first_arg",
    "drop_is_not_null_on_non_nullable_column",
    "drop_is_null_on_non_nullable_column",
    "drop_null_ordering_on_non_nullable_sort_key",
    "expand_is_not_null_of_coalesce",
    "expand_is_null_of_coalesce",
    "simplify_null_safe_comparison_on_non_nullable",
}

_TRUE = {"e": "lit", "value": {"bool": True}}
_FALSE = {"e": "lit", "value": {"bool": False}}


# --- fixtures ----------------------------------------------------------------


def _table() -> pa.Table:
    """`x` declared NOT NULL, `y` nullable, `f` a nullable float."""
    return pa.table(
        {
            "x": pa.array([1, 2, 3], type=pa.int64()),
            "y": pa.array([1, None, 3], type=pa.int64()),
            "f": pa.array([1.0, None, 3.0], type=pa.float64()),
        },
        schema=pa.schema(
            [
                pa.field("x", pa.int64(), nullable=False),
                pa.field("y", pa.int64()),
                pa.field("f", pa.float64()),
            ]
        ),
    )


@pytest.fixture
def ds():
    return bt.from_arrow(_table())


def _rewrite(dataset):
    """The plan after the REAL optimizer (every phase, every registered rule)."""
    return Optimizer(sources=dataset._sources).logical_rewrite(dataset._plan)


def _pred(node) -> dict:
    return node.predicate.to_ir()


def _item(node, i: int = 0) -> dict:
    return node.items[i].expr.to_ir()


def _first(plan, kind):
    return next(n for n in walk(plan) if isinstance(n, kind))


# --- registration -------------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert names >= RULE_NAMES


# --- IS NULL / IS NOT NULL over a NOT NULL column ------------------------------


def test_is_null_on_non_nullable_folds_to_false(ds):
    out = nb.drop_is_null_on_non_nullable_column(ds.filter(col("x").is_null())._plan, None)
    assert _pred(out) == _FALSE


def test_is_null_of_literal_folds_to_false(ds):
    # A `Lit` is the degenerate never-null expression — no separate rule needed.
    out = nb.drop_is_null_on_non_nullable_column(ds.select(r=lit(1).is_null())._plan, None)
    assert _item(out) == _FALSE


def test_is_null_of_derived_non_nullable_expression_folds(ds):
    out = nb.drop_is_null_on_non_nullable_column(ds.filter((col("x") + 1).is_null())._plan, None)
    assert _pred(out) == _FALSE


def test_is_null_is_idempotent(ds):
    once = nb.drop_is_null_on_non_nullable_column(ds.filter(col("x").is_null())._plan, None)
    assert nb.drop_is_null_on_non_nullable_column(once, None) is None


def test_is_null_does_not_fire_on_nullable_column(ds):
    assert nb.drop_is_null_on_non_nullable_column(ds.filter(col("y").is_null())._plan, None) is None


def test_is_null_does_not_fire_on_division(ds):
    # `div` can error rather than yield a value; it is not in the never-null whitelist.
    plan = ds.filter((col("x") / col("x")).is_null())._plan
    assert nb.drop_is_null_on_non_nullable_column(plan, None) is None


def test_is_null_does_not_fire_on_try_cast(ds):
    # `try_cast` *manufactures* NULL from an unconvertible non-null value.
    plan = ds.filter(col("x").try_cast("int64").is_null())._plan
    assert nb.drop_is_null_on_non_nullable_column(plan, None) is None


def test_is_not_null_on_non_nullable_folds_to_true(ds):
    out = nb.drop_is_not_null_on_non_nullable_column(ds.filter(col("x").is_not_null())._plan, None)
    assert _pred(out) == _TRUE


def test_is_not_null_of_literal_folds_to_true(ds):
    out = nb.drop_is_not_null_on_non_nullable_column(ds.select(r=lit(1).is_not_null())._plan, None)
    assert _item(out) == _TRUE


def test_is_not_null_does_not_fire_on_nullable_column(ds):
    plan = ds.filter(col("y").is_not_null())._plan
    assert nb.drop_is_not_null_on_non_nullable_column(plan, None) is None


# --- the NULL-vs-FALSE distinction inside a projection -------------------------


def test_null_check_folds_in_a_projection_because_it_is_total(ds):
    # `IS NULL` never yields NULL, so FALSE is its *value* on a NOT NULL column — the fold
    # is a value-level identity, not a filter-only one.
    out = nb.drop_is_null_on_non_nullable_column(ds.select(r=col("x").is_null())._plan, None)
    assert isinstance(out, Project)
    assert _item(out) == _FALSE


def test_nullable_null_check_is_never_folded_in_a_projection(ds):
    # Folding this would replace a *true* value with FALSE for row 2 — a wrong answer, not
    # merely a wrongly-dropped row.
    assert (
        nb.drop_is_null_on_non_nullable_column(ds.select(r=col("y").is_null())._plan, None) is None
    )


# --- nullability is unknown → no rewrite (never a guess) ------------------------


def test_no_rewrite_when_schema_is_unknown(ds):
    # Below an aggregate the analysis stops: an aggregate can emit NULL for a group.
    grouped = ds.group_by("y").agg(m=col("x").max())
    assert (
        nb.drop_is_null_on_non_nullable_column(grouped.filter(col("m").is_null())._plan, None)
        is None
    )


def test_no_rewrite_above_a_join(ds):
    # A join may null-extend a preserved side's counterpart, so the analysis stops there —
    # even for a column the *source* declares NOT NULL.
    joined = ds.join(bt.from_arrow(_table()), on="x", how="left")
    assert (
        nb.drop_is_null_on_non_nullable_column(joined.filter(col("x").is_null())._plan, None)
        is None
    )


# --- non-nullability propagated through a projection ---------------------------


def test_non_nullability_propagates_through_a_projection(ds):
    derived = ds.select(z=col("x") * 2)  # NOT NULL * literal → NOT NULL
    out = nb.drop_is_null_on_non_nullable_column(derived.filter(col("z").is_null())._plan, None)
    assert _pred(out) == _FALSE


def test_nullability_also_propagates_through_a_projection(ds):
    derived = ds.select(z=col("y") * 2)  # a nullable input taints the derived column
    plan = derived.filter(col("z").is_null())._plan
    assert nb.drop_is_null_on_non_nullable_column(plan, None) is None


# --- COALESCE ------------------------------------------------------------------


def test_coalesce_of_non_nullable_first_arg_unwraps(ds):
    out = nb.drop_coalesce_of_non_nullable_first_arg(ds.select(r=col("x").fill_null(0))._plan, None)
    assert _item(out) == Col("x").to_ir()


def test_coalesce_first_arg_is_idempotent(ds):
    once = nb.drop_coalesce_of_non_nullable_first_arg(
        ds.select(r=col("x").fill_null(0))._plan, None
    )
    assert nb.drop_coalesce_of_non_nullable_first_arg(once, None) is None


def test_coalesce_does_not_fire_on_nullable_first_arg(ds):
    plan = ds.select(r=col("y").fill_null(0))._plan
    assert nb.drop_coalesce_of_non_nullable_first_arg(plan, None) is None


def test_coalesce_does_not_fire_when_it_would_narrow_the_type(ds):
    # coalesce(int64_not_null, 1.5) is a DOUBLE; reducing it to `x` would make it a BIGINT.
    plan = ds.select(r=coalesce(col("x"), lit(1.5)))._plan
    assert nb.drop_coalesce_of_non_nullable_first_arg(plan, None) is None


def test_coalesce_truncates_after_a_non_nullable_arg(ds):
    plan = ds.select(r=coalesce(col("y"), col("x"), lit(0)))._plan
    out = nb.drop_coalesce_args_after_non_nullable(plan, None)
    assert _item(out) == coalesce(col("y"), col("x")).to_ir()


def test_coalesce_truncate_does_not_fire_when_all_args_are_nullable(ds):
    plan = ds.select(r=coalesce(col("y"), col("y"), col("y")))._plan
    assert nb.drop_coalesce_args_after_non_nullable(plan, None) is None


# --- IS NULL / IS NOT NULL pushed through COALESCE -----------------------------


def test_expand_is_null_of_coalesce(ds):
    out = ns.expand_is_null_of_coalesce(
        ds.filter(coalesce(col("y"), col("f")).is_null())._plan, None
    )
    assert _pred(out) == (col("y").is_null() & col("f").is_null()).to_ir()


def test_expand_is_null_of_coalesce_is_idempotent(ds):
    plan = ds.filter(coalesce(col("y"), col("f")).is_null())._plan
    once = ns.expand_is_null_of_coalesce(plan, None)
    assert ns.expand_is_null_of_coalesce(once, None) is None


def test_expand_is_not_null_of_coalesce(ds):
    plan = ds.filter(coalesce(col("y"), col("f")).is_not_null())._plan
    out = ns.expand_is_not_null_of_coalesce(plan, None)
    assert _pred(out) == (col("y").is_not_null() | col("f").is_not_null()).to_ir()


def test_expand_does_not_fire_without_a_coalesce(ds):
    assert ns.expand_is_null_of_coalesce(ds.filter(col("y").is_null())._plan, None) is None


# --- the null-safe comparison idiom -------------------------------------------


def test_null_safe_comparison_on_non_nullable_becomes_plain_equality(ds):
    plan = ds.filter(col("x").eq_missing(col("x")))._plan
    out = nb.simplify_null_safe_comparison_on_non_nullable(plan, None)
    assert _pred(out) == (col("x") == col("x")).to_ir()


def test_null_safe_comparison_kept_when_an_operand_is_nullable(ds):
    plan = ds.filter(col("x").eq_missing(col("y")))._plan
    assert nb.simplify_null_safe_comparison_on_non_nullable(plan, None) is None


# --- NOT canonicalization ------------------------------------------------------


def test_not_is_null_becomes_is_not_null(ds):
    out = ns.canonicalize_not_null_check(ds.filter(~col("y").is_null())._plan, None)
    assert _pred(out) == col("y").is_not_null().to_ir()


def test_not_is_not_null_becomes_is_null(ds):
    out = ns.canonicalize_not_null_check(ds.filter(~col("y").is_not_null())._plan, None)
    assert _pred(out) == col("y").is_null().to_ir()


def test_not_canonicalization_is_idempotent(ds):
    once = ns.canonicalize_not_null_check(ds.filter(~col("y").is_null())._plan, None)
    assert ns.canonicalize_not_null_check(once, None) is None


# --- COUNT(col) → COUNT(*) -----------------------------------------------------


def test_count_of_non_nullable_column_becomes_count_star(ds):
    plan = ds.group_by("y").agg(c=col("x").count())._plan
    out = ns.count_of_non_nullable_column_to_count_star(plan, None)
    assert out.aggregates[0].agg.func == "count_star"
    assert out.aggregates[0].agg.input is None


def test_count_of_nullable_column_is_left_alone(ds):
    plan = ds.group_by("x").agg(c=col("y").count())._plan
    assert ns.count_of_non_nullable_column_to_count_star(plan, None) is None


def test_count_star_rewrite_is_idempotent(ds):
    plan = ds.group_by("y").agg(c=col("x").count())._plan
    once = ns.count_of_non_nullable_column_to_count_star(plan, None)
    assert ns.count_of_non_nullable_column_to_count_star(once, None) is None


# --- sort: a null placement on a key that can hold no null ---------------------


def test_null_ordering_reset_on_non_nullable_sort_key(ds):
    plan = Sort(ds._plan, (SortKeySpec(Col("x"), False, True),))
    out = ns.drop_null_ordering_on_non_nullable_sort_key(plan, None)
    assert out.keys[0].nulls_first is False
    assert out.keys[0].descending is False


def test_null_ordering_kept_on_nullable_sort_key(ds):
    plan = Sort(ds._plan, (SortKeySpec(Col("y"), False, True),))
    assert ns.drop_null_ordering_on_non_nullable_sort_key(plan, None) is None


def test_null_ordering_reset_is_idempotent(ds):
    plan = Sort(ds._plan, (SortKeySpec(Col("x"), False, True),))
    once = ns.drop_null_ordering_on_non_nullable_sort_key(plan, None)
    assert ns.drop_null_ordering_on_non_nullable_sort_key(once, None) is None


# --- end to end: the rules fire through the REAL optimizer ---------------------


def test_optimizer_drops_a_filter_that_cannot_match(ds):
    # `x IS NULL` → FALSE → the canonical empty relation.
    plan = _rewrite(ds.filter(col("x").is_null()))
    assert not any(isinstance(n, Filter) for n in walk(plan))


def test_optimizer_drops_an_always_true_null_check(ds):
    plan = _rewrite(ds.filter(col("x").is_not_null()))
    assert not any(isinstance(n, Filter) for n in walk(plan))


def test_optimizer_unwraps_fill_null_on_a_non_nullable_column(ds):
    plan = _rewrite(ds.select(r=col("x").fill_null(0)))
    assert _item(_first(plan, Project)) == Col("x").to_ir()


def test_optimizer_rewrites_count_to_count_star(ds):
    plan = _rewrite(ds.group_by("y").agg(c=col("x").count()))
    aggs = [n for n in walk(plan) if hasattr(n, "aggregates")]
    assert any(spec.agg.func == "count_star" for n in aggs for spec in n.aggregates)


def test_optimizer_expands_and_folds_a_coalesce_null_check(ds):
    # coalesce(x, y) IS NULL → (x IS NULL AND y IS NULL) → (FALSE AND …) → FALSE → empty.
    plan = _rewrite(ds.filter(coalesce(col("x"), col("y")).is_null()))
    assert not any(isinstance(n, Filter) for n in walk(plan))


def test_optimizer_leaves_the_nullable_query_alone(ds):
    plan = _rewrite(ds.filter(col("y").is_null()))
    assert _pred(_first(plan, Filter)) == col("y").is_null().to_ir()
