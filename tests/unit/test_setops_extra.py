"""Plan-shape, idempotence, and does-not-fire tests for the `setops_extra` rules.

Every rule in the family is licensed by *set* semantics — an outer dedup dominating the
branches — so the negative that matters most is the same shape under `UNION ALL`, where the
duplicates the rewrite would collapse are the answer. The other negatives: a predicate pair
that only *looks* complementary (both NULL on a null row, so they partition nothing), two
branches that read different relations, and a duplicate-*sensitive* aggregate above the
union. Result-correctness vs DuckDB lives in `tests/differential/test_diff_setops_extra.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.setops_extra as se  # importing registers the rules
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.logical import Distinct, Filter, Union
from batcher.plan.visitor import walk

RULE_NAMES = {
    "absorb_subsumed_branch_in_distinct_union",
    "drop_union_dedup_on_semi_join_build",
    "drop_union_dedup_under_aggregate",
    "merge_distinct_union_of_complementary_filters",
    "merge_distinct_union_of_filters_on_same_input",
}


# --- fixtures ----------------------------------------------------------------


def _table() -> pa.Table:
    return pa.table({"a": [1, 2, 3, None], "b": [10, 20, 30, 40]})


def _not_null_table() -> pa.Table:
    return pa.table(
        {"x": pa.array([1, 2, 3], type=pa.int64())},
        schema=pa.schema([pa.field("x", pa.int64(), nullable=False)]),
    )


@pytest.fixture
def ds():
    return bt.from_arrow(_table())


@pytest.fixture
def nn():
    return bt.from_arrow(_not_null_table())


def _ctx(dataset):
    return Optimizer(sources=dataset._sources)._context()


def _rewrite(dataset):
    """The plan after the REAL optimizer (every phase, every registered rule)."""
    return Optimizer(sources=dataset._sources).logical_rewrite(dataset._plan)


def _unions(plan) -> list[Union]:
    return [n for n in walk(plan) if isinstance(n, Union)]


def _first(plan, kind):
    return next(n for n in walk(plan) if isinstance(n, kind))


# --- registration -------------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert names >= RULE_NAMES


# --- complementary filters partition the relation ------------------------------


def test_null_partition_merges_back_to_the_relation(ds):
    u = ds.filter(col("a").is_null()).union(ds.filter(col("a").is_not_null()), distinct=True)
    out = se.merge_distinct_union_of_complementary_filters(u._plan, _ctx(u))
    assert isinstance(out, Distinct)
    assert not _unions(out) and not [n for n in walk(out) if isinstance(n, Filter)]


def test_complementary_merge_is_idempotent(ds):
    # The rewrite consumes the `Union` it matched on, so the rule cannot fire on its own
    # output — the driver only ever offers it a `Union`.
    u = ds.filter(col("a").is_null()).union(ds.filter(col("a").is_not_null()), distinct=True)
    once = se.merge_distinct_union_of_complementary_filters(u._plan, _ctx(u))
    assert not _unions(once)


def test_complementary_merge_refuses_union_all(ds):
    # Bag semantics: the concatenation keeps every duplicate row of `a` — twice over if the
    # branches were merged back into one relation... and it reorders the rows besides.
    u = ds.filter(col("a").is_null()).union(ds.filter(col("a").is_not_null()), distinct=False)
    assert se.merge_distinct_union_of_complementary_filters(u._plan, _ctx(u)) is None


def test_complementary_merge_refuses_a_nullable_comparison(ds):
    # `a > 2` and `a <= 2` are BOTH null on the null row: it lands in neither branch, so the
    # pair does not partition and the row would be resurrected by the rewrite.
    u = ds.filter(col("a") > 2).union(ds.filter(col("a") <= 2), distinct=True)
    assert se.merge_distinct_union_of_complementary_filters(u._plan, _ctx(u)) is None


def test_complementary_merge_accepts_a_non_nullable_comparison(nn):
    u = nn.filter(col("x") > 2).union(nn.filter(col("x") <= 2), distinct=True)
    out = se.merge_distinct_union_of_complementary_filters(u._plan, _ctx(u))
    assert isinstance(out, Distinct)


def test_complementary_merge_refuses_different_relations(ds):
    other = bt.from_arrow(pa.table({"a": [7], "b": [7]}))
    u = ds.filter(col("a").is_null()).union(other.filter(col("a").is_not_null()), distinct=True)
    assert se.merge_distinct_union_of_complementary_filters(u._plan, _ctx(u)) is None


# --- filters on the same relation merge into one OR ----------------------------


def test_filters_on_the_same_relation_merge(ds):
    u = ds.filter(col("a") == 1).union(ds.filter(col("b") == 30), distinct=True)
    out = se.merge_distinct_union_of_filters_on_same_input(u._plan, _ctx(u))
    assert isinstance(out, Distinct)
    assert not _unions(out)
    assert out.input.predicate.to_ir() == ((col("a") == 1) | (col("b") == 30)).to_ir()


def test_filter_merge_is_idempotent(ds):
    u = ds.filter(col("a") == 1).union(ds.filter(col("b") == 30), distinct=True)
    once = se.merge_distinct_union_of_filters_on_same_input(u._plan, _ctx(u))
    assert not _unions(once)  # the matched `Union` is gone; the rule cannot re-fire


def test_filter_merge_refuses_union_all(ds):
    # A row satisfying BOTH predicates appears twice in the UNION ALL and once in the merge.
    u = ds.filter(col("a") == 1).union(ds.filter(col("b") == 30), distinct=False)
    assert se.merge_distinct_union_of_filters_on_same_input(u._plan, _ctx(u)) is None


def test_filter_merge_refuses_different_relations(ds):
    other = bt.from_arrow(pa.table({"a": [7], "b": [7]}))
    u = ds.filter(col("a") == 1).union(other.filter(col("b") == 30), distinct=True)
    assert se.merge_distinct_union_of_filters_on_same_input(u._plan, _ctx(u)) is None


def test_filter_merge_refuses_an_unfiltered_branch(ds):
    u = ds.union(ds.filter(col("b") == 30), distinct=True)
    assert se.merge_distinct_union_of_filters_on_same_input(u._plan, _ctx(u)) is None


# --- absorption of a subsumed branch -------------------------------------------


def test_filtered_branch_is_absorbed(ds):
    u = ds.union(ds.filter(col("a") == 1), distinct=True)
    out = se.absorb_subsumed_branch_in_distinct_union(u._plan, _ctx(u))
    assert isinstance(out, Distinct)
    assert not _unions(out)


def test_sorted_and_limited_branch_is_absorbed(ds):
    u = ds.union(ds.sort("a").limit(2), distinct=True)
    out = se.absorb_subsumed_branch_in_distinct_union(u._plan, _ctx(u))
    assert isinstance(out, Distinct)


def test_absorption_is_idempotent(ds):
    # Three branches, one absorbed: the output is still a `Union`, so the rule is offered its
    # own output and must decline it.
    other = bt.from_arrow(pa.table({"a": [7], "b": [7]}))
    u = ds.union(ds.filter(col("a") == 1), other, distinct=True)
    once = se.absorb_subsumed_branch_in_distinct_union(u._plan, _ctx(u))
    assert isinstance(once, Union) and len(once.inputs) == 2
    assert se.absorb_subsumed_branch_in_distinct_union(once, _ctx(u)) is None


def test_absorption_refuses_union_all(ds):
    # The filtered branch's rows are *additional copies* under bag semantics.
    u = ds.union(ds.filter(col("a") == 1), distinct=False)
    assert se.absorb_subsumed_branch_in_distinct_union(u._plan, _ctx(u)) is None


def test_absorption_refuses_a_branch_of_another_relation(ds):
    other = bt.from_arrow(pa.table({"a": [7], "b": [7]}))
    u = ds.union(other.filter(col("a") == 1), distinct=True)
    assert se.absorb_subsumed_branch_in_distinct_union(u._plan, _ctx(u)) is None


def test_absorption_leaves_a_projected_branch_alone(ds):
    # A `Project` rewrites the columns — its rows are not its input's rows.
    u = ds.union(ds.select(a=col("a"), b=col("b") + 0), distinct=True)
    assert se.absorb_subsumed_branch_in_distinct_union(u._plan, _ctx(u)) is None


# --- the union's dedup is redundant under a dedup-insensitive consumer ----------


def test_union_dedup_dropped_under_a_min_max_aggregate(ds):
    plan = ds.union(ds, distinct=True).group_by("a").agg(m=col("b").max())._plan
    out = se.drop_union_dedup_under_aggregate(plan, None)
    assert out.input.distinct is False


def test_union_dedup_kept_under_a_sum_aggregate(ds):
    plan = ds.union(ds, distinct=True).group_by("a").agg(s=col("b").sum())._plan
    assert se.drop_union_dedup_under_aggregate(plan, None) is None


def test_union_dedup_kept_under_a_count_aggregate(ds):
    plan = ds.union(ds, distinct=True).group_by("a").agg(c=col("b").count())._plan
    assert se.drop_union_dedup_under_aggregate(plan, None) is None


def test_union_dedup_under_aggregate_is_idempotent(ds):
    plan = ds.union(ds, distinct=True).group_by("a").agg(m=col("b").max())._plan
    once = se.drop_union_dedup_under_aggregate(plan, None)
    assert se.drop_union_dedup_under_aggregate(once, None) is None


def test_union_all_under_an_aggregate_is_left_alone(ds):
    plan = ds.union(ds, distinct=False).group_by("a").agg(m=col("b").max())._plan
    assert se.drop_union_dedup_under_aggregate(plan, None) is None


def test_union_dedup_dropped_on_a_semi_join_build_side(ds):
    plan = ds.join(ds.union(ds, distinct=True), on="a", how="semi")._plan
    assert se.drop_union_dedup_on_semi_join_build(plan, None).right.distinct is False


def test_union_dedup_dropped_on_an_anti_join_build_side(ds):
    plan = ds.join(ds.union(ds, distinct=True), on="a", how="anti")._plan
    assert se.drop_union_dedup_on_semi_join_build(plan, None).right.distinct is False


def test_union_dedup_kept_on_an_inner_join(ds):
    # An inner join emits one output row per matching right row: right duplicates matter.
    plan = ds.join(ds.union(ds, distinct=True), on="a", how="inner")._plan
    assert se.drop_union_dedup_on_semi_join_build(plan, None) is None


def test_semi_join_build_rewrite_is_idempotent(ds):
    plan = ds.join(ds.union(ds, distinct=True), on="a", how="semi")._plan
    once = se.drop_union_dedup_on_semi_join_build(plan, None)
    assert se.drop_union_dedup_on_semi_join_build(once, None) is None


# --- end to end: the rules fire through the REAL optimizer ---------------------


def test_optimizer_merges_the_null_partition(ds):
    plan = _rewrite(
        ds.filter(col("a").is_null()).union(ds.filter(col("a").is_not_null()), distinct=True)
    )
    assert not _unions(plan)
    assert isinstance(_first(plan, Distinct), Distinct)


def test_optimizer_merges_filters_on_one_relation(ds):
    plan = _rewrite(ds.filter(col("a") == 1).union(ds.filter(col("b") == 30), distinct=True))
    assert not _unions(plan)
    assert len([n for n in walk(plan) if isinstance(n, Filter)]) == 1


def test_optimizer_absorbs_a_subsumed_branch(ds):
    plan = _rewrite(ds.union(ds.filter(col("a") == 1), distinct=True))
    assert not _unions(plan)


def test_optimizer_relaxes_the_union_dedup_under_a_max(ds):
    plan = _rewrite(ds.union(ds, distinct=True).group_by("a").agg(m=col("b").max()))
    assert [u.distinct for u in _unions(plan)] == [False]


def test_optimizer_keeps_the_union_dedup_under_a_sum(ds):
    plan = _rewrite(ds.union(ds, distinct=True).group_by("a").agg(s=col("b").sum()))
    assert [u.distinct for u in _unions(plan)] == [True]


def test_optimizer_leaves_union_all_alone(ds):
    plan = _rewrite(ds.union(ds.filter(col("a") == 1), distinct=False))
    assert [u.distinct for u in _unions(plan)] == [False]
    assert len(_unions(plan)[0].inputs) == 2
