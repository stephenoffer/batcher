"""Plan-shape, idempotence and negative-case tests for the `agg_extra` rules."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import agg_extra as m
from batcher.plan.expr_ir import AggExpr, Col, Lit
from batcher.plan.logical import Aggregate, AggregateSpec, Distinct, Project, Projection

_RULE_NAMES = {
    "aggregate_without_aggs_to_distinct",
    "count_constant_to_count_star",
    "count_distinct_of_group_key",
    "count_of_group_key",
    "dedupe_group_keys",
    "deduplicate_aggregate_exprs",
    "drop_constant_group_key",
    "drop_distinct_before_agg",
    "fold_constant_grouped_aggregate",
    "redundant_aggregate_of_group_key",
    "sum_constant_to_count",
}


def _base(cols=None):
    return bt.from_pydict(cols or {"g": [1, 2, 2], "x": [10, 20, 30]})._plan


def test_all_rules_registered():
    registered = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert registered >= _RULE_NAMES


# --- dedupe_group_keys -------------------------------------------------------


def test_dedupe_group_keys_fires():
    agg = Aggregate(
        _base(),
        (Projection("a", Col("g")), Projection("b", Col("g"))),
        (AggregateSpec("s", AggExpr("sum", Col("x"))),),
    )
    out = m.dedupe_group_keys(agg, None)
    assert isinstance(out, Project)
    assert len(out.input.group_keys) == 1  # one physical key
    # a and b both re-derive from the surviving key.
    derived = {i.alias: i.expr.to_ir() for i in out.items}
    assert derived["a"] == Col("a").to_ir()
    assert derived["b"] == Col("a").to_ir()


def test_dedupe_group_keys_idempotent():
    agg = Aggregate(
        _base(),
        (Projection("a", Col("g")), Projection("b", Col("g"))),
        (AggregateSpec("s", AggExpr("sum", Col("x"))),),
    )
    out = m.dedupe_group_keys(agg, None)
    assert m.dedupe_group_keys(out.input, None) is None


def test_dedupe_group_keys_negative_distinct_keys():
    agg = Aggregate(
        _base(),
        (Projection("a", Col("g")), Projection("b", Col("x"))),
        (),
    )
    assert m.dedupe_group_keys(agg, None) is None


# --- drop_constant_group_key -------------------------------------------------


def test_drop_constant_group_key_fires():
    agg = Aggregate(
        _base(),
        (Projection("g", Col("g")), Projection("c", Lit(5))),
        (AggregateSpec("s", AggExpr("sum", Col("x"))),),
    )
    out = m.drop_constant_group_key(agg, None)
    assert isinstance(out, Project)
    assert len(out.input.group_keys) == 1
    assert {i.alias for i in out.items} == {"g", "c", "s"}


def test_drop_constant_group_key_idempotent():
    agg = Aggregate(
        _base(),
        (Projection("g", Col("g")), Projection("c", Lit(5))),
        (AggregateSpec("s", AggExpr("sum", Col("x"))),),
    )
    out = m.drop_constant_group_key(agg, None)
    assert m.drop_constant_group_key(out.input, None) is None


def test_drop_constant_group_key_negative_sole_key():
    # Sole constant key: dropping it would change empty-input semantics -> no fire.
    agg = Aggregate(
        _base(), (Projection("c", Lit(5)),), (AggregateSpec("s", AggExpr("sum", Col("x"))),)
    )
    assert m.drop_constant_group_key(agg, None) is None


# --- redundant_aggregate_of_group_key ---------------------------------------


def test_min_of_group_key_fires():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .group_by("g")
        .agg(mn=col("g").min(), s=col("x").sum())
        ._plan
    )
    out = m.redundant_aggregate_of_group_key(agg, None)
    assert isinstance(out, Project)
    assert [s.alias for s in out.input.aggregates] == ["s"]  # only the real agg remains
    folded = {i.alias: i.expr.to_ir() for i in out.items}
    assert folded["mn"] == Col("g").to_ir()


def test_min_of_group_key_idempotent():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(mn=col("g").min())._plan
    out = m.redundant_aggregate_of_group_key(agg, None)
    assert m.redundant_aggregate_of_group_key(out.input, None) is None


def test_min_of_non_key_negative():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(mn=col("x").min())._plan
    assert m.redundant_aggregate_of_group_key(agg, None) is None


# --- count_distinct_of_group_key --------------------------------------------


def test_count_distinct_of_group_key_fires():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(n=col("g").n_unique())._plan
    out = m.count_distinct_of_group_key(agg, None)
    assert isinstance(out, Project)
    assert out.input.aggregates == ()  # the distinct count is gone
    assert out.items[-1].expr.to_ir()["e"] == "case"


def test_count_distinct_of_group_key_idempotent():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .group_by("g")
        .agg(n=col("g").n_unique(), s=col("x").sum())
        ._plan
    )
    out = m.count_distinct_of_group_key(agg, None)
    assert m.count_distinct_of_group_key(out.input, None) is None


def test_count_distinct_of_non_key_negative():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(n=col("x").n_unique())._plan
    assert m.count_distinct_of_group_key(agg, None) is None


# --- count_of_group_key ------------------------------------------------------


def test_count_of_group_key_fires_and_reuses_count_star():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .group_by("g")
        .agg(c=col("g").count(), n=bt.count())
        ._plan
    )
    out = m.count_of_group_key(agg, None)
    assert isinstance(out, Project)
    # count_star (n) reused, not duplicated: still exactly one count_star inside.
    stars = [s for s in out.input.aggregates if s.agg.func == "count_star"]
    assert len(stars) == 1


def test_count_of_group_key_idempotent():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(c=col("g").count())._plan
    out = m.count_of_group_key(agg, None)
    assert m.count_of_group_key(out.input, None) is None


def test_count_of_non_key_negative():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(c=col("x").count())._plan
    assert m.count_of_group_key(agg, None) is None


# --- count_constant_to_count_star -------------------------------------------


def test_count_constant_fires():
    agg = bt.from_pydict({"g": [1, 2]}).group_by("g").agg(c=bt.lit(1).count())._plan
    out = m.count_constant_to_count_star(agg, None)
    assert isinstance(out, Aggregate)
    assert out.aggregates[0].agg.func == "count_star"


def test_count_constant_idempotent():
    agg = bt.from_pydict({"g": [1, 2]}).group_by("g").agg(c=bt.lit(1).count())._plan
    out = m.count_constant_to_count_star(agg, None)
    assert m.count_constant_to_count_star(out, None) is None


def test_count_of_real_column_negative():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(c=col("x").count())._plan
    assert m.count_constant_to_count_star(agg, None) is None


# --- sum_constant_to_count ---------------------------------------------------


def test_sum_constant_fires():
    agg = bt.from_pydict({"g": [1, 2]}).group_by("g").agg(s=bt.lit(2).sum())._plan
    out = m.sum_constant_to_count(agg, None)
    assert isinstance(out, Project)
    assert out.items[-1].expr.to_ir()["op"] == "mul"


def test_sum_constant_idempotent():
    agg = bt.from_pydict({"g": [1, 2]}).group_by("g").agg(s=bt.lit(2).sum())._plan
    out = m.sum_constant_to_count(agg, None)
    assert m.sum_constant_to_count(out.input, None) is None


def test_sum_float_constant_negative():
    # Float would not be bit-identical (rounding) -> must not fire.
    agg = bt.from_pydict({"g": [1, 2]}).group_by("g").agg(s=bt.lit(2.5).sum())._plan
    assert m.sum_constant_to_count(agg, None) is None


def test_sum_constant_global_negative():
    # Global sum of a constant over empty input is NULL, not 0 -> must not fire.
    agg = bt.from_pydict({"g": [1, 2]}).group_by().agg(s=bt.lit(2).sum())._plan
    assert m.sum_constant_to_count(agg, None) is None


# --- fold_constant_grouped_aggregate ----------------------------------------


def test_fold_constant_min_and_count_distinct_fire():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .group_by("g")
        .agg(mn=bt.lit(7).min(), n=bt.lit(3).n_unique(), s=col("x").sum())
        ._plan
    )
    out = m.fold_constant_grouped_aggregate(agg, None)
    assert isinstance(out, Project)
    assert [s.alias for s in out.input.aggregates] == ["s"]
    folded = {i.alias: i.expr.to_ir() for i in out.items}
    assert folded["mn"]["value"] == {"int": 7}
    assert folded["n"]["value"] == {"int": 1}


def test_fold_constant_idempotent():
    agg = bt.from_pydict({"g": [1, 2]}).group_by("g").agg(mn=bt.lit(7).min())._plan
    out = m.fold_constant_grouped_aggregate(agg, None)
    assert m.fold_constant_grouped_aggregate(out.input, None) is None


def test_fold_constant_global_negative():
    agg = bt.from_pydict({"g": [1, 2]}).group_by().agg(mn=bt.lit(7).min())._plan
    assert m.fold_constant_grouped_aggregate(agg, None) is None


# --- drop_distinct_before_agg ------------------------------------------------


def test_drop_distinct_before_agg_fires():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .distinct()
        .group_by("g")
        .agg(lo=col("x").min(), hi=col("x").max())
        ._plan
    )
    out = m.drop_distinct_before_agg(agg, None)
    assert isinstance(out, Aggregate)
    assert not isinstance(out.input, Distinct)


def test_drop_distinct_before_agg_idempotent():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .distinct()
        .group_by("g")
        .agg(lo=col("x").min())
        ._plan
    )
    out = m.drop_distinct_before_agg(agg, None)
    assert m.drop_distinct_before_agg(out, None) is None


def test_drop_distinct_before_agg_negative_sum():
    # SUM is duplicate-sensitive -> the Distinct must stay.
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .distinct()
        .group_by("g")
        .agg(s=col("x").sum())
        ._plan
    )
    assert m.drop_distinct_before_agg(agg, None) is None


# --- aggregate_without_aggs_to_distinct -------------------------------------


def test_group_only_to_distinct_fires():
    agg = Aggregate(_base(), (Projection("g", Col("g")), Projection("x", Col("x"))), ())
    out = m.aggregate_without_aggs_to_distinct(agg, None)
    assert isinstance(out, Distinct)
    assert isinstance(out.input, Project)


def test_group_only_to_distinct_idempotent():
    agg = Aggregate(_base(), (Projection("g", Col("g")),), ())
    out = m.aggregate_without_aggs_to_distinct(agg, None)
    # result is a Distinct, so the Aggregate rule cannot rematch it.
    assert not isinstance(out, Aggregate)


def test_group_only_negative_with_aggs():
    agg = bt.from_pydict({"g": [1, 2], "x": [1, 2]}).group_by("g").agg(s=col("x").sum())._plan
    assert m.aggregate_without_aggs_to_distinct(agg, None) is None


# --- deduplicate_aggregate_exprs --------------------------------------------


def test_deduplicate_aggregate_exprs_fires():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .group_by("g")
        .agg(a=col("x").sum(), b=col("x").sum())
        ._plan
    )
    out = m.deduplicate_aggregate_exprs(agg, None)
    assert isinstance(out, Project)
    assert len(out.input.aggregates) == 1  # computed once
    dup = next(i for i in out.items if i.alias == "b")
    assert dup.expr.to_ir() == Col("a").to_ir()


def test_deduplicate_aggregate_exprs_idempotent():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .group_by("g")
        .agg(a=col("x").sum(), b=col("x").sum())
        ._plan
    )
    out = m.deduplicate_aggregate_exprs(agg, None)
    assert m.deduplicate_aggregate_exprs(out.input, None) is None


def test_deduplicate_aggregate_exprs_negative_distinct():
    agg = (
        bt.from_pydict({"g": [1, 2], "x": [1, 2]})
        .group_by("g")
        .agg(a=col("x").sum(), b=col("x").min())
        ._plan
    )
    assert m.deduplicate_aggregate_exprs(agg, None) is None
