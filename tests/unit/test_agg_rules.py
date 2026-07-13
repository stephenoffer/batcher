"""Plan-shape, idempotence and refusal tests for the `agg_rules` (proven-metadata) rules.

Each rule gets: a *fires* test (the rewrite yields the intended shape), an *end-to-end* test
(it really fires through the real `Optimizer`, not just when called directly), and *refusal*
tests for the shapes it must decline — an ndv that is estimated rather than proven, an empty
input whose global aggregate must still emit its NULL/0 row, a NULL-bearing column.

Result-correctness vs DuckDB lives in `tests/differential/test_diff_agg_rules.py`.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import agg_rules as m
from batcher.plan.expr_ir import AggExpr, Col, Lit
from batcher.plan.logical import Aggregate, AggregateSpec, Limit, Project, Projection
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk

_RULE_NAMES = {
    "count_distinct_of_unique_column",
    "count_of_non_null_column",
    "drop_aggregate_over_single_row_input",
    "drop_dead_aggregate_output",
    "drop_group_key_functionally_determined_by_another",
    "global_count_star_from_exact_cardinality",
    "global_min_max_from_exact_bounds",
    "mean_of_constant_column",
    "merge_adjacent_aggregates_when_second_is_over_group_keys",
    "min_max_of_constant_column",
    "sum_of_constant_column",
}


def _exact(**kw) -> ColumnStat:
    return ColumnStat(provenance=Provenance.EXACT, **kw)


def _sketch(**kw) -> ColumnStat:
    return ColumnStat(provenance=Provenance.SKETCH, **kw)


def _stats(rows: int, **columns) -> list[SourceStatistics]:
    return [SourceStatistics(row_count=rows, columns=dict(columns))]


def _ds(rows: int = 3):
    ids = list(range(1, rows + 1))
    return bt.from_pydict(
        {"id": ids, "g": [f"g{i}" for i in ids], "k": [7] * rows, "x": [10 * i for i in ids]}
    )


def _ctx(ds, stats):
    return Optimizer(sources=ds._sources, source_stats=stats)._context()


def _rewrite(ds, plan, stats):
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(plan)


def _lit_value(expr):
    """The python value of a folded literal (the IR wraps it by type)."""
    assert isinstance(expr, Lit)
    return expr.value


def _aggregates(plan):
    return [n for n in walk(plan) if isinstance(n, Aggregate)]


def test_all_rules_registered():
    assert {r.name for r in DEFAULT_REGISTRY.rules()} >= _RULE_NAMES
    assert set(m.__all__) == _RULE_NAMES


# --- drop_group_key_functionally_determined_by_another ------------------------


def _fd_plan(ds):
    return ds.group_by("id", "g").agg(s=col("x").sum())._plan


def test_fd_group_key_fires_on_proven_unique_key():
    ds = _ds()
    plan = _fd_plan(ds)
    stats = _stats(3, id=_exact(ndv=3, null_count=0), g=_exact(ndv=3, null_count=0))
    out = m.drop_group_key_functionally_determined_by_another(plan, _ctx(ds, stats))
    assert isinstance(out, Project)
    assert [i.alias for i in out.items] == ["id", "g", "s"]  # schema preserved, in order
    inner = out.input
    assert isinstance(inner, Aggregate)
    assert [k.alias for k in inner.group_keys] == ["id"]  # only the unique key groups
    assert any(s.agg.func == "min" for s in inner.aggregates)  # `g` is carried as MIN(g)


def test_fd_group_key_idempotent():
    ds = _ds()
    stats = _stats(3, id=_exact(ndv=3, null_count=0))
    out = m.drop_group_key_functionally_determined_by_another(_fd_plan(ds), _ctx(ds, stats))
    assert m.drop_group_key_functionally_determined_by_another(out.input, _ctx(ds, stats)) is None


def test_fd_group_key_refuses_estimated_ndv():
    ds = _ds()
    stats = _stats(3, id=_sketch(ndv=3, null_count=0))  # measured, not proven
    assert (
        m.drop_group_key_functionally_determined_by_another(_fd_plan(ds), _ctx(ds, stats)) is None
    )


def test_fd_group_key_refuses_non_unique_ndv():
    ds = _ds()
    stats = _stats(3, id=_exact(ndv=2, null_count=0))  # a duplicate exists → groups merge
    assert (
        m.drop_group_key_functionally_determined_by_another(_fd_plan(ds), _ctx(ds, stats)) is None
    )


def test_fd_group_key_fires_end_to_end():
    ds = _ds()
    stats = _stats(3, id=_exact(ndv=3, null_count=0))
    out = _rewrite(ds, _fd_plan(ds), stats)
    assert any(len(a.group_keys) == 1 for a in _aggregates(out))


# --- count_distinct_of_unique_column ------------------------------------------


def _cd_plan(ds):
    return ds.group_by("g").agg(n=col("id").n_unique())._plan


def test_count_distinct_of_unique_column_fires():
    ds = _ds()
    stats = _stats(3, id=_exact(ndv=3, null_count=0))
    out = m.count_distinct_of_unique_column(_cd_plan(ds), _ctx(ds, stats))
    assert isinstance(out, Aggregate)
    assert [s.agg.func for s in out.aggregates] == ["count"]
    assert m.count_distinct_of_unique_column(out, _ctx(ds, stats)) is None  # idempotent


def test_count_distinct_of_unique_column_refuses_sketch_ndv():
    ds = _ds()
    stats = _stats(3, id=_sketch(ndv=3))
    assert m.count_distinct_of_unique_column(_cd_plan(ds), _ctx(ds, stats)) is None


def test_count_distinct_of_unique_column_fires_end_to_end():
    ds = _ds()
    stats = _stats(3, id=_exact(ndv=3, null_count=0))
    out = _rewrite(ds, _cd_plan(ds), stats)
    funcs = {s.agg.func for a in _aggregates(out) for s in a.aggregates}
    assert "count_distinct" not in funcs


# --- count_of_non_null_column -------------------------------------------------


def _count_plan(ds):
    return ds.group_by("g").agg(n=col("x").count())._plan


def test_count_of_non_null_column_fires():
    ds = _ds()
    stats = _stats(3, x=_exact(null_count=0))
    out = m.count_of_non_null_column(_count_plan(ds), _ctx(ds, stats))
    assert isinstance(out, Aggregate)
    assert out.aggregates[0].agg.func == "count_star"
    assert out.aggregates[0].agg.input is None


def test_count_of_non_null_column_refuses_when_nulls_present():
    ds = _ds()
    stats = _stats(3, x=_exact(null_count=1))  # COUNT(x) != COUNT(*) here
    assert m.count_of_non_null_column(_count_plan(ds), _ctx(ds, stats)) is None


def test_count_of_non_null_column_refuses_unproven_null_count():
    ds = _ds()
    stats = _stats(3, x=_sketch(null_count=0))
    assert m.count_of_non_null_column(_count_plan(ds), _ctx(ds, stats)) is None


def test_count_of_non_null_column_fires_end_to_end():
    ds = _ds()
    stats = _stats(3, x=_exact(null_count=0))
    out = _rewrite(ds, _count_plan(ds), stats)
    assert {s.agg.func for a in _aggregates(out) for s in a.aggregates} == {"count_star"}


# --- min_max_of_constant_column ------------------------------------------------


def _const_stats(rows: int = 3):
    return _stats(rows, k=_exact(min=7, max=7, null_count=0))


def test_min_max_of_constant_column_fires_grouped():
    ds = _ds()
    plan = ds.group_by("g").agg(lo=col("k").min(), hi=col("k").max())._plan
    out = m.min_max_of_constant_column(plan, _ctx(ds, _const_stats()))
    assert isinstance(out, Project)
    assert [i.alias for i in out.items] == ["g", "lo", "hi"]
    assert out.input.aggregates == ()  # both folded; the grouping key still forms the groups


def test_min_max_of_constant_column_refuses_null_bearing_column():
    ds = _ds()
    plan = ds.group_by("g").agg(lo=col("k").min())._plan
    stats = _stats(3, k=_exact(min=7, max=7, null_count=1))  # one value *plus* NULLs
    assert m.min_max_of_constant_column(plan, _ctx(ds, stats)) is None


def test_min_max_of_constant_column_refuses_empty_global_input():
    ds = _ds()
    plan = ds.group_by().agg(lo=col("k").min())._plan
    stats = _stats(0, k=_exact(min=7, max=7, null_count=0))  # empty: MIN must be NULL, not 7
    assert m.min_max_of_constant_column(plan, _ctx(ds, stats)) is None


def test_min_max_of_constant_column_fires_end_to_end():
    ds = _ds()
    plan = ds.group_by("g").agg(lo=col("k").min())._plan
    out = _rewrite(ds, plan, _const_stats())
    assert not any(s.agg.func == "min" for a in _aggregates(out) for s in a.aggregates)


# --- mean_of_constant_column ---------------------------------------------------


def test_mean_of_constant_column_fires():
    ds = _ds()
    plan = ds.group_by("g").agg(avg=col("k").mean())._plan
    out = m.mean_of_constant_column(plan, _ctx(ds, _const_stats()))
    assert isinstance(out, Project)
    assert _lit_value(out.items[-1].expr) == 7.0  # Float64, matching AVG's output type


def test_mean_of_constant_column_refuses_float_constant():
    ds = bt.from_pydict({"g": ["a"], "f": [0.1]})
    plan = ds.group_by("g").agg(avg=col("f").mean())._plan
    stats = _stats(3, f=_exact(min=0.1, max=0.1, null_count=0))
    assert m.mean_of_constant_column(plan, _ctx(ds, stats)) is None


def test_mean_of_constant_column_fires_end_to_end():
    ds = _ds()
    plan = ds.group_by("g").agg(avg=col("k").mean())._plan
    out = _rewrite(ds, plan, _const_stats())
    assert not any(s.agg.func == "mean" for a in _aggregates(out) for s in a.aggregates)


# --- sum_of_constant_column ----------------------------------------------------


def test_sum_of_constant_column_fires_grouped():
    ds = _ds()
    plan = ds.group_by("g").agg(t=col("k").sum())._plan
    out = m.sum_of_constant_column(plan, _ctx(ds, _const_stats()))
    assert isinstance(out, Project)
    assert [s.agg.func for s in out.input.aggregates] == ["count_star"]


def test_sum_of_constant_column_refuses_global():
    ds = _ds()
    plan = ds.group_by().agg(t=col("k").sum())._plan  # SUM over empty is NULL, 7*0 is 0
    assert m.sum_of_constant_column(plan, _ctx(ds, _const_stats())) is None


def test_sum_of_constant_column_fires_end_to_end():
    ds = _ds()
    plan = ds.group_by("g").agg(t=col("k").sum())._plan
    out = _rewrite(ds, plan, _const_stats())
    assert not any(s.agg.func == "sum" for a in _aggregates(out) for s in a.aggregates)


# --- global_min_max_from_exact_bounds ------------------------------------------


def test_global_min_max_from_exact_bounds_fires():
    ds = _ds()
    plan = ds.group_by().agg(lo=col("x").min(), t=col("x").sum())._plan
    stats = _stats(3, x=_exact(min=10, max=30, null_count=0))
    out = m.global_min_max_from_exact_bounds(plan, _ctx(ds, stats))
    assert isinstance(out, Project)
    assert _lit_value(out.items[0].expr) == 10
    assert [s.alias for s in out.input.aggregates] == ["t"]  # the SUM still executes


def test_global_min_max_from_exact_bounds_folds_the_whole_aggregate():
    ds = _ds()
    plan = ds.group_by().agg(hi=col("x").max())._plan
    stats = _stats(3, x=_exact(min=10, max=30, null_count=0))
    out = m.global_min_max_from_exact_bounds(plan, _ctx(ds, stats))
    assert isinstance(out, Project) and isinstance(out.input, Limit) and out.input.n == 1


def test_global_min_max_refuses_empty_input():
    ds = _ds()
    plan = ds.group_by().agg(hi=col("x").max())._plan
    stats = _stats(0, x=_exact(min=10, max=30, null_count=0))  # MAX over empty is NULL
    assert m.global_min_max_from_exact_bounds(plan, _ctx(ds, stats)) is None


def test_global_min_max_refuses_grouped():
    ds = _ds()
    plan = ds.group_by("g").agg(hi=col("x").max())._plan  # a per-group max is not the bound
    stats = _stats(3, x=_exact(min=10, max=30, null_count=0))
    assert m.global_min_max_from_exact_bounds(plan, _ctx(ds, stats)) is None


def test_global_min_max_fires_end_to_end():
    ds = _ds()
    plan = ds.group_by().agg(hi=col("x").max())._plan
    stats = _stats(3, x=_exact(min=10, max=30, null_count=0))
    assert not _aggregates(_rewrite(ds, plan, stats))


# --- global_count_star_from_exact_cardinality ----------------------------------


def test_global_count_star_from_exact_cardinality_fires():
    ds = _ds()
    plan = ds.group_by().agg(n=bt.count())._plan
    out = m.global_count_star_from_exact_cardinality(plan, _ctx(ds, _stats(3)))
    assert isinstance(out, Project) and isinstance(out.input, Limit)
    assert _lit_value(out.items[0].expr) == 3


def test_global_count_star_over_empty_keeps_the_row():
    ds = _ds()
    plan = ds.group_by().agg(n=bt.count(), t=col("x").sum())._plan
    out = m.global_count_star_from_exact_cardinality(plan, _ctx(ds, _stats(0)))
    # COUNT(*) over an empty relation is 0 — folded — but the aggregate stays for `t`,
    # so the global aggregate still emits its one (0, NULL) row.
    assert isinstance(out, Project) and isinstance(out.input, Aggregate)
    assert _lit_value(out.items[0].expr) == 0


def test_global_count_star_refuses_grouped():
    ds = _ds()
    plan = ds.group_by("g").agg(n=bt.count())._plan
    assert m.global_count_star_from_exact_cardinality(plan, _ctx(ds, _stats(3))) is None


def test_global_count_star_fires_end_to_end():
    ds = _ds()
    plan = ds.group_by().agg(n=bt.count())._plan
    assert not _aggregates(_rewrite(ds, plan, _stats(3)))


# --- drop_aggregate_over_single_row_input --------------------------------------


def test_drop_aggregate_over_single_row_input_fires():
    ds = _ds(1)
    plan = ds.group_by("g").agg(lo=col("x").min(), n=bt.count())._plan
    out = m.drop_aggregate_over_single_row_input(plan, _ctx(ds, _stats(1)))
    assert isinstance(out, Project)
    assert [i.alias for i in out.items] == ["g", "lo", "n"]
    assert not _aggregates(out)


def test_drop_aggregate_over_single_row_refuses_unfoldable_aggregate():
    ds = _ds(1)
    plan = ds.group_by("g").agg(t=col("x").sum())._plan  # SUM widens its type — not folded
    assert m.drop_aggregate_over_single_row_input(plan, _ctx(ds, _stats(1))) is None


def test_drop_aggregate_over_single_row_refuses_estimated_row_count():
    ds = _ds(4)
    # A filter's output size is *estimated* (selectivity), never proven — even if it in fact
    # leaves one row. Folding on an estimate would collapse a real aggregation.
    plan = ds.filter(col("x") > 35).group_by("g").agg(lo=col("x").min())._plan
    assert m.drop_aggregate_over_single_row_input(plan, _ctx(ds, _stats(4))) is None


def test_drop_aggregate_over_single_row_refuses_empty_global():
    ds = _ds(1)
    plan = ds.group_by().agg(lo=col("x").min())._plan  # 0 rows → still one NULL row out
    assert m.drop_aggregate_over_single_row_input(plan, _ctx(ds, _stats(0))) is None


def test_drop_aggregate_over_single_row_allows_empty_grouped():
    ds = _ds(1)
    plan = ds.group_by("g").agg(lo=col("x").min())._plan  # 0 rows → 0 rows either way
    assert isinstance(m.drop_aggregate_over_single_row_input(plan, _ctx(ds, _stats(0))), Project)


def test_drop_aggregate_over_single_row_fires_end_to_end():
    ds = _ds(1)
    plan = ds.group_by("g").agg(lo=col("x").min())._plan
    assert not _aggregates(_rewrite(ds, plan, _stats(1)))


# --- merge_adjacent_aggregates_when_second_is_over_group_keys ------------------


def _nested(ds, outer_keys=("g",)):
    inner = Aggregate(
        ds._plan,
        (Projection("g", Col("g")),),
        (AggregateSpec("t", AggExpr("sum", Col("x"))),),
    )
    return Aggregate(
        inner,
        tuple(Projection(k, Col(k)) for k in outer_keys),
        (AggregateSpec("hi", AggExpr("max", Col("t"))),),
    )


def test_merge_adjacent_aggregates_fires():
    ds = _ds()
    out = m.merge_adjacent_aggregates_when_second_is_over_group_keys(_nested(ds), None)
    assert isinstance(out, Project)
    assert [i.alias for i in out.items] == ["g", "hi"]
    assert len(_aggregates(out)) == 1  # only the inner aggregate survives


def test_merge_adjacent_aggregates_refuses_unfoldable_outer_aggregate():
    ds = _ds()
    inner = Aggregate(
        ds._plan, (Projection("g", Col("g")),), (AggregateSpec("t", AggExpr("sum", Col("x"))),)
    )
    outer = Aggregate(
        inner, (Projection("g", Col("g")),), (AggregateSpec("s", AggExpr("sum", Col("t"))),)
    )
    assert m.merge_adjacent_aggregates_when_second_is_over_group_keys(outer, None) is None


def test_merge_adjacent_aggregates_refuses_partial_key_cover():
    ds = _ds()
    inner = Aggregate(
        ds._plan,
        (Projection("g", Col("g")), Projection("id", Col("id"))),
        (AggregateSpec("t", AggExpr("sum", Col("x"))),),
    )
    outer = Aggregate(  # groups by a *subset* — that genuinely merges groups
        inner, (Projection("g", Col("g")),), (AggregateSpec("hi", AggExpr("max", Col("t"))),)
    )
    assert m.merge_adjacent_aggregates_when_second_is_over_group_keys(outer, None) is None


def test_merge_adjacent_aggregates_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _nested(ds), _stats(3))
    assert len(_aggregates(out)) == 1


# --- drop_dead_aggregate_output ------------------------------------------------


def _dead_plan(ds):
    agg = Aggregate(
        ds._plan,
        (Projection("g", Col("g")),),
        (
            AggregateSpec("t", AggExpr("sum", Col("x"))),
            AggregateSpec("dead", AggExpr("count_distinct", Col("x"))),
        ),
    )
    return Project(agg, (Projection("g", Col("g")), Projection("t", Col("t"))))


def test_drop_dead_aggregate_output_fires():
    ds = _ds()
    out = m.drop_dead_aggregate_output(_dead_plan(ds), None)
    assert isinstance(out, Project)
    assert [s.alias for s in out.input.aggregates] == ["t"]
    assert m.drop_dead_aggregate_output(out, None) is None  # idempotent


def test_drop_dead_aggregate_output_keeps_read_outputs():
    ds = _ds()
    plan = ds.group_by("g").agg(t=col("x").sum())._plan
    assert m.drop_dead_aggregate_output(Project(plan, (Projection("t", Col("t")),)), None) is None


def test_drop_dead_aggregate_output_refuses_emptying_a_keyless_aggregate():
    ds = _ds()
    agg = Aggregate(ds._plan, (), (AggregateSpec("t", AggExpr("sum", Col("x"))),))
    # Nothing above reads `t`, but an aggregate with neither a key nor a function has no
    # meaning — the rule leaves it alone.
    dead = Project(agg, (Projection("one", Lit(1)),))
    assert m.drop_dead_aggregate_output(dead, None) is None


def test_drop_dead_aggregate_output_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _dead_plan(ds), _stats(3))
    assert not any(s.agg.func == "count_distinct" for a in _aggregates(out) for s in a.aggregates)
