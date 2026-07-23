"""Plan-shape unit tests for the `limit_extra` rules.

The first half asserts each rule fires through the real `Optimizer` and yields the
intended shape. The second half pins the *non*-rewrites: a `LIMIT` is a positional
prefix, so a filter, a sample, a distinct, an unnest, or a `map_batches` must never
cross it, and a distinct `UNION` must never have its branches capped. Those are the
shapes where a plausible-looking rewrite returns the wrong rows.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.limit_extra import __all__ as _RULES
from batcher.plan.logical import Distinct, Filter, Limit, LogicalPlan, Project, Sample, Sort, Union
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk


def _ds() -> bt.Dataset:
    return bt.from_pydict({"x": [3, 1, 2, 5, 4], "v": [10, 20, 30, 40, 50]})


def _opt(ds: bt.Dataset, stats: list[SourceStatistics] | None = None) -> LogicalPlan:
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _kinds(plan: LogicalPlan) -> list[str]:
    return [type(node).__name__ for node in walk(plan)]


def _sorts(plan: LogicalPlan) -> list[Sort]:
    return [node for node in walk(plan) if isinstance(node, Sort)]


def test_eight_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert set(_RULES) <= names
    assert len(_RULES) == 8


# --- top-N rules -----------------------------------------------------------------


def test_topn_through_project_sinks_the_sort_below_the_projection():
    # The projection then evaluates `v * 2` for 2 rows instead of all 5.
    ds = _ds().select(col("x").alias("k"), (col("v") * 2).alias("w")).sort("k").limit(2)
    out = _opt(ds)
    assert isinstance(out, Project)
    assert isinstance(out.input, Sort)
    assert out.input.limit == 2
    assert [key.expr.name for key in out.input.keys] == ["x"]  # the alias was translated back


def test_topn_through_project_refuses_a_computed_sort_key():
    ds = _ds().select((col("x") * 2).alias("k")).sort("k").limit(2)
    out = _opt(ds)
    # The sort stays above the projection: its key is not a pass-through column.
    assert isinstance(out, Sort)
    assert isinstance(out.input, Project)


def test_push_topn_into_union_caps_every_branch():
    ds = _ds().union(_ds()).sort("x").limit(3)
    out = _opt(ds)
    assert isinstance(out, Sort) and out.limit == 3
    union = out.input
    assert isinstance(union, Union) and not union.distinct
    assert all(isinstance(b, Sort) and b.limit == 3 for b in union.inputs)


def test_push_topn_into_union_leaves_a_distinct_union_alone():
    # A distinct union dedups across branches: a row cut from one branch's top-k can
    # still change which rows survive the dedup.
    ds = _ds().union(_ds(), distinct=True).sort("x").limit(3)
    out = _opt(ds)
    unions = [n for n in walk(out) if isinstance(n, Union)]
    assert unions and all(not isinstance(b, Sort) for u in unions for b in u.inputs)


def test_collapse_topn_over_topn_keeps_the_tighter_cap():
    ds = _ds().sort("x").limit(10).sort("x").limit(3)
    out = _opt(ds)
    assert isinstance(out, Sort) and out.limit == 3
    assert not isinstance(out.input, Sort)


def test_collapse_topn_over_topn_refuses_different_keys():
    ds = _ds().sort("v").limit(10).sort("x").limit(3)
    out = _opt(ds)
    sorts = _sorts(out)
    assert len(sorts) == 2  # the inner ordering breaks the outer one's ties — it stays


def test_empty_topn_becomes_the_canonical_empty_marker():
    ds = _ds().sort("x").limit(0)
    out = _opt(ds)
    assert isinstance(out, Limit) and out.n == 0
    assert not _sorts(out)


# --- empty-marker + bounded-sample rules -----------------------------------------


def test_prune_input_of_empty_limit_deletes_the_dead_work():
    ds = _ds().filter(col("x") > 1).sort("x").distinct().limit(0)
    out = _opt(ds)
    assert isinstance(out, Limit) and out.n == 0
    assert _kinds(out) == ["Limit", "Scan"]


def test_prune_input_of_empty_limit_keeps_a_schema_changing_child():
    ds = _ds().select(col("x").alias("only")).limit(0)
    out = _opt(ds)
    assert out.available_columns() == ["only"]
    assert "Project" in _kinds(out)


def test_drop_limit_over_bounded_sample():
    ds = _ds().sample(n=2, seed=7).limit(5)
    out = _opt(ds)
    assert isinstance(out, Sample) and out.n == 2
    assert not [n for n in walk(out) if isinstance(n, Limit)]


def test_limit_below_the_sample_bound_is_kept():
    ds = _ds().sample(n=4, seed=7).limit(2)
    out = _opt(ds)
    assert isinstance(out, Limit) and out.n == 2
    assert isinstance(out.input, Sample)


def test_empty_limit_past_bounded_sample():
    ds = _ds().sample(n=2, seed=7).limit(5, offset=2)
    out = _opt(ds)
    assert isinstance(out, Limit) and out.n == 0


def test_offset_inside_the_sample_bound_is_kept():
    ds = _ds().sample(n=4, seed=7).limit(2, offset=1)
    out = _opt(ds)
    assert isinstance(out, Limit) and out.n == 2 and out.offset == 1


# --- sort-key pruning ------------------------------------------------------------


def _unique_v_stats() -> list[SourceStatistics]:
    return [
        SourceStatistics(
            row_count=5,
            columns={"v": ColumnStat(ndv=5, null_count=0, provenance=Provenance.EXACT)},
        )
    ]


def test_prune_sort_keys_after_unique_key():
    out = _opt(_ds().sort("v", "x"), _unique_v_stats())
    assert [key.expr.name for key in _sorts(out)[0].keys] == ["v"]


def test_sort_keys_before_the_unique_key_are_kept():
    out = _opt(_ds().sort("x", "v"), _unique_v_stats())
    assert [key.expr.name for key in _sorts(out)[0].keys] == ["x", "v"]


def test_sort_keys_are_not_pruned_without_exact_uniqueness():
    # No source statistics → the ndv is an estimate, and an estimate may never delete a
    # real tiebreak.
    out = _opt(_ds().sort("v", "x"))
    assert [key.expr.name for key in _sorts(out)[0].keys] == ["v", "x"]


def test_sort_keys_are_not_pruned_when_the_unique_column_is_nullable():
    stats = [
        SourceStatistics(
            row_count=5,
            columns={"v": ColumnStat(ndv=5, null_count=2, provenance=Provenance.EXACT)},
        )
    ]
    out = _opt(_ds().sort("v", "x"), stats)
    assert [key.expr.name for key in _sorts(out)[0].keys] == ["v", "x"]


# --- the shapes that must NOT be rewritten ---------------------------------------


def test_limit_never_sinks_below_a_filter():
    # `Limit(Filter(p, x), n)` keeps the first n *passing* rows; pushing the limit under
    # the filter would keep the passing rows of the first n — strictly fewer.
    ds = _ds().filter(col("x") > 3).limit(2)
    out = _opt(ds)
    assert isinstance(out, Limit)
    assert isinstance(out.input, Filter)


def test_limit_and_sample_are_never_reordered():
    below = _opt(_ds().sample(0.5, seed=1).limit(2))
    assert isinstance(below, Limit) and isinstance(below.input, Sample)

    above = _opt(_ds().limit(2).sample(0.5, seed=1))
    assert isinstance(above, Sample) and isinstance(above.input, Limit)


def test_limit_never_sinks_below_a_distinct():
    ds = _ds().distinct().limit(2)
    out = _opt(ds)
    assert isinstance(out, Limit)
    assert isinstance(out.input, Distinct)


def test_limit_never_sinks_below_an_unnest():
    ds = bt.from_pydict({"xs": [[1, 2], [], [3]]}).explode("xs").limit(2)
    out = _opt(ds)
    assert isinstance(out, Limit)
    assert type(out.input).__name__ == "Unnest"


def test_zero_limit_over_a_bounded_sample_stays_empty():
    # The `n == 0` marker must survive both sample rules (it is the empty relation).
    out = _opt(_ds().sample(n=3, seed=1).limit(0))
    assert isinstance(out, Limit) and out.n == 0
