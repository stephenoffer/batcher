"""Plan-shape unit tests for `count_distinct_to_distinct_count`.

The rewrite turns a lone ``COUNT(DISTINCT x) GROUP BY g`` into ``COUNT(x)`` over a
``DISTINCT(g, x)`` — reusing the radix-parallel distinct + count kernels. These tests
prove the rewrite changes the *plan* (distinct + plain count, no `count_distinct`) while
preserving the *result*; the cross-engine correctness is covered by the count-distinct
differential suite vs DuckDB.
"""

from __future__ import annotations

import batcher as bt
from batcher import col, count
from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.agg_pushdown import count_distinct_to_distinct_count
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Aggregate, Distinct, Project


def _ds():
    return bt.from_pydict(
        {"g": ["a", "a", "b", "b", "b"], "v": [1, 1, 2, 3, 3]}
    )


def _ctx(ds):
    est = StatsEstimator(ds._sources, learned={})
    return OptimizerContext(
        config=active_config(), sources=ds._sources, hub=None, estimator=est
    )


def test_rule_registered():
    assert "count_distinct_to_distinct_count" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_rewrites_lone_count_distinct():
    ds = _ds().group_by("g").agg(nd=col("v").n_unique())
    out = count_distinct_to_distinct_count(ds._plan, _ctx(ds))
    assert isinstance(out, Aggregate)
    # The distinct aggregate is gone — it is now a plain non-null COUNT…
    assert [s.agg.func for s in out.aggregates] == ["count"]
    # …over a Distinct(Project(group keys + value)).
    assert isinstance(out.input, Distinct)
    assert isinstance(out.input.input, Project)
    proj_cols = out.input.input.available_columns()
    assert "g" in proj_cols and "__count_distinct_value" in proj_cols


def test_no_fire_with_other_aggregates():
    # COUNT(DISTINCT v) alongside a row-level aggregate can't share one distinct.
    ds = _ds().group_by("g").agg(nd=col("v").n_unique(), n=count())
    assert count_distinct_to_distinct_count(ds._plan, _ctx(ds)) is None


def test_no_fire_for_approx_count_distinct():
    # approx_count_distinct is the bounded-memory HLL path — must not be rewritten.
    ds = _ds().group_by("g").agg(nd=col("v").approx_n_unique())
    assert count_distinct_to_distinct_count(ds._plan, _ctx(ds)) is None


def test_result_preserved_end_to_end():
    # The optimized query returns the same per-group distinct counts.
    got = _ds().group_by("g").agg(nd=col("v").n_unique()).collect().to_pydict()
    pairs = dict(zip(got["g"], got["nd"], strict=True))
    assert pairs == {"a": 1, "b": 2}
