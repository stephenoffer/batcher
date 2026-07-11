"""`requires_staging` — the routing fact that a join's operand spans two sources.

The one-shot distributed dispatcher co-partitions exactly two sources per join, so a join
over a join has no single-shot path and must be executed stage by stage. This predicate is
what turns that shape from a `PlanError` into the staged path, so it must be exact: too
narrow and 3+-table joins fail; too wide and ordinary queries pay a needless materialization.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col

pytest.importorskip("ray", reason="ray not installed")

from batcher.dist import requires_staging


def _d(name):
    return bt.from_pydict({name: [1], "k": [1], "d1": [1], "d2": [1], "v": [1]})


@pytest.fixture
def tables():
    return _d("a"), _d("b"), _d("c"), _d("e")


def test_two_table_join_needs_no_staging(tables):
    a, b, _c, _e = tables
    assert requires_staging(a.join(b, on="k")._plan) is False


def test_three_table_join_requires_staging(tables):
    a, b, c, _e = tables
    assert requires_staging(a.join(b, on="k").join(c, on="d1")._plan) is True


def test_four_table_join_requires_staging(tables):
    a, b, c, e = tables
    plan = a.join(b, on="k").join(c, on="d1").join(e, on="d2")._plan
    assert requires_staging(plan) is True


def test_join_nested_under_an_operator_is_still_found(tables):
    a, b, c, _e = tables
    deep = a.join(b, on="k").join(c, on="d1").group_by("k").agg(s=col("v").sum()).sort("s")
    assert requires_staging(deep._plan) is True


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b, c: a.group_by("k").agg(s=col("v").sum()),
        lambda a, b, c: a.sort("v"),
        lambda a, b, c: a.filter(col("v") > 0).limit(3),
        lambda a, b, c: a.join(b, on="k").sort("v"),  # a single join, wrapped
        lambda a, b, c: a.join(b, on="k").group_by("k").agg(s=col("v").sum()),
    ],
)
def test_one_shot_shapes_never_ask_for_staging(tables, build):
    """A needless `True` here costs every ordinary query an extra materialization."""
    a, b, c, _e = tables
    assert requires_staging(build(a, b, c)._plan) is False


def test_breaker_beneath_an_aggregate_requires_staging(tables):
    """`limit(100).group_by(k).agg(...)` — the aggregate executor would run the `Limit` once
    per partition and keep 100 rows on EACH worker."""
    a, _b, _c, _e = tables
    assert requires_staging(a.limit(100).group_by("k").agg(s=col("v").sum())._plan) is True


def test_nested_aggregate_requires_staging(tables):
    """`agg(agg(x))` — the outer aggregate would receive per-partition partial groups."""
    a, _b, _c, _e = tables
    nested = a.group_by("k").agg(s=col("v").sum()).group_by().agg(m=col("s").max())
    assert requires_staging(nested._plan) is True


def test_breaker_beneath_a_join_requires_staging(tables):
    """`limit(5).join(dim)` — each worker would keep its own 5 rows."""
    a, b, _c, _e = tables
    assert requires_staging(a.limit(5).join(b, on="k")._plan) is True


def test_breaker_beneath_a_window_requires_staging(tables):
    a, _b, _c, _e = tables
    assert requires_staging(a.limit(100).with_columns(t=col("v").sum().over("k"))._plan) is True


@pytest.mark.parametrize(
    "build",
    [
        # A breaker ABOVE another breaker is carried by `_split_at` and replayed on the
        # driver, never run per-partition — so these must NOT pay for staging.
        lambda a, b, c: a.group_by("k").agg(s=col("v").sum()).sort("s"),
        lambda a, b, c: a.group_by("k").agg(s=col("v").sum()).limit(5),
        lambda a, b, c: a.filter(col("v") > 0).group_by("k").agg(s=col("v").sum()),
        lambda a, b, c: a.sort("v").limit(5),
        # An aggregate over a JOIN and over a clean DISTINCT have real fused paths.
        lambda a, b, c: a.join(b, on="k").group_by("k").agg(s=col("v").sum()),
        lambda a, b, c: a.select("k").distinct().group_by().agg(n=col("k").count()),
    ],
)
def test_breaker_above_a_breaker_needs_no_staging(tables, build):
    """A false positive here costs every ordinary query an extra materialization."""
    a, b, c, _e = tables
    assert requires_staging(build(a, b, c)._plan) is False
