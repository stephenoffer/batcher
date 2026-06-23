"""Plan-shape unit tests for `eager_aggregation`."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.joins import eager_aggregation
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Aggregate, Join


def _emp():
    return bt.from_pydict({"dept_id": [1, 1, 2, 2, 3], "sal": [100, 200, 150, 300, 50]})


def _dept():
    return bt.from_pydict({"dept_id": [1, 2, 3], "name": ["eng", "sales", "ops"]})


def _ctx(ds, ndv=None):
    learned = {"__column_ndv__": ndv} if ndv else {}
    est = StatsEstimator(ds._sources, learned=learned)
    return OptimizerContext(
        config=active_config(), sources=ds._sources, hub=None, estimator=est
    )


def _grouped_max():
    return _emp().join(_dept(), on="dept_id").group_by("name").agg(top=col("sal").max())


def test_rule_registered():
    assert "eager_aggregation" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_pushes_partial_aggregate_below_join():
    ds = _grouped_max()
    out = eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0}))
    assert isinstance(out, Aggregate)
    assert isinstance(out.input, Join)
    assert isinstance(out.input.left, Aggregate)  # the pushed partial aggregate


def test_no_fire_without_reduction():
    # ndv == row count → grouping does not shrink the side → not worth pushing.
    ds = _grouped_max()
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 5.0})) is None


def test_no_fire_without_stats():
    # No ndv → the estimator can't prove a reduction → conservative no-op.
    ds = _grouped_max()
    assert eager_aggregation(ds._plan, _ctx(ds)) is None


def test_sum_not_pushed():
    # SUM is not fan-out-safe → never pushed.
    ds = _emp().join(_dept(), on="dept_id").group_by("name").agg(total=col("sal").sum())
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0})) is None


def test_right_side_aggregate_not_pushed():
    # The aggregate input is a right column → not a left-side push.
    ds = (
        _emp()
        .join(_dept(), on="dept_id")
        .group_by("dept_id")
        .agg(top=col("name").max())
    )
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0})) is None


def test_outer_join_not_pushed():
    ds = _emp().join(_dept(), on="dept_id", how="left").group_by("name").agg(top=col("sal").max())
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0})) is None


def test_idempotent():
    ds = _grouped_max()
    ctx = _ctx(ds, ndv={"dept_id": 3.0})
    once = eager_aggregation(ds._plan, ctx)
    # The pushed side is already reduced; a second push finds no further reduction.
    assert eager_aggregation(once, ctx) is None
