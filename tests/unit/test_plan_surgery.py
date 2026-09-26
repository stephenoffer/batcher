"""The adaptive loop's plan surgery sees every child the optimizer does."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.adaptive.plan_surgery import lowest_breaker, replace
from batcher.plan.logical import Aggregate, Scan
from batcher.plan.logical.join import AsofJoin
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _asof_over_an_aggregate():
    left = bt.from_arrow(pa.table({"t": [1, 2, 3, 4], "k": [1, 1, 2, 2]}))
    grouped = left.group_by("t").agg(n=bt.col("k").sum())
    right = bt.from_arrow(pa.table({"t": [0, 2, 4], "w": [10, 20, 30]}))
    return grouped.sort("t").join_asof(right.sort("t"), on="t")._plan


def test_a_breaker_beneath_an_asof_join_is_found():
    # `AsofJoin` is not a `Join` subclass, so a hand-written `Join`/`Union`/`.input` switch
    # reported no children for it and the stage loop could never cut beneath one.
    plan = _asof_over_an_aggregate()
    assert any(isinstance(n, AsofJoin) for n in walk(plan))
    found = lowest_breaker(plan)
    assert isinstance(found, Aggregate)


def test_replace_splices_through_an_asof_join():
    plan = _asof_over_an_aggregate()
    target = next(n for n in walk(plan) if isinstance(n, Aggregate))
    stub = next(n for n in walk(plan) if isinstance(n, Scan))
    spliced = replace(plan, target, stub)
    assert spliced is not plan
    assert not any(isinstance(n, Aggregate) for n in walk(spliced))
    assert isinstance(spliced, type(plan))


def test_staging_beneath_an_asof_join_keeps_the_result():
    left = bt.from_arrow(pa.table({"t": [1, 2, 3, 4, 3], "k": [1, 1, 2, 2, 5]}))
    right = bt.from_arrow(pa.table({"t": [0, 2, 4], "w": [10, 20, 30]}))
    ds = (
        left.group_by("t")
        .agg(n=bt.col("k").sum())
        .sort("t")
        .join_asof(right.sort("t"), on="t")
        .sort("t")
    )
    assert ds.collect(adaptive=True).to_pydict() == ds.collect(adaptive=False).to_pydict()
