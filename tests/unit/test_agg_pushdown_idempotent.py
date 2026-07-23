"""Structural-idempotency guards for the aggregate-through-join pushdown rules.

These rules (`eager_aggregation`, `pre_aggregation_through_join`,
`pre_aggregate_join_measures`) rewrite `Aggregate(Join(...))` into a pre-aggregated
join. Each is *documented* as idempotent, but that idempotency used to rest entirely on
the cost gate declining a second push (the already-reduced side shows no further
reduction). That makes correctness depend on an estimate — and when the estimate is
wrong (or bypassed), the second application is not merely wasteful: the LEFT-join
`count` merge (`count` → `sum(coalesce(__pm, 0))`) loses its `coalesce` on re-entry
(the outer aggregate is now a `sum`, not a `count`), so a fully-unmatched group's answer
flips from 0 to NULL, and the additive rules build an invalid join output with a
duplicated partial-column name.

The rules now carry a *structural* guard (`_already_grouped_by`) that refuses to re-push
onto a side already grouped by the join key, independent of the cost model. These tests
pin that guard by forcing the cost gate open, so they exercise the rewrite itself rather
than the estimate that normally hides the second firing.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules import agg_pushdown
from batcher.kyber.rules.agg_pushdown import (
    eager_aggregation,
    pre_aggregate_join_measures,
    pre_aggregation_through_join,
)
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Aggregate


def _ctx(ds):
    est = StatsEstimator(ds._sources)
    return OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)


def _force_gate_open(monkeypatch):
    # The cost gate is a performance heuristic, not a correctness guarantee: force it open
    # so a rewrite that is only "safe" because the gate declines the second push is exposed.
    monkeypatch.setattr(agg_pushdown, "_reduces_enough", lambda ctx, pushed, source: True)


def test_pre_aggregate_join_measures_idempotent(monkeypatch):
    # customer LEFT JOIN orders GROUP BY nation, COUNT(okey) — the TPC-H Q13 shape.
    cust = bt.from_pydict({"custkey": [1, 2, 3, 4], "nation": [10, 10, 20, 20]})
    orders = bt.from_pydict({"okey": [1, 2, 3, 4, 5], "custkey": [1, 1, 1, 2, 2]})
    ds = cust.join(orders, on="custkey", how="left").group_by("nation").agg(n=col("okey").count())
    ctx = _ctx(ds)
    _force_gate_open(monkeypatch)

    once = pre_aggregate_join_measures(ds._plan, ctx)
    assert isinstance(once, Aggregate)  # it fires the first time
    # Second application must be a no-op even with the gate forced open — re-firing would
    # drop the LEFT-join count's coalesce and corrupt the unmatched-group answer.
    assert pre_aggregate_join_measures(once, ctx) is None


def test_eager_aggregation_idempotent(monkeypatch):
    emp = bt.from_pydict({"dept_id": [1, 2, 3] * 4, "sal": list(range(12))})
    dept = bt.from_pydict({"dept_id": [1, 2, 3], "name": ["eng", "sales", "ops"]})
    ds = emp.join(dept, on="dept_id").group_by("name").agg(top=col("sal").max())
    ctx = _ctx(ds)
    _force_gate_open(monkeypatch)

    once = eager_aggregation(ds._plan, ctx)
    assert isinstance(once, Aggregate)
    assert eager_aggregation(once, ctx) is None


def test_pre_aggregation_through_join_idempotent(monkeypatch):
    # left side fans out on the key; right side is a GROUP BY on the key so it is *provably*
    # unique on it (the additive push's precondition).
    li = bt.from_pydict({"okey": [1, 1, 2, 2, 2, 3], "price": [10, 20, 30, 40, 50, 60]})
    orders = bt.from_pydict({"okey": [1, 2, 3], "prio": [1, 2, 3]})
    orders_unique = orders.group_by("okey").agg(p=col("prio").max())
    ds = li.join(orders_unique, on="okey").group_by("okey").agg(s=col("price").sum())
    ctx = _ctx(ds)
    _force_gate_open(monkeypatch)

    once = pre_aggregation_through_join(ds._plan, ctx)
    assert isinstance(once, Aggregate)
    assert pre_aggregation_through_join(once, ctx) is None
