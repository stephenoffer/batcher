"""Plan-shape, refusal and idempotence tests for `agg_algebra` (SUM linearity).

The rule collapses a family of ``SUM(base ± c)`` / ``SUM(base * c)`` onto one shared
``SUM(base)`` (+ ``COUNT(base)``), deriving each original output by a scalar projection.
Result-correctness vs DuckDB lives in `tests/differential/test_diff_agg_linearity.py`.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules import agg_algebra as m
from batcher.plan.logical import Aggregate, Project
from batcher.plan.visitor import walk


def _ds():
    return bt.from_pydict({"g": [1, 2, 3], "w": [10, 20, 30]})


def _ctx(ds):
    return Optimizer(sources=ds._sources)._context()


def _optimize(ds, plan):
    return Optimizer(sources=ds._sources).logical_rewrite(plan)


def _aggregates(plan):
    return [n for n in walk(plan) if isinstance(n, Aggregate)]


def test_fires_on_a_shifted_sum_family():
    ds = _ds()
    plan = ds.agg(**{f"s{i}": (col("w") + i).sum() for i in range(5)})._plan
    out = m.decompose_linear_sum_aggregates(plan, _ctx(ds))
    assert isinstance(out, Project)
    # Five SUM(w + i) collapse to exactly SUM(w) + COUNT(w).
    aggs = _aggregates(out)
    assert len(aggs) == 1
    assert sorted(s.agg.func for s in aggs[0].aggregates) == ["count", "sum"]
    # The projection still exposes all five original outputs, in order.
    assert [it.alias for it in out.items] == [f"s{i}" for i in range(5)]


def test_idempotent_does_not_refire_on_its_own_output():
    ds = _ds()
    plan = ds.agg(**{f"s{i}": (col("w") + i).sum() for i in range(5)})._plan
    out = m.decompose_linear_sum_aggregates(plan, _ctx(ds))
    # The rebuilt aggregate holds only SUM/COUNT, so the rule declines a second pass —
    # its own output can never re-trigger it, which is what keeps the fixpoint convergent.
    (rebuilt,) = _aggregates(out)
    assert m.decompose_linear_sum_aggregates(rebuilt, _ctx(ds)) is None
    # The full optimizer settles (a further re-optimization is a no-op).
    settled = _optimize(ds, _optimize(ds, plan))
    assert _optimize(ds, settled).to_ir() == settled.to_ir()


def test_refuses_a_lone_shifted_sum():
    # One SUM(w + 1) would become SUM(w) + COUNT(w) — two aggregates for one, MORE work.
    ds = _ds()
    plan = ds.agg(s=(col("w") + 1).sum())._plan
    assert m.decompose_linear_sum_aggregates(plan, _ctx(ds)) is None


def test_no_op_on_plain_sums():
    ds = _ds()
    plan = ds.agg(a=col("w").sum(), b=col("g").sum())._plan
    assert m.decompose_linear_sum_aggregates(plan, _ctx(ds)) is None


def test_shares_across_sub_and_mul_on_the_same_base():
    ds = _ds()
    plan = ds.agg(
        a=(col("w") + 1).sum(),
        b=(col("w") - 2).sum(),
        c=(col("w") * 3).sum(),
    )._plan
    out = m.decompose_linear_sum_aggregates(plan, _ctx(ds))
    assert isinstance(out, Project)
    aggs = _aggregates(out)[0].aggregates
    # One SUM(w) + one COUNT(w) serve all three (mul needs only the sum).
    assert sorted(s.agg.func for s in aggs) == ["count", "sum"]


def test_grouped_family_keeps_group_keys():
    ds = _ds()
    plan = ds.group_by("g").agg(**{f"s{i}": (col("w") + i).sum() for i in range(4)})._plan
    out = m.decompose_linear_sum_aggregates(plan, _ctx(ds))
    assert isinstance(out, Project)
    agg = _aggregates(out)[0]
    assert [k.alias for k in agg.group_keys] == ["g"]
    assert out.items[0].alias == "g"  # group key passed through first, in order


def test_fires_end_to_end_through_the_real_optimizer():
    ds = _ds()
    plan = ds.agg(**{f"s{i}": (col("w") + i).sum() for i in range(6)})._plan
    out = _optimize(ds, plan)
    # After the full rewrite the single aggregate carries far fewer specs than 6.
    assert all(len(a.aggregates) <= 2 for a in _aggregates(out))
