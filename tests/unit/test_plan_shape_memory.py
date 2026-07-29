"""The plan's *shape* is what decides its memory envelope, not its largest operator.

Two coupled gaps, both of which produced a confident envelope that was wrong by a large
factor on the most ordinary analytic plan shapes:

* `annotate_ops` sized a hash join at its **output** rows, where the resident state is the
  hash table over its **build** side. In a star schema those differ by the fan-out ratio.
* `PhysicalOp.inputs` was hardcoded empty, so Carbonite had no tree and could only take the
  largest single operator — which under-counts a bushy plan, in the direction that
  over-admits and OOMs.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import batcher as bt
from batcher.carbonite.memory.estimator import peak_operator_bytes
from batcher.config import active_config
from batcher.kyber.annotate import annotate_ops
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.optimizer import Optimizer
from batcher.plan.physical import PhysicalPlan

pytestmark = pytest.mark.unit


def _annotate(ds):
    est = CardinalityEstimator(ds._sources)
    ops = annotate_ops(ds._plan, est, active_config(), CostModel(est))
    return ops, PhysicalPlan(ir={}, output_schema=None, ops=ops)


def _fact(n: int = 100_000):
    return bt.from_pydict({"k": list(range(n)), "x": list(range(n))})


def _dim(n: int = 100):
    return bt.from_pydict({"k": list(range(n)), "y": list(range(n))})


# --- a join holds its build side, not its output ---------------------------------


def test_a_join_is_budgeted_at_its_build_side():
    # The star-schema shape, and the most common join in analytics: a large fact probing a
    # small dimension. The resident state is the dimension's hash table. Budgeting it at
    # the join's *output* — the fact fanned out by the match rate — over-sized a 100-row
    # build side by three orders of magnitude, and that figure is what admission checks,
    # what the spill decision reads, and what the distributed per-task grant derives from.
    ops, _ = _annotate(_fact().join(_dim(), on="k"))
    join = next(o for o in ops if o.kind == "Join")
    # Far below the join's own output volume, and in the neighbourhood of 100 rows of keys.
    assert 0 < join.bounds.m_max_bytes < 100_000


def test_the_join_envelope_agrees_with_the_cost_model():
    # Two subsystems sizing the same hash table must not disagree: `cost.py` ranks plans by
    # `mem = build_bytes` while `annotate_ops` hands Carbonite the figure it admits against.
    # A plan ranked on one number and admitted against another is the loop coming apart.
    ds = _fact().join(_dim(), on="k")
    est = CardinalityEstimator(ds._sources)
    model = CostModel(est)
    ops = annotate_ops(ds._plan, est, active_config(), model)
    join = next(o for o in ops if o.kind == "Join")
    assert join.bounds.m_max_bytes == pytest.approx(model.op_cost(ds._plan).mem, rel=0.01)


def test_a_fused_top_n_is_budgeted_at_its_heap():
    # A `Sort` carrying a limit holds a heap of `limit` rows, which is the whole reason to
    # fuse the two. Budgeting it at the relation makes the cheapest sort in the workload
    # look like the most expensive breaker in the plan. Annotated through the optimizer,
    # because the fusion that creates the shape is a rule.
    ds = _fact().sort("k").limit(10)
    est = CardinalityEstimator(ds._sources)
    fused = Optimizer(sources=ds._sources).logical_rewrite(ds._plan)
    ops = annotate_ops(fused, est, active_config(), CostModel(est))
    sort = next(o for o in ops if o.kind == "Sort")
    assert sort.bounds.m_max_bytes < 100_000


# --- the tree Kyber now hands over -----------------------------------------------


def test_annotate_records_the_plan_shape():
    # `inputs` was hardcoded empty and read by nothing, which is why the envelope below
    # could only ever be a `max`.
    ops, _ = _annotate(_fact().join(_dim(), on="k"))
    join = next(o for o in ops if o.kind == "Join")
    assert len(join.inputs) == 2
    scans = {int(o.op_id) for o in ops if o.kind == "Scan"}
    assert {int(i) for i in join.inputs} == scans


def test_a_bushy_plan_counts_the_breakers_that_are_live_together():
    # A join of two joins holds three hash tables at once. The largest-single reading
    # reports one of them; the concurrent reading is what admission actually needs.
    a, b, c, d = _fact(50_000), _dim(400), _dim(300), _dim(200)
    bushy = a.join(b, on="k").join(c.join(d, on="k"), on="k")
    ops, plan = _annotate(bushy)
    largest_single = max(o.bounds.m_max_bytes for o in ops)
    assert peak_operator_bytes(plan) > largest_single


def test_a_linear_plan_is_byte_for_byte_what_it_was():
    # The safety property: walking the schedule must not move a linear pipeline's
    # envelope, because a unary breaker's input has finished and released by the time its
    # own state is full.
    ds = _fact().group_by("k").agg(s=bt.col("x").sum()).sort("s")
    ops, plan = _annotate(ds)
    assert peak_operator_bytes(plan) == max(o.bounds.m_max_bytes for o in ops)


def test_an_unwired_plan_falls_back_to_the_previous_reading():
    # Every hand-built `PhysicalPlan` and test double carries no `inputs`. Those must get
    # exactly the pre-tree behavior rather than a zero.
    ops, _ = _annotate(_fact().join(_dim(), on="k"))
    bare = PhysicalPlan(ir={}, output_schema=None, ops=tuple(replace(o, inputs=()) for o in ops))
    assert peak_operator_bytes(bare) == max(o.bounds.m_max_bytes for o in ops)
