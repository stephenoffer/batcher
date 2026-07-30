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


def test_the_peak_names_every_operator_that_contributes_to_it():
    # `peak_contributors` is what keeps the rest of Carbonite honest about the envelope
    # having become a *sum*: on a bushy plan it must name both live tables, not just the
    # larger one.
    from batcher.carbonite.memory.estimator import peak_contributors

    a, b, c, d = _fact(50_000), _dim(400), _dim(300), _dim(200)
    _, plan = _annotate(a.join(b, on="k").join(c.join(d, on="k"), on="k"))
    contributors = peak_contributors(plan)
    assert len(contributors) > 1
    assert sum(o.bounds.m_max_bytes for o in contributors) == peak_operator_bytes(plan)


def test_a_linear_plan_names_exactly_its_dominant_breaker():
    from batcher.carbonite.memory.estimator import peak_contributors

    ds = _fact().group_by("k").agg(s=bt.col("x").sum()).sort("s")
    _, plan = _annotate(ds)
    contributors = peak_contributors(plan)
    assert len(contributors) == 1
    assert contributors[0].bounds.m_max_bytes == peak_operator_bytes(plan)


def test_a_guess_in_any_contributor_keeps_the_verdict_advisory():
    # The admission contract is that a plan sized from a guess may be routed out-of-core
    # but never *failed*. Once the envelope became a sum, reading only the largest term
    # would fail a legitimate query on the strength of a smaller guessed one.
    from dataclasses import replace as dc_replace

    from batcher.carbonite.policies.admission import _rests_on_a_guess
    from batcher.plan.physical import PhysicalPlan as PP
    from batcher.plan.stats import Provenance

    ops, _ = _annotate(_fact(50_000).join(_dim(400), on="k").join(_dim(300), on="k"))
    sized = [o for o in ops if o.bounds.m_max_bytes > 0]
    if len(sized) < 2:  # pragma: no cover - plan shape without two sized operators
        pytest.skip("plan has fewer than two sized operators")
    exact = tuple(
        dc_replace(o, properties=dc_replace(o.properties, provenance=Provenance.EXACT))
        for o in ops
    )
    all_exact = PP(ir={}, output_schema=None, ops=exact)
    binding = max(exact, key=lambda o: o.bounds.m_max_bytes)
    assert _rests_on_a_guess(all_exact, binding) is False

    # Now make the *smallest* contributor a guess: the verdict must go advisory.
    smallest = min(
        (o for o in exact if o.bounds.m_max_bytes > 0), key=lambda o: o.bounds.m_max_bytes
    )
    guessed = tuple(
        dc_replace(o, properties=dc_replace(o.properties, provenance=Provenance.DEFAULT))
        if o.op_id == smallest.op_id
        else o
        for o in exact
    )
    mixed = PP(ir={}, output_schema=None, ops=guessed)
    from batcher.carbonite.memory.estimator import peak_contributors

    if smallest.op_id not in {o.op_id for o in peak_contributors(mixed)}:
        pytest.skip("the smallest sized operator does not contribute to this plan's peak")
    assert _rests_on_a_guess(mixed, binding) is True


def test_a_warm_store_still_gets_the_concurrent_peak():
    """The correction must not switch itself off once the engine has learned something.

    `learned_plan_peak` used to hand the whole plan to the model's own flat aggregate, so
    the schedule walk applied *only on a cold store* — that is, only until the first query
    of any shape had run. A correction that silently stops applying once a system is warm is
    worse than one never written, because every cold-path test stays green.
    """
    from batcher.carbonite.memory.estimator import learned_plan_peak

    class _Doubling:
        def blend_peak(self, kind, planned):
            return planned * 2

    a, b, c, d = _fact(50_000), _dim(400), _dim(300), _dim(200)
    _, plan = _annotate(a.join(b, on="k").join(c.join(d, on="k"), on="k"))
    cold = peak_operator_bytes(plan)
    warm = learned_plan_peak(plan, _Doubling())
    largest_blended = 2 * max(o.bounds.m_max_bytes for o in plan.ops)
    # The model is honoured...
    assert warm == 2 * cold
    # ...and the schedule is still walked, rather than collapsing to a blended `max`.
    assert warm > largest_blended


def test_a_cold_store_is_exactly_the_plan_estimate():
    from batcher.carbonite.memory.estimator import learned_plan_peak

    _, plan = _annotate(_fact().join(_dim(), on="k"))
    assert learned_plan_peak(plan, None) == peak_operator_bytes(plan)
