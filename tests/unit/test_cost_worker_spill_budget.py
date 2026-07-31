"""The spill threshold is the worker's memory, not the driver's.

`api.orchestration.autoconfig` senses `max_memory_bytes` from *this process's* free RAM, and
on the usual cluster shape this process is a fat head node beside small workers. An operator
whose state sits between the two was predicted not to spill — and `cost.terms` calls costing a
spill at zero "the single largest cost error a plan can contain".

The correction combines the two figures with `min`, so the property that matters most is that a
single-node run cannot move: a machine's total RAM always exceeds its free RAM, so the sensed
value keeps winning there.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import active_config, set_config
from batcher.kyber.cost.terms import memory_budget, merge_io, merge_passes, spill_io

pytestmark = pytest.mark.unit

_GIB = 1 << 30


@pytest.fixture
def capped():
    """A config with an explicit 64 GiB envelope, standing in for a fat driver."""
    previous = active_config()
    cfg = dataclasses.replace(
        previous, memory=dataclasses.replace(previous.memory, max_memory_bytes=64 * _GIB)
    )
    set_config(cfg)
    try:
        yield cfg
    finally:
        set_config(previous)


def test_no_worker_figure_is_the_configured_budget(capped):
    """Every caller without a profile — which is every caller before this — is unchanged."""
    assert memory_budget() == pytest.approx(capped.spill_budget_bytes())
    assert memory_budget(0) == pytest.approx(capped.spill_budget_bytes())


def test_a_smaller_worker_lowers_the_budget(capped):
    """The fat-driver cluster: the plan is ranked against the node it will run on."""
    assert memory_budget(8 * _GIB) == pytest.approx(8 * _GIB * capped.memory.hard_limit)
    assert memory_budget(8 * _GIB) < capped.spill_budget_bytes()


def test_a_larger_worker_never_raises_it(capped):
    """`min`, not substitution. A worker's *total* RAM is not a licence to spill later.

    The configured figure is free RAM and the worker figure is total RAM, so a substitution
    would silently raise the threshold on every single-node run.
    """
    assert memory_budget(1024 * _GIB) == pytest.approx(capped.spill_budget_bytes())


def test_opting_out_of_bounded_memory_is_not_re_armed():
    """`unbounded_memory` means nothing spills, and a worker figure must not override it."""
    base = active_config()
    cfg = dataclasses.replace(base, memory=dataclasses.replace(base.memory, unbounded_memory=True))
    set_config(cfg)
    try:
        assert memory_budget(1 * _GIB) == 0.0
        assert spill_io(1e12, memory_budget(1 * _GIB)) == 0.0
    finally:
        set_config(base)


def test_an_operator_between_the_two_figures_now_costs_its_spill(capped):
    """The defect, stated as the case it produces.

    A 16 GiB hash table on an 8 GiB worker fits the driver's 64 GiB budget and does not fit the
    worker. Costed against the driver it was free; costed against the worker it is disk-bound.
    """
    state = 16.0 * _GIB
    assert spill_io(state) == 0.0
    assert spill_io(state, memory_budget(8 * _GIB)) > 0.0


def test_a_sort_over_budget_gains_merge_passes(capped):
    """The same correction reaches the out-of-core sort's pass count."""
    # Inside the driver's 64 GiB envelope (times `hard_limit`), outside a 4 GiB worker's.
    state = 32.0 * _GIB
    assert merge_passes(state) == 0.0
    assert merge_passes(state, memory_budget(4 * _GIB)) >= 1.0
    assert merge_io(state, memory_budget(4 * _GIB)) > 0.0


def test_the_cost_model_reads_the_profile(capped):
    """End to end: two models over one plan, differing only in the worker they target."""
    import batcher as bt
    from batcher import col
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel
    from batcher.plan.resource import HardwareProfile

    frame = bt.from_pydict({"k": list(range(500)), "v": list(range(500))})
    plan = frame.group_by("k").agg(total=col("v").sum())._plan
    estimator = CardinalityEstimator(frame._sources)
    # A tiny worker makes even this plan's state exceed its budget; a huge one does not.
    tiny = CostModel(estimator, hardware=HardwareProfile(memory_bytes=1))
    fat = CostModel(estimator, hardware=HardwareProfile(memory_bytes=1024 * _GIB))
    assert tiny.cost(plan).io > fat.cost(plan).io
    # And no profile ranks exactly as a fat one does, since the configured budget binds.
    bare = CostModel(estimator)
    assert bare.cost(plan).io == pytest.approx(fat.cost(plan).io)
