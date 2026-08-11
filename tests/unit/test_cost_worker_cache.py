"""The cache-residency term is the worker's cache, not the driver's.

The sibling of `test_cost_worker_spill_budget`, for the other machine-shaped multiplier in
`cost.terms`. `cache_factor` probed *this process's* last-level cache unconditionally, so on a
distributed run it described the driver — and on the usual cluster shape the driver is a fat
head node beside small workers. A driver publishing 100 MiB of L3 planning for 4 MiB workers
under-states the cache knee by more than an octave.

That term is not incidental: it multiplies the probe cost of every hash operator, so the error
lands on join ordering, aggregate, distinct and distinct union at once, and `terms` itself calls
it "precisely what join ordering is choosing between".

As with the spill budget, the property that matters most is that a single-node run cannot move:
`None` keeps the local probe, which is what every caller without a profile passes.
"""

from __future__ import annotations

import math

import pytest

from batcher._internal.hardware import l3_cache_bytes
from batcher.kyber.cost.terms import cache_factor

pytestmark = pytest.mark.unit

_MIB = 1 << 20


def test_no_cache_figure_probes_this_machine():
    """Every caller without a profile — which is every caller before this — is unchanged."""
    assert cache_factor(1e9) == pytest.approx(cache_factor(1e9, l3_cache_bytes() or None))


def test_a_resident_table_costs_nothing_extra():
    """Inside the cache there is no penalty, whichever cache is being described."""
    assert cache_factor(1 * _MIB, 8 * _MIB) == 1.0
    assert cache_factor(8 * _MIB, 8 * _MIB) == 1.0


def test_a_smaller_worker_cache_costs_more():
    """The fat-driver cluster: the same table misses more often on the node that runs it."""
    state = 256.0 * _MIB
    assert cache_factor(state, 4 * _MIB) > cache_factor(state, 64 * _MIB)


def test_the_penalty_grows_per_octave_and_is_capped():
    """The shape the constant describes: flat, then a knee per doubling, then a plateau."""
    one_octave = cache_factor(16 * _MIB, 8 * _MIB)
    two_octaves = cache_factor(32 * _MIB, 8 * _MIB)
    assert two_octaves - one_octave == pytest.approx(one_octave - 1.0)
    # Far past the knee every access already misses and there is nothing left to lose.
    assert cache_factor(1 << 50, 8 * _MIB) == cache_factor(1 << 60, 8 * _MIB)


def test_an_undetectable_cache_reports_no_opinion():
    """`0` is "this platform could not report it", and must not read as a zero-byte cache."""
    assert cache_factor(1e12, 0) == 1.0


def test_the_cost_model_reads_the_profile():
    """End to end: two models over one plan, differing only in the worker they target."""
    import batcher as bt
    from batcher import col
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel
    from batcher.plan.resource import HardwareProfile

    frame = bt.from_pydict({"k": list(range(5_000)), "v": list(range(5_000))})
    plan = frame.group_by("k").agg(total=col("v").sum())._plan
    estimator = CardinalityEstimator(frame._sources)
    # A cache small enough that this plan's group table overflows it, against one large enough
    # to hold it. Only the cache differs, so only the cache term can explain the gap.
    small = CostModel(estimator, hardware=HardwareProfile(l3_cache_bytes=4096))
    large = CostModel(estimator, hardware=HardwareProfile(l3_cache_bytes=1 << 34))
    assert small.cost(plan).cpu > large.cost(plan).cpu


def test_an_unprobeable_cluster_falls_back_to_the_local_probe():
    """`0` on the profile is "the workers could not be probed", not "they have no cache"."""
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel
    from batcher.plan.resource import HardwareProfile

    frame = bt.from_pydict({"k": [1, 2, 3]})
    estimator = CardinalityEstimator(frame._sources)
    unprobeable = CostModel(estimator, hardware=HardwareProfile(cpu_cores=8))
    bare = CostModel(estimator)
    assert unprobeable._cache_bytes is None
    assert bare._cache_bytes is None


def test_the_knee_lands_where_the_analytic_form_says():
    """An assertion the implementation cannot satisfy by construction.

    Pinning the penalty against the closed form it documents, rather than against itself, so a
    change to the constant or the log base fails here instead of silently re-pricing every hash
    operator in the engine.
    """
    cache = 8 * _MIB
    state = 128.0 * _MIB
    octaves = math.log2(state / cache)
    assert cache_factor(state, cache) == pytest.approx(1.0 + 0.35 * octaves)
