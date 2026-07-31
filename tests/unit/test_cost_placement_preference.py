"""PACK vs SPREAD: when concentrating a gang moves an exchange onto a faster tier.

The guarantee under test is that this only ever *adds* a PACK. Every fleet the shape model
cannot reason about, and every gang too wide to concentrate, must reach exactly the byte
threshold the rule used before.
"""

from __future__ import annotations

import pytest

from batcher.kyber.cost.placement import prefers_locality
from batcher.plan.resource import ClusterShape, HardwareProfile, NodeShape

pytestmark = pytest.mark.unit

_THRESHOLD = 4 * 1024 * 1024  # optimizer.locality_max_bytes


def _fleet(nodes: int, cores: int = 16, fabric: float = 25.0) -> ClusterShape:
    return ClusterShape(
        nodes=tuple(
            NodeShape(
                node_id=f"n{i}",
                cpu_cores=cores,
                memory_bytes=1 << 40,
                rack="r1",
                fabric_gbps=fabric,
            )
            for i in range(nodes)
        )
    )


def _profile(shape: ClusterShape, workers: int) -> HardwareProfile:
    return HardwareProfile.for_cluster(
        cpu_cores=16, memory_bytes=1 << 40, worker_count=workers, cluster=shape
    )


def test_a_small_shuffle_still_packs_whatever_the_fleet():
    """The original rule stands on its own terms and is unchanged."""
    for hardware in (HardwareProfile(), _profile(_fleet(16), 32), _profile(_fleet(64, 1), 64)):
        assert prefers_locality(hardware, 32, 1024.0, _THRESHOLD).pack is True


def test_a_large_shuffle_on_an_unreadable_fleet_spreads():
    """No shape, no new decision — the pre-existing SPREAD."""
    advice = prefers_locality(HardwareProfile(worker_count=32), 32, 1e9, _THRESHOLD)
    assert advice.pack is False
    assert advice.nodes == 0
    assert "unknown" in advice.reason


def test_a_large_shuffle_packs_when_the_gang_fits_dense_nodes():
    """The case the byte threshold got backwards: a big exchange onto few fat hosts."""
    advice = prefers_locality(_profile(_fleet(16, cores=16), 32), 32, 1e9, _THRESHOLD)
    assert advice.pack is True
    assert advice.nodes == 2
    assert advice.saving > 0.15


def test_a_gang_that_cannot_be_concentrated_spreads():
    """Sixty-four workers over one-core hosts span sixty-four nodes; packing is not available.

    This is also the guard on the failure-domain argument the module deliberately does not
    price: past a handful of nodes the concentration is not a concentration.
    """
    advice = prefers_locality(_profile(_fleet(64, cores=1), 64), 64, 1e9, _THRESHOLD)
    assert advice.pack is False
    assert advice.nodes == 64


def test_a_fleet_too_small_to_hold_the_gang_reports_no_opinion():
    """A fan-out wider than the whole fleet cannot be packed onto part of it."""
    advice = prefers_locality(_profile(_fleet(2, cores=4), 64), 64, 1e9, _THRESHOLD)
    assert advice.pack is False
    assert advice.nodes == 0


def test_no_discount_available_means_no_pack():
    """On a fabric faster than host memory, packing a relational gang saves nothing.

    The rule must follow the physics rather than the density: concentrating a fleet for a
    saving that is not there costs failure domain and read bandwidth for free.
    """
    advice = prefers_locality(_profile(_fleet(16, fabric=400.0), 32), 32, 1e9, _THRESHOLD)
    assert advice.pack is False
    assert advice.saving == pytest.approx(0.0)


def test_a_single_worker_is_not_a_placement_decision():
    assert prefers_locality(_profile(_fleet(16), 1), 1, 1e9, _THRESHOLD).pack is False


def test_the_advice_carries_a_reason_for_the_decision_log():
    """Every branch explains itself; a placement that moved for no stated reason is a bug."""
    for workers, shape in ((32, _fleet(16)), (64, _fleet(64, cores=1)), (32, _fleet(2, cores=4))):
        assert prefers_locality(_profile(shape, workers), workers, 1e9, _THRESHOLD).reason


def test_annotation_still_packs_a_small_breaker_end_to_end():
    """The wiring reaches `annotate_ops`, and a local plan's preference is unchanged."""
    import batcher as bt
    from batcher import col
    from batcher.config import active_config
    from batcher.kyber.annotate import annotate_ops
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel

    frame = bt.from_pydict({"k": list(range(64)), "v": list(range(64))})
    plan = frame.group_by("k").agg(total=col("v").sum())._plan
    estimator = CardinalityEstimator(frame._sources)
    ops = annotate_ops(plan, estimator, active_config(), CostModel(estimator))
    assert any(op.bounds.prefers_locality for op in ops)
