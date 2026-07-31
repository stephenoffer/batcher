"""Pricing a shuffled byte by the tier it crosses, and the guarantee that unknowns cost nothing.

Two properties matter more than any specific number here. The factor must be exactly `1.0`
wherever anything is unreadable, so no existing ranking moves on a fleet Batcher cannot see; and
it must never exceed `1.0`, so the tier model can only ever make an exchange cheaper than the
flat network price the axis charged before.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.cost.locality import locality_factor, locality_summary, tier_prices
from batcher.kyber.cost.shuffle import net_cost, shuffle_bytes
from batcher.plan.resource import ClusterShape, HardwareProfile, NodeShape

pytestmark = pytest.mark.unit


def _fleet(nodes: int, gpus: int, *, domain: int = 8, fabric: float = 400.0) -> ClusterShape:
    return ClusterShape(
        nodes=tuple(
            NodeShape(
                node_id=f"n{i}",
                cpu_cores=96,
                gpus=gpus,
                accelerator_type="NVIDIA_H100",
                gpu_memory_bytes=80 * (1 << 30),
                nvlink_domain=min(domain, gpus),
                rack="r1",
                fabric_gbps=fabric,
            )
            for i in range(nodes)
        )
    )


def _hardware(shape: ClusterShape, workers: int) -> HardwareProfile:
    return HardwareProfile.for_cluster(
        cpu_cores=96,
        memory_bytes=1 << 40,
        worker_count=workers,
        gpu_count=shape.total_gpus,
        gpu_memory_bytes=shape.binding_gpu_memory_bytes,
        accelerator_type="NVIDIA_H100",
        cluster=shape,
    )


def test_unknown_fleet_charges_the_flat_network_rate():
    """No shape, no discount — the ranking every existing deployment already has."""
    assert locality_factor(HardwareProfile(worker_count=32), 32) == 1.0
    assert locality_factor(None, 32) == 1.0


def test_single_worker_has_no_exchange_to_price():
    assert locality_factor(_hardware(_fleet(4, 8), 1), 1, unit="gpu") == 1.0


def test_unmeasurable_fabric_charges_the_flat_rate():
    """A fleet with no fabric figure keeps the flat price rather than guessing at one.

    The discount is a ratio of two bandwidths; without the denominator there is no ratio, and
    inventing one would put a fabricated number under a join order.
    """
    unrated = _fleet(4, 8, fabric=0.0)
    assert locality_factor(_hardware(unrated, 32), 32, unit="gpu") == 1.0
    assert tier_prices("NVIDIA_H100", fabric_gbps=0.0).measured is False


def test_dense_nodes_are_cheaper_to_shuffle_across_than_sparse_ones():
    """The distinction the tier model exists to draw, at identical device and worker counts."""
    dense = locality_factor(_hardware(_fleet(4, 8), 32), 32, unit="gpu")
    sparse = locality_factor(_hardware(_fleet(32, 1), 32), 32, unit="gpu")
    assert dense < sparse
    assert sparse == pytest.approx(1.0)


def test_denser_is_monotonically_cheaper():
    """Packing the same devices onto fewer hosts never makes the exchange cost more."""
    factors = [
        locality_factor(_hardware(_fleet(nodes, 32 // nodes), 32), 32, unit="gpu")
        for nodes in (32, 16, 8, 4, 2, 1)
    ]
    assert factors == sorted(factors, reverse=True)


@pytest.mark.parametrize("nodes,gpus", [(1, 8), (2, 4), (4, 8), (8, 2), (32, 1)])
@pytest.mark.parametrize("unit", ["cpu", "gpu"])
def test_the_factor_never_makes_an_exchange_dearer(nodes, gpus, unit):
    """A tier price is a discount or nothing; it can never inflate the flat byte count."""
    factor = locality_factor(_hardware(_fleet(nodes, gpus), nodes * gpus), nodes * gpus, unit=unit)
    assert 0.0 < factor <= 1.0


def test_a_device_fabric_is_priced_far_under_the_network():
    """NVLink against a 400 Gb/s NIC is a large discount, and the arithmetic is unit-correct.

    The device rate is published in gigabytes and the NIC's in gigabits; a comparison that
    skipped the conversion would look plausible and be wrong by eight.
    """
    prices = tier_prices("NVIDIA_H100", fabric_gbps=400.0)
    assert prices.intra_domain < 0.1
    assert prices.measured is True
    # A fabric faster than the device link removes the discount rather than inverting it.
    assert tier_prices("NVIDIA_H100", fabric_gbps=100_000.0).intra_domain == 1.0


def test_an_unlabelled_device_falls_back_to_the_host_tier():
    """With no device model, on-host traffic is priced at host memory — never at the NIC.

    Two workers on one host exchange through DRAM at worst, so charging them the network rate
    over-states exactly the placement a fan-out is trying to prefer.
    """
    unlabelled = tier_prices("", fabric_gbps=400.0)
    assert unlabelled.intra_domain == unlabelled.intra_node


def test_cpu_and_gpu_units_are_priced_differently_on_the_same_fleet():
    """A relational shuffle does not get the device fabric's discount."""
    hardware = _hardware(_fleet(4, 8), 32)
    assert locality_factor(hardware, 32, unit="gpu") < locality_factor(hardware, 32, unit="cpu")


def _aggregate_plan():
    """A group-by whose `net` axis is a real shuffle, built through the public API."""
    frame = bt.from_pydict({"k": list(range(1000)), "v": [i % 7 for i in range(1000)]})
    return frame.group_by("k").agg(total=col("v").sum())._plan


def test_net_cost_scales_linearly_with_the_locality_factor():
    """The factor multiplies the flat cost, and its default of `1.0` reproduces it exactly."""
    node = _aggregate_plan()
    rows_of, width_of = (lambda _n: 1000.0), (lambda _n: 64.0)
    flat = net_cost(node, rows_of, width_of, 8)
    assert flat > 0.0
    assert net_cost(node, rows_of, width_of, 8, 1.0) == pytest.approx(flat)
    assert net_cost(node, rows_of, width_of, 8, 0.5) == pytest.approx(flat * 0.5)


def test_net_cost_is_zero_on_one_worker_whatever_the_locality():
    node = _aggregate_plan()
    assert net_cost(node, lambda _n: 10.0, lambda _n: 8.0, 1, 0.01) == 0.0


def test_a_dense_fleet_ranks_a_shuffle_below_a_sparse_one_end_to_end():
    """The factor reaches the cost model, not just the helper — same query, two fleets.

    On a commodity 25 Gb/s fabric, which is what most rented capacity has, a relational
    exchange between two workers on one host moves through host memory at several times the
    NIC's rate. The dense fleet keeps a quarter of its shuffle on that path and the sparse one
    keeps none, and before the tier model the two were priced identically.
    """
    frame = bt.from_pydict({"k": list(range(1000)), "v": [i % 7 for i in range(1000)]})
    plan = frame.group_by("k").agg(total=col("v").sum())._plan
    estimator = CardinalityEstimator(frame._sources)
    dense = CostModel(estimator, workers=32, hardware=_hardware(_fleet(4, 8, fabric=25.0), 32))
    sparse = CostModel(estimator, workers=32, hardware=_hardware(_fleet(32, 1, fabric=25.0), 32))
    assert dense.cost(plan).net < sparse.cost(plan).net


def test_a_fast_fabric_removes_the_relational_discount():
    """On a 400 Gb/s fleet a host-memory copy is *slower* than the NIC, so no discount applies.

    The honest answer rather than a convenient one: the model prices a byte by the rate that
    carries it, and on that fabric an on-host relational exchange has nothing to gain.
    """
    hardware = _hardware(_fleet(4, 8, fabric=400.0), 32)
    assert locality_factor(hardware, 32, unit="cpu") == pytest.approx(1.0)


def test_a_cost_model_with_no_hardware_is_unchanged():
    """Every existing construction site passes no profile and must rank exactly as before."""
    frame = bt.from_pydict({"k": list(range(1000)), "v": [i % 7 for i in range(1000)]})
    plan = frame.group_by("k").agg(total=col("v").sum())._plan
    estimator = CardinalityEstimator(frame._sources)
    bare = CostModel(estimator, workers=32)
    unknown = CostModel(estimator, workers=32, hardware=HardwareProfile(worker_count=32))
    assert bare.cost(plan).net == pytest.approx(unknown.cost(plan).net)


def test_shuffle_bytes_is_unchanged():
    """The volume form itself is untouched — the tier model is a multiplier on top of it."""
    assert shuffle_bytes(1000.0, 100.0, 1) == 0.0
    assert shuffle_bytes(1000.0, 100.0, 2) == pytest.approx(50_000.0)


def test_summary_reports_whether_a_measurement_was_used():
    """A reader must be able to tell a measured ranking from a defaulted one."""
    flat = locality_summary(HardwareProfile(worker_count=8), 8)
    assert flat["factor"] == 1.0 and "flat" in flat["basis"]
    measured = locality_summary(_hardware(_fleet(4, 8), 32), 32, unit="gpu")
    assert measured["factor"] < 1.0
    assert sum(measured["shares"].values()) == pytest.approx(1.0)
    assert "NVIDIA_H100" in measured["basis"]
