"""Fabric-aware placement: keep a collective inside one NVLink domain, and say so when it can't.

These are pure topology decisions, so they run without Ray: records are constructed directly,
which is also how a caller receives them. The properties that matter are that a collective
fitting one domain is strict-packed onto that node, that one exceeding every domain is reported
as spanning the fabric rather than silently placed, and that an unreadable or unlabelled fleet
produces no hint at all instead of a confident wrong one.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.fabric import (
    GpuNodeTopology,
    devices_within_power_budget,
    domain_groups,
    fits_one_domain,
    interconnect_class,
    largest_local_domain,
    nvlink_domain_size,
    plan_collective,
    power_zone_load,
    rank_nodes_by_efficiency,
    topology_summary,
)

pytestmark = pytest.mark.unit


def _node(node_id: str, gpus: int = 8, model: str = "NVIDIA_H100", **kw) -> GpuNodeTopology:
    return GpuNodeTopology(node_id=node_id, gpus=gpus, accelerator_type=model, **kw)


def test_domain_is_capped_by_the_devices_a_node_actually_has() -> None:
    assert nvlink_domain_size("NVIDIA_H100", 8) == 8
    assert nvlink_domain_size("NVIDIA_H100", 2) == 2, "a 2-GPU node has a domain of 2"
    assert nvlink_domain_size("NVIDIA_L40S", 8) == 1, "PCIe-only parts have no coherent fabric"


def test_unknown_device_keeps_the_pre_existing_node_local_assumption() -> None:
    assert nvlink_domain_size("MADE_UP", 4) == 4
    assert nvlink_domain_size(None, 4) == 4


def test_a_collective_that_fits_one_domain_is_strict_packed() -> None:
    plan = plan_collective(8, (_node("a"), _node("b")))
    assert plan.strategy == "STRICT_PACK"
    assert not plan.spans_fabric
    assert plan.bundles == ({"GPU": 8.0, "CPU": 8.0},)
    assert plan.node_ids == ("a",)


def test_a_collective_wider_than_any_domain_is_reported_not_hidden() -> None:
    plan = plan_collective(16, (_node("a"), _node("b")))
    assert plan.spans_fabric
    assert plan.strategy == "PACK"
    assert len(plan.bundles) == 2
    assert "leaves the fast path" in plan.reason


def test_a_fleet_too_small_says_so() -> None:
    plan = plan_collective(32, (_node("a"), _node("b")))
    assert "will wait on capacity" in plan.reason
    assert plan.spans_fabric


def test_unreadable_topology_produces_no_placement_hint() -> None:
    plan = plan_collective(8, ())
    assert plan.bundles == ()
    assert plan.strategy == "PACK", "the pre-existing default, unchanged"
    assert "unreadable" in plan.reason


def test_bundles_prefer_the_largest_fabric_group() -> None:
    fleet = (
        _node("small", gpus=2, rack="r2"),
        _node("big-a", gpus=8, rack="r1", fabric="ib0"),
        _node("big-b", gpus=8, rack="r1", fabric="ib0"),
    )
    plan = plan_collective(16, fleet)
    assert set(plan.node_ids) == {"big-a", "big-b"}, "one rack and one fabric before crossing"


def test_domain_groups_are_ordered_by_capacity() -> None:
    fleet = (
        _node("x", gpus=2, rack="r2", fabric="ib1"),
        _node("y", gpus=8, rack="r1", fabric="ib0"),
        _node("z", gpus=8, rack="r1", fabric="ib0"),
    )
    keys = list(domain_groups(fleet))
    assert keys[0] == "r1/ib0"
    assert len(domain_groups(fleet)["r1/ib0"]) == 2


def test_unlabelled_nodes_group_alone_rather_than_together() -> None:
    fleet = (_node("a"), _node("b"))
    assert set(domain_groups(fleet)) == {"a", "b"}


def test_interconnect_class_ranks_by_locality() -> None:
    a = _node("a", rack="r1", fabric="ib0", zone="eu-north-1a")
    b = _node("b", rack="r1", fabric="ib0", zone="eu-north-1a")
    c = _node("c", rack="r2", fabric="ib1", zone="eu-north-1a")
    d = _node("d", rack="r9", fabric="ib9", zone="eu-north-1b")
    assert interconnect_class(a, a) == "nvlink"
    assert interconnect_class(a, b) == "intra-rack"
    assert interconnect_class(a, c) == "intra-zone"
    assert interconnect_class(a, d) == "cross-zone"


def test_a_pcie_only_node_is_not_reported_as_nvlink() -> None:
    node = _node("a", gpus=4, model="NVIDIA_L40S")
    assert interconnect_class(node, node) == "intra-node"


def test_fits_one_domain_is_conservative_without_topology() -> None:
    assert fits_one_domain(1, ())
    assert not fits_one_domain(4, ()), "unknown fleet plans for the slower path, never the faster"
    assert fits_one_domain(4, (_node("a"),))
    assert largest_local_domain(()) == 0


def test_power_zone_load_sums_whole_servers_per_zone() -> None:
    fleet = (
        _node("a", power_zone="busway-1"),
        _node("b", power_zone="busway-1"),
        _node("c", power_zone="busway-2"),
    )
    load = power_zone_load(fleet)
    assert load["busway-1"] == pytest.approx(2 * 8 * 875.0)  # 700 W + 25% host share
    assert load["busway-2"] == pytest.approx(8 * 875.0)


def test_unattributed_nodes_are_not_folded_into_a_labelled_zone() -> None:
    load = power_zone_load((_node("a", power_zone="busway-1"), _node("b")))
    assert set(load) == {"busway-1", ""}


def test_power_budget_bounds_what_a_zone_can_still_take() -> None:
    assert devices_within_power_budget(10_000.0, "NVIDIA_H100", already_drawn_watts=7_000.0) == 3
    assert devices_within_power_budget(0.0, "NVIDIA_H100") == -1, "unbudgeted is not zero room"
    assert devices_within_power_budget(10_000.0, "MADE_UP") == -1


def test_efficiency_ranking_of_nodes_stays_total() -> None:
    fleet = (
        _node("v100", model="NVIDIA_TESLA_V100"),
        _node("unknown", model="MADE_UP"),
        _node("h100", model="NVIDIA_H100"),
    )
    ranked = [n.node_id for n in rank_nodes_by_efficiency(fleet)]
    assert ranked[0] == "h100"
    assert set(ranked) == {"h100", "v100", "unknown"}, "every node stays a placement candidate"


def test_summary_reports_zero_labels_rather_than_pretending_one_rack() -> None:
    summary = topology_summary((_node("a"), _node("b", model="NVIDIA_A100_80G")))
    assert summary["gpus"] == 16
    assert summary["gpu_nodes"] == 2
    assert summary["largest_domain"] == 8
    assert summary["device_models"] == ["NVIDIA_A100_80G", "NVIDIA_H100"]
    assert summary["racks"] == 0
    assert summary["power_zones"] == 0
