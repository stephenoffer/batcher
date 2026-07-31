"""Placing a collective on a fleet whose devices have no vendor fabric between them.

A T4, L4, A10G, L40S or workstation part is PCIe-attached: its coherent domain is one device,
so the "fits one NVLink domain" test can never pass however many devices a node has. That left
a four-way collective on a four-device host going down the spanning path and being reported as
leaving the fast path -- when it had not left the machine at all. The distinction is worth
roughly an order of magnitude: an intra-host exchange runs at host-link rates, a spanning one
at NIC rates.

These tests pin the middle tier that closes that gap, and the `spans_nodes` / `link_class`
fields that let a caller tell the two apart. They construct records directly, so no Ray.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.fabric import GpuNodeTopology, plan_collective

pytestmark = pytest.mark.unit


def _t4(node_id: str, gpus: int = 4, **kw) -> GpuNodeTopology:
    """A node from this repo's own fleet: four PCIe-attached T4s, no NVLink between them."""
    return GpuNodeTopology(node_id=node_id, gpus=gpus, accelerator_type="T4", cpus=48.0, **kw)


def _h100(node_id: str, gpus: int = 8, **kw) -> GpuNodeTopology:
    return GpuNodeTopology(node_id=node_id, gpus=gpus, accelerator_type="H100", **kw)


def test_the_fleet_this_runs_on_has_no_coherent_domain() -> None:
    """The premise: a T4 node's domain is one device however many it holds."""
    assert _t4("a").local_domain == 1


def test_a_collective_that_fits_one_pcie_node_stays_on_it() -> None:
    plan = plan_collective(4, (_t4("a"), _t4("b")))
    assert plan.strategy == "STRICT_PACK", "it fits one host, so pin it to one host"
    assert plan.bundles == ({"GPU": 4.0, "CPU": 4.0},)
    assert plan.node_ids == ("a",)


def test_fitting_one_pcie_node_is_not_reported_as_leaving_the_machine() -> None:
    plan = plan_collective(4, (_t4("a"), _t4("b")))
    assert not plan.spans_nodes, "four devices on one host do not cross the network"
    assert plan.link_class == "intra-node"
    assert plan.spans_fabric, "but they genuinely do not share a vendor fabric"


def test_exceeding_one_pcie_node_does_span_nodes() -> None:
    plan = plan_collective(8, (_t4("a"), _t4("b")))
    assert plan.spans_nodes
    assert plan.strategy == "PACK"
    assert set(plan.node_ids) == {"a", "b"}
    assert plan.link_class in ("intra-rack", "intra-zone", "cross-zone")


def test_the_reason_names_the_host_link_rather_than_the_fast_path() -> None:
    reason = plan_collective(4, (_t4("a"),)).reason
    assert "stays off the network" in reason
    assert "host link" in reason


def test_nvlink_still_wins_over_a_merely_local_node() -> None:
    """An 8-GPU H100 node and an 8-GPU T4 node both fit; the coherent one must be picked."""
    plan = plan_collective(8, (_t4("pcie", gpus=8), _h100("nvlink")))
    assert plan.node_ids == ("nvlink",)
    assert plan.link_class == "nvlink"
    assert not plan.spans_fabric


def test_single_device_needs_no_fabric_at_all() -> None:
    plan = plan_collective(1, (_t4("a"),))
    assert plan.strategy == "STRICT_PACK"
    assert not plan.spans_nodes
    assert plan.link_class == "intra-node"


def test_link_class_reports_the_slowest_hop_when_spanning_zones() -> None:
    plan = plan_collective(8, (_t4("a", zone="us-west-2a"), _t4("b", zone="us-west-2b")))
    assert plan.spans_nodes
    assert plan.link_class == "cross-zone", "the collective runs at its slowest hop"


def test_a_fleet_too_small_still_reports_what_it_could_place() -> None:
    plan = plan_collective(32, (_t4("a"), _t4("b")))
    assert "8 of 32 devices" in plan.reason
    assert plan.spans_nodes


def test_an_unreadable_fleet_still_produces_no_hint() -> None:
    plan = plan_collective(4, ())
    assert plan.bundles == ()
    assert plan.strategy == "PACK"
    assert not plan.spans_nodes
    assert plan.link_class == ""
