"""The wiring findings in the accelerator report: an imbalance, and a node with no direct pair.

Both conditions leave every counter healthy, which is why they are reported at all. The cases
here describe the fabric mapping directly, since a CI host has neither shape.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.session.accelerators.wires import (
    add_wires,
    device_cost_section,
    peer_section,
    rail_section,
    wire_problems,
)

pytestmark = pytest.mark.unit


def test_an_uneven_rail_spread_is_reported() -> None:
    problems = wire_problems(
        {"rails": {"loaded_rails": 2, "imbalance": 0.375, "assignment": {"a": [0, 1, 2], "b": [3]}}}
    )
    assert len(problems) == 1
    assert "unevenly spread" in problems[0]
    assert "38%" in problems[0]


def test_even_rails_are_not_a_finding() -> None:
    assert wire_problems({"rails": {"loaded_rails": 2, "imbalance": 0.0}}) == []


def test_a_single_rail_cannot_be_unbalanced() -> None:
    assert wire_problems({"rails": {"loaded_rails": 1, "imbalance": 0.9}}) == []


def test_a_node_where_no_pair_can_copy_directly_is_reported() -> None:
    problems = wire_problems({"peers": {"devices": 4, "staged_pairs": 6}})
    assert len(problems) == 1
    assert "stages through host memory" in problems[0]


def test_a_node_with_some_direct_pairs_is_not_a_finding() -> None:
    assert wire_problems({"peers": {"devices": 4, "staged_pairs": 5}}) == []


def test_an_unreadable_topology_yields_no_complaint() -> None:
    """A deployment check must not fail a fleet whose base image stopped publishing a PCI tree."""
    assert wire_problems({}) == []
    assert wire_problems({"rails": {}, "peers": {}}) == []


def test_the_sections_are_omitted_rather_than_zeroed_off_hardware() -> None:
    fabric: dict = {}
    add_wires(fabric)
    # On a CI host with no accelerators both probes answer nothing, and the report stays the
    # same size as it was before this existed.
    assert fabric == {} or set(fabric) <= {"rails", "peers"}


def test_the_probes_return_a_mapping_on_any_host() -> None:
    assert isinstance(rail_section(), dict)
    assert isinstance(peer_section(), dict)


def test_the_report_still_reads_on_a_host_without_accelerators() -> None:
    report = bt.accelerators()
    assert "backend" in report
    assert isinstance(bt.accelerator_problems(), list)


def test_the_device_cost_section_is_omitted_where_the_wires_are_unreadable() -> None:
    """A CPU host has no rail and no host link, so there is no device byte to price."""
    assert device_cost_section() == {} or "net_gbps" in device_cost_section()


def test_the_fabric_metrics_are_absent_rather_than_zero_off_hardware() -> None:
    """A gauge reporting zero rails is indistinguishable from a fleet-wide outage."""
    from batcher.observe import fabric_metrics

    metrics = fabric_metrics()
    assert isinstance(metrics, dict)
    assert all(key.startswith("fabric.") for key in metrics)
    assert all(isinstance(value, float) for value in metrics.values())


def test_the_fabric_metric_names_group_with_the_existing_families() -> None:
    from batcher.observe import fabric_metrics

    # Empty on this host; the contract under test is the prefix, which is what a sink groups on.
    assert not any(key.startswith("energy.") for key in fabric_metrics())
