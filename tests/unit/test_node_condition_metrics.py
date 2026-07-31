"""The hardware conditions a GPU fleet has to scrape for, because nothing else reports them.

A host link at quarter width, memory repairing itself, an NVLink fabric that dropped, a port
accumulating symbol errors: every one leaves each query correct and a fraction as fast, so
none of them reaches a counter and none appears in a job's own timings. A fleet finds them by
alerting on them or does not find them at all.

The two properties these pin: the gauges reach the exposition an operator has already wired an
alert to, and a host where none of it is readable exports zeros rather than omitting the
section — a scrape config should not have to be conditional on the hardware it points at.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric.pcie import PcieLink
from batcher._internal.hardware.faults.counters import DeviceFaults
from batcher.carbonite.accel import HealthVerdict
from batcher.observe import metrics

pytestmark = pytest.mark.unit


@pytest.fixture
def clean_node(monkeypatch):
    """A host with nothing wrong and everything readable."""
    monkeypatch.setattr("batcher._internal.hardware.fabric.degraded_device_links", lambda: ())
    monkeypatch.setattr("batcher._internal.hardware.faults.faulted_devices", lambda: ())
    monkeypatch.setattr("batcher.carbonite.accel.assess_fleet", lambda: ())
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.nvlink_summary",
        lambda: {
            "devices": 8,
            "links": 144,
            "active_links": 144,
            "degraded_devices": 0,
            "peer_pairs": 28,
            "errors": {},
        },
    )
    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_error_total", lambda: {})
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma_summary",
        lambda: {
            "ports": 8,
            "active_ports": 8,
            "bandwidth_gbps": 3200.0,
            "link_layers": {},
            "rdma_available": True,
            "partition": "",
            "devices": [],
            "numa_nodes": [],
        },
    )


def test_a_healthy_node_reports_zeros_rather_than_nothing(clean_node):
    conditions = metrics._node_conditions()
    assert set(conditions) == set(metrics._NODE_CONDITION_HELP)
    assert all(value == 0 for value in conditions.values())


def test_an_unreadable_host_still_exports_the_section(monkeypatch):
    # A scrape config must not have to be conditional on the hardware it points at.
    def _boom():
        raise RuntimeError("no driver here")

    monkeypatch.setattr("batcher._internal.hardware.fabric.degraded_device_links", _boom)
    conditions = metrics._node_conditions()
    assert set(conditions) == set(metrics._NODE_CONDITION_HELP)
    assert all(value == 0 for value in conditions.values())


def test_each_condition_is_counted_from_the_probe_that_owns_it(monkeypatch, clean_node):
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.degraded_device_links",
        lambda: (PcieLink("0000:0c:00.0", gen=3, width=8, max_gen=5, max_width=16),),
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.faults.faulted_devices",
        lambda: (DeviceFaults(index=0, remap_pending=True, readable=True),),
    )
    monkeypatch.setattr(
        "batcher.carbonite.accel.assess_fleet",
        lambda: (
            HealthVerdict(device_index=0),
            HealthVerdict(device_index=1, state="quarantine", derate=0.0),
        ),
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.nvlink_summary",
        lambda: {
            "devices": 8,
            "links": 144,
            "active_links": 100,
            "degraded_devices": 3,
            "peer_pairs": 10,
            "errors": {},
        },
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.fabric_error_total",
        lambda: {"symbol_errors": 12, "link_downed": 1},
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma_summary",
        lambda: {
            "ports": 8,
            "active_ports": 6,
            "bandwidth_gbps": 2400.0,
            "link_layers": {},
            "rdma_available": True,
            "partition": "",
            "devices": [],
            "numa_nodes": [],
        },
    )
    conditions = metrics._node_conditions()
    assert conditions == {
        "degraded_links": 1,
        "faulted_devices": 1,
        "nvlink_down_devices": 3,
        "fabric_errors": 13,
        "fabric_ports_down": 2,
    }


def test_the_gauges_reach_the_prometheus_exposition(monkeypatch, clean_node):
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.degraded_device_links",
        lambda: (PcieLink("0000:0c:00.0", gen=3, width=8, max_gen=5, max_width=16),),
    )
    text = metrics.prometheus_text()
    assert "batcher_node_degraded_links 1" in text
    assert "# TYPE batcher_node_degraded_links gauge" in text
    # Every condition carries help text: a bare series name in a dashboard is a series
    # nobody knows how to act on.
    for name in metrics._NODE_CONDITION_HELP:
        assert f"# HELP batcher_node_{name} " in text


def test_the_snapshot_carries_the_section(clean_node):
    snapshot = metrics.metrics_snapshot()
    assert "node" in snapshot
    assert set(snapshot["node"]) == set(metrics._NODE_CONDITION_HELP)


def test_they_are_gauges_because_a_state_is_not_an_event():
    # "Three devices are on a degraded link" is actionable; the number of times that has
    # been true is not, and a counter would invite the second reading.
    text = metrics.prometheus_text()
    for name in metrics._NODE_CONDITION_HELP:
        assert f"# TYPE batcher_node_{name} gauge" in text
        assert f"batcher_node_{name}_total" not in text
