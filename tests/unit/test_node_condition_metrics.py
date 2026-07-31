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
from batcher.observe import metrics, node_metrics

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _detached_collector():
    """Detach the metrics collector afterwards, since it is a process-wide bus subscriber.

    A subscriber left attached tells the engine that per-query profiles are being consumed,
    which silently changes behavior for every later test — the same reason the metrics suite
    next door details its own collector.
    """
    import batcher.observe.metrics as m

    attached_before = m._detach is not None
    yield
    # Only undo an attachment *this* test caused, and undo it through `stop_metrics` so the
    # module can attach again. Calling the raw detach handle leaves it set, and
    # `start_metrics` then treats the collector as still attached — silencing every later
    # test, which is a worse leak than the one being avoided.
    if not attached_before:
        m.stop_metrics()


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
    conditions = node_metrics.node_conditions()
    assert set(conditions) == set(node_metrics.NODE_CONDITION_HELP)
    assert all(value == 0 for value in conditions.values())


def test_an_unreadable_host_still_exports_the_section(monkeypatch):
    # A scrape config must not have to be conditional on the hardware it points at.
    def _boom():
        raise RuntimeError("no driver here")

    monkeypatch.setattr("batcher._internal.hardware.fabric.degraded_device_links", _boom)
    conditions = node_metrics.node_conditions()
    assert set(conditions) == set(node_metrics.NODE_CONDITION_HELP)
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
    conditions = node_metrics.node_conditions()
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
    for name in node_metrics.NODE_CONDITION_HELP:
        assert f"# HELP batcher_node_{name} " in text


def test_the_snapshot_carries_the_section(clean_node):
    snapshot = metrics.metrics_snapshot()
    assert "node" in snapshot
    assert set(snapshot["node"]) == set(node_metrics.NODE_CONDITION_HELP)


def test_they_are_gauges_because_a_state_is_not_an_event():
    # "Three devices are on a degraded link" is actionable; the number of times that has
    # been true is not, and a counter would invite the second reading.
    text = metrics.prometheus_text()
    for name in node_metrics.NODE_CONDITION_HELP:
        assert f"# TYPE batcher_node_{name} gauge" in text
        assert f"batcher_node_{name}_total" not in text


# --- Per-device hardware gauges -------------------------------------------------------------


def test_a_device_s_readings_reach_the_scrape(monkeypatch):
    # Distinct from the `gpu.devices` series beside them, which the *engine* reports while an
    # inference stage runs. A node that is merely hot, capped, or idle publishes none of those,
    # and those are exactly the states that leave a job correct and a fraction as fast.
    monkeypatch.setattr(
        node_metrics,
        "device_readings",
        lambda: (("GPU-abc123", 540.0, 700.0, 71.0, 0.87),),
    )
    lines = node_metrics.device_gauges()
    assert 'batcher_device_power_watts{device="GPU-abc123"} 540.0' in lines
    assert 'batcher_device_power_limit_watts{device="GPU-abc123"} 700.0' in lines
    assert 'batcher_device_temperature_celsius{device="GPU-abc123"} 71.0' in lines
    assert 'batcher_device_utilization_ratio{device="GPU-abc123"} 0.87' in lines
    # Every gauge is declared before it is used, or a scrape rejects the whole payload.
    for suffix, _help, _pull in node_metrics._DEVICE_GAUGES:
        assert f"# TYPE batcher_device_{suffix} gauge" in lines


def test_an_idle_device_still_publishes_a_series(monkeypatch):
    # Zero is a real value here, not a missing one: an idle board draws watts and has a
    # temperature, and a series that disappears when it goes quiet breaks every rate and
    # average built on it.
    monkeypatch.setattr(node_metrics, "device_readings", lambda: (("0", 0.0, 0.0, 0.0, 0.0),))
    assert 'batcher_device_utilization_ratio{device="0"} 0.0' in node_metrics.device_gauges()


def test_a_host_with_no_devices_publishes_nothing(monkeypatch):
    monkeypatch.setattr(node_metrics, "device_readings", lambda: ())
    assert node_metrics.device_gauges() == []


def test_the_amd_boards_are_read_when_nvml_reports_nothing(monkeypatch):
    from batcher._internal.hardware.amd.devices import AmdDevice

    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", lambda: ())
    monkeypatch.setattr(
        "batcher._internal.hardware.amd.amd_devices",
        lambda: (
            AmdDevice(
                index=0,
                card="card0",
                unique_id="a1b2",
                power_watts=610.0,
                power_cap_watts=750.0,
                temperature_c=68.0,
                busy_percent=42,
            ),
        ),
    )
    readings = node_metrics.device_readings()
    assert readings == (("a1b2", 610.0, 750.0, 68.0, 0.42),)


def test_an_unreadable_device_tree_never_fails_a_scrape(monkeypatch):
    def boom():
        raise RuntimeError("driver gone")

    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", boom)
    assert node_metrics.device_readings() == ()
    assert node_metrics.device_gauges() == []


def test_the_gauges_appear_in_the_prometheus_payload(monkeypatch):
    monkeypatch.setattr(node_metrics, "device_readings", lambda: (("0", 1.0, 2.0, 3.0, 0.5),))
    monkeypatch.setattr(metrics, "device_gauges", node_metrics.device_gauges)
    text = metrics.prometheus_text()
    assert 'batcher_device_power_watts{device="0"} 1.0' in text
