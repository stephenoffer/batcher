"""The `planning` section of the accelerator report — what the optimizer made of the fleet.

The rest of that report describes the hardware. This section is the only place an operator can
see what the *optimizer* concluded from it, which is what makes two clusters' timings for the
same query comparable rather than mysterious.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.session.accelerators import planning as planning_mod
from batcher.plan.resource import ClusterShape, HardwareProfile, NodeShape

pytestmark = pytest.mark.unit


def _dense_profile(nodes: int = 4, gpus: int = 8, fabric: float = 25.0) -> HardwareProfile:
    shape = ClusterShape(
        nodes=tuple(
            NodeShape(
                node_id=f"n{i}",
                cpu_cores=96,
                memory_bytes=1 << 40,
                gpus=gpus,
                accelerator_type="NVIDIA_H100",
                gpu_memory_bytes=80 * (1 << 30),
                nvlink_domain=gpus,
                rack="r1",
                fabric_gbps=fabric,
            )
            for i in range(nodes)
        )
    )
    return HardwareProfile.for_cluster(
        cpu_cores=96,
        memory_bytes=1 << 40,
        worker_count=nodes * gpus,
        gpu_count=shape.total_gpus,
        gpu_memory_bytes=shape.binding_gpu_memory_bytes,
        cluster=shape,
    )


def test_omitted_when_there_is_no_readable_cluster(monkeypatch):
    """A single-node run gets no section rather than one restating the flat defaults."""
    monkeypatch.setattr(
        "batcher.api.orchestration.sizing.distributed_hardware", lambda: None, raising=True
    )
    report: dict = {}
    planning_mod.add_planning(report)
    assert "planning" not in report


def test_a_profile_with_no_shape_is_still_omitted(monkeypatch):
    """`worker_count` alone is not a topology, and reporting it as one would mislead."""
    monkeypatch.setattr(
        "batcher.api.orchestration.sizing.distributed_hardware",
        lambda: HardwareProfile(worker_count=32, gpu_count=32),
        raising=True,
    )
    report: dict = {}
    planning_mod.add_planning(report)
    assert "planning" not in report


def test_reports_the_shape_and_what_it_cost(monkeypatch):
    """With a shape, the section names the fleet and the discount it earned a shuffle."""
    monkeypatch.setattr(
        "batcher.api.orchestration.sizing.distributed_hardware",
        lambda: _dense_profile(),
        raising=True,
    )
    report: dict = {}
    planning_mod.add_planning(report)
    planning = report["planning"]
    assert planning["cluster"]["gpus"] == 32
    assert planning["cluster"]["largest_nvlink_domain"] == 8
    assert 0.0 < planning["shuffle_cost_factor"] < 1.0
    assert "Gb/s" in planning["shuffle_basis"]
    assert planning["device_exchange_cost_factor"] <= planning["shuffle_cost_factor"]


def test_one_device_per_host_earns_no_device_discount(monkeypatch):
    """A device fan-out over one-device hosts has no coherent fabric to exploit.

    Its *relational* shuffle still does: those hosts have ninety-six cores each, so a
    thirty-second of a CPU exchange stays on the host it started on. The two factors are
    genuinely different on one fleet, which is why the section reports both.
    """
    monkeypatch.setattr(
        "batcher.api.orchestration.sizing.distributed_hardware",
        lambda: _dense_profile(nodes=32, gpus=1),
        raising=True,
    )
    report: dict = {}
    planning_mod.add_planning(report)
    planning = report["planning"]
    assert planning["device_exchange_cost_factor"] == 1.0
    assert planning["shuffle_cost_factor"] < 1.0


def test_a_fleet_of_thin_hosts_earns_nothing_at_all(monkeypatch):
    """One core per host: there is no on-host tier for anything, so no discount anywhere."""
    shape = ClusterShape(
        nodes=tuple(
            NodeShape(node_id=f"n{i}", cpu_cores=1, gpus=1, fabric_gbps=25.0) for i in range(32)
        )
    )
    profile = HardwareProfile.for_cluster(
        cpu_cores=1, memory_bytes=1 << 34, worker_count=32, gpu_count=32, cluster=shape
    )
    monkeypatch.setattr(
        "batcher.api.orchestration.sizing.distributed_hardware", lambda: profile, raising=True
    )
    report: dict = {}
    planning_mod.add_planning(report)
    assert report["planning"]["shuffle_cost_factor"] == 1.0


def test_the_public_report_still_builds_without_a_cluster():
    """`bt.accelerators()` must keep working on a laptop, section or no section."""
    assert "backend" in bt.accelerators()
