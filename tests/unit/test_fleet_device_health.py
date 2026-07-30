"""Finding the sick node on a fleet, which the driver cannot see from where it stands.

NVML answers about the host it runs on, the kernel log about that host's driver, and `/sys`
about that host's wires. On a hundred-node cluster that means "no device is sick" and "no
device the driver can see is sick" are the same sentence, and only one of them is true. These
tests pin the fan-out that closes the gap, and the two directions it must not get wrong: every
accelerator node is probed rather than one per instance shape (a fault belongs to a board, not
to a shape), and a fleet that could not be probed is reported as unknown rather than healthy.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import hardware_probe

pytestmark = pytest.mark.unit


def _record(node: str, **kw) -> dict:
    base = {
        "node_id": node,
        "devices": 8,
        "quarantined": [],
        "degraded": [],
        "reasons": [],
        "reset_pending": [],
        "degraded_links": [],
        "nvlink": {"devices": 8, "links": 144, "active_links": 144, "degraded_devices": 0},
        "xid_readable": True,
    }
    return {**base, **kw}


def test_a_clean_fleet_has_no_unhealthy_nodes():
    records = (_record("a"), _record("b"))
    assert hardware_probe.unhealthy_nodes(records) == ()


@pytest.mark.parametrize(
    "fault",
    [
        {"quarantined": ["GPU-3"]},
        {"degraded": ["GPU-1"]},
        {"reset_pending": ["GPU-0"]},
        {"degraded_links": ["0000:0c:00.0"]},
        {"nvlink": {"devices": 8, "links": 144, "active_links": 100, "degraded_devices": 2}},
    ],
)
def test_every_fault_class_puts_a_node_on_the_drain_list(fault):
    records = (_record("healthy"), _record("sick", **fault))
    assert [r["node_id"] for r in hardware_probe.unhealthy_nodes(records)] == ["sick"]


def test_an_unprobeable_fleet_reports_nothing_rather_than_health(monkeypatch):
    # Empty from both means "we could not ask", which is why the caller checks the probe's own
    # result rather than reading an empty drain list as a clean bill.
    monkeypatch.setattr(hardware_probe, "cluster_device_health", lambda: ())
    assert hardware_probe.unhealthy_nodes() == ()


def test_no_ray_means_no_probe(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ray", None)
    assert hardware_probe.cluster_device_health() == ()


def test_the_worker_side_probe_reports_what_only_that_host_can_see(monkeypatch):
    from batcher.carbonite.accel import HealthVerdict

    monkeypatch.setattr(
        "batcher.carbonite.accel.assess_fleet",
        lambda: (
            HealthVerdict(device_index=0, uuid="GPU-0"),
            HealthVerdict(
                device_index=1, uuid="GPU-1", state="quarantine", reasons=("xid_79",), derate=0.0
            ),
        ),
    )
    monkeypatch.setattr("batcher.carbonite.accel.device_reset_candidates", lambda: ("GPU-0",))
    monkeypatch.setattr("batcher._internal.hardware.fabric.degraded_device_links", lambda: ())
    record = hardware_probe._device_health_on_this_worker()
    assert record["devices"] == 2
    assert record["quarantined"] == ["GPU-1"]
    assert record["reasons"] == ["xid_79"]
    assert record["reset_pending"] == ["GPU-0"]


def test_the_report_carries_the_drain_list(monkeypatch):
    import importlib

    report_mod = importlib.import_module("batcher.api.session.accelerators")
    monkeypatch.setattr(
        hardware_probe,
        "cluster_device_health",
        lambda: (_record("good"), _record("bad", quarantined=["GPU-2"], reasons=["ecc"])),
    )
    fleet: dict = {}
    report_mod._add_fleet_health(fleet)
    assert fleet["health"]["nodes_probed"] == 2
    assert [n["node_id"] for n in fleet["health"]["unhealthy"]] == ["bad"]
    assert fleet["health"]["unhealthy"][0]["reasons"] == ["ecc"]


def test_the_report_omits_health_entirely_off_a_cluster(monkeypatch):
    import importlib

    report_mod = importlib.import_module("batcher.api.session.accelerators")
    monkeypatch.setattr(hardware_probe, "cluster_device_health", lambda: ())
    fleet: dict = {}
    report_mod._add_fleet_health(fleet)
    assert fleet == {}


# --- Placement keeps a collective off a condemned node -------------------------------------


def _topology(*node_ids: str):
    from batcher.dist.executors.ray_runtime.fabric.topology import GpuNodeTopology

    return tuple(
        GpuNodeTopology(node_id=n, gpus=8, accelerator_type="NVIDIA_H100") for n in node_ids
    )


def _health_enabled(enabled: bool = True):
    """Scope `accelerator.health.enabled`, which is what gates the fleet probe."""
    from batcher.config import Config, config_context

    return config_context(Config.from_dict({"accelerator": {"health": {"enabled": enabled}}}))


def test_a_gang_bundle_avoids_a_node_with_a_condemned_device(monkeypatch):
    # A strict-pack collective has no way to route around a bad rank once placed: every rank
    # waits on the slowest, so the whole group runs at the bad device's rate.
    from batcher.dist.executors.ray_runtime.fabric import placement

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: (_record("good"), _record("bad", quarantined=["GPU-2"])),
    )
    with _health_enabled():
        kept = placement._without_unhealthy(_topology("good", "bad"))
    assert [n.node_id for n in kept] == ["good"]


def test_a_degraded_but_working_device_does_not_empty_the_node(monkeypatch):
    # A thermal or power clamp is often every node at once, and it is the clamp doing its job.
    from batcher.dist.executors.ray_runtime.fabric import placement

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: (_record("a", degraded=["GPU-1"]), _record("b", degraded=["GPU-0"])),
    )
    with _health_enabled():
        kept = placement._without_unhealthy(_topology("a", "b"))
    assert [n.node_id for n in kept] == ["a", "b"]


def test_an_unreadable_fleet_is_not_an_unhealthy_one(monkeypatch):
    from batcher.dist.executors.ray_runtime.fabric import placement

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health", lambda: ()
    )
    with _health_enabled():
        kept = placement._without_unhealthy(_topology("a", "b"))
    assert len(kept) == 2


def test_a_wholly_condemned_fleet_still_gets_a_placement(monkeypatch):
    # Refusing to place work at all is worse than placing it on a fleet whose state the
    # operator has already been told about.
    from batcher.dist.executors.ray_runtime.fabric import placement

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: (_record("a", quarantined=["x"]), _record("b", quarantined=["y"])),
    )
    with _health_enabled():
        kept = placement._without_unhealthy(_topology("a", "b"))
    assert len(kept) == 2


def test_the_probe_is_not_run_when_health_checking_is_off(monkeypatch):
    from batcher.dist.executors.ray_runtime.fabric import placement

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: pytest.fail("probed the fleet with health checking disabled"),
    )
    with _health_enabled(enabled=False):
        assert len(placement._without_unhealthy(_topology("a"))) == 1


# --- Sampling, so a scheduling path does not pay a fleet round trip ------------------------


def test_the_probe_is_sampled_rather_than_run_per_caller(monkeypatch):
    # The placement filter asks per placement decision. Unsampled, that puts a task-per-node
    # round trip on a scheduling path.
    calls = []
    monkeypatch.setattr(
        hardware_probe, "_probe_fleet_health", lambda: calls.append(1) or (_record("a"),)
    )
    hardware_probe.reset_fleet_health()
    for _ in range(10):
        assert hardware_probe.cluster_device_health() == (_record("a"),)
    assert len(calls) == 1
    hardware_probe.reset_fleet_health()
    hardware_probe.cluster_device_health()
    assert len(calls) == 2


def test_an_unreadable_fleet_is_not_cached(monkeypatch):
    # A cluster seconds from coming up must not have its unavailability held for half a
    # minute of placement decisions.
    calls = []
    monkeypatch.setattr(hardware_probe, "_probe_fleet_health", lambda: calls.append(1) or ())
    hardware_probe.reset_fleet_health()
    for _ in range(3):
        assert hardware_probe.cluster_device_health() == ()
    assert len(calls) == 3
