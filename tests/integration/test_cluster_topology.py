"""Cluster-aware transport resolution: Flight on a real cluster, disk locally.

The disk shuffle writes to a driver-local ``work_dir`` that worker nodes can't
reach, so on a genuine multi-node cluster the data plane MUST go through Carbonite
/ Arrow Flight. ``transport="auto"`` (the surface default) makes that choice from
the live cluster topology. These tests pin that resolution and its overrides.
"""

from __future__ import annotations

import pytest

from batcher.config import Config, DistributedConfig, config_context

ray = pytest.importorskip("ray", reason="ray not installed")

from _ray_cluster import init_test_ray, shutdown_test_ray  # noqa: E402  (after importorskip)
from batcher import dist  # noqa: E402  (after importorskip)
from batcher.dist.executors.ray_runtime import scaling  # noqa: E402  (after importorskip)


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(2)
    yield
    shutdown_test_ray(started)


@pytest.fixture
def _fake_nodes(monkeypatch):
    """Patch ``ray.nodes`` to report a chosen number of alive worker nodes.

    Each record carries a ``Resources`` dict, because the topology readers size the fan-out
    from **per-node** resources (head excluded, drain excluded) rather than from
    ``ray.cluster_resources()`` totals. A bare ``{"Alive": True}`` therefore reads as a node
    with zero CPUs, and `clamp_workers` takes its documented "Ray reports no CPUs" no-op
    branch instead of the clamp the test means to exercise.

    Each call also drops the `_LIVE_TTL_S` topology window. That window reuses `ray.nodes()`
    for 50 ms to keep a distributed query from making O(nodes) GCS round trips, so two
    different cluster shapes stubbed inside one test are otherwise both answered from the
    first snapshot — a test that reads process state, and one that passes or fails depending
    on how fast the box is.
    """

    def _set(n: int, cpus_per_node: float = 1.0) -> None:
        nodes = [
            {"Alive": True, "Resources": {"CPU": cpus_per_node, "memory": float(2**30)}}
            for _ in range(n)
        ]
        monkeypatch.setattr(ray, "nodes", lambda: nodes)
        scaling._reset_topology_cache()

    return _set


def test_explicit_transport_passes_through(_fake_nodes):
    """An explicit choice is never overridden by topology."""
    _fake_nodes(5)
    assert dist.resolve_transport("disk", 2) == "disk"
    assert dist.resolve_transport("flight", 2) == "flight"


def test_single_node_resolves_to_disk(_fake_nodes):
    _fake_nodes(1)
    assert dist.resolve_transport("auto", 2) == "disk"


def test_multi_node_resolves_to_flight(_fake_nodes):
    """The key cluster-correctness property: multi-node defaults to Flight, never
    the driver-local disk shuffle."""
    _fake_nodes(3)
    assert dist.resolve_transport("auto", 2) == "flight"


def test_shared_filesystem_keeps_disk_on_multi_node(_fake_nodes):
    """A configured shared filesystem makes the disk shuffle cluster-safe, so
    ``auto`` may keep disk even multi-node (and never needs to probe topology)."""
    _fake_nodes(4)
    cfg = Config().replace(distributed=DistributedConfig(shared_filesystem=True))
    with config_context(cfg):
        assert dist.resolve_transport("auto", 2) == "disk"


def test_cluster_topology_counts_alive_nodes(monkeypatch):
    """Dead nodes are not counted, and CPUs are summed from the nodes that survive."""
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [
            {"Alive": True, "Resources": {"CPU": 4.0}},
            {"Alive": False, "Resources": {"CPU": 4.0}},
            {"Alive": True, "Resources": {"CPU": 4.0}},
        ],
    )
    scaling._reset_topology_cache()
    topo = dist.cluster_topology()
    assert topo["nodes"] == 2
    assert topo["cpus"] == 8.0  # the two alive nodes only; the dead one contributes nothing


def test_clamp_workers_to_available_cpus(monkeypatch):
    """Worker fan-out is clamped to schedulable CPUs so the gang-scheduling placement
    group can't over-subscribe the cluster (the autoscaler hint is best-effort)."""
    from batcher.dist.executors import ray_runtime

    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True, "Resources": {"CPU": 2.0}}])
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 2.0})
    scaling._reset_topology_cache()
    assert ray_runtime.clamp_workers(8) == 2  # clamped down to available
    assert ray_runtime.clamp_workers(1) == 1  # already fits


def test_clamp_workers_waits_for_autoscale_then_uses_new_nodes(monkeypatch):
    """On an autoscaling cluster (`autoscale_wait_s > 0`), a big job waits for the
    nodes it asked for to arrive and then runs at the larger fan-out — instead of
    clamping to the pre-scale size and leaving the new capacity for the next job."""
    from batcher.dist.executors import ray_runtime

    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True}])
    # The cluster grows 2 → 4 → 8 CPUs over successive polls (autoscaler provisioning).
    cpus = iter([2.0, 2.0, 4.0, 8.0])
    last = [8.0]

    def _resources():
        last[0] = next(cpus, last[0])  # grow over polls, then hold at the last value
        return {"CPU": last[0]}

    monkeypatch.setattr(ray, "cluster_resources", _resources)
    cfg = Config().replace(
        distributed=DistributedConfig(autoscale_wait_s=5.0, autoscale_poll_s=0.01)
    )
    with config_context(cfg):
        assert ray_runtime.clamp_workers(8) == 8  # waited for the scale-up, then used it


def test_autoscale_request_high_water_then_reclaims(monkeypatch):
    """The autoscaler floor is the high-water mark over in-flight query scopes, and
    resets to 0 only when the LAST scope ends — so concurrent queries compose (a scope
    never lowers a sibling's floor) and a finished big query stops pinning the cluster
    scaled up (the autoscaler can reclaim the idle nodes)."""
    from batcher.dist.executors.ray_runtime import autoscale_request

    # Patch the module that *defines* the floor lifecycle, so its same-module calls to
    # `_apply_autoscale_floor` are intercepted. That module is `autoscale_request`, not
    # `scaling` — patching the latter silently stopped applying when the lifecycle moved,
    # and a monkeypatch that no longer binds is a test that passes while testing nothing.
    # `_apply_autoscale_floor` takes (cpus, gpus, resources); capture the CPU floor (the
    # GPU floor and the custom resources stay empty for these CPU-only scopes).
    applied: list[int] = []

    def _capture(cpus, gpus=0, resources=()):
        applied.append(cpus)

    monkeypatch.setattr(autoscale_request, "_apply_autoscale_floor", _capture)
    monkeypatch.setattr(autoscale_request, "_autoscale_active", 0)
    monkeypatch.setattr(autoscale_request, "_autoscale_floor", 0)
    monkeypatch.setattr(autoscale_request, "_autoscale_gpu_floor", 0)

    autoscale_request.request_autoscale(100)  # scope A: scale to 100
    autoscale_request.request_autoscale(50)  # scope B: max(100, 50) → still 100
    assert applied == [100, 100]
    # A ends, B still in flight → floor unchanged (no reclaim)
    autoscale_request.release_autoscale()
    assert applied == [100, 100]
    # last scope ends → reset to 0 so idle nodes are reclaimed
    autoscale_request.release_autoscale()
    assert applied == [100, 100, 0]


def test_clamp_workers_autoscale_times_out_and_clamps(monkeypatch):
    """If the requested nodes never arrive (a fixed cluster, or the autoscaler is at
    its max), the bounded wait elapses and the job clamps to what is actually there —
    it never hangs on a scale-up that cannot happen."""
    from batcher.dist.executors import ray_runtime

    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True, "Resources": {"CPU": 2.0}}])
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 2.0})  # never grows
    scaling._reset_topology_cache()
    cfg = Config().replace(
        distributed=DistributedConfig(autoscale_wait_s=0.05, autoscale_poll_s=0.01)
    )
    with config_context(cfg):
        assert ray_runtime.clamp_workers(8) == 2  # waited briefly, then clamped to 2


def test_auto_distribute_resolution(monkeypatch):
    """distributed="auto" uses the cluster only when connected to a multi-node one;
    explicit True/False always win."""
    from batcher.api.terminal import _resolve_distributed

    assert _resolve_distributed(True) is True
    assert _resolve_distributed(False) is False
    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True}, {"Alive": True}])
    scaling._reset_topology_cache()
    assert _resolve_distributed("auto") is True
    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True}])
    # Drop the 50 ms topology window: without this the single-node stub below is answered
    # from the two-node snapshot above, and the assertion passes only on a slow machine.
    scaling._reset_topology_cache()
    assert _resolve_distributed("auto") is False
