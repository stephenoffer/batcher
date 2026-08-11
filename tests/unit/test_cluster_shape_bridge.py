"""The bridge from live Ray topology into the neutral shape Kyber plans against.

`fabric.shape.cluster_shape()` is the only path by which the optimizer learns that its fleet
has a structure at all: how many nodes, how dense with devices, which racks and zones, how wide
a coherent fabric. Everything downstream of it — the interconnect-tier discount on a shuffled
byte, the exchange width, the binding per-worker core and memory figures — is derived from what
it says.

It had no tests. That matters more here than the line count suggests, because every failure of
this function is silent: it returns a *plausible* shape, the optimizer plans against it without
complaint, and the only symptom is a plan ranked with the wrong network cost.

Two properties are pinned here and neither is cosmetic:

* the shape describes the machines that will run the work, so the Ray head and any draining
  node are excluded — a head is routinely the *smallest* node in a fleet, and the binding
  per-worker figures take the smallest;
* a node's availability zone is read under whichever label its provider actually wrote, since
  only the KubeRay spelling was recognized and every other deployment reported no zone at all.
"""

from __future__ import annotations

import sys
from typing import ClassVar

import pytest

from batcher.dist.executors.ray_runtime import hardware_probe as hp
from batcher.dist.executors.ray_runtime import scaling
from batcher.dist.executors.ray_runtime.fabric import shape as shape_mod

pytestmark = pytest.mark.unit

_HEAD = "node:__internal_head__"


def _node(node_id, cpus, *, memory=0, gpus=0.0, accel=None, labels=None, head=False):
    resources = {"CPU": cpus, "GPU": gpus, "memory": float(memory)}
    if head:
        resources[_HEAD] = 1.0
    tags = dict(labels or {})
    if accel:
        tags["ray.io/accelerator-type"] = accel
    return {"NodeID": node_id, "Alive": True, "Resources": resources, "Labels": tags}


@pytest.fixture
def fake_ray(monkeypatch):
    """Install a stub `ray` whose node list each test sets, with no snapshot in force."""

    class _Ray:
        # Class attributes on purpose: each test assigns `fake_ray.records`, and the stub is
        # installed in `sys.modules` as the module object itself rather than instantiated.
        records: ClassVar[list[dict]] = []
        initialized: ClassVar[bool] = True

        @classmethod
        def is_initialized(cls):
            return cls.initialized

        @classmethod
        def nodes(cls):
            return cls.records

    monkeypatch.setitem(sys.modules, "ray", _Ray)
    token = scaling._TOPOLOGY.set(None)
    monkeypatch.setattr(scaling, "draining_node_ids", frozenset)
    yield _Ray
    scaling._TOPOLOGY.reset(token)


def test_the_head_is_not_part_of_the_fleet_shape(fake_ray):
    """A head node hosts no worker, and it distorts every figure derived from the shape.

    `binding_cpu_cores` takes the *smallest* node and a head is routinely the smallest, so
    including it binds the whole plan's per-worker sizing to a machine that runs none of the
    work. `total_cores` drives `exchange_width`, and a wider exchange keeps less of itself
    local, so head cores also quietly remove the locality discount.
    """
    fake_ray.records = [
        _node("head", 4.0, memory=8 << 30, head=True),
        _node("w1", 64.0, memory=256 << 30),
        _node("w2", 64.0, memory=256 << 30),
    ]
    shape = shape_mod.cluster_shape()
    assert [n.node_id for n in shape.nodes] == ["w1", "w2"]
    assert shape.total_cores == 128
    assert shape.binding_cpu_cores == 64
    assert shape.binding_memory_bytes == 256 << 30


def test_a_draining_node_is_not_part_of_the_fleet_shape(fake_ray, monkeypatch):
    """A draining node is alive, advertises its full resources, and is going away. Sizing a
    plan against capacity already committed to disappearing is what the drain list exists to
    prevent, and the shape is one more place that sizing happens."""
    fake_ray.records = [
        _node("w1", 64.0, memory=256 << 30),
        _node("doomed", 8.0, memory=16 << 30),
    ]
    monkeypatch.setattr(scaling, "draining_node_ids", lambda: frozenset({"doomed"}))
    shape = shape_mod.cluster_shape()
    assert [n.node_id for n in shape.nodes] == ["w1"]


def test_a_head_only_cluster_still_produces_a_shape(fake_ray):
    """Survivors-or-nothing: a single-node run is its head and has to be described."""
    fake_ray.records = [_node("head", 8.0, memory=32 << 30, head=True)]
    assert [n.node_id for n in shape_mod.cluster_shape().nodes] == ["head"]


@pytest.mark.parametrize(
    "label",
    [
        "topology.kubernetes.io/zone",
        "failure-domain.beta.kubernetes.io/zone",
        "ray.io/availability-zone",
    ],
)
def test_the_zone_is_read_under_whichever_label_the_provider_wrote(fake_ray, label):
    """Only the current KubeRay spelling was recognized, which covers a KubeRay cluster and
    nothing else: a cluster launched on EC2 or GCE by the Ray cluster launcher carries the
    cloud's own label, and a pre-1.17 Kubernetes carries the `failure-domain.beta` form. On
    those fleets every node reported no zone, so a multi-AZ exchange was indistinguishable
    from a single-rack one."""
    fake_ray.records = [
        _node("a", 16.0, labels={label: "us-east-1a"}),
        _node("b", 16.0, labels={label: "us-east-1b"}),
    ]
    assert shape_mod.cluster_shape().zones == 2


def test_an_unlabelled_fleet_reports_no_zones(fake_ray):
    fake_ray.records = [_node("a", 16.0), _node("b", 16.0)]
    assert shape_mod.cluster_shape().zones == 0


def test_device_memory_and_fabric_width_come_from_the_model(fake_ray):
    """Ray never reports device memory, so it is recovered from the advertised model — and the
    coherent fabric width is capped at the node's own device count, since a two-device node has
    a domain of two whatever an eight-way part's datasheet says."""
    fake_ray.records = [
        _node("dense", 96.0, gpus=8.0, accel="H100"),
        _node("pair", 32.0, gpus=2.0, accel="H100"),
    ]
    shape = shape_mod.cluster_shape()
    dense, pair = shape.nodes[0], shape.nodes[1]
    assert dense.gpu_memory_bytes == 80 * (1 << 30)
    assert dense.local_domain == 8
    assert pair.local_domain == 2
    assert shape.max_gpus_per_node == 8


def test_an_unrecognized_model_reports_unknown_rather_than_a_guess(fake_ray):
    """An unknown model leaves every VRAM-sized decision on its existing default. A fabricated
    figure would instead size a shard against memory the device may not have.

    The fabric width degrades differently, and deliberately: it falls back to the node's whole
    device count — "no narrower than the node", the assumption in force before the shape
    existed — rather than to a datasheet width nothing matched.
    """
    fake_ray.records = [_node("odd", 32.0, gpus=4.0, accel="SomeFutureGPU")]
    node = shape_mod.cluster_shape().nodes[0]
    assert node.gpu_memory_bytes == 0
    assert node.local_domain == 4


def test_a_node_with_no_schedulable_cores_holds_no_share(fake_ray):
    fake_ray.records = [_node("w1", 16.0), _node("coreless", 0.0)]
    assert [n.node_id for n in shape_mod.cluster_shape().nodes] == ["w1"]


def test_the_shape_is_stable_across_reads_of_an_unchanged_cluster(fake_ray):
    """The shape reaches the plan cache key, so a set-ordered tuple would invalidate every
    memoized plan on a cluster that had not changed at all."""
    fake_ray.records = [_node("c", 16.0), _node("a", 16.0), _node("b", 16.0)]
    first = shape_mod.cluster_shape()
    fake_ray.records = list(reversed(fake_ray.records))
    assert shape_mod.cluster_shape() == first


def test_an_unreadable_cluster_reports_an_empty_shape(fake_ray):
    """Empty makes every locality-aware decision report the flat answer it gave before, so a
    caller never has to test for it."""
    fake_ray.initialized = False
    assert not shape_mod.cluster_shape().known


# --- Devices the fleet has taken out of rotation ----------------------------------------------


def test_quarantined_devices_reach_the_shape(fake_ray, monkeypatch):
    """`unhealthy_gpus` was declared, documented, derived into `healthy_gpus`, summarized, and
    read by `exchange_width` — and never once set. Every health-aware sizing decision in the
    engine was inert, so a device fan-out was sized onto boards the scheduler refuses to place
    on and the placement group pended."""
    fake_ray.records = [
        _node("sick", 96.0, gpus=8.0, accel="H100"),
        _node("well", 96.0, gpus=8.0, accel="H100"),
    ]
    monkeypatch.setattr(
        shape_mod,
        "_node_records",
        lambda: fake_ray.records,
    )
    monkeypatch.setattr(
        hp,
        "sampled_device_health",
        lambda: ({"node_id": "sick", "quarantined": ["GPU-a", "GPU-b"], "degraded": ["GPU-b"]},),
    )
    shape = shape_mod.cluster_shape()
    by_id = {n.node_id: n for n in shape.nodes}
    # Deduplicated: a device is routinely reported as both quarantined and degraded.
    assert by_id["sick"].unhealthy_gpus == 2
    assert by_id["sick"].healthy_gpus == 6
    assert by_id["well"].unhealthy_gpus == 0
    assert shape.total_gpus == 16
    assert shape.healthy_gpus == 14
    assert shape.exchange_width("gpu") == 14


def test_a_stale_health_record_cannot_drive_healthy_devices_negative(fake_ray, monkeypatch):
    """A record naming devices a resized node no longer has must be capped, not trusted."""
    fake_ray.records = [_node("shrunk", 32.0, gpus=2.0, accel="H100")]
    monkeypatch.setattr(shape_mod, "_node_records", lambda: fake_ray.records)
    monkeypatch.setattr(
        hp,
        "sampled_device_health",
        lambda: ({"node_id": "shrunk", "quarantined": [f"GPU-{i}" for i in range(8)]},),
    )
    node = shape_mod.cluster_shape().nodes[0]
    assert node.unhealthy_gpus == 2
    assert node.healthy_gpus == 0


def test_planning_never_triggers_a_fleet_wide_health_probe(fake_ray, monkeypatch):
    """Probing from the planner would put a round trip to every accelerator node on every
    optimize. Absence of a sample reads as "assume healthy", which is what held before."""
    fake_ray.records = [_node("a", 32.0, gpus=4.0, accel="H100")]
    monkeypatch.setattr(shape_mod, "_node_records", lambda: fake_ray.records)

    def explode():  # pragma: no cover - the point is that it is never called
        raise AssertionError("cluster_shape() probed the fleet's device health")

    monkeypatch.setattr(hp, "_probe_fleet_health", explode)
    hp.reset_fleet_health()
    assert shape_mod.cluster_shape().nodes[0].unhealthy_gpus == 0
