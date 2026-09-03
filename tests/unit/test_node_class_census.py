"""The node census: one pass over the fleet, two views, and the aggregates weighted.

`scaling.node_classes()` reports one record per node, and twelve call sites read it. All
but one ask an *aggregate* question — how many workers fit, is any node wide enough to
pack onto, which zone is best, what device models are present — and none of those needs to
know which machine is which. On the clusters this engine targets that distinction is the
difference between a driver that scales and one that does not: building one ten-key dict
per node was 250 ms of a single query against a synthetic 100,000-node fleet.

So the extraction now happens once into a class index, and two views hang off it:
`node_class_census()` (one entry per class, with a `count`) and `node_classes()` (the
per-node expansion, for the one caller that needs `node_id`). These tests pin that the two
views agree, that the aggregate consumers weight by `count`, and that a scope pays for the
pass once.

The risk this file exists to catch is specific and silent: **a consumer that reads the
census and ignores `count` under-counts the fleet by the number of nodes per class**, sizes
a fan-out to the number of instance types, and produces a plan that merely runs slowly.
"""

from __future__ import annotations

import sys
from typing import ClassVar

import pytest

from batcher.dist.executors.ray_runtime import capacity, scaling

pytestmark = pytest.mark.unit

_HEAD = "node:__internal_head__"


def _node(node_id, cpus, *, memory=0.0, gpus=0.0, zone="z1", spot=False, head=False):
    resources = {"CPU": cpus, "GPU": gpus, "memory": float(memory)}
    if head:
        resources[_HEAD] = 1.0
    labels = {"ray.io/availability-zone": zone}
    if spot:
        labels["ray.io/market-type"] = "spot"
    return {"NodeID": node_id, "Alive": True, "Resources": resources, "Labels": labels}


@pytest.fixture
def fake_ray(monkeypatch):
    """A stub `ray` whose node list each test sets, with no snapshot in force."""

    class _Ray:
        records: ClassVar[list[dict]] = []

        @classmethod
        def is_initialized(cls):
            return True

        @classmethod
        def nodes(cls):
            return cls.records

        @classmethod
        def cluster_resources(cls):
            return {"CPU": sum(n["Resources"]["CPU"] for n in cls.records)}

    monkeypatch.setitem(sys.modules, "ray", _Ray)
    token = scaling._TOPOLOGY.set(None)
    monkeypatch.setattr(scaling, "draining_node_ids", frozenset)
    # Nameplate everywhere, so a class is not split by a co-tenant's transient reservation.
    monkeypatch.setattr(capacity, "_live_free_cpus_by_node", lambda: None)
    scaling._reset_topology_cache()
    yield _Ray
    scaling._TOPOLOGY.reset(token)
    scaling._reset_topology_cache()


def test_the_census_and_the_per_node_view_describe_the_same_fleet(fake_ray):
    """Counts sum to the node total, and each class's fields are its members' fields."""
    fake_ray.records = [
        _node("a", 64.0, memory=512e9),
        _node("b", 64.0, memory=512e9),
        _node("c", 64.0, memory=512e9),
        _node("d", 16.0, memory=64e9, zone="z2"),
    ]
    census = scaling.node_class_census()
    per_node = scaling.node_classes()

    assert len(per_node) == 4
    assert len(census) == 2, "three identical nodes are one class"
    assert sum(entry["count"] for entry in census) == len(per_node)

    # Every per-node record is reproduced by exactly one class, `node_id` aside.
    for row in per_node:
        matches = [
            entry
            for entry in census
            if all(entry[k] == v for k, v in row.items() if k != "node_id")
        ]
        assert len(matches) == 1, row


def test_placeable_workers_counts_every_node_not_every_class(fake_ray):
    """The under-count a consumer that ignored `count` would produce, pinned as a number.

    Four 64-core nodes host eight 8-core workers each. Reading the census without weighting
    would see two *classes* and answer 16, which is a fan-out a third the size the cluster
    can place — and nothing would fail, the query would just run narrow.
    """
    fake_ray.records = [_node(f"n{i}", 64.0, memory=512e9) for i in range(4)]
    assert len(scaling.node_class_census()) == 1
    assert capacity.placeable_workers(8.0) == 32


def test_a_zone_choice_weights_by_the_nodes_in_it(fake_ray):
    """The bigger zone wins on capacity, which it only does if its class count is counted.

    One 64-core node in `z1` against four in `z2`: unweighted, both zones look like a single
    class and the tie breaks arbitrarily.
    """
    fake_ray.records = [_node("a", 64.0, memory=512e9, zone="z1")] + [
        _node(f"b{i}", 64.0, memory=512e9, zone="z2") for i in range(4)
    ]
    chosen = capacity.preferred_fleet_zone(4, capacity.Demand(num_cpus=8.0))
    assert chosen == {"ray.io/availability-zone": "z2"}


def test_a_scope_extracts_the_fleet_once(fake_ray):
    """The pass is memoized for the length of a `topology_scope()`, and only there.

    Counting reads of the node list rather than timing them: the point is that a phase with
    a dozen callers makes one pass, and that outside a scope the autoscale wait still sees
    the cluster change.
    """
    fake_ray.records = [_node(f"n{i}", 64.0, memory=512e9) for i in range(3)]
    reads = 0
    original = fake_ray.nodes

    def counted():
        nonlocal reads
        reads += 1
        return original()

    fake_ray.nodes = counted
    try:
        with scaling.topology_scope():
            before = reads
            for _ in range(5):
                scaling.node_class_census()
                scaling.node_classes()
                scaling.alive_node_count()
            assert reads == before, "a scope must not re-read the node list"
        scaling._reset_topology_cache()
        outside = reads
        scaling.node_class_census()
        assert reads > outside, "outside a scope the fleet is read live"
    finally:
        fake_ray.nodes = original


def test_only_the_per_node_view_carries_identity(fake_ray):
    """`node_id` is what separates the two views, and the spot set is why it still exists."""
    fake_ray.records = [
        _node("safe", 64.0, memory=512e9),
        _node("cheap", 64.0, memory=512e9, spot=True),
    ]
    assert {row["node_id"] for row in scaling.node_classes()} == {"safe", "cheap"}
    assert all("node_id" not in entry for entry in scaling.node_class_census())
    spot = {row["node_id"] for row in scaling.node_classes() if row["preemptible"]}
    assert spot == {"cheap"}
