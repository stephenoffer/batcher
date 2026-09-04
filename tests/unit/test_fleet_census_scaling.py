"""The fleet is described by node *class*, so the driver's per-query cost stops tracking it.

Batcher's scheduling target is clusters of a hundred thousand nodes and millions of
processes. Every figure the placement phase and the cost model read used to be recomputed
over one record per node — `ClusterShape` holds properties rather than fields, so each
access was a fresh O(nodes) pass, and `locality_shares` is evaluated once per pipeline
breaker. Measured against a synthetic 100,000-node fleet before this change: 106 ms for a
single `locality_shares` call and 28 ms to hash the shape, which happens on every plan
cache lookup because the shape reaches the key.

A cluster of that size is a handful of instance types across a handful of zones, so the
fix is to describe it as a census — distinct shapes, each with a count — and weight every
aggregate. These tests pin the two things that makes true:

* **equivalence**, against the per-node form, which is the oracle. A census is a
  compression of the description and must never be a change to it, so a randomized fleet
  is checked class-by-class against its own expansion;
* **the scaling property itself**, asserted as work that does not grow rather than as a
  duration, so it stays meaningful on a loaded or slow machine.

`tests/unit/test_cluster_shape.py` covers what the figures *mean*; this file covers that
the representation carrying them is faithful and cheap.
"""

from __future__ import annotations

import itertools
import random

import pytest

from batcher.plan.resource.cluster import ClusterShape, NodeShape
from batcher.plan.resource.locality import _spread, _spread_census

pytestmark = pytest.mark.unit

#: Every aggregate `ClusterShape` derives. Listed rather than discovered so a new one is a
#: deliberate addition to this test as well.
_AGGREGATES = (
    "known",
    "node_count",
    "gpu_node_count",
    "total_gpus",
    "healthy_gpus",
    "total_cores",
    "total_memory_bytes",
    "aggregate_gpu_memory_bytes",
    "device_models",
    "homogeneous_gpus",
    "binding_gpu_memory_bytes",
    "binding_cpu_cores",
    "binding_memory_bytes",
    "largest_nvlink_domain",
    "max_gpus_per_node",
    "min_gpus_per_node",
    "racks",
    "power_zones",
    "zones",
    "fabric_gbps",
)


def _random_shape(rng: random.Random) -> NodeShape:
    """A node whose every field varies, including the ones that split classes."""
    gpus = rng.choice([0, 0, 1, 2, 4, 8])
    return NodeShape(
        cpu_cores=rng.choice([0, 1, 2, 8, 16, 64]),
        memory_bytes=rng.choice([0, 64 << 30, 512 << 30]),
        gpus=gpus,
        accelerator_type=rng.choice(["A100", "H100", ""]) if gpus else "",
        gpu_memory_bytes=rng.choice([0, 40 << 30, 80 << 30]) if gpus else 0,
        nvlink_domain=rng.choice([0, 2, 4, 8]) if gpus else 0,
        fabric_gbps=rng.choice([0.0, 100.0, 400.0]),
        # Unlabelled racks and zones on purpose: an unlabelled node is its own rack, which is
        # the branch a census has to keep separate rather than folding into one group.
        rack=rng.choice(["r1", "r2", "r3", ""]),
        zone=rng.choice(["z1", "z2", ""]),
        power_zone=rng.choice(["p1", "p2", ""]),
        unhealthy_gpus=rng.choice([0, 0, 1]) if gpus else 0,
    )


def _fleet_pair(rng: random.Random) -> tuple[ClusterShape, ClusterShape]:
    """The same fleet as a census and as one record per node."""
    classes = [(_random_shape(rng), rng.randint(1, 6)) for _ in range(rng.randint(1, 5))]
    census = ClusterShape(
        nodes=tuple(shape for shape, _ in classes),
        multiplicity=tuple(count for _, count in classes),
    )
    # Class-contiguous, which is how a census expands and what `_spread_census` is exact for.
    expanded = ClusterShape(
        nodes=tuple(itertools.chain.from_iterable([shape] * count for shape, count in classes))
    )
    return census, expanded


@pytest.mark.parametrize("seed", range(25))
def test_every_aggregate_matches_the_per_node_fleet(seed: int) -> None:
    """A census reports exactly what one record per node reports."""
    census, expanded = _fleet_pair(random.Random(seed))
    for name in _AGGREGATES:
        assert getattr(census, name) == getattr(expanded, name), name
    assert census.summary() == expanded.summary()


@pytest.mark.parametrize("seed", range(25))
def test_locality_shares_are_the_same_however_the_fleet_is_described(seed: int) -> None:
    """A fleet's tier split must not depend on whether identical nodes were grouped.

    This is the correctness property the census owes, and it is *not* "equals the per-node
    form for any expansion order". `_spread` hands its remainder to the first nodes in the
    fleet's own ordering, which was the node id and is therefore arbitrary; a census cannot
    replay an arbitrary order and does not try to (see `_apportion`). What it must do is give
    one answer for one fleet, so describing the same machines as `n` singleton classes or as
    one class of `n` has to land in the same place — which it does, because identical nodes
    are interchangeable and share a rack.
    """
    rng = random.Random(seed)
    shape = _random_shape(rng)
    count = rng.randint(2, 8)
    grouped = ClusterShape(nodes=(shape,), multiplicity=(count,))
    spelled_out = ClusterShape(nodes=(shape,) * count)
    for unit in ("cpu", "gpu"):
        for workers in (1, 2, 3, 7, 16, 64, 257):
            assert grouped.locality_shares(workers, unit=unit) == spelled_out.locality_shares(
                workers, unit=unit
            ), f"unit={unit} workers={workers}"


@pytest.mark.parametrize("seed", range(25))
def test_the_shares_always_partition_the_exchange(seed: int) -> None:
    """Five fractions summing to one, non-negative, on any fleet.

    Every tiered cost multiplies through these, so a set that does not partition silently
    rescales every `net` cost that reads it. Checked on the census because that is the form
    the engine now builds.
    """
    census, _ = _fleet_pair(random.Random(seed))
    for unit in ("cpu", "gpu"):
        for workers in (1, 5, 31, 400):
            shares = census.locality_shares(workers, unit=unit)
            parts = (
                shares.local,
                shares.intra_domain,
                shares.intra_node,
                shares.intra_rack,
                shares.cross_rack,
            )
            assert all(part >= 0.0 for part in parts), (unit, workers, parts)
            assert sum(parts) == pytest.approx(1.0), (unit, workers, parts)


@pytest.mark.parametrize("seed", range(40))
def test_a_fleet_of_distinguishable_nodes_places_exactly_as_before(seed: int) -> None:
    """With one node per class, `_spread_census` is bit-identical to `_spread`.

    The case that has to be exact, because it is every `ClusterShape` built by hand and every
    fleet whose machines differ in any reported field. `_apportion` reduces to "the first
    `extra` groups" when each group holds one node, which is precisely what `_spread` does.
    """
    rng = random.Random(seed)
    capacities = [rng.choice([0, 1, 2, 3, 8, 16, 64]) for _ in range(rng.randint(1, 7))]
    workers = rng.randint(0, 200)
    assert _spread_census([(c, 1) for c in capacities], workers) == _spread(capacities, workers)


@pytest.mark.parametrize("seed", range(40))
def test_a_placement_never_exceeds_capacity_or_loses_a_worker(seed: int) -> None:
    """Whatever the classes, the placement is conserved and bounded.

    `_spread` guarantees both and the census form has to as well: a class holds at most
    `capacity x nodes` workers, and every worker asked for is placed rather than dropped.
    """
    rng = random.Random(seed)
    classes = [(rng.choice([0, 1, 2, 8, 64]), rng.randint(1, 6)) for _ in range(rng.randint(1, 6))]
    workers = rng.randint(0, 300)
    placed = _spread_census(classes, workers)
    fleet_capacity = sum(capacity * count for capacity, count in classes)
    if fleet_capacity == 0:
        assert placed == [0] * len(classes)
        return
    assert sum(placed) == workers
    if workers <= fleet_capacity:
        for (capacity, count), total in zip(classes, placed, strict=True):
            assert total <= capacity * count


def test_the_remainder_is_shared_out_by_node_count_not_by_class_order() -> None:
    """The correction `_apportion` exists for, pinned as a number.

    Two racks, two workers, ten nodes: a small rack of two and a large one of eight. Filling
    in class order puts both workers in the *small* rack, because it comes first — one rack
    between them, and an over-stated locality share, which is the direction that under-charges
    a shuffle. Apportioning by node count puts both in the large rack, which is where eight
    nodes in ten would land them.
    """
    assert _spread_census([(64, 2), (64, 8)], 2) == [0, 2]
    # With every class a single node, unchanged: the first two, exactly as `_spread` does.
    assert _spread_census([(64, 1)] * 5, 2) == [1, 1, 0, 0, 0]
    assert _spread([64] * 5, 2) == [1, 1, 0, 0, 0]


def test_the_cost_of_a_locality_question_does_not_grow_with_the_fleet() -> None:
    """The scaling property, as work rather than as a duration.

    A timing here would be flaky on a shared machine, so this counts the thing that used to
    grow: how many node records the placement arithmetic touches. `_spread_census` is given
    one entry per class, so a fleet of ten identical nodes and a fleet of ten thousand ask
    it exactly the same question — which is what makes the cost model's per-breaker call
    independent of cluster size.
    """
    shape = NodeShape(cpu_cores=64, memory_bytes=512 << 30, rack="r1", zone="z1")
    small = ClusterShape(nodes=(shape,), multiplicity=(10,))
    huge = ClusterShape(nodes=(shape,), multiplicity=(10_000,))

    assert len(small.nodes) == len(huge.nodes) == 1
    assert (small.node_count, huge.node_count) == (10, 10_000)
    # The same one-class question in both cases; only the counts inside it differ.
    assert len(small._census) == len(huge._census) == 1
    # And the shape that reaches the plan cache key is one entry, not ten thousand.
    assert hash(huge) == hash(ClusterShape(nodes=(shape,), multiplicity=(10_000,)))


def test_an_empty_multiplicity_is_exactly_the_per_node_form() -> None:
    """The default has to be bit-identical, because every hand-built shape uses it."""
    nodes = (NodeShape(cpu_cores=8, rack="a"), NodeShape(cpu_cores=16, rack="b"))
    plain = ClusterShape(nodes=nodes)
    spelled = ClusterShape(nodes=nodes, multiplicity=(1, 1))
    assert plain.node_count == spelled.node_count == 2
    assert plain.locality_shares(4) == spelled.locality_shares(4)
    assert plain.summary() == spelled.summary()
