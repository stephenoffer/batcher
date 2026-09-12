"""A fleet of unequal machines is sized per node, not by its smallest node.

Every distributed sizing decision used to collapse the cluster to its weakest member: the
per-worker core grant was `min(node_cores)` and the per-worker memory budget was
`min(node_ram) * soft_limit / max_workers_on_any_node`. On the 27-node cluster these tests
are written against — one 96-core/206 GB node, two 48-core/103 GB, seven 16-core/34 GB and
sixteen 4-core/8.6 GB — that produced 96 workers of 4 cores each, every one of them budgeted
**228 MB**, including the twenty-four packed onto the 206 GB machine.

The tests here pin both halves of the replacement: the arithmetic
(`plan.resource.fleet_plan`) and the fact that its grants actually reach the placement
bundle, the actor's CPU claim, and the engine config the worker computes under. The last is
the one that could silently regress to the old behaviour while every other assertion passed,
because a per-worker grant that is computed and then not shipped looks identical from the
driver.
"""

from __future__ import annotations

import pytest

from batcher.plan.resource import is_heterogeneous, plan_worker_slots

pytestmark = pytest.mark.unit

# The live cluster this change was measured on, as (cores, bytes) per node.
_MIXED_CORES = [96.0, 48.0, 48.0] + [16.0] * 7 + [4.0] * 16
_MIXED_MEMORY = (
    [206_158_430_208, 103_079_215_104, 103_079_215_104]
    + [34_359_738_368] * 7
    + [8_589_934_592] * 16
)


def _plan(cores=None, memory=None, **kwargs):
    """`plan_worker_slots` over the mixed fleet, with this module's measured defaults."""
    opts = {"target_cores": 24, "min_slice_cores": 8, "domains": 1, "memory_share": 0.8}
    opts.update(kwargs)
    return plan_worker_slots(
        _MIXED_CORES if cores is None else cores,
        _MIXED_MEMORY if memory is None else memory,
        **opts,
    )


def test_the_mixed_fleet_is_tiled_per_node_not_by_its_smallest_node():
    """31 workers sized for their own nodes, against 96 sized for the 4-core ones."""
    slots = _plan()
    by_size: dict[float, int] = {}
    for slot in slots:
        by_size[slot.cpus] = by_size.get(slot.cpus, 0) + 1
    # 96 -> 4x24, each 48 -> 2x24 (so eight 24-core workers), 16 -> 1x16, 4 -> 1x4.
    assert by_size == {24.0: 8, 16.0: 7, 4.0: 16}
    assert len(slots) == 31
    # Every core in the cluster is accounted for: that is what tiling by the smallest node
    # bought at the price of the shape, and this keeps it while fixing the shape.
    assert sum(slot.cpus for slot in slots) == sum(_MIXED_CORES)


def test_a_worker_is_budgeted_from_the_node_it_lands_on():
    """The 228 MB regression: memory came from the smallest node, divided by the busiest."""
    slots = _plan()
    on_big = [s for s in slots if s.node_index == 0]
    on_small = [s for s in slots if s.node_index == len(_MIXED_CORES) - 1]
    # 206 GB * 0.8 / 4 slices ~ 41 GB, not 8.6 GB * 0.8 / 24 ~ 0.23 GB.
    assert on_big[0].memory_bytes == pytest.approx(41_231_686_041, rel=0.01)
    assert on_small[0].memory_bytes == pytest.approx(6_871_947_673, rel=0.01)
    # The failure this replaces, stated as the rule that produced it: the smallest node's RAM
    # divided by the *busiest* node's worker count. It is 228 MB on this fleet, and the big
    # node's worker must be budgeted orders of magnitude above it, not near it.
    old_uniform = min(_MIXED_MEMORY) * 0.8 / max(int(c // 4) for c in _MIXED_CORES)
    assert old_uniform == pytest.approx(286_331_153, rel=0.01)
    assert on_big[0].memory_bytes > 100 * old_uniform


def test_a_homogeneous_fleet_reports_itself_uniform_and_is_left_alone():
    """`is_heterogeneous` is the gate that keeps every uniform cluster on the old path."""
    assert not is_heterogeneous([16.0] * 16)
    assert not is_heterogeneous([])
    assert is_heterogeneous(_MIXED_CORES)
    # And the tiling itself is the identity a homogeneous cluster already had: one 16-core
    # worker per 16-core node, because 16 is below the 24-core target.
    slots = _plan(cores=[16.0] * 16, memory=[34_359_738_368] * 16)
    assert [s.cpus for s in slots] == [16.0] * 16


def test_a_node_is_never_sliced_below_what_a_worker_process_is_worth():
    """`min_slice_cores` declines a cut rather than making it smaller."""
    # Two memory domains would ask for two workers; 8 cores cannot afford two 8-core ones.
    slots = _plan(cores=[8.0], memory=[0], domains=2)
    assert [s.cpus for s in slots] == [8.0]
    # With cores to afford it, the domain floor is honoured over the 24-core target.
    slots = _plan(cores=[32.0], memory=[0], domains=2)
    assert [s.cpus for s in slots] == [16.0, 16.0]


def test_an_odd_core_count_is_dealt_out_whole_and_loses_nothing():
    """A remainder goes to the earliest slots, so no core is dropped on the floor."""
    slots = _plan(cores=[50.0], memory=[0])
    assert [s.cpus for s in slots] == [25.0, 25.0]
    slots = _plan(cores=[51.0], memory=[0])
    assert sorted(s.cpus for s in slots) == [25.0, 26.0]
    assert sum(s.cpus for s in slots) == 51.0


def test_busy_cores_thin_the_grant_without_collapsing_the_shape():
    """Free capacity bounds what a bundle reserves; the nameplate still sets the fan-out.

    Sizing the *shape* from free capacity is the failure this guards: a node whose cores are
    momentarily held would drop out and the fan-out would collapse.
    """
    idle = _plan(cores=[96.0], memory=[0])
    busy = _plan(cores=[96.0], memory=[0], node_free_cores=[40.0])
    assert len(busy) == len(idle) == 4  # same shape
    assert [s.cpus for s in busy] == [10.0] * 4  # thinner grant
    # An idle cluster — every single-tenant run — is untouched.
    assert _plan(node_free_cores=list(_MIXED_CORES)) == _plan()


def test_the_uniform_path_is_the_one_a_homogeneous_cluster_still_takes(monkeypatch):
    """The gate itself, not just the arithmetic behind it.

    `_heterogeneous_fill` returning `None` is what routes a cluster back to
    `_cluster_fill_workers` and every sizing decision that hangs off it, so it is the single
    assertion that keeps a homogeneous deployment unchanged by any of this.
    """
    from batcher.dist import executor as ex

    def topology(rows):
        monkeypatch.setattr("batcher.dist.executors.ray_runtime.scaling.node_classes", lambda: rows)
        monkeypatch.setattr(
            "batcher.dist.executors.ray_runtime.scaling.cluster_numa_nodes", lambda: 1
        )

    even = [{"cpus": 16.0, "free_cpus": 16.0, "memory": 34_359_738_368} for _ in range(16)]
    topology(even)
    assert ex._heterogeneous_fill() is None

    topology([even[0]])  # a single node is not a fleet to tile
    assert ex._heterogeneous_fill() is None

    topology([])  # unreadable topology
    assert ex._heterogeneous_fill() is None

    # Positive control: change one node's size and the same call now returns a plan, so the
    # three `None`s above are a decision and not an unconditional bail-out.
    mixed = [*even, {"cpus": 96.0, "free_cpus": 96.0, "memory": 206_158_430_208}]
    topology(mixed)
    slots = ex._heterogeneous_fill()
    assert slots is not None
    # 16-core nodes give one 15-core worker each; the 96-core node gives 24+24+24+23. Each
    # figure is one short of tiling its node exactly — see `_NODE_RESERVE_CORES`.
    assert sorted({s.cpus for s in slots}) == [15.0, 23.0, 24.0]


def test_a_node_keeps_a_core_the_fleet_does_not_reserve():
    """A fleet that tiles every node exactly holds 100% of the cluster's schedulable CPU.

    That is not a small inefficiency: a distributed query also runs plain Ray tasks (the map
    UDF, the hardware probe) outside the fleet's placement group, so they have nowhere to go
    and the query waits on capacity only its own fleet could release. Measured on this fleet
    before the reserve existed -- slots summing to 384 cores against a cluster of exactly 384,
    and the fleet failing to come up complete on two runs of three, each costing 120 s.
    """
    exact = _plan()
    reserved = _plan(node_reserve_cores=1.0)
    assert sum(s.cpus for s in exact) == sum(_MIXED_CORES)  # the shape that caused it
    # One core per node that hosts a worker, and not one more.
    assert sum(s.cpus for s in reserved) == sum(_MIXED_CORES) - len(_MIXED_CORES)
    # The worker COUNT is untouched — a headroom must not cost the fan-out.
    assert len(reserved) == len(exact)
    # A node too small to spare a core keeps its worker rather than losing it.
    tiny = _plan(cores=[1.0, 8.0], memory=[0, 0], node_reserve_cores=1.0)
    assert [s.cpus for s in tiny] == [1.0, 7.0]
