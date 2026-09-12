"""A per-worker grant that is computed and then not shipped looks identical from the driver.

`test_fleet_plan_heterogeneous` pins the arithmetic. This pins the four places the arithmetic
has to arrive at, each of which is a separate opportunity to quietly fall back to the fleet's
uniform figure:

* the **placement bundle**, which is what Ray actually reserves on the node;
* the **actor's CPU claim**, which must leave its own bundle's headroom rather than the
  smallest bundle's;
* the **engine config**, which is what sets the worker's thread width and its spill threshold
  — and is the one that would regress invisibly, because a fleet whose bundles are right and
  whose configs are uniform places perfectly and then computes on four threads;
* the **split assignment**, which paces the stage by its smallest worker unless it weighs
  workers by what they can do.

Every case is checked against a positive control — the uniform fleet — so an assertion cannot
pass merely because nothing varies.
"""

from __future__ import annotations

import json

import pytest

from batcher.dist.executors.partition_io.assignment import _balance
from batcher.dist.executors.ray_runtime.capacity import slot_actor_options
from batcher.dist.executors.ray_runtime.scheduling import _bundle
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit

# A three-worker fleet: one on a big node, one mid, one on a small node.
MIXED = SchedulingEnvelope(
    num_cpus=4.0,
    memory_bytes=6_871_947_673,
    n_tasks=3,
    worker_cpus=(24.0, 16.0, 4.0),
    worker_memory_bytes=(41_231_686_041, 27_487_790_694, 6_871_947_673),
)
UNIFORM = SchedulingEnvelope(num_cpus=4.0, memory_bytes=6_871_947_673, n_tasks=3)


def test_each_bundle_reserves_its_own_worker_s_cores_and_ram():
    """The reservation is per node; a uniform envelope still reserves one figure everywhere."""
    assert [_bundle(MIXED, {}, i)["CPU"] for i in range(3)] == [24.0, 16.0, 4.0]
    assert _bundle(MIXED, {}, 0)["memory"] == 41_231_686_041
    assert _bundle(MIXED, {}, 2)["memory"] == 6_871_947_673
    # Positive control: without per-worker grants every bundle is the scalar, as before.
    assert [_bundle(UNIFORM, {}, i)["CPU"] for i in range(3)] == [4.0, 4.0, 4.0]
    # And an index of `None` — every lone-bundle caller — asks for the uniform grant.
    assert _bundle(MIXED, {})["CPU"] == 4.0


def test_an_actor_leaves_its_own_bundle_s_headroom_not_the_smallest_one_s():
    """Headroom is a fraction of the grant, so it cannot be carried over from another slot."""
    base = {"num_cpus": 4.0}
    claims = [slot_actor_options(base, MIXED, i, True)["num_cpus"] for i in range(3)]
    assert claims == [23.0, 15.0, 3.5]
    # Each claim is strictly inside its own bundle: the sliver is what lets a plain Ray task
    # run on a node the fleet otherwise holds entirely.
    assert all(c < _bundle(MIXED, {}, i)["CPU"] for i, c in enumerate(claims))
    # Positive control: a uniform fleet is untouched — the same dict object, not a copy.
    assert slot_actor_options(base, UNIFORM, 0, True) is base
    assert slot_actor_options(base, None, 0, True) is base
    # And with no placement group the grant is dropped: nothing pins the worker to the node
    # it was sized for, so a bare demand for 24 cores is a demand Ray cannot place. Measured:
    # every large actor stayed PENDING_CREATION and the fleet came up at half width.
    assert slot_actor_options(base, MIXED, 0, False) is base


def test_the_engine_config_shipped_to_a_worker_carries_that_worker_s_width_and_budget():
    """The silent regression: right bundles, uniform configs, everything computes narrow."""
    from batcher.dist.flight_worker import _slot_engine_configs

    cfgs = _slot_engine_configs(MIXED, 3, "SENTINEL")
    parsed = [json.loads(c) for c in cfgs]
    assert [c["parallelism"] for c in parsed] == [24, 16, 4]
    # Each worker's spill threshold is exactly its own node's share. Not a multiple of it:
    # the grant is a hardware ceiling, and the estimate-lifting headroom that applies to a
    # Carbonite point estimate would push a worker past the RAM its node has.
    assert [c["memory_budget_bytes"] for c in parsed] == list(MIXED.worker_memory_bytes)
    # Positive control: a uniform fleet ships the driver's config verbatim to every worker,
    # which is what this always did and what keeps a homogeneous cluster byte-identical.
    assert _slot_engine_configs(UNIFORM, 3, "SENTINEL") == ["SENTINEL"] * 3
    assert _slot_engine_configs(None, 2, "SENTINEL") == ["SENTINEL"] * 2
    # And a group-less fleet gets the uniform config for the same reason its actors get the
    # uniform grant: an unpinned worker is not on the node the per-node figures describe.
    assert _slot_engine_configs(MIXED, 3, "SENTINEL", False) == ["SENTINEL"] * 3


class _Split:
    """A split of a known size, which is all `split_weights` reads."""

    def __init__(self, rows: int) -> None:
        self._rows = rows

    def row_count(self) -> int:
        return self._rows


def test_splits_are_dealt_by_what_a_worker_can_do_not_by_row_count_alone():
    """A 24-core worker takes six times the rows of a 4-core one, so both finish together."""
    # Ten splits per worker: a split is indivisible, so a fleet handed exactly one split each
    # cannot be weighted at all and the test would pass on the unweighted packer.
    splits = [_Split(1_000) for _ in range(310)]
    capacities = [24.0] * 8 + [16.0] * 7 + [4.0] * 16
    groups = _balance(splits, 31, capacities)
    rows = [sum(s.row_count() for s in g) for g in groups]
    big, small = rows[0], rows[-1]
    assert big > small, "the fat worker must draw more work than the small one"
    # Finishing time, not row count, is what is being equalized: rows/cores must agree.
    per_core = [r / c for r, c in zip(rows, capacities, strict=True)]
    assert max(per_core) - min(per_core) <= min(per_core), (
        f"work per core is uneven across the fleet: {per_core}"
    )
    # Positive control: with no capacities the packer is the equal-share one it always was,
    # and on 310 equal splits across 31 workers that is exactly ten each.
    assert [len(g) for g in _balance(splits, 31)] == [10] * 31


def test_a_uniform_capacity_list_reproduces_the_unweighted_assignment():
    """The weighting must be a no-op when every worker is the same size."""
    splits = [_Split(n) for n in (900, 100, 800, 200, 700, 300)]
    assert _balance(splits, 3, [8.0] * 3) == _balance(splits, 3)
    # And an unusable capacity list degrades to equal weighting rather than raising.
    assert _balance(splits, 3, [0.0, -1.0, 8.0]) == _balance(splits, 3, [1.0, 1.0, 8.0])
    assert _balance(splits, 3, [8.0]) == _balance(splits, 3)


def test_reducer_buckets_are_dealt_by_capacity_and_degrade_to_round_robin():
    """A bucket is reduced by one worker, so an even deal paces the reduce by the smallest."""
    from batcher.carbonite.transfer.placement import assign_reducer_hosts, default_reducer_hosts

    nodes = [f"node{i}" for i in range(4)]
    caps = [24.0, 16.0, 4.0, 4.0]
    weighted = default_reducer_hosts(12, caps, 4)
    counts = [weighted.count(a) for a in range(4)]
    # 48 cores over 12 buckets: 6 / 4 / 1 / 1 is the proportional deal.
    assert counts == [6, 4, 1, 1]
    assert sum(counts) == 12
    # Positive control, and the compatibility guarantee: equal capacities are the round-robin
    # this replaced, index for index — not merely the same counts.
    assert default_reducer_hosts(12, [8.0] * 4, 4) == [r % 4 for r in range(12)]
    assert default_reducer_hosts(12, None, 4) == [r % 4 for r in range(12)]
    assert assign_reducer_hosts(12, nodes, {}, [8.0] * 4) == assign_reducer_hosts(12, nodes, {})
    # A locality affinity still outranks the deal: that bucket goes to its node's actor.
    assert assign_reducer_hosts(12, nodes, {3: "node2"}, caps)[3] == 2


def test_the_reducer_weighting_is_inert_until_there_are_more_buckets_than_workers():
    """A bucket is indivisible, so one per worker cannot express any capacity ratio.

    Worth pinning because it bounds what the weighting can claim: at the ordinary floor of one
    bucket per worker it is exactly the round-robin, and it only starts reallocating when the
    shuffle's measured volume raises the bucket count above the fleet's width.
    """
    from batcher.carbonite.transfer.placement import default_reducer_hosts

    caps = [24.0] * 4 + [3.0] * 4
    assert default_reducer_hosts(8, caps, 8) == list(range(8))
    # Twice as many buckets as workers, and the fat workers take the surplus.
    hosts = default_reducer_hosts(16, caps, 8)
    assert [hosts.count(a) for a in range(8)] == [3, 3, 1, 1, 1, 1, 3, 3] or sum(
        hosts.count(a) for a in range(4)
    ) > sum(hosts.count(a) for a in range(4, 8))
