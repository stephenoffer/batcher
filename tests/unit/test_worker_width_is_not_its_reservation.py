"""A scheduling reserve must cost the fleet a slot, never a worker thread.

Both fan-out fills hold a core back on every node so the query's own plain Ray tasks have
somewhere to run -- `dist.executor._headroom_grant` on the uniform path,
`plan.resource.fleet_plan`'s `node_reserve_cores` on the per-node one. That reserve is a
*scheduling* concession and is the right one. But the same figure also sized each worker's
rayon pool, because `engine_config_json` read the reservation, so the headroom was silently
taking a thread as well as a slot.

The cost is `1/node_cores`, which is why it went unnoticed: a percent on the 96-core node the
mechanism was written against, and **a quarter of the machine** on the 4-core instances a wide
fleet is actually built from. Measured on a 100 x 4-core cluster at TPC-H sf100, every worker
ran three threads on a four-core node and the cluster could not exceed 75% however much work
it was given.

Each test here has its positive control beside it, because the failure mode of a "fix" to this
is to stop reserving at all -- which restores the placement deadlock the reserve exists to
prevent, and would pass any test that only checked the width.
"""

from __future__ import annotations

import json

import pytest

from batcher.dist.executors.ray_runtime.lifecycle import engine_config_json
from batcher.plan.resource import SchedulingEnvelope
from batcher.plan.resource.fleet_plan import plan_worker_slots


def _width(env: SchedulingEnvelope, index: int | None = None) -> int:
    """The rayon width the data plane is shipped for worker `index` under `env`."""
    return json.loads(engine_config_json(num_cpus=env.slot_compute_cpus(index)))["parallelism"]


# --------------------------------------------------------------------- the uniform fill
def test_a_thinned_reservation_still_ships_the_full_width():
    """The 100 x 4-core case: reserve 3 so a task can be placed, compute on all 4."""
    env = SchedulingEnvelope(num_cpus=3.0, n_tasks=100, compute_cpus=4.0)
    assert env.num_cpus == 3.0, "the reservation must stay thinned"
    assert _width(env) == 4


def test_an_unthinned_fleet_is_unchanged():
    """The positive control for the field's default: no reserve, no divergence."""
    env = SchedulingEnvelope(num_cpus=16.0, n_tasks=8)
    assert env.slot_compute_cpus() == 16.0
    assert _width(env) == 16


def test_a_raised_reservation_is_never_narrowed_by_a_stale_width():
    """`_even_cpu_share` can raise `num_cpus` after the width was recorded."""
    env = SchedulingEnvelope(num_cpus=8.0, n_tasks=4, compute_cpus=3.0)
    assert _width(env) == 8


# ------------------------------------------------------------------- the per-node fill
@pytest.mark.parametrize("reserve", [0.0, 1.0])
def test_the_per_node_plan_reserves_cores_without_losing_threads(reserve):
    """One 4-core node and one 24-core node, with and without the reserve.

    The reserve must move the *grant* and leave the *width* on the node's own cores, and the
    4-core node is where the two diverge most.
    """
    slots = plan_worker_slots(
        [4.0, 24.0],
        [8 << 30, 96 << 30],
        target_cores=24.0,
        min_slice_cores=8.0,
        node_reserve_cores=reserve,
    )
    small = slots[0]
    assert small.compute_cpus == 4.0, "the small node's worker computes on its whole node"
    assert small.cpus == (3.0 if reserve else 4.0), "and reserves one core less when asked to"


def test_the_reserve_is_still_taken_off_the_reservation():
    """The positive control: a change that stopped reserving would pass every test above."""
    reserved = plan_worker_slots(
        [4.0] * 3, [8 << 30] * 3, target_cores=24.0, min_slice_cores=8.0, node_reserve_cores=1.0
    )
    tiled = plan_worker_slots(
        [4.0] * 3, [8 << 30] * 3, target_cores=24.0, min_slice_cores=8.0, node_reserve_cores=0.0
    )
    assert sum(s.cpus for s in reserved) == 9.0
    assert sum(s.cpus for s in tiled) == 12.0
    assert [s.compute_cpus for s in reserved] == [s.compute_cpus for s in tiled] == [4.0] * 3


def test_a_slot_carries_its_own_width_through_the_envelope():
    """An unequal fleet's widths are positional, like its grants."""
    env = SchedulingEnvelope(
        num_cpus=3.0,
        n_tasks=2,
        worker_cpus=(3.0, 23.0),
        worker_compute_cpus=(4.0, 24.0),
    )
    assert (_width(env, 0), _width(env, 1)) == (4, 24)
