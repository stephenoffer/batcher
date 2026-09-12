"""A map task submitted into the shuffle fleet's own reservation must fit one bundle.

`_placeable_scheduling`'s second answer runs a task stage *inside* the held fleet's placement
group. That is the only option when the stage reads an intermediate published on those actors,
and it is the shape the fan-out sizing does not know about: the share is capped at a node
(`_placeable_node_cores`), while the bundles were sized from the query's envelope.

Ray checks a task's whole request against a **single bundle**, so the mismatch is not a queue,
it is a `ValueError` at submit -- `Cannot schedule _map_udf_task with the placement group
because the resource request {'CPU': 16.0, 'memory': 0} cannot fit into any bundles for the
placement group, [{'CPU': 8.0, ...} x 4]` -- and the query fails where the fallback exists to
keep it running. Each actor claims all but `fleet_task_headroom` of its own bundle, so that
headroom is the largest request that can actually be placed beside it.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors import map as mapmod
from batcher.dist.executors.ray_runtime import fleet_task_headroom

pytestmark = pytest.mark.unit


class _PG:
    def __init__(self, bundles: list[dict]) -> None:
        self.bundle_specs = bundles


class _Strategy:
    def __init__(self, pg) -> None:
        self.placement_group = pg


def _sched(bundles: list[dict] | None) -> dict:
    pg = _PG(bundles) if bundles is not None else None
    return {"scheduling_strategy": _Strategy(pg), "memory": 0}


def test_a_node_sized_share_is_cut_to_what_a_bundle_keeps_free() -> None:
    """The failure this exists for: 16 CPU asked, 8-CPU bundles, one core of headroom."""
    sched = _sched([{"CPU": 8.0, "memory": 1048576.0}] * 4)

    clamped = mapmod._clamped_to_fleet_bundles([16.0, 16.0, 4.0], sched)

    assert clamped == [fleet_task_headroom(8.0)] * 3
    assert all(c <= 8.0 for c in clamped), "every share must fit one bundle"


def test_a_share_already_inside_the_headroom_is_untouched() -> None:
    """It is a ceiling, not a resize: a small task keeps its own sizing."""
    sched = _sched([{"CPU": 16.0}] * 8)  # headroom 1.0

    assert mapmod._clamped_to_fleet_bundles([0.125, 0.5, 4.0], sched) == [0.125, 0.5, 1.0]


def test_the_narrowest_bundle_binds() -> None:
    """A heterogeneous fleet places a task on whichever bundle takes it, so size for the
    one that can."""
    sched = _sched([{"CPU": 16.0}, {"CPU": 4.0}])

    assert mapmod._clamped_to_fleet_bundles([9.0], sched) == [fleet_task_headroom(4.0)]


def test_a_bundle_with_no_room_still_leaves_a_placeable_share() -> None:
    """`fleet_task_headroom` is 0 at one core. A share of 0 is not a task Ray can run."""
    sched = _sched([{"CPU": 1.0}])

    assert mapmod._clamped_to_fleet_bundles([8.0], sched) == [mapmod._MIN_TASK_CPU]


def test_an_unreadable_group_leaves_the_stage_alone() -> None:
    """Never shrink a stage on a guess: no group, no clamp."""
    for sched in ({}, _sched(None), _sched([]), _sched([{"GPU": 1.0}])):
        assert mapmod._clamped_to_fleet_bundles([16.0, 2.0], sched) == [16.0, 2.0]
