"""The fleet planner must not treat its own warm fleet as another tenant's load.

Sizing each worker against its node's *free* cores is right when the busy cores belong to
somebody else and wrong when they belong to us. A warm session fleet holds most of the cluster
by design -- that is the point of `reuse_session_fleet` -- so before this the second query of a
session read a nearly-full cluster and cut its workers out of the remainder.

Measured on the 28-node / 384-core mixed cluster: an idle cluster planned 32 slots totalling
357 cores, and the very next query planned 32 slots totalling 32. The numbers below are that
cluster's shape, scaled down, so a failure here reads as the defect it guards.

What makes this worth a test rather than a comment: nothing downstream fails when the grants
collapse. The query still returns the right answer, every other test still passes, and the only
symptom is that the cluster runs at a fraction of its hardware -- which no assertion anywhere
else is looking at.
"""

from __future__ import annotations

import pytest

from batcher.dist.executor import _free_cores_ignoring_our_own_fleet


@pytest.fixture
def held(monkeypatch):
    """Set what `session_fleet_held_cores` reports for this test."""

    def _set(mapping: dict[str, float]):
        monkeypatch.setattr("batcher.dist.fleet.held.session_fleet_held_cores", lambda: mapping)

    return _set


def _rows(*specs: tuple[str, float, float]) -> list[dict]:
    """`(node_id, nameplate, free)` triples as `node_classes()` records."""
    return [{"node_id": n, "cpus": c, "free_cpus": f} for n, c, f in specs]


def test_our_own_fleet_does_not_make_a_node_look_busy(held):
    """The defect: a warm fleet's own reservation collapsed the next query's grants."""
    held({"big": 92.0, "small": 3.0})
    rows = _rows(("big", 96.0, 4.0), ("small", 4.0, 1.0))

    assert _free_cores_ignoring_our_own_fleet(rows) == [96.0, 4.0]


def test_another_tenant_still_counts_as_busy(held):
    """The positive control: cores we do NOT hold must stay unavailable.

    Without this, "ignore the free-core figure" would pass the test above while over-committing
    every shared cluster -- a fix that is indistinguishable from deleting the check.
    """
    held({})
    rows = _rows(("big", 96.0, 10.0), ("small", 4.0, 1.0))

    assert _free_cores_ignoring_our_own_fleet(rows) == [10.0, 1.0]


def test_a_holding_never_exceeds_the_nameplate(held):
    """Adding a holding back can only restore capacity the node actually has."""
    held({"big": 90.0})
    rows = _rows(("big", 96.0, 50.0))  # 50 + 90 would be 140 on a 96-core box

    assert _free_cores_ignoring_our_own_fleet(rows) == [96.0]


def test_a_node_we_hold_nothing_on_is_untouched(held):
    """Per-node attribution: a holding on one machine must not free cores on another."""
    held({"big": 92.0})
    rows = _rows(("big", 96.0, 4.0), ("other", 16.0, 2.0))

    assert _free_cores_ignoring_our_own_fleet(rows) == [96.0, 2.0]


def test_an_unreadable_holding_keeps_the_old_behaviour(held):
    """`{}` is the documented fallback, and it must be the pre-existing free-core figure."""
    held({})
    rows = _rows(("a", 16.0, 5.0), ("b", 4.0, 4.0))

    assert _free_cores_ignoring_our_own_fleet(rows) == [5.0, 4.0]


def test_a_missing_free_figure_falls_back_to_the_nameplate(held):
    """`node_classes` reports `free_cpus` as `None` when the GCS figure is unreadable."""
    held({})
    rows = [{"node_id": "a", "cpus": 16.0, "free_cpus": None}]

    assert _free_cores_ignoring_our_own_fleet(rows) == [16.0]


def test_the_planner_recovers_the_whole_fleet_shape(held):
    """End to end through `plan_worker_slots`: the shape must match the idle cluster's.

    This is the assertion that would have caught the defect. The two calls differ only in how
    busy the cluster looks, and a planner that believes its own fleet is a stranger cuts 32
    one-core workers out of the second one.
    """
    from batcher.plan.resource.fleet_plan import plan_worker_slots

    cores = [96.0, 48.0, 16.0, 4.0]
    memory = [206 * 2**30, 103 * 2**30, 34 * 2**30, 8 * 2**30]

    def plan(free: list[float]) -> float:
        slots = plan_worker_slots(
            cores,
            memory,
            target_cores=24.0,
            min_slice_cores=3.0,
            node_free_cores=free,
            node_reserve_cores=1.0,
        )
        return sum(s.cpus for s in slots)

    idle = plan(list(cores))
    warm_before_fix = plan([4.0, 2.0, 1.0, 1.0])  # what a warm fleet leaves visible
    held({"n0": 92.0, "n1": 46.0, "n2": 15.0, "n3": 3.0})
    rows = _rows(("n0", 96.0, 4.0), ("n1", 48.0, 2.0), ("n2", 16.0, 1.0), ("n3", 4.0, 1.0))
    warm_after_fix = plan(_free_cores_ignoring_our_own_fleet(rows))

    assert warm_before_fix < idle, "the fixture no longer reproduces the collapse it guards"
    assert warm_after_fix == idle
