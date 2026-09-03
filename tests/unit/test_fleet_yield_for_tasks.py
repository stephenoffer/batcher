"""A warm shuffle fleet must hand its cores back to a stage that runs as Ray *tasks*.

A fleet is a placement-group reservation of the cluster's **whole** CPU capacity — one
worker per node, holding that node's cores. A stage whose work is plain Ray tasks submits
them *outside* that reservation, so while the fleet is up those tasks are not slow, they are
unschedulable: `{'CPU': 0.125}: 1+ pending` against `384.0/384.0`, forever, with no error
and no timeout.

`session_fleet_lease` already documents this deadlock and closes the half it can see: a
query that never shuffles no longer takes the fleet hold on entry. This is the other half. A
**staged** query whose first stage does shuffle takes the hold legitimately, and its next
stage is a map — reproduced single-process on `tests/integration/test_distributed.py`, where
`test_distributed_multi_table_join_matches_single_node` hung indefinitely with the driver
parked in `gather_map_results` and one 0.125-CPU task pending against a full cluster.

The yield is refused under exactly the conditions a *respawn* is refused
(`_session_fleet_resizable`): an operator mid-shuffle, or an intermediate published on the
actors that a teardown would destroy.
"""

from __future__ import annotations

import pytest

from batcher.dist.fleet import _fleet, plan_id

pytestmark = pytest.mark.unit


class _FakeFleet:
    """Stands in for a `ShuffleFleet` holding the cluster's cores."""

    def __init__(self) -> None:
        self.cleaned = 0

    def cleanup(self) -> None:
        self.cleaned += 1


@pytest.fixture
def warm(monkeypatch):
    """A cached session fleet with no lease outstanding, and a cluster with no free CPU."""
    fleet = _FakeFleet()
    monkeypatch.setattr(_fleet, "_SESSION", fleet, raising=False)
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 0, raising=False)
    monkeypatch.setattr(_fleet, "_SESSION_QUERY_LEASES", 0, raising=False)
    monkeypatch.setattr(_fleet, "_SESSION_TIMER", None, raising=False)
    monkeypatch.setattr(_fleet, "_free_cluster_cpus", lambda: 0.0)
    monkeypatch.setattr(plan_id, "active_query_scopes", lambda: 1)
    yield fleet
    _fleet._SESSION = None
    _fleet._SESSION_LEASES = 0
    _fleet._SESSION_QUERY_LEASES = 0


def test_a_fleet_holding_the_whole_cluster_is_released_for_a_task_stage(warm):
    """The deadlock case: nothing is free, so the task can never be placed."""
    assert _fleet.yield_session_fleet(0.125) is True
    assert warm.cleaned == 1
    assert _fleet._SESSION is None, "a yielded fleet must not be handed out again"


def test_a_cluster_that_can_place_the_task_keeps_its_warm_fleet(warm, monkeypatch):
    """Yielding costs a respawn, so it is only worth it when the stage is actually stuck."""
    monkeypatch.setattr(_fleet, "_free_cluster_cpus", lambda: 8.0)
    assert _fleet.yield_session_fleet(0.125) is False
    assert warm.cleaned == 0
    # The *largest* task is what decides it: free capacity below that one ask still deadlocks.
    assert _fleet.yield_session_fleet(16.0) is True
    assert warm.cleaned == 1


def test_an_operator_mid_shuffle_keeps_the_fleet(warm):
    """An operator lease means a shuffle is running over these actors right now.

    `_SESSION_LEASES` above `_SESSION_QUERY_LEASES` is that state — and it is also the state
    of a `FlightMaterializedSource` published on the fleet, whose lease is held until the
    next stage has read it. Killing the actors under either is a wrong answer, not a slow one.
    """
    _fleet._SESSION_LEASES = 1
    assert _fleet.yield_session_fleet(0.125) is False
    assert warm.cleaned == 0


def test_a_second_concurrent_query_keeps_the_fleet(warm, monkeypatch):
    """Two pipelines share one fleet; one of them must not tear it out from under the other."""
    monkeypatch.setattr(plan_id, "active_query_scopes", lambda: 2)
    assert _fleet.yield_session_fleet(0.125) is False
    assert warm.cleaned == 0


def test_an_unreadable_cluster_never_costs_a_teardown(warm, monkeypatch):
    """`_free_cluster_cpus` answers `inf` when Ray cannot be asked, which declines the yield."""
    monkeypatch.setattr(_fleet, "_free_cluster_cpus", lambda: float("inf"))
    assert _fleet.yield_session_fleet(0.125) is False
    assert warm.cleaned == 0


def test_no_cached_fleet_is_a_no_op(monkeypatch):
    """The common case — nothing warm to release — costs nothing and answers False."""
    monkeypatch.setattr(_fleet, "_SESSION", None, raising=False)
    assert _fleet.yield_session_fleet(1.0) is False
