"""A warm shuffle fleet belongs to the Ray session it was spawned in, and to no other.

The session fleet is reused across `collect()` calls, so it outlives the query that spawned
it — and can outlive the cluster. Before the fleet carried its session, a driver that
reconnected (a head restart, a `ray.shutdown()` and a new `ray.init`) kept handing the old
cluster's placement group to `held_placement_group`, a task stage was placed into that
reservation, and Ray parked every task in its infeasible queue: the query hung with the
cluster idle. Measured on `test_placement_locality` followed by a distributed UDF map.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import scheduling
from batcher.dist.fleet import _fleet

pytestmark = pytest.mark.unit


@pytest.fixture
def _session(monkeypatch):
    """The Ray session key the driver reports, settable by the test."""
    current = {"key": "01000000@node-a"}
    monkeypatch.setattr(scheduling, "ray_session_key", lambda: current["key"])
    return current


def _fleet_now() -> _fleet.ShuffleFleet:
    """A fleet stamped with the current session, the way `ShuffleFleet.spawn` stamps it."""
    return _fleet.ShuffleFleet(
        [object()], "pg-of-this-cluster", [""], 1, "{}", 1, session=scheduling.ray_session_key()
    )


def test_spawn_stamps_the_fleet_with_the_session_it_was_spawned_in(monkeypatch, _session):
    import batcher.dist.flight_worker as flight_worker

    monkeypatch.setattr(flight_worker, "new_plan_id", lambda: 7)
    monkeypatch.setattr(_fleet, "_spawn_fleet_with_addrs", lambda *a: ([object()], "pg", [""]))

    assert _fleet.ShuffleFleet.spawn(1, 1, "{}").session == "01000000@node-a"


def test_a_fleet_from_a_previous_session_lends_no_placement_group(monkeypatch, _session):
    monkeypatch.setattr(_fleet, "_SESSION", _fleet_now())
    # Positive control: in its own session the held reservation is handed out.
    assert _fleet.held_placement_group() == "pg-of-this-cluster"

    _session["key"] = "01000000@node-b"  # same job id, new cluster
    assert _fleet.held_placement_group() is None


def test_acquire_respawns_rather_than_reusing_a_previous_sessions_fleet(monkeypatch, _session):
    monkeypatch.setattr(_fleet, "_SESSION", _fleet_now())
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 0)
    _session["key"] = "01000000@node-b"

    def _never_cleanup(self):
        raise AssertionError("a dead cluster's fleet must be dropped, not torn down")

    monkeypatch.setattr(_fleet.ShuffleFleet, "cleanup", _never_cleanup)
    # A liveness ping against the old handles would block for its whole timeout; the
    # session check must reject the fleet before it is ever pinged.
    monkeypatch.setattr(_fleet, "_session_fleet_alive", lambda f: pytest.fail("pinged"))
    fresh = _fleet_now()
    monkeypatch.setattr(_fleet.ShuffleFleet, "spawn", classmethod(lambda cls, *a: fresh))

    got = _fleet._acquire_session_fleet(1, 1, "{}")
    assert got is fresh
    assert got.session == "01000000@node-b"


def test_cleanup_of_a_previous_sessions_fleet_touches_nothing(monkeypatch, _session):
    """The idle-release timer armed in one session fires in the next; it must act on nothing."""
    import sys
    import types

    calls: list[str] = []
    fake = types.ModuleType("ray")
    fake.kill = lambda actor: calls.append("kill")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake)
    import batcher.dist.executors.ray_runtime as rt

    monkeypatch.setattr(rt, "release_placement", lambda pg: calls.append("release"))
    fleet = _fleet_now()
    _session["key"] = "01000000@node-b"

    fleet.cleanup()

    assert calls == []
    assert fleet.actors == [] and fleet.pg is None
    # Positive control: in its own session the same teardown does act.
    _session["key"] = "01000000@node-a"
    live = _fleet_now()
    live.cleanup()
    assert calls == ["kill", "release"]
