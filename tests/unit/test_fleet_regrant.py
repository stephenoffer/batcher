"""A reused shuffle fleet must run under the *borrowing* query's grant, not the spawner's.

A `_FlightWorker` is built from the grant of whichever query spawned it: its credit window
(1 credit = 1 in-flight batch) and the `EngineConfig` its every local `execute_plan` runs
under (memory budget, morsel size, parallelism). The session fleet outlives one query, so
without a re-grant every later query in the session silently runs under the first query's
budget.

Measured on a 9-node cluster, TPC-H sf10 (`lineitem ⋈ orders`, group-by) — same plan, same
data, same 8 live actors:

    fleet spawned by the join   (credits=64, memory_budget=372 MB):   0.6 s
    fleet spawned by a COUNT(*) (credits=1,  memory_budget=1 MB)  :   3.2 s

A global count needs one reducer and a megabyte, and Carbonite correctly grants it exactly
that; the join that followed then inherited `credits=1`, so its Flight exchange held one
batch in flight at a time. Any cheap query poisoned every expensive query after it.
"""

from __future__ import annotations

import pytest

from batcher.dist.fleet import _fleet

pytestmark = pytest.mark.unit


class _FakeActor:
    """Stands in for a `_FlightWorker` Ray actor handle."""

    def __init__(self) -> None:
        self.grants: list[tuple[int, str]] = []

    class _Method:
        def __init__(self, owner: _FakeActor) -> None:
            self.owner = owner

        def remote(self, credits: int, cfg_json: str):
            self.owner.grants.append((credits, cfg_json))
            return ("ref", credits, cfg_json)

    @property
    def set_grant(self) -> _FakeActor._Method:
        return _FakeActor._Method(self)


def _fleet_of(n: int, credits: int, cfg_json: str) -> _fleet.ShuffleFleet:
    actors = [_FakeActor() for _ in range(n)]
    return _fleet.ShuffleFleet(actors, None, [""] * n, credits, cfg_json, 1)


@pytest.fixture
def _no_ray(monkeypatch):
    """`_regrant_fleet` only needs `ray.get` to resolve the fan-out of `set_grant` calls."""
    import sys
    import types

    fake = types.ModuleType("ray")
    fake.get = lambda refs: list(refs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake)
    return fake


def test_regrant_pushes_the_borrowers_grant_to_every_worker(_no_ray):
    fleet = _fleet_of(3, credits=1, cfg_json='{"memory_budget_bytes": 1048576}')

    _fleet._regrant_fleet(fleet, credits=64, cfg_json='{"memory_budget_bytes": 389909338}')

    # Every worker is re-granted — a straggler left on the old grant would serialize the
    # exchange all by itself (the shuffle is only as wide as its narrowest credit window).
    for actor in fleet.actors:
        assert actor.grants == [(64, '{"memory_budget_bytes": 389909338}')]
    # And the fleet records the grant it now carries, so the next acquire sees it as current.
    assert fleet.credits == 64
    assert fleet.cfg_json == '{"memory_budget_bytes": 389909338}'


def test_acquire_regrants_a_fleet_spawned_by_a_cheaper_query(monkeypatch, _no_ray):
    """The bug: a COUNT(*)'s 1-credit / 1 MB fleet served the join that followed it."""
    cheap = _fleet_of(8, credits=1, cfg_json='{"memory_budget_bytes": 1048576}')
    monkeypatch.setattr(_fleet, "_SESSION", cheap)
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 0)
    monkeypatch.setattr(_fleet, "_session_fleet_alive", lambda _f: True)
    monkeypatch.setattr(
        _fleet.ShuffleFleet, "spawn", classmethod(lambda *a, **k: pytest.fail("must not respawn"))
    )

    got = _fleet._acquire_session_fleet(8, 64, '{"memory_budget_bytes": 389909338}')

    # Reused (no respawn: a fleet holds the cluster's whole CPU capacity, so respawning it
    # while it is still being reaped cannot be placed) — but re-granted.
    assert got is cheap
    assert got.credits == 64
    for actor in got.actors:
        assert actor.grants == [(64, '{"memory_budget_bytes": 389909338}')]


def test_acquire_leaves_a_matching_fleet_untouched(monkeypatch, _no_ray):
    """The warm path this cache exists for: the same query again re-grants nothing."""
    cfg = '{"memory_budget_bytes": 389909338}'
    warm = _fleet_of(8, credits=64, cfg_json=cfg)
    monkeypatch.setattr(_fleet, "_SESSION", warm)
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 0)
    monkeypatch.setattr(_fleet, "_session_fleet_alive", lambda _f: True)

    got = _fleet._acquire_session_fleet(8, 64, cfg)

    assert got is warm
    assert all(not a.grants for a in got.actors)  # no RPC at all


def test_acquire_respawns_a_fleet_too_narrow_to_re_grant(monkeypatch, _no_ray):
    """Width is the one thing a re-grant cannot fix."""
    narrow = _fleet_of(2, credits=64, cfg_json="{}")
    monkeypatch.setattr(_fleet.ShuffleFleet, "cleanup", lambda _self: None)
    monkeypatch.setattr(_fleet, "_SESSION", narrow)
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 0)
    monkeypatch.setattr(_fleet, "_session_fleet_alive", lambda _f: True)
    spawned: list[int] = []

    def _spawn(_cls, workers, credits, cfg_json):
        spawned.append(workers)
        return _fleet_of(workers, credits, cfg_json)

    monkeypatch.setattr(_fleet.ShuffleFleet, "spawn", classmethod(_spawn))

    got = _fleet._acquire_session_fleet(8, 64, "{}")

    assert spawned == [8]
    assert len(got.actors) == 8


def test_a_staged_query_widens_the_fleet_it_leases_itself(monkeypatch):
    """A query's own query-scope lease must not veto the resize it is held across.

    Runs against the real `ray` module (unlike the tests above): opening the lease mints a
    plan id, which reaches `flight_worker` — a module that needs a genuine `ray.remote` to
    import. Nothing here reaches the cluster: the fleet is faked and `spawn` is patched.

    `api.adaptive.staging` takes a `session_fleet_lease` for the whole query *before* its
    first stage runs, so the raw lease count is already 1 when that stage's operator asks
    for a fleet. Testing the raw count therefore made the too-narrow branch unreachable on
    the staged path, and a staged query inherited whatever fan-out the first query of the
    process created — permanently, since nothing else resizes it.
    """
    narrow = _fleet_of(2, credits=64, cfg_json="{}")
    monkeypatch.setattr(_fleet, "_SESSION", narrow)
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 0)
    monkeypatch.setattr(_fleet, "_SESSION_QUERY_LEASES", 0)
    monkeypatch.setattr(_fleet, "_session_fleet_alive", lambda _f: True)
    monkeypatch.setattr(_fleet.ShuffleFleet, "cleanup", lambda _self: None)
    monkeypatch.setattr(
        _fleet.ShuffleFleet,
        "spawn",
        classmethod(lambda _c, w, credits, cfg_json: _fleet_of(w, credits, cfg_json)),
    )

    with _fleet.session_fleet_lease():
        got = _fleet._acquire_session_fleet(8, 64, "{}")

    assert len(got.actors) == 8


def test_a_leased_fleet_is_never_torn_down(monkeypatch, _no_ray):
    """Another operator is mid-shuffle over its actors; killing them would fail the query.

    The lease here is an *operator* lease (no matching query-scope lease), which is exactly
    the kind `_session_fleet_resizable` still refuses on.
    """
    narrow = _fleet_of(2, credits=64, cfg_json="{}")
    monkeypatch.setattr(
        _fleet.ShuffleFleet,
        "cleanup",
        lambda _self: pytest.fail("must not tear down a leased fleet"),
    )
    monkeypatch.setattr(_fleet, "_SESSION", narrow)
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 1)
    monkeypatch.setattr(_fleet, "_session_fleet_alive", lambda _f: True)
    monkeypatch.setattr(
        _fleet.ShuffleFleet, "spawn", classmethod(lambda *a, **k: pytest.fail("must not respawn"))
    )

    assert _fleet._acquire_session_fleet(8, 64, "{}") is narrow
