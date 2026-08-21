"""Many pipelines on one cluster: shared actors, isolated shuffle namespaces.

The warm session fleet (`dist.fleet`) is what lets several `collect()` calls reuse one
set of `_FlightWorker` actors instead of each reserving the cluster's whole CPU capacity.
Sharing the *actors* is intended; sharing the *ticket namespace* is not. A shuffle ticket
is `(plan, stage, src, dst, epoch)`, and `plan` is the fence that keeps one query's
published partitions unreadable by another. These tests pin that the fence is minted per
**query**, not per fleet spawn — the distinction is invisible until two pipelines run at
once over the same fleet, at which point identical tickets mean one query silently reads
the other's buckets.
"""

from __future__ import annotations

import threading

import pyarrow as pa
import pytest

import batcher as bt
import batcher.dist.fleet._fleet as _fleet
import batcher.dist.fleet.plan_id as plan_id_mod
import batcher.dist.fleet.query as query_mod
from batcher.dist import flight_worker as fw
from batcher.dist.fleet.plan_id import query_plan_id, query_shuffle_scope


class _FakeFleet:
    """A session fleet stand-in: real `plan_id`/grant, no Ray actors."""

    def __init__(self, plan_id: int, n: int = 2, num_cpus: float = 0.0) -> None:
        self.plan_id = plan_id
        self.actors = [object() for _ in range(n)]
        self.pg = None
        self.addrs = ["host:1"] * n
        self.credits = 8
        self.cfg_json = "{}"
        # The per-worker core grant `_fleet_is_too_thin` reads. `ShuffleFleet` defaults it to
        # 0.0 meaning "not recorded", which that check reads as "leave the fleet alone" — the
        # behaviour every test here assumes, since none of them is about respawn-on-collapse.
        self.num_cpus = num_cpus

    @property
    def workers(self) -> int:
        return len(self.actors)


def _install_fake_session(monkeypatch, fleet: _FakeFleet) -> None:
    """Pin a warm session fleet in place, with liveness/teardown stubbed out."""
    monkeypatch.setattr(_fleet, "_SESSION", fleet, raising=False)
    monkeypatch.setattr(_fleet, "_SESSION_LEASES", 0, raising=False)
    # Reset the scope counter too: it gates fleet re-grant and rival-fleet spawning, so a
    # value leaked from another test would silently change what those paths decide.
    monkeypatch.setattr(plan_id_mod, "_ACTIVE_SCOPES", 0, raising=False)
    monkeypatch.setattr(_fleet, "_SESSION_TIMER", None, raising=False)
    monkeypatch.setattr(_fleet, "_session_fleet_alive", lambda _f: True)
    monkeypatch.setattr(_fleet, "_regrant_fleet", lambda _f, _c, _j: None)


def test_concurrent_queries_sharing_a_fleet_get_distinct_plan_ids(monkeypatch):
    """Two queries borrowing one warm fleet must not mint the same shuffle plan id.

    Both borrow the *same* actors — that is the point of the session fleet — so the only
    thing separating their published partitions is the ticket's `plan` field.
    """
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))

    ids = []
    for _ in range(2):
        with _fleet.session_fleet_lease():
            _fleet.acquire_fleet(2, 8, "{}")
            ids.append(fw.current_plan_id())

    assert ids[0] != ids[1], f"both queries fenced under the same plan id {ids[0]}"


def test_concurrent_queries_sharing_a_fleet_do_not_collide_on_tickets(monkeypatch):
    """The end the plan id serves: identical shuffle coordinates stay distinct tickets.

    Same stage, same mapper, same reducer, same epoch — the exact coordinate two
    concurrent group-bys over one fleet both produce.
    """
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))

    tickets = []
    for _ in range(2):
        with _fleet.session_fleet_lease():
            _fleet.acquire_fleet(2, 8, "{}")
            tickets.append(str(fw._ticket(0, 3, 5, 0)))

    assert tickets[0] != tickets[1], f"both queries published under ticket {tickets[0]}"


def _run_second_pipeline_while(second, *, first):
    """Run `second` in its own query scope while a first pipeline holds one.

    Concurrent pipelines are separate *contexts*, not nested blocks — a new thread starts
    with a fresh `ContextVar` context, so the second genuinely opens its own scope and the
    process sees two active queries. Nesting would model one query's stages instead.
    """
    done = threading.Event()
    error: list = []

    def _run():
        try:
            with _fleet.session_fleet_lease():
                second()
        except BaseException as exc:  # surface it rather than hanging the assert
            error.append(exc)
        finally:
            done.set()

    with _fleet.session_fleet_lease():
        first()
        t = threading.Thread(target=_run)
        t.start()
        assert done.wait(timeout=30), "the concurrent pipeline did not finish"
    t.join(timeout=30)
    if error:
        raise error[0]


def test_a_second_query_does_not_regrant_a_fleet_in_use(monkeypatch):
    """An arriving pipeline must not retune workers another is already shuffling over.

    `_regrant_fleet` rewrites each worker's credit window and `EngineConfig` in place. It
    exists to stop a cheap query's grant poisoning the *next* expensive one, but applied
    while a concurrent query is mid-shuffle it does the same damage to the *current* one.
    """
    fleet = _FakeFleet(plan_id=4242)
    _install_fake_session(monkeypatch, fleet)
    regrants: list[str] = []
    monkeypatch.setattr(_fleet, "_regrant_fleet", lambda _f, _c, j: regrants.append(j))

    # Pipeline B runs in its own thread — a genuinely separate context, which is what a
    # concurrent pipeline is. (Nesting the lease would model one query's stages, not two
    # queries, and would correctly be counted once.)
    _run_second_pipeline_while(
        lambda: _fleet.acquire_fleet(2, 64, '{"memory_budget_bytes":999}'),
        first=lambda: _fleet.acquire_fleet(2, 8, "{}"),
    )

    assert regrants == [], f"a concurrent pipeline was re-granted mid-shuffle: {regrants}"


def test_a_lone_query_still_regrants_the_warm_fleet(monkeypatch):
    """The uncontended case must keep re-granting — that is what the fleet reuse is for.

    Guards the fix above from over-reaching into the single-pipeline path, where inheriting
    the previous query's grant is the measured 5x regression `_regrant_fleet` prevents.
    """
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))
    regrants: list[str] = []
    monkeypatch.setattr(_fleet, "_regrant_fleet", lambda _f, _c, j: regrants.append(j))

    with _fleet.session_fleet_lease():
        _fleet.acquire_fleet(2, 64, '{"memory_budget_bytes":999}')

    assert regrants == ['{"memory_budget_bytes":999}']


def test_non_adaptive_queries_are_fenced_too(monkeypatch):
    """The commonest distributed shape must be fenced, not just the adaptive one.

    `adaptive="auto"` resolves to False for a plain distributed `group_by`, so those
    queries never take a `session_fleet_lease`. They still borrow the warm session fleet
    (on by default), so without their own scope they inherited the *fleet's* id — shared
    by every concurrent pipeline — and published byte-identical tickets.
    """
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))

    tickets = []
    for _ in range(2):
        with query_shuffle_scope():  # what `execute_distributed` now opens per query
            _fleet.acquire_fleet(2, 8, "{}")
            tickets.append(str(fw._ticket(0, 3, 5, 0)))

    assert tickets[0] != tickets[1], f"both queries published under ticket {tickets[0]}"


def test_non_adaptive_queries_also_block_a_mid_flight_regrant(monkeypatch):
    """The re-grant guard must count non-adaptive queries too, not just leased ones.

    Both protections key off "is another query running". When that was counted only by the
    adaptive path's lease, a concurrent *non-adaptive* query — the commonest kind — was
    invisible, and the fleet was re-granted out from under it mid-shuffle.
    """
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))
    regrants: list[str] = []
    monkeypatch.setattr(_fleet, "_regrant_fleet", lambda _f, _c, j: regrants.append(j))

    done = threading.Event()

    def pipeline_b():
        with query_shuffle_scope():  # a one-shot query: scope, no lease
            _fleet.acquire_fleet(2, 64, '{"memory_budget_bytes":999}')
        done.set()

    with query_shuffle_scope():  # pipeline A, one-shot, mid-shuffle
        _fleet.acquire_fleet(2, 8, "{}")
        t = threading.Thread(target=pipeline_b)
        t.start()
        assert done.wait(timeout=30)
    t.join(timeout=30)

    assert regrants == [], f"a concurrent one-shot pipeline was re-granted: {regrants}"


def test_the_distributed_entry_point_opens_a_scope():
    """Pin the fence to the entry point, so a new caller cannot forget to open one."""
    from batcher.dist.executor import execute_distributed

    assert hasattr(execute_distributed, "__wrapped__"), (
        "execute_distributed is no longer wrapped in a query shuffle scope"
    )


def test_a_staged_query_keeps_one_fence_across_stages():
    """Nesting must inherit, never re-mint: a stage's intermediate is read by the next.

    The adaptive loop opens the scope once per query and each stage's
    `execute_distributed` opens another inside it. Re-minting there would republish the
    next stage under a different plan id and orphan the intermediate.
    """
    with query_shuffle_scope():
        outer = query_plan_id()
        with query_shuffle_scope():
            assert query_plan_id() == outer, "a stage re-minted the query's fence"
        assert query_plan_id() == outer, "the query's fence did not survive its stage"
    assert query_plan_id() is None, "the fence outlived the query"


def _force_query_fleet_path(monkeypatch, spawned: list) -> None:
    """Take `maybe_spawn_query_fleet` past its config/topology guards to the spawn decision."""
    from batcher.config import active_config

    cfg = active_config()
    monkeypatch.setattr(type(cfg.distributed), "persistent_fleet", property(lambda _s: True))
    # `maybe_spawn_query_fleet` lives in `fleet.query`, not `fleet._fleet`, so the names it
    # resolves are that module's — patching them on `_fleet` silently patches nothing.
    monkeypatch.setattr(query_mod, "available_cpu_count", lambda: 8)
    mod = "batcher.dist.executors.ray_runtime"
    monkeypatch.setattr(f"{mod}._ensure_ray", lambda *_a, **_k: None)
    # Sized against the fleet's own per-worker grant, so the stub takes the extra arguments
    # rather than pinning a signature this test is not about.
    monkeypatch.setattr(f"{mod}.clamp_workers", lambda w, *_a, **_k: w)
    monkeypatch.setattr(f"{mod}.current_envelope", lambda *_a, **_k: None)
    monkeypatch.setattr(f"{mod}.resolve_transport", lambda *_a: "flight")
    monkeypatch.setattr(f"{mod}.request_autoscale", lambda *_a, **_k: None)
    monkeypatch.setattr(f"{mod}.release_autoscale", lambda *_a, **_k: None)
    monkeypatch.setattr("batcher.dist.flight_aggregate._shuffle_credits", lambda *_a: 8)
    monkeypatch.setattr(f"{mod}.engine_config_json", lambda *_a, **_k: "{}")
    monkeypatch.setattr(
        _fleet.ShuffleFleet,
        "spawn",
        classmethod(lambda _c, w, *_a: spawned.append(w) or _FakeFleet(1, w)),
    )


def test_a_concurrent_pipeline_shares_the_fleet_instead_of_reserving_a_second(monkeypatch):
    """A second pipeline must not gang-request the cluster a first one is already running on.

    A fleet asks for one worker per node holding that node's cores, so two fleets cannot
    both be placed. `maybe_spawn_query_fleet` frees the warm fleet first to avoid exactly
    that — but the free is a no-op while another pipeline holds it, and spawning anyway
    degrades to whatever few workers can be placed rather than failing outright.
    """
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))
    spawned: list = []
    _force_query_fleet_path(monkeypatch, spawned)

    # Pipeline A is already running in another context when B asks for a fleet.
    got: list = []
    _run_second_pipeline_while(
        lambda: got.append(query_mod.maybe_spawn_query_fleet(8, "flight")), first=lambda: None
    )
    fleet = got[0]

    assert fleet is None, "a second pipeline reserved its own cluster-wide fleet"
    assert spawned == [], f"a competing gang reservation was issued: {spawned}"


def test_a_lone_pipeline_still_spawns_its_query_fleet(monkeypatch):
    """The uncontended adaptive path must keep its persistent fleet — the staged-query win."""
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))
    spawned: list = []
    _force_query_fleet_path(monkeypatch, spawned)
    monkeypatch.setattr(query_mod, "release_session_fleet", lambda: None)

    with _fleet.session_fleet_lease():
        fleet = query_mod.maybe_spawn_query_fleet(8, "flight")

    assert fleet is not None and spawned == [8]


def test_plan_id_is_isolated_across_threads(monkeypatch):
    """A second pipeline in another thread must not retarget the first one's tickets.

    `_current_plan_id` used to be a module global, so the *later* query's
    `set_current_plan_id` retroactively changed the tickets the earlier query's driver
    was still building — it would fetch under an id its workers never published to.
    """
    _install_fake_session(monkeypatch, _FakeFleet(plan_id=4242))

    seen: dict[str, int] = {}
    started = threading.Event()
    other_done = threading.Event()

    def other() -> None:
        with _fleet.session_fleet_lease():
            _fleet.acquire_fleet(2, 8, "{}")
            seen["other"] = fw.current_plan_id()
        other_done.set()

    with _fleet.session_fleet_lease():
        _fleet.acquire_fleet(2, 8, "{}")
        mine_before = fw.current_plan_id()
        started.set()
        t = threading.Thread(target=other)
        t.start()
        other_done.wait(timeout=30)
        mine_after = fw.current_plan_id()
    t.join(timeout=30)

    assert mine_before == mine_after, "a concurrent pipeline retargeted this query's tickets"
    assert seen["other"] != mine_before, "the concurrent pipeline shared this query's fence"


def test_two_real_pipelines_run_at_once_and_both_are_correct():
    """End to end: two distributed pipelines at once, each matching the single-node oracle.

    The acceptance test for the whole concurrency story — it exercises Ray bring-up, fleet
    sharing, the shuffle fence, and the shared memory envelope in one go, against the
    single-node result as the oracle. It caught the `ray.init` race: started together on a
    cold Ray, both pipelines saw "not initialized", both called `ray.init`, and the loser
    died on Ray's own "called ray.init twice" assertion before it ran a single row.
    """
    ray = pytest.importorskip("ray")
    assert ray is not None

    table = pa.table({"k": [i % 7 for i in range(4000)], "v": [i % 5 for i in range(4000)]})
    query = lambda: bt.from_arrow(table).group_by("k").agg(s=bt.col("v").sum())  # noqa: E731
    oracle = query().collect().to_pydict()
    expected = dict(zip(oracle["k"], oracle["s"], strict=True))

    results: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}

    def pipeline(tag: str) -> None:
        try:
            out = query().collect(distributed=True, num_workers=2).to_pydict()
            results[tag] = dict(zip(out["k"], out["s"], strict=True))
        except BaseException as exc:  # report it rather than hanging the join
            errors[tag] = exc

    threads = [threading.Thread(target=pipeline, args=(t,)) for t in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)

    assert not errors, f"a concurrent pipeline failed: {errors}"
    assert results["A"] == expected, "pipeline A disagreed with the single-node oracle"
    assert results["B"] == expected, "pipeline B disagreed with the single-node oracle"
