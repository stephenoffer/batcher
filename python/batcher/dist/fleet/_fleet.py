"""A query-lifetime shuffle-actor fleet for the adaptive Flight path.

Each Flight-shuffle operator (aggregate, join, sort, window) by default spawns its
own `_FlightWorker` fleet + placement group and tears it down when it finishes. For
an adaptive multi-stage query that is wasteful and, worse, *blocks the data plane
from staying on the workers between stages*: keeping a stage's result on persistent
actors while the next stage reserves a fresh SPREAD placement group makes the new
gang reservation contend with the still-held bundles and deadlock.

`ShuffleFleet` removes that hazard by reserving **one** placement group + worker
fleet for the whole query and installing it as an ambient handle. Every Flight
operator that runs under it *borrows* the fleet instead of spawning its own, so a
stage's intermediate stays partitioned on the workers (a `FlightMaterializedSource`)
and the next stage reads its bucket in place — no driver collect, no per-stage
placement churn, hence no second reservation to deadlock against. The fleet is owned
by the adaptive loop (`api.adaptive.execute_adaptive`) and freed once, at query end.

The fleet is ambient (a `ContextVar`, mirroring the scheduling-envelope pattern in
`dist.executors.ray_runtime`) so it reaches each operator without threading through
every signature. With no fleet installed, every operator spawns and frees its own —
the pre-existing behavior — so single-node == distributed stays bit-identical.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading

from batcher._internal.errors import ResourceError
from batcher._internal.logging import note_suppressed
from batcher.dist.fleet.plan_id import active_query_scopes, adopt_plan_id, query_shuffle_scope

__all__ = [
    "ShuffleFleet",
    "acquire_fleet",
    "current_fleet",
    "release_fleet",
    "release_session_fleet",
    "release_session_lease",
    "reset_fleet",
    "session_fleet_lease",
    "set_fleet",
]

# The shuffle fleet in force for the current adaptive query, if any. Ambient so a
# Flight operator borrows it without it being threaded through every call.
_FLEET: contextvars.ContextVar[ShuffleFleet | None] = contextvars.ContextVar(
    "batcher_shuffle_fleet", default=None
)

# How many of `_SESSION_LEASES` are query-scoped (`session_fleet_lease`) rather than held
# by a running operator or by a published intermediate. Counted separately because the two
# kinds mean opposite things for resizing: see `_session_fleet_resizable`. Guarded by
# `_SESSION_LOCK`, like `_SESSION_LEASES` itself.
_SESSION_QUERY_LEASES = 0


def _spawn_fleet_with_addrs(workers: int, credits: int, cfg_json: str, plan_id: int | None = None):
    """Spawn the worker fleet and fetch their Flight addresses, releasing the gang on failure.

    Returns ``(actors, placement_group, addrs)``. If anything between reserving the
    placement group and collecting every worker's advertised address fails (an actor
    that can't bind its Flight server, a node lost mid-spawn, an interrupt), the actors
    are killed and the placement group released before the error propagates — otherwise
    the reserved gang would leak (no `ShuffleFleet` is constructed, so its `cleanup`
    never runs). The single guarded spawn point both `ShuffleFleet.spawn` and the
    transient `acquire_fleet` path go through.
    """
    import ray

    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import release_placement
    from batcher.dist.flight_worker import spawn_flight_workers

    actors, pg = spawn_flight_workers(workers, credits, cfg_json, plan_id)
    ok = False
    try:
        # Bounded wait for every worker to advertise its Flight address. An un-placeable
        # actor (the request outran the schedulable node count) or one lost to a spot
        # preemption mid-spawn would otherwise leave `ray.get` blocking FOREVER — the whole
        # query hangs on fleet startup. Instead, wait up to the placement timeout and
        # proceed with whichever workers came up, killing the stragglers: the mergeable
        # shuffle algebra makes any (>=1) worker count result-identical, so a smaller fleet
        # is a scheduling degradation, never a wrong answer.
        addr_refs = [a.addr.remote() for a in actors]
        timeout = max(1.0, active_config().distributed.placement_timeout_s)
        ready, pending = ray.wait(addr_refs, num_returns=len(addr_refs), timeout=timeout)
        if pending:
            # A fleet asks for one worker per node holding that node's cores, i.e. the
            # cluster's *whole* CPU capacity (`_even_cpu_share`). So it is placeable only
            # when the cluster is genuinely idle — and the most common reason it isn't is
            # a fleet that was torn down microseconds ago, whose actors Ray has not yet
            # reaped. Degrading immediately turns that transient into a *cached* 1-2 worker
            # fleet that then serves the rest of the session (measured: an 8-worker
            # distributed join left running on 2 workers, 0.6 s -> 16 s). Give the
            # reclamation one more placement window before accepting a smaller fleet.
            ready, pending = ray.wait(addr_refs, num_returns=len(addr_refs), timeout=timeout)
        if pending:
            ready_set = set(ready)
            for a, ref in zip(actors, addr_refs, strict=True):
                if ref not in ready_set:
                    with contextlib.suppress(Exception):
                        ray.kill(a)  # a straggler that never came up — reclaim its slot
            actors = [a for a, ref in zip(actors, addr_refs, strict=True) if ref in ready_set]
            addr_refs = ready
            _warn_degraded_fleet(len(actors), workers, timeout)
        if not actors:  # nothing came up at all — a real, actionable failure, not a hang
            raise ResourceError(
                f"no distributed worker became available within {timeout:.0f}s: "
                f"{_fleet_demand_reason() or 'the cluster is over-subscribed or unschedulable'}"
                "; retry or reduce num_workers"
            )
        addrs = list(ray.get(addr_refs))
        ok = True
        return actors, pg, addrs
    finally:
        if not ok:
            for a in actors:
                with contextlib.suppress(Exception):
                    ray.kill(a)
            release_placement(pg)


def _fleet_demand_reason() -> str | None:
    """Why the fleet's workers could not be placed, in the ask's own terms, or `None`.

    "The cluster is over-subscribed or unschedulable" names both possibilities and
    distinguishes neither, which is the wrong half of the answer to give someone whose query
    just failed: the two have opposite fixes. Asking a worker for more cores than any node
    has is settled by changing the grant; a cluster somebody else is holding is settled by
    waiting or by looking at who. The topology already knows which one it is.
    """
    try:
        from batcher.dist.executors.ray_runtime.capacity import Demand, describe_pending_demand
        from batcher.dist.executors.ray_runtime.scheduling import current_envelope

        return describe_pending_demand(Demand.from_envelope(current_envelope()))
    except Exception as exc:  # pragma: no cover - a diagnosis never replaces the failure
        note_suppressed("dist", "diagnose the unplaceable fleet", exc)
        return None


def _warn_degraded_fleet(placed: int, wanted: int, timeout: float) -> None:
    """Say so when a fleet comes up narrower than it asked for.

    This is the most expensive silent degradation on the distributed path and it left no
    trace at all: the stragglers are killed, the survivors serve the query, and — on the
    session-cached path — the rest of the session too. Measured on an 8-worker distributed
    join that came up with 2: **0.6 s becomes 16 s**, with nothing anywhere to connect the
    two. A query that runs at a quarter of its width should not have to be inferred from a
    stopwatch.

    Best-effort: reporting a degradation must not turn it into a failure.
    """
    if placed >= wanted:
        return
    try:
        from batcher._internal.logging import get_logger, log_kv

        log_kv(
            get_logger("dist"),
            logging.WARNING,
            "shuffle fleet came up narrower than requested; the query runs at reduced width",
            placed=placed,
            requested=wanted,
            waited_s=round(timeout * 2, 1),
            reason=_fleet_demand_reason() or "workers did not advertise in time",
        )
    except Exception as exc:  # pragma: no cover - observation must never fail a spawn
        note_suppressed("dist", "report the degraded fleet", exc)


class ShuffleFleet:
    """One placement group + `_FlightWorker` fleet reused across a query's stages.

    Holds the actors, their advertised Flight addresses, and the grant (credits +
    engine config) they were spawned with, so a borrowing operator runs every stage
    against the *same* fleet with the *same* worker count. `cleanup()` is the single
    teardown point — the adaptive loop calls it once, in its `finally`.
    """

    __slots__ = ("actors", "addrs", "cfg_json", "credits", "num_cpus", "pg", "plan_id")

    def __init__(
        self,
        actors,
        pg,
        addrs,
        credits: int,
        cfg_json: str,
        plan_id: int,
        num_cpus: float = 0.0,
    ) -> None:
        self.actors = actors
        self.pg = pg
        self.addrs = addrs
        self.credits = credits
        self.cfg_json = cfg_json
        # The query's shuffle plan id, set on the driver whenever this fleet is
        # borrowed so every borrowing operator's tickets fence to this query.
        self.plan_id = plan_id
        # The per-worker core grant these actors hold. Recorded because worker *count* alone
        # cannot tell a healthy fleet from a degenerate one: 175 one-core workers and 16
        # sixteen-core workers occupy the same cluster, and only the second can use it. See
        # `_fleet_is_too_thin`.
        self.num_cpus = num_cpus

    @property
    def workers(self) -> int:
        """The fixed worker count for the whole query (the fleet's actor count)."""
        return len(self.actors)

    @classmethod
    def spawn(cls, workers: int, credits: int, cfg_json: str) -> ShuffleFleet:
        """Gang-schedule `workers` actors once and cache their advertised addresses."""
        from batcher.dist.flight_worker import new_plan_id

        plan_id = new_plan_id()
        actors, pg, addrs = _spawn_fleet_with_addrs(workers, credits, cfg_json, plan_id)
        return cls(actors, pg, addrs, credits, cfg_json, plan_id, _wanted_grant())

    def cleanup(self) -> None:
        """Kill the fleet's actors and release its placement group (idempotent)."""
        import ray

        from batcher.dist.executors.ray_runtime import release_placement

        for a in self.actors:
            with contextlib.suppress(Exception):
                ray.kill(a)
        self.actors = []
        release_placement(self.pg)
        self.pg = None


# --- Session fleet: one warm fleet reused across separate distributed queries -------
# Guards `_SESSION` (the cached cross-query fleet) and its idle-release timer. A query
# fleet (the adaptive-loop `ContextVar` above) always wins over this; this only serves
# the otherwise-transient per-operator spawn so a second `collect()` starts warm.
_SESSION_LOCK = threading.RLock()
_SESSION: ShuffleFleet | None = None
_SESSION_TIMER: threading.Timer | None = None
# Outstanding borrows of the session fleet. The idle timer may only fire when this is
# zero: "idle" means *no operator is using the fleet*, not "N seconds since someone
# acquired it". Armed at acquire time, the timer would `ray.kill` the actors out from
# under any query that ran longer than `session_fleet_idle_s` (30s by default) — which
# is every large distributed join, and which surfaced as a mid-query `ActorDiedError`
# ("killed by ray.kill") from the shuffle's own recovery path.
_SESSION_LEASES = 0


def _wanted_grant() -> float:
    """The per-worker core grant the caller's scheduling envelope asks for, or 0 if unknown.

    `execute_distributed` has already resolved the fan-out and installed it as the ambient
    envelope by the time a fleet is spawned or borrowed, so this is the sizing decision for
    *this* query rather than a second guess at it. 0 means "no envelope", which every
    comparison below treats as "do not judge".
    """
    try:
        from batcher.dist.executors.ray_runtime import current_envelope

        env = current_envelope()
        return float(env.num_cpus) if env is not None else 0.0
    except Exception as exc:
        note_suppressed("dist", "read the scheduling envelope's worker grant", exc)
        return 0.0


#: How much thinner than the current sizing a cached fleet may be before it is respawned.
#:
#: A cached fleet is normally kept: respawning costs 1-2s and the whole point of the session
#: fleet is to skip that. But `dist.executor._placeable_grant` sizes the grant from *free*
#: capacity so the gang can be placed, and when a query's own map tasks (or the previous
#: fleet) already hold the cores it thins all the way to one core per worker. That fleet then
#: occupies the cluster, so the next sizing sees no free capacity either and thins again —
#: and because `_acquire_session_fleet` only ever respawned a fleet that was too *narrow*, a
#: 175-worker one-core fleet is never narrower than a 16-worker request and the process stays
#: on one-core workers for the rest of its life. Measured: TPC-H sf10 distributed went from
#: 27s for all 22 queries to over 20 minutes reaching q9.
#:
#: Half is deliberately loose. The comparison is against a grant that is itself derived from
#: live capacity, so it wobbles by a core or two between queries for reasons that are not a
#: pathology; respawning on that would trade the ratchet for churn. A factor of two only
#: fires on the collapse this exists to catch.
_FLEET_THINNESS_TOLERANCE = 0.5


def _fleet_is_too_thin(fleet: ShuffleFleet, wanted: float) -> bool:
    """Whether `fleet`'s per-worker grant has collapsed far below what this query wants.

    Both figures must be known and positive: a fleet spawned before the grant was recorded,
    or a query with no envelope, is left alone rather than respawned on a guess.
    """
    return bool(
        fleet.num_cpus > 0 and wanted > 0 and fleet.num_cpus < wanted * _FLEET_THINNESS_TOLERANCE
    )


def _session_fleet_alive(fleet: ShuffleFleet) -> bool:
    """Whether every actor in `fleet` is still reachable (cheap liveness ping)."""
    import ray

    if not fleet.actors:
        return False
    try:
        ray.get([a.addr.remote() for a in fleet.actors], timeout=10.0)
        return True
    except Exception:
        return False


def _arm_idle_release(idle_s: float) -> None:
    """(Re)start the idle timer that tears down the session fleet after `idle_s`."""
    global _SESSION_TIMER
    if _SESSION_TIMER is not None:
        _SESSION_TIMER.cancel()
    if idle_s <= 0:
        return
    _SESSION_TIMER = threading.Timer(idle_s, release_session_fleet)
    _SESSION_TIMER.daemon = True
    _SESSION_TIMER.start()


def _regrant_fleet(fleet: ShuffleFleet, credits: int, cfg_json: str) -> None:
    """Re-grant a reused fleet's workers for the query about to borrow it.

    A worker is built from the grant of whichever query *spawned* it — its credit window
    (1 credit = 1 in-flight batch) and the `EngineConfig` its every local `execute_plan`
    runs under (memory budget, morsel size, parallelism). Reusing the fleet without
    re-granting therefore runs every later query in the session under the *first* query's
    budget. Measured on the 9-node cluster, TPC-H sf10 (`lineitem ⋈ orders`, group-by):

        fleet spawned by the join   : credits=64, memory_budget=372 MB ->  0.6 s
        fleet spawned by a COUNT(*) : credits=1,  memory_budget=1 MB   ->  3.2 s

    Same plan, same data, same 8 live actors — the join simply inherited the count's grant,
    so its Flight exchange held one batch in flight at a time against a 1 MB budget. Any
    cheap query poisoned every expensive query after it.

    Re-granting is two attribute writes per worker. Respawning instead would be the obvious
    alternative and is the wrong one: a fleet asks for one worker per node holding that
    node's cores — the cluster's entire CPU capacity — so a respawn issued while the fleet
    it replaces is still being reaped cannot be placed, and the spawn silently degrades to
    the 1-2 workers it *can* place (measured: the same join at 16 s on a 2-worker fleet).
    """
    import ray

    ray.get([a.set_grant.remote(credits, cfg_json) for a in fleet.actors])
    fleet.credits = credits
    fleet.cfg_json = cfg_json


def _session_fleet_resizable() -> bool:
    """Whether the cached session fleet may be torn down and respawned right now.

    Not every lease means the same thing. An **operator** lease (`acquire_fleet`) says a
    shuffle is running over these actors right now, and a lease still held after an
    operator returns says it left a `FlightMaterializedSource` published on them; both die
    with a respawn. A **query-scope** lease (`session_fleet_lease`) says only that a staged
    query intends to use the fleet across its stages — it is taken by
    `api.adaptive.staging` *before* the first stage runs, when nothing has been published.

    Testing the raw count conflated the two, and the query-scope lease is the common case:
    by the time a staged query's first operator asked for a fleet, the counter it had
    incremented itself was already 1, so the too-narrow/too-thin test could never fire. A
    staged query therefore inherited whatever fan-out the first query of the process
    happened to create, for the life of the process — a 2-worker fleet serving an 8-worker
    query, with no way back up. Since the grant also collapses to one core per worker
    against a busy cluster (`_placeable_grant`), that is how a process gets permanently
    stuck on a degenerate fleet: exactly the state `_fleet_is_too_thin` detects and was
    then unable to act on.

    So the fleet is resizable when every outstanding lease is query-scoped and at most one
    query holds it — nobody is mid-shuffle, and nothing is published that a respawn would
    destroy. An unleased fleet satisfies this trivially, which is the historical rule.
    """
    from batcher.dist.fleet.plan_id import active_query_scopes

    return _SESSION_LEASES <= _SESSION_QUERY_LEASES and active_query_scopes() <= 1


def _acquire_session_fleet(workers: int, credits: int, cfg_json: str) -> ShuffleFleet:
    """Get the warm session fleet, spawning (or respawning it) as needed.

    A cached fleet wide enough for this query is reused — that is the whole point of the
    session fleet, and what turns a ~3 s warm query into ~1 s (a spawn is a placement group
    + N actors + N Flight servers) — but it is **re-granted** first, so it runs under *this*
    query's credits and `EngineConfig` rather than the spawning query's (`_regrant_fleet`,
    which is where the 5x regression that motivated this lives).

    Only a fleet that is too **narrow** is torn down and respawned: that is the one thing a
    re-grant cannot fix. A fleet still in use (leased) is never torn down — the borrower is
    mid-shuffle over its actors — so it is reused as-is and the next uncontended acquire
    resizes it.

    Re-granting is skipped while a **second query** holds the fleet: the in-place rewrite
    would retune workers a concurrent query is already shuffling over — the poisoning
    `_regrant_fleet` prevents between queries, now mid-flight. The arriving query runs
    under the incumbent's grant: a scheduling degradation, never a wrong answer.

    A fleet whose actors died (preemption) is torn down and respawned transparently.

    Takes a **lease** on the fleet: the idle timer is cancelled for as long as any
    operator holds one, and re-armed only by the matching `release_session_lease`. The
    borrower MUST release it (the Flight operators do so via `release_fleet`).
    """
    global _SESSION, _SESSION_LEASES, _SESSION_TIMER

    with _SESSION_LOCK:
        # In use ⇒ not idle. Stop any pending teardown before handing the fleet out.
        if _SESSION_TIMER is not None:
            _SESSION_TIMER.cancel()
            _SESSION_TIMER = None
        if _SESSION is not None and not _session_fleet_alive(_SESSION):
            with contextlib.suppress(Exception):
                _SESSION.cleanup()
            _SESSION = None
        # Too narrow *or too thin* for this query, and nobody is mid-shuffle over it →
        # respawn. Width alone was the whole test, which is what let the one-core collapse
        # in `_FLEET_THINNESS_TOLERANCE` persist for the life of the process: a fleet with
        # ten times the workers at a sixteenth of the cores each is never "too narrow".
        if (
            _SESSION is not None
            and _session_fleet_resizable()
            and (len(_SESSION.actors) < workers or _fleet_is_too_thin(_SESSION, _wanted_grant()))
        ):
            with contextlib.suppress(Exception):
                _SESSION.cleanup()
            _SESSION = None
        if _SESSION is None:
            _SESSION = ShuffleFleet.spawn(workers, credits, cfg_json)
        elif active_query_scopes() <= 1 and (_SESSION.credits, _SESSION.cfg_json) != (
            credits,
            cfg_json,
        ):
            # Wide enough, but granted for someone else's query. Re-grant, don't respawn —
            # and only when no concurrent pipeline is shuffling over these same workers.
            with contextlib.suppress(Exception):
                _regrant_fleet(_SESSION, credits, cfg_json)
        _SESSION_LEASES += 1
        return _SESSION


def release_session_lease() -> None:
    """Drop one borrow of the session fleet; re-arm the idle timer when the last one goes."""
    global _SESSION_LEASES
    from batcher.config import active_config

    with _SESSION_LOCK:
        if _SESSION_LEASES > 0:
            _SESSION_LEASES -= 1
        if _SESSION_LEASES == 0 and _SESSION is not None:
            _arm_idle_release(active_config().distributed.session_fleet_idle_s)


@contextlib.contextmanager
def session_fleet_lease():
    """Hold the session fleet for the lifetime of one distributed query.

    The per-operator lease (`acquire_fleet` / `release_fleet`) protects the fleet only
    while an operator is *running*. A staged query also needs it alive **between** stages:
    an intermediate left partitioned on the workers (a `FlightMaterializedSource`) is read
    in place by the next stage, so tearing the fleet down in the gap destroys the
    intermediate. This query-scoped lease holds the floor above zero for the whole run, so
    the idle timer can only fire once the query — not merely one operator — is done.

    Leasing before the fleet exists is fine and intended: the counter gates teardown, and
    the first operator to need a fleet spawns it under the already-held lease.

    This is also where the query's shuffle **plan id** is minted: the one scope that means
    "one query", and while the fleet under it may be shared with other pipelines, the id
    must not be.

    This lease is counted in `_SESSION_QUERY_LEASES` as well, because it must not veto the
    fleet resize it is held *across*: a query that finds the cached fleet too narrow has to
    be able to respawn it on its first acquire, and this lease is its own. See
    `_session_fleet_resizable`.
    """
    global _SESSION_LEASES, _SESSION_QUERY_LEASES, _SESSION_TIMER

    with _SESSION_LOCK:
        if _SESSION_TIMER is not None:
            _SESSION_TIMER.cancel()
            _SESSION_TIMER = None
        _SESSION_LEASES += 1
        _SESSION_QUERY_LEASES += 1
    try:
        with query_shuffle_scope():  # the fence, and the one place a query is counted
            yield
    finally:
        with _SESSION_LOCK:
            _SESSION_QUERY_LEASES = max(0, _SESSION_QUERY_LEASES - 1)
        release_session_lease()


def release_session_fleet() -> None:
    """Tear down the cached session fleet and release its cluster cores (idempotent).

    Called by the idle timer, and available to a caller that wants to free the cluster
    immediately (e.g. before handing it to another engine). A no-op when no fleet is
    cached, and — critically — when the fleet is still leased: killing the actors under a
    running query is what a naive time-since-acquire timer used to do.
    """
    global _SESSION, _SESSION_TIMER
    with _SESSION_LOCK:
        if _SESSION_TIMER is not None:
            _SESSION_TIMER.cancel()
            _SESSION_TIMER = None
        if _SESSION_LEASES > 0:
            return  # an operator is still shuffling over it — never kill mid-query
        if _SESSION is not None:
            with contextlib.suppress(Exception):
                _SESSION.cleanup()
            _SESSION = None


def acquire_fleet(workers: int, credits: int, cfg_json: str):
    """Borrow the query/session fleet, or spawn a transient one for this operator.

    Returns ``(actors, pg, addrs, workers, owns)``. Precedence:

    1. A query-lifetime fleet (the adaptive loop's ambient `ContextVar`) — every Flight
       operator MUST borrow it (``owns`` False); spawning its own placement group would
       contend with the fleet's held bundles and deadlock.
    2. The warm **session fleet** (when `reuse_session_fleet` is on) — reused across
       separate `collect()` calls so a short query skips the ~1-2s fleet spawn. Returned
       with ``owns`` False so the per-operator teardown leaves it warm for the next query.
    3. Otherwise spawn a transient fleet the caller tears down (``owns`` True) — the
       pre-existing per-operator path (single-node == distributed stays bit-identical).
    """
    fleet = current_fleet()
    if fleet is not None:
        # Re-assert this operator's plan id on the driver, so its tree-combine tickets
        # fence to the same query the workers are publishing under. Prefer the id minted
        # for *this query* over the fleet's spawn-time one: a shared fleet's id is common
        # to every pipeline borrowing it, which is exactly the collision we are avoiding.
        adopt_plan_id(fleet.plan_id)
        return fleet.actors, fleet.pg, fleet.addrs, fleet.workers, False

    from batcher.config import active_config

    if active_config().distributed.reuse_session_fleet:
        session = _acquire_session_fleet(workers, credits, cfg_json)
        adopt_plan_id(session.plan_id)
        return session.actors, session.pg, session.addrs, session.workers, False

    actors, pg, addrs = _spawn_fleet_with_addrs(workers, credits, cfg_json)
    return actors, pg, addrs, workers, True


def release_fleet(actors, pg, owns: bool) -> None:
    """The teardown paired with `acquire_fleet` — every Flight operator's ``finally``.

    Mirrors the three acquisition paths: a transient fleet (``owns``) is killed and its
    placement group released; a borrowed **query** fleet is left alone (the adaptive loop
    owns its lifetime); a borrowed **session** fleet has its lease dropped, which re-arms
    the idle timer only once no operator is still using it.
    """
    if owns:
        import ray

        from batcher.dist.executors.ray_runtime import release_placement

        for a in actors:
            with contextlib.suppress(Exception):
                ray.kill(a)
        release_placement(pg)
    elif current_fleet() is None:
        release_session_lease()  # borrowed the session fleet — hand the lease back


def current_fleet() -> ShuffleFleet | None:
    """The shuffle fleet in force for the current adaptive query, if any."""
    return _FLEET.get()


def set_fleet(fleet: ShuffleFleet | None) -> contextvars.Token:
    """Install `fleet` as the ambient fleet; returns a token to `reset` it after."""
    return _FLEET.set(fleet)


def reset_fleet(token: contextvars.Token) -> None:
    _FLEET.reset(token)
