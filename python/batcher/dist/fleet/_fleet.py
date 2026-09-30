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

from batcher._internal.logging import get_logger, log_kv, note_suppressed
from batcher.dist.fleet.plan_id import active_query_scopes, adopt_plan_id, query_shuffle_scope
from batcher.dist.fleet.spawn import (
    _spawn_fleet_with_addrs,
)

__all__ = [
    "ShuffleFleet",
    "acquire_fleet",
    "borrows_session_fleet",
    "current_fleet",
    "held_placement_group",
    "release_fleet",
    "release_session_fleet",
    "release_session_lease",
    "reset_fleet",
    "session_fleet_lease",
    "set_fleet",
    "yield_session_fleet",
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

#: Whether the *current* query's `session_fleet_lease` has taken its hold on the fleet yet.
#:
#: Tri-state, and all three states are distinct: `None` means no query lease is in scope at
#: all (a bare `acquire_fleet`, which owns its own lease and must not take a second one),
#: `False` means a query lease is open and has not yet taken its hold, `True` means it has.
#:
#: A ContextVar rather than a counter because the hold belongs to one query: `acquire_fleet`
#: promotes the lease of the query that called it, and a concurrent pipeline's lease is
#: unaffected.
_QUERY_HOLD: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "batcher_session_query_hold", default=None
)


class ShuffleFleet:
    """One placement group + `_FlightWorker` fleet reused across a query's stages.

    Holds the actors, their advertised Flight addresses, and the grant (credits +
    engine config) they were spawned with, so a borrowing operator runs every stage
    against the *same* fleet with the *same* worker count. `cleanup()` is the single
    teardown point — the adaptive loop calls it once, in its `finally`.
    """

    __slots__ = ("actors", "addrs", "cfg_json", "credits", "num_cpus", "pg", "plan_id", "session")

    def __init__(
        self,
        actors,
        pg,
        addrs,
        credits: int,
        cfg_json: str,
        plan_id: int,
        num_cpus: float = 0.0,
        session: str | None = None,
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
        # The Ray session the actors and placement group live in. A warm fleet outlives the
        # query that spawned it, and so can outlive the *cluster*: after a reconnect its
        # placement group names a reservation the new cluster never had, and a stage placed
        # into it sits in Ray's infeasible queue forever. See `_from_this_session`.
        self.session = session

    @property
    def workers(self) -> int:
        """The fixed worker count for the whole query (the fleet's actor count)."""
        return len(self.actors)

    @classmethod
    def spawn(cls, workers: int, credits: int, cfg_json: str) -> ShuffleFleet:
        """Gang-schedule `workers` actors once and cache their advertised addresses."""
        from batcher.dist.flight_worker import new_plan_id

        plan_id = new_plan_id()
        from batcher.dist.executors.ray_runtime.scheduling import ray_session_key

        actors, pg, addrs = _spawn_fleet_with_addrs(workers, credits, cfg_json, plan_id)
        return cls(
            actors, pg, addrs, credits, cfg_json, plan_id, _wanted_grant(), ray_session_key()
        )

    def cleanup(self) -> None:
        """Kill the fleet's actors and release its placement group (idempotent).

        A fleet from a Ray session this driver has left is only forgotten. Its handles name
        the old cluster's actors and reservation, and acting on them after a reconnect sends
        `kill`/`remove_placement_group` to the *new* cluster: the idle-release timer armed in
        one session fires in the next, which is how a stale teardown landed on a live GCS.
        """
        import ray

        from batcher.dist.executors.ray_runtime import release_placement

        if not _from_this_session(self):
            self.actors, self.pg = [], None
            return
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


def _from_this_session(fleet: ShuffleFleet) -> bool:
    """Whether `fleet` was spawned in the Ray session this driver is attached to now.

    `False` only on positive evidence of a different session. A fleet whose session could
    not be read at spawn keeps the behaviour it had before the stamp existed, and the
    liveness ping in `_acquire_session_fleet` still guards it.
    """
    stamp = getattr(fleet, "session", None)  # a handle-shaped stand-in carries no stamp
    if stamp is None:
        return True
    from batcher.dist.executors.ray_runtime.scheduling import ray_session_key

    return stamp == ray_session_key()


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

    The config is re-granted **per worker** on a fleet whose nodes are unequal, for the same
    reason the spawn ships one per worker: a re-grant carrying the uniform config would undo
    the per-node sizing on every reuse, so the *first* query on a warm fleet would use the big
    nodes and every later one would not. A uniform fleet re-grants `cfg_json` itself, which is
    what this always did.
    """
    import ray

    from batcher.dist.executors.ray_runtime import current_envelope
    from batcher.dist.flight_worker import _slot_engine_configs

    cfgs = _slot_engine_configs(current_envelope(), len(fleet.actors), cfg_json, bool(fleet.pg))
    ray.get([a.set_grant.remote(credits, cfgs[i]) for i, a in enumerate(fleet.actors)])
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
        if _SESSION is not None and not _from_this_session(_SESSION):
            # Spawned on a cluster this driver has since left. Dropped, not cleaned up: its
            # actor handles and placement group belong to that cluster, not this one.
            _SESSION = None
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
        # The query lease takes its hold here, on the first acquire, rather than on entry —
        # see `session_fleet_lease` for why holding it from entry deadlocks a query whose
        # operators are tasks rather than fleet actors.
        if _QUERY_HOLD.get() is False:
            _QUERY_HOLD.set(True)
            _SESSION_LEASES += 1
        return _SESSION


def borrow_warm_session_fleet() -> list | None:
    """The warm session fleet's actors, leased, or None when no fleet is warm; never spawns.

    For work that runs *on* the fleet's actors without shuffling, which is only worth doing
    when they already exist: the aligned executor's key-range units, which as plain tasks
    waited ~0.5 s per query for a worker lease and forced the warm fleet down to find cores
    (and the next shuffling query to pay a respawn). The lease is released with
    `release_session_lease`, like any other borrow.
    """
    global _SESSION_LEASES, _SESSION_TIMER

    with _SESSION_LOCK:
        if _SESSION is None or not _from_this_session(_SESSION) or _FLEET.get() is not None:
            return None
        if _SESSION_TIMER is not None:
            _SESSION_TIMER.cancel()
            _SESSION_TIMER = None
        _SESSION_LEASES += 1
        return list(_SESSION.actors)


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

    **It takes hold on the query's first `acquire_fleet`, not on entry**, and that
    distinction is the difference between a warm fleet and a deadlock. A warm fleet is a
    *placement-group reservation of the whole cluster* — measured here as four
    `_FlightWorker`s holding 24 cores each on a 96-core box — and a query that never shuffles
    still ran under this lease. So a broadcast join, whose probe is Ray *tasks* rather than
    fleet actors, cancelled the idle timer on entry and then waited forever for cores that
    only the timer it had just cancelled could free. Nothing errored: `ray.wait` inside
    `gather_with_backups`, `{'CPU': 24.0}: 8+ pending` against `96.0/96.0`, indefinitely.

    Deferring the hold costs nothing the docstring above promises. The purpose is to keep
    the fleet alive *between* a staged query's operators, and a query with no first operator
    on the fleet has no gap to protect; once one acquires, the hold is taken and every later
    gap is covered exactly as before.
    """
    global _SESSION_QUERY_LEASES

    with _SESSION_LOCK:
        _SESSION_QUERY_LEASES += 1
    held = _QUERY_HOLD.set(False)
    try:
        with query_shuffle_scope():  # the fence, and the one place a query is counted
            yield
    finally:
        promoted = _QUERY_HOLD.get() is True
        _QUERY_HOLD.reset(held)
        with _SESSION_LOCK:
            _SESSION_QUERY_LEASES = max(0, _SESSION_QUERY_LEASES - 1)
        if promoted:
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


def _free_cluster_cpus() -> float:
    """CPUs one node can hand a task right now, or `inf` when it cannot be read.

    Cores reserved inside a placement group count as *used* here even while the bundle sits
    idle, which is exactly the accounting `yield_session_fleet` needs: a warm fleet holding
    the whole cluster reads as zero free, because that is what a plain task sees.

    The most free on any **one** node, not the cluster's sum: a task is placed on a single
    node. A fleet holding 15 of each worker's 16 cores leaves eight free cores across eight
    workers, which the sum called room for an 8-CPU task that no node could take. The task
    then waited out the fleet's idle timer, 30 s on a query whose work took 0.3 s (TPC-H q10
    at SF1, aligned units of 8 CPUs).

    `inf` on failure, so an unreadable cluster never causes a teardown.
    """
    import ray

    try:
        from ray._private.state import available_resources_per_node

        per_node = available_resources_per_node()
        if per_node:
            return max(float(r.get("CPU", 0.0)) for r in per_node.values())
        return float(ray.available_resources().get("CPU", 0.0))
    except Exception as exc:
        note_suppressed("dist", "read the cluster's free CPU", exc)
        return float("inf")


def yield_session_fleet(needed_cpus: float) -> bool:
    """Release the warm fleet's cores for a stage whose work is Ray **tasks**, not actors.

    A fleet is a placement-group reservation of the cluster's *whole* CPU capacity (one
    worker per node holding that node's cores — see `_even_cpu_share`). A stage that runs
    as plain Ray tasks submits them **outside** that reservation, so when the fleet is up
    those tasks have nowhere to go and the barrier waits forever: `{'CPU': 0.125}: 1+
    pending` against `384.0/384.0`, no error, no timeout.

    `session_fleet_lease` already documents this deadlock and fixes the half it can see —
    a query that *never* shuffles no longer takes the hold on entry. This is the other
    half: a **staged** query whose first stage does shuffle takes the hold legitimately,
    and then its next stage is a map. Reproduced single-process on
    `tests/integration/test_distributed.py`, where it hung indefinitely.

    Yielding is safe under exactly the condition a respawn is (`_session_fleet_resizable`):
    no operator is mid-shuffle and nothing is published on the actors that a teardown would
    destroy. The next `acquire_fleet` respawns transparently, so the cost of yielding
    unnecessarily is one fleet spawn — which is why it is asked only when the cluster
    genuinely cannot place the caller's largest task.

    Args:
        needed_cpus: The largest single CPU ask among the tasks about to be submitted. A
            task can never run while the cluster's free CPU is below this, however long
            the barrier waits.

    Returns:
        Whether the fleet was torn down.
    """
    global _SESSION, _SESSION_TIMER

    if needed_cpus <= 0:
        return False
    with _SESSION_LOCK:
        if _SESSION is None:
            return False
        free = _free_cluster_cpus()
        if free >= needed_cpus:
            return False
        if not _session_fleet_resizable():
            # An intermediate published on these actors, or a second pipeline shuffling over
            # them, makes the teardown a wrong answer rather than a slow one — so the caller
            # keeps the fleet and runs inside its reservation instead
            # (`fleet_task_options`). Reported because it is the state a stall would be
            # explained by, and the lease counts are what distinguish the two cases.
            log_kv(
                get_logger("dist"),
                logging.DEBUG,
                "keeping the warm fleet: an intermediate is published on it",
                needed_cpus=needed_cpus,
                free_cpus=free,
                leases=_SESSION_LEASES,
                query_leases=_SESSION_QUERY_LEASES,
                query_scopes=active_query_scopes(),
            )
            return False
        fleet, _SESSION = _SESSION, None
        if _SESSION_TIMER is not None:
            _SESSION_TIMER.cancel()
            _SESSION_TIMER = None
    with contextlib.suppress(Exception):
        fleet.cleanup()
    log_kv(
        get_logger("dist"),
        logging.INFO,
        "released the warm shuffle fleet so this stage's tasks can be placed",
        needed_cpus=needed_cpus,
    )
    return True


def held_placement_group():
    """The placement group of whatever fleet this process is holding, or None.

    The query fleet first (the adaptive loop's ambient handle), then the warm session
    fleet. Read by a task stage that cannot be placed on the open cluster: the bundles keep
    a sliver free (`fleet_task_headroom`) so it can run inside the reservation instead of
    pending against it forever.

    Returns:
        A Ray placement group, or None when no fleet is up.
    """
    fleet = _FLEET.get() or _SESSION
    if fleet is None or not _from_this_session(fleet):
        return None
    return getattr(fleet, "pg", None)


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


def borrows_session_fleet() -> bool:
    """Whether the next `acquire_fleet` will take a lease on the warm **session** fleet.

    Must be asked *before* the acquire. `acquire_fleet` has three branches — borrow the
    adaptive loop's query fleet, borrow the session fleet, spawn a transient one — and only the
    middle one takes a lease that has to be handed back. `current_fleet()` separates the first
    from the other two, but only until the acquire installs one; asking afterwards is how a
    lease meant for one path gets released on another.

    A stage that *publishes* its output cannot hand the lease back in its own `finally` (that
    would arm the idle timer against an intermediate nothing has read yet), so it passes this
    to its `FlightMaterializedSource`, which returns it at `cleanup()`.

    Returns:
        True when this call site would be the one to take the session lease.
    """
    return current_fleet() is None


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
