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
import threading

__all__ = [
    "ShuffleFleet",
    "acquire_fleet",
    "current_fleet",
    "maybe_spawn_query_fleet",
    "release_session_fleet",
    "reset_fleet",
    "set_fleet",
]

# The shuffle fleet in force for the current adaptive query, if any. Ambient so a
# Flight operator borrows it without it being threaded through every call.
_FLEET: contextvars.ContextVar[ShuffleFleet | None] = contextvars.ContextVar(
    "batcher_shuffle_fleet", default=None
)


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

    from batcher.dist.executors.ray_runtime import release_placement
    from batcher.dist.flight_worker import spawn_flight_workers

    actors, pg = spawn_flight_workers(workers, credits, cfg_json, plan_id)
    ok = False
    try:
        addrs = list(ray.get([a.addr.remote() for a in actors]))
        ok = True
        return actors, pg, addrs
    finally:
        if not ok:
            for a in actors:
                with contextlib.suppress(Exception):
                    ray.kill(a)
            release_placement(pg)


class ShuffleFleet:
    """One placement group + `_FlightWorker` fleet reused across a query's stages.

    Holds the actors, their advertised Flight addresses, and the grant (credits +
    engine config) they were spawned with, so a borrowing operator runs every stage
    against the *same* fleet with the *same* worker count. `cleanup()` is the single
    teardown point — the adaptive loop calls it once, in its `finally`.
    """

    __slots__ = ("actors", "addrs", "cfg_json", "credits", "pg", "plan_id")

    def __init__(self, actors, pg, addrs, credits: int, cfg_json: str, plan_id: int) -> None:
        self.actors = actors
        self.pg = pg
        self.addrs = addrs
        self.credits = credits
        self.cfg_json = cfg_json
        # The query's shuffle plan id, set on the driver whenever this fleet is
        # borrowed so every borrowing operator's tickets fence to this query.
        self.plan_id = plan_id

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
        return cls(actors, pg, addrs, credits, cfg_json, plan_id)

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


def _acquire_session_fleet(workers: int, credits: int, cfg_json: str) -> ShuffleFleet:
    """Get the warm session fleet, spawning (or respawning a dead one) as needed.

    The cached fleet's worker count wins for the whole session (a borrowing operator
    fans out over the actors that exist), so a later query with a different data-driven
    `workers` reuses the warm fleet instead of churning a new placement group. A fleet
    whose actors died (preemption) is torn down and respawned transparently.
    """
    global _SESSION
    from batcher.config import active_config

    with _SESSION_LOCK:
        if _SESSION is not None and not _session_fleet_alive(_SESSION):
            with contextlib.suppress(Exception):
                _SESSION.cleanup()
            _SESSION = None
        if _SESSION is None:
            _SESSION = ShuffleFleet.spawn(workers, credits, cfg_json)
        _arm_idle_release(active_config().distributed.session_fleet_idle_s)
        return _SESSION


def release_session_fleet() -> None:
    """Tear down the cached session fleet and release its cluster cores (idempotent).

    Called by the idle timer, and available to a caller that wants to free the cluster
    immediately (e.g. before handing it to another engine). A no-op when no fleet is
    cached.
    """
    global _SESSION, _SESSION_TIMER
    with _SESSION_LOCK:
        if _SESSION_TIMER is not None:
            _SESSION_TIMER.cancel()
            _SESSION_TIMER = None
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
        # Re-assert the borrowed fleet's plan id on the driver, so this operator's
        # tree-combine tickets fence to the same query the workers were spawned for.
        from batcher.dist.flight_worker import set_current_plan_id

        set_current_plan_id(fleet.plan_id)
        return fleet.actors, fleet.pg, fleet.addrs, fleet.workers, False

    from batcher.config import active_config

    if active_config().distributed.reuse_session_fleet:
        from batcher.dist.flight_worker import set_current_plan_id

        session = _acquire_session_fleet(workers, credits, cfg_json)
        set_current_plan_id(session.plan_id)
        return session.actors, session.pg, session.addrs, session.workers, False

    actors, pg, addrs = _spawn_fleet_with_addrs(workers, credits, cfg_json)
    return actors, pg, addrs, workers, True


def current_fleet() -> ShuffleFleet | None:
    """The shuffle fleet in force for the current adaptive query, if any."""
    return _FLEET.get()


def set_fleet(fleet: ShuffleFleet | None) -> contextvars.Token:
    """Install `fleet` as the ambient fleet; returns a token to `reset` it after."""
    return _FLEET.set(fleet)


def reset_fleet(token: contextvars.Token) -> None:
    _FLEET.reset(token)


def maybe_spawn_query_fleet(num_workers: int | None, transport: str) -> ShuffleFleet | None:
    """Spawn a query-lifetime fleet when the adaptive Flight path warrants one.

    Returns a `ShuffleFleet` only when `distributed.persistent_fleet` is enabled and
    the resolved transport is Flight on a genuine multi-worker cluster; otherwise
    `None`, so the caller leaves each operator to spawn its own fleet (the default,
    bit-identical path). The worker count is fixed here for the whole query so every
    stage shuffles over the same fleet.
    """
    from batcher.config import active_config

    if not active_config().distributed.persistent_fleet:
        return None

    import math
    import os

    from batcher.dist.executors.ray_runtime import (
        _ensure_ray,
        clamp_workers,
        engine_config_json,
        release_autoscale,
        request_autoscale,
        resolve_transport,
    )

    workers = num_workers or (os.cpu_count() or 4)
    _ensure_ray(workers)
    # Ask the autoscaler for the fleet's cores and wait (bounded) for them while sizing
    # it; release the request once sized — the spawned actors keep the nodes busy, so the
    # autoscaler never reclaims them under the fleet, and the floor needn't stay pinned.
    request_autoscale(math.ceil(workers))
    try:
        workers = clamp_workers(workers)
        if workers <= 1 or resolve_transport(transport, workers) != "flight":
            return None
        from batcher.dist.flight_aggregate import _shuffle_credits

        # An adaptive query fleet reserves its own placement group; free any warm
        # session fleet first so its held bundles can't deadlock the new gang request.
        release_session_fleet()
        return ShuffleFleet.spawn(workers, _shuffle_credits(), engine_config_json())
    finally:
        release_autoscale()
