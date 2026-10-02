"""Bringing a shuffle fleet up: gang-schedule the workers, collect their addresses, report.

Split out of `_fleet`, which keeps what happens to a fleet once it exists (reuse across
queries, leases, re-grants, release). Everything here runs once per spawn: reserve the
placement group, start the `_FlightWorker` actors, wait (bounded) for each to advertise its
Flight address, release the gang on any failure, tell same-node workers about each other, and
say why a fleet came up smaller than asked for.
"""

from __future__ import annotations

import contextlib
import logging

from batcher._internal.errors import ResourceError
from batcher._internal.logging import get_logger, log_kv, note_suppressed

__all__ = [
    "_fleet_demand_reason",
    "_spawn_fleet_with_addrs",
    "_tell_workers_about_node_peers",
    "_warn_degraded_fleet",
]


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
        _tell_workers_about_node_peers(actors, addrs)
        ok = True
        return actors, pg, addrs
    finally:
        if not ok:
            for a in actors:
                with contextlib.suppress(Exception):
                    ray.kill(a)
            release_placement(pg)


def _tell_workers_about_node_peers(actors, addrs) -> None:
    """Tell each worker whether another worker landed on its node.

    A shuffle address is `{node_ip}:{port}`, so two workers share a node exactly when their
    addresses' hosts match — the fleet already has every address by this point, and nothing
    else in the system does. The workers use it to skip mirroring buckets into shared memory
    when no other process on the node could read one (`ShuffleSession._shm_mirror_ok`), which
    is the ordinary shape of a fleet of small nodes: one worker per node, so every mirrored
    file is written, never read, and unlinked.

    Best-effort and one round-trip per fleet, not per query. A failure here leaves every
    worker at its default of "assume a peer", which is exactly today's behaviour, so this can
    only ever remove wasted work — never correctness, since a missing mirror already falls
    back to Flight.
    """
    import ray

    from batcher.carbonite.transfer.lifecycle import host_of

    try:
        seen: dict[str, int] = {}
        hosts = [host_of(a) for a in addrs]
        for h in hosts:
            seen[h] = seen.get(h, 0) + 1
        ray.get(
            [
                actor.set_shm_peers.remote(seen.get(host, 0) > 1)
                for actor, host in zip(actors, hosts, strict=True)
            ]
        )
    except Exception as exc:
        note_suppressed("dist", "tell the fleet whether its workers share nodes", exc)


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
    except Exception as exc:  # a diagnosis never replaces the failure
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
        log_kv(
            get_logger("dist"),
            logging.WARNING,
            "shuffle fleet came up narrower than requested; the query runs at reduced width",
            placed=placed,
            requested=wanted,
            waited_s=round(timeout * 2, 1),
            reason=_fleet_demand_reason() or "workers did not advertise in time",
        )
    except Exception as exc:  # observation must never fail a spawn
        note_suppressed("dist", "report the degraded fleet", exc)
