"""Shuffle-output replication: turn a worker loss into a re-fetch, not a recompute.

A mapper publishes its buckets on its own Flight server, so losing that worker normally
forces a *recompute* — re-read its source partition from object storage and re-run the
map, usually the longest phase of the query. Placing a second copy on an off-node
survivor lets a reducer fetch the byte-identical bucket instead, at the cost of one extra
network copy.

That trade is affordable only because of the mergeable algebra: what a mapper publishes
is **pre-aggregated partial state**, typically far smaller than the source that produced
it, so copying it is much cheaper than regenerating it. Spark cannot make the same trade
— its shuffle carries raw rows, which is why it recomputes the map stage when a node
takes its shuffle files with it.

**The epoch invariant.** A replica is published under the ticket of the epoch it was
copied at. A recompute reincarnates its source to the *next* epoch, so the replica's
ticket no longer matches — and an unregistered ticket reads back as an **empty bucket
rather than an error**. A reducer allowed to fall back to a stale replica would therefore
silently drop that mapper's rows instead of failing. Two rules follow, and both are
load-bearing rather than defensive:

1. Advertise a replica only once its ``replicate_buckets`` call has **acked** (that ack is
   all-or-nothing across every bucket).
2. **Retire a source's replicas when it is recomputed** — done by the caller, which owns
   the lineage.

Scope: every Flight shuffle — aggregate (flat reduce and combiner tree), join, sort, and
window. Nothing here is aggregate-specific: a shuffle is a set of published buckets keyed
by ``(src, bucket)``, and that is all a replica copies, which is why one placement
function serves all four operators.
"""

from __future__ import annotations

import contextlib

from batcher._internal import events
from batcher._internal.logging import note_suppressed
from batcher.dist.flight_worker import current_plan_id

__all__ = ["replicate_shuffle_output", "retire_replicas"]


def retire_replicas(replicas, src: int, worker: int, shuffle: str) -> None:
    """Drop source `src`'s advertised replicas, before its output is recomputed.

    Every shuffle's recovery path must call this *before* it republishes a lost source.
    A replica was copied under the ticket of the epoch it was taken at; a recompute
    reincarnates the source to the next epoch, so the replica's ticket no longer
    resolves — and an unregistered ticket reads back as an **empty bucket rather than an
    error**. A reducer allowed to fall over to the stale copy would therefore silently
    drop that mapper's rows and return a wrong answer with nothing turning red. See the
    epoch invariant in this module's docstring.

    A no-op when replication is off (`replicas is None`), so a caller never has to guard.

    Args:
        replicas: The per-source replica lists, mutated in place, or `None`.
        src: The source whose output is about to be recomputed.
        worker: The worker that held it, for the recovery event.
        shuffle: Which shuffle this is (`aggregate`/`join`/`sort`/`window`), for the event.
    """
    if replicas is None or src >= len(replicas):
        return
    if replicas[src]:
        events.publish(
            events.RECOVERY,
            name=shuffle,
            event="replica_retired",
            shuffle=shuffle,
            src=src,
            worker=worker,
            replicas=len(replicas[src]),
        )
    replicas[src] = []


def replicate_shuffle_output(actors, addrs, n_reducers, workers, dead, stages=(0,)):
    """Place a second copy of every mapper's buckets on an off-node survivor.

    Best-effort by construction: anything that fails leaves that source unreplicated and
    it degrades to the recompute path, so replication can never make a query fail.

    Args:
        actors: The worker actor handles, indexed by worker id.
        addrs: ``addrs[src]`` is the Flight address source `src` published its buckets on.
        n_reducers: Bucket count per mapper, so a replica copies every one of them.
        workers: Live worker count.
        dead: Workers already known gone; never given a copy to hold.
        stages: The shuffle stages this operator published. Aggregate, sort and window
            each publish one (stage 0); a **join publishes two** — the left side on stage
            0 and the right on stage 1, under one address. A replica of a join mapper is
            only usable if it holds *both*, so every stage's ack is required before the
            host is advertised (see the all-or-nothing rule below).

    Returns:
        ``replicas[src] = [addr, ...]`` for the reduce gather to fall over to, or ``None``
        when replication is off (``shuffle_replication <= 1``), the cluster is too small
        to host an independent copy, or nothing could be placed.
    """
    from batcher.config import active_config

    factor = active_config().distributed.shuffle_replication
    if factor <= 1 or workers < 2:
        return None

    import ray

    from batcher.carbonite.resilience.replication import assign_replica_hosts

    try:
        nodes = ray.get([actors[i].node_id.remote() for i in range(workers)])
        worker_addrs = ray.get([actors[i].addr.remote() for i in range(workers)])
    except Exception as exc:  # replication is an optimization; a probe failure keeps recompute
        # Noted, not silent: a probe that always fails turns replication permanently off,
        # and every worker loss then pays a full map-stage recompute with nothing on the
        # record to say the cheaper path was never available.
        note_suppressed("dist", "probe workers for replica placement", exc)
        return None

    index_of = {a: i for i, a in enumerate(worker_addrs)}
    # A source recovered during the map barrier lives on a different worker than its
    # index, so resolve the primary from the address it actually published on.
    primaries = {src: index_of[a] for src, a in enumerate(addrs) if a in index_of}
    if not primaries:
        return None

    assignment = assign_replica_hosts(primaries, nodes, factor, frozenset(dead or ()))
    refs: dict[tuple[int, int], list[object]] = {}
    for src, hosts in assignment.items():
        for host in hosts:
            with contextlib.suppress(Exception):
                refs[(src, host)] = [
                    actors[host].replicate_buckets.remote(
                        addrs[src], src, n_reducers, stage, 0, current_plan_id()
                    )
                    for stage in stages
                ]

    replicas: list[list[str]] = [[] for _ in range(len(addrs))]
    if not refs:
        return None
    # Wait for every ack **together**, then read them. Each `ray.get` used to block in
    # turn, so the acks were collected serially — `workers x factor` sequential round trips
    # on the map barrier, the point of the query where the reduce is already waiting. One
    # `ray.wait` for all of them makes the waiting concurrent, and reading each ref
    # afterwards keeps the per-source error isolation the degradation story depends on: a
    # source whose replica never acked must keep recompute, not fail the query.
    pending = [ref for stage_refs in refs.values() for ref in stage_refs]
    with contextlib.suppress(Exception):
        ray.wait(pending, num_returns=len(pending))
    for (src, _host), stage_refs in refs.items():
        try:
            # All-or-nothing *across stages*, for the same reason `replicate_buckets` is
            # all-or-nothing across buckets: a join replica holding the left side but not
            # the right is a half-filled copy, and an unregistered ticket reads back as an
            # empty bucket rather than an error — so a reducer falling over to it would
            # silently emit an under-joined result. Every stage acks or the host is not
            # advertised at all. Each ack is this host's address, so any one of them names it.
            acks = [ray.get(ref) for ref in stage_refs]
        except Exception as exc:  # unacked ⇒ never advertised; that source keeps recompute
            note_suppressed("dist", "collect a replica acknowledgement", exc)
            continue
        if acks:
            replicas[src].append(acks[0])
    return replicas if any(replicas) else None
