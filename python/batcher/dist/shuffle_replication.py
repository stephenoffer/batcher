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

Scope: this serves the flat aggregate reduce. A wide shuffle (``workers > fan_in``)
reduces through the combiner tree, which does not thread replicas yet and still degrades
to recompute.
"""

from __future__ import annotations

import contextlib

from batcher._internal.logging import note_suppressed
from batcher.dist.flight_worker import current_plan_id

__all__ = ["replicate_shuffle_output"]


def replicate_shuffle_output(actors, addrs, n_reducers, workers, dead):
    """Place a second copy of every mapper's buckets on an off-node survivor.

    Best-effort by construction: anything that fails leaves that source unreplicated and
    it degrades to the recompute path, so replication can never make a query fail.

    Args:
        actors: The worker actor handles, indexed by worker id.
        addrs: ``addrs[src]`` is the Flight address source `src` published its buckets on.
        n_reducers: Bucket count per mapper, so a replica copies every one of them.
        workers: Live worker count.
        dead: Workers already known gone; never given a copy to hold.

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
    except Exception:  # replication is an optimization; a probe failure keeps recompute
        return None

    index_of = {a: i for i, a in enumerate(worker_addrs)}
    # A source recovered during the map barrier lives on a different worker than its
    # index, so resolve the primary from the address it actually published on.
    primaries = {src: index_of[a] for src, a in enumerate(addrs) if a in index_of}
    if not primaries:
        return None

    assignment = assign_replica_hosts(primaries, nodes, factor, frozenset(dead or ()))
    refs: dict[tuple[int, int], object] = {}
    for src, hosts in assignment.items():
        for host in hosts:
            with contextlib.suppress(Exception):
                refs[(src, host)] = actors[host].replicate_buckets.remote(
                    addrs[src], src, n_reducers, 0, 0, current_plan_id()
                )

    replicas: list[list[str]] = [[] for _ in range(len(addrs))]
    for (src, _host), ref in refs.items():
        try:
            replicas[src].append(ray.get(ref))
        except Exception as exc:  # unacked ⇒ never advertised; that source keeps recompute
            note_suppressed("dist", "collect a replica acknowledgement", exc)
            continue
    return replicas if any(replicas) else None
