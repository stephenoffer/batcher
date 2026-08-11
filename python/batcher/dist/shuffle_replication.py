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

**The tree's interior levels are replicated too**, by ``replicate_interior_outputs``. A wide
shuffle reduces through a combiner tree, and for a long time only its *leaves* carried
copies: an interior combiner's merged partial lived on exactly one node, so losing that node
threw away every level built so far and restarted the tree from the leaves — the recompute
the leaf replicas exist to avoid, reintroduced one level up. The interior copy is also the
cheapest one in the shuffle, because a level's output is the merge of ``fan_in`` partials and
is therefore the smallest state anywhere in the tree.

Interior replicas need no epoch fence, which is the one way they differ. A replica's hazard
is *staleness* — a ticket that outlives the data it named — and an interior ticket cannot go
stale within the attempt that made it: every level's fallbacks are built fresh inside a
single ``_tree_reduce`` call and no reference to them escapes it. A retry rebuilds the tree
from the leaves and re-advertises only what it just copied, so last attempt's interior copies
are unreachable rather than wrong.
"""

from __future__ import annotations

import contextlib

from batcher._internal import events
from batcher._internal.logging import note_suppressed
from batcher.dist.flight_worker import current_plan_id

__all__ = [
    "placement_probe",
    "replicate_interior_outputs",
    "replicate_shuffle_output",
    "retire_replicas",
]


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


def placement_probe(actors, workers):
    """What replica placement needs to know about the fleet, or `None` if it can't run.

    Returns `(nodes, index_of, suspect, factor, preemptible)`: the node each worker sits on,
    the worker index each shuffle address belongs to, the workers the fault ledger is
    quarantining, the configured replication factor, and the workers sitting on spot capacity.
    `None` when replication is off, the fleet is too small to hold an independent copy, or the
    probe itself failed.

    A replica exists to die independently of its primary, so putting the only spare copy on a
    worker the ledger has been quarantining defeats the whole point — that copy is the one
    least likely to be there when it is needed. `suspect` deprioritizes rather than excludes,
    so a fleet where most workers are suspect still gets its copies placed.

    `preemptible` is the same argument one failure domain out. A spot reclamation takes an
    instance group rather than a machine, so a copy on a second spot node dies in the same
    wave as its primary — on exactly the fleet the `spot` profile turns replication on for.
    """
    from batcher.config import active_config

    factor = active_config().distributed.shuffle_replication
    if factor <= 1 or workers < 2:
        return None

    import ray

    from batcher.dist.executors.ray_runtime.policies import node_ledger

    try:
        nodes = ray.get([actors[i].node_id.remote() for i in range(workers)])
        worker_addrs = ray.get([actors[i].addr.remote() for i in range(workers)])
    except Exception as exc:  # replication is an optimization; a probe failure keeps recompute
        # Noted, not silent: a probe that always fails turns replication permanently off,
        # and every worker loss then pays a full map-stage recompute with nothing on the
        # record to say the cheaper path was never available.
        note_suppressed("dist", "probe workers for replica placement", exc)
        return None

    ledger = node_ledger()
    blocked = ledger.blocked_keys() if ledger is not None else ()
    suspect = frozenset(int(k) for k in blocked if k.isdigit())
    return nodes, {a: i for i, a in enumerate(worker_addrs)}, suspect, factor, _spot_workers(nodes)


def _spot_workers(nodes) -> frozenset[int]:
    """Worker indices whose node is spot capacity, empty when that cannot be read.

    Read from the driver's own topology rather than probed on each worker: it is one label
    lookup the driver has already paid for elsewhere, against `workers` extra round trips.
    Empty on any failure, which ranks every candidate the same and leaves the placement
    exactly as it was — a market label that cannot be read is not evidence about the fleet.
    """
    try:
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        spot_nodes = {n["node_id"] for n in node_classes() if n.get("preemptible")}
    except Exception as exc:
        note_suppressed("dist", "read node market types for replica placement", exc)
        return frozenset()
    return frozenset(i for i, node in enumerate(nodes) if node in spot_nodes)


def replicate_interior_outputs(actors, outputs, workers, dead, probe=None):
    """Place a second copy of each combiner-tree interior partial on an off-node survivor.

    One level of the tree at a time. `outputs[i]` is the `(addr, ticket)` the i-th combine
    task of this level published, and the returned `fallbacks[i]` are the addresses holding a
    copy of it — positionally, because that is how `_combine_sources` indexes replicas.

    Without this, a tree's interior was single-copy however high `shuffle_replication` was
    set: losing one combiner discarded every level built so far and restarted from the
    leaves. The copy is cheap here in a way it is nowhere else in the shuffle — a level's
    output is `fan_in` partials already merged into one — and it is what makes a wide
    shuffle's fault tolerance match a narrow one's rather than degrade to recompute at the
    first level.

    Best-effort throughout: an output with no acked copy simply gets an empty fallback list
    and keeps the recompute path, so this can never fail a query.

    Args:
        actors: The worker actor handles, indexed by worker id.
        outputs: `(addr, ticket)` per combine task of this level, in task order.
        workers: Live worker count.
        dead: Workers already known gone; never given a copy to hold.

    Returns:
        `fallbacks[i] = [addr, ...]` positional over `outputs`, or `None` when replication is
        off or nothing could be placed.
    """
    probe = placement_probe(actors, workers) if probe is None else probe
    if probe is None or not outputs:
        return None
    nodes, index_of, suspect, factor, spot = probe

    import ray

    from batcher.carbonite.resilience.replication import assign_replica_hosts

    # The "source" here is the combine task, and its primary is the worker that published it.
    primaries = {i: index_of[addr] for i, (addr, _t) in enumerate(outputs) if addr in index_of}
    if not primaries:
        return None
    assignment = assign_replica_hosts(
        primaries, nodes, factor, frozenset(dead or ()), suspect, spot
    )

    refs: dict[tuple[int, int], object] = {}
    for i, hosts in assignment.items():
        addr, ticket = outputs[i]
        for host in hosts:
            with contextlib.suppress(Exception):
                refs[(i, host)] = actors[host].replicate_tickets.remote(
                    addr, [ticket], current_plan_id()
                )

    if not refs:
        return None
    fallbacks: list[list[str]] = [[] for _ in outputs]
    with contextlib.suppress(Exception):
        ray.wait(list(refs.values()), num_returns=len(refs))
    for (i, _host), ref in refs.items():
        try:
            fallbacks[i].append(ray.get(ref))
        except Exception as exc:  # unacked ⇒ never advertised; that partial keeps recompute
            note_suppressed("dist", "collect an interior replica acknowledgement", exc)
    return fallbacks if any(fallbacks) else None


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
    probe = placement_probe(actors, workers)
    if probe is None:
        return None
    nodes, index_of, suspect, factor, spot = probe

    import ray

    from batcher.carbonite.resilience.replication import assign_replica_hosts

    # A source recovered during the map barrier lives on a different worker than its
    # index, so resolve the primary from the address it actually published on.
    primaries = {src: index_of[a] for src, a in enumerate(addrs) if a in index_of}
    if not primaries:
        return None

    assignment = assign_replica_hosts(
        primaries, nodes, factor, frozenset(dead or ()), suspect, spot
    )
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
