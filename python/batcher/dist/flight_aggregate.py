"""Distributed aggregation over an Arrow Flight shuffle (object store bypassed).

The disk-shuffle path (`executor._distributed_aggregate`) routes partial state
through Arrow-IPC files. This path instead runs long-lived Ray actors that each
host a node-local Flight server: mappers PUBLISH their hash-partitioned partials
on their own server and only advertise their `addr`; reducers FETCH their bucket
from every mapper over credit-bounded Flight streaming. Only addresses + tickets
(and the small finalized results) ever transit Ray — no `RecordBatch` becomes a
Ray object, and the heavy shuffle never touches the object store. This is the
true multi-node data plane the architecture calls for; it runs cross-process on
one host (Ray local mode) exactly as it would cross-node.
"""

from __future__ import annotations

import contextlib
import json

import pyarrow as pa

from batcher.carbonite import ResourceManager
from batcher.dist.executor import (
    _apply_above,
    _empty_agg_table,
    _ensure_ray,
    _relabel_single_source,
)
from batcher.dist.executors.partition_io import partition_descriptors
from batcher.dist.executors.ray_runtime import engine_config_json, release_placement
from batcher.dist.flight_worker import _ticket
from batcher.io.source import Source
from batcher.plan.logical import Aggregate, LogicalPlan

__all__ = ["execute_aggregate_flight"]


def _shuffle_credits(requested: int = 0) -> int:
    """The credit window Carbonite grants for this shuffle's reducer<-mapper channels.

    One credit = one in-flight `RecordBatch`, so the window bounds each channel's
    buffered memory. Decided once on the driver (control plane) and shipped to the
    actors as a plain int — no per-row work crosses Ray.

    When the execution carries a `SchedulingEnvelope`, its `credits` field is the
    window Carbonite already granted from the operator's measured/estimated
    `c_max_credits` — so the shuffle starts metadata-driven instead of from a blind
    default. Otherwise fall back to a fresh grant of `requested`.
    """
    from batcher.dist.executors.ray_runtime import current_envelope

    env = current_envelope()
    if env is not None and env.credits > 0:
        return env.credits
    return ResourceManager().grant_credits(requested)


def _shuffle_fan_in() -> int:
    """Carbonite's bound on how many upstreams a shuffle node fans in.

    A reduce over more than this many partials becomes a tree of combiner stages so
    no node ever reads from more than `shuffle_fan_in` upstreams at once. A floor of
    2 (a binary tree) is enforced here so a misconfigured value can't make the tree
    degenerate or fail to converge — it is a hard minimum, not a separate knob.
    """
    from batcher.config import active_config

    return max(2, active_config().flow_control.shuffle_fan_in)


def execute_aggregate_flight(
    above: list[LogicalPlan],
    agg: Aggregate,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    materialize: bool = True,
):
    """Distributed aggregation over a Flight shuffle, resilient to worker loss.

    A lost worker's shuffle output is recomputed from its source partition (still on
    disk) on a surviving worker and the reducers retry — Spark-style lineage
    recovery, coordinated by Carbonite's `ShuffleRecovery`. `_fault_inject` is a
    test-only hook: the worker ids to kill after the map barrier to exercise the
    recovery path.

    `materialize=False` (with no post-aggregate operators, on the flat-reduce path)
    keeps the result on the worker actors and returns a `FlightMaterializedSource`:
    each reducer publishes its finalized bucket on its own Flight server and the
    driver gets only `(addr, ticket, rows)` handles, so the next adaptive stage fetches
    the intermediate shared-nothing instead of collecting it through the driver. The
    actors stay alive (the source owns them) until its `cleanup()`. The wide
    tree-reduce path still collects (returns a table); the caller handles either.
    """
    import ray

    from batcher.dist.fleet import acquire_fleet

    _ensure_ray(workers)

    gk = json.dumps([{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys])
    aj = json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])
    map_plan, sid = _relabel_single_source(agg.input)
    map_ir = json.dumps(map_plan.to_ir())
    n_keys = len(agg.group_keys)

    # Carbonite grants the credit window once on the driver and the locality-aware
    # ShuffleSession on each actor uses it (and skips the network for same-process
    # buckets); both are decided here in the control plane. Only consulted when this
    # call spawns its own fleet — a borrowed fleet carries the grant it was spawned with.
    credits = _shuffle_credits()
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    # Borrow the query-lifetime fleet if the adaptive loop installed one (pins the
    # worker count to the fleet's, so every stage shuffles over the same actors);
    # otherwise spawn one we tear down. `owns` gates teardown.
    actors, pg, addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_reducers = 1 if n_keys == 0 else workers

    keep_actors = False  # set when a FlightMaterializedSource takes ownership of them
    try:
        partitions = partition_descriptors(sources[sid], workers)

        # MAP barrier: every mapper publishes ALL its buckets on its own Flight server.
        ray.get(
            [
                actors[i].map_publish.remote(map_ir, gk, aj, partitions[i], n_keys, n_reducers)
                for i in range(workers)
            ]
        )

        # Simulate worker loss after the map barrier (test hook): the killed workers'
        # published buckets vanish, so the reduce must recompute them.
        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        # A wide shuffle (more upstreams than the fan-in bound) reduces through a
        # combiner tree so no node fans in more than `fan_in` streams; a small one
        # uses the flat reduce. Both carry lineage recompute on worker loss.
        fan_in = _shuffle_fan_in()
        # Locality-aware reducer placement (opt-in): host each reducer where its bucket
        # concentrates, so its fetches become same-node hits. None ⇒ default round-robin.
        reducer_hosts = _locality_reducer_hosts(actors, n_reducers, workers)
        if workers > fan_in:
            batches = _tree_reduce_with_recovery(
                actors, list(addrs), partitions, map_ir, gk, aj, n_keys, n_reducers, fan_in, workers
            )
        elif materialize is False and not above:
            # Keep the result on the actors: each reducer publishes its bucket and the
            # driver gets only handles. The actors stay alive (the source owns them).
            from batcher.dist.fleet import FlightMaterializedSource

            handles = _reduce_with_recovery(
                actors,
                list(addrs),
                partitions,
                map_ir,
                gk,
                aj,
                n_keys,
                n_reducers,
                workers,
                materialize=True,
                reducer_hosts=reducer_hosts,
            )
            schema = handles[0][3] if handles else _empty_agg_table(agg).schema
            src_handles = [(a, t, n) for a, t, n, _s in handles]
            keep_actors = True
            # A borrowed fleet outlives this stage and is freed once by the adaptive
            # loop, so the source must NOT own the actors/pg (its `cleanup()` no-ops);
            # only a self-spawned fleet is handed to the source to tear down.
            src_actors = actors if owns else None
            src_pg = pg if owns else None
            return FlightMaterializedSource(src_handles, schema, src_actors, src_pg)
        else:
            batches = _reduce_with_recovery(
                actors,
                list(addrs),
                partitions,
                map_ir,
                gk,
                aj,
                n_keys,
                n_reducers,
                workers,
                reducer_hosts=reducer_hosts,
            )
    finally:
        # Only tear down a fleet we spawned; a borrowed one is the query's, freed once
        # by the adaptive loop. `keep_actors` further defers a self-spawned fleet to
        # the FlightMaterializedSource that took ownership of it.
        if owns and not keep_actors:
            for a in actors:
                with contextlib.suppress(Exception):
                    ray.kill(a)
            release_placement(pg)

    table = pa.Table.from_batches(batches) if batches else _empty_agg_table(agg)
    return table if not above else _apply_above(above, table)


def _locality_reducer_hosts(actors, n_reducers, workers):
    """Host-actor index per reducer, placing each where its bucket's bytes concentrate
    (locality-aware scheduling), or ``None`` to keep the default round-robin.

    ``None`` when the feature is off, when nothing is concentrated (an evenly-spread
    shuffle), or on any error — so the reduce path is unchanged in the common case.
    Result-preserving: which actor hosts a reducer never changes the output.
    """
    from batcher.config import active_config

    if not active_config().distributed.locality_aware_scheduling:
        return None

    import ray

    from batcher.carbonite.transfer.placement import assign_reducer_hosts, reducer_affinity

    try:
        nodes = ray.get([actors[i].node_id.remote() for i in range(workers)])
        per_mapper = ray.get([actors[i].published_bucket_bytes.remote() for i in range(workers)])
    except Exception:  # locality is best-effort; a probe failure keeps default placement
        return None
    bucket_node_bytes: dict[int, dict[str, int]] = {}
    for i, sizes in enumerate(per_mapper):
        node = nodes[i]
        for r, nbytes in sizes.items():
            bucket_node_bytes.setdefault(r, {})[node] = (
                bucket_node_bytes.setdefault(r, {}).get(node, 0) + nbytes
            )
    affinity = reducer_affinity(bucket_node_bytes)
    if not affinity:
        return None  # nothing concentrated ⇒ default placement is as good
    return assign_reducer_hosts(n_reducers, nodes, affinity)


def _reduce_with_recovery(
    actors,
    mapper_addrs,
    partitions,
    map_ir,
    gk,
    aj,
    n_keys,
    n_reducers,
    workers,
    *,
    materialize=False,
    reducer_hosts=None,
):
    """Run the reduce stage under Carbonite recompute-on-worker-loss recovery.

    Reducers are hosted on live workers; a reducer that reports an unreachable
    mapper (or whose host actor has died) drives a recompute of the lost source
    partition on a surviving worker, then a retry. Returns the finalized batches, or —
    when `materialize` — the `(addr, ticket, rows, schema)` handles of each reducer's
    bucket left published on its host actor's Flight server.
    """
    import ray

    from batcher._internal.errors import ResourceError
    from batcher.carbonite.resilience import ShuffleLineage, ShuffleRecovery, gather_with_backups
    from batcher.dist.executors.ray_runtime import (
        draining_workers,
        recovery_policy,
        speculation_policy,
    )

    dead: set[int] = set()
    # Per-source lineage: a recompute `reincarnate()`s the source to the next epoch,
    # so the regenerated partition is published *and* fetched under a fresh ticket and
    # a zombie worker's stale partial can never be read. Epoch 0 (no recompute) keeps
    # the tickets — and the clean-run behavior — bit-identical.
    lineage: dict[int, ShuffleLineage] = {}

    def _epochs() -> dict[int, int]:
        return {src: lin.epoch for src, lin in lineage.items()}

    def _pick_live(avoid: set[int]) -> int:
        for i in range(workers):
            if i not in dead and i not in avoid:
                return i
        raise ResourceError("no surviving worker to recover the shuffle on")

    # A reducer that returns "ok" fetched *all* its sources completely, so its result
    # is final and deterministic — cache it across recovery rounds (keyed by reducer
    # id) and never re-run it. Only pending (failed / not-yet-run) reducers re-launch,
    # so one lost mapper costs one recompute + the re-fetch of the affected reducers,
    # not a re-run of the whole reduce stage (the amplification that hurt most on a
    # churning spot/autoscaling cluster).
    done: dict[int, object] = {}

    def _host_for(r: int, avoid: set[int]) -> int:
        # The locality-aware host when given (a reducer placed near its data), else the
        # default `reducer r → actor r`; a dead/avoided host falls back to any survivor.
        # `avoid` lets a straggler's backup land on a *different* live worker than the
        # slow original (so the backup can actually win the race).
        h = reducer_hosts[r] if reducer_hosts is not None else r
        return h if h not in dead and h not in avoid else _pick_live(avoid)

    def attempt():
        failed = set()
        # Launch every *pending* reducer concurrently across the fleet, then collect via
        # `gather_with_backups`: a reducer that runs far slower than its peers (a
        # degraded-but-alive node — common on heterogeneous/spot clusters) gets a backup
        # copy on a different live worker, and the barrier takes whichever finishes first.
        # Deterministic reducers ⇒ the backup is byte-identical, so speculation changes
        # only *when* a bucket arrives, never *what* it holds. A reducer whose host dies
        # is classified as a lost host (recompute), exactly as the serial path did.
        epochs = _epochs()
        method = "reduce_fetch_publish" if materialize else "reduce_fetch"
        ref_host: dict[object, int] = {}

        def _launch(r: int, avoid: set[int]):
            host = _host_for(r, avoid)
            ref = getattr(actors[host], method).remote(gk, aj, mapper_addrs, r, epochs)
            ref_host[ref] = host
            return ref

        pending = [r for r in range(n_reducers) if r not in done]
        refs = [_launch(r, set()) for r in pending]

        def _relaunch(idx: int):
            # Back the straggler up on a *different* live worker; if it is the only
            # survivor, fall back to relaunching anywhere live (correct, just not faster).
            try:
                return _launch(pending[idx], {ref_host[refs[idx]]})
            except ResourceError:
                return _launch(pending[idx], set())

        def _on_failure(_idx: int, ref: object, _exc: Exception):
            return ("__dead__", ref_host.get(ref))  # the host of the last-failed copy

        results = gather_with_backups(refs, _relaunch, speculation_policy(), on_failure=_on_failure)
        for r, (status, payload) in zip(pending, results, strict=True):
            if status == "ok":
                done[r] = payload  # complete + deterministic → cache, never re-run
            elif status == "__dead__":
                if payload is not None:  # the reducer's host died — its mapper data too
                    dead.add(payload)
                    failed.add(payload)
            else:
                failed.update(payload)
        return [p for p in done.values() if p is not None], failed

    def recompute(failed_srcs):
        for src in failed_srcs:
            dead.add(src)  # an unreachable mapper means that worker is gone
            lineage[src] = lineage.get(src, ShuffleLineage(0, src)).reincarnate()
            target = _pick_live({src})
            mapper_addrs[src] = ray.get(
                actors[target].map_publish.remote(
                    map_ir, gk, aj, partitions[src], n_keys, n_reducers, src, lineage[src].epoch
                )
            )

    # Proactive spot-preemption migration: move any draining worker's mapper output to
    # a survivor *before* it is reclaimed, so a known-imminent loss costs no recovery
    # round (and no idle-timeout stall on a hung-but-draining peer). Best-effort — a
    # failure here just falls through to the reactive recompute the loop already does.
    proactive = draining_workers(actors, workers)
    if proactive:
        with contextlib.suppress(Exception):
            recompute(proactive)

    finals = ShuffleRecovery(recovery_policy()).run(attempt, recompute)
    if materialize:
        # Handles: (addr, ticket, rows, schema); empty buckets returned None (dropped).
        return [h for h in finals if h is not None]
    return [b for b in finals if b is not None and b.num_rows > 0]


def _tree_reduce(actors, leaf_addrs, n_reducers, gk, aj, fan_in, workers, dead=None):
    """Combine each bucket's `workers` leaf partials into one via a combiner tree.

    Each round groups a bucket's current partials into chunks of `fan_in`, and a
    *live* worker combines each chunk (at most `fan_in` fetches) and republishes the
    merged partial. After log_fan_in(workers) rounds one partial per bucket remains,
    which is finalized. No node ever reads from more than `fan_in` upstreams, so
    per-node fan-in stays bounded as the cluster grows to many thousands. Workers in
    `dead` are never assigned combine work (their leaf inputs are expected to have
    been recomputed onto a live worker's address in `leaf_addrs`). Returns the
    finalized batches. Raises if a combine touches a lost worker, so the caller's
    recovery loop can recompute and retry.
    """
    import ray

    dead = dead or set()
    live = [i for i in range(workers) if i not in dead]

    # frontier[r]: the (addr, ticket) sources currently holding bucket r's partials.
    frontier = {
        r: [(leaf_addrs[src], _ticket(0, src, r)) for src in range(workers)]
        for r in range(n_reducers)
    }
    stage = 1
    while any(len(srcs) > fan_in for srcs in frontier.values()):
        tasks, next_frontier, assign = [], {r: [] for r in range(n_reducers)}, 0
        for r in range(n_reducers):
            srcs = frontier[r]
            for i in range(0, len(srcs), fan_in):
                chunk = srcs[i : i + fan_in]
                if len(chunk) == 1:
                    next_frontier[r].append(chunk[0])  # nothing to combine yet
                    continue
                tasks.append((r, live[assign % len(live)], chunk, _ticket(stage, assign, r)))
                assign += 1
        new_addrs = ray.get(
            [
                actors[combiner].combine_publish.remote(gk, aj, chunk, out_ticket)
                for (_r, combiner, chunk, out_ticket) in tasks
            ]
        )
        for (r, _combiner, _chunk, out_ticket), addr in zip(tasks, new_addrs, strict=True):
            next_frontier[r].append((addr, out_ticket))
        frontier, stage = next_frontier, stage + 1

    # Final level: each bucket has <= fan_in sources — one combine+finalize per bucket.
    finals = ray.get(
        [
            actors[live[r % len(live)]].combine_finalize_fetch.remote(gk, aj, frontier[r])
            for r in range(n_reducers)
        ]
    )
    return [b for b in finals if b is not None and b.num_rows > 0]


def _tree_reduce_with_recovery(
    actors, leaf_addrs, partitions, map_ir, gk, aj, n_keys, n_reducers, fan_in, workers
):
    """Run the tree reduce under Carbonite recompute-on-worker-loss recovery.

    A lost worker takes its leaf partial (and any interior partials it held) with
    it. Recovery regenerates the lost leaf partition from its source (still on disk)
    onto a surviving worker and restarts the tree, which rebuilds every interior
    partial fresh — so a single bounded-fan-in mechanism is also fault-tolerant.
    """
    import ray

    from batcher.carbonite.resilience import ShuffleRecovery
    from batcher.dist.executors.ray_runtime import recovery_policy

    dead: set[int] = set()

    def _detect_dead():
        # Ping every live actor *concurrently* (one ray.get over all refs), not one
        # at a time — O(workers) serial RPCs each recovery round is slow on a big
        # cluster (N9). A ref that raises marks its actor dead.
        candidates = [i for i in range(workers) if i not in dead]
        refs = [actors[i].addr.remote() for i in candidates]
        for i, ref in zip(candidates, refs, strict=True):
            try:
                ray.get(ref)
            except ray.exceptions.RayActorError:
                dead.add(i)  # actor i is gone; its leaf partition (i) must be remade

    def attempt():
        try:
            return _tree_reduce(
                actors, leaf_addrs, n_reducers, gk, aj, fan_in, workers, dead
            ), set()
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            before = set(dead)
            _detect_dead()
            return None, (dead - before or {-1})  # -1: force a retry even if nothing new

    def recompute(failed):
        for src in (s for s in failed if isinstance(s, int) and s >= 0):
            target = next(j for j in range(workers) if j not in dead)
            leaf_addrs[src] = ray.get(
                actors[target].map_publish.remote(
                    map_ir, gk, aj, partitions[src], n_keys, n_reducers, src
                )
            )

    return ShuffleRecovery(recovery_policy()).run(attempt, recompute)
