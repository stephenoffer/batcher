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

from batcher._internal import events
from batcher._internal.logging import note_suppressed
from batcher.carbonite import ResourceManager
from batcher.dist.adaptive_sizing import aggregate_reducer_count, record_aggregate_cardinality
from batcher.dist.executor import (
    _apply_above,
    _empty_agg_table,
    _ensure_ray,
    _relabel_single_source,
)
from batcher.dist.executors.partition_io import consumer_pushdown, partition_descriptors
from batcher.dist.executors.ray_runtime import (
    engine_config_json,
    map_barrier,
    release_placement,
    shuffle_partitions,
)
from batcher.dist.fleet.plan_id import next_result_stage
from batcher.dist.flight_worker import _ticket, current_plan_id
from batcher.dist.shuffle_replication import replicate_shuffle_output, retire_replicas
from batcher.io.source import Source
from batcher.plan.ir_specs import agg_spec_json
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
    _fault_inject_map: set[int] | None = None,
):
    """Distributed aggregation over a Flight shuffle, resilient to worker loss.

    A lost worker's shuffle output is recomputed from its source partition (still on
    disk) on a surviving worker and the reducers retry — Spark-style lineage recovery.
    Worker loss is survived in *both* phases: `map_barrier` relocates a source whose
    worker dies while mapping, `ShuffleRecovery` one whose worker dies before the reduce
    fetches it. `_fault_inject` / `_fault_inject_map` are test-only hooks: the worker ids
    to kill after / before the map barrier.

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

    gk, aj = agg_spec_json(agg)
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
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_reducers = 1 if n_keys == 0 else aggregate_reducer_count(agg, shuffle_partitions(workers))

    keep_actors = False  # set when a FlightMaterializedSource takes ownership of them
    try:
        # Push the map prefix's projection + predicate into the per-worker read, so each
        # mapper reads ONLY the columns/rows its plan needs straight from object storage —
        # not the whole (wide) source. Without this a worker reads every column and the
        # `map_ir` project discards the surplus *after* paying to fetch it: TPC-H lineitem
        # has 16 columns but this agg needs 3, so the un-pushed read moved ~5x the bytes
        # and dominated the scan (sf10 map barrier 18s → ~4s). Mirrors single-node
        # projection/predicate pushdown; the `map_ir` filter re-checks, so pushdown is safe.
        # `map_plan`'s scan was relabeled to source 0, so key the analysis on 0, not on the
        # source's original index: a staged plan whose input is an intermediate (source id >
        # 0) missed the lookup and silently read every column.
        # Ask about the aggregate *over* the prefix, not the prefix alone. The prefix of a
        # plain `group_by(k).agg(sum(v))` is a bare `Scan`, and a bare scan requires every
        # column it has — so the projection came back as the source's full schema and the
        # pruning described above silently did nothing for the commonest aggregate shape (it
        # bit only when the user's own filter/project happened to sit in the prefix).
        # `consumer_pushdown` re-parents the aggregate so Kyber's analysis sees what the
        # reducers actually read.
        projection, predicate = consumer_pushdown(agg, map_plan)
        partitions = partition_descriptors(
            sources[sid],
            workers,
            projection=projection,
            predicate=predicate,
            worker_addrs=fleet_addrs,
        )

        if _fault_inject_map:  # test hook: kill before the barrier, so nothing publishes
            for i in _fault_inject_map:
                ray.kill(actors[i])

        # MAP barrier: every mapper publishes ALL its buckets on its own Flight server,
        # under worker-loss recovery. A spot preemption *here* — the map phase reads the
        # source from object storage and is usually the longest part of the query — would
        # fail the whole query on a bare `ray.get`; instead the lost worker's source is
        # republished on a survivor under the same `src`, so the reducers' tickets still
        # resolve, and `dead` keeps the reduce off a worker that is gone.
        addrs, dead = map_barrier(
            workers,
            lambda host, src: actors[host].map_publish.remote(
                map_ir, gk, aj, partitions[src], n_keys, n_reducers, src, 0, current_plan_id()
            ),
        )

        # Placed HERE, as soon as the buckets exist and before anything can take a worker
        # away — replicating after a loss would be probing a corpse. `None` (the default
        # factor of 1) leaves the reduce byte-identical to the unreplicated path.
        replicas = replicate_shuffle_output(actors, addrs, n_reducers, workers, dead)

        # Simulate worker loss after the map barrier (test hook): the killed workers'
        # published buckets vanish, so the reduce must recompute them — or, with
        # replication on, re-fetch them from the survivor holding the copy.
        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        # A wide shuffle (more upstreams than the fan-in bound) reduces through a
        # combiner tree so no node fans in more than `fan_in` streams; a small one
        # uses the flat reduce. Both carry lineage recompute on worker loss.
        fan_in = _shuffle_fan_in()
        # Locality-aware reducer placement (opt-in): host each reducer where its bucket
        # concentrates, so its fetches become same-node hits. None ⇒ default round-robin.
        reducer_hosts = _locality_reducer_hosts(actors, n_reducers, workers, fleet_addrs)
        reduce_args = (actors, addrs, partitions, map_ir, gk, aj, n_keys, n_reducers, workers)
        if workers > fan_in:
            batches = _tree_reduce_with_recovery(
                actors,
                addrs,
                partitions,
                map_ir,
                gk,
                aj,
                n_keys,
                n_reducers,
                fan_in,
                workers,
                dead,
                replicas,
            )
        else:
            # `on_actors`: keep the result on the workers — each reducer publishes its
            # bucket and the driver gets only handles, so the next adaptive stage reads
            # the intermediate in place. Otherwise the reducers return their batches.
            on_actors = materialize is False and not above
            out = _reduce_with_recovery(
                *reduce_args,
                materialize=on_actors,
                reducer_hosts=reducer_hosts,
                dead=dead,
                replicas=replicas,
            )
            if on_actors:
                from batcher.dist.fleet import FlightMaterializedSource

                schema = out[0][3] if out else _empty_agg_table(agg).schema
                keep_actors = True  # the source owns them now
                # A borrowed fleet outlives this stage and is freed once by the adaptive
                # loop, so the source must NOT own the actors/pg (its `cleanup()` no-ops);
                # only a self-spawned fleet is handed to the source to tear down.
                src_actors, src_pg = (actors, pg) if owns else (None, None)
                handles = [(a, t, n) for a, t, n, _s in out]
                return FlightMaterializedSource(handles, schema, src_actors, src_pg)
            batches = out
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
    record_aggregate_cardinality(agg, table.num_rows)
    return table if not above else _apply_above(above, table)


def _locality_reducer_hosts(actors, n_reducers, workers, fleet_addrs=None):
    """Host-actor index per reducer, placing each where its bucket's bytes concentrate
    (locality-aware scheduling), or ``None`` to keep the default round-robin.

    ``None`` when the feature is off, when the whole fleet is on one node (every fetch is
    already same-node, so placement has nothing to win), when nothing is concentrated (an
    evenly-spread shuffle), or on any error — so the reduce path is unchanged in the
    common case. Result-preserving: which actor hosts a reducer never changes the output.

    Node identity comes from the workers' advertised shuffle addresses when the caller has
    them, which costs nothing (the driver holds them already) and is the *same* identity
    `select_mode` routes on — so placement and transport agree on what "same node" means.
    Falling back to a `node_id` probe costs a round-trip per worker, and paid it even to
    discover a single-node fleet where the answer was always `None`.
    """
    from batcher.config import active_config

    if not active_config().distributed.locality_aware_scheduling:
        return None

    import ray

    from batcher.carbonite.transfer.lifecycle import host_of
    from batcher.carbonite.transfer.placement import assign_reducer_hosts, reducer_affinity

    try:
        if fleet_addrs and len(fleet_addrs) >= workers and all(fleet_addrs[:workers]):
            nodes = [host_of(a) for a in fleet_addrs[:workers]]
        else:
            nodes = ray.get([actors[i].node_id.remote() for i in range(workers)])
        if len(set(nodes)) <= 1:
            return None  # one node: every fetch is same-node already
        per_mapper = ray.get([actors[i].published_bucket_bytes.remote() for i in range(workers)])
    except Exception as exc:  # locality is best-effort; a probe failure keeps default placement
        note_suppressed("dist", "probe reducer host locality", exc)
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
    dead=None,
    replicas=None,
):
    """Run the reduce stage under Carbonite recompute-on-worker-loss recovery.

    Reducers are hosted on live workers; a reducer that reports an unreachable
    mapper (or whose host actor has died) drives a recompute of the lost source
    partition on a surviving worker, then a retry. Returns the finalized batches, or —
    when `materialize` — the `(addr, ticket, rows, schema)` handles of each reducer's
    bucket left published on its host actor's Flight server. `dead` seeds the workers the
    map barrier already lost, so no reducer is scheduled onto an actor that is gone.
    """
    import ray

    from batcher._internal.errors import ResourceError
    from batcher.carbonite.resilience import (
        ShuffleLineage,
        ShuffleRecovery,
        SourcePlacement,
        gather_with_backups,
    )
    from batcher.dist.executors.ray_runtime import (
        draining_workers,
        recovery_policy,
        speculation_policy,
    )

    dead: set[int] = set(dead or ())
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

    # Where each source's latest map output lives. Identity until a recompute relocates a
    # source, after which the source id and its host are different numbers — and it is the
    # HOST that dies. `map_barrier` keeps the same mapping for the same reason
    # (`ray_runtime/policies.py::_on_lost`).
    placement = SourcePlacement(workers)

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
        # One stage id for every bucket of THIS published result, so two materialized
        # intermediates of the same query cannot share a ticket (`next_result_stage`).
        result_stage = next_result_stage() if materialize else 0
        ref_host: dict[object, int] = {}

        def _launch(r: int, avoid: set[int]):
            host = _host_for(r, avoid)
            extra = (result_stage,) if materialize else ()
            ref = getattr(actors[host], method).remote(
                gk, aj, mapper_addrs, r, epochs, replicas, current_plan_id(), *extra
            )
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
                    # `payload` is a HOST id; `failed` carries SOURCE ids (the other branch
                    # reports unreachable sources). Translate through the current placement
                    # so a relocated source is recomputed and an unrelated one is not —
                    # on a clean run this is `{payload}`, exactly as before.
                    failed.update(placement.sources_on(payload))
            else:
                failed.update(payload)
        return [p for p in done.values() if p is not None], failed

    def recompute(failed_srcs):
        for src in failed_srcs:
            # The HOST holding `src` is what died — which is `src` itself only until this
            # source has been relocated once. Marking `src` unconditionally would re-mark an
            # already-dead worker and leave the real one live for `_pick_live`/`_host_for`
            # to hand out again, spending the recovery budget on a host that cannot answer.
            host = placement.host_of(src)
            dead.add(host)
            # Retire this source's replicas BEFORE reincarnating it: a stale replica holds
            # the old epoch's ticket, which reads back as an EMPTY bucket rather than an
            # error, so falling back to it would silently drop this mapper's rows. See the
            # epoch invariant in `dist/shuffle_replication.py`.
            retire_replicas(replicas, src, host, "aggregate")
            lineage[src] = lineage.get(src, ShuffleLineage(0, src)).reincarnate()
            target = _pick_live({host})
            placement.relocate(src, target)  # it lives here now, not on `src`
            mapper_addrs[src] = ray.get(
                actors[target].map_publish.remote(
                    map_ir,
                    gk,
                    aj,
                    partitions[src],
                    n_keys,
                    n_reducers,
                    src,
                    lineage[src].epoch,
                    current_plan_id(),
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

    finals = ShuffleRecovery(recovery_policy(), label="aggregate").run(attempt, recompute)
    if materialize:
        # Handles: (addr, ticket, rows, schema); empty buckets returned None (dropped).
        return [h for h in finals if h is not None]
    return [b for b in finals if b is not None and b.num_rows > 0]


def _tree_reduce(actors, leaf_addrs, n_reducers, gk, aj, fan_in, workers, dead=None, replicas=None):
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
    # fallbacks[r][i]: replica addresses for frontier[r][i], carried POSITIONALLY alongside
    # the frontier because `gather_combine` indexes replicas by source position, not by
    # worker id. Only the leaf level has any: an interior combiner's output is published on
    # a single node and is never replicated, so it contributes an empty list and a loss
    # there still costs a recompute round.
    fallbacks = {
        r: [
            list(replicas[src]) if replicas and src < len(replicas) else []
            for src in range(workers)
        ]
        for r in range(n_reducers)
    }
    stage = 1
    while any(len(srcs) > fan_in for srcs in frontier.values()):
        tasks, next_frontier, assign = [], {r: [] for r in range(n_reducers)}, 0
        next_fallbacks: dict[int, list[list[str]]] = {r: [] for r in range(n_reducers)}
        for r in range(n_reducers):
            srcs = frontier[r]
            for i in range(0, len(srcs), fan_in):
                chunk = srcs[i : i + fan_in]
                chunk_reps = fallbacks[r][i : i + fan_in]
                if len(chunk) == 1:
                    next_frontier[r].append(chunk[0])  # nothing to combine yet
                    next_fallbacks[r].append(chunk_reps[0])  # its replicas carry forward
                    continue
                tasks.append(
                    (r, live[assign % len(live)], chunk, _ticket(stage, assign, r), chunk_reps)
                )
                assign += 1
        new_addrs = ray.get(
            [
                actors[combiner].combine_publish.remote(gk, aj, chunk, out_ticket, chunk_reps)
                for (_r, combiner, chunk, out_ticket, chunk_reps) in tasks
            ]
        )
        for (r, _combiner, _chunk, out_ticket, _reps), addr in zip(tasks, new_addrs, strict=True):
            next_frontier[r].append((addr, out_ticket))
            next_fallbacks[r].append([])  # a combined partial exists on one node only
        frontier, fallbacks, stage = next_frontier, next_fallbacks, stage + 1

    # Final level: each bucket has <= fan_in sources — one combine+finalize per bucket.
    finals = ray.get(
        [
            actors[live[r % len(live)]].combine_finalize_fetch.remote(
                gk, aj, frontier[r], fallbacks[r]
            )
            for r in range(n_reducers)
        ]
    )
    return [b for b in finals if b is not None and b.num_rows > 0]


def _tree_reduce_with_recovery(
    actors,
    leaf_addrs,
    partitions,
    map_ir,
    gk,
    aj,
    n_keys,
    n_reducers,
    fan_in,
    workers,
    dead=None,
    replicas=None,
):
    """Run the tree reduce under Carbonite recompute-on-worker-loss recovery.

    A lost worker takes its leaf partial (and any interior partials it held) with
    it. Recovery regenerates the lost leaf partition from its source (still on disk)
    onto a surviving worker and restarts the tree, which rebuilds every interior
    partial fresh — so a single bounded-fan-in mechanism is also fault-tolerant. `dead`
    seeds the workers the map barrier already lost, so the tree never assigns them work.
    """
    import ray

    from batcher.carbonite.resilience import ShuffleRecovery
    from batcher.dist.executors.ray_runtime import (
        is_recoverable_task_failure,
        recovery_policy,
    )

    dead: set[int] = set(dead or ())

    def _detect_dead():
        # Ping every live actor *concurrently* (one ray.get over all refs), not one
        # at a time — O(workers) serial RPCs each recovery round is slow on a big
        # cluster. A ref that raises marks its actor dead.
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
                actors, leaf_addrs, n_reducers, gk, aj, fan_in, workers, dead, replicas
            ), set()
        except ray.exceptions.RayTaskError as exc:
            # A combine fetches inside its task, so a lost peer arrives here as a
            # `RetryableShuffleError` wrapped in a `RayTaskError` — the same type a user's
            # failing UDF produces. Only the transport-classified ones are recoverable;
            # anything else is deterministic, and retrying it would spend the whole recovery
            # budget re-running a bug and then bury the real traceback under
            # `ResourceError("shuffle did not recover...")`. Re-raise so the user sees the
            # actual exception. Mirrors `gather_map_results`, which re-raises `RayTaskError`
            # outright on the flat path.
            if not is_recoverable_task_failure(exc):
                raise
            before = set(dead)
            _detect_dead()
            return None, (dead - before or {-1})  # -1: force a retry even if nothing new
        except ray.exceptions.RayActorError:
            before = set(dead)
            _detect_dead()
            return None, (dead - before or {-1})  # -1: force a retry even if nothing new

    def recompute(failed):
        for src in (s for s in failed if isinstance(s, int) and s >= 0):
            # Retire this source's replicas before republishing it — the same epoch
            # invariant the flat path enforces (see `dist/shuffle_replication.py`): a
            # stale replica's ticket reads back as an EMPTY bucket, not an error, so
            # falling back to it would silently drop this mapper's rows. The tree
            # republishes at the leaf, so the recomputed primary is the only valid copy.
            if replicas is not None and src < len(replicas):
                if replicas[src]:
                    events.publish(
                        events.RECOVERY,
                        name="aggregate",
                        event="replica_retired",
                        shuffle="aggregate",
                        src=src,
                        replicas=len(replicas[src]),
                    )
                replicas[src] = []
            target = next(j for j in range(workers) if j not in dead)
            leaf_addrs[src] = ray.get(
                actors[target].map_publish.remote(
                    map_ir, gk, aj, partitions[src], n_keys, n_reducers, src, 0, current_plan_id()
                )
            )

    return ShuffleRecovery(recovery_policy(), label="aggregate").run(attempt, recompute)
