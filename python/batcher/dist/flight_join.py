"""Distributed hash join over an Arrow Flight shuffle (object store bypassed).

Co-partitions both join sides by key over two shuffle stages, then joins each
co-located bucket. Like the aggregate Flight path, only `(addr, ticket)` strings
(and the small results) transit Ray — the heavy data moves node→node over
credit-bounded Flight, never through the object store. The per-worker Flight
endpoint and the credit window come from the shared `_FlightWorker` /
`_shuffle_credits` in `flight_aggregate`.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.carbonite.resilience import SourcePlacement
from batcher.dist.adaptive_sizing import row_shuffle_reducer_count
from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import (
    empty_descriptor,
    partition_descriptors,
    source_pushdown,
)
from batcher.dist.executors.plan_analysis import empty_result_table
from batcher.dist.executors.ray_runtime import (
    engine_config_json,
    map_barrier,
    map_partitions,
    shuffle_partitions,
    skew_join_salt,
)
from batcher.dist.executors.ray_runtime.metering import drain_worker_metrics
from batcher.dist.fleet import acquire_fleet, release_fleet
from batcher.dist.fleet.plan_id import next_result_stage, next_stage_base
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.dist.flight_broadcast import broadcast_eligible, execute_broadcast_join_flight
from batcher.dist.flight_worker import current_plan_id
from batcher.dist.shuffle_replication import replicate_shuffle_output, retire_replicas
from batcher.dist.skew import join_skew_key, resolve_hot_keys, salting_preserves_result
from batcher.io.source import Source
from batcher.plan.ir_specs import agg_spec_json, task_scan_ir
from batcher.plan.logical import Aggregate, Join, LogicalPlan

__all__ = ["execute_join_flight"]


def execute_join_flight(
    above: list[LogicalPlan],
    join: Join,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    fused_agg: Aggregate | None = None,
    combine_partials: bool = False,
    materialize: bool = True,
    _fault_inject_map: set[int] | None = None,
    hub=None,
    metrics_out=None,
):
    """Co-partition both join sides over a Flight shuffle and join per bucket.

    `materialize=False` (an *intermediate* stage of a multi-join query, with nothing
    fused above) leaves each reducer's joined bucket published on its host actor and
    returns a `FlightMaterializedSource` over the handles, so the next stage's mappers
    fetch it shared-nothing. Otherwise the reducers' batches are collected into a
    `pa.Table` (the returned type is therefore one or the other — the caller handles
    both, exactly as for the aggregate).

    `fused_agg` is folded INTO the reduce so only small aggregated/partial buckets reach
    the driver — the full join never materializes on the head (exchange elimination). When
    its group keys ⊇ the join key every group lands in one bucket, so each reducer's
    per-bucket aggregate is complete and the driver concatenates. When they do NOT (set
    `combine_partials`), a group spans buckets: each reducer emits its PARTIAL state and
    the driver does the cross-bucket `combine_finalize` (standard mergeable two-phase),
    so an aggregate over an arbitrary join still runs fully distributed instead of
    collecting the whole join to the driver to aggregate it single-node.

    Left and right mappers publish their key-hashed buckets on their own Flight
    servers (shuffle stages 0 and 1); reducer r fetches bucket r from every mapper
    on both sides and runs the local join. A lost worker's buckets are recomputed
    from its source partitions (still on disk) on a survivor — Spark-style lineage
    recovery, matching the aggregate path — in the map phase via `map_barrier`, and
    thereafter via `ShuffleRecovery`. Object store bypassed. `_fault_inject` /
    `_fault_inject_map` are test-only hooks: worker ids to kill after / before the map
    barrier."""
    import ray

    # A broadcast-marked join replicates its small build side and shuffles nothing — see
    # `flight_broadcast`. Tried first and only for the join types where it yields the same
    # relation; it returns None (build side empty or over the measured budget) to fall
    # through to the co-partition shuffle below, which is always correct.
    #
    # `combine_partials` is not passed on: it distinguishes a reducer that may finalize its
    # bucket from one that may not, and a broadcast join co-partitions nothing, so its
    # aggregate is *always* a partial the driver closes. The flag has no counterpart here.
    if broadcast_eligible(join):
        out = execute_broadcast_join_flight(
            above, join, sources, workers, fused_agg=fused_agg, materialize=materialize
        )
        if out is not None:
            return out

    nat = engine()
    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    # Restrict each side to exactly the columns the join OUTPUT carries (Kyber already
    # pruned `join.output` to what the consumer above needs) plus the join keys, so the
    # shuffle moves only those columns — not the whole wide table. Without this each side
    # ships every source column (the relabeled per-side sub-plan is a bare scan, so its
    # own `source_pushdown` can't see that the join above keeps just a few): TPC-H lineitem
    # has 17 columns but a `…⋈ orders GROUP BY priority` join needs 2, so the un-pruned
    # shuffle moved ~8× the bytes and dominated the join.
    left_need = {o.name for o in join.output if o.side == "left"} | set(join.left_keys)
    right_need = {o.name for o in join.output if o.side == "right"} | set(join.right_keys)
    left_plan, lsid = _relabel_single_source(_project_join_side(join.left, left_need))
    right_plan, rsid = _relabel_single_source(_project_join_side(join.right, right_need))
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    join_ir = json.dumps(
        {
            **join.shape_ir(),
            "left": task_scan_ir(),
            "right": task_scan_ir(1),
        }
    )
    # A fused aggregate's group keys/aggregates (over the join output columns), shipped to
    # each reducer to fold its joined bucket down before it leaves the worker.
    gk = aj = None
    if fused_agg is not None:
        gk, aj = agg_spec_json(fused_agg)

    # 0-row schema probes so reducers can type the null-extended side of an outer join.
    def probe(sub_ir, source):
        empty = pa.RecordBatch.from_pylist([], schema=source.schema())
        out = nat.execute_plan(sub_ir, [[empty]], cfg_json)
        return out[0] if out else empty

    credits = _shuffle_credits()
    # Two consecutive ticket stages for THIS join's map buckets (left, right). Every join
    # used the fixed stages 0 and 1, so a query with two joins had two sets of map buckets at
    # identical coordinates on the same worker — see `fleet.plan_id.next_stage_base`.
    stage_base = next_stage_base(2)

    # Borrow the query-lifetime fleet when the adaptive loop installed one; else spawn
    # one we tear down. Every Flight operator must borrow it — spawning a second
    # placement group would contend with the fleet's held bundles and deadlock.
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    # A join exchanges raw rows, so the bucket count is what bounds the build-side hash
    # table a reducer holds at once. Sized by the exchanged volume (`row_shuffle_reducer_count`),
    # never below the one-per-worker floor. `right_plan` was relabeled to read source 0, so the
    # estimator is handed that one source — passing the whole list would size the build side
    # from whichever table happens to be source 0.
    n_buckets = row_shuffle_reducer_count(right_plan, shuffle_partitions(workers), sources, rsid)
    # Keep the join's output ON the workers only when it is a plain intermediate: nothing
    # to apply above it, and no fused aggregate (which already shrinks the bucket to a
    # partial before it leaves — a far smaller thing to hand back than the join itself).
    publish = materialize is False and not above and fused_agg is None
    keep_actors = False  # set when a FlightMaterializedSource takes ownership of them
    try:
        # Each side reads only the columns/rows its map prefix needs (join keys + carried
        # output), not the whole wide table — the biggest scan win on a star-schema join.
        # `_relabel_single_source` rewrote each side's scan to source **0**, so the analysis
        # must be keyed on 0 — not on the side's original id in `sources`. Keyed on `rsid`
        # (always 1 for the right operand) the lookup missed every time and returned "no
        # pushdown", so the build side read EVERY column of its table from storage and the
        # map prefix then threw the surplus away. The left side (`lsid == 0`) coincidentally
        # agreed with the relabeled id, which is why only the right side paid for it.
        lproj, lpred = source_pushdown(left_plan, 0)
        rproj, rpred = source_pushdown(right_plan, 0)
        # More map partitions than workers where the sources have splits to fill them, so a
        # straggler holds a fraction of a node's share rather than all of it (see
        # `map_partitions`). Both sides map through ONE barrier under a single source id, so
        # the two lists have to be the same length — and each side's count is bounded by its
        # own splits. Pad the shorter one with no-op partitions rather than re-planning the
        # longer one's splits, which on a star-schema join is the expensive side.
        ceiling = map_partitions(workers)
        lparts = partition_descriptors(
            sources[lsid],
            workers,
            projection=lproj,
            predicate=lpred,
            worker_addrs=fleet_addrs,
            max_partitions=ceiling,
        )
        rparts = partition_descriptors(
            sources[rsid],
            workers,
            projection=rproj,
            predicate=rpred,
            worker_addrs=fleet_addrs,
            max_partitions=ceiling,
        )
        n_sources = max(len(lparts), len(rparts))
        lparts += [empty_descriptor(sources[lsid], lproj) for _ in range(n_sources - len(lparts))]
        rparts += [empty_descriptor(sources[rsid], rproj) for _ in range(n_sources - len(rparts))]
        placement = SourcePlacement(workers)

        # Simulate worker loss BEFORE the map barrier (test hook).
        if _fault_inject_map:
            for i in _fault_inject_map:
                ray.kill(actors[i])

        # MAP barrier under worker-loss recovery: a worker preempted while mapping has
        # BOTH its sides republished on one survivor under the same `src`, so the single
        # `mapper_addrs[src]` the reducers dial still resolves. Both sides go out as ONE
        # actor call (`map_publish_join`), so the barrier awaits a single ref that fails if
        # *either* side fails — issuing two calls and awaiting only the second hid a
        # left-side error behind a healthy-looking address and mislabelled it, three
        # retries later, as an unreachable worker.
        def _sides(src: int):
            return (
                (left_ir, list(join.left_keys), lparts[src]),
                (right_ir, list(join.right_keys), rparts[src]),
            )

        # Skew-aware salting. A single hot join key sends `fraction x rows` to ONE reducer
        # however wide the shuffle is, so the join does not merely stop scaling — it goes
        # *backwards*. Measured on this path, 40M ⋈ 10M with 40% of the probe on one key
        # (`benchmarks/BENCHMARK_RESULTS.md`): a uniform join of the same shape and size runs
        # 3,712 -> 2,016 -> 1,657 ms across 2, 4 and 8 workers, while the skewed one runs
        # 6,138 -> 3,332 -> **12,801** ms. Adding workers shrinks every reducer except the
        # one that matters, and the widening fan-out is then pure overhead against it.
        #
        # Salting splits that key across sub-buckets (replicating its build rows to each),
        # which is a pure scheduling change: `salting_preserves_result` states the three
        # conditions, and `finalize` is the one that matters here — a reducer closing a fused
        # aggregate is relying on co-partitioning that salting deliberately breaks.
        finalize = not combine_partials
        skew = None
        if salting_preserves_result(join, reducer_finalizes=fused_agg is not None and finalize):
            cfg_salt, frac = skew_join_salt()
            hot, salt = resolve_hot_keys(
                join,
                sources,
                join_skew_key(left_ir, right_ir, join),
                frac,
                n_buckets,
                cfg_salt,
                lambda: _detect_hot_keys_flight(
                    actors,
                    (left_ir, join.left_keys[0], lparts),
                    (right_ir, join.right_keys[0], rparts),
                    frac,
                ),
            )
            skew = (hot, salt) if salt else None

        def _launch(host: int, src: int):
            left, right = _sides(src)
            return actors[host].map_publish_join.remote(
                left, right, n_buckets, src, 0, current_plan_id(), stage_base, skew
            )

        mapper_addrs, mapper_dead = map_barrier(
            n_sources, _launch, workers=workers, placement=placement
        )

        # Placed HERE, as soon as the buckets exist and before anything can take a worker
        # away — replicating after a loss would be probing a corpse. A join's mapper
        # publishes BOTH sides under one address, so one replica covers both and a lost
        # worker costs no re-read of either source. `None` (the default factor of 1)
        # leaves the reduce byte-identical to the unreplicated path.
        replicas = replicate_shuffle_output(
            actors, mapper_addrs, n_buckets, workers, mapper_dead, stages=(0, 1)
        )

        # Simulate worker loss after the map barrier (test hook).
        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        lschema = probe(left_ir, sources[lsid])
        rschema = probe(right_ir, sources[rsid])
        out = _join_reduce_with_recovery(
            actors,
            mapper_addrs,
            (lparts, rparts),
            (left_ir, right_ir),
            (list(join.left_keys), list(join.right_keys)),
            join_ir,
            (lschema, rschema),
            n_buckets,
            workers,
            gk,
            aj,
            finalize=finalize,
            dead=mapper_dead,
            publish=publish,
            replicas=replicas,
            stage_base=stage_base,
            skew=skew,
            placement=placement,
        )
        if publish:
            from batcher.dist.fleet import FlightMaterializedSource

            # `out` is the reducers' `(addr, ticket, rows, schema)` handles; an empty
            # bucket publishes nothing and is simply absent.
            handles = [(a, t, n) for a, t, n, _s in out]
            schema = out[0][3] if out else _join_output_schema(join, lschema, rschema)
            keep_actors = True  # the source holds the buckets; the fleet must outlive us
            # A borrowed fleet is the query's (freed once by the adaptive loop), so the
            # source must not own it; only a self-spawned fleet is handed over to tear down.
            src_actors, src_pg = (actors, pg) if owns else (None, None)
            return FlightMaterializedSource(handles, schema, src_actors, src_pg)
        batches = out
    finally:
        # Collect what the workers measured before anything below can kill them. Nothing
        # subscribes to the event bus inside a Ray worker, so the measurements are pulled;
        # this is the one point every exit path passes through with the actors still alive.
        drain_worker_metrics(actors, hub, metrics_out)
        # A published result leaves its buckets ON the actors, so a fleet we own is handed
        # to the source rather than torn down here.
        if not keep_actors:
            release_fleet(actors, pg, owns)

    # Non-fusable fused aggregate: reducers shipped PARTIAL state (one per bucket); the
    # group spans buckets, so do the cross-bucket combine+finalize here on the small
    # partials (workers × groups rows), not on the full join.
    if combine_partials and fused_agg is not None:
        final = nat.combine_finalize(gk, aj, batches) if batches else None
        table = pa.Table.from_batches([final]) if final is not None else _empty_fused(fused_agg)
        return table if not above else _apply_above(above, table)

    if batches:
        table = pa.Table.from_batches(batches)
    elif fused_agg is not None:
        table = _empty_fused(fused_agg)
    else:
        table = empty_result_table(join, [o.alias for o in join.output])
    return table if not above else _apply_above(above, table)


def _detect_hot_keys_flight(actors, left, right, fraction: float) -> tuple[list[str], float]:
    """Detect the join key's hot values across BOTH sides, on the workers.

    Each side's splits are scanned in place by the actor that owns them (Misra-Gries per
    split), and the local counts are summed here; a value is hot when it clears `fraction`
    of *that side's* total rows. Both sides are examined because either one can be the
    skewed one, and a hot key on the build side overloads a reducer exactly as a hot probe
    key does.

    The Flight counterpart of the disk path's `_detect_hot_keys`, differing only in that
    the splits are descriptors read by an actor rather than files read by a task — the data
    is never pulled to the driver either way.

    Args:
        actors: The worker fleet.
        left: `(sub_ir, key_name, partitions)` for the probe side.
        right: `(sub_ir, key_name, partitions)` for the build side.
        fraction: The share of a side's rows a value must hold to count as hot.

    Returns:
        `(hot_values, max_share)` — the values sorted, and the largest one's measured share
        of its side, which is what sizes the salt fan-out (see `skew.resolve_hot_keys`).
        `([], 0.0)` when neither side is skewed.
    """
    import ray

    hot: set[str] = set()
    peak = 0.0
    for sub_ir, key_name, parts in (left, right):
        refs = [
            actors[i % len(actors)].heavy_hitters.remote(sub_ir, key_name, parts[i], fraction)
            for i in range(len(parts))
        ]
        counts: dict[str, int] = {}
        total = 0
        for pairs, n in ray.get(refs):
            total += n
            for v, c in pairs:
                counts[v] = counts.get(v, 0) + c
        if not total:
            continue
        for v, c in counts.items():
            if c >= fraction * total:
                hot.add(v)
                peak = max(peak, c / total)
    return sorted(hot), peak


def _empty_fused(fused_agg: Aggregate) -> pa.Table:
    """The empty result table for a fused post-join aggregate (group keys + aggregates)."""
    keys = [k.alias for k in fused_agg.group_keys]
    return empty_result_table(fused_agg, keys + [s.alias for s in fused_agg.aggregates])


def _project_join_side(side: LogicalPlan, needed: set[str]) -> LogicalPlan:
    """Wrap a join side in a `Project` selecting only `needed` columns, so its scan reads
    and shuffles just those — `source_pushdown` then maps them through any rename/filter
    to the actual scan columns. A no-op when nothing can be pruned (needed ⊇ available),
    so a `SELECT *`-style join whose output Kyber did not prune is unchanged.
    """
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Project, Projection

    avail = side.available_columns()
    keep = [c for c in avail if c in needed]  # preserve the side's column order
    if not keep or len(keep) == len(avail):
        return side
    return Project(side, tuple(Projection(alias=c, expr=Col(c)) for c in keep))


def _join_output_schema(join: Join, lschema: pa.RecordBatch, rschema: pa.RecordBatch) -> pa.Schema:
    """The join's output schema, for a result whose every bucket came back empty.

    Built from `join.output` against the two sides' probe schemas, so a
    `FlightMaterializedSource` over zero buckets still advertises the columns the next
    stage's plan is typed against (an empty relation must still have a schema).
    """
    sides = {"left": lschema.schema, "right": rschema.schema}
    fields = []
    for o in join.output:
        side = sides[o.side]
        field = side.field(side.get_field_index(o.name))
        fields.append(field.with_name(o.alias))
    return pa.schema(fields)


def _join_reduce_with_recovery(
    actors,
    addrs,
    parts,
    irs,
    keys,
    join_ir,
    schemas,
    n_buckets,
    workers,
    gk=None,
    aj=None,
    finalize=True,
    dead=None,
    publish=False,
    replicas=None,
    stage_base=0,
    skew=None,
    placement=None,
):
    """Run the join reduce under recompute-on-worker-loss recovery.

    Each reducer is hosted on a live worker; one that reports an unreachable mapper
    fetches the byte-identical buckets from a `replicas` survivor, which holds *both*
    sides (a join replica is all-or-nothing across the two shuffle stages). Only a
    source whose every copy is gone drives a recompute of both sides from their on-disk
    source partitions onto a survivor, then a retry. Returns the joined batches — or,
    with `publish`, the `(addr, ticket, rows, schema)` handles of the buckets left ON
    the workers.

    `dead` seeds the workers the map barrier already lost, so no reducer is hosted on
    an actor that is gone.

    `skew` is the `(hot_keys, salt_count)` the map barrier bucketed under, and it MUST be
    carried here: a recompute republishes a lost source's buckets, and bucketing that
    replacement plainly while its peers are salted would send the hot key's replacement
    rows to a different reducer than the build rows they must meet — a silently short join
    result, on the recovery path, which no test that never loses a worker would see.
    """
    import ray

    from batcher.dist.executors.ray_runtime import run_bucket_reduce

    # One stage id for every bucket of THIS published result. Without it two materialized
    # intermediates of the same query share a ticket and the second overwrites the first —
    # see `fleet.plan_id.next_result_stage`.
    result_stage = next_result_stage() if publish else 0
    lparts, rparts = parts
    left_ir, right_ir = irs
    left_keys, right_keys = keys
    lschema, rschema = schemas

    def remote_reduce(host: int, bucket: int):
        # `publish`: leave the joined bucket on the worker and return a handle, so an
        # intermediate join never round-trips through the driver (see `reduce_join_publish`).
        # Otherwise the reducer ships its batches back.
        if publish:
            return actors[host].reduce_join_publish.remote(
                join_ir,
                addrs,
                bucket,
                lschema,
                rschema,
                None,
                replicas,
                current_plan_id(),
                result_stage,
                stage_base,
            )
        return actors[host].reduce_join.remote(
            join_ir,
            addrs,
            bucket,
            lschema,
            rschema,
            gk,
            aj,
            finalize,
            None,
            replicas,
            current_plan_id(),
            stage_base,
        )

    def republish(target: int, src: int) -> None:
        # Retire before republishing — a replica taken at the old epoch reads back as an
        # empty bucket, not an error. See `dist/shuffle_replication.py::retire_replicas`.
        retire_replicas(replicas, src, target, "join")
        # One call for both sides, as in the map barrier: awaiting only the right side would
        # let a left-side failure pass for a successful republish.
        addrs[src] = ray.get(
            actors[target].map_publish_join.remote(
                (left_ir, left_keys, lparts[src]),
                (right_ir, right_keys, rparts[src]),
                n_buckets,
                src,
                0,
                current_plan_id(),
                stage_base,
                skew,
            )
        )

    done = run_bucket_reduce(
        kind="join",
        n_buckets=n_buckets,
        workers=workers,
        actors=actors,
        remote_reduce=remote_reduce,
        republish=republish,
        dead=dead,
        mapper_addrs=addrs,
        replicas=replicas,
        placement=placement,
    )
    if publish:
        # One `(addr, ticket, rows, schema)` handle per non-empty bucket, still on its worker.
        return [p for p in done.values() if p]
    out = []
    for res in done.values():
        if res:
            out.extend(b for b in res if b.num_rows > 0)
    return out
