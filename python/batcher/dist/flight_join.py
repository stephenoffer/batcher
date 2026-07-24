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
from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
from batcher.dist.executors.plan_analysis import empty_result_table
from batcher.dist.executors.ray_runtime import (
    engine_config_json,
    map_barrier,
    shuffle_partitions,
)
from batcher.dist.fleet import acquire_fleet, release_fleet
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.dist.flight_worker import current_plan_id
from batcher.io.source import Source
from batcher.plan.ir_specs import agg_spec_json
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
            "op": "hash_join",
            "left": {"op": "scan", "source_id": 0},
            "right": {"op": "scan", "source_id": 1},
            "left_keys": list(join.left_keys),
            "right_keys": list(join.right_keys),
            "join_type": join.join_type,
            "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in join.output],
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

    # Borrow the query-lifetime fleet when the adaptive loop installed one; else spawn
    # one we tear down. Every Flight operator must borrow it — spawning a second
    # placement group would contend with the fleet's held bundles and deadlock.
    actors, pg, _addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = shuffle_partitions(workers)
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
        lparts = partition_descriptors(sources[lsid], workers, projection=lproj, predicate=lpred)
        rparts = partition_descriptors(sources[rsid], workers, projection=rproj, predicate=rpred)

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

        def _launch(host: int, src: int):
            left, right = _sides(src)
            return actors[host].map_publish_join.remote(
                left, right, n_buckets, src, 0, current_plan_id()
            )

        mapper_addrs, mapper_dead = map_barrier(workers, _launch)

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
            finalize=not combine_partials,
            dead=mapper_dead,
            publish=publish,
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
):
    """Run the join reduce under recompute-on-worker-loss recovery.

    Each reducer is hosted on a live worker; one that reports an unreachable mapper
    (or whose host died) drives a recompute of that worker's *both* sides (the join
    co-partitions left and right) from their on-disk source partitions onto a
    survivor, then a retry. Returns the joined batches — or, with `publish`, the
    `(addr, ticket, rows, schema)` handles of the buckets left ON the workers.

    `dead` seeds the workers the map barrier already lost, so no reducer is hosted on
    an actor that is gone.
    """
    import ray

    from batcher.dist.executors.ray_runtime import run_bucket_reduce

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
                join_ir, addrs, bucket, lschema, rschema, None, None, current_plan_id()
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
            None,
            current_plan_id(),
        )

    def republish(target: int, src: int) -> None:
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
    )
    if publish:
        # One `(addr, ticket, rows, schema)` handle per non-empty bucket, still on its worker.
        return [p for p in done.values() if p]
    out = []
    for res in done.values():
        if res:
            out.extend(b for b in res if b.num_rows > 0)
    return out
