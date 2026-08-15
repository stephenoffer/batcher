"""Distributed sort over an Arrow Flight shuffle (object store bypassed).

Range-partitions by the leading sort key across workers, sorts each range, and
concatenates the ranges in key order — globally sorted, no final merge. The range
boundaries come from a **sample pass**: each worker samples its OWN split's
leading-key quantile grid (so the input is never read on the driver, unlike the
disk sort), and the driver merges the small grids into `n_buckets-1` boundaries. The
rows then move node→node over credit-bounded Flight, never through the object
store. Reuses the shared `_FlightWorker` and the same Spark-style lineage recovery.

Boundary precision only affects *balance*, never correctness: range-partition →
per-range sort → ordered concat is order-preserving for any boundaries, because the
boundaries are deduplicated and `searchsorted(side="right")` keeps equal keys in one
bucket. Restricted (by the dispatcher) to a leading key that is a plain column over
a breaker-free single source.
"""

from __future__ import annotations

import json
import logging

import pyarrow as pa

from batcher._internal.logging import get_logger, log_kv
from batcher._internal.native import engine
from batcher.carbonite.resilience import SourcePlacement
from batcher.dist.adaptive_sizing import row_shuffle_reducer_count
from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import (
    merge_boundaries,
    partition_descriptors,
    plan_hot_split,
    sample_probs,
    source_pushdown,
)
from batcher.dist.executors.plan_analysis import empty_result_table
from batcher.dist.executors.ray_runtime import (
    engine_config_json,
    map_barrier,
    map_partitions,
    shuffle_partitions,
)
from batcher.dist.executors.ray_runtime.metering import drain_worker_metrics
from batcher.dist.fleet import acquire_fleet, release_fleet
from batcher.dist.fleet.plan_id import next_result_stage, next_stage_base
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.dist.flight_worker import current_plan_id
from batcher.dist.shuffle_replication import replicate_shuffle_output, retire_replicas
from batcher.dist.sort_boundaries import (
    load_learned_grids,
    persist_grids,
    sort_key_identity,
    sort_key_is_string,
    sort_shape_key,
)
from batcher.io.source import Source
from batcher.plan.ir_specs import sort_keys_ir, task_scan_ir
from batcher.plan.logical import LogicalPlan, Sort

__all__ = ["execute_sort_flight", "execute_topn_flight"]

_log = get_logger("dist.sort")


def _phase(name: str, seconds: float, **fields: object) -> None:
    """Record one distributed-sort phase timing on the central logger.

    These timings used to be `print`s behind a `BATCHER_SORT_PROFILE` env var: invisible to
    the log file, unfilterable, on stdout in the middle of a user's results, and unknown to
    the dashboard. As DEBUG records on `batcher.dist.sort` they answer to the same
    `log_level` as everything else, and the phase name and duration are structured fields
    rather than a sentence — so "which phase dominates this sort" is a query, not a grep.
    """
    log_kv(_log, logging.DEBUG, "sort phase", phase=name, seconds=round(seconds, 3), **fields)


def _sort_ir(keys, limit, input_ir):
    """The sort IR over `input_ir` carrying `keys` and `limit` (None = no limit)."""
    return json.dumps(
        {
            "op": "sort",
            "input": input_ir,
            "keys": sort_keys_ir(keys),
            "limit": limit,
        }
    )


def execute_topn_flight(
    above: list[LogicalPlan],
    sort: Sort,
    sources: list[Source],
    workers: int,
) -> pa.Table:
    """A distributed **top-N** (`ORDER BY ... LIMIT k`) with NO shuffle — mergeable.

    The global top-N is the top-N of the union of each worker's top-N, so every worker
    reads its own split, runs the map prefix + the single-node top-N heap (`sort+limit`),
    and ships only `k` rows. This skips the full range-partition sort entirely (which would
    shuffle every row just to slice the first `k`), the dominant cost for a small `k`.

    The driver folds those into a running top-`k` one worker at a time rather than gathering
    every one of them first. Both halves of that matter and for different reasons. Holding
    them makes the driver's peak `workers x k` rows, which is unbounded in the cluster: at
    the guarded ceiling of a million rows a two-hundred-worker fleet would put two hundred
    million rows through one node to return a million. And gathering first puts the whole
    merge *after* the barrier, a Θ(workers · k log k) serial tail that grows exactly as the
    map phase in front of it shrinks. Folding makes the peak `2k` and leaves one merge behind
    the barrier instead of `workers` of them. It is `streaming_topn`'s fold, one level up.
    """
    import ray

    nat = engine()
    _ensure_ray(workers)
    cfg_json = engine_config_json()
    map_plan, sid = _relabel_single_source(sort.input)
    # Per-worker plan: read the split (scan 0) → map prefix → local top-N heap.
    local_ir = _sort_ir(sort.keys, sort.limit, map_plan.to_ir())
    # Driver merge plan: top-N over the concatenated per-worker top-Ns (scan 0).
    merge_ir = _sort_ir(sort.keys, sort.limit, task_scan_ir())

    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, _shuffle_credits(), cfg_json)
    try:
        # Partition to the fleet's ACTUAL worker count. `acquire_fleet` may hand back a
        # reused session fleet whose size differs from the requested `workers` (it reassigns
        # `workers` to that size), so `parts` must be built here, after the fleet is known —
        # otherwise parts and actors mismatch: a larger fleet indexes past `parts`, a smaller
        # one silently drops the tail partitions' rows (a wrong result). `execute_sort_flight`
        # already orders it this way.
        # `map_plan`'s scan was relabeled to source 0, so key the analysis on 0, not on the
        # source's original index: a staged plan whose input is an intermediate (source id >
        # 0) missed the lookup and silently read every column.
        projection, predicate = source_pushdown(map_plan, 0)
        # Contiguous, source-ordered partitions. A top-N keeps only `k` of the rows it
        # orders, so which of several rows tied at the `k`-th place survives is decided by
        # input order — and the load-balanced split pick hands one partition non-adjacent
        # runs of the source, which selects a different tied row than single-node does.
        # Measured over a 40-value key across twelve files: `ORDER BY k LIMIT 137` returned
        # the right 137 keys and a different set of rows. Order preservation outranks the
        # locality `worker_addrs` asks for, which is `assign_splits`' own stated priority.
        parts = partition_descriptors(
            sources[sid],
            workers,
            projection=projection,
            predicate=predicate,
            preserve_order=True,
            worker_addrs=fleet_addrs,
        )
        refs = [actors[i].local_topn.remote(local_ir, parts[i]) for i in range(workers)]
        merged: list = []
        for i, ref in enumerate(refs):
            # In worker order, not arrival order. The fold has to be bounded, but it must
            # not become *arrival*-ordered: `LIMIT k` over rows that tie at the k-th place
            # may return any of them, and which ones it returns would then vary run to run
            # on the same data. Worker order is fixed by the partitioning, so this returns
            # exactly what the one-shot merge did. Waiting on worker `i` still overlaps the
            # merge with every later worker's scan, so the serial tail is one merge rather
            # than `workers` of them.
            arrived = [b for b in ray.get(ref) if b.num_rows > 0]
            refs[i] = None  # drop the ref so the worker's copy can be freed
            if arrived:
                merged = list(nat.execute_plan(merge_ir, [merged + arrived], cfg_json))
    finally:
        release_fleet(actors, pg, owns)

    if merged:
        table = pa.Table.from_batches(merged)
    else:
        table = pa.table({k.expr.name: [] for k in sort.keys})
    return table if not above else _apply_above(above, table)


def execute_sort_flight(
    above: list[LogicalPlan],
    sort: Sort,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    _fault_inject_map: set[int] | None = None,
    hub=None,
    metrics_out=None,
    materialize: bool = True,
):
    """Range-partition by the leading key over Flight, sort each range, concat in order.

    Worker loss is survived in every phase: `map_barrier` reprocesses a split whose worker
    dies while sampling or range-partitioning, and `ShuffleRecovery` recomputes a range
    bucket whose worker dies before the reduce fetches it. `_fault_inject` /
    `_fault_inject_map` are test-only hooks: worker ids to kill after / before the map
    barrier.

    `materialize=False` leaves each range bucket published on the worker that sorted it
    and returns a `FlightMaterializedSource` over the handles **in range order** — which
    is the sorted order, since the ranges are globally ordered against one another. A sort
    is row-preserving, so the `driver_concat` phase below moves the entire relation through
    one process; this is the path that removes it. Declined (and a table returned) when
    something is stacked `above` or the sort carries a `limit` that must slice the assembled
    result, so the caller has to handle either type."""
    import time as _tt0

    import ray

    _enter = _tt0.perf_counter()
    _ensure_ray(workers)
    _phase("ensure_ray", _tt0.perf_counter() - _enter)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    key = sort.keys[0]  # caller guarantees a plain-column leading key
    key_name = key.expr.name
    desc, nulls_first = key.descending, key.nulls_first
    map_plan, sid = _relabel_single_source(sort.input)
    map_ir = json.dumps(map_plan.to_ir())
    sort_ir = json.dumps(
        {
            **sort.shape_ir(),
            "input": task_scan_ir(),
        }
    )
    credits = _shuffle_credits()

    # A `limit` slices the assembled result and `above` has nothing to apply itself to
    # without one, so both keep the collect. Everything else stays on the workers.
    publish = materialize is False and not above and sort.limit is None
    keep_actors = False  # set when a FlightMaterializedSource takes ownership of them

    import time as _tt

    _ps = _tt.perf_counter()
    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a
    # second placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    # A sort exchanges raw rows and sorts one bucket at a time, so the bucket count is what
    # bounds the run a reducer materializes. Sized by volume, never below the floor.
    n_buckets = row_shuffle_reducer_count(map_plan, shuffle_partitions(workers), sources, sid)
    _phase("acquire_fleet", _tt.perf_counter() - _ps, workers=workers, buckets=n_buckets)
    try:
        # Push the map prefix's projection + predicate into the read so each worker
        # fetches only the columns/rows it needs (the sort keys + carried output), not
        # the whole wide source — the projection the `map_ir` would otherwise discard
        # after paying to read it (see flight_aggregate).
        # `map_plan`'s scan was relabeled to source 0, so key the analysis on 0, not on the
        # source's original index: a staged plan whose input is an intermediate (source id >
        # 0) missed the lookup and silently read every column.
        projection, predicate = source_pushdown(map_plan, 0)
        # More map partitions than workers where the source has splits to fill them, so a
        # straggler holds a fraction of a node's share rather than all of it (see
        # `map_partitions`). `len(parts)` is the source count from here on — for the sample
        # barrier too, which simply merges more quantile grids.
        # A sort carrying a `limit` too large for the shuffle-free top-N still *slices* its
        # ordered result, so it selects among rows tied at the cut and needs the same
        # source-ordered partitions the top-N does. An unlimited sort returns every row, so
        # the pick is free and the balanced, locality-aware one is better.
        parts = partition_descriptors(
            sources[sid],
            workers,
            projection=projection,
            predicate=predicate,
            preserve_order=sort.limit is not None,
            worker_addrs=fleet_addrs,
            max_partitions=map_partitions(workers),
        )
        n_sources = len(parts)
        placement = SourcePlacement(workers)

        import time as _t

        # Simulate worker loss BEFORE the sample/map barriers (test hook).
        if _fault_inject_map:
            for i in _fault_inject_map:
                ray.kill(actors[i])

        # Both barriers run under worker-loss recovery, sharing one `dead` view of the
        # fleet: a worker preempted while sampling or range-partitioning has its split
        # reprocessed on a survivor rather than failing the whole sort. Sampling only
        # reads (nothing to republish); range-publish carries `src` so the relocated
        # buckets keep the ticket the reducers dial.
        dead: set[int] = set()

        # One ticket stage for THIS sort's shuffle. The stage used to be the literal 0, so
        # two sorts in one query (or a sort beside a window) published byte-identical
        # tickets on the same worker and the second overwrote the first — the collision
        # that made a join read another join's buckets. See `fleet.plan_id.next_stage_base`.
        stage_base = next_stage_base(1)

        # SAMPLE: each worker samples its own split's leading-key distribution.
        #
        # Skipped outright when this sort shape has been sampled before. The barrier
        # executes the entire mapped prefix — scan, pushed predicate, projection — over
        # every split to produce a few dozen floats per worker, and `range_publish` below
        # then executes that same prefix a second time to do the actual partitioning. A
        # learned grid removes the first of those two passes. It is safe even when stale:
        # boundaries decide only which reducer a row lands on, and the ordered concat is
        # correct for any monotone boundary list (see this module's header and
        # `dist/sort_boundaries.py`), so a grid that no longer fits the data costs balance
        # and never a row.
        _s = _t.perf_counter()
        # WHICH relation and WHICH type: a bare-scan `map_ir` is a positional source id with
        # no schema, so every single-source sort in the process hashed alike and shared one
        # grid — a wrong-typed one raises in the range partitioner, and a wrong-relation one
        # silently puts the whole input in a single bucket. See
        # `dist/sort_boundaries.sort_shape_key`. `expect_strings` re-checks on load, so an
        # entry written under the old colliding digest re-samples instead of raising.
        key_is_str = sort_key_is_string(sources[sid], key_name)
        shape_key = sort_shape_key(map_ir, key_name, sort_key_identity(sources[sid], key_name))
        grids = load_learned_grids(shape_key, key_is_str)
        learned = grids is not None
        if not learned:
            # The grid is sized against the *bucket* count rather than fixed: `n_buckets`
            # runs well above the source count on a volume-sized shuffle, and a grid that
            # resolves a boundary only to a fraction of a bucket overloads the unluckiest
            # reducer by that much while every other one waits on it. See `sample_probs`.
            probs = sample_probs(n_buckets, n_sources)
            grids, dead = map_barrier(
                n_sources,
                lambda host, src: actors[host].sample_quantiles.remote(
                    map_ir, key_name, probs, parts[src]
                ),
                dead=dead,
                workers=workers,
            )
            persist_grids(shape_key, grids)
        # Cut into exactly `n_buckets` ranges: `shuffle_partitions` can trim the reducer
        # count below `workers` (the `max_shuffle_partitions` cap / learned fan-out), and
        # `merge_boundaries(grids, workers)` would emit up to `workers-1` boundaries — more
        # than `n_buckets-1` — routing rows past the last bucket and panicking the range
        # partitioner. Size the boundaries by the actual bucket count. This is also why the
        # *grids* are what persist rather than the boundaries: the bucket count moves
        # between runs, so a stored boundary list would be the wrong length.
        boundaries = merge_boundaries(grids, n_buckets)
        # A range partition must keep equal keys together, so one dominant value pins its
        # whole share on a single reducer however wide the shuffle is — the busiest bucket
        # stops shrinking as workers are added. `plan_hot_split` gives that value a bucket of
        # its own and spreads it over `subs` of them, one per contiguous run of mappers,
        # which is sound precisely because those rows tie. `None` leaves everything as it was.
        split = plan_hot_split(grids, boundaries, n_buckets, nulls_first, desc)
        if split is not None:
            boundaries, n_buckets, hot_bucket, subs = split
            n_physical = n_buckets + subs - 1
        else:
            hot_bucket, subs, n_physical = -1, 0, n_buckets
        _phase("sample", _t.perf_counter() - _s, buckets=n_physical, learned=learned)

        # MAP: range-partition each split by the boundaries and publish raw rows.
        _s = _t.perf_counter()
        mapper_addrs, dead = map_barrier(
            n_sources,
            lambda host, src: actors[host].range_publish.remote(
                map_ir,
                key_name,
                boundaries,
                n_buckets,
                nulls_first,
                desc,
                parts[src],
                src,
                0,
                current_plan_id(),
                stage_base,
                hot_bucket,
                subs,
                n_sources,
            ),
            dead=dead,
            workers=workers,
            placement=placement,
        )
        _phase("map_range_publish", _t.perf_counter() - _s)

        # Placed HERE, as soon as the buckets exist and before anything can take a worker
        # away — replicating after a loss would be probing a corpse. A sort's buckets are
        # raw rows rather than pre-aggregated state, so the copy is larger than the
        # aggregate's; it is still cheaper than re-reading the source and re-running the
        # sample + range partition. `None` (the default factor of 1) leaves the reduce
        # byte-identical to the unreplicated path.
        replicas = replicate_shuffle_output(actors, mapper_addrs, n_physical, workers, dead)

        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        _s = _t.perf_counter()
        results = _sort_reduce_with_recovery(
            actors,
            mapper_addrs,
            parts,
            map_ir,
            key_name,
            boundaries,
            sort_ir,
            nulls_first,
            desc,
            n_buckets,
            workers,
            dead=dead,
            replicas=replicas,
            stage_base=stage_base,
            placement=placement,
            hot_bucket=hot_bucket,
            subs=subs,
            n_physical=n_physical,
            publish=publish,
        )
        _phase("reduce_gather_sort", _t.perf_counter() - _s)
        if publish:
            from batcher.dist.fleet import FlightMaterializedSource

            # In range order (reversed for a descending sort), so reading the handles in
            # sequence reproduces the concatenation this replaces. An empty range published
            # nothing and is simply absent.
            order = range(n_physical - 1, -1, -1) if desc else range(n_physical)
            published = [h for r in order if (h := results.get(r)) is not None]
            handles = [(addr, ticket, rows) for addr, ticket, rows, _schema in published]
            schema = (
                published[0][3]
                if published
                else empty_result_table(sort, sort.available_columns()).schema
            )
            keep_actors = True  # the source holds the buckets; the fleet must outlive us
            # A borrowed fleet is the query's (freed once by the adaptive loop), so the
            # source must not own it; only a self-spawned fleet is handed over to tear down.
            src_actors, src_pg = (actors, pg) if owns else (None, None)
            return FlightMaterializedSource(handles, schema, src_actors, src_pg)
    finally:
        # Collect what the workers measured before anything below can kill them. Nothing
        # subscribes to the event bus inside a Ray worker, so the measurements are pulled;
        # this is the one point every exit path passes through with the actors still alive.
        drain_worker_metrics(actors, hub, metrics_out)
        # A published result leaves its buckets ON the actors, so a fleet we own is handed
        # to the source rather than torn down here.
        if not keep_actors:
            release_fleet(actors, pg, owns)

    # Concatenate the ranges in leading-key order (reversed for a descending sort) —
    # each bucket is globally ordered relative to the others, so no final merge.
    #
    # Over `n_buckets`, NOT `workers`. `shuffle_partitions` is documented to treat the worker
    # count as a floor and raise the bucket count toward `workers x
    # shuffle_partition_multiplier` once the shuffle's volume has been measured, so the two
    # are equal only on a cold store. Walking `range(workers)` then read the first `workers`
    # range buckets and silently dropped every row in the rest — a short result from a sort,
    # with no error, appearing only after a shape had run once. The disk sort
    # (`executors/sort.py`) has always used `n_buckets` here.
    _pc = _tt.perf_counter()
    order = range(n_physical - 1, -1, -1) if desc else range(n_physical)
    out: list[pa.RecordBatch] = []
    for r in order:
        out.extend(b for b in results.get(r, []) if b.num_rows > 0)
    table = (
        pa.Table.from_batches(out) if out else empty_result_table(sort, sort.available_columns())
    )
    _phase("driver_concat", _tt.perf_counter() - _pc, rows=table.num_rows)
    if sort.limit is not None:
        table = table.slice(0, sort.limit)
    _phase("total", _tt.perf_counter() - _enter)
    return table if not above else _apply_above(above, table)


def _sort_reduce_with_recovery(
    actors,
    addrs,
    parts,
    map_ir,
    key_name,
    boundaries,
    sort_ir,
    nulls_first,
    desc,
    n_buckets,
    workers,
    dead=None,
    replicas=None,
    stage_base=0,
    placement=None,
    hot_bucket=-1,
    subs=0,
    n_physical=None,
    publish=False,
):
    """Run the sort reduce under recompute-on-worker-loss recovery.

    Returns a `{bucket_id: sorted_batches}` dict so the driver can concatenate the
    ranges in key order regardless of completion order — or, with `publish`, a
    `{bucket_id: (addr, ticket, rows, schema)}` dict of the buckets left ON their workers.
    A reducer reporting an unreachable mapper fetches the byte-identical bucket from a
    `replicas` survivor; only a source whose every copy is gone drives a recompute of that
    worker's range bucket from its on-disk source partition onto a survivor, then a retry.
    """
    import ray

    from batcher.dist.executors.ray_runtime import run_bucket_reduce

    # One stage id for every bucket of THIS published result, so two materialized
    # intermediates in one query cannot share a ticket (see `fleet.plan_id.next_result_stage`).
    result_stage = next_result_stage() if publish else 0

    def remote_reduce(host: int, bucket: int):
        if publish:
            return actors[host].sort_reduce_publish.remote(
                sort_ir,
                addrs,
                bucket,
                None,
                replicas,
                current_plan_id(),
                stage_base,
                result_stage,
            )
        return actors[host].sort_reduce.remote(
            sort_ir, addrs, bucket, None, replicas, current_plan_id(), stage_base
        )

    def republish(target: int, src: int) -> None:
        # Retire before republishing — a replica taken at the old epoch reads back as an
        # empty bucket, not an error. See `dist/shuffle_replication.py::retire_replicas`.
        retire_replicas(replicas, src, target, "sort")
        addrs[src] = ray.get(
            actors[target].range_publish.remote(
                map_ir,
                key_name,
                boundaries,
                n_buckets,
                nulls_first,
                desc,
                parts[src],
                src,
                0,
                current_plan_id(),
                stage_base,
                hot_bucket,
                subs,
                len(parts),
            )
        )

    n_physical = n_buckets if n_physical is None else n_physical
    done = run_bucket_reduce(
        kind="sort",
        n_buckets=n_physical,
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
        # Keyed by bucket so the caller orders the handles by range; an empty range
        # published nothing and is dropped rather than coerced to an empty handle.
        return {bucket: payload for bucket, payload in done.items() if payload}
    # Keyed by bucket so the driver concatenates ranges in key order; an "ok" reduce over an
    # empty range returns None, coerced to [] so the concatenation never sees a hole.
    return {bucket: (payload or []) for bucket, payload in done.items()}
