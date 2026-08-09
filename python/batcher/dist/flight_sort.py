"""Distributed sort over an Arrow Flight shuffle (object store bypassed).

Range-partitions by the leading sort key across workers, sorts each range, and
concatenates the ranges in key order — globally sorted, no final merge. The range
boundaries come from a **sample pass**: each worker samples its OWN split's
leading-key quantile grid (so the input is never read on the driver, unlike the
disk sort), and the driver merges the small grids into `workers-1` boundaries. The
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
from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import (
    merge_boundaries,
    partition_descriptors,
    source_pushdown,
)
from batcher.dist.executors.plan_analysis import empty_result_table
from batcher.dist.executors.ray_runtime import (
    engine_config_json,
    map_barrier,
    shuffle_partitions,
)
from batcher.dist.fleet import acquire_fleet, release_fleet
from batcher.dist.fleet.plan_id import next_stage_base
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.dist.flight_worker import current_plan_id
from batcher.dist.shuffle_replication import replicate_shuffle_output, retire_replicas
from batcher.dist.sort_boundaries import load_learned_grids, persist_grids, sort_shape_key
from batcher.io.source import Source
from batcher.plan.ir_specs import sort_keys_ir
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


# Per-worker CDF sample granularity: a fine grid (33 probe points) so the merged
# boundaries balance the ranges well. Precision affects only balance, not result.
_SAMPLE_PROBS = [i / 32 for i in range(33)]


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
    and ships only `k` rows; the driver merges the `workers x k` rows with one more
    `sort+limit`. This skips the full range-partition sort entirely (which would shuffle
    every row just to slice the first `k`), the dominant cost for a small `k`.
    """
    import ray

    nat = engine()
    _ensure_ray(workers)
    cfg_json = engine_config_json()
    map_plan, sid = _relabel_single_source(sort.input)
    # Per-worker plan: read the split (scan 0) → map prefix → local top-N heap.
    local_ir = _sort_ir(sort.keys, sort.limit, map_plan.to_ir())
    # Driver merge plan: top-N over the concatenated per-worker top-Ns (scan 0).
    merge_ir = _sort_ir(sort.keys, sort.limit, {"op": "scan", "source_id": 0})

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
        parts = partition_descriptors(
            sources[sid],
            workers,
            projection=projection,
            predicate=predicate,
            worker_addrs=fleet_addrs,
        )
        results = ray.get([actors[i].local_topn.remote(local_ir, parts[i]) for i in range(workers)])
    finally:
        release_fleet(actors, pg, owns)

    gathered = [b for r in results for b in r if b.num_rows > 0]
    merged = nat.execute_plan(merge_ir, [gathered], cfg_json) if gathered else []
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
) -> pa.Table:
    """Range-partition by the leading key over Flight, sort each range, concat in order.

    Worker loss is survived in every phase: `map_barrier` reprocesses a split whose worker
    dies while sampling or range-partitioning, and `ShuffleRecovery` recomputes a range
    bucket whose worker dies before the reduce fetches it. `_fault_inject` /
    `_fault_inject_map` are test-only hooks: worker ids to kill after / before the map
    barrier."""
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
            "op": "sort",
            "input": {"op": "scan", "source_id": 0},
            "keys": sort_keys_ir(sort.keys),
            "limit": sort.limit,
        }
    )
    credits = _shuffle_credits()

    import time as _tt

    _ps = _tt.perf_counter()
    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a
    # second placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = shuffle_partitions(workers)
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
        parts = partition_descriptors(
            sources[sid],
            workers,
            projection=projection,
            predicate=predicate,
            worker_addrs=fleet_addrs,
        )

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
        shape_key = sort_shape_key(map_ir, key_name)
        grids = load_learned_grids(shape_key)
        learned = grids is not None
        if not learned:
            grids, dead = map_barrier(
                workers,
                lambda host, src: actors[host].sample_quantiles.remote(
                    map_ir, key_name, _SAMPLE_PROBS, parts[src]
                ),
                dead=dead,
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
        _phase("sample", _t.perf_counter() - _s, buckets=n_buckets, learned=learned)

        # MAP: range-partition each split by the boundaries and publish raw rows.
        _s = _t.perf_counter()
        mapper_addrs, dead = map_barrier(
            workers,
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
            ),
            dead=dead,
        )
        _phase("map_range_publish", _t.perf_counter() - _s)

        # Placed HERE, as soon as the buckets exist and before anything can take a worker
        # away — replicating after a loss would be probing a corpse. A sort's buckets are
        # raw rows rather than pre-aggregated state, so the copy is larger than the
        # aggregate's; it is still cheaper than re-reading the source and re-running the
        # sample + range partition. `None` (the default factor of 1) leaves the reduce
        # byte-identical to the unreplicated path.
        replicas = replicate_shuffle_output(actors, mapper_addrs, n_buckets, workers, dead)

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
        )
        _phase("reduce_gather_sort", _t.perf_counter() - _s)
    finally:
        release_fleet(actors, pg, owns)

    # Concatenate the ranges in leading-key order (reversed for a descending sort) —
    # each bucket is globally ordered relative to the others, so no final merge.
    _pc = _tt.perf_counter()
    order = range(workers - 1, -1, -1) if desc else range(workers)
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
):
    """Run the sort reduce under recompute-on-worker-loss recovery.

    Returns a `{bucket_id: sorted_batches}` dict so the driver can concatenate the
    ranges in key order regardless of completion order. A reducer reporting an
    unreachable mapper fetches the byte-identical bucket from a `replicas` survivor;
    only a source whose every copy is gone drives a recompute of that worker's range
    bucket from its on-disk source partition onto a survivor, then a retry.
    """
    import ray

    from batcher.dist.executors.ray_runtime import run_bucket_reduce

    def remote_reduce(host: int, bucket: int):
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
            )
        )

    done = run_bucket_reduce(
        kind="sort",
        n_buckets=n_buckets,
        workers=workers,
        actors=actors,
        remote_reduce=remote_reduce,
        republish=republish,
        dead=dead,
        mapper_addrs=addrs,
        replicas=replicas,
    )
    # Keyed by bucket so the driver concatenates ranges in key order; an "ok" reduce over an
    # empty range returns None, coerced to [] so the concatenation never sees a hole.
    return {bucket: (payload or []) for bucket, payload in done.items()}
