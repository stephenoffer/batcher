"""Distributed *global* (no-``PARTITION BY``) ordered window over an Arrow Flight shuffle.

``row_number() OVER (ORDER BY t)`` and running aggregates need one global row order, so the
hash shuffle every partitioned window uses is unavailable: there is one partition and it
cannot be split by key. Until this module, that made an ordered global window the one window
shape with **no distributed path at all** -- over splittable data it raised rather than
silently running the whole relation on one node.

It splits along the *order* instead. Range-partition the rows by the single order key into
buckets that are ordered relative to each other (the same sample-and-cut the distributed sort
uses, so equal keys never span a boundary), window each bucket on its own worker, then shift
each bucket's result by the prior buckets' contribution. The offset algebra, and the exact
set of functions it covers, is `offsets` -- shared with the single-node streamer so the
distributed answer and the streamed answer are computed by one implementation rather than two.

What scales, and what doesn't. The read, the range partition and the per-bucket window kernel
-- every superlinear term -- run once per bucket across the fleet, so they divide by the
worker count. What stays central is one vectorized ``add``/``if_else`` per bucket on the
driver, because the offsets are a prefix scan over buckets and each bucket's shift is a
single scalar. That pass is linear in the rows the driver was already going to concatenate
for `collect`, so it costs a constant factor on top of the gather, not a second sort.

Rows move node-to-node over credit-bounded Flight and never through the Ray object store;
only ``(addr, ticket)`` strings transit Ray. The per-worker Flight endpoint, credit window and
`_FlightWorker` actor are the shared ones from `flight_aggregate`, and worker loss is survived
in both phases exactly as the sort's range shuffle survives it.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import (
    merge_boundaries,
    partition_descriptors,
    sample_probs,
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
from batcher.dist.global_window.offsets import (
    OrderedBucketOffsets,
    bucket_order,
    inject_avg_helpers,
)
from batcher.dist.shuffle_replication import replicate_shuffle_output, retire_replicas
from batcher.dist.sort_boundaries import load_learned_grids, persist_grids, sort_shape_key
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Window

__all__ = ["execute_global_window_flight"]


def execute_global_window_flight(
    above: list[LogicalPlan],
    window: Window,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    _fault_inject_map: set[int] | None = None,
) -> pa.Table:
    """Range-partition by the order key over Flight, window per bucket, offset on the driver.

    `_fault_inject` / `_fault_inject_map` are test-only hooks: worker ids to kill after /
    before the map barrier, matching the sort and hash-window paths.
    """
    import ray

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    key = window.order_keys[0]  # caller guarantees a single plain-column order key
    key_name = key.expr.name
    desc, nulls_first = key.descending, key.nulls_first

    map_plan, sid = _relabel_single_source(window.input)
    map_ir = json.dumps(map_plan.to_ir())
    # The reduce runs the window over its bucket as a single in-memory source 0. `to_ir()`
    # memoizes and hands back the plan's shared dict/list, so copy the pieces rewritten here
    # before touching them — mutating the cached structures would corrupt every later use of
    # the same plan.
    win_ir = dict(window.to_ir())
    win_ir["input"] = {"op": "scan", "source_id": 0}
    win_ir["functions"] = list(win_ir["functions"])
    # `avg` is offset through its running sum and count, so ask the kernel for those two
    # alongside it under private aliases; the driver reads them back per bucket and drops
    # them before the rows are returned, so the output schema is unchanged.
    avg_helpers = inject_avg_helpers(window, win_ir)
    win_json = json.dumps(win_ir)
    credits = _shuffle_credits()

    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a second
    # placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = shuffle_partitions(workers)
    try:
        # Read only the columns/rows the window's map prefix needs. `map_plan`'s scan was
        # relabeled to source 0, so key the analysis on 0, not on the source's original index.
        projection, predicate = source_pushdown(map_plan, 0)
        parts = partition_descriptors(
            sources[sid],
            workers,
            projection=projection,
            predicate=predicate,
            worker_addrs=fleet_addrs,
        )

        if _fault_inject_map:  # test hook: kill before the barrier, so nothing publishes
            for i in _fault_inject_map:
                ray.kill(actors[i])

        # One ticket stage for THIS window's shuffle. A literal 0 would collide with any
        # other shuffle of the same query publishing byte-identical tickets, which is how a
        # reducer ends up reading another operator's buckets. See `fleet.plan_id`.
        stage_base = next_stage_base(1)
        dead: set[int] = set()

        # SAMPLE: each worker samples its own split's order-key distribution, so the input is
        # never read on the driver. A learned grid for this shape skips the pass entirely; it
        # is safe even when stale, because boundaries decide only which bucket a row lands in
        # and the offset algebra is correct for any monotone boundary list.
        shape_key = sort_shape_key(map_ir, key_name)
        grids = load_learned_grids(shape_key)
        if grids is None:
            # Sized against the bucket count, as the sort's is: a boundary is placed to
            # within `1/g` of a sampler's rows, so cutting more buckets than there are
            # samplers needs a finer grid to keep the ranges even. Precision affects only
            # balance -- any monotone boundary list keeps equal keys together, which is all
            # the offset algebra needs.
            probs = sample_probs(n_buckets, workers)
            grids, dead = map_barrier(
                workers,
                lambda host, src: actors[host].sample_quantiles.remote(
                    map_ir, key_name, probs, parts[src]
                ),
                dead=dead,
            )
            persist_grids(shape_key, grids)
        # Cut into exactly `n_buckets` ranges: `shuffle_partitions` can trim the reducer count
        # below `workers`, and boundaries sized for `workers` would route rows past the last
        # bucket and panic the range partitioner.
        boundaries = merge_boundaries(grids, n_buckets)

        # MAP: range-partition each split by the boundaries and publish each bucket.
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

        # Placed HERE, as soon as the buckets exist and before anything can take a worker
        # away — replicating after a loss would be probing a corpse.
        replicas = replicate_shuffle_output(actors, mapper_addrs, n_buckets, workers, dead)

        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        results = _window_reduce_with_recovery(
            actors,
            mapper_addrs,
            parts,
            map_ir,
            key_name,
            boundaries,
            win_json,
            nulls_first,
            desc,
            n_buckets,
            workers,
            dead=dead,
            replicas=replicas,
            stage_base=stage_base,
        )
    finally:
        release_fleet(actors, pg, owns)

    # Walk the buckets in global key order and shift each one's window columns to their
    # global values. Buckets are ordered relative to each other by construction, so this
    # prefix scan is the whole of what the driver has to do sequentially.
    offsets = OrderedBucketOffsets(window, avg_helpers)
    out: list[pa.RecordBatch] = []
    for b in bucket_order(n_buckets, desc):
        batches = [x for x in results.get(b, []) if x.num_rows > 0]
        if not batches:
            continue
        out.extend(offsets.apply(pa.Table.from_batches(batches)).to_batches())
    table = (
        pa.Table.from_batches(out)
        if out
        else empty_result_table(window, window.available_columns())
    )
    return table if not above else _apply_above(above, table)


def _window_reduce_with_recovery(
    actors,
    addrs,
    parts,
    map_ir,
    key_name,
    boundaries,
    win_json,
    nulls_first,
    desc,
    n_buckets,
    workers,
    dead=None,
    replicas=None,
    stage_base=0,
):
    """Run the per-bucket window reduce under recompute-on-worker-loss recovery.

    Returns a ``{bucket_id: windowed_batches}`` dict so the driver applies the offsets in
    key order regardless of completion order. A reducer reporting an unreachable mapper
    fetches the byte-identical bucket from a `replicas` survivor; only a source whose every
    copy is gone drives a recompute of that worker's range bucket onto a survivor, then a
    retry — matching the sort's range shuffle exactly.
    """
    import ray

    from batcher.dist.executors.ray_runtime import run_bucket_reduce

    def remote_reduce(host: int, bucket: int):
        # `reduce_window` fetches the bucket and runs the given plan over it; the plan here
        # is the window IR, the same one the single-node kernel would run over these rows.
        return actors[host].reduce_window.remote(
            win_json, addrs, bucket, None, replicas, current_plan_id(), stage_base
        )

    def republish(target: int, src: int) -> None:
        # Retire before republishing — a replica taken at the old epoch reads back as an
        # empty bucket, not an error. See `dist/shuffle_replication.py::retire_replicas`.
        retire_replicas(replicas, src, target, "global_window")
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
        kind="global_window",
        n_buckets=n_buckets,
        workers=workers,
        actors=actors,
        remote_reduce=remote_reduce,
        republish=republish,
        dead=dead,
        mapper_addrs=addrs,
        replicas=replicas,
    )
    # Keyed by bucket so the driver offsets ranges in key order; an "ok" reduce over an empty
    # range returns None, coerced to [] so the walk never sees a hole.
    return {bucket: (payload or []) for bucket, payload in done.items()}
