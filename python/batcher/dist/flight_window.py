"""Distributed window functions over an Arrow Flight shuffle (object store bypassed).

Hash-shuffles the raw input rows by the window's partition keys so every whole
partition lands on one reducer, which runs the ordinary window operator over its
rows — identical to single-node. Like the aggregate/join Flight paths, only
`(addr, ticket)` strings (and the small results) transit Ray; the rows move
node→node over credit-bounded Flight, never through the object store. The
per-worker Flight endpoint, credit window, and `_FlightWorker` actor are the shared
ones from `flight_aggregate`; a lost worker's bucket is recomputed from its source
partition (still on disk) on a survivor — the same Spark-style lineage recovery.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
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
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Window

__all__ = ["execute_window_flight"]


def execute_window_flight(
    above: list[LogicalPlan],
    window: Window,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    _fault_inject_map: set[int] | None = None,
) -> pa.Table:
    """Hash-shuffle rows by the window's partition keys over Flight, window per bucket.

    Mappers publish their key-hashed row buckets on their own Flight servers
    (one stage per shuffle); reducer r fetches bucket r from every mapper and runs the
    window operator over the whole partition. Worker loss is survived in both phases:
    `map_barrier` relocates a source whose worker dies while mapping, `ShuffleRecovery`
    recomputes one whose worker dies before the reduce fetches it. `_fault_inject` /
    `_fault_inject_map` are test-only hooks: worker ids to kill after / before the map
    barrier."""
    import ray

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    # Caller guarantees every partition key is a plain column; shuffle by their names.
    key_names = [k.name for k in window.partition_keys]
    map_plan, sid = _relabel_single_source(window.input)
    map_ir = json.dumps(map_plan.to_ir())
    # The reduce runs the window over its bucket as a single in-memory source 0.
    win_ir = window.to_ir()
    win_ir["input"] = {"op": "scan", "source_id": 0}
    win_json = json.dumps(win_ir)
    credits = _shuffle_credits()

    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a
    # second placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = shuffle_partitions(workers)
    try:
        # Read only the columns/rows the window's map prefix needs (see flight_aggregate).
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

        if _fault_inject_map:  # test hook: kill before the barrier, so nothing publishes
            for i in _fault_inject_map:
                ray.kill(actors[i])

        # MAP barrier under worker-loss recovery: a worker preempted while mapping has its
        # source republished on a survivor under the same `src`, so the reducers' tickets
        # still resolve. A bare `ray.get` here failed the whole query on one preemption —
        # in the map phase, which reads the source and dominates the query's runtime.
        # One ticket stage for THIS window's shuffle. It used to be the literal 0, so a
        # window sharing a fleet with another shuffle of the same query published
        # byte-identical tickets and one overwrote the other — the collision that made a
        # join read another join's buckets. See `fleet.plan_id.next_stage_base`.
        stage = next_stage_base(1)
        addrs, dead = map_barrier(
            workers,
            lambda host, src: actors[host].map_publish_raw.remote(
                map_ir, key_names, parts[src], n_buckets, stage, src, 0, current_plan_id()
            ),
        )

        # Placed HERE, as soon as the buckets exist and before anything can take a worker
        # away — replicating after a loss would be probing a corpse. `None` (the default
        # factor of 1) leaves the reduce byte-identical to the unreplicated path.
        replicas = replicate_shuffle_output(actors, addrs, n_buckets, workers, dead)

        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        batches = _window_reduce_with_recovery(
            actors,
            addrs,
            parts,
            map_ir,
            key_names,
            win_json,
            n_buckets,
            workers,
            dead=dead,
            replicas=replicas,
            stage=stage,
        )
    finally:
        release_fleet(actors, pg, owns)

    table = (
        pa.Table.from_batches(batches)
        if batches
        else pa.table({c: [] for c in window.available_columns()})
    )
    return table if not above else _apply_above(above, table)


def _window_reduce_with_recovery(
    actors,
    addrs,
    parts,
    map_ir,
    key_names,
    win_json,
    n_buckets,
    workers,
    dead=None,
    replicas=None,
    stage=0,
):
    """Run the window reduce under recompute-on-worker-loss recovery.

    A reducer that reports an unreachable mapper (or whose host died) fetches the
    byte-identical bucket from a `replicas` survivor; only a source whose every copy is
    gone drives a recompute from its on-disk source partition onto a survivor, then a
    retry — matching the aggregate/join paths. Returns the windowed batches. `dead` seeds
    the workers the map barrier already lost.
    """
    import ray

    from batcher.dist.executors.ray_runtime import run_bucket_reduce

    def remote_reduce(host: int, bucket: int):
        return actors[host].reduce_window.remote(
            win_json, addrs, bucket, None, replicas, current_plan_id(), stage
        )

    def republish(target: int, src: int) -> None:
        # Retire before republishing — a replica taken at the old epoch reads back as an
        # empty bucket, not an error. See `dist/shuffle_replication.py::retire_replicas`.
        retire_replicas(replicas, src, target, "window")
        addrs[src] = ray.get(
            actors[target].map_publish_raw.remote(
                map_ir, key_names, parts[src], n_buckets, stage, src, 0, current_plan_id()
            )
        )

    done = run_bucket_reduce(
        kind="window",
        n_buckets=n_buckets,
        workers=workers,
        actors=actors,
        remote_reduce=remote_reduce,
        republish=republish,
        dead=dead,
        mapper_addrs=addrs,
        replicas=replicas,
    )
    out: list[pa.RecordBatch] = []
    for res in done.values():
        if res:
            out.extend(b for b in res if b.num_rows > 0)
    return out
