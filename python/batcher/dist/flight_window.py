"""Keyed row shuffle over an Arrow Flight shuffle (object store bypassed).

Hash-shuffles raw rows by key columns so every row of a key lands on one reducer, which then
runs the ordinary single-node operator over its bucket — identical to single-node. Like the
aggregate/join Flight paths, only `(addr, ticket)` strings (and the small results) transit
Ray; the rows move node→node over credit-bounded Flight, never through the object store. The
per-worker Flight endpoint, credit window, and `_FlightWorker` actor are the shared ones from
`flight_aggregate`; a lost worker's bucket is recomputed from its source partition (still on
disk) on a survivor — the same Spark-style lineage recovery.

Two operators distribute this way rather than by moving partial aggregate state: a
**partitioned window**, whose kernel needs every row of a partition at once, and a **keyed
dedup**, whose answer is one of the rows itself. They differ only in the map plan — a window
ships its raw rows, a dedup runs the whole dedup on its own partition first and ships one row
per key, which is most of the shuffle volume gone — so `_execute_keyed_flight` below takes both
plans as IR and cares about neither. The disk-transport counterpart is
`executors/keyed_shuffle.py`.

The module keeps its window-named path deliberately: `execute_window_flight` and the internals
around it are monkeypatch targets in the replication and recovery suites, and a patch that
follows a moved name silently stops applying while the test keeps passing.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher.carbonite.resilience import SourcePlacement
from batcher.dist.adaptive_sizing import row_shuffle_reducer_count
from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
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
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Window

__all__ = ["execute_keyed_shuffle_flight", "execute_window_flight"]


def execute_window_flight(
    above: list[LogicalPlan],
    window: Window,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    _fault_inject_map: set[int] | None = None,
    hub=None,
    metrics_out=None,
    materialize: bool = True,
):
    """Hash-shuffle rows by the window's partition keys over Flight, window per bucket.

    Mappers publish their key-hashed row buckets on their own Flight servers
    (one stage per shuffle); reducer r fetches bucket r from every mapper and runs the
    window operator over the whole partition. Worker loss is survived in both phases:
    `map_barrier` relocates a source whose worker dies while mapping, `ShuffleRecovery`
    recomputes one whose worker dies before the reduce fetches it. `_fault_inject` /
    `_fault_inject_map` are test-only hooks: worker ids to kill after / before the map
    barrier.

    `materialize=False` leaves each windowed bucket on the worker that computed it and
    returns a `FlightMaterializedSource`. A window emits one row per input row, so the
    collect it replaces moves the entire relation through the driver."""
    # Caller guarantees every partition key is a plain column; shuffle by their names.
    map_plan, sid = _relabel_single_source(window.input)
    return _execute_keyed_flight(
        above,
        map_plan=map_plan,
        reduce_ir=_scan_rooted_ir(window),
        key_names=[k.name for k in window.partition_keys],
        out_schema=empty_result_table(window, window.available_columns()).schema,
        sources=sources,
        source_id=sid,
        workers=workers,
        _fault_inject=_fault_inject,
        _fault_inject_map=_fault_inject_map,
        hub=hub,
        metrics_out=metrics_out,
        materialize=materialize,
    )


def execute_keyed_shuffle_flight(
    above: list[LogicalPlan],
    *,
    map_plan: LogicalPlan,
    reduce_ir: str,
    key_names: list[str],
    out_schema: pa.Schema,
    sources: list[Source],
    source_id: int,
    workers: int,
    hub=None,
    metrics_out=None,
    materialize: bool = True,
):
    """Run an already-decomposed keyed row shuffle over Flight.

    The caller supplies the map plan (which for a mergeable reduction is the operator itself,
    so each mapper pre-reduces its own partition) and the reduce plan rooted on a scan of the
    bucket. Used by the keyed dedup; see `executors/distinct.py`.

    Args:
        above: Operators above the shuffled one; run single-node on the collected result.
        map_plan: The per-partition map plan, already relabeled to read source 0.
        reduce_ir: The reducer's plan as JSON IR, rooted on a scan of its bucket.
        key_names: Columns to hash-shuffle by.
        out_schema: The operator's output schema, carrying its real column *types* —
            what a shuffle whose every bucket came back empty returns. See
            `executors.keyed_shuffle.keyed_row_shuffle` for why names alone were wrong.
        sources: The query's bound sources.
        source_id: Which of them `map_plan` reads.
        workers: How many workers to spread the map phase over.
        hub: Metadata hub the workers' measured operator metrics are recorded into.
        metrics_out: When given, each worker's `ExecMetrics` document is appended to it.
        materialize: `False` to leave each reducer's bucket on its worker and hand back a
            `FlightMaterializedSource` the next stage reads in place. Only honored when
            nothing is stacked `above`.

    Returns:
        The collected result table with `above` applied, or — under `materialize=False`
        with no `above` — a `FlightMaterializedSource` over the reducers' buckets.
    """
    return _execute_keyed_flight(
        above,
        map_plan=map_plan,
        reduce_ir=reduce_ir,
        key_names=key_names,
        out_schema=out_schema,
        sources=sources,
        source_id=source_id,
        workers=workers,
        hub=hub,
        metrics_out=metrics_out,
        materialize=materialize,
    )


def _scan_rooted_ir(node: LogicalPlan) -> str:
    """`node`'s IR with its input replaced by a scan of source 0 — the reduce-side plan."""
    ir = node.to_ir()
    ir["input"] = {"op": "scan", "source_id": 0}
    return json.dumps(ir)


def _execute_keyed_flight(
    above: list[LogicalPlan],
    *,
    map_plan: LogicalPlan,
    reduce_ir: str,
    key_names: list[str],
    out_schema: pa.Schema,
    sources: list[Source],
    source_id: int,
    workers: int,
    _fault_inject: set[int] | None = None,
    _fault_inject_map: set[int] | None = None,
    hub=None,
    metrics_out=None,
    materialize: bool = True,
):
    """The shared driver: shuffle rows by `key_names`, run `reduce_ir` per bucket."""
    import ray

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    sid = source_id
    map_ir = json.dumps(map_plan.to_ir())
    win_json = reduce_ir
    credits = _shuffle_credits()

    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a
    # second placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    # A window exchanges raw rows and materializes a whole partition-run per bucket, so the
    # bucket count is what bounds that run. Sized by volume, never below the floor.
    n_buckets = row_shuffle_reducer_count(map_plan, shuffle_partitions(workers), sources, source_id)
    # There is nothing to apply `above` to once the buckets stay on the workers, so a
    # stacked operator keeps the collect.
    publish = materialize is False and not above
    keep_actors = False  # set when a FlightMaterializedSource takes ownership of them
    try:
        # Read only the columns/rows the window's map prefix needs (see flight_aggregate).
        # `map_plan`'s scan was relabeled to source 0, so key the analysis on 0, not on the
        # source's original index: a staged plan whose input is an intermediate (source id >
        # 0) missed the lookup and silently read every column.
        projection, predicate = source_pushdown(map_plan, 0)
        # More map partitions than workers where the source has splits to fill them, so a
        # straggler holds a fraction of a node's share rather than all of it (see
        # `map_partitions`). `len(parts)` is the source count from here on.
        parts = partition_descriptors(
            sources[sid],
            workers,
            projection=projection,
            predicate=predicate,
            worker_addrs=fleet_addrs,
            max_partitions=map_partitions(workers),
        )
        n_sources = len(parts)
        placement = SourcePlacement(workers)

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
            n_sources,
            lambda host, src: actors[host].map_publish_raw.remote(
                map_ir, key_names, parts[src], n_buckets, stage, src, 0, current_plan_id()
            ),
            workers=workers,
            placement=placement,
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
            placement=placement,
            publish=publish,
        )
        if publish:
            from batcher.dist.fleet import FlightMaterializedSource

            # `batches` is the reducers' `(addr, ticket, rows, schema)` handles; an empty
            # bucket published nothing and is simply absent. A window's and a dedup's
            # results are both multisets, so bucket order carries no meaning.
            handles = [(addr, ticket, rows) for addr, ticket, rows, _schema in batches]
            schema = batches[0][3] if batches else out_schema
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

    table = (
        pa.Table.from_batches(batches) if batches else pa.Table.from_batches([], schema=out_schema)
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
    placement=None,
    publish=False,
):
    """Run the window reduce under recompute-on-worker-loss recovery.

    A reducer that reports an unreachable mapper (or whose host died) fetches the
    byte-identical bucket from a `replicas` survivor; only a source whose every copy is
    gone drives a recompute from its on-disk source partition onto a survivor, then a
    retry — matching the aggregate/join paths. Returns the windowed batches, or — with
    `publish` — the `(addr, ticket, rows, schema)` handles of the buckets left ON their
    workers. `dead` seeds the workers the map barrier already lost.
    """
    import ray

    from batcher.dist.executors.ray_runtime import run_bucket_reduce

    # One stage id for every bucket of THIS published result, so two materialized
    # intermediates in one query cannot share a ticket (`fleet.plan_id.next_result_stage`).
    result_stage = next_result_stage() if publish else 0

    def remote_reduce(host: int, bucket: int):
        if publish:
            return actors[host].reduce_window_publish.remote(
                win_json, addrs, bucket, None, replicas, current_plan_id(), stage, result_stage
            )
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
        placement=placement,
    )
    if publish:
        # One `(addr, ticket, rows, schema)` handle per non-empty bucket, still on its worker.
        return [h for h in done.values() if h]
    out: list[pa.RecordBatch] = []
    for res in done.values():
        if res:
            out.extend(b for b in res if b.num_rows > 0)
    return out
