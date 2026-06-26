"""Distributed aggregation over a disk Arrow-IPC shuffle.

Pipeline:  map (run the sub-plan on a source partition → `partial_aggregate`) →
hash-shuffle partial state to disk → reduce (`combine_finalize` per key
partition) → collect → run any post-aggregation operators single-node. The
mergeable primitives are reused verbatim, so the result equals single-node.
"""

from __future__ import annotations

import json
import tempfile

import pyarrow as pa

from batcher.dist.executors.partition_io import (
    _apply_above,
    _empty_agg_table,
    _partition_source,
    source_pushdown,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source
from batcher.dist.executors.ray_runtime import _ensure_ray, _rmtree, engine_config_json
from batcher.io.source import Source
from batcher.plan.logical import Aggregate, LogicalPlan


def _distributed_aggregate(
    above: list[LogicalPlan],
    agg: Aggregate,
    sources: list[Source],
    workers: int,
    hub=None,
    *,
    materialize: bool = True,
    metrics_out=None,
):
    """Distribute `agg` over a disk shuffle. Returns a `pa.Table` (collected), or —
    when ``materialize=False`` and there are no post-aggregate operators — a
    `MaterializedSource` over the reducers' on-disk IPC output, so the adaptive
    executor scans the intermediate in place for the next stage instead of pulling
    every reducer's result back to the driver. Its work_dir is kept alive and owned
    by the returned source's `cleanup()`."""
    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers

    group_keys_json = json.dumps(
        [{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys]
    )
    aggregates_json = json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])
    map_plan, source_id = _relabel_single_source(agg.input)
    map_ir = json.dumps(map_plan.to_ir())
    n_keys = len(agg.group_keys)
    # Global aggregate (no keys) cannot shuffle by key → a single reducer.
    n_reducers = 1 if n_keys == 0 else workers

    # Push the sub-plan's projection + predicate into the source read (the map_ir
    # still re-checks the filter, so this is a pure I/O optimization).
    projection, predicate = source_pushdown(map_plan, 0)

    work_dir = tempfile.mkdtemp(prefix="batcher_shuffle_")
    keep_dir = False  # set when a MaterializedSource takes ownership of work_dir
    try:
        # Resolve and partition the single source into `workers` map inputs.
        partitions = _partition_source(
            sources[source_id], workers, work_dir, projection=projection, predicate=predicate
        )

        from batcher.carbonite.resilience import gather_with_backups
        from batcher.dist.executors.ray_runtime import speculation_policy

        pol = speculation_policy()

        # MAP: run the sub-plan on each partition, partial-aggregate, hash-shuffle.
        # A map task is a pure function of its partition, so a straggler can be
        # backed up (deterministic → identical output); `gather_with_backups` is a
        # plain barrier when speculation is disabled.
        def _map_for(mid: int):
            return _map_task.remote(
                map_ir,
                group_keys_json,
                aggregates_json,
                partitions[mid],
                n_keys,
                n_reducers,
                work_dir,
                mid,
                cfg_json,
            )

        map_refs = [_map_for(mid) for mid in range(len(partitions))]
        # Each mapper returns (per-reducer paths, sub-plan metrics). The driver
        # records the workers' measured operator metrics into the hub so the cost
        # model calibrates from distributed runs too (the measure→consume loop is
        # not single-node-only). Best-effort, by operator kind.
        map_results = gather_with_backups(map_refs, _map_for, pol)
        shuffle_paths = [paths for paths, _metrics in map_results]
        _record_worker_metrics(hub, (m for _paths, m in map_results), metrics_out)

        # REDUCE: each reducer combines+finalizes the partials routed to it.
        def _reduce_for(r: int):
            inputs = [paths[r] for paths in shuffle_paths]
            return _reduce_task.remote(group_keys_json, aggregates_json, inputs, work_dir, r)

        reduce_refs = [_reduce_for(r) for r in range(n_reducers)]
        result_paths = gather_with_backups(reduce_refs, _reduce_for, pol)  # [(path, rows)]

        # Keep the result partitioned on disk for the next adaptive stage: hand back
        # a MaterializedSource over the reducer IPC files (exact row count from the
        # tasks) and skip the read-back/collect entirely. Only when there are no
        # post-aggregate operators (the adaptive stage shape); otherwise collect so
        # `_apply_above` can run them.
        if not materialize and not above:
            from batcher.dist.executors.partition_io import materialize_reduce_output

            keep_dir = True
            return materialize_reduce_output(result_paths, work_dir, _empty_agg_table(agg).schema)

        from batcher.dist.shuffle_io import read_ipc

        agg_batches: list[pa.RecordBatch] = []
        for p, _rows in result_paths:
            if p is not None:
                agg_batches.extend(read_ipc(p))
    finally:
        if not keep_dir:
            _rmtree(work_dir)

    agg_table = pa.Table.from_batches(agg_batches) if agg_batches else _empty_agg_table(agg)

    # Run any post-aggregation operators single-node over the (small) result.
    if not above:
        return agg_table
    return _apply_above(above, agg_table)


def _record_worker_metrics(hub, metrics_jsons, metrics_out=None) -> None:
    """Record distributed workers' sub-plan metrics into the hub (driver side).

    Calibration buckets by operator kind, so the workers' sub-plan-local op_ids need
    no global correlation. When `metrics_out` is given, each worker's parsed op-list is
    also appended to it — the channel the conductor's `QueryProfile` uses to surface the
    distributed map sub-plan (a separate op-id space, shown as its own section). Best-effort
    — never breaks a query."""
    import json

    from batcher.config import active_config

    morsel_rows = active_config().execution.morsel_rows
    for metrics_json in metrics_jsons:
        if not metrics_json:
            continue
        if hub is not None:
            from batcher import core

            core.record_exec_metrics(hub, metrics_json, morsel_rows)
        if metrics_out is not None:
            try:
                metrics_out.append(json.loads(metrics_json).get("ops", []))
            except (ValueError, TypeError):
                pass


def _map_task(
    map_ir,
    group_keys_json,
    aggregates_json,
    part_path,
    n_keys,
    n_reducers,
    work_dir,
    mapper_id,
    engine_config,
):
    import os as _os

    import batcher._native as nat
    from batcher.dist.executors.partition_io import read_partition
    from batcher.dist.shuffle_io import write_ipc

    # Metered: the worker measures its sub-plan's per-operator runtime facts and
    # ships them back so the driver can feed the cost-model calibration loop.
    rows, metrics_json = nat.execute_plan_metered(
        map_ir, [read_partition(part_path)], engine_config
    )
    partial = nat.partial_aggregate(group_keys_json, aggregates_json, rows)

    if n_keys == 0:
        path = _os.path.join(work_dir, f"m{mapper_id}_r0.arrow")
        write_ipc([partial], path)
        return [path], metrics_json

    buckets = nat.partition_batches([partial], list(range(n_keys)), n_reducers)
    paths = []
    for r, bucket in enumerate(buckets):
        path = _os.path.join(work_dir, f"m{mapper_id}_r{r}.arrow")
        write_ipc(bucket, path)
        paths.append(path)
    return paths, metrics_json


def _reduce_task(group_keys_json, aggregates_json, input_paths, work_dir, reducer_id):
    """Combine+finalize the partials routed to this reducer. Returns
    ``(ipc_path, row_count)`` for a non-empty bucket, else ``(None, 0)`` — the exact
    count lets the driver size a materialized intermediate without reading it back."""
    import os as _os

    import batcher._native as nat
    from batcher.dist.shuffle_io import read_ipc, write_ipc

    partials: list = []
    for path in input_paths:
        partials.extend(read_ipc(path))
    if not partials:
        return (None, 0)
    result = nat.combine_finalize(group_keys_json, aggregates_json, partials)
    if result.num_rows == 0:
        return (None, 0)
    path = _os.path.join(work_dir, f"reduce_{reducer_id}.arrow")
    write_ipc([result], path)
    return (path, result.num_rows)
