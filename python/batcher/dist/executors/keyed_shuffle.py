"""Shuffle raw rows by key columns, then run one plan per bucket.

Two operators distribute this way rather than by moving partial aggregate *state*: a
partitioned window, whose kernel needs every row of a partition at once, and a keyed dedup,
whose surviving row is one of the rows themselves. Both hash-shuffle by their key columns —
so equal keys land on one reducer and a key is never split — and both then run the ordinary
single-node operator over the bucket. The concatenation of the reducers is what single-node
execution produces.

What differs between them is only the *map* plan. A window has nothing useful to do before
the shuffle, so it ships its raw rows. A dedup's reduction is mergeable, so the map side runs
the whole dedup on its own partition first and ships one row per key per partition — usually
most of the shuffle volume gone, and the reason a keyed dedup scales with workers instead of
with the network. The driver below takes both plans as IR and cares about neither.

The shuffle itself is Arrow IPC files on shared storage; the Flight transport is the same
decomposition over credit-bounded streams, in `dist/flight_window.py` (which keeps its
window-named path because the replication suite monkeypatches into it).
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.dist.executors.partition_io import _apply_above, _partition_source
from batcher.dist.executors.ray_runtime import (
    _ensure_ray,
    _rmtree,
    engine_config_json,
    record_worker_metrics,
    shuffle_partitions,
)
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["keyed_row_shuffle", "scan_rooted_ir"]


def scan_rooted_ir(node: LogicalPlan) -> str:
    """`node`'s IR with its input replaced by a scan of source 0 — the reduce-side plan.

    A reducer holds its bucket as one in-memory relation, so the operator it runs is the
    plan node re-rooted on that relation. Going through `to_ir()` and swapping the `input`
    keeps the operator's own encoding in `plan`, where the wire contract lives.
    """
    ir = node.to_ir()
    ir["input"] = {"op": "scan", "source_id": 0}
    return json.dumps(ir)


def keyed_row_shuffle(
    above: list[LogicalPlan],
    *,
    map_plan: LogicalPlan,
    reduce_ir: str,
    key_indices: list[int],
    out_schema: pa.Schema,
    source: Source,
    workers: int,
    hub=None,
    tag: str,
    metrics_out=None,
    materialize: bool = True,
):
    """Hash-shuffle `map_plan`'s rows by `key_indices`, run `reduce_ir` per bucket, collect.

    Args:
        above: Operators that sit above the shuffled one; run single-node on the result.
        map_plan: The per-partition map plan, already relabeled to read source 0.
        reduce_ir: The reducer's plan, rooted on a scan of its bucket (`scan_rooted_ir`).
        key_indices: Column positions of the shuffle key in `map_plan`'s output.
        out_schema: The operator's output schema, carrying its real column *types*. A
            shuffle where every bucket came back empty has no batch to take a schema from,
            and this is what it returns instead. It used to be a list of *names*, from
            which the empty result was built as `pa.table({name: []})` — every column
            typed `null`. That is a `distributed != single-node` divergence in column
            types on exactly the empty case, which is where such a divergence hides
            longest: a downstream concat, typed projection or `write.parquet` then breaks
            only when the filter happened to match nothing.
        source: The source to partition across mappers.
        workers: How many workers to spread the map phase over.
        hub: Metadata hub the workers' measured operator metrics are recorded into.
        tag: Short name for this operator, used in scratch filenames and metrics.
        materialize: `False` to hand back the reducers' output as a `MaterializedSource`
            the next stage scans in place, instead of collecting it here. A window is
            row-preserving and a keyed dedup is close to it, so what this path collects is
            the size of the *relation*, not of a summary — a Θ(N) term on one node, and the
            largest one left in the distributed executor. Only honored when nothing is
            stacked `above` (there is nothing to apply the operators to otherwise).
        metrics_out: When given, each worker's `ExecMetrics` document is appended to it for
            the conductor's `QueryProfile`. The hub always learns from these workers; without
            this channel the *profile* did not, so a distributed window or dedup reported no
            per-operator detail and no machine cost at all — unlike the aggregate route,
            which has carried it all along.

    Returns:
        The collected result table with `above` applied, or — under `materialize=False`
        with no `above` — a `MaterializedSource` over the reducers' output, which owns
        `work_dir` and reclaims it on `cleanup()`.
    """
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.shuffle_io import distributed_work_dir

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers
    map_ir = json.dumps(map_plan.to_ir())
    n_reducers = shuffle_partitions(workers)

    work_dir = distributed_work_dir(f"batcher_{tag}shuffle_")
    keep_dir = False  # set when a MaterializedSource takes ownership of work_dir
    try:
        partitions = _partition_source(source, workers, work_dir)
        pol = speculation_policy()

        # Each task is a pure function of its partition file, so a straggler can be backed
        # up (deterministic → identical output); `gather_with_backups` is a plain barrier
        # when speculation is disabled (the default), matching every other disk shuffle.
        def _map_for(mid: int):
            return _map_task.remote(
                map_ir,
                json.dumps(key_indices),
                partitions[mid],
                n_reducers,
                work_dir,
                mid,
                cfg_json,
                tag,
            )

        map_results = gather_with_backups(
            [_map_for(mid) for mid in range(len(partitions))], _map_for, pol, stage="shuffle.map"
        )
        shuffle_paths = [paths for paths, _metrics in map_results]  # [mapper][reducer] = path
        record_worker_metrics(hub, (m for _paths, m in map_results), metrics_out)

        def _reduce_for(r: int):
            return _reduce_task.remote(
                reduce_ir, [paths[r] for paths in shuffle_paths], work_dir, r, cfg_json, tag
            )

        reduce_results = gather_with_backups(
            [_reduce_for(r) for r in range(n_reducers)], _reduce_for, pol, stage="shuffle.reduce"
        )
        # The reduce runs the operator over a whole key partition — the breaker whose
        # measured time and peak bytes the cost model and memory model are fit from.
        record_worker_metrics(hub, (m for _p, _n, m in reduce_results), metrics_out)

        if not materialize and not above:
            from batcher.dist.executors.partition_io import materialize_reduce_output

            keep_dir = True
            return materialize_reduce_output(
                [(p, n) for p, n, _m in reduce_results], work_dir, out_schema
            )

        from batcher.dist.shuffle_io import read_ipc

        out_batches: list[pa.RecordBatch] = []
        for p, _n, _m in reduce_results:
            if p is not None:
                out_batches.extend(read_ipc(p))
    finally:
        if not keep_dir:
            _rmtree(work_dir)

    table = (
        pa.Table.from_batches(out_batches)
        if out_batches
        else pa.Table.from_batches([], schema=out_schema)
    )
    return table if not above else _apply_above(above, table)


def _map_task(
    map_ir, key_indices_json, part_path, n_reducers, work_dir, mapper_id, engine_config, tag
):
    nat = engine()
    from batcher.dist.executors.partition_io import read_partition
    from batcher.dist.executors.ray_runtime import execute_metered
    from batcher.dist.shuffle_io import write_shuffle_buckets

    rows, metrics_json = execute_metered(map_ir, [read_partition(part_path)], engine_config)
    buckets = nat.partition_batches(rows, json.loads(key_indices_json), n_reducers)
    return write_shuffle_buckets(buckets, work_dir, f"{tag}m", mapper_id), metrics_json


def _reduce_task(reduce_ir, input_paths, work_dir, reducer_id, engine_config, tag):
    """Run the operator over this bucket and publish it. Returns `(path, rows, metrics)`.

    The row count is exact and is reported for the same reason the aggregate's reducer
    reports it: a caller keeping the result partitioned needs to size the intermediate
    without reading it back, and reading it back is the thing being avoided."""
    import os as _os

    from batcher.dist.executors.ray_runtime import execute_metered
    from batcher.dist.shuffle_io import read_ipc, write_ipc

    batches: list = []
    for path in input_paths:
        batches.extend(read_ipc(path))
    if not batches:
        return (None, 0, "")
    # Every row of every key in this bucket is present, so the operator computes here
    # exactly what it would single-node.
    result, metrics_json = execute_metered(reduce_ir, [batches], engine_config)
    rows = sum(b.num_rows for b in result) if result else 0
    if rows == 0:
        return (None, 0, metrics_json)
    path = _os.path.join(work_dir, f"{tag}reduce_{reducer_id}.arrow")
    write_ipc(result, path)
    return (path, rows, metrics_json)
