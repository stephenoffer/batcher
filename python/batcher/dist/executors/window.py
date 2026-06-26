"""Distributed window functions over a disk Arrow-IPC shuffle.

Window functions are computed *per partition*, so hash-shuffling input rows by the
window's partition keys co-locates every whole partition on a single reducer; the
reducer runs the ordinary window operator over its rows and the concatenation of
all reducers is identical to single-node execution. Unlike the aggregate shuffle
(which moves partial *state*), this moves the raw rows and reuses the *same* hash
partitioner so a partition is never split across reducers.

Restricted to windows whose partition keys are plain columns and whose input is a
breaker-free single source; anything else falls back to single-node.
"""

from __future__ import annotations

import json
import tempfile

import pyarrow as pa

from batcher.dist.executors.partition_io import _apply_above, _partition_source
from batcher.dist.executors.plan_analysis import _relabel_single_source
from batcher.dist.executors.ray_runtime import _ensure_ray, _rmtree, engine_config_json
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Window


def _distributed_window(
    above: list[LogicalPlan], window: Window, sources: list[Source], workers: int
) -> pa.Table:
    """Run `window` across `workers` by hash-shuffling rows by its partition keys."""
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import speculation_policy

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers

    # Partition-key column positions in the window input's output (caller guarantees
    # every partition key is a plain `Col`).
    cols = window.input.available_columns()
    pk_indices = [cols.index(k.name) for k in window.partition_keys]

    map_plan, source_id = _relabel_single_source(window.input)
    map_ir = json.dumps(map_plan.to_ir())
    # The reduce runs the window over its bucket (a single in-memory source 0).
    win_ir = window.to_ir()
    win_ir["input"] = {"op": "scan", "source_id": 0}
    win_json = json.dumps(win_ir)
    n_reducers = workers

    work_dir = tempfile.mkdtemp(prefix="batcher_winshuffle_")
    try:
        partitions = _partition_source(sources[source_id], workers, work_dir)
        pol = speculation_policy()

        # Each task is a pure function of its partition file, so a straggler can be
        # backed up (deterministic → identical output); `gather_with_backups` is a
        # plain barrier when speculation is disabled (the default) — matching the
        # resilience wrapper every other disk-shuffle operator uses.
        def _map_for(mid: int):
            return _map_task.remote(
                map_ir, json.dumps(pk_indices), partitions[mid], n_reducers, work_dir, mid, cfg_json
            )

        shuffle_paths = gather_with_backups(
            [_map_for(mid) for mid in range(len(partitions))], _map_for, pol
        )  # shuffle_paths[mapper][reducer] = path

        def _reduce_for(r: int):
            return _reduce_task.remote(
                win_json, [paths[r] for paths in shuffle_paths], work_dir, r, cfg_json
            )

        result_paths = gather_with_backups(
            [_reduce_for(r) for r in range(n_reducers)], _reduce_for, pol
        )

        from batcher.dist.shuffle_io import read_ipc

        out_batches: list[pa.RecordBatch] = []
        for p in result_paths:
            if p is not None:
                out_batches.extend(read_ipc(p))
    finally:
        _rmtree(work_dir)

    table = (
        pa.Table.from_batches(out_batches)
        if out_batches
        else pa.table({c: [] for c in window.available_columns()})
    )
    return table if not above else _apply_above(above, table)


def _map_task(map_ir, pk_indices_json, part_path, n_reducers, work_dir, mapper_id, engine_config):
    import os as _os

    import batcher._native as nat
    from batcher.dist.executors.partition_io import read_partition
    from batcher.dist.shuffle_io import write_ipc

    rows = nat.execute_plan(map_ir, [read_partition(part_path)], engine_config)
    pk_indices = json.loads(pk_indices_json)
    buckets = nat.partition_batches(rows, pk_indices, n_reducers)
    paths = []
    for r, bucket in enumerate(buckets):
        path = _os.path.join(work_dir, f"wm{mapper_id}_r{r}.arrow")
        write_ipc(bucket, path)
        paths.append(path)
    return paths


def _reduce_task(win_json, input_paths, work_dir, reducer_id, engine_config):
    import os as _os

    import batcher._native as nat
    from batcher.dist.shuffle_io import read_ipc, write_ipc

    batches: list = []
    for path in input_paths:
        batches.extend(read_ipc(path))
    if not batches:
        return None
    # The whole window partition for every key in this bucket is present, so the
    # window operator computes the same values it would single-node.
    result = nat.execute_plan(win_json, [batches], engine_config)
    if not result or sum(b.num_rows for b in result) == 0:
        return None
    path = _os.path.join(work_dir, f"winreduce_{reducer_id}.arrow")
    write_ipc(result, path)
    return path
