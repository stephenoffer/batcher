"""Distributed sort over a disk Arrow-IPC shuffle.

Range-partition by the *leading* sort key across workers (equal leading-key values
deterministically land in the same bucket, so no value spans a boundary), sort each
range by *all* sort keys in parallel, then concatenate the ranges in leading-key
order — globally sorted, with no final merge. The range boundaries come from a
**sample pass**: each worker sketches its OWN partition's leading-key quantile grid
via the mergeable KLL sketch (`column_quantiles`), so the input is never read on the
driver — only the small grids cross back, which the driver merges into `workers-1`
boundaries. This mirrors `flight_sort`, but shuffles through driver-local IPC files
instead of Arrow Flight. Single- and multi-key sorts both go through this path (the
leading key must be a plain column to range-partition on).
"""

from __future__ import annotations

import json
import tempfile

import pyarrow as pa

from batcher.dist.executors.partition_io import _apply_above, _partition_source, merge_boundaries
from batcher.dist.executors.plan_analysis import _relabel_single_source
from batcher.dist.executors.ray_runtime import _ensure_ray, _rmtree, engine_config_json
from batcher.dist.shuffle_io import read_ipc
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Sort

# Per-worker CDF sample granularity: a fine grid (33 probe points) so the merged
# boundaries balance the ranges well. Precision affects only balance, not result.
_SAMPLE_PROBS = [i / 32 for i in range(33)]


def _distributed_sort(
    above: list[LogicalPlan], sort: Sort, sources: list[Source], workers: int
) -> pa.Table:
    """Sample boundaries, range-partition each split, sort each range in parallel,
    then concatenate the ranges in leading-key order — globally sorted, no merge."""
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import speculation_policy

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers
    # The leading key drives range partitioning and concatenation order; every key
    # drives the per-bucket sort.
    key = sort.keys[0]
    key_name = key.expr.name
    desc, nulls_first = key.descending, key.nulls_first
    map_plan, sid = _relabel_single_source(sort.input)
    map_ir = json.dumps(map_plan.to_ir())
    sort_ir = json.dumps(
        {
            "op": "sort",
            "input": {"op": "scan", "source_id": 0},
            "keys": [
                {"expr": k.expr.to_ir(), "descending": k.descending, "nulls_first": k.nulls_first}
                for k in sort.keys
            ],
            "limit": sort.limit,
        }
    )
    n_buckets = workers

    work_dir = tempfile.mkdtemp(prefix="batcher_dsort_")
    try:
        # Partition the source into per-worker map inputs (no data read on driver).
        partitions = _partition_source(sources[sid], workers, work_dir)
        pol = speculation_policy()

        # SAMPLE: each worker sketches its own partition's leading-key grid; the
        # driver merges the small grids into range boundaries (rows never cross).
        # Every task is a pure function of its partition file, so a straggler can be
        # backed up (deterministic → identical output); `gather_with_backups` is a
        # plain barrier when speculation is disabled (the default).
        def _sample_for(w: int):
            return _sample_task.remote(map_ir, key_name, _SAMPLE_PROBS, partitions[w], cfg_json)

        grids = gather_with_backups(
            [_sample_for(w) for w in range(len(partitions))], _sample_for, pol
        )
        boundaries = merge_boundaries(grids, workers)

        # MAP: range-partition each split by the boundaries, one IPC file per bucket.
        def _range_for(w: int):
            return _range_task.remote(
                map_ir,
                key_name,
                boundaries,
                n_buckets,
                nulls_first,
                desc,
                partitions[w],
                work_dir,
                w,
                cfg_json,
            )

        map_paths = gather_with_backups(
            [_range_for(w) for w in range(len(partitions))], _range_for, pol
        )

        # REDUCE: each bucket gathers its shard from every mapper, sorts the range.
        def _reduce_for(r: int):
            return _sort_reduce_task.remote(
                sort_ir, [paths[r] for paths in map_paths], work_dir, r, cfg_json
            )

        sorted_paths = gather_with_backups(
            [_reduce_for(r) for r in range(n_buckets)], _reduce_for, pol
        )

        # Concatenate the ranges in leading-key order (reversed for a descending
        # sort) — each bucket is globally ordered relative to the others, no merge.
        order = range(n_buckets - 1, -1, -1) if desc else range(n_buckets)
        out: list[pa.RecordBatch] = []
        for r in order:
            if sorted_paths[r] is not None:
                out.extend(read_ipc(sorted_paths[r]))
        result = (
            pa.Table.from_batches(out)
            if out
            else pa.table({c: [] for c in sort.available_columns()})
        )
        if sort.limit is not None:
            result = result.slice(0, sort.limit)
    finally:
        _rmtree(work_dir)

    return result if not above else _apply_above(above, result)


def _sample_task(map_ir, key_name, probs, part_path, engine_config):
    import batcher._native as nat
    from batcher.dist.executors.partition_io import read_partition

    rows = nat.execute_plan(map_ir, [read_partition(part_path)], engine_config)
    n = sum(b.num_rows for b in rows)
    if n == 0:
        return ([], 0)
    grid = nat.column_quantiles([key_name], rows, list(probs)).get(key_name, [])
    return (grid, n)


def _range_task(
    map_ir,
    key_name,
    boundaries,
    n_buckets,
    nulls_first,
    desc,
    part_path,
    work_dir,
    mapper_id,
    engine_config,
):
    import os as _os

    import batcher._native as nat
    from batcher.dist.executors.partition_io import bucketize, read_partition
    from batcher.dist.shuffle_io import write_ipc

    rows = nat.execute_plan(map_ir, [read_partition(part_path)], engine_config)
    schema = rows[0].schema if rows else pa.schema([])
    buckets = bucketize(rows, key_name, boundaries, n_buckets, nulls_first, desc)
    paths = []
    for r in range(n_buckets):
        path = _os.path.join(work_dir, f"m{mapper_id}_r{r}.arrow")
        # An empty bucket still gets a schema-only file so every mapper publishes
        # exactly `n_buckets` paths (the reducer indexes by bucket); empty batches
        # are filtered out before the sort.
        batches = buckets[r] or [pa.RecordBatch.from_pylist([], schema=schema)]
        write_ipc(batches, path)
        paths.append(path)
    return paths


def _sort_reduce_task(sort_ir, input_paths, work_dir, reducer_id, engine_config):
    import os as _os

    import batcher._native as nat
    from batcher.dist.shuffle_io import write_ipc

    rows: list = []
    for p in input_paths:
        rows.extend(read_ipc(p))
    rows = [b for b in rows if b.num_rows > 0]
    if not rows:
        return None
    out = nat.execute_plan(sort_ir, [rows], engine_config)
    if not out:
        return None
    path = _os.path.join(work_dir, f"sorted_{reducer_id}.arrow")
    write_ipc(out, path)
    return path
