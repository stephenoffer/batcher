"""Distributed *global* ordered window over a disk Arrow-IPC shuffle.

The same ordered-bucket-offset algebra `flight` runs, shuffled through driver-local IPC files
instead of Arrow Flight -- so the route exists on both transports, exactly as the distributed
sort does. `dist/executors/ray_runtime.choose_transport` picks disk on a single node, and a
window shape that worked on a cluster and raised on one box would be the worse failure.

Every phase here is the distributed sort's, unchanged: `executors.sort`'s sample, range-
partition and reduce tasks are *generic over the plan they run*, so this module reuses them
rather than restating three Ray tasks. The reduce is handed the window IR where the sort hands
it the sort IR; that is the entire difference in the shuffle. The driver then walks the
buckets in key order and applies the offsets.

Reaching the tasks through the module (`_sort.<name>.remote`) rather than importing them by
name is load-bearing: `_ensure_ray` rebinds each one to its `ray.remote` wrapper on the
module, so a by-name import would capture the *unwrapped* function and call it locally.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher.dist.executors import sort as _sort
from batcher.dist.executors.partition_io import (
    _apply_above,
    _partition_source,
    merge_boundaries,
    sample_probs,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source, empty_result_table
from batcher.dist.executors.ray_runtime import (
    _ensure_ray,
    _rmtree,
    engine_config_json,
    record_worker_metrics,
    shuffle_partitions,
)
from batcher.dist.global_window.offsets import (
    OrderedBucketOffsets,
    bucket_order,
    inject_avg_helpers,
)
from batcher.dist.shuffle_io import distributed_work_dir, read_ipc
from batcher.dist.sort_boundaries import load_learned_grids, persist_grids, sort_shape_key
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Window

__all__ = ["execute_global_window_disk"]


def execute_global_window_disk(
    above: list[LogicalPlan],
    window: Window,
    sources: list[Source],
    workers: int,
    hub=None,
) -> pa.Table:
    """Range-partition by the order key, window each range in parallel, offset on the driver."""
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import speculation_policy

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers

    key = window.order_keys[0]  # caller guarantees a single plain-column order key
    key_name = key.expr.name
    desc, nulls_first = key.descending, key.nulls_first

    map_plan, sid = _relabel_single_source(window.input)
    map_ir = json.dumps(map_plan.to_ir())
    # The reduce runs the window over its bucket as a single in-memory source 0. `to_ir()`
    # memoizes and hands back the plan's shared structures, so copy what is rewritten here.
    win_ir = dict(window.to_ir())
    win_ir["input"] = {"op": "scan", "source_id": 0}
    win_ir["functions"] = list(win_ir["functions"])
    avg_helpers = inject_avg_helpers(window, win_ir)
    win_json = json.dumps(win_ir)
    n_buckets = shuffle_partitions(workers)

    work_dir = distributed_work_dir("batcher_dglobalwin_")
    try:
        partitions = _partition_source(sources[sid], workers, work_dir)
        pol = speculation_policy()

        # SAMPLE: each worker sketches its own partition's order-key grid; the driver merges
        # the small grids into range boundaries (rows never cross). Every task is a pure
        # function of its partition file, so a straggler can be backed up.
        # The grid is derived from the bucket-to-sampler ratio rather than fixed, and
        # `sample_probs` is shared with the sort rather than restated here: a boundary is
        # placed to within `1/g` of a sampler's rows, so a wide cut needs a finer grid to
        # keep the ranges even. Two statements of that arithmetic is how the two paths
        # would drift into disagreeing about the sample's shape.
        probs = sample_probs(n_buckets, len(partitions))

        def _sample_for(w: int):
            return _sort._sample_task.remote(map_ir, key_name, probs, partitions[w], cfg_json)

        # A learned grid skips this barrier outright — the sample pass runs the whole mapped
        # prefix over every partition to return a few hundred floats, and the range pass then
        # runs it again. Safe when stale: the offset algebra is correct for any monotone
        # boundary list, so an out-of-date grid costs balance and never a row.
        shape_key = sort_shape_key(map_ir, key_name)
        grids = load_learned_grids(shape_key)
        if grids is None:
            grids = gather_with_backups(
                [_sample_for(w) for w in range(len(partitions))], _sample_for, pol
            )
            persist_grids(shape_key, grids)
        # Size the cut by the ACTUAL reducer count: `shuffle_partitions` can trim it below the
        # mapper fan-out, and boundaries sized for `workers` would route rows into bucket ids
        # past the last bucket and panic the range partitioner.
        boundaries = merge_boundaries(grids, n_buckets)

        # MAP: range-partition each split by the boundaries, one IPC file per bucket. Equal
        # keys land in one bucket, so no peer group spans a boundary and each bucket can be
        # windowed on its own.
        def _range_for(w: int):
            return _sort._range_task.remote(
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

        map_results = gather_with_backups(
            [_range_for(w) for w in range(len(partitions))], _range_for, pol
        )
        map_paths = [paths for paths, _metrics in map_results]
        record_worker_metrics(hub, (m for _paths, m in map_results))

        # REDUCE: each bucket gathers its shard from every mapper and runs the WINDOW over
        # the range — the same generic reduce the sort uses, given a different plan.
        def _reduce_for(r: int):
            return _sort._sort_reduce_task.remote(
                win_json, [paths[r] for paths in map_paths], work_dir, r, cfg_json
            )

        reduce_results = gather_with_backups(
            [_reduce_for(r) for r in range(n_buckets)], _reduce_for, pol
        )
        # `_sort_reduce_task` reports `(path, rows, metrics)`; the row count is for a caller
        # keeping the result partitioned, which this one does not — it has per-bucket window
        # offsets to apply before the buckets mean anything.
        windowed_paths = [path for path, _rows, _metrics in reduce_results]
        # The reduce task *is* the window breaker: its `peak_bytes` is the only measurement of
        # what a distributed global window costs in memory, and the memory model that decides
        # spilling is fit from exactly these rows.
        record_worker_metrics(hub, (m for _path, _rows, m in reduce_results))

        # Walk the buckets in global key order, shifting each one's window columns to their
        # global values. Buckets are ordered relative to each other by construction.
        offsets = OrderedBucketOffsets(window, avg_helpers)
        out: list[pa.RecordBatch] = []
        for r in bucket_order(n_buckets, desc):
            if windowed_paths[r] is None:
                continue
            batches = [b for b in read_ipc(windowed_paths[r]) if b.num_rows > 0]
            if not batches:
                continue
            out.extend(offsets.apply(pa.Table.from_batches(batches)).to_batches())
        result = (
            pa.Table.from_batches(out)
            if out
            else empty_result_table(window, window.available_columns())
        )
    finally:
        _rmtree(work_dir)

    return result if not above else _apply_above(above, result)
