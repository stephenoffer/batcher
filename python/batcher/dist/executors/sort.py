"""Distributed sort over a disk Arrow-IPC shuffle.

Range-partition by the *leading* sort key across workers (equal leading-key values
deterministically land in the same bucket, so no value spans a boundary), sort each
range by *all* sort keys in parallel, then concatenate the ranges in leading-key
order — globally sorted, with no final merge. The range boundaries come from a
**sample pass**: each worker sketches its OWN partition's leading-key quantile grid
via the shared sampler (`sample_key_grid`), so the input is never read on the
driver — only the small grids cross back, which the driver merges into `n_buckets-1`
boundaries. The grids are also *learned* (`dist/sort_boundaries.py`), so a second run
of the same sort shape skips that barrier and reads its input once instead of twice.
This mirrors `flight_sort`, but shuffles through driver-local IPC files
instead of Arrow Flight. Single- and multi-key sorts both go through this path (the
leading key must be a plain column to range-partition on).
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.dist.adaptive_sizing import row_shuffle_reducer_count
from batcher.dist.executors.partition_io import (
    _apply_above,
    _partition_source,
    merge_boundaries,
    plan_hot_split,
    sample_probs,
    source_pushdown,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source, empty_result_table
from batcher.dist.executors.ray_runtime import (
    _ensure_ray,
    _rmtree,
    engine_config_json,
    record_worker_metrics,
    shuffle_partitions,
)
from batcher.dist.shuffle_io import distributed_work_dir, read_ipc
from batcher.dist.sort_boundaries import (
    load_learned_grids,
    persist_grids,
    sort_key_identity,
    sort_key_is_string,
    sort_shape_key,
)
from batcher.io.source import Source
from batcher.plan.ir_specs import task_scan_ir
from batcher.plan.logical import LogicalPlan, Sort


def _distributed_sort(
    above: list[LogicalPlan],
    sort: Sort,
    sources: list[Source],
    workers: int,
    hub=None,
    metrics_out=None,
    *,
    materialize: bool = True,
):
    """Sample boundaries, range-partition each split, sort each range in parallel,
    then concatenate the ranges in leading-key order — globally sorted, no merge.

    `metrics_out` collects each worker's `ExecMetrics` document for the conductor's
    profile. The hub has always learned from these workers; the *profile* had no channel,
    so a distributed sort reported no per-operator detail — and the reduce task **is** the
    sort breaker, the one measurement of what a distributed sort costs in memory.

    `materialize=False` hands the reducers' output back as a `MaterializedSource` over the
    range buckets **in leading-key order** instead of concatenating them on the driver. A
    sort is row-preserving, so what the concatenation costs is the size of the whole
    relation on one node — the largest driver term the distributed executor had left, and
    the reason `iter_batches(distributed=True)` over an `ORDER BY` used to hold the entire
    result in the driver before yielding its first batch. The ordering survives because it
    was never produced by the concatenation: each bucket is a *range*, globally ordered
    against every other bucket, so listing the files in that same order and reading them
    in sequence is exactly the row order the concatenation had
    (`MaterializedSource.iter_batches` and `.splits()` both preserve file order).

    Declined — and a collected table returned — when anything is stacked `above` (there
    would be nothing to apply it to) or the sort carries a `limit`, whose final `slice`
    needs the assembled result. Callers must handle either return type.
    """
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
            **sort.shape_ir(),
            "input": task_scan_ir(),
        }
    )
    # A sort exchanges raw rows and sorts one bucket at a time, so the bucket count bounds
    # the run a reducer materializes; size it by volume (`row_shuffle_reducer_count`), never
    # below the one-per-worker floor. `map_plan` reads relabeled source 0, so hand the
    # estimator exactly that source.
    n_buckets = row_shuffle_reducer_count(map_plan, shuffle_partitions(workers), sources, sid)

    # A `limit` slices the assembled result, so it needs the assembly; `above` has nothing
    # to be applied to without it. Everything else can stay where it was computed.
    keep_partitioned = materialize is False and not above and sort.limit is None
    keep_dir = False  # set when a MaterializedSource takes ownership of work_dir

    work_dir = distributed_work_dir("batcher_dsort_")
    try:
        # Partition the source into per-worker map inputs (no data read on driver), with
        # the prefix's projection and predicate pushed into the split read as the Flight
        # sort already does. The map plan re-checks the filter, so this is I/O only — but
        # this operator reads its input *twice* (sample, then range-partition), so it is the
        # one where the saving is doubled. `map_plan`'s scan was relabeled to source 0.
        projection, predicate = source_pushdown(map_plan, 0)
        # A sort carrying a `limit` too large for the shuffle-free top-N still *slices*, so
        # it selects among rows tied at the cut and needs the same source-ordered partitions
        # `_distributed_topn` does. An unlimited sort returns every row, so the pick is free
        # and the balanced one is better.
        partitions = _partition_source(
            sources[sid],
            workers,
            work_dir,
            projection=projection,
            predicate=predicate,
            preserve_order=sort.limit is not None,
        )
        pol = speculation_policy()

        # SAMPLE: each worker sketches its own partition's leading-key grid; the
        # driver merges the small grids into range boundaries (rows never cross).
        # Every task is a pure function of its partition file, so a straggler can be
        # backed up (deterministic → identical output); `gather_with_backups` is a
        # plain barrier when speculation is disabled (the default).
        # The grid is sized against the *bucket* count, not fixed: `n_buckets` runs well
        # above the partition count on a volume-sized shuffle, and a grid that resolves a
        # boundary only to a quarter of a bucket overloads the unluckiest reducer by that
        # much while every other one waits for it. See `sample_probs`.
        probs = sample_probs(n_buckets, len(partitions))

        def _sample_for(w: int):
            return _sample_task.remote(map_ir, key_name, probs, partitions[w], cfg_json)

        # A learned grid removes the sample barrier entirely. The sample pass executes the
        # *whole* mapped prefix — scan, pushed predicate, projection — over every partition
        # to return a few hundred floats, and the range pass then executes the identical
        # prefix again to bucketize the rows it just measured and threw away. On a
        # scan-dominated sort that is close to half the job, and it is a full synchronous
        # barrier, so it is serial fraction as well as work. Safe when stale by the same
        # argument the boundaries themselves rest on: buckets are globally ordered for any
        # monotone boundary list, so a grid that no longer describes the data costs balance
        # and can never cost a row or an ordering (see `dist/sort_boundaries.py`). The
        # *grids* persist rather than the boundaries because the bucket count moves between
        # runs, so a stored boundary list would be the wrong length.
        # WHICH relation and WHICH type: a bare-scan `map_ir` is a positional source id with
        # no schema, so every single-source sort in the process hashed alike and shared one
        # grid — a wrong-typed one raises in the range partitioner, and a wrong-relation one
        # silently puts the whole input in a single bucket. See
        # `dist/sort_boundaries.sort_shape_key`. `expect_strings` re-checks on load, so an
        # entry written under the old colliding digest re-samples instead of raising.
        key_is_str = sort_key_is_string(sources[sid], key_name)
        shape_key = sort_shape_key(map_ir, key_name, sort_key_identity(sources[sid], key_name))
        grids = load_learned_grids(shape_key, key_is_str)
        if grids is None:
            grids = gather_with_backups(
                [_sample_for(w) for w in range(len(partitions))],
                _sample_for,
                pol,
                stage="sort.sample",
            )
            persist_grids(shape_key, grids)
        # Boundaries must cut into exactly `n_buckets` ranges, so size the split by the
        # actual reducer count — NOT `workers`. `shuffle_partitions` can trim the reducer
        # count below the mapper fan-out (the `max_shuffle_partitions` cap, the learned
        # fan-out), and `merge_boundaries(grids, workers)` would then emit up to `workers-1`
        # boundaries — more than `n_buckets-1` — routing rows into bucket ids past the last
        # bucket and panicking the range partitioner. (The out-of-core sort already uses
        # `n_buckets` here.)
        boundaries = merge_boundaries(grids, n_buckets)

        # A range partition must keep equal keys together, so one dominant value pins its
        # share of the rows on a single reducer however wide the shuffle is — the busiest
        # bucket simply stops shrinking as workers are added. `plan_hot_split` gives that
        # value a bucket of its own and spreads it over `subs` of them, one per contiguous
        # run of mappers, which is sound precisely because those rows all tie. `None` when
        # there is no such value, and then nothing below changes.
        split = plan_hot_split(grids, boundaries, n_buckets, nulls_first, desc)
        if split is not None:
            boundaries, n_buckets, hot_bucket, subs = split
            n_physical = n_buckets + subs - 1
        else:
            hot_bucket, subs, n_physical = -1, 0, n_buckets

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
                hot_bucket,
                subs,
                len(partitions),
            )

        map_results = gather_with_backups(
            [_range_for(w) for w in range(len(partitions))], _range_for, pol, stage="sort.map"
        )
        map_paths = [paths for paths, _metrics in map_results]
        record_worker_metrics(hub, (m for _paths, m in map_results), metrics_out)

        # REDUCE: each bucket gathers its shard from every mapper, sorts the range.
        def _reduce_for(r: int):
            return _sort_reduce_task.remote(
                sort_ir, [paths[r] for paths in map_paths], work_dir, r, cfg_json
            )

        reduce_results = gather_with_backups(
            [_reduce_for(r) for r in range(n_physical)], _reduce_for, pol, stage="sort.reduce"
        )
        sorted_paths = [(path, rows) for path, rows, _metrics in reduce_results]
        # The reduce task *is* the sort breaker: its `peak_bytes` is the only measurement
        # of what a distributed sort actually costs in memory, and the memory model that
        # decides spilling is fit from exactly these rows.
        record_worker_metrics(hub, (m for _path, _rows, m in reduce_results), metrics_out)

        # Leading-key order (reversed for a descending sort) — each bucket is globally
        # ordered relative to the others, so this is the sorted relation with no merge.
        order = range(n_physical - 1, -1, -1) if desc else range(n_physical)

        if keep_partitioned:
            from batcher.dist.executors.partition_io import materialize_reduce_output

            keep_dir = True  # the source owns `work_dir` and reclaims it on `cleanup()`
            return materialize_reduce_output(
                [sorted_paths[r] for r in order],
                work_dir,
                empty_result_table(sort, sort.available_columns()).schema,
            )

        out: list[pa.RecordBatch] = []
        for r in order:
            if sorted_paths[r][0] is not None:
                out.extend(read_ipc(sorted_paths[r][0]))
        result = (
            pa.Table.from_batches(out)
            if out
            else empty_result_table(sort, sort.available_columns())
        )
        if sort.limit is not None:
            result = result.slice(0, sort.limit)
    finally:
        if not keep_dir:
            _rmtree(work_dir)

    return result if not above else _apply_above(above, result)


def _distributed_topn(
    above: list[LogicalPlan],
    sort: Sort,
    sources: list[Source],
    workers: int,
) -> pa.Table:
    """`ORDER BY ... LIMIT k` with **no exchange at all** — the mergeable top-N.

    A top-N does not need the rows co-located, only compared: a row among the global `k`
    smallest is among its own partition's `k` smallest, because its partition holds a subset
    of the relation and its rank there is no worse than its rank globally. So the union of
    the per-partition top-`k`s contains the answer, and re-applying the same `sort + limit`
    to that union selects exactly it.

    That makes this the one relational shape here that scales *perfectly*: the map phase is
    Θ(N/W) and there is no shuffle to pay the exchange's Θ(W²) stream count, no boundary
    sample, and no reduce barrier. `_distributed_sort` answered the same query by
    range-partitioning **every row** across the cluster and then slicing the first `k` off
    the front — an all-to-all exchange of the whole relation to return ten rows. The Flight
    transport has had `execute_topn_flight` for this; the disk transport had nothing, so the
    strategy was chosen by which transport the topology resolved to rather than by the query.

    Each worker folds its partition a chunk at a time (`streaming_topn`), so it never
    materializes a partition to pick `k` rows out of it, and publishes its `k` to a file
    rather than returning them — the driver then folds those in worker order, one at a time,
    so its peak is `2k` rows rather than `workers x k` and the merge overlaps the map phase.
    Worker order, not arrival order: `LIMIT k` over rows that tie at the `k`-th place may
    return any of them, and an arrival-ordered fold would vary that between runs.

    Args:
        above: Operators stacked above the sort, re-applied to the result.
        sort: The sort node, carrying a non-null `limit`.
        sources: The bound sources for the whole plan.
        workers: Worker fan-out.

    No `hub`/`metrics_out`: this path reports no per-operator metrics, for the reason
    `_topn_task` states, and taking the parameters to ignore them would say otherwise.

    Returns:
        The top-`k` rows, in sort order.
    """
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import speculation_policy

    nat = engine()
    _ensure_ray(workers)
    cfg_json = engine_config_json()
    map_plan, sid = _relabel_single_source(sort.input)
    # The worker's plan is the map prefix with the sort+limit on top; the merge plan is the
    # same sort+limit over the already-projected output, which is what makes re-applying it
    # to the union of the workers' answers the merge.
    local_ir = json.dumps({**sort.shape_ir(), "input": map_plan.to_ir()})
    merge_ir = json.dumps({**sort.shape_ir(), "input": task_scan_ir()})

    work_dir = distributed_work_dir("batcher_dtopn_")
    try:
        # Push the prefix's projection and predicate into the split read, as the Flight
        # top-N does. The map plan re-checks the filter, so this is I/O only — but it is
        # per-node I/O, which is the term that has to fall with the fleet for the shape to
        # scale at all. `map_plan`'s scan was relabeled to source 0, so ask about 0.
        projection, predicate = source_pushdown(map_plan, 0)
        # Contiguous, source-ordered partitions. A top-N keeps only `k` of the rows it
        # orders, so which of several rows tied at the `k`-th place survives is decided by
        # input order — and the load-balanced split pick hands one partition non-adjacent
        # splits, which selects a different tied row than single-node does. Measured on 24
        # files with a 5,000-value key: `ORDER BY k LIMIT 137` returned the right 137 keys
        # either way, and a different set of rows under the balanced pick. `_contiguous`
        # still fills each partition to an equal share of the weight, so this gives up
        # reordering rather than balance.
        partitions = _partition_source(
            sources[sid],
            workers,
            work_dir,
            projection=projection,
            predicate=predicate,
            preserve_order=True,
        )
        pol = speculation_policy()

        def _topn_for(w: int):
            return _topn_task.remote(local_ir, merge_ir, partitions[w], work_dir, w, cfg_json)

        paths = gather_with_backups(
            [_topn_for(w) for w in range(len(partitions))], _topn_for, pol, stage="topn.map"
        )

        merged: list[pa.RecordBatch] = []
        for path in paths:
            if path is None:
                continue
            arrived = [b for b in read_ipc(path) if b.num_rows > 0]
            if arrived:
                merged = list(nat.execute_plan(merge_ir, [merged + arrived], cfg_json))
        result = (
            pa.Table.from_batches(merged)
            if merged
            else empty_result_table(sort, sort.available_columns())
        )
    finally:
        _rmtree(work_dir)

    return result if not above else _apply_above(above, result)


def _topn_task(local_ir, merge_ir, part_path, work_dir, task_id, engine_config):
    """This partition's own top-`k`, folded a chunk at a time and written as one IPC file.

    Returns the path, or `None` for a partition that yielded no rows.

    Unmetered, unlike the sort's tasks, and that is a stated gap rather than an oversight:
    `streaming_topn` runs the plan once per chunk, so no single execution's `ExecMetrics`
    describe the operator, and reporting one chunk's would teach the memory model that a
    top-N costs a chunk. Metering nothing is the honest option until the fold learns to
    accumulate."""
    import os as _os

    nat = engine()
    from batcher.dist.executors.partition_io import iter_partition, streaming_topn
    from batcher.dist.shuffle_io import write_ipc

    top = [
        b
        for b in streaming_topn(nat, local_ir, merge_ir, iter_partition(part_path), engine_config)
        if b.num_rows > 0
    ]
    if not top:
        return None
    return write_ipc(top, _os.path.join(work_dir, f"topn_{task_id}.arrow"))


def _sample_task(map_ir, key_name, probs, part_path, engine_config):
    nat = engine()
    from batcher.dist.executors.partition_io import read_partition, sample_key_grid

    rows = nat.execute_plan(map_ir, [read_partition(part_path)], engine_config)
    n = sum(b.num_rows for b in rows)
    if n == 0:
        return ([], 0)
    return (sample_key_grid(rows, key_name, list(probs)), n)


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
    hot_bucket=-1,
    subs=0,
    n_mappers=1,
):
    import os as _os

    from batcher.dist.executors.partition_io import (
        bucketize,
        hot_sub_bucket,
        read_partition,
        split_hot_bucket,
    )
    from batcher.dist.executors.ray_runtime import execute_metered
    from batcher.dist.shuffle_io import write_ipc

    rows, metrics_json = execute_metered(map_ir, [read_partition(part_path)], engine_config)
    schema = rows[0].schema if rows else pa.schema([])
    buckets = bucketize(rows, key_name, boundaries, n_buckets, nulls_first, desc)
    if hot_bucket >= 0 and subs > 1:
        # This mapper's share of the dominant value goes to exactly one sub-bucket, chosen
        # by mapper id so the driver's ordered concatenation still reads them in mapper
        # order — every row here ties on the key, so that order is the whole contract.
        buckets = split_hot_bucket(
            buckets, hot_bucket, subs, hot_sub_bucket(mapper_id, n_mappers, subs, desc)
        )
    n_buckets = len(buckets)
    paths = []
    for r in range(n_buckets):
        path = _os.path.join(work_dir, f"m{mapper_id}_r{r}.arrow")
        # An empty bucket still gets a schema-only file so every mapper publishes
        # exactly `n_buckets` paths (the reducer indexes by bucket); empty batches
        # are filtered out before the sort.
        batches = buckets[r] or [pa.RecordBatch.from_pylist([], schema=schema)]
        write_ipc(batches, path)
        paths.append(path)
    return paths, metrics_json


def _sort_reduce_task(sort_ir, input_paths, work_dir, reducer_id, engine_config):
    """Sort this range bucket and publish it. Returns `(path, rows, metrics)`.

    The row count is exact and is reported for the same reason the aggregate's and the
    keyed shuffle's reducers report it: a caller keeping the result partitioned
    (`materialize=False`) needs to size the intermediate without reading it back, and
    reading it back is precisely the thing being avoided.
    """
    import os as _os

    from batcher.dist.executors.ray_runtime import execute_metered
    from batcher.dist.shuffle_io import write_ipc

    rows: list = []
    for p in input_paths:
        rows.extend(read_ipc(p))
    rows = [b for b in rows if b.num_rows > 0]
    if not rows:
        return (None, 0, "")
    out, metrics_json = execute_metered(sort_ir, [rows], engine_config)
    if not out:
        return (None, 0, metrics_json)
    path = _os.path.join(work_dir, f"sorted_{reducer_id}.arrow")
    write_ipc(out, path)
    return (path, sum(b.num_rows for b in out), metrics_json)
