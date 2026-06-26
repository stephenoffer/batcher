"""The distributed executor — the dispatcher.

Inspects a plan's shape and routes it to the matching distributed operator
(map_batches, aggregate, join, sort, distinct, window, union), each of which reuses
the engine's mergeable primitives so its result is identical to single-node
execution. Shapes that can't be distributed yet fall back to the multi-core
single-node engine.

The per-operator implementations live in the `executors` subpackage; plan
analysis in `executors.plan_analysis`, partitioning/post-breaker helpers in
`executors.partition_io`, and Ray lifecycle + fallback in `executors.ray_runtime`.
The internal helpers re-exported
here (`_apply_above`, `_empty_agg_table`, `_ensure_ray`, `_partition_source`,
`_relabel_single_source`, `_rmtree`) keep `from batcher.dist.executor import ...`
working for the Flight and spill paths.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pyarrow as pa

# Re-exported (`X as X`) so the Flight + spill paths can keep importing these
# helpers from `batcher.dist.executor` after the split.
from batcher.dist.executors.partition_io import _apply_above as _apply_above
from batcher.dist.executors.partition_io import _empty_agg_table as _empty_agg_table
from batcher.dist.executors.partition_io import _partition_source as _partition_source
from batcher.dist.executors.partition_io import source_pushdown

# Used by the dispatcher below.
from batcher.dist.executors.plan_analysis import (
    _has_breaker,
    _is_linear_map_pipeline,
    _single_source,
    _split_at,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source as _relabel_single_source
from batcher.dist.executors.plan_analysis import _source_ids as _source_ids
from batcher.dist.executors.ray_runtime import _ensure_ray as _ensure_ray
from batcher.dist.executors.ray_runtime import _rmtree as _rmtree
from batcher.dist.executors.ray_runtime import (
    _single_node,
    clamp_workers,
    engine_config_json,
    release_autoscale,
    request_autoscale,
    reset_scheduling_envelope,
    resolve_transport,
    set_scheduling_envelope,
)
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Join,
    LogicalPlan,
    Sort,
    Union,
    Window,
)
from batcher.plan.resource import SchedulingEnvelope

__all__ = ["execute_distributed"]


def execute_distributed(
    plan: LogicalPlan,
    sources: list[Source],
    num_workers: int | None = None,
    transport: str = "disk",
    envelope: SchedulingEnvelope | None = None,
    hub=None,
    *,
    materialize: bool = True,
    metrics_out=None,
):
    """Execute a plan across Ray workers, falling back to single-node when needed.

    `transport="flight"` shuffles aggregation partials over Arrow Flight between
    worker actors (object store bypassed) instead of the default disk Arrow-IPC
    files; the result is identical. `transport="auto"` (the surface default) is
    resolved here from cluster topology — Flight on a real multi-node cluster
    (where the disk shuffle's driver-local `work_dir` is unreachable cross-node),
    disk on a single node / shared filesystem.

    `envelope` is Carbonite's metadata-driven scheduling grant: its `n_tasks` sets
    the worker fan-out (replacing a blind `os.cpu_count()`) and its per-task
    resources are applied to every Ray task via `.options(...)` at wrap time. It is
    installed as the ambient grant for the duration of this call. `hub` lets the
    GPU map/inference path record measured utilization for next-run adaptation.

    `materialize=False` lets a stage keep its result partitioned on disk and return a
    `MaterializedSource` (over the disk-shuffle path, where it is supported) instead
    of collecting every reducer's output back to the driver — the adaptive executor
    scans that intermediate in place for the next stage. Shapes that don't support it
    still return a collected `pa.Table`, so the caller must handle either.
    """
    if envelope is not None and num_workers is None:
        workers = max(1, envelope.n_tasks)
    else:
        workers = num_workers or (os.cpu_count() or 4)

    # Set the grant first so the up-front `_ensure_ray` wraps tasks with it; then ask
    # the autoscaler for the cores this query wants (released in the `finally` so a
    # one-off big job doesn't pin the cluster scaled-up), clamp the fan-out to
    # schedulable capacity, and pick the transport from the resulting topology.
    num_cpus = envelope.num_cpus if envelope is not None else 1.0
    token = set_scheduling_envelope(envelope)
    request_autoscale(math.ceil(workers * num_cpus))
    try:
        _ensure_ray(workers)
        clamped = clamp_workers(workers, num_cpus)
        # Carbonite sized the per-task memory hint against its *desired* fan-out;
        # once the cluster clamp reduces (or the data-driven want exceeds) it, each
        # real task holds a larger share. Rescale the soft memory hint to the actual
        # worker count and re-install the grant so `.options(memory=)` is honest.
        if envelope is not None and clamped != workers:
            envelope = _rescale_envelope(envelope, workers, clamped)
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
            _ensure_ray(clamped)
        workers = clamped
        transport = resolve_transport(transport, workers)
        return _dispatch(
            plan, sources, workers, transport, hub, materialize=materialize, metrics_out=metrics_out
        )
    finally:
        release_autoscale()  # let the autoscaler reclaim what this query scaled up
        reset_scheduling_envelope(token)


def _rescale_envelope(
    envelope: SchedulingEnvelope, desired: int, actual: int
) -> SchedulingEnvelope:
    """Rescale a scheduling grant from its desired fan-out to the actual one.

    The per-task memory hint was `peak // desired`; with `actual` tasks each holds
    `peak // actual`, so scale the hint by `desired / actual` (and update `n_tasks`).
    """
    import dataclasses

    actual = max(1, actual)
    memory_bytes = int(envelope.memory_bytes * desired / actual) if envelope.memory_bytes else 0
    return dataclasses.replace(envelope, n_tasks=actual, memory_bytes=memory_bytes)


def _is_splittable_source(source: Source) -> bool:
    """Whether `source` yields real per-chunk splits workers can read directly (Parquet
    row-groups, lakehouse fragments) — vs an in-memory/iterator source that returns one
    `WholeSourceSplit` and would have to be shipped to the workers. Only the former is
    worth distributing for a breaker-free scan."""
    from batcher.io.splits import WholeSourceSplit

    try:
        splits = source.splits()
    except Exception:
        return False
    return bool(splits) and not (len(splits) == 1 and isinstance(splits[0], WholeSourceSplit))


def _fusable_join_aggregate(agg: Aggregate) -> bool:
    """Whether `agg` is an aggregate over an inner join, grouped by (a superset of) the
    join key — so it can be distributed by reusing the join's co-partitioning.

    Requires the join key to appear among the group keys as a plain column: then every
    group shares one key value, lands in one co-partitioned bucket, and each reducer's
    per-bucket aggregate is complete (no cross-bucket combine needed, so any aggregate
    — even a non-mergeable one — is correct).
    """
    j = agg.input
    if not isinstance(j, Join) or j.join_type != "inner":
        return False
    if not (_single_source(j.left) and _single_source(j.right)):
        return False
    group_cols = {gk.expr.name for gk in agg.group_keys if isinstance(gk.expr, Col)}
    return bool(j.left_keys) and set(j.left_keys) <= group_cols


def _dispatch(
    plan: LogicalPlan,
    sources: list[Source],
    workers: int,
    transport: str,
    hub=None,
    *,
    materialize: bool = True,
    metrics_out=None,
):
    # Batch-inference / embedding pipelines (map_batches): distribute the linear
    # map chain across workers — the Ray Data competitor path.
    from batcher.core.udf import has_map_batches

    if has_map_batches(plan):
        if _is_linear_map_pipeline(plan) and _single_source(plan):
            # A linear chain with a stateless-CPU prefix feeding a GPU/load-once stage
            # streams the two with the stages overlapped (CPU prepares k+1 while the GPU
            # runs k), when enabled; otherwise it runs embarrassingly parallel. Any such
            # chain qualifies (incl. CPU→GPU→postprocess); shapes with no overlap to win
            # fall back to the non-overlapped map.
            from batcher.config import active_config

            if active_config().distributed.stream_inference:
                from batcher.dist.executors.plan_analysis import split_at_first_pool_boundary

                if split_at_first_pool_boundary(plan) is not None:
                    from batcher.dist.streaming import stream_distributed_pipeline

                    return stream_distributed_pipeline(plan, sources, workers, hub)
            from batcher.dist.executors.map import _distributed_map

            return _distributed_map(plan, sources, workers, hub)
        # map + relational breaker: not distributable yet → single-node UDF path.
        from batcher import core

        batches = core.execute_with_udfs(plan, sources)
        return pa.Table.from_batches(batches) if batches else pa.table({})

    # Breaker-free scan/filter/project over a SPLITTABLE source (Parquet row-groups,
    # lakehouse fragments): fan the read out so each worker reads its own splits in
    # parallel — the distributed-scan case — instead of one node reading the whole
    # source. In-memory/iterator sources stay single-node (shipping them to workers
    # costs more than the parallel CPU saves). Reuses `_distributed_map`'s stateless
    # task path (no UDF/GPU ⇒ one task per partition).
    if _is_linear_map_pipeline(plan) and _single_source(plan):
        sid = next(iter(_source_ids(plan)))
        if sid < len(sources) and _is_splittable_source(sources[sid]):
            from batcher.dist.executors.map import _distributed_map

            return _distributed_map(plan, sources, workers, hub)

    agg_split = _split_at(plan, Aggregate)
    if agg_split is not None:
        above, agg = agg_split
        if _single_source(agg.input):
            if transport == "flight":
                from batcher.dist.flight_aggregate import execute_aggregate_flight

                # When a query-lifetime `ShuffleFleet` is ambient (the adaptive loop
                # installed one), `materialize=False` keeps the stage's result on the
                # fleet's workers and returns a `FlightMaterializedSource` the next
                # stage reads in place — no driver collect. With no fleet the operator
                # spawns its own and collects, exactly as before.
                return execute_aggregate_flight(
                    above, agg, sources, workers, materialize=materialize
                )
            from batcher.dist.executors.aggregate import _distributed_aggregate

            return _distributed_aggregate(
                above, agg, sources, workers, hub, materialize=materialize, metrics_out=metrics_out
            )
        # Aggregate directly over an inner join, grouped by (a superset of) the join
        # key: reuse the join's co-partitioning to aggregate per bucket — no second
        # shuffle, and the full join never lands on the driver (exchange elimination).
        # The disk co-partition shuffle handles this; Flight keeps the fallback.
        if transport != "flight" and _fusable_join_aggregate(agg):
            from batcher.dist.executors.join import _distributed_join_aggregate

            return _distributed_join_aggregate(above, agg, agg.input, sources, workers)

    join_split = _split_at(plan, Join)
    if join_split is not None:
        above, join = join_split
        if _single_source(join.left) and _single_source(join.right):
            if transport == "flight":
                from batcher.dist.flight_join import execute_join_flight

                return execute_join_flight(above, join, sources, workers)
            from batcher.dist.executors.join import _distributed_join

            return _distributed_join(above, join, sources, workers, materialize=materialize)

    # ASOF join with `by` keys: co-partition both sides by the `by` keys (equal `by`
    # values hash together, so each bucket is an independent ASOF join). A keyless
    # ASOF needs one global order on `on` → stays single-node.
    asof_split = _split_at(plan, AsofJoin)
    if asof_split is not None:
        above, asof = asof_split
        if asof.left_by and _single_source(asof.left) and _single_source(asof.right):
            return _distributed_asof(above, asof, sources, workers)

    # A top-level sort over a scannable input distributes via range partitioning on
    # the leading key (which must be a plain column); secondary keys may be anything.
    sort_split = _split_at(plan, Sort)
    if sort_split is not None:
        above, sort = sort_split
        from batcher.plan.expr_ir import Col

        if (
            _single_source(sort.input)
            and sort.keys
            and isinstance(sort.keys[0].expr, Col)
            and not _has_breaker(sort.input)
        ):
            if transport == "flight":
                from batcher.dist.flight_sort import execute_sort_flight

                return execute_sort_flight(above, sort, sources, workers)
            from batcher.dist.executors.sort import _distributed_sort

            return _distributed_sort(above, sort, sources, workers)

    # DISTINCT over a breaker-free single source: dedup via the aggregate shuffle.
    distinct_split = _split_at(plan, Distinct)
    if distinct_split is not None:
        above, distinct = distinct_split
        if _single_source(distinct.input) and not _has_breaker(distinct.input):
            from batcher.dist.executors.distinct import _distributed_distinct

            return _distributed_distinct(
                above, distinct, sources, workers, transport, materialize=materialize
            )

    # A window partitioned by plain columns over a breaker-free source: hash-shuffle
    # rows by the partition keys so each partition is computed whole on one reducer.
    window_split = _split_at(plan, Window)
    if window_split is not None:
        above, window = window_split
        from batcher.plan.expr_ir import Col

        if (
            _single_source(window.input)
            and not _has_breaker(window.input)
            and window.partition_keys
            and all(isinstance(k, Col) for k in window.partition_keys)
        ):
            if transport == "flight":
                from batcher.dist.flight_window import execute_window_flight

                return execute_window_flight(above, window, sources, workers)
            from batcher.dist.executors.window import _distributed_window

            return _distributed_window(above, window, sources, workers)

    # UNION: distribute each branch independently, then concatenate (+ dedup).
    union_split = _split_at(plan, Union)
    if union_split is not None:
        above, union = union_split
        from batcher.dist.executors.union import _distributed_union

        return _distributed_union(above, union, sources, workers, transport)

    # Unsupported shape → multi-core single-node engine.
    return _single_node(plan, sources)


# --- ASOF join (co-partition by the `by` keys) --------------------------------
# Lives here (not in the `executors` subpackage, which is at its file-count ceiling)
# alongside the dispatch that routes to it. An ASOF match only ever pairs rows that
# share a `by` group, and `partition_batches` hashes equal `by` values to the same
# bucket on both sides, so each bucket is an independent ASOF join whose union is the
# full result. It reuses the equi-join's generic map/reduce tasks verbatim — only the
# partition keys (`by`) and the reducer IR (`asof_join`) differ.


def _asof_reducer_ir(asof: AsofJoin) -> dict:
    """IR for the per-bucket ASOF join of a left input (source 0) and right input
    (source 1). Mirrors `AsofJoin.to_ir()` but substitutes the per-task scans."""
    return {
        "op": "asof_join",
        "left": {"op": "scan", "source_id": 0},
        "right": {"op": "scan", "source_id": 1},
        "left_on": asof.left_on,
        "right_on": asof.right_on,
        "left_by": list(asof.left_by),
        "right_by": list(asof.right_by),
        "backward": asof.direction == "backward",
        "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in asof.output],
    }


def _distributed_asof(
    above: list[LogicalPlan], asof: AsofJoin, sources: list[Source], workers: int
) -> pa.Table:
    """Co-partition both sides by the `by` keys and ASOF-join each bucket in parallel."""
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.join import _join_map_task, _join_reduce_task
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.shuffle_io import read_ipc

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers

    left_plan, left_sid = _relabel_single_source(asof.left)
    right_plan, right_sid = _relabel_single_source(asof.right)
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    asof_ir = json.dumps(_asof_reducer_ir(asof))
    left_proj, left_pred = source_pushdown(left_plan, 0)
    right_proj, right_pred = source_pushdown(right_plan, 0)

    work_dir = tempfile.mkdtemp(prefix="batcher_asof_")
    try:
        left_parts = _partition_source(
            sources[left_sid], workers, work_dir, tag="L", projection=left_proj, predicate=left_pred
        )
        right_parts = _partition_source(
            sources[right_sid],
            workers,
            work_dir,
            tag="R",
            projection=right_proj,
            predicate=right_pred,
        )

        pol = speculation_policy()

        # Co-partition each side by its `by` keys with the generic join map task.
        def _left_map_for(i: int):
            return _join_map_task.remote(
                left_ir, list(asof.left_by), left_parts[i], workers, work_dir, "L", i, cfg_json
            )

        def _right_map_for(i: int):
            return _join_map_task.remote(
                right_ir, list(asof.right_by), right_parts[i], workers, work_dir, "R", i, cfg_json
            )

        left_paths = gather_with_backups(
            [_left_map_for(i) for i in range(len(left_parts))], _left_map_for, pol
        )  # [mapper][bucket]
        right_paths = gather_with_backups(
            [_right_map_for(i) for i in range(len(right_parts))], _right_map_for, pol
        )

        def _reduce_for(r: int):
            l_inputs = [paths[r] for paths in left_paths]
            r_inputs = [paths[r] for paths in right_paths]
            return _join_reduce_task.remote(asof_ir, l_inputs, r_inputs, work_dir, r, cfg_json)

        result_paths = gather_with_backups(
            [_reduce_for(r) for r in range(workers)], _reduce_for, pol
        )

        batches: list[pa.RecordBatch] = []
        for p, _rows in result_paths:
            if p is not None:
                batches.extend(read_ipc(p))
    finally:
        _rmtree(work_dir)

    if not batches:
        names = [o.alias for o in asof.output]
        result = pa.table({n: pa.array([], pa.null()) for n in names})
    else:
        result = pa.Table.from_batches(batches)
    return result if not above else _apply_above(above, result)
