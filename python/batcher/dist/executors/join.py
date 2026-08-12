"""Distributed join: a broadcast path and a co-partition hash-shuffle path.

When the planner marks a join ``broadcast`` (a small build side), the small side is
replicated to every worker and the big side is range-split with **no shuffle** — far
cheaper than moving both sides. Otherwise both sides are co-partitioned by join key:
equal keys hash to the same bucket, so per-bucket joins are independent and their
union is the full join (the textbook distributed hash join over disk Arrow-IPC files).
"""

from __future__ import annotations

import json
import os

import pyarrow as pa

from batcher._internal.hardware import l3_cache_bytes
from batcher._internal.native import engine
from batcher.dist.executors.partition_io import (
    _apply_above,
    _partition_source,
    source_pushdown,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source, empty_result_table
from batcher.dist.executors.ray_runtime import (
    _ensure_ray,
    _rmtree,
    engine_config_json,
    record_worker_metrics,
)
from batcher.dist.skew import join_skew_key, resolve_hot_keys, salting_preserves_result
from batcher.io.source import Source
from batcher.plan.distribution import BROADCAST_SAFE_JOINS
from batcher.plan.ir_specs import binary_task_ir
from batcher.plan.logical import Aggregate, Join, LogicalPlan
from batcher.plan.types import retained_bytes, total_retained_bytes

# Join types for which broadcasting the right (build) side and range-splitting the
# left (probe) side is correct. Read from `plan.distribution` rather than restated,
# because the GPU fan-out asks the same question and the two must not drift: a join
# type that is broadcast-safe on one backend and not the other is a duplicated row.
# RIGHT/FULL keep the co-partition shuffle even when the planner marked them
# broadcast (every strategy yields the same relation).
_BROADCAST_SAFE = BROADCAST_SAFE_JOINS

# Join types where the probe (left) side may be pruned by a bloom over the build
# (right) side's keys: a probe row with no matching build key contributes nothing
# (inner) or is excluded anyway (left-semi), so dropping it early changes nothing.
_BLOOM_SAFE = frozenset({"inner", "semi"})

# Fixed bloom capacity so every build-side mapper's bloom shares dimensions (a
# requirement to merge them). ~1M keys at 1% false positives ≈ 1.2 MB shipped to the
# probe mappers; a larger build side just raises the false-positive rate (still
# correct, only less pruning).
_BLOOM_EXPECTED_ITEMS = 1 << 20

# `"auto"` bloom engagement thresholds. The bloom pays off when the probe (left) side
# is much larger than the build (right) side — a selective fact⋈dimension — and the
# probe is big enough to repay building + shipping the bloom. Below these it is pure
# overhead, so `"auto"` keeps the plain shuffle.
_BLOOM_AUTO_PROBE_RATIO = 4.0  # probe rows ≥ 4× build rows
_BLOOM_AUTO_MIN_PROBE_ROWS = 1 << 20  # ~1M probe rows before a bloom is worth building


def _bloom_engaged(policy: bool | str, join: Join, sources: list[Source]) -> bool:
    """Resolve the `runtime_bloom_join` policy to a concrete on/off for this join.

    ``True``/``False`` force it; ``"auto"`` (the default) defers to a cardinality
    estimate so the bloom engages only where it pays — no user tuning.
    """
    if policy is True or policy is False:
        return policy
    return _bloom_beneficial(join, sources)


def _bloom_beneficial(join: Join, sources: list[Source]) -> bool:
    """Whether a runtime bloom is estimated to pay off for this shuffle join.

    The bloom prunes the probe (left) side using the build (right) side's keys, so it
    wins when the probe is much larger than the build (`_BLOOM_AUTO_PROBE_RATIO`) and
    large enough to repay building it (`_BLOOM_AUTO_MIN_PROBE_ROWS`). Best-effort and
    result-neutral: an unknown estimate reads as "not beneficial" (keep the plain
    shuffle), and being wrong only trades performance, never correctness.
    """
    from batcher.config import active_config
    from batcher.kyber.cardinality import CardinalityEstimator

    try:
        est = CardinalityEstimator(sources, {}, active_config().optimizer.cardinality)
        probe = est.estimate(join.left).rows
        build = est.estimate(join.right).rows
    except Exception:
        return False
    return (
        build > 0
        and probe >= build * _BLOOM_AUTO_PROBE_RATIO
        and probe >= _BLOOM_AUTO_MIN_PROBE_ROWS
    )


def salting_is_safe(join: Join, reducer_ir: str | None) -> bool:
    """Whether skew salting may engage for this shuffle join without changing the answer.

    Salting spreads a hot key's probe rows across several reducers (replicating its build
    rows to each), so the key's work fans across the cluster instead of overloading one
    node. That is result-preserving **only when each reducer's output is concatenated**.

    It is *not* safe when the reducer FINALIZES — a fused join+aggregate
    (`_distributed_join_aggregate`, which passes `reducer_ir`). That fusion is correct
    precisely because co-partitioning by the join key puts every group in exactly one
    bucket, letting a reducer finalize its bucket locally. Salting deliberately breaks
    that precondition: each of the salted reducers would finalize a *partial* group, and
    the union would carry several half-summed rows for the hot key instead of one correct
    row. No error is raised — the query simply returns a wrong answer.

    This guard is load-bearing rather than defensive, because salting engages on the
    **default** path: a measured hot key turns it on even when `skew_join_salt` is 0
    (see `dist.skew.resolve_hot_keys`).

    The conditions themselves live in `skew.salting_preserves_result`, shared with the
    Flight transport; this only translates *this* path's spelling of "the reducer
    finalizes" — a non-None `reducer_ir` — into the question that function asks.

    Args:
        join: The join being scheduled; salting needs a single equi-key and a left-driven
            join type, since it replicates build rows and must not duplicate output rows.
        reducer_ir: The per-bucket reducer override. `None` means the reducer is the plain
            join, whose buckets are concatenated — the only shape salting is safe for.

    Returns:
        True when salting may engage.
    """
    return salting_preserves_result(join, reducer_finalizes=reducer_ir is not None)


def _join_reducer_ir(join: Join) -> dict:
    """IR for the per-task join of a left input (source 0) with a right input
    (source 1). Mirrors `Join.to_ir()` but substitutes the per-task scans for the
    original inputs, carrying the planner's physical `strategy` through (not
    dropped). Used by both the shuffle reducer (co-partitioned buckets) and the
    broadcast task (left chunk joined with the full right).
    """
    return binary_task_ir(join)


def _distributed_join(
    above: list[LogicalPlan],
    join: Join,
    sources: list[Source],
    workers: int,
    *,
    materialize: bool = True,
    hub=None,
):
    """Run a distributed join: the broadcast path when the planner marked it broadcast
    on a broadcast-safe join type, else the co-partition shuffle. `materialize=False`
    lets the co-partition path keep its result partitioned (a `MaterializedSource`) for
    the next adaptive stage; the broadcast path always collects (caller handles both)."""
    if join.strategy == "broadcast" and join.join_type in _BROADCAST_SAFE:
        return _broadcast_join(above, join, sources, workers, hub=hub)
    return _shuffle_join(above, join, sources, workers, materialize=materialize, hub=hub)


def _shuffle_join(
    above: list[LogicalPlan],
    join: Join,
    sources: list[Source],
    workers: int,
    *,
    reducer_ir: str | None = None,
    output_names: list[str] | None = None,
    materialize: bool = True,
    hub=None,
):
    """Co-partition both sides by join key and join each bucket in parallel.

    `reducer_ir`/`output_names` override the per-bucket reducer (default: the join
    itself). Co-partitioning by the join key means equal keys share a bucket, so a
    reducer that *also* aggregates (group keys ⊇ join key) computes complete groups
    locally — `_distributed_join_aggregate` uses this to distribute a post-join
    aggregate without a second shuffle.
    """

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers

    left_plan, left_sid = _relabel_single_source(join.left)
    right_plan, right_sid = _relabel_single_source(join.right)
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())

    join_ir = reducer_ir if reducer_ir is not None else json.dumps(_join_reducer_ir(join))

    left_proj, left_pred = source_pushdown(left_plan, 0)
    right_proj, right_pred = source_pushdown(right_plan, 0)

    from batcher.dist.shuffle_io import distributed_work_dir

    work_dir = distributed_work_dir("batcher_join_")
    keep_dir = False  # set when a MaterializedSource takes ownership of work_dir
    try:
        left_parts = _partition_source(
            sources[left_sid],
            workers,
            work_dir,
            tag="L",
            projection=left_proj,
            predicate=left_pred,
        )
        right_parts = _partition_source(
            sources[right_sid],
            workers,
            work_dir,
            tag="R",
            projection=right_proj,
            predicate=right_pred,
        )

        from batcher.carbonite.resilience import gather_with_backups
        from batcher.dist.executors.ray_runtime import (
            runtime_bloom_join,
            skew_join_salt,
            speculation_policy,
        )

        pol = speculation_policy()

        # Skew-aware salting: when a single left-driven join key has hot values, spread
        # the probe (left) hot rows across `salt` reducers and replicate the build
        # (right) hot rows to all of them, so a huge x huge hot key fans across the
        # cluster instead of overloading one reducer. `salt == 0` → plain co-partition,
        # bit-identical to single-node. Cold keys always hash identically on both sides, so
        # the joined relation is unchanged.
        #
        # Which values are hot, and how far to fan them, is `dist.skew.resolve_hot_keys` —
        # shared with the Flight transport rather than restated, so the two cannot disagree
        # about whether a join is skewed. Only the pre-pass differs: here the splits are
        # files read by Ray tasks, there they are descriptors read by fleet actors.
        hot: list[str] = []
        salt = 0
        if salting_is_safe(join, reducer_ir):
            cfg_salt, frac = skew_join_salt()

            def _detect() -> tuple[list[str], float]:
                lk, rk = join.left_keys[0], join.right_keys[0]
                left_hot, l_share = _detect_hot_keys(left_parts, left_ir, lk, frac, cfg_json)
                right_hot, r_share = _detect_hot_keys(right_parts, right_ir, rk, frac, cfg_json)
                return sorted(left_hot | right_hot), max(l_share, r_share)

            hot, salt = resolve_hot_keys(
                join,
                sources,
                join_skew_key(left_ir, right_ir, join),
                frac,
                workers,
                cfg_salt,
                _detect,
            )

        # Runtime bloom reduction: build a bloom over the (smaller) build/right side's
        # keys and prune the probe/left side before its shuffle. Inner/semi (where
        # dropping a non-matching probe row is a no-op). Multi-key safe — the bloom is
        # built and probed over the row-encoded key *tuple* (the shared `key_rows`
        # encoding in `bc_py::bloom`), so composite keys prune exactly as single keys do;
        # corresponding left/right key order makes the encodings align. The engage
        # decision is the cardinality-driven `"auto"` policy by default (True/False force
        # it) — so a selective fact⋈dimension gets the bloom with no user tuning while a
        # balanced join keeps the plain shuffle.
        use_bloom = (
            join.join_type in _BLOOM_SAFE
            and bool(join.left_keys)
            and len(join.left_keys) == len(join.right_keys)
            and _bloom_engaged(runtime_bloom_join(), join, sources)
        )

        # Map tasks are pure functions of their partition → straggler-backup-safe.
        def _left_map_for(i: int, bloom_bytes=None):
            return _join_map_task.remote(
                left_ir,
                list(join.left_keys),
                left_parts[i],
                workers,
                work_dir,
                "L",
                i,
                cfg_json,
                hot,
                salt,
                False,
                bloom_bytes,
            )

        def _right_map_for(i: int, build_bloom=False):
            return _join_map_task.remote(
                right_ir,
                list(join.right_keys),
                right_parts[i],
                workers,
                work_dir,
                "R",
                i,
                cfg_json,
                hot,
                salt,
                True,
                None,
                build_bloom,
                _BLOOM_EXPECTED_ITEMS,
            )

        if use_bloom:
            nat = engine()
            # Build side first, so its merged bloom is ready to prune the probe side.
            right_results = gather_with_backups(
                [_right_map_for(i, build_bloom=True) for i in range(len(right_parts))],
                lambda i: _right_map_for(i, build_bloom=True),
                pol,
            )
            right_paths = [paths for paths, _bloom in right_results]
            merged_bloom = nat.merge_blooms([b for _p, b in right_results if b is not None])
            left_paths = gather_with_backups(
                [_left_map_for(i, bloom_bytes=merged_bloom) for i in range(len(left_parts))],
                lambda i: _left_map_for(i, bloom_bytes=merged_bloom),
                pol,
            )  # [mapper][bucket]
        else:
            left_paths = gather_with_backups(
                [_left_map_for(i) for i in range(len(left_parts))], _left_map_for, pol
            )  # [mapper][bucket]
            right_paths = gather_with_backups(
                [_right_map_for(i) for i in range(len(right_parts))], _right_map_for, pol
            )

        def _reduce_for(r: int):
            # `if paths` skips mappers that produced no buckets — a probe partition the
            # bloom pruned to empty (see `_join_map_task`).
            l_inputs = [paths[r] for paths in left_paths if paths]
            r_inputs = [paths[r] for paths in right_paths if paths]
            return _join_reduce_task.remote(join_ir, l_inputs, r_inputs, work_dir, r, cfg_json)

        reduce_results = gather_with_backups(
            [_reduce_for(r) for r in range(workers)], _reduce_for, pol
        )  # [(path, rows, metrics)]
        result_paths = [(path, rows) for path, rows, _m in reduce_results]
        # The reducer *is* the hash-join breaker: its `rows_build` and `peak_bytes` are the
        # only measurement of what a distributed join's hash table costs. Without them the
        # learned memory model — the one that decides whether the next join spills — is fit
        # entirely from single-node runs.
        record_worker_metrics(hub, (m for _p, _r, m in reduce_results))

        # Keep the join result partitioned on disk for the next adaptive stage.
        if not materialize and not above:
            from batcher.dist.executors.partition_io import materialize_reduce_output

            names = output_names if output_names is not None else [o.alias for o in join.output]
            fallback = empty_result_table(join, names).schema
            keep_dir = True
            return materialize_reduce_output(result_paths, work_dir, fallback)

        from batcher.dist.shuffle_io import read_ipc

        batches: list[pa.RecordBatch] = []
        for p, _rows in result_paths:
            if p is not None:
                batches.extend(read_ipc(p))
    finally:
        if not keep_dir:
            _rmtree(work_dir)

    if not batches:
        names = output_names if output_names is not None else [o.alias for o in join.output]
        result = empty_result_table(join, names)
    else:
        result = pa.Table.from_batches(batches)

    return result if not above else _apply_above(above, result)


def _distributed_join_aggregate(
    above: list[LogicalPlan],
    agg: Aggregate,
    join: Join,
    sources: list[Source],
    workers: int,
    hub=None,
) -> pa.Table:
    """Distribute an aggregate over an inner join by reusing the join's co-partitioning.

    The shuffle join co-partitions both sides by the join key, so equal keys share a
    bucket. When the aggregate's group keys include the join key, every group lies
    entirely within one bucket — so each reducer joins *and* aggregates its bucket and
    the union of reducer outputs is the complete result. This is exchange elimination:
    no second shuffle for the aggregate, and the full join never materializes on the
    driver (the dispatcher's fallback would collect it there). Forces the co-partition
    shuffle (never broadcast, whose range-split would scatter a group across chunks).
    """
    agg_ir = agg.to_ir()
    reducer_ir = json.dumps(
        {
            "op": "aggregate",
            "input": _join_reducer_ir(join),
            "group_keys": agg_ir["group_keys"],
            "aggregates": agg_ir["aggregates"],
        }
    )
    return _shuffle_join(
        above,
        join,
        sources,
        workers,
        reducer_ir=reducer_ir,
        output_names=agg.available_columns(),
        hub=hub,
    )


def _broadcast_join(
    above: list[LogicalPlan],
    join: Join,
    sources: list[Source],
    workers: int,
    hub=None,
) -> pa.Table:
    """Broadcast the small (right/build) side of an equi-join, falling back to the
    shuffle join when it will not fit.

    The shuffle fallback also covers the empty build side, so left/anti semantics over an
    empty right stay correct without a hand-built empty schema.
    """
    result = broadcast_probe_join(
        above, join, join.left, join.right, _join_reducer_ir(join), sources, workers, hub=hub
    )
    if result is None:
        return _shuffle_join(above, join, sources, workers, hub=hub)
    return result


def broadcast_probe_join(
    above: list[LogicalPlan],
    node: LogicalPlan,
    left: LogicalPlan,
    right: LogicalPlan,
    reducer_ir: dict,
    sources: list[Source],
    workers: int,
    hub=None,
) -> pa.Table | None:
    """Broadcast the small (right/build) side to every worker and split the big
    (left/probe) side — no shuffle of either side's keys.

    The build side is materialized once on the driver and shipped to every probe task,
    which joins its left chunk against the full right. Because each probe task sees the
    **whole** right side, every left row's match set is computed within its own partition,
    and each left row belongs to exactly one partition — so the union of the per-partition
    results is the full relation, with nothing duplicated and nothing missed. That argument
    holds for any join whose output is a function of (one left row, all right rows), which
    is exactly `_BROADCAST_SAFE`: `right`/`full` are excluded because a right row's
    matched-ness is a global question no single partition can answer.

    Nothing here reads the join *condition*, only the `reducer_ir` the caller supplies —
    which is what lets the equi-join and the inequality (`RangeJoin`) path share it. Their
    only real difference is what to do when the build side does not fit, so that decision
    is the caller's: this returns **None** when the right side is empty or exceeds the
    broadcast budget, rather than picking a fallback the caller may not have.

    Args:
        above: Operators stacked above the join, re-applied to the result.
        node: The join node itself, used for the empty result's schema.
        left: The probe side, partitioned across workers.
        right: The build side, broadcast whole to every worker.
        reducer_ir: Per-task IR joining source 0 (left chunk) with source 1 (full right).
        sources: The bound sources for the whole plan.
        workers: Probe-task fan-out.
        hub: Optional metadata hub for worker metrics.

    Returns:
        The joined table, or None if the build side is empty or too large to broadcast.
    """
    nat = engine()
    from batcher.dist.shuffle_io import read_ipc, write_ipc
    from batcher.io.source import read_source

    _ensure_ray(workers)
    cfg_json = engine_config_json()

    left_plan, left_sid = _relabel_single_source(left)
    right_plan, right_sid = _relabel_single_source(right)
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    join_ir = json.dumps(reducer_ir)
    left_proj, left_pred = source_pushdown(left_plan, 0)
    right_proj, right_pred = source_pushdown(right_plan, 0)

    # Materialize the build side once on the driver, then fall back to a shuffle join if
    # it is empty OR its *actual* size exceeds the broadcast threshold — a runtime guard
    # against a planner under-estimate that would OOM by replicating an over-large side.
    from batcher.config import active_config

    right_in = read_source(sources[right_sid], right_proj, right_pred)
    right_full = nat.execute_plan(right_ir, [right_in], cfg_json)
    # Measured on what the build side *retains*, not what it addresses: a batch cut by
    # offset from a larger one pins the parent, so `nbytes` here would pass a side well
    # over the threshold and then replicate it to every worker — the exact OOM this guard
    # exists to prevent, reached through the guard.
    if (
        not right_full
        or sum(b.num_rows for b in right_full) == 0
        or total_retained_bytes(right_full)
        # `workers`, so the runtime guard asks the same question the planner did. Left at
        # the single-node default it would re-decline, on measurement, the very broadcasts
        # the planner widened the threshold to admit — and silently, as a fallback.
        > active_config().optimizer.resolved_broadcast_max_bytes(l3_cache_bytes(), workers)
    ):
        return None

    from batcher.dist.shuffle_io import distributed_work_dir

    work_dir = distributed_work_dir("batcher_bcast_")
    try:
        right_path = os.path.join(work_dir, "broadcast_right.arrow")
        write_ipc(right_full, right_path)
        left_parts = _partition_source(
            sources[left_sid],
            workers,
            work_dir,
            tag="L",
            projection=left_proj,
            predicate=left_pred,
        )

        # Each probe task is a deterministic function of its left partition (read from
        # disk) joined against the broadcast right, so a slow survivor can be backed up
        # and the barrier takes whichever copy finishes first — the same straggler
        # mitigation the shuffle-join barriers use. `relaunch(i)` re-issues task i
        # identically (it writes the same `bcast_join_i` path with the same bytes).
        # Speculation is opt-in: with `speculation_max_backups == 0` (default) this is a
        # plain ordered gather, so the result is unchanged.
        from batcher.carbonite.resilience import gather_with_backups
        from batcher.dist.executors.ray_runtime import speculation_policy

        def _probe_task(i: int):
            return _broadcast_join_task.remote(
                left_ir, join_ir, left_parts[i], right_path, work_dir, i, cfg_json
            )

        refs = [_probe_task(i) for i in range(len(left_parts))]
        probe_results = gather_with_backups(refs, _probe_task, speculation_policy())
        record_worker_metrics(hub, (m for _path, ms in probe_results for m in ms))
        batches: list[pa.RecordBatch] = []
        for p, _metrics in probe_results:
            if p is not None:
                batches.extend(read_ipc(p))
    finally:
        _rmtree(work_dir)

    if not batches:
        result = empty_result_table(node, node.available_columns())
    else:
        result = pa.Table.from_batches(batches)
    return result if not above else _apply_above(above, result)


# Probe-chunk byte target: the left partition streams through the broadcast join in
# chunks of about this size, so a task's peak memory is one chunk + the (small) broadcast
# side + that chunk's output — not the whole left partition.
_BROADCAST_PROBE_CHUNK_BYTES = 32 << 20


def _broadcast_join_task(
    left_ir, join_ir, left_part_path, right_path, work_dir, task_id, engine_config
):
    from batcher.dist.executors.partition_io import iter_partition
    from batcher.dist.shuffle_io import read_ipc

    # The broadcast (right) side is small by construction — keep it resident, then stream
    # the (large) left partition past it one chunk at a time (`_stream_broadcast_join`),
    # so the left partition never has to fit in memory at once.
    path = os.path.join(work_dir, f"bcast_join_{task_id}.arrow")
    # One `ExecMetrics` document per probe chunk: each chunk is a real join of that chunk
    # against the full build side, so each is a legitimate observation for the cost and
    # memory models (peak = broadcast side + one chunk + its output).
    metrics: list[str] = []
    out_path = _stream_broadcast_join(
        left_ir,
        iter_partition(left_part_path),
        join_ir,
        read_ipc(right_path),
        path,
        engine_config,
        _BROADCAST_PROBE_CHUNK_BYTES,
        metrics_sink=metrics,
    )
    return out_path, metrics


def _stream_broadcast_join(
    left_ir,
    left_batches,
    join_ir,
    right_full,
    out_path,
    engine_config,
    chunk_bytes,
    metrics_sink: list[str] | None = None,
):
    """Join a streamed left side against a resident broadcast `right_full`, writing the
    output incrementally — peak memory is one probe chunk + the broadcast side + that
    chunk's output. Returns `out_path` if any rows were written, else None.

    When `metrics_sink` is given, each chunk's join `ExecMetrics` document is appended to
    it, so the broadcast path feeds the cost/memory models the shuffle path already feeds.
    The left sub-plan stays unmetered: it is the map prefix, already measured wherever it
    runs, and metering it here would double-count the same scan against every chunk.

    A plain helper (not a Ray task), so the chunked-probe logic is unit-testable in
    process; `_broadcast_join_task` wires it to partition IO on the worker.
    """
    import pyarrow as pa

    nat = engine()
    from batcher.dist.executors.ray_runtime import execute_metered

    sink = writer = None
    rows = 0
    try:
        for chunk in _byte_chunks(left_batches, chunk_bytes):
            left_rows = nat.execute_plan(left_ir, [chunk], engine_config)
            joined, metrics_json = execute_metered(join_ir, [left_rows, right_full], engine_config)
            if metrics_sink is not None and metrics_json:
                metrics_sink.append(metrics_json)
            for b in joined:
                if not b.num_rows:
                    continue
                if writer is None:
                    sink = pa.OSFile(out_path, "wb")
                    writer = pa.ipc.new_stream(sink, b.schema)
                writer.write_batch(b)
                rows += b.num_rows
    finally:
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()
    return out_path if rows else None


def _byte_chunks(batches, target_bytes: int):
    """Group an iterable of batches into lists of about `target_bytes` each (always at
    least one batch per chunk), so a streaming consumer bounds its working set.

    Sized by retained bytes: the bound is on memory held, and a batch that windows a
    larger parent holds the parent whatever `nbytes` reports."""
    chunk: list = []
    size = 0
    for b in batches:
        chunk.append(b)
        size += retained_bytes(b)
        if size >= target_bytes:
            yield chunk
            chunk, size = [], 0
    if chunk:
        yield chunk


def _detect_hot_keys(parts, subplan_ir, key_name, fraction, cfg_json) -> tuple[set[str], float]:
    """Detect the hot values of `key_name` across a side's partitions (Misra-Gries).

    Each partition reports its local heavy hitters and row count; a value is hot
    globally when its summed count clears `fraction` of the side's total rows.
    Returns the hot values rendered as strings (matching `salted_partition_batches`)
    and the hottest one's **measured** share of the side — the figure that sizes the salt
    fan-out, which `fraction` (the threshold a value must merely clear) cannot supply.
    `(set(), 0.0)` when nothing is skewed → the caller falls back to the plain shuffle.
    """
    import ray

    refs = [_join_detect_task.remote(subplan_ir, key_name, p, fraction, cfg_json) for p in parts]
    counts: dict[str, int] = {}
    total = 0
    for pairs, n in ray.get(refs):
        total += n
        for v, c in pairs:
            counts[v] = counts.get(v, 0) + c
    if total == 0:
        return set(), 0.0
    threshold = fraction * total
    hot = {v for v, c in counts.items() if c >= threshold}
    peak = max((counts[v] / total for v in hot), default=0.0)
    return hot, peak


def _join_detect_task(subplan_ir, key_name, part_path, fraction, engine_config):
    nat = engine()
    from batcher.dist.executors.partition_io import read_partition

    rows = nat.execute_plan(subplan_ir, [read_partition(part_path)], engine_config)
    n = sum(b.num_rows for b in rows)
    if n == 0:
        return [], 0
    hh = nat.heavy_hitters([key_name], rows, fraction)
    return [(v, int(c)) for v, c in hh.get(key_name, [])], n


def _join_map_task(
    subplan_ir,
    key_names,
    part_path,
    n_buckets,
    work_dir,
    side,
    mapper_id,
    engine_config,
    hot_keys=(),
    salt_count=0,
    replicate=False,
    bloom_bytes=None,
    build_bloom=False,
    bloom_expected=0,
):

    nat = engine()
    from batcher.dist.executors.partition_io import read_partition
    from batcher.dist.shuffle_io import write_shuffle_buckets

    rows = nat.execute_plan(subplan_ir, [read_partition(part_path)], engine_config)
    schema = rows[0].schema
    key_idx = [schema.get_field_index(k) for k in key_names]
    # Runtime bloom reduction: drop probe rows whose key can't be on the build side
    # before bucketing/shuffling. A superset filter (no false negatives) → the joined
    # relation is unchanged, only the rows written shrink.
    if bloom_bytes is not None:
        rows = nat.bloom_filter_batches(rows, key_idx, bloom_bytes)
        # Fully pruned by the bloom → this probe partition matches nothing. Skip
        # partitioning, writing, and shuffling its (empty) buckets entirely; the
        # reduce tolerates a mapper that produced no buckets.
        if not any(b.num_rows for b in rows):
            return []
    # Salted shuffle for a single hot join key; otherwise the plain co-partition.
    if salt_count > 0 and hot_keys and len(key_idx) == 1:
        buckets = nat.salted_partition_batches(
            rows, key_idx, n_buckets, list(hot_keys), salt_count, replicate
        )
    else:
        buckets = nat.partition_batches(rows, key_idx, n_buckets)

    paths = write_shuffle_buckets(buckets, work_dir, f"{side}_m", mapper_id)
    # The build side returns a bloom over its keys alongside its buckets, so the
    # driver can merge them and prune the probe side. Built over the full materialized
    # side (pre-bucketing); all mappers size identically so the blooms merge.
    if build_bloom:
        return paths, nat.build_key_bloom(rows, key_idx, bloom_expected)
    return paths


def _join_reduce_task(join_ir, left_paths, right_paths, work_dir, reducer_id, engine_config):
    import json as _json
    import os as _os

    from batcher.dist.executors.ray_runtime import execute_metered
    from batcher.dist.shuffle_io import read_ipc, reduce_envelope, write_ipc

    # When the two co-partitioned buckets together exceed the spill budget, reduce them
    # out-of-core — re-partition into sub-buckets on disk and join one pair at a time —
    # so a large or skewed bucket never has to fit in memory at once. Small buckets keep
    # the direct in-memory join (no spill overhead).
    budget = reduce_envelope(engine_config).budget
    total = sum(_safe_file_size(p) for p in (*left_paths, *right_paths))
    if budget and total > budget:
        from batcher.dist.spill_breakers import reduce_join_paths_spilling

        jd = _json.loads(join_ir)
        n_sub = max(2, -(-total // budget))  # ceil(total / budget): each pair ~ one budget
        result = reduce_join_paths_spilling(
            join_ir,
            list(jd["left_keys"]),
            list(jd["right_keys"]),
            left_paths,
            right_paths,
            work_dir,
            n_sub,
            engine_config,
        )
        # The out-of-core branch joins sub-bucket pairs internally; it reports no metrics.
        metrics_json = ""
    else:
        left: list = []
        for p in left_paths:
            left.extend(read_ipc(p))
        right: list = []
        for p in right_paths:
            right.extend(read_ipc(p))
        # An inner equi-join needs rows on BOTH sides; a reducer whose co-partitioned
        # bucket is empty on one side (a key present in only one input — routine under
        # skew or a high reducer count) produces no rows, so it contributes nothing to
        # the join or to an aggregate fused above it. Short-circuit rather than hand the
        # native engine a schema-less empty input, which it cannot type the result from.
        # Strictly inner `hash_join`: a left/right/full join keeps the non-empty side's
        # unmatched rows, and this same reducer also runs `asof_join` (left-style — every
        # left row is emitted with a null match even when the right bucket is empty).
        top = _json.loads(join_ir)
        join_node = top.get("input", top) if top.get("op") == "aggregate" else top
        if (
            join_node.get("op") == "hash_join"
            and join_node.get("join_type", "inner") == "inner"
            and (not any(b.num_rows for b in left) or not any(b.num_rows for b in right))
        ):
            return (None, 0, "")
        result, metrics_json = execute_metered(join_ir, [left, right], engine_config)

    rows = sum(b.num_rows for b in result) if result else 0
    if rows == 0:
        return (None, 0, metrics_json)
    path = _os.path.join(work_dir, f"join_reduce_{reducer_id}.arrow")
    write_ipc(result, path)
    return (path, rows, metrics_json)


def _safe_file_size(path: str) -> int:
    import os as _os

    try:
        return _os.path.getsize(path)
    except OSError:
        return 0
