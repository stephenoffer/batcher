"""Distributed aggregation over a disk Arrow-IPC shuffle.

Pipeline:  map (run the sub-plan on a source partition → `partial_aggregate`) →
hash-shuffle partial state to disk → reduce (`combine_finalize` per key
partition) → collect → run any post-aggregation operators single-node. The
mergeable primitives are reused verbatim, so the result equals single-node.

The map side takes one sub-plan per *branch* (`shuffle_branches`), which is the one
input in the ordinary case and the branch list when the aggregate reads a `UNION ALL`.
Every branch's partials land in the same bucket space, so the reducers see exactly what
they would have seen had the union been concatenated first — `combine` is associative and
commutative, which is the whole reason that holds. This is what keeps a reduced union
(`union(...).group_by(...)`, and the `intersect`/`except_` lowering) off the driver.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher._internal.native import engine
from batcher.dist.executors.partition_io import (
    _apply_above,
    _partition_source,
    consumer_pushdown,
)
from batcher.dist.executors.plan_analysis import (
    _empty_agg_table,
    _relabel_single_source,
    shuffle_branches,
)
from batcher.dist.executors.ray_runtime import (
    _ensure_ray,
    _rmtree,
    engine_config_json,
    record_worker_metrics,
    shuffle_partitions,
)
from batcher.io.source import Source
from batcher.plan.ir_specs import agg_spec_json
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
    by the returned source's `cleanup()`.

    `agg.input` is either the usual breaker-free single-source prefix or a `UNION ALL` of
    them, whose branches all map into this one shuffle (`shuffle_branches`)."""
    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers

    group_keys_json, aggregates_json = agg_spec_json(agg)
    branches = shuffle_branches(agg.input)
    if branches is None:
        raise PlanError(
            "the distributed aggregate shuffle needs a breaker-free single-source map "
            f"prefix (or a UNION ALL of them), got {type(agg.input).__name__}"
        )
    n_keys = len(agg.group_keys)
    # Global aggregate (no keys) cannot shuffle by key → a single reducer.
    n_reducers = 1 if n_keys == 0 else shuffle_partitions(workers)

    from batcher.dist.shuffle_io import distributed_work_dir

    work_dir = distributed_work_dir("batcher_shuffle_")
    keep_dir = False  # set when a MaterializedSource takes ownership of work_dir
    try:
        # One `(sub-plan IR, partition)` map task per partition of every branch. A single
        # input is the one-branch case and produces exactly the task list it always did;
        # a UNION ALL contributes its own partitions of its own source, all feeding this
        # one shuffle. Each branch is pushed down and partitioned separately because each
        # reads its own source: the projection a branch needs is a fact about that branch.
        map_specs: list[tuple[str, str]] = []
        for b, branch in enumerate(branches):
            map_plan, source_id = _relabel_single_source(branch)
            # Push the projection + predicate into the source read (the map_ir still re-checks
            # the filter, so this is a pure I/O optimization). Ask about the aggregate *over*
            # the map prefix: the prefix of a plain `group_by(k).agg(sum(v))` is a bare `Scan`,
            # which requires every column it has — so asking about the prefix alone read the
            # whole wide table to answer a two-column aggregate. See `consumer_pushdown`.
            projection, predicate = consumer_pushdown(agg, map_plan)
            parts = _partition_source(
                sources[source_id],
                workers,
                work_dir,
                tag=f"P{b}",
                projection=projection,
                predicate=predicate,
            )
            branch_ir = json.dumps(map_plan.to_ir())
            map_specs.extend((branch_ir, p) for p in parts)

        from batcher.carbonite.resilience import gather_with_backups
        from batcher.dist.executors.ray_runtime import speculation_policy

        pol = speculation_policy()

        # MAP: run the sub-plan on each partition, partial-aggregate, hash-shuffle.
        # A map task is a pure function of its partition, so a straggler can be
        # backed up (deterministic → identical output); `gather_with_backups` is a
        # plain barrier when speculation is disabled.
        def _map_for(mid: int):
            branch_ir, part = map_specs[mid]
            return _map_task.remote(
                branch_ir,
                group_keys_json,
                aggregates_json,
                part,
                n_keys,
                n_reducers,
                work_dir,
                mid,
                cfg_json,
            )

        map_refs = [_map_for(mid) for mid in range(len(map_specs))]
        # Each mapper returns (per-reducer paths, sub-plan metrics). The driver
        # records the workers' measured operator metrics into the hub so the cost
        # model calibrates from distributed runs too (the measure→consume loop is
        # not single-node-only). Best-effort, by operator kind.
        map_results = gather_with_backups(map_refs, _map_for, pol, stage="aggregate.map")
        shuffle_paths = [paths for paths, _metrics in map_results]
        record_worker_metrics(hub, (m for _paths, m in map_results), metrics_out)

        # COMBINE: collapse each bucket's mapper partials through a tree of bounded-fan-in
        # combines, so no reducer reads more than `shuffle_fan_in` of them. Skipped
        # entirely (and costing nothing) when there are already few enough mappers.
        bucket_inputs = _tree_combine_buckets(
            shuffle_paths, n_reducers, group_keys_json, aggregates_json, work_dir, cfg_json, pol
        )

        # REDUCE: each reducer combines+finalizes the partials routed to it. The worker
        # config (shipped `cfg_json`) carries its memory envelope, so a reducer whose merged
        # group state would exceed it spills out of core instead of OOMing.
        def _reduce_for(r: int):
            return _reduce_task.remote(
                group_keys_json, aggregates_json, bucket_inputs[r], work_dir, r, cfg_json
            )

        reduce_refs = [_reduce_for(r) for r in range(n_reducers)]
        result_paths = gather_with_backups(
            reduce_refs, _reduce_for, pol, stage="aggregate.reduce"
        )  # [(path, rows)]

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

    nat = engine()
    from batcher.dist.executors.partition_io import read_partition
    from batcher.dist.executors.ray_runtime import execute_metered
    from batcher.dist.shuffle_io import write_ipc, write_shuffle_buckets

    # Metered: the worker measures its sub-plan's per-operator runtime facts and
    # ships them back so the driver can feed the cost-model calibration loop.
    rows, metrics_json = execute_metered(map_ir, [read_partition(part_path)], engine_config)
    partial = nat.partial_aggregate(group_keys_json, aggregates_json, rows)

    if n_keys == 0:
        path = _os.path.join(work_dir, f"m{mapper_id}_r0.arrow")
        write_ipc([partial], path)
        return [path], metrics_json

    buckets = nat.partition_batches([partial], list(range(n_keys)), n_reducers)
    return write_shuffle_buckets(buckets, work_dir, "m", mapper_id), metrics_json


# How many times the fan-in the mapper count must reach before the combiner tree engages.
# A level of the disk tree costs a task, a write and a read per chunk against a saving of
# `m - f - m/f` combines, so at `m` just past `f` the trade is roughly even; at `4f` the
# saving is five times the tasks spent, which is where it is worth taking unconditionally.
_TREE_MIN_MAPPERS = 4


def _combine_task(
    group_keys_json, aggregates_json, input_paths, work_dir, out_name, engine_config=""
):
    """Merge a chunk of a bucket's partials into ONE partial file — an interior level of
    the combiner tree.

    `combine`, never `combine_finalize`: an interior node's output is consumed by another
    combine, and finalizing early would aggregate an average of averages. The distinction
    is the whole reason the tree is expressible at all — partial state is closed under
    `combine`, and only the last step leaves that algebra.

    **Declines rather than OOMs.** The reducer has an out-of-core fold
    (`combine_finalize_spilling`) for the case where merged group state exceeds the shipped
    memory envelope; an interior combine has no such thing, because spilling is only defined
    for the step that *finalizes*. So a chunk whose inputs already exceed the envelope hands
    its paths back untouched instead of merging them, and that bucket reaches the reducer as
    wide as it was — which is exactly the input the spilling fold is written for. Without
    this the tree would turn a high-cardinality aggregate that used to spill and finish into
    one that dies in a level nobody can see.

    Returns the paths the next level should read: one merged file normally, the inputs
    unchanged when declined, and an empty list when every input was empty (so an empty
    bucket costs no file)."""
    import os as _os

    nat = engine()
    from batcher.dist.shuffle_io import read_ipc, reduce_envelope, write_ipc

    budget = reduce_envelope(engine_config).budget
    if budget > 0:
        on_disk = sum(_os.path.getsize(p) for p in input_paths if _os.path.exists(p))
        # On-disk bytes are a floor for the in-memory state (IPC never expands), so a
        # chunk under the envelope is safe to merge and the common case pays one stat
        # per input for that certainty.
        if on_disk > budget:
            return list(input_paths)

    running = None
    for path in input_paths:
        batch = read_ipc(path)
        if not batch:
            continue
        merged = batch if running is None else [running, *batch]
        running = nat.combine(group_keys_json, aggregates_json, merged)
    if running is None:
        return []
    return [write_ipc([running], _os.path.join(work_dir, out_name))]


def _tree_combine_buckets(
    shuffle_paths, n_reducers, group_keys_json, aggregates_json, work_dir, cfg_json, pol
):
    """Collapse every bucket's mapper partials to at most `shuffle_fan_in` files, in
    `ceil(log_fan_in(mappers))` parallel levels.

    Without this the reduce side is a **line**: reducer `r` opens one file per mapper and
    folds them one after another, so its critical path is Θ(mappers). That term grows as
    the cluster does — one mapper per worker is the floor — while the map phase it runs
    after shrinks, which is Amdahl's serial fraction appearing precisely at the scale that
    motivated adding nodes. Measured as a shape rather than a constant: at `W` workers the
    reduce does `W` sequential combines of a `G`-group state, so `T_reduce = Θ(W·G)` and
    total time bottoms out around `W = sqrt(N/G)` and *rises* past it.

    Rebracketing the same fold as a tree of arity `f` costs the identical number of
    combines but puts `ceil(m/f)` of them in each level and runs the levels' tasks in
    parallel, so the critical path is `Θ(f·log_f W)` — the term that lets total time keep
    falling as workers are added. It is sound for any `f` because `combine` is associative
    and commutative, so every bracketing of the same partials is the same state; `f` buys
    critical path against per-node fan-out and never buys correctness.

    This is the disk transport's half of what `flight_aggregate._tree_reduce` already does
    over Flight, and it applies to the keyless global aggregate too — there `n_reducers` is
    1, so the line was `W` long on a single node with the rest of the cluster idle.

    **It engages later than the Flight tree does, and deliberately.** Flight's interior
    partials live in memory, so a level costs a combine and nothing else; here a level costs
    a task launch, a file write and a file read per chunk, and those are charged against a
    saving of `m - f - m/f` combines. Just past `m > f` that trade is roughly even and can
    be negative when the group state is small — so the tree waits for `_TREE_MIN_MAPPERS`
    times the fan-in, where the saving is several times the tasks it spends. Below that a
    shuffle takes the flat fold it always did and pays nothing for this existing.

    Args:
        shuffle_paths: One list of `n_reducers` bucket paths per mapper.
        n_reducers: The bucket count.
        group_keys_json: The aggregate's group-key spec.
        aggregates_json: The aggregate's expression spec.
        work_dir: The shared shuffle scratch directory.
        cfg_json: The shipped worker engine config, which carries the memory envelope an
            interior combine declines against.
        pol: The speculation policy for the per-level barrier.

    Returns:
        `bucket_inputs[r]`, the (at most `fan_in`) partial paths the reducer for bucket `r`
        should fold. The mapper paths themselves when no level was needed.
    """
    from batcher.config import active_config
    from batcher.dist.reduction import chunks

    n_mappers = len(shuffle_paths)
    fan_in = max(2, active_config().flow_control.shuffle_fan_in)
    bucket_inputs = [[paths[r] for paths in shuffle_paths] for r in range(n_reducers)]
    if n_mappers < _TREE_MIN_MAPPERS * fan_in:
        return bucket_inputs

    from batcher.carbonite.resilience import gather_with_backups

    level = 0
    while max((len(b) for b in bucket_inputs), default=0) > fan_in:
        # One task per (bucket, chunk), flattened into a single barrier so a level's tasks
        # across every bucket run concurrently — the parallelism the tree exists for. A
        # chunk of one carries straight over: it is already the merge of itself, and a task
        # would cost a full read and write to reproduce bytes that exist.
        merged: list[list[str]] = [[] for _ in range(n_reducers)]
        specs: list[tuple[int, int, list[str]]] = []
        for r, inputs in enumerate(bucket_inputs):
            for i, chunk in enumerate(chunks(inputs, fan_in)):
                if len(chunk) == 1:
                    merged[r].append(chunk[0])
                else:
                    specs.append((r, i, list(chunk)))

        def _combine_for(t: int, _specs=specs, _level=level):
            r, i, chunk = _specs[t]
            return _combine_task.remote(
                group_keys_json,
                aggregates_json,
                chunk,
                work_dir,
                f"c{_level}_r{r}_{i}.arrow",
                cfg_json,
            )

        refs = [_combine_for(t) for t in range(len(specs))]
        out = gather_with_backups(refs, _combine_for, pol, stage=f"aggregate.combine.{level}")
        for (r, _i, _chunk), paths in zip(specs, out, strict=True):
            merged[r].extend(paths)
        widest_before = max((len(b) for b in bucket_inputs), default=0)
        bucket_inputs = merged
        # A level that shrank nothing is one where every chunk declined on memory, and a
        # further level would decline identically — so stop rather than spin. The bucket goes
        # to the reducer as wide as it started, which is the out-of-core fold's input.
        if max((len(b) for b in bucket_inputs), default=0) >= widest_before:
            break
        level += 1
    return bucket_inputs


def _reduce_task(
    group_keys_json, aggregates_json, input_paths, work_dir, reducer_id, engine_config=""
):
    """Combine+finalize the partials routed to this reducer. Returns
    ``(ipc_path, row_count)`` for a non-empty bucket, else ``(None, 0)`` — the exact
    count lets the driver size a materialized intermediate without reading it back.

    The partials are folded **incrementally** with `combine` — each input merged into one
    running state (bounded by the group cardinality) and then dropped — rather than
    materializing them all at once. So a high-fan-in or skewed reducer's peak memory is one
    running state + one input, not the sum of its inputs. `combine` is associative and
    commutative, so the folded result is what combining them all in one call gives — the
    mergeable-algebra invariant.

    How *many* inputs there are is `_tree_combine_buckets`'s answer, not the mapper count:
    the tree collapses a wide bucket to at most `shuffle_fan_in` partials first, so this
    fold's length no longer grows with the cluster. Bounded memory was never the same
    property as a bounded critical path, and folding a `W`-long list incrementally only ever
    bought the first.

    The running state itself is still O(groups): a high-cardinality `GROUP BY` / `DISTINCT`
    can make it exceed one worker's RAM even though each *input* is bounded. When the shipped
    memory envelope (`memory_budget_bytes`) is set and this reducer's on-disk input exceeds
    it, the fold is done **out of core** by `combine_finalize_spilling` — it reads the shuffle
    files one at a time and grace-partitions them to disk, so the reducer completes instead of
    OOMing. Result-identical to the in-memory fold (the mergeable algebra holds out-of-core).
    On-disk bytes are a floor for the in-memory state (IPC never expands), so the small-state
    common case keeps the fast in-memory fold and pays nothing."""
    import os as _os

    nat = engine()
    from batcher.dist.shuffle_io import read_ipc, reduce_envelope, write_ipc

    envelope = reduce_envelope(engine_config)
    if envelope.budget > 0:
        on_disk = sum(_os.path.getsize(p) for p in input_paths if _os.path.exists(p))
        if on_disk > envelope.budget:
            result = nat.combine_finalize_spilling(
                group_keys_json,
                aggregates_json,
                list(input_paths),
                envelope.budget,
                envelope.spill_dir or work_dir,
                envelope.compression,
            )
            if result.num_rows == 0:
                return (None, 0)
            path = _os.path.join(work_dir, f"reduce_{reducer_id}.arrow")
            write_ipc([result], path)
            return (path, result.num_rows)

    running = None
    for path in input_paths:
        batch = read_ipc(path)
        if not batch:
            continue
        merged = batch if running is None else [running, *batch]
        running = nat.combine(group_keys_json, aggregates_json, merged)
    if running is None:
        return (None, 0)
    result = nat.combine_finalize(group_keys_json, aggregates_json, [running])
    if result.num_rows == 0:
        return (None, 0)
    path = _os.path.join(work_dir, f"reduce_{reducer_id}.arrow")
    write_ipc([result], path)
    return (path, result.num_rows)
