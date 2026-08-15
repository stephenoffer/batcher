"""Distributed write — parallel data-file writers + one driver-side commit.

Each worker writes its own shard's data files (its own ``part-NNNNN`` files, with
Hive partitioning if requested) and returns a list of `WrittenFile` locators —
no data flows back through the driver. The driver concatenates the locators into
one `WriteManifest` (a commutative merge) and the caller performs a single
`commit`. This is the file-sink form of the two-phase write the lakehouse sinks
build on for ACID commits.

Three entry points:

* `_distributed_write` re-shards an already-collected result table — the last
  resort, for shapes that could not keep their result partitioned.
* `_distributed_write_plan` is the *streaming* path: each worker reads its own
  source partition, runs the (breaker-free) plan, and writes its output directly,
  so a result larger than the driver's memory never lands on the driver.
* `_distributed_write_partitioned` is the same streaming path over a breaker's
  result that stayed partitioned (`materialize=False`), which is what keeps an
  `ORDER BY` / `GROUP BY` / `JOIN` followed by a write off the driver too.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from batcher._internal.mathx import ceil_div
from batcher._internal.native import engine
from batcher.io.base._layout import FileLayout
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan
from batcher.plan.types import logical_bytes, retained_bytes


def _distributed_write(
    sink: Any,
    table: pa.Table,
    path: str,
    partition_by: list[str] | None,
    workers: int,
    *,
    layout: FileLayout | None = None,
    resume: bool = False,
) -> WriteManifest:
    """Write `table` as `workers` shards in parallel, returning the merged manifest.

    The result is always a directory of ``part-*`` files (one per shard), so
    shards never collide. Single-node callers use `Sink.write_partitioned`
    directly; this path is for an already-collected distributed result.

    `layout` and `resume` are the caller's file-sizing and idempotence requests, and both
    have to reach the worker: a shard resolves its own row cap (the whole table *is* here,
    but the split is by shard, so a cap resolved against the whole table would be wrong by
    a factor of `workers`), and a `resume` the workers never saw made a re-run after a
    preemption rewrite every file it had already finished.
    """
    from batcher.dist.executors.ray_runtime import _ensure_ray, gather_map_results

    _ensure_ray(workers)
    shards = _write_shards(table, partition_by, workers)
    layout = layout or FileLayout()

    # Gather with preemption recovery: a write-shard whose worker is lost is resubmitted
    # onto a survivor. Each shard writes a deterministic `part-{idx}` file, so a resubmit
    # overwrites any partial file the dead worker left — idempotent, no orphan.
    results: list[list[WrittenFile]] = gather_map_results(
        lambda idx: _write_shard.remote(
            sink, shards[idx], path, partition_by, idx, layout.for_shard(idx, len(shards)), resume
        ),
        len(shards),
    )
    return WriteManifest(tuple(f for shard_files in results for f in shard_files))


def _write_shards(table: pa.Table, partition_by: list[str] | None, workers: int) -> list[pa.Table]:
    """Divide a collected result into the shards the write workers each take.

    Without `partition_by` a shard is a contiguous row range — every row is equivalent, so
    equal ranges are the balanced split.

    With `partition_by` the split is **by partition key** instead, because row ranges are
    what makes the small-files problem: a key's rows are spread over every range, so each
    of the W shards writes its own file into each of the P partition directories and the
    write emits W x P files rather than P. At 8 workers and 200 daily partitions that is
    1,600 files where 200 were asked for, and each is 1/8th the size — which is the shape
    that makes the *next* query slow, since a scan then pays 1,600 footer reads. Grouping
    whole keys into a shard is Spark's ``repartition($"p").write.partitionBy("p")``, done
    without making the caller ask for it.

    Grouping by key alone would trade one problem for its mirror image: three partitions
    over a hundred workers is three shards, so ninety-seven workers idle while three write
    the entire result. So a key larger than an even share of the rows is *split* across
    shards — the same skew rule the shuffle uses, applied to the write. A key under that
    share stays whole and lands in one file; a key over it becomes as many files as it has
    even shares, which is what a caller wants anyway from a partition too big to be one
    file. Pieces are then packed largest-first onto the currently-lightest shard (LPT bin
    packing), so the residue after splitting does not pile onto one worker.

    Splitting stops at `skew_min_bucket_rows`, the same floor the join's skew detection
    uses, for the same reason: below it a "share" is smaller than the shards are worth,
    and splitting there would recreate the small-files problem out of nothing on a result
    that comfortably fits in one file per key.

    The division is deterministic — a stable sort over a deterministic run order — which
    `resume` depends on: a re-run must assign the same rows to the same part file.

    Args:
        table: The collected result to divide.
        partition_by: Hive partition columns, or None for an unpartitioned write.
        workers: The most shards to produce.

    Returns:
        One table per shard, always at least one (an empty result still writes an empty
        shard, so the output directory exists and reads back as an empty relation).
    """
    n = table.num_rows
    if not partition_by or n == 0:
        per = max(1, ceil_div(n, workers))
        shards = [table.slice(i * per, per) for i in range(workers) if i * per < n]
        return shards or [table.slice(0, 0)]

    import pyarrow.compute as pc

    from batcher.config import active_config
    from batcher.io.base._hive import hive_partition_run_starts

    keys = [(c, "ascending") for c in partition_by]
    ordered = table.take(pc.sort_indices(table, sort_keys=keys))
    starts = hive_partition_run_starts(ordered, list(partition_by), pc)
    bounds = zip(starts, [*starts[1:], ordered.num_rows], strict=True)
    share = max(ceil_div(n, workers), active_config().execution.skew_min_bucket_rows, 1)
    pieces: list[pa.Table] = []
    for begin, end in bounds:
        rows = end - begin
        if rows <= share:
            pieces.append(ordered.slice(begin, rows))
            continue
        # A key bigger than one worker's share: cut it into shares so it can run wide.
        # The length is clamped to the run's own end -- `Table.slice` clamps only to the
        # end of the *table*, so an unclamped last piece runs on into the next key and
        # those rows are then written twice.
        pieces.extend(ordered.slice(o, min(share, end - o)) for o in range(begin, end, share))
    if len(pieces) <= 1:
        return [ordered]
    buckets: list[list[pa.Table]] = [[] for _ in range(min(workers, len(pieces)))]
    loads = [0] * len(buckets)
    # Two pieces of one key landing in the same shard is harmless: `write_partitioned`
    # regroups a shard by key, so they rejoin into a single file there.
    for piece in sorted(pieces, key=lambda t: -t.num_rows):
        lightest = loads.index(min(loads))
        buckets[lightest].append(piece)
        loads[lightest] += piece.num_rows
    return [pa.concat_tables(b) for b in buckets if b]


def _write_shard(
    sink: Any,
    shard: pa.Table,
    path: str,
    partition_by: list[str] | None,
    idx: int,
    layout: FileLayout | None = None,
    resume: bool = False,
) -> list[WrittenFile]:
    return sink.write_partitioned(
        shard,
        path,
        partition_by=partition_by,
        file_index=idx,
        resume=resume,
        max_rows_per_file=_shard_rows_per_file(shard, layout),
    )


def _shard_rows_per_file(table: pa.Table, layout: FileLayout | None) -> int | None:
    """Resolve `layout` against the shard the worker is holding.

    `num_files` and `target_bytes_per_file` both need the data's size, which on a
    streaming distributed write exists only on the worker — the driver deliberately never
    sees these rows. Resolving here is what makes the two layouts mean the same thing
    single-node and distributed instead of being silently dropped on a cluster.
    """
    if layout is None:
        return None
    return layout.rows_per_file(table.num_rows, logical_bytes(table))


def _distributed_write_partitioned(
    result,
    path: str,
    fmt: str,
    sink_kwargs: dict[str, Any] | None,
    partition_by: list[str] | None,
    workers: int,
    *,
    layout: FileLayout | None = None,
    resume: bool = False,
) -> WriteManifest:
    """Write a breaker's already-partitioned result from the workers that hold it.

    `result` is what a stage run with ``materialize=False`` hands back: a
    `MaterializedSource` over the reducers' Arrow-IPC buckets, or a
    `FlightMaterializedSource` over buckets still resident on the fleet. Either advertises
    one `Split` per bucket, so this is the ordinary streaming distributed write with a
    scan of that intermediate as its plan — each worker reads one bucket and writes it,
    and only `WrittenFile` locators come back.

    That is the whole point: a `GROUP BY`, `ORDER BY`, `JOIN`, `DISTINCT` or window
    followed by a write used to be collected onto the driver in full and re-sharded from
    there, so the one terminal whose output is the size of the result was also the one
    that funnelled it through a single process. For the row-preserving breakers (sort,
    window) that is the entire relation.

    Args:
        result: The stage's partitioned result — any source advertising per-bucket splits.
        path: Destination path or table identifier.
        fmt: Sink format name.
        sink_kwargs: Constructor arguments for the worker-side sink.
        partition_by: Hive partition columns, or None.
        workers: Worker fan-out.
        layout: The caller's file-sizing request, resolved per shard on the worker.
        resume: Skip shards whose part file already exists.

    Returns:
        The merged `WriteManifest` over every worker's written files.
    """
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    scan = Scan(0, SchemaRef.from_arrow(result.schema()))
    return _distributed_write_plan(
        scan,
        [result],
        path,
        fmt,
        sink_kwargs,
        partition_by,
        workers,
        layout=layout,
        resume=resume,
    )


def _distributed_write_plan(
    plan: LogicalPlan,
    sources: list[Source],
    path: str,
    fmt: str,
    sink_kwargs: dict[str, Any] | None,
    partition_by: list[str] | None,
    workers: int,
    *,
    layout: FileLayout | None = None,
    resume: bool = False,
) -> WriteManifest:
    """Streaming distributed write for a breaker-free single-source plan.

    Each worker reads its source partition (with projection + predicate pushed),
    runs the plan, and writes its output files directly to the sink — only
    `WrittenFile` manifests return to the driver, so the full result never
    materializes there and no shared filesystem is required (the input partition is
    a split-manifest the worker reads from storage, or a shipped batch list).
    """
    from batcher import core
    from batcher.dist.executors.map import _distributed_map
    from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
    from batcher.dist.executors.plan_analysis import _relabel_single_source
    from batcher.dist.executors.ray_runtime import (
        _ensure_ray,
        engine_config_json,
        gather_map_results,
    )

    # A plan carrying a Python UDF (`map_batches` / batch inference) cannot be shipped as JSON
    # IR — the native engine can't run a Python callable, and `MapBatches.to_ir()` raises. Such
    # a plan goes through the UDF-aware distributed map instead, with the sink bound into each
    # worker so the post-inference rows are written where they were produced. Without this a
    # billion-row embedding write collected its whole result onto the driver first.
    if core.has_map_batches(plan):
        return _distributed_map(
            plan,
            sources,
            workers,
            write_spec={
                "fmt": fmt,
                "sink_kwargs": sink_kwargs,
                "path": path,
                "partition_by": partition_by,
                "layout": layout,
                "resume": resume,
            },
        )

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers
    map_plan, sid = _relabel_single_source(plan)
    map_ir = json.dumps(map_plan.to_ir())
    projection, predicate = source_pushdown(map_plan, 0)

    parts = partition_descriptors(sources[sid], workers, projection=projection, predicate=predicate)
    # Recover a preempted write-shard onto a survivor; the worker recomputes its
    # partition from the durable split descriptor and rewrites its `part-{idx}` file
    # (deterministic name ⇒ idempotent overwrite, no orphaned partial output).
    shard_layout = layout or FileLayout()
    results: list[list[WrittenFile]] = gather_map_results(
        lambda idx: _write_plan_shard.remote(
            map_ir,
            parts[idx],
            fmt,
            sink_kwargs,
            path,
            partition_by,
            idx,
            cfg_json,
            shard_layout.for_shard(idx, len(parts)),
            resume,
        ),
        len(parts),
    )
    return WriteManifest(tuple(f for shard_files in results for f in shard_files))


def _write_plan_shard(
    map_ir: str,
    partition: dict,
    fmt: str,
    sink_kwargs: dict[str, Any] | None,
    path: str,
    partition_by: list[str] | None,
    idx: int,
    engine_config: str,
    layout: FileLayout | None = None,
    resume: bool = False,
) -> list[WrittenFile]:
    """Read this shard's partition, run the plan over it, and write its files.

    The shard **streams** wherever it can: its rows are read, mapped and encoded a chunk at
    a time, so a worker holds one chunk rather than its whole share of the result. That is
    what makes the write scale with the data instead of with the cluster. Materializing
    made per-worker memory `rows / workers`, so doubling the input on a fixed cluster
    doubled every worker's peak — the write finished only because someone had sized the
    cluster to the answer, and a job that grew OOMed a node at a time.

    Two shapes still materialize, and both need the whole shard by definition rather than
    by omission:

    * `partition_by`, which fans rows out by key. Streaming it would emit a file per
      partition *per chunk*, turning one file per key into hundreds of small ones — the
      small-files problem `_write_shards` exists to avoid, recreated inside the shard.
    * `num_files`, which names a total and so cannot be resolved before the rows are
      counted. `target_bytes_per_file` is estimated from the first chunk's bytes-per-row
      instead, since it asks for a size rather than a count.
    """
    nat = engine()
    from batcher.dist.executors.partition_io import (
        iter_partition_descriptor,
        read_partition_descriptor,
    )
    from batcher.io.sink import SINKS

    sink = SINKS.get(fmt)(**(sink_kwargs or {}))
    if partition_by or (layout is not None and layout.num_files is not None):
        batches = read_partition_descriptor(partition)
        out = nat.execute_plan(map_ir, [batches], engine_config) if batches else []
        if not out or sum(b.num_rows for b in out) == 0:
            return []
        table = pa.Table.from_batches(out)
        return sink.write_partitioned(
            table,
            path,
            partition_by=partition_by,
            file_index=idx,
            resume=resume,
            max_rows_per_file=_shard_rows_per_file(table, layout),
        )

    mapped = _map_stream(nat, map_ir, iter_partition_descriptor(partition), engine_config)
    first = next(mapped, None)
    if first is None:
        # No rows at all. The driver writes one empty file centrally for the whole write
        # (it is the only party that knows *every* shard was empty), so a shard that
        # writes its own here would leave an extra empty part beside the real ones.
        return []
    from itertools import chain

    stream = chain([first], mapped)
    cap = _stream_rows_per_file(first, layout)
    if cap is None:
        # One file per shard, named exactly as the materializing path named it, so a
        # `resume` re-run and a reader's glob both see the same layout as before.
        return [
            sink.write_stream_shard(
                stream, path, file_index=idx, schema=first.schema, resume=resume
            )
        ]
    return sink.write_stream_parts(
        stream,
        path,
        max_rows_per_file=cap,
        schema=first.schema,
        file_index=idx,
        resume=resume,
    )


def _map_stream(nat: Any, map_ir: str, batches: Any, engine_config: str) -> Any:
    """Run `map_ir` over `batches` a chunk at a time, yielding the mapped output batches.

    The plan reaching here is breaker-free (`_distributed_write_plan` only routes those),
    so running it over a prefix of the input gives exactly the rows running it over the
    whole input would have given for that prefix — which is what lets the shard be encoded
    before the rest of it has been read.

    The chunk budget is the shuffle map side's `_FOLD_CHUNK_BYTES`, deliberately and not by
    coincidence: it is the same question (how much of a partition a worker may hold while
    folding it) with the same answer, and a second constant would drift from it.
    """
    from batcher.dist.executors.partition_io.folds import _FOLD_CHUNK_BYTES

    chunk: list = []
    size = 0
    for batch in batches:
        chunk.append(batch)
        size += retained_bytes(batch)
        if size >= _FOLD_CHUNK_BYTES:
            yield from nat.execute_plan(map_ir, [chunk], engine_config)
            chunk, size = [], 0
    if chunk:
        yield from nat.execute_plan(map_ir, [chunk], engine_config)


def _stream_rows_per_file(first: pa.RecordBatch, layout: FileLayout | None) -> int | None:
    """The row cap a streaming shard rolls files over at, or None for one file per shard.

    `target_bytes_per_file` is resolved against the *first batch's* bytes-per-row rather
    than the shard's, because a streaming shard never holds its own total. Both are
    estimates of the same ratio, and a batch of 16,384 rows is a large enough sample of it
    that the resulting files land near the target.
    """
    if layout is None or layout.is_default:
        return None
    if layout.max_rows_per_file is not None:
        return layout.max_rows_per_file
    if layout.target_bytes_per_file is not None and (width := logical_bytes(first)) > 0:
        return max(1, first.num_rows * layout.target_bytes_per_file // width)
    return None
