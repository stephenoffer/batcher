"""Shared partitioning + post-breaker helpers for the distributed operators.

`_partition_source` assigns a source's *splits* to per-worker partition files —
lazily, so the driver never materializes the whole source for a splittable
source (Parquet row-groups, lakehouse fragments, …): it writes a tiny pickled
*split manifest* per worker and each worker reads only its slice directly from
storage. A source that cannot subdivide (in-memory / iterator) falls back to the
eager read-and-range-slice path, reproducing the previous behavior exactly.

`read_partition` is the worker-side reader that accepts either kind of partition
file, and `_apply_above` re-runs operators carried above a breaker single-node.

`merge_boundaries` + `bucketize` are the *range*-partitioning helpers (the distributed
sort's split-by-value step), shared by the disk and Flight sort paths so they stay in
lockstep — a distributed sort range-partitions rows by the leading key so the buckets
concatenate, in bucket order, to a globally sorted result with no final merge.
"""

from __future__ import annotations

import dataclasses
import os
import pickle

import pyarrow as pa

from batcher.dist.executors.scan_read import (
    _SCAN_PREFETCH,
    _SPLIT_TARGET_BYTES,
    _read_split_batches,
)
from batcher.io.source import InMemorySource, Source
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.logical import LogicalPlan, Scan
from batcher.plan.schema import SchemaRef


def source_pushdown(plan: LogicalPlan, source_id: int) -> tuple[list[str] | None, dict | None]:
    """Compute the projection + pushed predicate for `source_id` in `plan`.

    Mirrors what single-node execution pushes to a source, so a distributed map
    task reads only the columns/rows it needs. Returns ``(None, None)`` if the
    analysis can't run (e.g. an opaque `MapBatches` node) — the worker then reads
    everything and the engine's operators filter/project, which is still correct.

    **Pass the whole operator sub-tree, not just its map prefix** — see
    `consumer_pushdown`, which is what most shuffle operators want.
    """
    try:
        from batcher.kyber.rules.projections import (
            required_columns_per_source,
            required_predicates_per_source,
        )

        projection = required_columns_per_source(plan).get(source_id)
        predicate = required_predicates_per_source(plan).get(source_id)
        return projection, predicate
    except Exception:
        return None, None


def consumer_pushdown(
    consumer: LogicalPlan, map_plan: LogicalPlan, source_id: int = 0
) -> tuple[list[str] | None, dict | None]:
    """`source_pushdown` for a shuffle operator's map prefix, *including* the operator above it.

    A shuffle operator's map prefix (`agg.input`) alone answers the wrong question: the prefix
    of `group_by("k").agg(sum("v"))` is a bare `Scan`, which requires every column it has, so
    the projection came back as the source's full schema — the whole wide table read off disk
    to answer a two-column aggregate (27 GB vs 6 GB at 1B rows). Re-parenting the operator onto
    the relabeled prefix restores the context Kyber's `required_columns_per_source` needs, so
    dist schedules against Kyber's own decision (`["k","v"]`, what single-node reads) instead
    of a worse re-derived one. Pass-through operators (sort/window/distinct) narrow nothing.
    """
    import dataclasses

    try:
        rooted = dataclasses.replace(consumer, input=map_plan)
    except Exception:
        return source_pushdown(map_plan, source_id)
    return source_pushdown(rooted, source_id)


def _scan_splits(
    source: Source,
    workers: int,
    predicate: dict | None = None,
    projection: list[str] | None = None,
) -> list[Split]:
    """A source's splits sized for a `workers`-wide distributed read.

    `predicate` is the filter pushed to this scan. A source that can answer it from
    metadata — a lakehouse table format, whose log records per-file bounds — returns
    only the files that can match, so the eliminated ones never become tasks at all
    (see `io.source.plan_splits`). Sources that cannot prune return everything, and the
    engine's `Filter` re-checks the rows, so this is always safe.

    One split per native chunk (Parquet row-group) is ideal for *balance* and *prefetch
    overlap*, but a dataset with thousands of small row-groups makes per-request latency
    dominate, so we coalesce adjacent ones up to `_SPLIT_TARGET_BYTES`. The catch: over-
    coalescing a *small* dataset collapses it below the fan-out — sf10's 60 row-group
    splits became 10 whole-file splits at a 64 MB target, starving the 8 workers of
    balance and prefetch and *regressing* it 16.6 s → 40 s. So coalesce only while the
    result still has at least `workers x prefetch` splits (enough to fill every worker's
    prefetch window); below that, keep the fine splits. Large datasets (sf100: 4,900 →
    1,700) still coalesce; small ones keep their parallelism.
    """
    from batcher.io.source import plan_splits

    # Plan the COALESCED shape first, and only fall back to the fine one if coalescing
    # collapsed the dataset below the fan-out. This is the same decision as before — coalescing
    # only ever *reduces* the split count, so "fine is already at or below the floor" and
    # "coalesced is below the floor" select the same branch — but it inverts *which* case pays
    # for a second plan. Before, the large dataset (the one whose plan is expensive) always
    # planned twice: a full pass over every file, discarded, then an identical pass. Measured
    # at 50,000 Parquet files that was 28.4 s → 58.3 s, and every footer was read twice because
    # the footer cache holds 1,024 entries. Now the large dataset plans once, and only a *small*
    # one — where a second plan is cheap by definition — can pay for two.
    floor = max(1, workers) * max(1, _SCAN_PREFETCH)
    coalesced = plan_splits(
        source, target_size=_SPLIT_TARGET_BYTES, predicate=predicate, projection=projection
    )
    if len(coalesced) >= floor:
        return coalesced
    return plan_splits(source, predicate=predicate, projection=projection)


def _balance(splits: list[Split], workers: int) -> list[list[Split]]:
    """Greedily bin-pack splits into `workers` groups balanced by row count.

    Splits with an unknown row count are weighted as 1 so they spread evenly.
    Largest-first assignment keeps the per-worker load roughly equal.
    """
    groups: list[list[Split]] = [[] for _ in range(workers)]
    loads = [0] * workers
    ordered = sorted(splits, key=lambda s: s.row_count() or 0, reverse=True)
    for s in ordered:
        i = min(range(workers), key=lambda w: loads[w])
        groups[i].append(s)
        loads[i] += s.row_count() or 1
    return groups


def _contiguous(splits: list[Split], workers: int) -> list[list[Split]]:
    """Group splits into `workers` contiguous, source-ordered runs (order preserved).

    Unlike `_balance` (which reorders splits largest-first for even load), group 0 holds the
    source's first splits, group 1 the next, and so on — each a contiguous near-equal-count
    run. Callers whose correctness needs the concatenation of per-partition results to
    reproduce the source's global row order (distributed `LIMIT` / `with_row_index`) require
    this: a `_balance` assignment puts non-adjacent splits in one partition, so a per-partition
    prefix interleaves rows from different parts of the source.
    """
    groups: list[list[Split]] = [[] for _ in range(workers)]
    if workers <= 0 or not splits:
        return groups
    target = max(1, -(-sum(s.row_count() or 1 for s in splits) // workers))  # ceil per group
    w, load = 0, 0
    for s in splits:
        groups[w].append(s)
        load += s.row_count() or 1
        if load >= target and w < workers - 1:
            w, load = w + 1, 0
    return groups


def _slice_rows_evenly(batches: list[pa.RecordBatch], workers: int) -> list[list[pa.RecordBatch]]:
    """Split an ordered batch list into `workers` groups of near-equal total row count.

    Batches are sliced (zero-copy) at group boundaries, so even a single large batch is
    spread evenly across every worker rather than assigned whole to one — order preserved,
    the first ``total % workers`` groups getting one extra row. Empty groups are returned
    as ``[]`` (the caller substitutes a schema-only empty batch) so the per-worker shape
    stays uniform. The row counts of the returned groups sum to the input's total.
    """
    total = sum(b.num_rows for b in batches)
    groups: list[list[pa.RecordBatch]] = [[] for _ in range(workers)]
    if workers <= 1 or total == 0:
        groups[0] = [b for b in batches if b.num_rows]
        return groups
    base, extra = divmod(total, workers)
    targets = [base + (1 if i < extra else 0) for i in range(workers)]
    w, filled = 0, 0
    for b in batches:
        off, n = 0, b.num_rows
        while off < n:
            if w < workers - 1 and filled >= targets[w]:
                w += 1
                filled = 0
                continue
            room = (targets[w] - filled) if w < workers - 1 else (n - off)
            take = min(room, n - off)
            groups[w].append(b.slice(off, take))
            off += take
            filled += take
    return groups


def _partition_source(
    source: Source,
    workers: int,
    work_dir: str,
    tag: str = "P",
    projection: list[str] | None = None,
    predicate: dict | None = None,
) -> list[str]:
    """Assign a source's splits to `workers` partition files.

    Splittable sources write one pickled split-manifest per worker (no data read
    on the driver), carrying the source's pushed `projection`/`predicate` so each
    worker reads only the columns/rows it needs. Non-splittable sources fall back
    to eager range-slicing into Arrow-IPC files (projection/predicate are applied
    once on the driver before slicing). Either kind is read back with
    `read_partition`.
    """
    splits = _scan_splits(source, workers, predicate, projection)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return _eager_range_split(source, workers, work_dir, tag, projection, predicate)

    from batcher.config import active_config
    from batcher.dist.shuffle_io import write_ipc

    # Broken-record policy resolved on the driver, embedded per manifest → reaches workers.
    on_read_error = active_config().distributed.on_read_error
    meta = {"projection": projection, "predicate": predicate, "on_read_error": on_read_error}
    schema = source.schema()
    paths = []
    for i, group in enumerate(_balance(splits, workers)):
        if group:
            path = os.path.join(work_dir, f"{tag}_part_{i}.splits")
            with open(path, "wb") as fh:
                pickle.dump({"splits": group, **meta}, fh)
        else:
            # Empty group: a schema-only IPC partition keeps map tasks uniform.
            path = os.path.join(work_dir, f"{tag}_part_{i}.arrow")
            write_ipc([_projected_empty_batch(schema, projection)], path)
        paths.append(path)
    return paths


def _eager_range_split(
    source: Source,
    workers: int,
    work_dir: str,
    tag: str,
    projection: list[str] | None = None,
    predicate: dict | None = None,
) -> list[str]:
    """Stream a non-splittable source round-robin into per-worker IPC partition files.

    Reads the source one batch at a time (`iter_source`) and distributes batches
    across the worker files, so the driver never materializes the whole source — a
    larger-than-RAM streaming input is partitioned in bounded memory rather than
    OOMing the driver. Projection/predicate are applied during the streaming read so
    the IPC shards carry only the needed columns/rows, matching a splittable source's
    per-worker reads. Each worker re-partitions by key downstream, so round-robin vs
    range assignment never changes the result.
    """
    from batcher.dist.shuffle_io import write_ipc_round_robin
    from batcher.io.source import iter_source

    paths = [os.path.join(work_dir, f"{tag}_part_{i}.arrow") for i in range(workers)]
    write_ipc_round_robin(
        iter_source(source, projection, predicate),
        _projected_schema(source, projection),
        paths,
    )
    return paths


def _projected_schema(source: Source, projection: list[str] | None) -> pa.Schema:
    """The source's schema restricted to `projection` (the empty-partition schema)."""
    schema = source.schema()
    if projection is None:
        return schema
    return pa.schema([schema.field(c) for c in projection])


def partition_descriptors(
    source: Source,
    workers: int,
    projection: list[str] | None = None,
    predicate: dict | None = None,
    preserve_order: bool = False,
) -> list[dict]:
    """Partition a source into `workers` in-memory descriptors — no shared filesystem.

    Unlike `_partition_source` (which writes per-worker files to a driver-local
    `work_dir`), this returns descriptors meant to be passed as Ray task/actor args,
    so the Flight (multi-node) path needs no shared filesystem:

    * **Splittable** sources yield a small split-manifest per worker — only split
      *references* (file + row-group), so each worker reads its slice directly from
      storage. Nothing but the manifest crosses Ray; fully shared-nothing.
    * **Non-splittable** (in-memory / iterator) sources are eagerly read and
      range-sliced into per-worker batch lists. Those batches are driver-resident
      already, so shipping them as args is bounded input movement (not shuffle).

    `preserve_order` assigns splittable splits as contiguous source-ordered runs
    (`_contiguous`) instead of load-balanced (`_balance`), so the partition-index-assembled
    concatenation reproduces the source's global row order — required by the order-sensitive
    `LIMIT` / `with_row_index` paths (the in-memory branch is already order-preserving).

    Read back with `read_partition_descriptor`.
    """
    splits = _scan_splits(source, workers, predicate, projection)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        from batcher.io.source import iter_source

        # Range-slice the source into per-worker batch lists of near-equal ROW COUNT, not
        # round-robin whole batches: a source that arrives as one large batch (a
        # `from_arrow` table, an in-memory image/tensor set) must still fan out evenly
        # across every worker instead of landing entirely on worker 0 while the rest sit
        # idle — the imbalance the round-robin-by-batch scheme produced whenever the batch
        # count was below the worker count (the common case for GPU/UDF inputs, which are
        # few, wide rows). Slices are zero-copy views, so this is bounded one-time input
        # movement (Ray task args), not shuffle, and matches the disk path's range split.
        # `_slice_rows_evenly` cuts contiguous, source-ordered slices, so this branch already
        # satisfies `preserve_order`.
        proj_schema = _projected_schema(source, projection)
        batches = list(iter_source(source, projection, predicate))
        groups = _slice_rows_evenly(batches, workers)
        empty = pa.RecordBatch.from_pylist([], schema=proj_schema)
        return [{"batches": g or [empty]} for g in groups]

    schema = source.schema()
    descriptors: list[dict] = []
    assign = _contiguous if preserve_order else _balance
    for group in assign(splits, workers):
        if group:
            descriptors.append({"splits": group, "projection": projection, "predicate": predicate})
        else:
            # Empty group: a schema-only batch keeps the per-worker shape uniform.
            cols = projection or schema.names
            empty_schema = pa.schema([schema.field(c) for c in cols])
            descriptors.append({"batches": [pa.RecordBatch.from_pylist([], schema=empty_schema)]})
    return descriptors


def descriptor_rows(desc: dict) -> int:
    """Approximate row count of a partition descriptor — for skew-aware per-task sizing.

    Splittable partitions sum their splits' footer-derived row counts (no I/O — the
    count was captured when the split was built); in-memory partitions sum their batch
    sizes. Used to give each distributed task a CPU share proportional to its data (a
    heavier partition gets more cores, a tiny one a fraction), so per-task allocation
    tracks data skew that LPT balancing could not fully even out.
    """
    if "splits" in desc:
        total = 0
        for s in desc["splits"]:
            rows = getattr(s, "rows", None)
            if rows is None and hasattr(s, "row_count"):
                try:
                    rows = s.row_count()
                except Exception:
                    rows = None
            total += rows or 0
        return total
    return sum(b.num_rows for b in desc.get("batches", []))


def read_partition_descriptor(desc: dict) -> list[pa.RecordBatch]:
    """Read a descriptor from `partition_descriptors` (split-manifest or batch list).

    A split-manifest reads each split directly from storage with the pushed
    projection/predicate (splits that don't accept a predicate ignore it — the
    engine's `Filter` re-checks, so pushdown stays safe); a batch list is returned
    as-is. A manifest fully eliminated by predicate pushdown still returns one
    schema-only batch so downstream native operators always have a schema.
    """
    if "splits" in desc:
        projection, predicate, splits = desc["projection"], desc["predicate"], desc["splits"]
        out = list(_read_split_batches(splits, projection, predicate))
        if not out and splits:
            schema = splits[0].schema()
            if projection is not None:
                schema = pa.schema([schema.field(c) for c in projection])
            out = [pa.RecordBatch.from_pylist([], schema=schema)]
        return out
    return desc["batches"]


def iter_partition_descriptor(desc: dict):
    """Yield a descriptor's batches one at a time — the streaming form of
    `read_partition_descriptor`, so the map side can aggregate a partition without
    holding it whole. Same split/empty-schema handling."""
    if "splits" in desc:
        projection, predicate, splits = desc["projection"], desc["predicate"], desc["splits"]
        emitted = False
        for b in _read_split_batches(splits, projection, predicate):
            emitted = True
            yield b
        if not emitted and splits:
            schema = splits[0].schema()
            if projection is not None:
                schema = pa.schema([schema.field(c) for c in projection])
            yield pa.RecordBatch.from_pylist([], schema=schema)
        return
    yield from desc["batches"]


# The streaming map-side folds (`streaming_partial_aggregate`, `streaming_topn`) live in
# `folds.py` — a family of their own, kept out of this file's line budget.


def _projected_empty_batch(schema: pa.Schema, projection) -> pa.RecordBatch:
    """One empty batch carrying `schema` (restricted to `projection`) — the schema a
    fully-pruned/empty partition still hands downstream so the native ops get a schema."""
    if projection is not None:
        schema = pa.schema([schema.field(c) for c in projection])
    return pa.RecordBatch.from_pylist([], schema=schema)


def read_partition(path: str) -> list[pa.RecordBatch]:
    """Read a partition file written by `_partition_source` (manifest or IPC).

    A split-manifest carries the pushed `projection`/`predicate`; each split reads
    only the needed columns/rows. Splits that don't accept a predicate ignore it
    (the engine's `Filter` re-checks), so pushdown is always safe.
    """
    if path.endswith(".splits"):
        with open(path, "rb") as fh:
            manifest = pickle.load(fh)
        projection = manifest["projection"]
        predicate = manifest["predicate"]
        splits = manifest["splits"]
        on_read_error = manifest.get("on_read_error", "error")
        out = list(_read_split_batches(splits, projection, predicate, on_read_error))
        if not out and splits:
            out = [_projected_empty_batch(splits[0].schema(), projection)]
        return out
    from batcher.dist.shuffle_io import read_ipc

    return read_ipc(path)


def iter_partition(path: str):
    """Yield a partition's batches one at a time — the streaming form of `read_partition`.

    Lets a consumer process a large partition without holding all of it in memory (the
    broadcast-join probe streams its left partition this way). Same manifest/IPC handling
    and same empty-partition schema guarantee as `read_partition`.
    """
    if path.endswith(".splits"):
        with open(path, "rb") as fh:
            manifest = pickle.load(fh)
        projection, predicate, splits = (
            manifest["projection"],
            manifest["predicate"],
            manifest["splits"],
        )
        on_read_error = manifest.get("on_read_error", "error")
        emitted = False
        for b in _read_split_batches(splits, projection, predicate, on_read_error):
            emitted = True
            yield b
        if not emitted and splits:
            yield _projected_empty_batch(splits[0].schema(), projection)
        return
    with pa.OSFile(path, "rb") as src, pa.ipc.open_stream(src) as reader:
        yield from reader


def materialize_reduce_output(result_paths, work_dir: str, fallback_schema: pa.Schema):
    """Wrap a breaker's reducer IPC output as a `MaterializedSource` (no driver collect).

    `result_paths` is the reducers' `[(ipc_path, row_count)]` (with `(None, 0)` for an
    empty bucket); the source owns `work_dir` and reclaims it on `cleanup()`. Shared by
    the distributed aggregate and join `materialize=False` paths.
    """
    from batcher.io.source import MaterializedSource
    from batcher.io.splits import IpcFileSplit

    files = [(p, n) for p, n in result_paths if p is not None]
    schema = IpcFileSplit(files[0][0]).schema() if files else fallback_schema
    return MaterializedSource(files, schema, work_dir=work_dir)


def _apply_above(above: list[LogicalPlan], agg_table: pa.Table) -> pa.Table:
    """Re-apply the operators above the aggregate to its result, single-node.

    Runs through `_single_node` — the same optimize-then-execute-locally fallback the rest of
    `dist` uses — rather than constructing a `Dataset`. Reaching for the public `api` from
    inside `dist` would make the conductor a dependency of the thing it conducts (an
    `api -> dist -> api` cycle), and `dist` already owns a runner for exactly this.
    """
    from batcher.dist.executors.ray_runtime import _single_node

    plan: LogicalPlan = Scan(0, SchemaRef.from_arrow(agg_table.schema))
    for node in reversed(above):  # innermost (closest to agg) first
        plan = dataclasses.replace(node, input=plan)
    # A zero-row table has NO batches (pyarrow drops empty chunks) and `InMemorySource`
    # needs one, so `filter(<no match>).distinct()` used to die here. Feed it a schema-only
    # batch: same rows, same schema, no crash.
    batches = agg_table.to_batches() or [pa.RecordBatch.from_pylist([], schema=agg_table.schema)]
    return _single_node(plan, [InMemorySource(batches)])
