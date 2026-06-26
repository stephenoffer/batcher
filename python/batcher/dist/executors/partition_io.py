"""Shared partitioning + post-breaker helpers for the distributed operators.

`_partition_source` assigns a source's *splits* to per-worker partition files —
lazily, so the driver never materializes the whole source for a splittable
source (Parquet row-groups, lakehouse fragments, …): it writes a tiny pickled
*split manifest* per worker and each worker reads only its slice directly from
storage. A source that cannot subdivide (in-memory / iterator) falls back to the
eager read-and-range-slice path, reproducing the previous behavior exactly.

`read_partition` is the worker-side reader that accepts either kind of partition
file. `_apply_above` re-runs operators carried above a breaker single-node;
`_empty_agg_table` builds the schema-only empty aggregate result.

`merge_boundaries` + `bucketize` are the *range*-partitioning helpers (the distributed
sort's split-by-value step), shared by the disk and Flight sort paths so they stay in
lockstep — a distributed sort range-partitions rows by the leading key so the buckets
concatenate, in bucket order, to a globally sorted result with no final merge.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
from inspect import signature

import pyarrow as pa

from batcher.io.source import InMemorySource, Source
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.logical import Aggregate, LogicalPlan, Scan
from batcher.plan.schema import SchemaRef


def source_pushdown(plan: LogicalPlan, source_id: int) -> tuple[list[str] | None, dict | None]:
    """Compute the projection + pushed predicate for `source_id` in `plan`.

    Mirrors what single-node execution pushes to a source, so a distributed map
    task reads only the columns/rows it needs. Returns ``(None, None)`` if the
    analysis can't run (e.g. an opaque `MapBatches` node) — the worker then reads
    everything and the engine's operators filter/project, which is still correct.
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
    splits = source.splits()
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return _eager_range_split(source, workers, work_dir, tag, projection, predicate)

    from batcher.dist.shuffle_io import write_ipc

    schema = source.schema()
    paths = []
    for i, group in enumerate(_balance(splits, workers)):
        if group:
            path = os.path.join(work_dir, f"{tag}_part_{i}.splits")
            with open(path, "wb") as fh:
                pickle.dump({"splits": group, "projection": projection, "predicate": predicate}, fh)
        else:
            # Empty group: a schema-only IPC partition keeps map tasks uniform.
            cols = projection or schema.names
            empty_schema = pa.schema([schema.field(c) for c in cols])
            path = os.path.join(work_dir, f"{tag}_part_{i}.arrow")
            write_ipc([pa.RecordBatch.from_pylist([], schema=empty_schema)], path)
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

    Read back with `read_partition_descriptor`.
    """
    splits = source.splits()
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        from batcher.io.source import iter_source

        # Stream the source round-robin into per-worker batch lists, holding one
        # batch at a time rather than concatenating the whole source into a Table
        # first (which doubled driver memory). The batches still ship as Ray task
        # args — bounded one-time input movement, not shuffle — but peak driver
        # memory drops to a single batch plus the per-worker references.
        proj_schema = _projected_schema(source, projection)
        groups: list[list[pa.RecordBatch]] = [[] for _ in range(workers)]
        for i, b in enumerate(iter_source(source, projection, predicate)):
            groups[i % workers].append(b)
        empty = pa.RecordBatch.from_pylist([], schema=proj_schema)
        return [{"batches": g or [empty]} for g in groups]

    schema = source.schema()
    descriptors: list[dict] = []
    for group in _balance(splits, workers):
        if group:
            descriptors.append({"splits": group, "projection": projection, "predicate": predicate})
        else:
            # Empty group: a schema-only batch keeps the per-worker shape uniform.
            cols = projection or schema.names
            empty_schema = pa.schema([schema.field(c) for c in cols])
            descriptors.append({"batches": [pa.RecordBatch.from_pylist([], schema=empty_schema)]})
    return descriptors


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
        out: list[pa.RecordBatch] = []
        for s in splits:
            out.extend(_split_read(s, projection, predicate))
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
        for s in splits:
            for b in _split_read(s, projection, predicate):
                emitted = True
                yield b
        if not emitted and splits:
            schema = splits[0].schema()
            if projection is not None:
                schema = pa.schema([schema.field(c) for c in projection])
            yield pa.RecordBatch.from_pylist([], schema=schema)
        return
    yield from desc["batches"]


def streaming_partial_aggregate(nat, map_ir, gk, aj, batches, engine_config, chunk_bytes=16 << 20):
    """Fold a partition's batches through the (breaker-free) map prefix + partial
    aggregate into one running partial, a byte-bounded chunk at a time.

    The map-side of a shuffle never holds the whole partition or the whole mapped output:
    peak is one chunk + the running partial. Correct by the mergeable invariant — combine
    of per-chunk partials equals one partial over the whole partition (the map prefix is
    breaker-free, so per-chunk application matches whole-partition application).
    """
    running = None
    chunk: list = []
    size = 0

    def fold(rows):
        nonlocal running
        mapped = nat.execute_plan(map_ir, [rows], engine_config)
        partial = nat.partial_aggregate(gk, aj, mapped)
        running = partial if running is None else nat.combine(gk, aj, [running, partial])

    for b in batches:
        chunk.append(b)
        size += b.nbytes
        if size >= chunk_bytes:
            fold(chunk)
            chunk, size = [], 0
    if chunk:
        fold(chunk)
    if running is None:  # empty partition → the empty (schema-bearing) partial
        running = nat.partial_aggregate(gk, aj, nat.execute_plan(map_ir, [[]], engine_config))
    return running


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
        out: list[pa.RecordBatch] = []
        for s in splits:
            out.extend(_split_read(s, projection, predicate))
        if not out and splits:
            # A partition fully eliminated by predicate pushdown (or an empty
            # source) still must carry a schema, so downstream operators — notably
            # the native partial-aggregate, which can't run "over empty input with
            # no schema" — have one. Emit a single schema-only batch.
            schema = splits[0].schema()
            if projection is not None:
                schema = pa.schema([schema.field(c) for c in projection])
            out = [pa.RecordBatch.from_pylist([], schema=schema)]
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
        emitted = False
        for s in splits:
            for b in _split_read(s, projection, predicate):
                emitted = True
                yield b
        if not emitted and splits:
            schema = splits[0].schema()
            if projection is not None:
                schema = pa.schema([schema.field(c) for c in projection])
            yield pa.RecordBatch.from_pylist([], schema=schema)
        return
    with pa.OSFile(path, "rb") as src, pa.ipc.open_stream(src) as reader:
        yield from reader


def _split_read(split: Split, projection: list[str] | None, predicate: dict | None) -> list:
    """Read a split, passing `predicate` only if its `read` accepts one."""
    if predicate is not None and "predicate" in signature(split.read).parameters:
        return split.read(projection, predicate=predicate)
    return split.read(projection)


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
    """Re-apply the operators above the aggregate to its result, single-node."""
    from batcher.api.dataset import Dataset

    plan: LogicalPlan = Scan(0, SchemaRef.from_arrow(agg_table.schema))
    for node in reversed(above):  # innermost (closest to agg) first
        plan = dataclasses.replace(node, input=plan)
    return Dataset(plan, [InMemorySource(agg_table.to_batches())]).collect()


def _empty_agg_table(agg: Aggregate) -> pa.Table:
    names = [k.alias for k in agg.group_keys] + [s.alias for s in agg.aggregates]
    return pa.table({n: pa.array([], pa.null()) for n in names})


def merge_boundaries(grids: list[tuple[list[float], int]], workers: int) -> list[float]:
    """Merge per-worker quantile grids into `workers-1` deduplicated range boundaries.

    Each grid is a `(sampled_cdf, row_count)` pair; empty/zero-row grids are dropped.
    Splits are roughly equal-size, so an unweighted concat of the sampled CDFs
    approximates the global distribution; the evenly spaced split points of that
    concat are the boundaries (dedup means equal keys never span a boundary). Returns
    `[]` for a single worker (one bucket).
    """
    import numpy as np

    samples = [np.asarray(grid, dtype=float) for grid, n in grids if grid and n]
    if not samples:
        return []
    qs = np.linspace(0, 1, workers + 1)[1:-1]
    if len(qs) == 0:
        return []
    return np.unique(np.quantile(np.concatenate(samples), qs)).tolist()


def bucketize(
    batches: list[pa.RecordBatch],
    key_name: str,
    boundaries: list[float],
    n_buckets: int,
    nulls_first: bool,
    descending: bool,
) -> list[list[pa.RecordBatch]]:
    """Split `batches` into `n_buckets` lists by the leading key's `boundaries`.

    Bucket `b` holds keys in the `b`-th open interval of `boundaries`, so the buckets
    are globally ordered and equal keys never span a boundary
    (`searchsorted(side="right")` keeps equal keys in one bucket). Nulls go to whichever
    end the caller's final concatenation places first/last: for a descending sort the
    driver concatenates buckets high→low, so the "front" bucket is `n_buckets-1` (else
    `0`); nulls land in the front bucket when `nulls_first`, else the opposite end —
    matching single-node null ordering exactly. Boundary precision affects only balance,
    never the result.

    The per-row bucketing + scatter runs in the Rust data plane
    (`nat.range_partition_batches`, the range counterpart of the hash
    `partition_batches`), so this stays off the per-row Python hot path.
    """
    if not batches:
        return [[] for _ in range(n_buckets)]
    import batcher._native as nat

    key_index = batches[0].schema.get_field_index(key_name)
    return nat.range_partition_batches(
        list(batches), key_index, list(boundaries), n_buckets, nulls_first, descending
    )
