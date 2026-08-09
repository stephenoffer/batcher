"""Distributed streaming terminals — pull a distributed result back in bounded memory.

The single-node streaming terminal (`stream._iter_batches`) yields a plan's result one
batch at a time. These are its distributed analogues, so `iter_batches(distributed=)`
never funnels the whole distributed result through the driver:

- `iter_distributed` runs a top-level *breaker* with ``materialize=False`` so each
  reducer keeps its bucket partitioned (on disk or the Flight fleet), then yields one
  bucket at a time — peak driver memory is a single reducer's output.
- a breaker-free *scan/filter/project* over a splittable source fans the read out across
  workers and streams each worker's output back one partition at a time
  (`stream_distributed_map`); `distributable_scan_source` decides when that applies.

Split out of `stream` so that file stays within its size budget; `stream` imports these.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["distributable_scan_source", "iter_distributed", "iter_distributed_scan"]


def distributable_scan_source(plan: LogicalPlan, sources: list[Source]) -> int | None:
    """The source id if `plan` is a breaker-free single-source scan/filter/project over a
    SPLITTABLE source (worth fanning out across workers), else ``None``.

    `map_batches` pipelines and in-memory/iterator sources are excluded — the former
    needs the actor-pool path, the latter would have to be shipped to the workers.
    """
    from batcher import core
    from batcher.dist.executor import _is_splittable_source
    from batcher.dist.executors.plan_analysis import _single_source
    from batcher.plan.visitor import scanned_source_ids

    if not _single_source(plan) or core.has_map_batches(plan):
        return None
    ids = scanned_source_ids(plan)
    if len(ids) != 1:
        return None
    sid = next(iter(ids))
    if sid >= len(sources) or not _is_splittable_source(sources[sid]):
        return None
    return sid


def _rechunk(b: pa.RecordBatch, batch_size: int | None) -> Iterator[pa.RecordBatch]:
    """Yield `b` whole, or in `batch_size`-row slices."""
    if batch_size is None:
        yield b
        return
    for off in range(0, b.num_rows, batch_size):
        yield b.slice(off, batch_size)


def iter_distributed(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    num_workers: int | None,
    transport: str,
    batch_size: int | None,
) -> Iterator[pa.RecordBatch]:
    """Stream a distributed breaker's result back to the driver one bucket at a time.

    The stage runs with ``materialize=False`` so each reducer keeps its bucket
    partitioned (on disk or on the Flight fleet) instead of every bucket being pulled
    into one driver table. The driver then reads one bucket at a time and yields it, so
    peak driver memory is a *single* reducer's output, not the whole result. Shapes that
    don't support ``materialize=False`` return a collected table, which is simply
    re-chunked. The partitioned intermediate is freed once iteration finishes.

    The whole generator runs inside one `query_shuffle_scope`, and that is load-bearing
    rather than tidy. Scope exit evicts the query's shuffle buckets, on the premise —
    true everywhere else — that leaving the scope means the query is over. It is false
    here: this terminal *deliberately* returns handles to buckets the driver has not read
    yet, so the scope `run_relational` opens internally would close, evict them, and leave
    the reads to find nothing. They would not raise. An unregistered ticket reads back as
    an **empty bucket, not an error** (the epoch invariant in `dist/shuffle_replication`),
    so `iter_batches(distributed=True)` silently returned **zero rows** under the Flight
    transport on a multi-node cluster, while `collect()` on the same plan returned the
    right answer — the exact wrong-answer bug `dist/fleet/eviction.py`'s own docstring
    says premature eviction would cause. Holding an enclosing scope makes the inner one
    reentrant (it detects the ambient plan id and neither re-mints nor evicts), so the
    buckets are freed when *iteration* finishes, which is when this query is actually over.
    """
    from batcher import core
    from batcher.api.orchestration import run_relational
    from batcher.dist.fleet.plan_id import query_shuffle_scope

    ctx = core.ExecutionContext(
        columns=columns,
        hub=core.default_hub(),
        num_workers=num_workers,
        transport=transport,
    )
    with query_shuffle_scope():
        result, _ = run_relational(plan, sources, ctx, distributed=True, materialize=False)
        if isinstance(result, pa.Table):
            batches = (
                result.to_batches()
                if batch_size is None
                else result.to_batches(max_chunksize=batch_size)
            )
            yield from batches
            return
        try:
            for b in result.iter_batches():
                yield from _rechunk(b, batch_size)
        finally:
            if callable(getattr(result, "cleanup", None)):
                result.cleanup()


def iter_distributed_scan(
    plan: LogicalPlan, sources: list[Source], num_workers: int | None, batch_size: int | None
) -> Iterator[pa.RecordBatch]:
    """Fan a breaker-free splittable scan out across workers and stream each worker's
    output back one partition at a time — parallel reads with the driver holding only one
    partition's result. The caller guarantees the shape (`distributable_scan_source`)."""

    from batcher.dist.executors.map import stream_distributed_map

    workers = num_workers or available_cpu_count()
    for b in stream_distributed_map(plan, sources, workers):
        yield from _rechunk(b, batch_size)
