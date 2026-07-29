"""Windowed streaming helpers for `map_batches` (UDF) pipelines.

A `map_batches` UDF parallelizes across `num_workers` only when handed several batches
at once, so the streaming paths drive it in *windows* — enough rows per round to fill
the worker pool, small enough to keep driver memory bounded. These helpers are shared by
the streaming iterator (`stream.py`) and the streaming write (`core.py`); they carry no
execution state, so both layers import them without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pyarrow as pa

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["max_map_workers", "peek_stream", "stream_map_aggregate", "stream_windowed"]


def stream_map_aggregate(
    plan: LogicalPlan,
    mapped: Iterator[pa.RecordBatch],
    batch_size: int | None,
) -> Iterator[pa.RecordBatch]:
    """Fold an already-mapped stream into one running aggregate state, then yield it.

    `plan` is a top-level `Aggregate` over a `map_batches` input; `mapped` streams that
    map's output (windowed + parallel, bounded memory). Each mapped batch is partial-
    aggregated and combined into a single running `_AggFold` state, so a large `map→agg`
    never materializes the whole mapped output — the streaming (out-of-core) counterpart
    of the materializing collect path, with the same mergeable result.
    """
    import dataclasses

    from batcher.core.streaming import _AggFold, _rebatch
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    first = next(mapped, None)
    if first is None:
        return
    # The map is already applied, so the fold's input is a plain scan of the mapped schema
    # (identity) — it only partial-aggregates + combines, never re-runs the UDF.
    ident = dataclasses.replace(plan, input=Scan(0, SchemaRef.from_arrow(first.schema)))
    fold = _AggFold(ident)
    fold.push(first)
    for batch in mapped:
        fold.push(batch)
    result = fold.finalize()
    if result is not None:
        yield from _rebatch(result, batch_size)


def max_map_workers(plan: LogicalPlan) -> int:
    """The largest `num_workers` across the plan's `map_batches` stages (>=1).

    Sizes the streaming window so a `map_batches` UDF is handed enough batches per
    round to fill its worker pool — the difference between serial and all-cores.
    """
    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import children

    best = 1
    if isinstance(plan, MapBatches):
        best = max(best, int(plan.num_workers or 1))
    # Children come from `plan.visitor`, which caches each node class's child-bearing
    # fields — rather than re-deriving them per node with `dataclasses.fields`.
    for child in children(plan):
        best = max(best, max_map_workers(child))
    return best


def stream_windowed(
    source: Source,
    run_window: Callable[[list[pa.RecordBatch]], list[pa.RecordBatch]],
    target_rows: int,
    batch_size: int | None,
) -> Iterator[pa.RecordBatch]:
    """Stream `source` through `run_window` in windows of ~`target_rows` rows.

    Accumulates source batches until the window holds ~`target_rows` rows, applies the
    (parallel) UDF pipeline to the whole window at once, and yields the results — so the
    per-batch calls fan across the worker pool (the pipeline re-morselizes the window
    internally) while driver memory stays bounded to one window (+ its output), never the
    whole input. Sizing by rows (not batch count) keeps the bound stable whether the
    source yields many small batches or a few large row-groups.
    """
    from batcher.io.source import iter_source

    def flush(buf):
        out = [b for b in run_window(buf) if b.num_rows]
        if not out:
            return
        if batch_size is None:
            # Coalesce the window's per-morsel outputs into one batch so a downstream
            # sink writes a few large row-groups, not thousands of tiny ones (an
            # incremental Parquet write over morsel-sized batches is many times slower
            # than over window-sized ones). One window's rows are already resident, so
            # this adds no memory beyond the bound the window already sets.
            combined = pa.Table.from_batches(out).combine_chunks().to_batches()
            yield from (b for b in combined if b.num_rows)
        else:
            for b in out:
                for off in range(0, b.num_rows, batch_size):
                    yield b.slice(off, batch_size)

    buf: list[pa.RecordBatch] = []
    rows = 0
    for batch in iter_source(source, None, None):
        if batch.num_rows == 0:
            continue
        buf.append(batch)
        rows += batch.num_rows
        if rows >= target_rows:
            yield from flush(buf)
            buf = []
            rows = 0
    if buf:
        yield from flush(buf)


def peek_stream(
    batches: Iterator[pa.RecordBatch],
    empty_schema: Callable[[], pa.Schema],
) -> tuple[pa.Schema, Iterator[pa.RecordBatch]]:
    """Pull the first batch to learn the output schema, then re-chain it in.

    Reading the schema off the first streamed batch avoids a whole extra pass over an
    opaque (`map_batches`) pipeline just to type its output. `empty_schema` supplies the
    schema only when the stream is empty (so a sink can still write a valid empty file);
    it is called at most once, on the empty path, where the source is already exhausted.
    """
    import itertools

    try:
        first = next(batches)
    except StopIteration:
        return empty_schema(), iter(())
    return first.schema, itertools.chain([first], batches)
