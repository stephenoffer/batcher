"""The neutral read helpers every executor calls a `Source` through.

A source is read via these functions rather than directly, so the duck-typed
extensions — predicate pushdown (`supports_predicate`), cheap statistics
(`statistics()`), and predicate-pruned split planning (`plan_splits`) — stay opt-in: a
connector that implements none of them is still read correctly, just without the
optimization. `read_source` also normalizes the empty case, so every scan hands the
engine at least one (schema-carrying) batch.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import pyarrow as pa

from batcher.io.source.base import Source
from batcher.io.splits import Split
from batcher.plan.source_stats import SourceStatistics

__all__ = [
    "continues_across_passes",
    "is_bounded",
    "iter_source",
    "plan_splits",
    "read_source",
    "source_statistics",
]


def plan_splits(
    source: Source,
    target_size: int | None = None,
    predicate: dict | None = None,
    projection: list[str] | None = None,
) -> list[Split]:
    """A source's splits, with the pushed `predicate` used to skip files when it can.

    A connector whose metadata can rule a whole data file out — a lakehouse table
    format, whose transaction log records each file's partition values and column
    bounds — declares ``splits(target_size=..., predicate=...)`` and returns only the
    files that can contain a matching row. That elimination happens at plan time, so a
    pruned file is never listed, never opened, and never becomes a worker task.

    Sources that take no `predicate` are called without one and return every split;
    the engine's own `Filter` re-checks the rows regardless, so an unpruned plan is
    always correct and merely reads more. This keeps pruning strictly opt-in, the same
    way `supports_predicate` does for row-level pushdown.

    Args:
        source: The source to plan a read over.
        target_size: Optional coalescing target in bytes.
        predicate: The predicate Kyber pushed to this scan, if any.
        projection: The columns Kyber pushed to this scan, if any. A *warehouse* has to be
            told which columns to read when its read is **created** — a BigQuery read session
            fixes `selected_fields`, a SQL query fixes its `SELECT` list — so a projection that
            only arrives at `Split.read` time is a client-side slice of data the server already
            sent. Sources that can only project after the fact simply do not declare it.

    Returns:
        The splits to read, pruned where the source could prune them.
    """
    kwargs: dict[str, object] = {"target_size": target_size}
    if predicate is not None and _accepts(source.splits, "predicate"):
        kwargs["predicate"] = predicate
    if projection is not None and _accepts(source.splits, "projection"):
        kwargs["projection"] = projection
    return source.splits(**kwargs)


def _accepts(splits_fn: object, name: str) -> bool:
    """Whether a source's `splits` takes a `name` keyword."""
    try:
        return name in inspect.signature(splits_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - a builtin/C callable
        return False


def source_statistics(source: Source) -> SourceStatistics | None:
    """Cheap statistics a connector declares without scanning data, or None.

    A source may implement `statistics() -> SourceStatistics | None` (footer /
    manifest / catalog metadata). For sources that don't, this falls back to
    wrapping `row_count()` so the exact-or-unknown row count still reaches the
    estimator. Duck-typed (like `supports_predicate`) so no connector is forced
    to implement the richer method. Best-effort: a failing probe yields None
    rather than breaking planning.
    """
    stats_fn = getattr(source, "statistics", None)
    if callable(stats_fn):
        try:
            result = stats_fn()
        except Exception:
            result = None
        if result is not None:
            return result
    try:
        rows = source.row_count()
    except Exception:
        return None
    return None if rows is None else SourceStatistics(row_count=rows)


def is_bounded(source: Source) -> bool:
    """Whether `source` is finite (a `collect()` would terminate).

    Sources are bounded by default; only unbounded streaming sources (brokers,
    incremental file discovery, an explicitly-unbounded `from_batches`) declare
    ``bounded = False``. Read via `getattr` so any duck-typed source is treated as
    bounded unless it opts out.
    """
    return getattr(source, "bounded", True)


def continues_across_passes(source: Source) -> bool:
    """Whether a fresh `iter_batches()` *continues* the stream rather than replaying it.

    An unbounded source can end an `iter_batches()` generator for two very different
    reasons, and a streaming driver must not confuse them:

    * **It continues.** The incremental file source ends a pass when the directory holds
      nothing new, and its durable seen-store means the next pass returns only what has
      arrived since. A broker is the same: its offsets carry forward. Asking again is how
      the stream keeps flowing, so a driver that stops here stops the query.
    * **It replays.** An `IteratorSource` built from a batch factory calls that factory
      again, from the beginning. Asking again yields the *same rows a second time*, so a
      driver that re-opens it duplicates the entire stream, forever.

    Nothing in the `Source` protocol distinguished the two, so this is declared: a source
    opts in with ``continues_across_passes = True``. Default False, because replaying is
    the answer that is merely wrong rather than catastrophic — a stream that stops early
    is visible in `is_active`; one that re-reads its input in a loop writes duplicates to
    the sink until someone notices.

    Args:
        source: The source to ask.

    Returns:
        True when re-opening the source continues the stream.
    """
    return getattr(source, "continues_across_passes", False)


def iter_source(
    source: Source,
    projection: list[str] | None = None,
    predicate: dict | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream `source` batch-by-batch, pushing `predicate` only to capable sources
    whose `iter_batches` accepts one.

    The streaming path's `Filter` re-checks every batch, so a source that ignores
    the predicate is still correct — this is the bounded-memory analogue of
    `read_source`. Sources whose `iter_batches` lacks a `predicate` parameter are
    called with projection only (no signature break).
    """
    if predicate is not None and getattr(source, "supports_predicate", False):
        from inspect import signature

        if "predicate" in signature(source.iter_batches).parameters:
            return source.iter_batches(projection, predicate=predicate)  # type: ignore[call-arg]
    return source.iter_batches(projection)


def read_source(
    source: Source,
    projection: list[str] | None = None,
    predicate: dict | None = None,
) -> list[pa.RecordBatch]:
    """Read `source` with projection, passing a pushed `predicate` only to sources
    that declare ``supports_predicate``.

    The engine retains its `Filter` operator regardless, so a source that ignores
    (or partially applies) the predicate still produces correct results — pushdown
    is a pure I/O optimization. Capable sources translate the predicate IR via
    `batcher.io.predicate` to their backend filter.

    Always returns at least one batch. A `RecordBatch` is the only carrier of a schema
    across the FFI boundary, and the engine's pipeline breakers (join, aggregate,
    distinct, sort) cannot name their output columns without it. Sources disagree on the
    empty case — an in-memory table yields one zero-row batch, a zero-row Parquet file
    yields none — so the boundary normalizes it here rather than asking every connector
    to remember. Reading nothing is routine: an incremental batch with no new rows, a
    table whose rows were all deleted, a partition pruned away entirely.
    """
    if predicate is not None and getattr(source, "supports_predicate", False):
        batches = source.read(projection, predicate=predicate)  # type: ignore[call-arg]
    else:
        batches = source.read(projection)
    if batches:
        return batches
    schema = source.schema()
    if projection is not None:
        schema = pa.schema([schema.field(schema.get_field_index(c)) for c in projection])
    return [pa.RecordBatch.from_pylist([], schema=schema)]
