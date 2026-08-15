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

from batcher._internal.logging import note_suppressed
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
    "watermark_partition_columns",
    "watermark_partitions",
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


def watermark_partition_columns(source: Source) -> tuple[str, ...]:
    """The columns of `source`'s batches that say which partition a row arrived from.

    An event-time watermark over a partitioned stream is the *minimum* over per-partition
    maxima, and computing that minimum needs each row attributed to a partition. A broker
    already carries the attribution in its batches (``topic`` and ``partition``), so this
    names those columns rather than inventing a side channel; a source that declares none
    is treated as a single partition, where a minimum and a maximum agree.

    Read via `getattr` so any duck-typed source is unpartitioned unless it opts in.

    Args:
        source: The source to ask.

    Returns:
        The partition-identifying column names, empty when the source has none.
    """
    return tuple(getattr(source, "watermark_partition_columns", ()) or ())


def watermark_partitions(source: Source) -> tuple[tuple[object, ...], ...]:
    """The partitions `source` expects to read from, before any of them has delivered.

    "Which partitions exist" and "which partitions have sent me something" differ exactly
    at startup, and that difference is a watermark bug: a minimum over the partitions seen
    so far is a minimum over partition 0 alone while partition 1 is still connecting, which
    over-claims event time and drops partition 1's first rows as late. A source that can
    enumerate its partitions up front closes that window.

    Best-effort by design. Discovery talks to a broker, and a broker that will not answer
    must not prevent the query from starting — the tracker then simply learns its partitions
    as they deliver, which is the behavior of a source that cannot enumerate at all.

    Args:
        source: The source to ask.

    Returns:
        One tuple of partition-column values per expected partition, empty when the source
        cannot enumerate them.
    """
    discover = getattr(source, "watermark_partitions", None)
    if discover is None:
        return ()
    try:
        return tuple(tuple(p) for p in discover())
    except Exception as exc:
        # An unreachable broker degrades this to "learn partitions as they arrive", which is
        # exactly the contract for a source that never had the method. Failing the query over
        # an optional startup optimization would be the worse answer — but the degradation is
        # the startup window this exists to close, so it is on the record rather than silent.
        note_suppressed("io", "enumerate the source's watermark partitions", exc)
        return ()


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
    limit: int | None = None,
    ordering: tuple[tuple[str, bool, bool], ...] | None = None,
) -> list[pa.RecordBatch]:
    """Read `source` with projection, passing a pushed `predicate` only to sources
    that declare ``supports_predicate``, and a row cap only to those declaring
    ``supports_limit``.

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

    Args:
        source: The relation to read.
        projection: Columns the scan must produce, or None for all of them.
        predicate: Kyber's pushed predicate, or None.
        limit: The most rows the plan needs from this source
            (`PhysicalPlan.source_limits`), or None for no cap. A ceiling and never a
            floor: a source free to return more is still correct, because the engine keeps
            its own `Limit`.
        ordering: The ordering `limit` is taken in (`PhysicalPlan.source_orderings`), for
            a top-N. Unlike `limit` this is not free to ignore *selectively*: a source
            that cannot apply the ordering must not apply the cap either, since the first
            n of an unordered read is not the first n of a sorted one.

    Returns:
        The source's batches, never an empty list.
    """
    # Both extras are opt-in per source and passed only when the source declares them, so
    # a connector that never heard of either keeps the plain two-argument `read`.
    extras: dict[str, object] = {}
    if predicate is not None and getattr(source, "supports_predicate", False):
        extras["predicate"] = predicate
    if limit is not None and getattr(source, "supports_limit", False):
        extras["limit"] = limit
    if ordering and getattr(source, "supports_ordering", False):
        extras["ordering"] = ordering
    batches = source.read(projection, **extras)  # type: ignore[call-arg]
    if batches:
        return batches
    schema = source.schema()
    if projection is not None:
        schema = pa.schema([schema.field(schema.get_field_index(c)) for c in projection])
    return [pa.RecordBatch.from_pylist([], schema=schema)]
