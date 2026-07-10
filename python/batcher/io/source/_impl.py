"""Lazy data sources.

A `Source` knows its schema without reading data, and reads (optionally a column
subset) only when asked. The `projection` parameter is the hook the optimizer's
projection-pushdown pass uses: it sets which columns the scan must produce, so a
columnar source (Parquet) reads only those.

File-format sources (Parquet/CSV/JSON/…) live one-per-file under `io/formats/`
and register into the `SOURCES` registry; this module re-exports them and owns
the non-file sources (in-memory, streaming-iterator) plus the `Source` protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable

import pyarrow as pa

from batcher.io.formats import SOURCES, CSVSource, JSONSource, ParquetSource
from batcher.io.splits import IpcFileSplit, Split, WholeSourceSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = [
    "SOURCES",
    "CSVSource",
    "Checkpointable",
    "InMemorySource",
    "IteratorSource",
    "JSONSource",
    "MaterializedSource",
    "ParquetSource",
    "Source",
    "Split",
    "is_bounded",
    "is_checkpointable",
    "iter_source",
    "read_source",
    "source_statistics",
]


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


@runtime_checkable
class Source(Protocol):
    """A lazily-readable relation.

    `bounded` (default ``True``) marks whether the source is finite. Unbounded
    sources (Kafka and other brokers, incremental file discovery) set it ``False``
    so terminal operations choose a streaming path and `collect()` refuses to
    materialize an infinite stream. Read it via `is_bounded` to honor the default.
    """

    bounded: bool

    def schema(self) -> pa.Schema:
        """The full schema of the source, without reading the data."""
        ...

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read the source, optionally only `projection` columns."""
        ...

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Yield record batches lazily (the streaming read path)."""
        ...

    def row_count(self) -> int | None:
        """The number of rows, if known cheaply without reading data (else None)."""
        ...

    def identity(self) -> str:
        """A stable identifier for this source (for keyed metadata/learning)."""
        ...

    # Optional (duck-typed via `source_statistics`): a connector may also expose
    #   def statistics(self) -> SourceStatistics | None
    # returning footer/manifest/catalog row counts and per-column min/max/null/ndv
    # known without scanning. Sources that don't implement it fall back to
    # `row_count()`. Not a required Protocol method so `runtime_checkable` still
    # accepts the many sources that predate it.

    def splits(self, target_size: int | None = None) -> list[Split]:
        """Independently-readable slices for distributed/parallel reads.

        A source that cannot subdivide returns a single `WholeSourceSplit`.
        """
        ...

    # Optional (duck-typed via `is_checkpointable`): a *replayable* streaming source
    # may also expose
    #   def snapshot_position(self) -> dict        # what it has read through
    #   def seek(self, position: dict) -> None     # resume from a recorded position
    # so a streaming query can checkpoint offsets and resume exactly-once after a
    # restart (Kafka offsets, Kinesis sequence numbers, a rate cursor). Not required
    # Protocol methods, so non-replayable sources are simply at-least-once.


@runtime_checkable
class Checkpointable(Protocol):
    """A streaming source whose read position can be snapshotted and resumed."""

    def snapshot_position(self) -> dict: ...

    def seek(self, position: dict) -> None: ...


def is_checkpointable(source: Source) -> bool:
    """Whether `source` supports offset snapshot/seek for exactly-once recovery."""
    return callable(getattr(source, "snapshot_position", None)) and callable(
        getattr(source, "seek", None)
    )


# Overflow-safe narrow → wide numeric widenings the engine normalizes to at the FFI
# boundary anyway (Int8/16/32, UInt8/16/32 → Int64; Float16/32 → Float64). Doing it once
# here — instead of the Rust boundary re-casting on *every* query — turns a per-query
# O(rows) `cast` (arrow's int16→int64 cast is ~47 ms / 10 M rows) into a one-time cost.
# UInt64 (can overflow Int64), dictionaries, and everything else are left to the Rust
# `normalize_batch`, which stays the correctness backstop (a no-op on already-wide cols).
_WIDEN_NARROW: dict[pa.DataType, pa.DataType] = {
    pa.int8(): pa.int64(),
    pa.int16(): pa.int64(),
    pa.int32(): pa.int64(),
    pa.uint8(): pa.int64(),
    pa.uint16(): pa.int64(),
    pa.uint32(): pa.int64(),
    pa.float16(): pa.float64(),
    pa.float32(): pa.float64(),
}


def _widen_schema(schema: pa.Schema, targets: dict[str, pa.DataType]) -> pa.Schema:
    """`schema` with each `targets` column retyped to its widened type (metadata only)."""
    return pa.schema([f.with_type(targets[f.name]) if f.name in targets else f for f in schema])


class InMemorySource:
    """A relation already materialized as Arrow record batches.

    Narrow numeric columns (Int8/16/32, UInt8/16/32, Float16/32) are widened to
    Int64/Float64 — the types the engine normalizes to at the FFI boundary anyway — but
    **lazily and per column, with caching**: only the columns a query actually reads are
    cast, and the cast happens once (arrow's int16→int64 cast is ~47 ms / 10 M rows, and
    the Rust boundary would otherwise redo it on *every* query). This is skipped when
    ``shrink_output_dtypes`` is on — that opt-in path re-narrows pass-through outputs from
    the source widths, which pre-widening would erase — so that path keeps the Rust
    boundary as the (correctness-equivalent) fallback.
    """

    __slots__ = (
        "_batches",
        "_cache",
        "_mean_cache",
        "_ndv_cache",
        "_schema",
        "_stats",
        "_sum_cache",
        "_targets",
        "_valuecount_cache",
    )
    bounded = True
    # The batches are already in RAM, so a statistics pass over them costs no I/O. The
    # conductor reads this before sketching cold-start distinct counts on the query path
    # (`api.terminal._metadata.seed_column_ndv`); a file-backed source leaves it False and
    # learns its ndv from the batches a run already scanned, rather than re-reading them.
    resident = True
    # `identity()` is shape-based for in-memory data (see `identity`), so two *different*
    # in-memory relations that share a schema+size collide on it. That is fine for the
    # optimizer's shape keys, but NOT for the cross-query source-stats cache, whose entries
    # (row count, column min/max) depend on the actual data. So in-memory stats are never
    # stored under the shared identity — they are memoized per instance in `statistics()`.
    stable_stats_identity = False

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        if not batches:
            raise ValueError("InMemorySource requires at least one record batch")
        from batcher.config import active_config

        self._batches = batches
        src_schema = batches[0].schema
        if active_config().execution.shrink_output_dtypes:
            self._targets: dict[str, pa.DataType] = {}
            self._schema = src_schema
        else:
            self._targets = {f.name: t for f in src_schema if (t := _WIDEN_NARROW.get(f.type))}
            self._schema = _widen_schema(src_schema, self._targets) if self._targets else src_schema
        self._cache: dict[tuple[int, str], pa.Array] = {}
        self._stats: object | None = None
        self._ndv_cache: dict[str, int | None] = {}
        self._mean_cache: dict[str, float | None] = {}
        self._sum_cache: dict[str, float | int | None] = {}
        self._valuecount_cache: dict[tuple[str, str, object], int | None] = {}

    def schema(self) -> pa.Schema:
        return self._schema

    def _build_column(self, name: str) -> pa.ChunkedArray | pa.Array:
        """Column `name` across all batches as one (narrow-int widened) Arrow column."""
        chunks = [self._widened(bi, name, b.column(name)) for bi, b in enumerate(self._batches)]
        return pa.chunked_array(chunks) if len(chunks) > 1 else chunks[0]

    def column_ndv(self, name: str) -> int | None:
        """EXACT distinct count of `name`'s non-null values, computed once and cached."""
        if name not in self._ndv_cache:
            from batcher.io.source import inmemory_stats

            self._ndv_cache[name] = inmemory_stats.column_ndv(self._build_column, name)
        return self._ndv_cache[name]

    def column_mean(self, name: str) -> float | None:
        """EXACT average of `name`'s non-null values, computed once and cached."""
        if name not in self._mean_cache:
            from batcher.io.source import inmemory_stats

            self._mean_cache[name] = inmemory_stats.column_mean(self._build_column, name)
        return self._mean_cache[name]

    def column_sum(self, name: str) -> float | int | None:
        """EXACT total of `name`'s non-null values (exactly representable), else None."""
        if name not in self._sum_cache:
            from batcher.io.source import inmemory_stats

            self._sum_cache[name] = inmemory_stats.column_sum(self._build_column, name)
        return self._sum_cache[name]

    def column_predicate_count(self, op: str, name: str, value: object) -> int | None:
        """EXACT surviving count of ``name <op> value`` (nulls excluded), cached per key.

        `op` ∈ eq/ne/lt/le/gt/ge. Lets a repeat ``COUNT(*) WHERE col <op> v`` be answered
        from metadata — the learned-metadata moat for any single-column comparison filter."""
        key = (op, name, value)
        if key not in self._valuecount_cache:
            from batcher.io.source import inmemory_stats

            self._valuecount_cache[key] = inmemory_stats.column_predicate_count(
                self._build_column, op, name, value
            )
        return self._valuecount_cache[key]

    def statistics(self):
        """EXACT per-column min/max/null-count over the ordered columns, computed once.

        An in-memory relation is immutable, so its column bounds are exact and constant; the
        one vectorized pass (memoized per instance) lets Kyber answer an unfiltered
        ``MIN``/``MAX`` from metadata on every subsequent run — the learned-metadata moat a
        static optimizer can't match. See `io.inmemory_stats` for the computation.
        """
        if self._stats is None:
            from batcher.io.source import inmemory_stats

            rows = sum(b.num_rows for b in self._batches)
            self._stats = inmemory_stats.statistics(self._build_column, self._schema, rows)
        return self._stats

    def _widened(self, bi: int, name: str, col: pa.Array) -> pa.Array:
        """The widened (and cached) form of column `name` in batch `bi`, or `col` as-is."""
        target = self._targets.get(name)
        if target is None:
            return col
        key = (bi, name)
        arr = self._cache.get(key)
        if arr is None:
            import pyarrow.compute as pc

            arr = pc.cast(col, target)
            self._cache[key] = arr
        return arr

    def _project(self, bi: int, b: pa.RecordBatch, projection: list[str] | None) -> pa.RecordBatch:
        if projection is None and not self._targets:
            return b
        names = projection if projection is not None else b.schema.names
        selected = b.select(names)
        if not self._targets:
            return selected
        arrays = [self._widened(bi, n, selected.column(j)) for j, n in enumerate(names)]
        return pa.RecordBatch.from_arrays(
            arrays, schema=_widen_schema(selected.schema, self._targets)
        )

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return [self._project(i, b, projection) for i, b in enumerate(self._batches)]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for i, b in enumerate(self._batches):
            yield self._project(i, b, projection)

    def row_count(self) -> int | None:
        return sum(b.num_rows for b in self._batches)

    def identity(self) -> str:
        # In-memory data has no stable cross-run identity; key by schema + size.
        return f"mem:{self._schema}:{self.row_count()}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]


class IteratorSource:
    """A streaming relation backed by a re-iterable factory of record batches.

    `factory` is a zero-argument callable returning a *fresh* iterator of
    `pyarrow.RecordBatch` each time it is called (so the source can be read more
    than once, e.g. plan-build validation then execution). The schema must be
    supplied up front since the data is not materialized. This is the entry point
    for unbounded / larger-than-memory streaming inputs.

    `bounded` defaults to ``True`` (a finite generator); pass ``bounded=False`` for
    a genuinely unbounded stream so `collect()` refuses to materialize it.
    """

    __slots__ = ("_bounded", "_factory", "_schema")

    def __init__(
        self,
        factory: Callable[[], Iterator[pa.RecordBatch]],
        schema: pa.Schema,
        *,
        bounded: bool = True,
    ) -> None:
        self._factory = factory
        self._schema = schema
        self._bounded = bounded

    @property
    def bounded(self) -> bool:
        return self._bounded

    def schema(self) -> pa.Schema:
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for b in self._factory():
            yield b.select(projection) if projection is not None else b

    def row_count(self) -> int | None:
        return None  # streaming sources have unknown (possibly unbounded) length.

    def identity(self) -> str:
        return f"stream:{self._schema}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]


class MaterializedSource:
    """A relation whose batches live on disk as Arrow IPC files (one per producer).

    Produced by a distributed stage that kept its result *partitioned* instead of
    collecting it to the driver: the adaptive executor scans it in place for the next
    stage (shared-nothing, via `IpcFileSplit`s), and its exact `row_count` feeds the
    optimizer's build-side/broadcast choices (provenance ``EXACT`` via the
    `row_count` fallback). `cleanup()` removes the backing files once the query no
    longer needs the intermediate.
    """

    __slots__ = ("_files", "_schema", "_work_dir")
    bounded = True

    def __init__(
        self,
        files: list[tuple[str, int]],
        schema: pa.Schema,
        work_dir: str | None = None,
    ) -> None:
        self._files = files  # (ipc_path, exact_row_count) per producer partition
        self._schema = schema
        self._work_dir = work_dir

    def schema(self) -> pa.Schema:
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        out: list[pa.RecordBatch] = []
        for path, _rows in self._files:
            out.extend(IpcFileSplit(path).read(projection))
        return out

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for path, _rows in self._files:
            yield from IpcFileSplit(path).iter_batches(projection)

    def row_count(self) -> int | None:
        return sum(rows for _path, rows in self._files)

    def identity(self) -> str:
        return f"materialized:{self._schema}:{self.row_count()}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [IpcFileSplit(path, rows) for path, rows in self._files]

    def cleanup(self) -> None:
        """Delete the backing IPC files' work directory (best-effort)."""
        if self._work_dir:
            import shutil

            shutil.rmtree(self._work_dir, ignore_errors=True)
