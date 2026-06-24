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
    """
    if predicate is not None and getattr(source, "supports_predicate", False):
        return source.read(projection, predicate=predicate)  # type: ignore[call-arg]
    return source.read(projection)


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


class InMemorySource:
    """A relation already materialized as Arrow record batches."""

    __slots__ = ("_batches", "_schema")
    bounded = True

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        if not batches:
            raise ValueError("InMemorySource requires at least one record batch")
        self._batches = batches
        self._schema = batches[0].schema

    def schema(self) -> pa.Schema:
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        if projection is None:
            return self._batches
        return [b.select(projection) for b in self._batches]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for b in self._batches:
            yield b.select(projection) if projection is not None else b

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
