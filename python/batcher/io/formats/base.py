"""Format contracts and registries — the seam new IO formats plug into.

A *format* is a named (``"parquet"`` / ``"csv"`` / ``"json"`` / …) pairing of a
read path and a write path over Arrow. `SourceFormat` and `SinkFormat` are the
minimal `Protocol`s a format implements; each `io/formats/<fmt>.py` module
registers its concrete source/sink classes into the `SOURCES` / `SINKS`
registries here. Adding a format (Iceberg, Delta, Lance, Kafka, …) is one new
file that imports these registries and registers — no edit to `source.py` /
`sink.py`.

Concrete file formats subclass the Template-Method bases in `io.base`
(`FileSource` / `FileSink`), which supply the shared path/filesystem/schema/split
machinery; these protocols describe what the resulting classes expose.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

from batcher._internal.registry import Registry

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["SINKS", "SOURCES", "SinkFormat", "SourceFormat"]


@runtime_checkable
class SourceFormat(Protocol):
    """A lazily-readable relation backed by one format's files.

    Constructed with a path (single file, directory, or glob). Reads are lazy:
    `schema` is known without reading the data, and `read` / `iter_batches`
    honor an optional column `projection` for projection pushdown. `splits`
    advertises independently-readable slices for distributed reads.
    """

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
        """The **exact** number of rows, if known cheaply without reading data (else None).

        Exact is a requirement, not a hope. When the conductor has collected no
        `SourceStatistics` for a source, the estimator falls back to this and tags the
        result `Provenance.EXACT` — which is the provenance that lets a terminal be answered
        from metadata *without executing*. A source that returns an estimate here makes
        `count()` return a number the data does not support, silently.

        An estimate belongs in `SourceStatistics(row_count=..., exact_rows=False)`, which
        exists for exactly this distinction: it informs cost and cardinality and can never
        answer a terminal. A catalog figure such as Postgres `reltuples` or Mongo
        `estimatedDocumentCount` goes there. Return `None` here instead.
        """
        ...

    def identity(self) -> str:
        """A stable identifier for this source (for keyed metadata/learning)."""
        ...

    def splits(self, target_size: int | None = None) -> list[Split]:
        """Independently-readable slices for distributed/parallel reads."""
        ...


@runtime_checkable
class SinkFormat(Protocol):
    """A writer that persists a whole Arrow table to a path."""

    def write(self, table: pa.Table, path: str) -> None:
        """Write `table` to `path` in this format."""
        ...


# Format families whose modules are imported the first time a format is looked up rather
# than when `batcher.io.formats` is imported. Between them they reach every SQL warehouse,
# NoSQL store, message broker, lakehouse table format, and ML shard layout Batcher can
# read — the widest part of the IO surface and the least used, since a given process
# names at most one or two of them. Importing them all up front was the single largest
# component of `import batcher`, and it dragged `ssl`, `http`, `email`, and `asyncio` in
# behind it for a process that may only ever read a Parquet file.
#
# The file formats a pipeline reaches for by default — Parquet, CSV, JSON, text, binary,
# the multimodal blob readers, the robotics logs — stay eager in `formats/__init__`: they
# are cheap, they are re-exported by name, and deferring them would only move the cost.
_DEFERRED_FAMILIES = ("lakehouse", "ml", "nosql", "sql", "streaming")


def _load_deferred_families() -> None:
    """Import the deferred format families, so their modules register themselves.

    Runs at most once, driven by `Registry.complete` from the first lookup that a
    partially-populated registry could answer wrongly. Self-registration is untouched: a
    new connector is still one new module in its category package that registers on
    import, exactly as `add-an-io-format-or-connector` describes.
    """
    import importlib

    for family in _DEFERRED_FAMILIES:
        importlib.import_module(f"batcher.io.formats.{family}")


# Registries of format readers/writers, keyed by format name ("parquet"/"csv"/…).
# Concrete classes register into these from each `io/formats/<fmt>.py` module.
SOURCES: Registry[type] = Registry("source", on_miss=_load_deferred_families)
SINKS: Registry[type] = Registry("sink", on_miss=_load_deferred_families)
