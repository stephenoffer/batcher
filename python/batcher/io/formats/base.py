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
        """The number of rows, if known cheaply without reading data (else None)."""
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


# Registries of format readers/writers, keyed by format name ("parquet"/"csv"/…).
# Concrete classes register into these from each `io/formats/<fmt>.py` module.
SOURCES: Registry[type] = Registry("source")
SINKS: Registry[type] = Registry("sink")
