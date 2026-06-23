"""Data sinks — persisting query results.

Sinks write an Arrow table to storage. Kept behind a small protocol + registry so
new formats (and partitioned / streaming writers) slot in uniformly. The
per-format writers live one-per-file under `io/formats/` and register into the
`SINKS` registry; this module re-exports them and owns the `Sink` protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pyarrow as pa

from batcher.io.formats import SINKS, CSVSink, JSONSink, ParquetSink
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = ["SINKS", "CSVSink", "JSONSink", "ParquetSink", "Sink"]


@runtime_checkable
class Sink(Protocol):
    """A writer that persists Arrow tables to storage.

    `write` produces a single file; `write_partitioned` writes one shard of a
    (possibly Hive-partitioned) directory write; `commit` finalizes a write
    atomically from the collected manifest (a no-op for plain file sinks).
    """

    def write(self, table: pa.Table, path: str) -> WrittenFile: ...

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,
        file_index: int = 0,
    ) -> list[WrittenFile]: ...

    def commit(self, manifest: WriteManifest, path: str) -> None: ...
