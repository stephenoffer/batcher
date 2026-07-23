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

    Examples:
        .. doctest::

            >>> from batcher.io import ParquetSink, Sink
            >>> isinstance(ParquetSink(), Sink)
            True
    """

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Write the whole table to a single file at `path`, atomically.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import ParquetSink
                >>> table = pa.table({"x": [1, 2, 3]})
                >>> ParquetSink().write(table, "out.parquet").rows  # doctest: +SKIP
                3

        Args:
            table: The rows to persist.
            path: Destination file URI.

        Returns:
            The file that was written, with its row count and size.
        """
        ...

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Write `table` under directory `path` as one shard of a directory write.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import ParquetSink
                >>> table = pa.table({"c": ["a", "b"], "x": [1, 2]})
                >>> sink = ParquetSink()
                >>> len(sink.write_partitioned(table, "out", partition_by=["c"]))  # doctest: +SKIP
                2

        Args:
            table: The rows to persist.
            path: Destination directory URI.
            partition_by: Columns to encode as Hive ``col=value`` directories.
                They are dropped from the data, since the path carries them.
            file_index: This shard's index, which names its part files so
                concurrent writers never collide.

        Returns:
            One entry per file written by this shard.
        """
        ...

    def commit(self, manifest: WriteManifest, path: str) -> None:
        """Finalize a write from the manifest every shard contributed to.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSink, WriteManifest
                >>> ParquetSink().commit(WriteManifest(), "out")  # a no-op for file sinks

        Args:
            manifest: Every file the write produced, merged across shards.
            path: The write's destination root.
        """
        ...
