"""`ParquetSink` — the Parquet writer.

Thin: `FileSink` owns atomicity, Hive partitioning, sharding, and the manifest; this
adds the format's own encode and its incremental (row-group-at-a-time) stream writer.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher.io.base import FileSink
from batcher.io.formats.base import SINKS

__all__ = ["ParquetSink"]


@SINKS.register("parquet")
class ParquetSink(FileSink):
    """Write a Parquet file, zstd-compressed by default.

    Supports incremental writes: `write_stream` appends one row-group per batch, so a
    `read → transform → write` pipeline never materializes its result.

    Examples:
        .. doctest::

            >>> import pyarrow as pa  # doctest: +SKIP
            >>> from batcher.io import ParquetSink  # doctest: +SKIP
            >>> sink = ParquetSink(compression="snappy")  # doctest: +SKIP
            >>> sink.write(pa.table({"x": [1]}), "o.parquet").rows  # doctest: +SKIP
            1
    """

    format_name = "parquet"

    __slots__ = ("_file_token", "compression")

    def __init__(self, compression: str = "zstd", file_token: str | None = None) -> None:
        self.compression = compression
        self._file_token = file_token

    @property
    def suffix(self) -> str:
        """``.parquet``, or ``-{file_token}.parquet`` when this write must not reuse a name.

        `FileSink` names shards ``part-{index}.parquet``, which is deterministic by design —
        a retried shard overwrites itself, so a resumed write is idempotent. But a write that
        must **leave existing files alone** cannot use it: the name collides with a file the
        write never read, and overwriting it destroys data.

        Two writes need exactly that. A copy-on-write ``MERGE`` rewrites only the data files
        its key-pruning proved could match, and must not clobber the ones it skipped. A Delta
        append writes files the transaction log references forever, so reusing a name
        silently rewrites history and breaks time travel. A per-write `file_token` makes
        every file distinct while keeping the Hive layout and the `file_index` that separates
        concurrent shards.

        It is a **constructor option**, not a sink subclass, because a distributed write
        rebuilds its sink on each worker from `sink_kwargs` — a token carried on the object
        would be lost in transit, and every shard would collide.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSink
                >>> ParquetSink().suffix
                '.parquet'
                >>> ParquetSink(file_token="a1b2").suffix
                '-a1b2.parquet'

        Returns:
            The file extension this write appends to each shard's name.
        """
        return f"-{self._file_token}.parquet" if self._file_token else ".parquet"

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        import pyarrow.parquet as pq

        pq.write_table(table, fh, compression=self.compression)

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        import pyarrow.parquet as pq

        return pq.ParquetWriter(fh, schema, compression=self.compression)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write_batch(batch)

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
