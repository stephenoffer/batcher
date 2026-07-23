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

# Rows to accumulate before flushing a row group. A row group is the unit of parallelism,
# of column statistics, and of dictionary/compression state, so its size is a real
# trade-off — but one row group *per incoming batch* is far off the mark in the small
# direction. A streaming write hands the sink one morsel at a time, which produced 4,096-row
# row groups: TPC-H sf1 `lineitem` came out as **1,459** row groups against DuckDB's 49, and
# the file was *larger* (218.9 MB vs 207.1 MB) despite ZSTD against DuckDB's Snappy, because
# dictionaries and compression state reset at every boundary and 1,459 footers' worth of
# statistics is not free to write or to parse. 128Ki rows is the conventional target (DuckDB
# writes ~122.9K) and bounds the buffer at one row group of live rows.
_ROW_GROUP_ROWS = 128 * 1024


class _BufferedRowGroupWriter:
    """Batches a stream of morsels into properly-sized Parquet row groups.

    `pyarrow.ParquetWriter.write_batch` starts a new row group per call, so writing a
    morsel at a time yields a row group per morsel. This accumulates them and flushes once
    the buffer reaches `_ROW_GROUP_ROWS`, keeping the write streaming (never more than one
    row group of rows is held) while producing a normal file layout.
    """

    __slots__ = ("_buf", "_rows", "_schema", "_writer")

    def __init__(self, writer: Any, schema: pa.Schema) -> None:
        self._writer = writer
        self._schema = schema
        self._buf: list[pa.RecordBatch] = []
        self._rows = 0

    def add(self, batch: pa.RecordBatch) -> None:
        """Buffer one batch, flushing a row group once the target is reached."""
        if batch.num_rows == 0:
            return
        self._buf.append(batch)
        self._rows += batch.num_rows
        if self._rows >= _ROW_GROUP_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        self._writer.write_table(pa.Table.from_batches(self._buf, schema=self._schema))
        self._buf.clear()
        self._rows = 0

    def close(self) -> None:
        """Flush the trailing partial row group, then close the underlying writer."""
        self._flush()
        self._writer.close()


@SINKS.register("parquet")
class ParquetSink(FileSink):
    """Write a Parquet file, zstd-compressed by default.

    Supports incremental writes: `write_stream` accumulates batches into normal-sized row
    groups and flushes them as it goes, so a `read → transform → write` pipeline never
    materializes its result and still produces a conventional file layout.

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

    def __init__(
        self, compression: str = "zstd", file_token: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)  # carries filesystem= / storage_options=
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

        # `row_group_size` matters in BOTH directions. Left to pyarrow's default a large
        # in-memory table becomes ONE row group, which defeats row-group pruning and
        # per-row-group read parallelism entirely; the streaming path had the opposite
        # problem (one group per morsel). Both converge on the same target.
        pq.write_table(
            table,
            fh,
            compression=self.compression,
            write_page_index=True,
            row_group_size=_ROW_GROUP_ROWS,
        )

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        import pyarrow.parquet as pq

        # `write_page_index` emits the ColumnIndex/OffsetIndex — per-*page* min/max plus the
        # row offsets to seek by. Without it a reader can only skip whole row groups, so a
        # highly selective predicate still decodes one entirely: on TPC-H sf1 `lineitem`,
        # `l_orderkey < 100` matches 105 rows but decodes 122,880 (a 1,170x amplification).
        # Batcher's native reader now consumes this (`bc-io::page_index`): a pushed predicate
        # becomes a `RowSelection` over the surviving pages, measured at 61x less data decoded
        # and 17.5x faster on a 2M-row group. Spark/DuckDB/Databricks readers use it too.
        # Cost is a few KB of footer per file.
        writer = pq.ParquetWriter(fh, schema, compression=self.compression, write_page_index=True)
        # Wrapped so morsels accumulate into normal-sized row groups instead of one row
        # group per morsel — see `_BufferedRowGroupWriter`.
        return _BufferedRowGroupWriter(writer, schema)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.add(batch)

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
