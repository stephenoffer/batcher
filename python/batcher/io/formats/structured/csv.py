"""CSV format — lazy read + write via pyarrow, with byte-range splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Any

import pyarrow as pa

from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import FileSplit, Split, read_aligned_range

__all__ = ["CSVRangeSplit", "CSVSink", "CSVSource"]


@dataclass(frozen=True, slots=True)
class CSVRangeSplit:
    """A newline-aligned byte range of a CSV file, parsed with the file's header.

    The header line is prepended to every non-leading range so pyarrow parses each
    range with the correct column names; ranges cover each data row exactly once.
    """

    path: str
    start: int
    end: int

    def _header(self) -> bytes:
        fs = resolve_filesystem(self.path)
        with fs.open(self.path) as fh:
            return fh.readline()

    def _table(self, projection: list[str] | None) -> pa.Table:
        import io

        import pyarrow.csv as pacsv

        data = read_aligned_range(self.path, self.start, self.end)
        if self.start != 0:
            data = self._header() + data  # supply column names to a mid-file range
        if not data.strip():
            from batcher.io.formats.base import SOURCES

            empty = SOURCES.get("csv")(self.path).schema().empty_table()
            return empty.select(projection) if projection is not None else empty
        table = pacsv.read_csv(io.BytesIO(data))
        return table.select(projection) if projection is not None else table

    def schema(self) -> pa.Schema:
        from batcher.io.formats.base import SOURCES

        return SOURCES.get("csv")(self.path).schema()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._table(projection).to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection).to_batches()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"csv:{self.path}:{self.start}-{self.end}"


@SOURCES.register("csv")
class CSVSource(FileSource):
    """One or more CSV files (single file, directory, or glob).

    Large files are split into newline-aligned byte ranges (`CSVRangeSplit`) so a
    single multi-GB CSV reads in parallel across workers; small files use one split
    each. Schema is inferred by pyarrow on first access.
    """

    suffix = ".csv"
    format_name = "csv"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        import pyarrow.csv as pacsv

        return pacsv.read_csv(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        import pyarrow.csv as pacsv

        table = pacsv.read_csv(fh)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _file_splits(self, path: str, target_size: int | None) -> list[Split]:
        # Default byte-range split size (so one huge file fans across workers instead
        # of reading on a single node) is the configured `ExecutionConfig.split_bytes`.
        chunk = target_size or active_config().execution.split_bytes
        try:
            size = self._fs.size(path)
        except (OSError, ValueError):
            return [FileSplit(self.format_name, path)]
        if size <= chunk:
            return [FileSplit(self.format_name, path)]
        return [
            CSVRangeSplit(path, start, min(start + chunk, size)) for start in range(0, size, chunk)
        ]


@SINKS.register("csv")
class CSVSink(FileSink):
    """Write a CSV file."""

    suffix = ".csv"
    format_name = "csv"

    __slots__ = ()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        import pyarrow.csv as pacsv

        pacsv.write_csv(table, fh)

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        import pyarrow.csv as pacsv

        return pacsv.CSVWriter(fh, schema)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write(batch)

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
