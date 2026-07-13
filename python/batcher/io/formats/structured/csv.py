"""CSV format — lazy read + write via pyarrow, with byte-range splits."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Any

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import FileSplit, Split, read_aligned_range

__all__ = ["CSVRangeSplit", "CSVSink", "CSVSource"]

# Below this many rows a single-file CSV write stays serial: the thread-pool + buffer
# overhead isn't worth it. Above it, encode row ranges concurrently (one range per core).
_CSV_PARALLEL_MIN_ROWS = 200_000
# Batches per core held in a streaming CSV write's parallel-encode window (bounds memory).
_CSV_STREAM_WINDOW_PER_CORE = 2


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

    Examples:
        .. doctest::

            >>> from batcher.io import CSVSource  # doctest: +SKIP
            >>> src = CSVSource("s3://bucket/events/*.csv")  # doctest: +SKIP
            >>> src.schema().names  # doctest: +SKIP
            ['id', 'ts']
    """

    suffix = ".csv"
    format_name = "csv"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        import pyarrow.csv as pacsv

        # Infer the schema from the first block only (the streaming reader's schema is known
        # after one block) instead of reading the whole file — schema inference for a scan
        # runs during planning, so reading a multi-GB CSV end-to-end here would read it once
        # for the schema and again for the data. First-block inference is what pyarrow's own
        # streaming read commits to (and what DuckDB/Polars sample), so the schema matches.
        return pacsv.open_csv(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        import pyarrow.csv as pacsv

        # Push the projection into the parse (`include_columns`) so pyarrow only *converts*
        # the wanted columns — a projected scan skips the (often costly) string/decimal
        # conversion of the columns it drops, instead of parsing all then selecting.
        convert = (
            pacsv.ConvertOptions(include_columns=projection) if projection is not None else None
        )
        table = pacsv.read_csv(fh, convert_options=convert)
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
    """Write a CSV file.

    Row ranges are encoded concurrently into in-memory buffers (only the first
    carries the header) and written back to back, so a single-file write is not
    bottlenecked on pyarrow's single-threaded CSV writer.

    Examples:
        .. doctest::

            >>> import pyarrow as pa  # doctest: +SKIP
            >>> from batcher.io import CSVSink  # doctest: +SKIP
            >>> CSVSink().write(pa.table({"x": [1, 2]}), "out.csv").rows  # doctest: +SKIP
            2
    """

    suffix = ".csv"
    format_name = "csv"

    __slots__ = ()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        import pyarrow.csv as pacsv

        # A single-file CSV write is otherwise a serial encode (pyarrow's writer is
        # single-threaded) — the slow path that loses a directory-vs-file race to an
        # engine that shards its write. But CSV is just row-wise text, so encode row
        # ranges CONCURRENTLY (pyarrow's CSV encoder releases the GIL) into in-memory
        # buffers — only the first carries the header — and write them back to back.
        n = table.num_rows
        workers = min(n // _CSV_PARALLEL_MIN_ROWS, available_cpu_count())
        if workers <= 1:
            pacsv.write_csv(table, fh)
            return
        rows = -(-n // workers)  # ceil
        slices = [(i, table.slice(off, rows)) for i, off in enumerate(range(0, n, rows))]

        def _encode(item: tuple[int, pa.Table]) -> pa.Buffer:
            idx, chunk = item
            sink = pa.BufferOutputStream()
            pacsv.write_csv(chunk, sink, write_options=pacsv.WriteOptions(include_header=idx == 0))
            return sink.getvalue()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            buffers = list(pool.map(_encode, slices))  # order preserved
        for buf in buffers:
            fh.write(memoryview(buf))

    def write_stream(self, batches, path, *, schema=None, resume=False):  # type: ignore[override]
        """Stream to one CSV file with a **parallel** encode, in bounded memory.

        The base streaming write appends one batch at a time through pyarrow's
        single-threaded `CSVWriter` — the serial encode that loses a directory-vs-file
        race. CSV rows are independent text, so instead accumulate a window of batches and
        encode them CONCURRENTLY to byte buffers (pyarrow's CSV encoder releases the GIL;
        only the first batch of the whole stream carries the header), writing each window
        back to back. Peak memory is one window, not the whole result — still out-of-core.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> from batcher.io import CSVSink  # doctest: +SKIP
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # doctest: +SKIP
                >>> CSVSink().write_stream(ds.iter_batches(), "out.csv").rows  # doctest: +SKIP
                3

        Args:
            batches: The batches to encode, consumed one at a time.
            path: Destination file URI.
            schema: Schema used to write a valid empty file when `batches` yields
                nothing.
            resume: Leave an already-present (hence complete) file untouched.
        """
        from itertools import chain

        from batcher.io.base import _safe_size
        from batcher.io.manifest import WrittenFile

        fs = resolve_filesystem(path)
        if resume and fs.exists(path):
            return WrittenFile(path=path, rows=0, bytes=_safe_size(fs, path))
        it = iter(batches)
        first = next(it, None)
        rows = 0
        with fs.atomic_writer(path) as fh:
            if first is None:
                self._write_file(schema.empty_table() if schema is not None else pa.table({}), fh)
            else:
                rows = self._encode_stream_parallel(chain([first], it), fh)
        return WrittenFile(path=path, rows=rows, bytes=_safe_size(fs, path))

    def _encode_stream_parallel(self, batches: Iterator[pa.RecordBatch], fh: IO[Any]) -> int:
        import pyarrow.csv as pacsv

        def _encode(item: tuple[int, pa.RecordBatch]) -> pa.Buffer:
            idx, batch = item
            sink = pa.BufferOutputStream()
            pacsv.write_csv(
                pa.table(batch), sink, write_options=pacsv.WriteOptions(include_header=idx == 0)
            )
            return sink.getvalue()

        cores = available_cpu_count()
        window = max(1, cores * _CSV_STREAM_WINDOW_PER_CORE)
        rows = 0
        buf: list[pa.RecordBatch] = []
        emitted = 0

        def flush() -> int:
            with ThreadPoolExecutor(max_workers=min(len(buf), cores)) as pool:
                for out in pool.map(_encode, [(emitted + i, b) for i, b in enumerate(buf)]):
                    fh.write(memoryview(out))
            return len(buf)

        for batch in batches:
            if not batch.num_rows:
                continue
            buf.append(batch)
            rows += batch.num_rows
            if len(buf) >= window:
                emitted += flush()
                buf = []
        if buf:
            flush()
        return rows

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        import pyarrow.csv as pacsv

        return pacsv.CSVWriter(fh, schema)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write(batch)

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
