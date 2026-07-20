"""CSV format — lazy read + write via pyarrow, with byte-range splits."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import SchemaError
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
    # The source's declared schema, carried because a worker rebuilds the reader from the
    # split alone: without it the range re-infers from its own bytes, and a range that
    # happens to hold only integers disagrees with the source and with its sibling ranges.
    declared_schema: pa.Schema | None = None

    def _header(self) -> bytes:
        fs = resolve_filesystem(self.path)
        with fs.open(self.path) as fh:
            return fh.readline()

    def _table(self, projection: list[str] | None) -> pa.Table:
        import io

        import pyarrow.csv as pacsv

        schema = self.schema()
        data = read_aligned_range(self.path, self.start, self.end)
        if self.start != 0:
            data = self._header() + data  # supply column names to a mid-file range
        if not data.strip():
            empty = schema.empty_table()
            return empty.select(projection) if projection is not None else empty
        # Force each range to the file's declared column types. pyarrow infers types
        # independently per `read_csv` call, so without this an early range that happens
        # to hold only integers parses that column as int64 while a later range with a
        # string parses it as string — the ranges of one file disagree with each other
        # and with the source schema. Pinning `column_types` makes every range parse to
        # the same schema the source advertises.
        convert = pacsv.ConvertOptions(column_types=schema)
        table = pacsv.read_csv(io.BytesIO(data), convert_options=convert)
        return table.select(projection) if projection is not None else table

    def schema(self) -> pa.Schema:
        if self.declared_schema is not None:
            return self.declared_schema
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


@contextlib.contextmanager
def _mismatch_reported(source: CSVSource):
    """Turn pyarrow's conversion error into one that says what to do about it.

    A CSV column is typed from the first block, so a value further down that does not fit
    is not a corrupt file — it is inference having been shown too little. The raw error
    names the offending value but not the inferred type, nor that the type is declarable,
    which is the whole of the fix.
    """
    try:
        yield
    except pa.ArrowInvalid as exc:
        raise SchemaError(
            f"CSV value does not fit the inferred column type in {source._path!r}: {exc}. "
            "The schema is inferred from the file's first block, so a value further down "
            "may not fit it. Declare the type instead — "
            'bt.read.csv(path, schema=pa.schema([("col", pa.string()), ...])).'
        ) from exc


@SOURCES.register("csv")
class CSVSource(FileSource):
    """One or more CSV files (single file, directory, or glob).

    Large files are split into newline-aligned byte ranges (`CSVRangeSplit`) so a
    single multi-GB CSV reads in parallel across workers; small files use one split
    each.

    **The schema is a contract, and CSV cannot see the future.** It is inferred from the
    first block — what pyarrow's streaming reader commits to, and what DuckDB and Polars
    sample — so a column that is integral for a million rows and then holds ``"N/A"``
    was inferred wrong. Every read path therefore *pins* the advertised schema, so all of
    them agree. They did not before: `schema()` said `int64`, `read()` re-inferred over
    the whole file and silently returned `string` (contradicting the schema the engine had
    already planned against), and `iter_batches()` raised. One file, three answers.

    Pass `schema=` to declare the truth when inference cannot reach it — the escape hatch
    the mismatch error points at.

    Examples:
        .. doctest::

            >>> from batcher.io import CSVSource  # doctest: +SKIP
            >>> src = CSVSource("s3://bucket/events/*.csv")  # doctest: +SKIP
            >>> src.schema().names  # doctest: +SKIP
            ['id', 'ts']
    """

    suffix = ".csv"
    format_name = "csv"

    __slots__ = ("_declared_schema",)

    def __init__(self, path: str, *, schema: pa.Schema | None = None, **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        self._declared_schema = schema

    def schema(self) -> pa.Schema:
        """The declared schema when one was given, else the one inferred from the file.

        Inference reads only the file's first block, so a column that is integral for a
        million rows and then holds ``"N/A"`` is inferred wrong. Declaring the schema is
        the escape hatch, and every read path is pinned to whatever this returns.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> import batcher as bt
                >>> _ = open("late.csv", "w").write("k,v\\n1,10\\n2,oops\\n")
                >>> declared = pa.schema([("k", pa.int64()), ("v", pa.string())])
                >>> bt.read.csv("late.csv", schema=declared).schema.field("v").type
                DataType(string)

        Returns:
            The schema every read path of this source will produce.
        """
        return self._declared_schema if self._declared_schema is not None else super().schema()

    def _reader_kwargs(self) -> dict[str, object]:
        """A declared schema changes how a split parses, so a worker must rebuild it."""
        base = super()._reader_kwargs()
        return (
            {**base, "schema": self._declared_schema} if self._declared_schema is not None else base
        )

    def _convert_options(self, projection: list[str] | None) -> Any:
        """Parse options pinning the advertised column types (and the projection).

        Pinning is what makes the read paths agree with `schema()` and with each other.
        Without it pyarrow re-infers per read, over a different amount of data each time.
        """
        import pyarrow.csv as pacsv

        types = {field.name: field.type for field in self.schema()}
        if projection is not None:
            types = {name: t for name, t in types.items() if name in set(projection)}
        return pacsv.ConvertOptions(include_columns=projection, column_types=types)

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

        # The projection is pushed into the parse (`include_columns`) so pyarrow only
        # *converts* the wanted columns, and the types are pinned so this path cannot
        # disagree with `schema()` — it used to re-infer over the whole file and return a
        # widened type the engine had not planned for.
        with _mismatch_reported(self):
            table = pacsv.read_csv(fh, convert_options=self._convert_options(projection))
        return table.to_batches()

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one CSV a block at a time rather than decoding it whole.

        `read_csv` materializes the entire decoded table — measured at ~2.2x the file size
        in peak RSS for a 225 MB CSV — which makes `iter_batches` streaming in name only and
        caps the file size a worker can handle. `open_csv` returns pyarrow's incremental
        reader, so peak memory is one block regardless of how large the file is.
        """
        import pyarrow.csv as pacsv

        with self._fs.open(path) as fh:
            reader = pacsv.open_csv(fh, convert_options=self._convert_options(projection))
            with _mismatch_reported(self):
                yield from reader

    def _file_splits(
        self,
        path: str,
        target_size: int | None,
        predicate: dict | None = None,  # noqa: ARG002 (CSV has no footer statistics to prune with)
    ) -> list[Split]:
        # Default byte-range split size (so one huge file fans across workers instead
        # of reading on a single node) is the configured `ExecutionConfig.split_bytes`.
        chunk = target_size or active_config().execution.split_bytes
        try:
            size = self._fs.size(path)
        except (OSError, ValueError):
            return [FileSplit(self.format_name, path)]
        if size <= chunk:
            return [FileSplit(self.format_name, path, self._reader_kwargs())]
        return [
            CSVRangeSplit(path, start, min(start + chunk, size), self._declared_schema)
            for start in range(0, size, chunk)
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
