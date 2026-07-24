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
from batcher.io.detect import compression_for_path
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.structured._csv_diagnostics import (
    invalid_utf8_error,
    mismatch_reported,
)
from batcher.io.formats.structured._csv_options import (
    CSVReadOptions,
    resolve_read_options,
    resolve_write_options,
)
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
    # The source's parse/convert vocabulary (delimiter, quoting, null/boolean tokens),
    # carried for exactly the reason `declared_schema` is: a worker rebuilds the parse from
    # the split alone. A range that re-parses a semicolon-separated file with the default
    # comma does not fail — it produces one wide string column per row, silently, on the
    # distributed path only. Only options that leave rows and byte offsets alone appear
    # here; `CSVReadOptions.range_safe` is what keeps the rest from ever reaching a range.
    options: dict[str, Any] | None = None

    def _options(self) -> CSVReadOptions:
        return resolve_read_options(self.options or {})

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
        #
        # The projection is pushed into the parse (`include_columns`) rather than applied
        # with `.select` afterwards, matching `CSVSource._read_file`. Selecting afterwards
        # still *converts* every column of every row — on a wide table that is nearly the
        # whole cost of the read, paid on the distributed path (this split IS the
        # distributed CSV read) to build columns that are immediately discarded. Pinning
        # `column_types` to the projected subset keeps each range agreeing with the source
        # schema on the columns it actually produces.
        types = {field.name: field.type for field in schema}
        if projection is not None:
            types = {name: t for name, t in types.items() if name in set(projection)}
        options = self._options()
        return pacsv.read_csv(
            io.BytesIO(data),
            parse_options=options.parse_options(),
            convert_options=options.convert_options(column_types=types, include_columns=projection),
        )

    def schema(self) -> pa.Schema:
        if self.declared_schema is not None:
            return self.declared_schema
        from batcher.io.formats.base import SOURCES

        return SOURCES.get("csv")(self.path, **(self.options or {})).schema()

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

    __slots__ = ("_options",)

    def __init__(self, path: str, **kwargs: Any) -> None:
        # The base owns the source-wide keywords (paths, filesystem, credentials, error
        # tolerance); everything else is CSV vocabulary. The split is read from the base
        # signature rather than restated here so a keyword added to `FileSource` reaches it
        # instead of being rejected as an unknown CSV option.
        import inspect

        from batcher.io.base._options import split_base_options

        base = set(inspect.signature(FileSource.__init__).parameters) - {"self", "path"}
        # `split_base_options` folds the base *aliases* in before splitting, so `usecols`
        # is recognized as the base's `columns` rather than mistaken for CSV vocabulary
        # and handed to the CSV option builder, which has no such field.
        base_kwargs, own = split_base_options(kwargs, base)
        super().__init__(path, **base_kwargs)
        self._options = resolve_read_options(own)

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
        if self._options.declared_schema is not None:
            return self._options.declared_schema
        # `dtype=`/`schema_overrides=`/`parse_dates=[...]` are per-column overrides laid over
        # inference, so they are applied *here*, on the one schema every read path pins to —
        # never inside a read path, which is how the three of them would drift apart again.
        return self._options.resolve_schema(super().schema())

    def _estimated_row_count(self, byte_total: int | None) -> int | None:
        """An advisory row count extrapolated from a byte sample (CSV has no footer).

        CSV reaches the estimator with no row count, so a join against one was sized from the
        planner's default. Scaling the first file's average row width by the dataset's on-disk
        size gives a far better cardinality — advisory (`exact_rows=False`), O(1) I/O. The
        header is discounted; `byte_total` (already computed by `statistics()`) is reused.
        """
        from batcher.io.stats.row_estimate import estimate_delimited_rows

        return estimate_delimited_rows(
            self._fs, self._files(), has_header=self._options.has_header, total_bytes=byte_total
        )

    def _reader_kwargs(self) -> dict[str, object]:
        """Every option that changes what a parse produces, so a worker rebuilds it exactly.

        The resolved schema rides along rather than the caller's `dtype` overrides: a worker
        holds one file, so re-deriving the schema there would re-infer from that file's rows
        and could disagree with the schema the plan was built against.
        """
        return {**super()._reader_kwargs(), **self._options.as_kwargs(), "schema": self.schema()}

    def _convert_options(self, projection: list[str] | None) -> Any:
        """Convert options pinning the advertised column types (and the projection).

        Pinning is what makes the read paths agree with `schema()` and with each other.
        Without it pyarrow re-infers per read, over a different amount of data each time.
        """
        types = {field.name: field.type for field in self.schema()}
        if projection is not None:
            types = {name: t for name, t in types.items() if name in set(projection)}
        return self._options.convert_options(column_types=types, include_columns=projection)

    def _parse_kwargs(self, projection: list[str] | None, *, pin_types: bool) -> dict[str, Any]:
        """The pyarrow option objects shared by schema inference and every read path.

        There is one builder because there must be one answer. Inference is the only caller
        that does not pin column types — it is the thing deciding them — but it must see the
        identical delimiter, quoting, header framing and null vocabulary, or the schema it
        commits to describes a different parse than the one that later runs.
        """
        return {
            "read_options": self._options.read_options(),
            "parse_options": self._options.parse_options(),
            "convert_options": (
                self._convert_options(projection)
                if pin_types
                else self._options.convert_options(include_columns=projection)
            ),
        }

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        import pyarrow.csv as pacsv

        # Infer the schema from the first block only (the streaming reader's schema is known
        # after one block) instead of reading the whole file — schema inference for a scan
        # runs during planning, so reading a multi-GB CSV end-to-end here would read it once
        # for the schema and again for the data. First-block inference is what pyarrow's own
        # streaming read commits to (and what DuckDB/Polars sample), so the schema matches.
        schema = pacsv.open_csv(fh, **self._parse_kwargs(None, pin_types=False)).schema
        # pyarrow reports undecodable bytes by *typing the column `binary`* rather than by
        # failing, which turns a corrupt file into a successful read of a differently-typed
        # table. Nothing downstream can tell that apart from a column of genuine bytes, so
        # the refusal has to happen here, where the choice was made.
        binary_cols = [f.name for f in schema if pa.types.is_binary(f.type)]
        if binary_cols:
            raise invalid_utf8_error(
                self._path, f"column(s) {binary_cols} could not be decoded as text"
            )
        return schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        import pyarrow.csv as pacsv

        # The projection is pushed into the parse (`include_columns`) so pyarrow only
        # *converts* the wanted columns, and the types are pinned so this path cannot
        # disagree with `schema()` — it used to re-infer over the whole file and return a
        # widened type the engine had not planned for.
        with mismatch_reported(self._path):
            table = pacsv.read_csv(fh, **self._parse_kwargs(projection, pin_types=True))
        return table.to_batches()

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one CSV a block at a time rather than decoding it whole.

        `read_csv` materializes the entire decoded table — measured at ~2.2x the file size
        in peak RSS for a 225 MB CSV — which makes `iter_batches` streaming in name only and
        caps the file size a worker can handle. `open_csv` returns pyarrow's incremental
        reader, so peak memory is one block regardless of how large the file is.
        """
        import pyarrow.csv as pacsv

        # `_open` rather than `_fs.open`, so `events.csv.gz` streams here too.
        with self._open(path) as fh:
            reader = pacsv.open_csv(fh, **self._parse_kwargs(projection, pin_types=True))
            with mismatch_reported(self._path):
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
        # A byte range is only meaningful over the bytes that are actually on disk, and only
        # when the file's row framing is uniform across ranges. A compressed file has neither
        # (an offset into the gzip stream is not an offset into the CSV), and options such as
        # `skip_rows` or a headerless file would be re-applied to every range — dropping or
        # inventing rows, silently, on the distributed path alone. Both cases read as one
        # whole-file split instead: slower for one huge file, never wrong.
        if compression_for_path(path) is not None or not self._options.range_safe:
            return [FileSplit(self.format_name, path, self._reader_kwargs())]
        try:
            size = self._fs.size(path)
        except (OSError, ValueError):
            # `_reader_kwargs()` here too, not just on the sized branch below. A file whose
            # size cannot be read is exactly the object-store/BYO-backend case that most
            # needs the caller's `filesystem=`/`storage_options=` and its `on_error` policy
            # carried to the worker — dropping them rebuilt a reader that resolves its own
            # backend from the environment and fails fast on a read the caller declared
            # tolerant.
            return [FileSplit(self.format_name, path, self._reader_kwargs())]
        if size <= chunk:
            return [FileSplit(self.format_name, path, self._reader_kwargs())]
        # The advertised schema, not the caller's raw `schema=`, so every range is pinned to
        # the same types the plan was built against even when they came from inference.
        schema, options = self.schema(), self._options.range_kwargs()
        return [
            CSVRangeSplit(path, start, min(start + chunk, size), schema, options)
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

    __slots__ = ("_options",)

    def __init__(self, **kwargs: Any) -> None:
        import inspect

        base = set(inspect.signature(FileSink.__init__).parameters) - {"self"}
        super().__init__(**{k: v for k, v in kwargs.items() if k in base})
        self._options = resolve_write_options({k: v for k, v in kwargs.items() if k not in base})

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        import pyarrow.csv as pacsv

        # A single-file CSV write is otherwise a serial encode (pyarrow's writer is
        # single-threaded) — the slow path that loses a directory-vs-file race to an
        # engine that shards its write. But CSV is just row-wise text, so encode row
        # ranges CONCURRENTLY (pyarrow's CSV encoder releases the GIL) into in-memory
        # buffers — only the first carries the header — and write them back to back.
        table = self._options.apply_nulls(table)
        n = table.num_rows
        workers = min(n // _CSV_PARALLEL_MIN_ROWS, available_cpu_count())
        if workers <= 1:
            options = self._options.write_options(include_header=True)
            pacsv.write_csv(table, fh, write_options=options)
            return
        rows = -(-n // workers)  # ceil
        slices = [(i, table.slice(off, rows)) for i, off in enumerate(range(0, n, rows))]

        def _encode(item: tuple[int, pa.Table]) -> pa.Buffer:
            idx, chunk = item
            sink = pa.BufferOutputStream()
            # `include_header=idx == 0` is the chunk's turn; `write_options` ANDs it with the
            # caller's `header=`, so `header=False` suppresses it on the first chunk too.
            pacsv.write_csv(
                chunk, sink, write_options=self._options.write_options(include_header=idx == 0)
            )
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
            # `idx` counts batches across the WHOLE stream, not within a window, so exactly
            # one chunk is ever offered the header; `write_options` then honors `header=`.
            pacsv.write_csv(
                self._options.apply_nulls(pa.table(batch)),
                sink,
                write_options=self._options.write_options(include_header=idx == 0),
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

        # The writer is opened against the schema `apply_nulls` will actually hand it — an
        # all-string one when `null_value=` is set — or every appended batch would be
        # rejected for not matching the schema the writer was opened with.
        return pacsv.CSVWriter(
            fh,
            self._options.null_schema(schema),
            write_options=self._options.write_options(include_header=True),
        )

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write(self._options.apply_nulls(pa.table(batch)))

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
