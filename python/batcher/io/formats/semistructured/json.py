"""JSON format — newline-delimited (line) JSON read + write."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher._internal.hardware import available_cpu_count
from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.base._bad_rows import bad_row_handler
from batcher.io.base._options import BASE_SOURCE_OPTIONS, OptionSpec
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.semistructured.json_tolerance import read_json_records
from batcher.io.splits import FileSplit, LineRangeSplit, Split

__all__ = ["JSONSink", "JSONSource"]

#: The JSON reader's keyword vocabulary. Batcher reads **newline-delimited** JSON, which is
#: what `pandas.read_json(..., lines=True)` and `polars.read_ndjson` produce, so `lines=True`
#: is the shape already assumed and accepting it is free. The options that describe a
#: *different* shape (`lines=False`, a non-``records`` `orient`) are refused by name rather
#: than ignored: ignoring them would read a JSON array file as one malformed record per line
#: and report a schema nobody asked for, which is a wrong answer rather than a missing
#: feature. `columns`/`n_rows` are handled by `FileSource` for every format at once.
_JSON_READ_OPTIONS = OptionSpec(
    "json",
    base=BASE_SOURCE_OPTIONS,
    # `lines` is canonical rather than ignored because its *value* decides whether the file
    # is even readable: `lines=False` names a JSON-array file, a different format. It is
    # validated in `__init__` and never reaches the base.
    canonical=("lines", "on_bad_lines"),
    aliases={"on_bad_rows": "on_bad_lines"},
    ignored={
        "typ": "Batcher always produces a table, never a Series.",
        "precise_float": (
            "floats are parsed exactly; there is no fast-but-lossy mode to opt out of."
        ),
    },
    unsupported={
        "mode": (
            "Spark's read mode has no single Batcher spelling because it conflates two "
            "independent decisions. Pass on_bad_lines='skip' for DROPMALFORMED and "
            "on_bad_lines='error' (the default) for FAILFAST. PERMISSIVE, which keeps a "
            "malformed record and parks its text in a corrupt-record column, has no "
            "equivalent."
        ),
        "ignore_errors": (
            "Polars folds two behaviors into this flag. Pass on_bad_lines='skip' to drop "
            "records that are not valid JSON, and declare the column with schema= if what "
            "you want is for a value that does not fit the inferred type to survive."
        ),
        "orient": (
            "Batcher reads newline-delimited JSON (one object per line), the only orient "
            "that streams and splits. Convert an array-of-objects file first, e.g. "
            "pandas.read_json(p).to_json(out, orient='records', lines=True)."
        ),
        "index_col": (
            "Batcher has no row index; every column is a real column. Drop index_col=, and "
            "select the column explicitly."
        ),
        "chunksize": (
            "reads are already streamed in batches. Use ds.iter_batches() to consume them, "
            "or n_rows= to bound the read."
        ),
    },
)

# Below this many rows a JSON write encodes serially (pandas `to_json` if available, else
# stdlib per row); above it, the encode fans across processes (pandas holds the GIL).
_JSON_PARALLEL_MIN_ROWS = 200_000
_JSON_COUNTER = 0
# Bytes of raw NDJSON `_iter_file` decodes per step. `pyarrow.json` has no incremental
# reader (no `open_json` counterpart to `open_csv`), so streaming means cutting the file at
# newline boundaries and decoding one cut at a time. 8 MiB keeps peak memory at a window
# rather than a file while still amortizing the parse.
_JSON_STREAM_CHUNK_BYTES = max(1 << 16, int(os.environ.get("BATCHER_JSON_CHUNK_BYTES", 8 << 20)))


def _newline_chunks(fh: IO[Any], size: int) -> Iterator[bytes]:
    """Cut `fh` into ``~size``-byte pieces that each end on a line boundary.

    A piece ending at the last ``\\n`` in a block holds only whole NDJSON records and so
    decodes on its own; the tail past that newline carries into the next piece. Every piece
    concatenated reproduces the file exactly — no record split, duplicated, or dropped. A
    block with no newline (a record longer than `size`) accumulates rather than being cut,
    so a wide row is read whole; a final line without a newline is yielded as its own piece.
    """
    remainder = b""
    while True:
        block = fh.read(size)
        if not block:
            break
        block = remainder + block
        cut = block.rfind(b"\n")
        if cut == -1:
            remainder = block  # one record spans the whole block — keep accumulating
            continue
        yield block[: cut + 1]
        remainder = block[cut + 1 :]
    if remainder.strip():
        yield remainder


def _ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _rewind(fh: IO[Any]) -> bool:
    """Seek `fh` back to the start, reporting whether the stream can be re-read."""
    try:
        if not fh.seekable():
            return False
        fh.seek(0)
    except (OSError, ValueError, AttributeError):
        return False
    return True


from batcher.io.formats.semistructured.json_encoding import (  # noqa: E402
    _disable_json_proc,
    _json_encode_shard,
    _json_pool,
    _json_proc_disabled,
    _json_write_part,
    _ndjson_bytes,
)


@SOURCES.register("json")
class JSONSource(FileSource):
    """One or more newline-delimited (line) JSON files (file, directory, or glob).

    Large files are split into newline-aligned byte ranges (`LineRangeSplit`), so a
    single multi-GB NDJSON file is read in parallel across workers; small files use
    one split each. `pyarrow.json.read_json` reads each range whole, so per-task
    memory scales with the split size, not the whole file.

    Examples:
        .. doctest::

            >>> from batcher.io import JSONSource  # doctest: +SKIP
            >>> src = JSONSource("s3://bucket/events/*.json")  # doctest: +SKIP
            >>> src.schema().names  # doctest: +SKIP
            ['id', 'payload']
    """

    suffix = ".json"
    format_name = "json"

    __slots__ = ("_on_bad_lines",)

    def __init__(self, path: Any, **kwargs: Any) -> None:
        """Open an NDJSON source, accepting the pandas/Polars JSON reader vocabulary."""
        # Named `base_kwargs` (not `opts`) so the structural guard in
        # `tests/unit/test_split_fidelity_matrix.py` can see that the caller's keywords —
        # `on_error`, `schema_mode`, `files`, `columns`, `n_rows` — really are forwarded.
        base_kwargs = _JSON_READ_OPTIONS.resolve(kwargs)
        lines = base_kwargs.pop("lines", True)
        self._on_bad_lines = str(base_kwargs.pop("on_bad_lines", "error"))
        # Validated here, not at the first bad record: under `on_error='skip'` a late raise
        # is swallowed as an unreadable file, so a misspelled tolerance flag would turn into
        # silent whole-corpus loss.
        bad_row_handler(self._on_bad_lines)
        if not lines:
            raise FormatError(
                "json: lines=False names a JSON-array file (a single '[...]' document), "
                "which is a different format from the newline-delimited JSON Batcher "
                "reads — one object per line, so it streams and splits. Convert it first, "
                "e.g. pandas.read_json(p).to_json(out, orient='records', lines=True)."
            )
        super().__init__(path, **base_kwargs)

    def _estimated_row_count(self, byte_total: int | None) -> int | None:
        """An advisory row count from a byte sample — NDJSON has no footer to count.

        Every line is exactly one record (NDJSON has no header), so the shared delimited
        estimator extrapolates a count from the first file's average line width and the
        dataset's on-disk size. Advisory (`statistics()` marks it `exact_rows=False`) and
        O(1) I/O at plan time — enough to size a join against a JSON source, which otherwise
        reached the estimator with no cardinality at all. `byte_total` is the size
        `statistics()` already computed, reused so the file sizes are not swept twice.
        """
        from batcher.io.stats.row_estimate import estimate_delimited_rows

        return estimate_delimited_rows(
            self._fs, self._files(), has_header=False, total_bytes=byte_total
        )

    def _reader_kwargs(self) -> dict[str, object]:
        """The base kwargs plus this source's bad-record mode.

        Emitted only when it differs from the default, so a strict read's split kwargs (and
        therefore its `identity()`) are byte-identical to what they were before tolerance
        existed. Without this a tolerated read stayed tolerant on the driver and reverted to
        fail-fast on every worker — the shape of defect that passes every local test.
        """
        extra: dict[str, object] = {}
        if self._on_bad_lines != "error":
            extra["on_bad_lines"] = self._on_bad_lines
        return {**super()._reader_kwargs(), **extra}

    def _policy(self, path: str = "", *, observe: bool = True):
        """This source's bad-record policy, or None when a bad record must abort the read."""
        return bad_row_handler(
            self._on_bad_lines, path or self._path, format_name="json", observe=observe
        )

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        # `observe=False`: inference parses the same records the read is about to, so
        # counting here would report every dropped record twice.
        return read_json_records(fh, None, self._policy(observe=False)).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:

        if projection is not None:
            parse_options = self._projection_parse_options(projection)
            if parse_options is not None:
                try:
                    return read_json_records(fh, parse_options, self._policy()).to_batches()
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    # The explicit schema could not parse this file, but free inference
                    # might (a value the unified schema does not describe). Rewind and take
                    # the original path so pushdown can only ever *add* speed, never turn a
                    # readable file into an error. If the handle cannot be rewound the
                    # fallback would read from a consumed stream and silently return a
                    # truncated file, so the error is re-raised instead.
                    if not _rewind(fh):
                        raise

        table = read_json_records(fh, None, self._policy())
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one NDJSON file in newline-aligned windows rather than decoding it whole.

        Without this, `iter_batches()` fell through to `_read_file`, which hands the entire
        handle to `read_json` — so streaming was in name only: batches were yielded out of a
        table that already held the whole file, and a file larger than the worker's envelope
        failed before the first batch. `CSVSource._iter_file` fixed the same defect by moving
        to `open_csv`; `pyarrow.json` ships no such incremental reader, so `_newline_chunks`
        cuts the window here and each piece decodes on its own.

        **Every window is pinned to `self.schema()`**, which is what makes the pieces agree.
        `read_json` infers per call, so an all-integer window would type a column `int64`
        while a later one holding a float types it `double`, and a field absent from a window
        would be missing from that batch entirely — the windows of one file disagreeing with
        each other and with the schema the engine planned against. Pinning is what
        `_projection_parse_options`, `LineRangeSplit`, and every `CSVSource` read path already
        do, and it is why this returns exactly what `_read_file` returns.

        Falls back to the whole-file read when no schema can be pinned, and when the file
        yields no window at all — an empty file must keep raising `read_json`'s "Empty JSON
        file" rather than quietly reading as zero rows.
        """
        parse_options = self._projection_parse_options(projection)
        if parse_options is None:
            yield from super()._iter_file(path, projection)
            return

        produced = False
        with self._fs.open(path) as fh:
            for window in _newline_chunks(fh, _JSON_STREAM_CHUNK_BYTES):
                if not window.strip():
                    continue
                table = read_json_records(window, parse_options, self._policy(path))
                if not table.num_rows:
                    continue
                produced = True
                yield from table.to_batches()
        if not produced:
            yield from super()._iter_file(path, projection)

    def _projection_parse_options(self, projection: list[str] | None) -> Any:
        """`ParseOptions` that parse **only** `projection`, or None to read everything.

        `read_json` has no `columns=` argument, so the projected read used to parse every
        column of every record and then throw the unwanted ones away — the decode cost and
        the peak memory of a 200-column log file were paid in full to answer a two-column
        query. An `explicit_schema` listing just the projected fields, with unexpected fields
        `ignore`d, makes the parser skip the rest outright: they are never converted, never
        allocated, never freed.

        The schema comes from `self.schema()` — the source's own whole-file inference — so
        the types are *exactly* the ones the un-projected read would have produced, and the
        result is byte-identical to `read_json(...).select(projection)` (verified for nested
        struct and list columns, and for column ordering, which follows `projection` rather
        than file order). A column the unified schema has but this file lacks parses as all
        nulls, which is what `_normalize` would have filled in anyway.

        A `projection` of None pins the *whole* advertised schema instead of narrowing it.
        `_iter_file` needs that: a streamed window must be pinned whether or not a projection
        was pushed, or the windows of one file infer different types from each other.

        Returns None if the schema is unavailable or does not describe every projected
        column — with nothing trustworthy to force, reading everything stays correct.
        """
        import pyarrow.json as pajson

        try:
            schema = self.schema()
            fields = list(schema) if projection is None else [schema.field(n) for n in projection]
        except Exception:
            return None  # inference unavailable/incomplete → no pushdown, same answer
        if not fields:
            return None
        return pajson.ParseOptions(
            explicit_schema=pa.schema(fields),
            # Anything outside the projection is skipped rather than inferred — the whole
            # point. Without this, unlisted fields are still parsed and appended.
            unexpected_field_behavior="ignore",
        )

    def _file_splits(
        self,
        path: str,
        target_size: int | None,
        predicate: dict | None = None,  # noqa: ARG002 (NDJSON has no footer statistics to prune with)
    ) -> list[Split]:
        # Default byte-range split size (so one huge NDJSON file fans across workers
        # instead of reading on a single node) is `ExecutionConfig.split_bytes`.
        chunk = target_size or active_config().execution.split_bytes
        # `_reader_kwargs()` has to ride the split: a worker rebuilds the reader as
        # `SOURCES.get("json")(path, **kwargs)`, so anything omitted here silently reverts to
        # its constructor default. Omitting it made `read.json(..., on_error="skip")` a no-op
        # on every split-based path (the distributed executor, the streaming reader) — an
        # explicitly tolerated read quietly became fail-fast — and dropped a caller's
        # `filesystem=`/`storage_options=`, so a worker resolved its own backend from the
        # environment and read a *different store* than the driver was configured for.
        kwargs = self._reader_kwargs()
        try:
            size = self._fs.size(path)
        except (OSError, ValueError):
            return [FileSplit(self.format_name, path, kwargs)]
        if size <= chunk:
            return [FileSplit(self.format_name, path, kwargs)]
        return [
            LineRangeSplit(self.format_name, path, start, min(start + chunk, size), kwargs)
            for start in range(0, size, chunk)
        ]


@SINKS.register("json")
class JSONSink(FileSink):
    """Write newline-delimited (line) JSON.

    pyarrow has no JSON writer, so each row is serialized as one JSON object per
    line via the stdlib — the shape `JSONSource` / `pyarrow.json.read_json` reads.

    Examples:
        .. doctest::

            >>> import pyarrow as pa  # doctest: +SKIP
            >>> from batcher.io import JSONSink  # doctest: +SKIP
            >>> JSONSink().write(pa.table({"x": [1, 2]}), "out.json").rows  # doctest: +SKIP
            2
    """

    suffix = ".json"
    format_name = "json"

    __slots__ = ("_on_bad_lines",)

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        # A per-row `json.dumps` over `to_pylist()` is catastrophically slow (~35x behind a
        # directory-sharding engine). Prefer pandas' C-accelerated `to_json` (NDJSON), and
        # for a large table fan the encode across PROCESSES — pandas' encoder holds the GIL,
        # so threads don't help, but NDJSON chunks concatenate cleanly. Each worker writes
        # its own shard file (no result IPC); the driver just streams them back to back. Any
        # failure (no pandas, non-import-safe entrypoint) falls back to a correct serial path.
        n = table.num_rows
        workers = min(n // _JSON_PARALLEL_MIN_ROWS, available_cpu_count()) if n else 0
        if workers > 1 and not _json_proc_disabled():
            try:
                self._write_parallel(table, fh, workers)
                return
            except Exception:
                _disable_json_proc()  # a broken pool must not poison later writes
        self._write_serial(table, fh)

    def _write_parts(self, table, directory, file_index, resume, max_rows_per_file):  # type: ignore[override]
        """Write a directory's part files across PROCESSES (pandas' JSON encoder holds the
        GIL, so the base's thread-per-part write serializes). Each worker encodes and writes
        one part from its IPC chunk — no result IPC, no concat — which is the difference
        between a directory JSON write that loses to a sharding engine and one that beats it.
        """
        from batcher.io.manifest import WrittenFile

        n = table.num_rows
        base = super()._write_parts
        if max_rows_per_file is None or n <= max_rows_per_file or _json_proc_disabled():
            return base(table, directory, file_index, resume, max_rows_per_file)
        tasks = [
            (
                _ipc_bytes(table.slice(start, max_rows_per_file)),
                f"{directory}/part-{file_index:05d}-{ci:05d}{self.suffix}",
                resume,
            )
            for ci, start in enumerate(range(0, n, max_rows_per_file))
        ]
        try:
            pool = _json_pool(min(len(tasks), available_cpu_count()))
            parts = list(pool.map(_json_write_part, tasks))
        except Exception:
            # A non-import-safe entrypoint can't fork a worker; fall back to the base's
            # thread-per-part write (correct, just serial for JSON's GIL-bound encoder).
            _disable_json_proc()
            return base(table, directory, file_index, resume, max_rows_per_file)
        return [WrittenFile(path=p, rows=r, bytes=b) for p, r, b in parts]

    def write_stream(self, batches, path, *, schema=None, resume=False):  # type: ignore[override]
        """Stream NDJSON to one file, encoding one batch at a time (bounded memory).

        The base `write_stream` buffers the whole result into one table before encoding
        (JSON has no incremental pyarrow writer). NDJSON rows are independent, so instead
        encode each incoming batch to NDJSON bytes and write it straight through — a
        breaker-free read→transform→write over a huge source never materializes on the
        driver. Atomic and `resume`-safe like the base; `schema` writes a valid empty
        file when the stream yields nothing.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> from batcher.io import JSONSink  # doctest: +SKIP
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # doctest: +SKIP
                >>> JSONSink().write_stream(ds.iter_batches(), "out.json").rows  # doctest: +SKIP
                3

        Args:
            batches: The batches to encode, consumed one at a time.
            path: Destination file URI.
            schema: Schema used to write a valid empty file when `batches` yields
                nothing.
            resume: Leave an already-present (hence complete) file untouched.
        """
        from itertools import chain

        def encode(first, rest, fh) -> int:
            rows = 0
            for batch in chain([first], rest):
                if not batch.num_rows:
                    continue
                fh.write(_ndjson_bytes(pa.Table.from_batches([batch])))
                rows += batch.num_rows
            return rows

        return self._stream_to_file(batches, path, schema=schema, resume=resume, encode=encode)

    def _write_serial(self, table: pa.Table, fh: IO[Any]) -> None:
        fh.write(_ndjson_bytes(table))

    def _write_parallel(self, table: pa.Table, fh: IO[Any], workers: int) -> None:
        import contextlib
        import shutil

        global _JSON_COUNTER
        _JSON_COUNTER += 1
        from batcher._internal.site.container import shm_root

        # Sized against the table, because a `/dev/shm` large enough in general is not large
        # enough for this write, and the failure lands mid-encode as ENOSPC.
        root = shm_root(table.nbytes)
        rows = -(-n // workers) if (n := table.num_rows) else 1
        tasks = []
        for i, off in enumerate(range(0, table.num_rows, rows)):
            out = os.path.join(root, f"bcjson_{os.getpid()}_{_JSON_COUNTER}_{i}")
            tasks.append((_ipc_bytes(table.slice(off, rows)), out))
        try:
            out_paths = list(_json_pool(workers).map(_json_encode_shard, tasks))
            for out in out_paths:
                with open(out, "rb") as src:
                    shutil.copyfileobj(src, fh)
        finally:
            for _ipc, out in tasks:
                with contextlib.suppress(OSError):
                    os.remove(out)
