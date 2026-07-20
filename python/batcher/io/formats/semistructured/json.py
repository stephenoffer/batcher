"""JSON format — newline-delimited (line) JSON read + write."""

from __future__ import annotations

import json
import math
import os
from typing import IO, Any

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import FileSplit, LineRangeSplit, Split

__all__ = ["JSONSink", "JSONSource"]

# Below this many rows a JSON write encodes serially (pandas `to_json` if available, else
# stdlib per row); above it, the encode fans across processes (pandas holds the GIL).
_JSON_PARALLEL_MIN_ROWS = 200_000
_JSON_COUNTER = 0


def _nullable_int_mapper(arrow_type: pa.DataType) -> Any:
    """Map an integer Arrow type to pandas' nullable integer dtype (else default).

    ``Table.to_pandas`` upcasts an integer column that contains a null to float64 —
    which silently turns ``9007199254740993`` into ``9007199254740992.0`` and changes
    the column's type on a JSON round-trip. Mapping integer columns to pandas' nullable
    integer extension dtypes keeps every value exact and integer-typed through
    ``to_json``.
    """
    import pandas as pd

    if pa.types.is_integer(arrow_type):
        return pd.ArrowDtype(arrow_type)
    return None


def _table_to_ndjson(table: pa.Table) -> bytes:
    """Encode `table` as newline-delimited JSON bytes via pandas' C-accelerated writer."""
    df = table.to_pandas(types_mapper=_nullable_int_mapper)
    ndjson = df.to_json(orient="records", lines=True)
    if ndjson and not ndjson.endswith("\n"):
        ndjson += "\n"  # so shard outputs concatenate into valid NDJSON
    return ndjson.encode("utf-8")


def _schema_has_float(schema: pa.Schema) -> bool:
    """True if `schema` holds a floating-point value anywhere (nested included)."""

    def _has_float(t: pa.DataType) -> bool:
        if pa.types.is_floating(t):
            return True
        if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            return _has_float(t.value_type)
        if pa.types.is_struct(t):
            return any(_has_float(t.field(i).type) for i in range(t.num_fields))
        if pa.types.is_map(t):
            return _has_float(t.key_type) or _has_float(t.item_type)
        return False

    return any(_has_float(f.type) for f in schema)


def _sanitize_nonfinite(value: Any) -> Any:
    """Replace NaN/±Inf floats with None (JSON has no non-finite; match pandas → ``null``)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_sanitize_nonfinite(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_nonfinite(v) for k, v in value.items()}
    return value


def _table_to_ndjson_exact(table: pa.Table) -> bytes:
    """Encode via the stdlib, so every float round-trips bit-for-bit.

    pandas' ``to_json`` rounds floats to ``double_precision`` (default 10) decimal
    places — ``3.141592653589793`` becomes ``3.1415926536`` — and even the maximum
    ``double_precision=15`` can round the largest double up to ``inf``. Python's
    ``json.dumps`` renders each float with ``repr`` (the shortest round-tripping form),
    so the value read back equals the value written. Raises on non-JSON-native leaves
    (timestamp/decimal/bytes), letting the caller fall back to the pandas encoder.
    """
    if table.num_rows == 0:
        # A 0-byte file is not valid NDJSON (`pyarrow.json.read_json` rejects it as
        # "Empty JSON file"); emit a single newline for a readable empty file, matching
        # the pandas encoder's output so a float-schema empty write reads back cleanly.
        return b"\n"
    return b"".join(
        (json.dumps(_sanitize_nonfinite(row)) + "\n").encode("utf-8") for row in table.to_pylist()
    )


def _ndjson_bytes(table: pa.Table) -> bytes:
    """Encode `table` as NDJSON, preserving float precision, with a pandas fast path.

    Float columns route through the exact stdlib encoder (pandas' ``to_json`` silently
    truncates them); float-free tables take pandas' faster C encoder. Either way falls
    back to the other on failure so a missing pandas or a non-JSON-native leaf still
    produces output.
    """
    if _schema_has_float(table.schema):
        try:
            return _table_to_ndjson_exact(table)
        except (TypeError, ValueError):
            pass  # mixed with a non-JSON-native leaf (e.g. timestamp) — use pandas
    try:
        return _table_to_ndjson(table)
    except Exception:
        return _table_to_ndjson_exact(table)


def _ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


_JSON_POOL: Any = None
_JSON_POOL_SIZE = 0
# Set once if the JSON process pool proves unusable this session (e.g. a non-import-safe
# entrypoint forkserver/spawn can't fork a child from). After that every JSON write stays
# on the serial/thread path — never re-attempting (and re-breaking) the pool per write.
_JSON_PROC_DISABLED = False


def _disable_json_proc() -> None:
    """Disable the JSON process pool for the rest of the session (idempotent)."""
    global _JSON_POOL, _JSON_POOL_SIZE, _JSON_PROC_DISABLED
    _JSON_PROC_DISABLED = True
    if _JSON_POOL is not None:
        _JSON_POOL.shutdown(wait=False)
        _JSON_POOL = None
        _JSON_POOL_SIZE = 0


def _json_pool(n: int) -> Any:
    """A process-lifetime pool for JSON encoding, grown lazily and reused across writes.

    Standing the forkserver pool up once (not per write) is what keeps a JSON write from
    paying ~1s of child-spawn each time; torn down at interpreter exit.
    """
    global _JSON_POOL, _JSON_POOL_SIZE
    import atexit
    from concurrent.futures import ProcessPoolExecutor

    if _JSON_POOL is None or n > _JSON_POOL_SIZE:
        if _JSON_POOL is not None:
            _JSON_POOL.shutdown(wait=False)
        else:
            atexit.register(lambda: _JSON_POOL and _JSON_POOL.shutdown(wait=False))
        from batcher._internal.hardware import process_start_method_context

        ctx = process_start_method_context()
        _JSON_POOL = ProcessPoolExecutor(max_workers=n, mp_context=ctx)
        _JSON_POOL_SIZE = n
    return _JSON_POOL


def _json_encode_shard(task: tuple[bytes, str]) -> str:
    """Worker: decode an IPC chunk, encode it to NDJSON, write it to `out_path`."""
    ipc, out_path = task
    with pa.ipc.open_stream(pa.py_buffer(ipc)) as reader:
        table = reader.read_all()
    with open(out_path, "wb") as fh:
        fh.write(_ndjson_bytes(table))
    return out_path


def _json_write_part(task: tuple[bytes, str, bool]) -> tuple[str, int, int]:
    """Worker: encode an IPC chunk to NDJSON and write it as one output part file.

    Uses the filesystem's atomic writer so a part is either complete or absent (resume-safe
    like the base sink). Returns ``(path, rows, bytes)`` for the manifest.
    """
    from batcher.io.filesystem import resolve_filesystem

    ipc, path, resume = task
    fs = resolve_filesystem(path)
    with pa.ipc.open_stream(pa.py_buffer(ipc)) as reader:
        table = reader.read_all()
    if resume and fs.exists(path):
        return path, table.num_rows, _size_or_zero(fs, path)
    data = _ndjson_bytes(table)
    with fs.atomic_writer(path) as fh:
        fh.write(data)
    return path, table.num_rows, len(data)


def _rewind(fh: IO[Any]) -> bool:
    """Seek `fh` back to the start, reporting whether the stream can be re-read."""
    try:
        if not fh.seekable():
            return False
        fh.seek(0)
    except (OSError, ValueError, AttributeError):
        return False
    return True


def _size_or_zero(fs: Any, path: str) -> int:
    try:
        return fs.size(path)
    except (OSError, ValueError):
        return 0


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

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        import pyarrow.json as pajson

        return pajson.read_json(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        import pyarrow.json as pajson

        if projection is not None:
            parse_options = self._projection_parse_options(projection)
            if parse_options is not None:
                try:
                    return pajson.read_json(fh, parse_options=parse_options).to_batches()
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    # The explicit schema could not parse this file, but free inference
                    # might (a value the unified schema does not describe). Rewind and take
                    # the original path so pushdown can only ever *add* speed, never turn a
                    # readable file into an error. If the handle cannot be rewound the
                    # fallback would read from a consumed stream and silently return a
                    # truncated file, so the error is re-raised instead.
                    if not _rewind(fh):
                        raise

        table = pajson.read_json(fh)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _projection_parse_options(self, projection: list[str]) -> Any:
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

        Returns None if the schema is unavailable or does not describe every projected
        column — with nothing trustworthy to force, reading everything stays correct.
        """
        import pyarrow.json as pajson

        try:
            schema = self.schema()
            fields = [schema.field(name) for name in projection]
        except Exception:
            return None  # inference unavailable/incomplete → no pushdown, same answer
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
        try:
            size = self._fs.size(path)
        except (OSError, ValueError):
            return [FileSplit(self.format_name, path)]
        if size <= chunk:
            return [FileSplit(self.format_name, path)]
        return [
            LineRangeSplit(self.format_name, path, start, min(start + chunk, size))
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

    __slots__ = ()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        # A per-row `json.dumps` over `to_pylist()` is catastrophically slow (~35x behind a
        # directory-sharding engine). Prefer pandas' C-accelerated `to_json` (NDJSON), and
        # for a large table fan the encode across PROCESSES — pandas' encoder holds the GIL,
        # so threads don't help, but NDJSON chunks concatenate cleanly. Each worker writes
        # its own shard file (no result IPC); the driver just streams them back to back. Any
        # failure (no pandas, non-import-safe entrypoint) falls back to a correct serial path.
        n = table.num_rows
        workers = min(n // _JSON_PARALLEL_MIN_ROWS, available_cpu_count()) if n else 0
        if workers > 1 and not _JSON_PROC_DISABLED:
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
        if max_rows_per_file is None or n <= max_rows_per_file or _JSON_PROC_DISABLED:
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

        from batcher.io.base import _safe_size
        from batcher.io.filesystem import resolve_filesystem
        from batcher.io.manifest import WrittenFile

        fs = resolve_filesystem(path)
        if resume and fs.exists(path):
            return WrittenFile(path=path, rows=0, bytes=_safe_size(fs, path))
        it = iter(batches)
        first = next(it, None)
        rows = 0
        with fs.atomic_writer(path) as fh:
            if first is None:
                empty = schema.empty_table() if schema is not None else pa.table({})
                self._write_serial(empty, fh)
            else:
                for batch in chain([first], it):
                    if not batch.num_rows:
                        continue
                    fh.write(_ndjson_bytes(pa.Table.from_batches([batch])))
                    rows += batch.num_rows
        return WrittenFile(path=path, rows=rows, bytes=_safe_size(fs, path))

    def _write_serial(self, table: pa.Table, fh: IO[Any]) -> None:
        fh.write(_ndjson_bytes(table))

    def _write_parallel(self, table: pa.Table, fh: IO[Any], workers: int) -> None:
        import contextlib
        import shutil
        import tempfile

        global _JSON_COUNTER
        _JSON_COUNTER += 1
        root = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
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
