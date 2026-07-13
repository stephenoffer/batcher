"""JSON format — newline-delimited (line) JSON read + write."""

from __future__ import annotations

import json
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


def _table_to_ndjson(table: pa.Table) -> bytes:
    """Encode `table` as newline-delimited JSON bytes via pandas' C-accelerated writer."""
    ndjson = table.to_pandas().to_json(orient="records", lines=True)
    if ndjson and not ndjson.endswith("\n"):
        ndjson += "\n"  # so shard outputs concatenate into valid NDJSON
    return ndjson.encode("utf-8")


def _ndjson_bytes(table: pa.Table) -> bytes:
    """`_table_to_ndjson` with a stdlib fallback when pandas is unavailable."""
    try:
        return _table_to_ndjson(table)
    except Exception:
        return b"".join((json.dumps(row) + "\n").encode("utf-8") for row in table.to_pylist())


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
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if _JSON_POOL is None or n > _JSON_POOL_SIZE:
        if _JSON_POOL is not None:
            _JSON_POOL.shutdown(wait=False)
        else:
            atexit.register(lambda: _JSON_POOL and _JSON_POOL.shutdown(wait=False))
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("forkserver" if "forkserver" in methods else "fork")
        _JSON_POOL = ProcessPoolExecutor(max_workers=n, mp_context=ctx)
        _JSON_POOL_SIZE = n
    return _JSON_POOL


def _json_encode_shard(task: tuple[bytes, str]) -> str:
    """Worker: decode an IPC chunk, encode it to NDJSON, write it to `out_path`."""
    ipc, out_path = task
    with pa.ipc.open_stream(pa.py_buffer(ipc)) as reader:
        table = reader.read_all()
    with open(out_path, "wb") as fh:
        fh.write(_table_to_ndjson(table))
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
    data = _table_to_ndjson(table)
    with fs.atomic_writer(path) as fh:
        fh.write(data)
    return path, table.num_rows, len(data)


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

        table = pajson.read_json(fh)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _file_splits(self, path: str, target_size: int | None) -> list[Split]:
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
