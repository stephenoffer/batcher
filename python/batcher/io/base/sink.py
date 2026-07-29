"""`FileSink` — the Template-Method base every file-format writer subclasses.

Owns filesystem resolution, atomic writes, Hive partitioning, the per-file manifest,
and the parallel-friendly `write_partitioned` (one worker calls it for its shard), so
a concrete format is a tiny subclass overriding only `_write_file`.

The `Sink` protocol itself lives in `io.sink`; this base structurally satisfies it.
"""

from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import IO, Any, ClassVar

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.io.base._paths import normalize_path
from batcher.io.base._transient import with_retry
from batcher.io.filesystem import FileSystem, resolve_filesystem
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = ["FileSink"]

_HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"


def _partition_run_starts(ordered: pa.Table, cols: list[str], pc: Any) -> list[int]:
    """Row offsets where a new partition-key run begins in a key-sorted table.

    A row starts a run when any key column differs from the previous row, where "differs"
    treats NULL as equal to NULL and NaN as equal to NaN — the grouping `group_by` gives
    them, and the opposite of what `equal` gives.
    """
    n = ordered.num_rows
    if n == 0:
        return []
    changed = None
    for name in cols:
        column = ordered.column(name)
        previous, current = column.slice(0, n - 1), column.slice(1, n - 1)
        same = pc.fill_null(pc.equal(previous, current), False)
        same = pc.or_(same, pc.and_(pc.is_null(previous), pc.is_null(current)))
        if pa.types.is_floating(column.type):
            both_nan = pc.and_(
                pc.fill_null(pc.is_nan(previous), False),
                pc.fill_null(pc.is_nan(current), False),
            )
            same = pc.or_(same, both_nan)
        differs = pc.invert(pc.fill_null(same, False))
        changed = differs if changed is None else pc.or_(changed, differs)
    # `changed[i]` compares row i+1 against row i, so a True at i starts a run at i+1.
    return [0, *(i + 1 for i, flag in enumerate(changed.to_pylist()) if flag)]


# Write-side retry, mirroring the read path's `_READ_RETRY_*` so a deployment tunes one
# idea rather than two. See `FileSink.write` for why retrying a write is safe.
_WRITE_RETRY_ATTEMPTS = max(1, int(os.environ.get("BATCHER_WRITE_RETRY_ATTEMPTS", "3")))
_WRITE_RETRY_BACKOFF_S = max(0.0, float(os.environ.get("BATCHER_WRITE_RETRY_BACKOFF_S", "0.5")))


class FileSink(ABC):
    """Base for a format writer: single-file, partitioned, or one distributed shard.

    Subclasses set `suffix`/`format_name` and override `_write_file`. The base
    owns filesystem resolution, Hive partitioning, the per-file manifest, and the
    parallel-friendly `write_partitioned` (one worker calls it for its shard).

    Examples:
        .. doctest::

            >>> from batcher.io import FileSink, ParquetSink
            >>> issubclass(ParquetSink, FileSink)
            True
    """

    suffix: ClassVar[str] = ""
    format_name: ClassVar[str] = ""

    __slots__ = ("_filesystem", "_storage_options")

    def __init__(
        self,
        *,
        filesystem: object = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        # Bring-your-own filesystem / credentials, mirroring `FileSource`. Every sink is
        # reconstructed on a worker from `sink_kwargs`, so a subclass with its own
        # `__init__` MUST accept these and forward via `super().__init__(...)`, or a
        # distributed write resolves against the worker's env vars instead of the caller's.
        self._filesystem = filesystem
        self._storage_options = storage_options

    def _resolve(self, path: str) -> FileSystem:
        """Resolve `path`'s filesystem, honoring the caller's `filesystem`/`storage_options`."""
        return resolve_filesystem(
            path, filesystem=self._filesystem, storage_options=self._storage_options
        )

    @staticmethod
    def _dest(path: Any) -> str:
        """The destination as a plain string URI, from a `Path`/`os.PathLike`/``~`` value.

        Writers take a destination from the same vocabulary readers take a source from, so
        the coercion is the same one (`io.base._paths.normalize_path`) rather than a second
        set of rules. Applied at each public entry point, because the path is not only
        resolved to a filesystem — it is also recorded in the returned `WrittenFile`, and a
        manifest holding a `PosixPath` breaks the commit that reads it back.
        """
        return normalize_path(path, what="the write destination")

    def write(self, table: pa.Table, path: str, *, resume: bool = False) -> WrittenFile:
        """Write the whole table to a single file at `path`, atomically.

        The bytes become visible at `path` only once the write completes — a crash
        mid-write leaves any prior file intact (no truncated output), closing Ray
        Data's overwrite data-loss (ray#62019). Local writes go via a temp file +
        atomic rename; object stores write directly (a single PUT is already atomic).

        A **transient** failure is retried with jittered backoff, as the read path already
        does. A fast pipeline feeding a directory write bursts concurrent PUTs at one key
        prefix, which is what makes a store answer `SlowDown`/503 — a property of the moment,
        not of the data, that used to kill a job at 99%. Safe here because a failed attempt
        publishes nothing: the atomic writer discards the partial and the table is still in
        memory, so the retry writes the same bytes. `write_stream` has no equivalent — its
        batches are an iterator a retry cannot rewind.

        Examples:
            .. doctest::

                >>> import pyarrow as pa  # doctest: +SKIP
                >>> from batcher.io import ParquetSink  # doctest: +SKIP
                >>> table = pa.table({"x": [1, 2]})  # doctest: +SKIP
                >>> ParquetSink().write(table, "out.parquet").rows  # doctest: +SKIP
                2

        Args:
            table: The rows to persist.
            path: Destination file URI.
            resume: Leave a file already present at `path` untouched and report it
                as-is. Writes are atomic, so an existing file is a complete one —
                re-running a crashed job skips the work it finished.

        Returns:
            The file that was written, with its row count and size on storage.
        """
        path = self._dest(path)
        fs = self._resolve(path)
        if resume and fs.exists(path):
            return WrittenFile(path=path, rows=table.num_rows, bytes=_safe_size(fs, path))

        def _put() -> None:
            with fs.atomic_writer(path) as fh:
                self._write_file(table, fh)

        with_retry(_put, attempts=_WRITE_RETRY_ATTEMPTS, backoff_base_s=_WRITE_RETRY_BACKOFF_S)
        return WrittenFile(path=path, rows=table.num_rows, bytes=_safe_size(fs, path))

    def write_stream(
        self,
        batches: Iterator[pa.RecordBatch],
        path: str,
        *,
        schema: pa.Schema | None = None,
        resume: bool = False,
    ) -> WrittenFile:
        """Stream `batches` into one file at `path`, holding a single batch at a time.

        The bounded-memory counterpart of `write`: a breaker-free `read→transform→
        write` pipeline never materializes the whole result on the driver. Formats with
        an incremental writer (Parquet/CSV/Arrow append row-groups via
        `_open_stream_writer`) stream truly; the rest fall back to buffering one table,
        so any format stays correct. Atomic and `resume`-safe like `write`.

        Examples:
            .. doctest::

                >>> import batcher as bt  # doctest: +SKIP
                >>> from batcher.io import ParquetSink  # doctest: +SKIP
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # doctest: +SKIP
                >>> sink = ParquetSink()  # doctest: +SKIP
                >>> sink.write_stream(ds.iter_batches(), "out.parquet").rows  # doctest: +SKIP
                3

        Args:
            batches: The batches to persist, consumed one at a time.
            path: Destination file URI.
            schema: Schema used to write a valid empty file when `batches` yields
                nothing.
            resume: Leave an already-present (hence complete) file untouched.

        Returns:
            The file that was written, with its row count and size on storage.
        """
        from itertools import chain

        def encode(first: pa.RecordBatch, rest: Iterator[pa.RecordBatch], fh: IO[Any]) -> int:
            rows = 0
            if (writer := self._open_stream_writer(fh, first.schema)) is None:
                table = pa.Table.from_batches(list(chain([first], rest)))
                self._write_file(table, fh)
                return table.num_rows
            for batch in chain([first], rest):
                if batch.num_rows:
                    self._write_batch(writer, batch)
                    rows += batch.num_rows
            self._close_stream_writer(writer)
            return rows

        return self._stream_to_file(batches, path, schema=schema, resume=resume, encode=encode)

    def _stream_to_file(
        self,
        batches: Iterator[pa.RecordBatch],
        path: str,
        *,
        schema: pa.Schema | None,
        resume: bool,
        encode: Callable[[pa.RecordBatch, Iterator[pa.RecordBatch], IO[Any]], int],
    ) -> WrittenFile:
        """Run `encode` inside the scaffold every single-file streaming write shares.

        Destination normalization, filesystem resolution, the `resume` short-circuit, the
        atomic writer, the empty-stream case, and the `WrittenFile` accounting are identical
        for every format; only the encoding differs. A subclass that wants a different
        encoding strategy (NDJSON straight-through, a parallel CSV window) overrides
        `write_stream` and calls this with its own `encode` rather than restating the
        scaffold — which is how two of them came to resolve the filesystem with the
        module-level `resolve_filesystem`, silently dropping the caller's `storage_options`
        and `filesystem`, and to skip `_dest`, recording an un-normalized path in the
        manifest.

        Args:
            batches: The batches to persist, consumed one at a time.
            path: Destination file URI, normalized here.
            schema: Schema used to write a valid empty file when `batches` yields nothing.
            resume: Leave an already-present (hence complete) file untouched.
            encode: Given the first batch, the rest of the iterator, and the open file
                handle, write them and return the row count.

        Returns:
            The file that was written, with its row count and size on storage.
        """
        path = self._dest(path)
        fs = self._resolve(path)
        if resume and fs.exists(path):
            # Atomic writes ⇒ an existing file is a complete one; skip the redone work.
            # The exact row count needs a footer read, so it is best-effort here.
            return WrittenFile(path=path, rows=0, bytes=_safe_size(fs, path))
        it = iter(batches)
        first = next(it, None)
        rows = 0
        with fs.atomic_writer(path) as fh:
            if first is None:
                empty = schema.empty_table() if schema is not None else pa.table({})
                self._write_file(empty, fh)
            else:
                rows = encode(first, it, fh)
        return WrittenFile(path=path, rows=rows, bytes=_safe_size(fs, path))

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any | None:  # noqa: ARG002 (extension-point args used by overrides)
        """Open an incremental writer over `fh`, or None to buffer (the default).

        Formats that can append a batch at a time (Parquet/CSV/Arrow) return a writer
        object driven by `_write_batch`/`_close_stream_writer`; the default None makes
        `write_stream` buffer one table — correct, just not bounded-memory.
        """
        return None

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        """Append one batch to an open incremental `writer` (see `_open_stream_writer`)."""
        raise NotImplementedError

    def _close_stream_writer(self, writer: Any) -> None:
        """Flush and close an incremental `writer` (see `_open_stream_writer`)."""
        raise NotImplementedError

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,
        file_index: int = 0,
        resume: bool = False,
        max_rows_per_file: int | None = None,
    ) -> list[WrittenFile]:
        """Write `table` under directory `path` as one shard (`file_index`).

        Without `partition_by`, writes ``<path>/part-{file_index:05d}<suffix>``.
        With `partition_by`, writes Hive-layout ``<path>/c=v/.../part-…`` files,
        dropping the partition columns from the data (they live in the path).

        Examples:
            .. doctest::

                >>> import pyarrow as pa  # doctest: +SKIP
                >>> from batcher.io import ParquetSink  # doctest: +SKIP
                >>> table = pa.table({"c": ["a", "b"], "x": [1, 2]})  # doctest: +SKIP
                >>> sink = ParquetSink()  # doctest: +SKIP
                >>> len(sink.write_partitioned(table, "out", partition_by=["c"]))  # doctest: +SKIP
                2

        Args:
            table: The rows to persist.
            path: Destination directory URI.
            partition_by: Columns to encode as Hive ``col=value`` directories.
            file_index: This shard's index. It names the part files, so concurrent
                shards never collide.
            resume: Skip any (atomically written, hence complete) file that exists.
            max_rows_per_file: Cap each file's row count, splitting a large
                (sub)table into several parts. Honored *per partition* — the bug
                where Ray Data ignores ``min_rows_per_file`` alongside
                ``partition_cols``.

        Returns:
            One entry per file this shard wrote.
        """
        path = self._dest(path)
        fs = self._resolve(path)
        if not partition_by:
            fs.mkdirs(path, exist_ok=True)
            return self._write_parts(table, path, file_index, resume, max_rows_per_file)

        parts = list(self._hive_partition(table, partition_by))

        def _write_partition(item: tuple[list[tuple[str, Any]], pa.Table]) -> list[WrittenFile]:
            key_vals, sub = item
            sub_dir = "/".join([path, *(f"{c}={_hive_str(v)}" for c, v in key_vals)])
            fs.mkdirs(sub_dir, exist_ok=True)
            return [
                replace(w, partition_values=dict(key_vals))
                for w in self._write_parts(sub, sub_dir, file_index, resume, max_rows_per_file)
            ]

        # Write the partition directories CONCURRENTLY: each is an independent subtree
        # (its own mkdirs + encode + PUT), and the columnar encode/compression releases
        # the GIL, so a high-cardinality partitioned write no longer emits one partition
        # after another (the serial loop that loses a directory-vs-file race). Bounded to
        # the CPU count; `_write_parts` still parallelizes the chunks within a partition.
        # Partitions are disjoint dirs, so order is irrelevant (the manifest merge is
        # commutative), but results are kept in partition order for deterministic output.
        if len(parts) <= 1:
            return [w for item in parts for w in _write_partition(item)]
        workers = min(len(parts), available_cpu_count())
        out: list[WrittenFile] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for written in pool.map(_write_partition, parts):  # order preserved
                out.extend(written)
        return out

    def _write_parts(
        self,
        table: pa.Table,
        directory: str,
        file_index: int,
        resume: bool,
        max_rows_per_file: int | None,
    ) -> list[WrittenFile]:
        """Write `table` into `directory` as one part file, or several capped at
        `max_rows_per_file` rows. Chunk file names carry both the shard `file_index`
        and the chunk index so parts never collide across distributed shards."""
        if max_rows_per_file is None or table.num_rows <= max_rows_per_file:
            name = f"{directory}/part-{file_index:05d}{self.suffix}"
            return [self.write(table, name, resume=resume)]

        # Write the parts concurrently: the columnar encode + compression (Parquet/
        # Arrow/CSV) runs in the C++ layer with the GIL released, so a thread per part
        # turns a serial N-file write into a parallel one (a directory write — Ray Data's
        # default output shape — otherwise writes each shard back to back). Slices are
        # zero-copy views over one already-resident table, so this adds no memory.
        chunks = [
            (chunk_idx, table.slice(start, max_rows_per_file))
            for chunk_idx, start in enumerate(range(0, table.num_rows, max_rows_per_file))
        ]

        def _write_chunk(item: tuple[int, pa.Table]) -> WrittenFile:
            chunk_idx, chunk = item
            name = f"{directory}/part-{file_index:05d}-{chunk_idx:05d}{self.suffix}"
            return self.write(chunk, name, resume=resume)

        if len(chunks) == 1:
            return [_write_chunk(chunks[0])]
        workers = min(len(chunks), available_cpu_count())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_write_chunk, chunks))  # order preserved

    def commit(self, manifest: WriteManifest, path: str) -> None:
        """Finalize a write by publishing a `_SUCCESS` completion marker.

        A file sink publishes each data file as it is written, so there is nothing to make
        visible at commit time. What was missing is a way to tell a *complete* output
        directory from a half-written one: a distributed write that died at 90% leaves a
        directory of valid Parquet files that reads back cleanly and silently short. The
        marker is written only after every shard's manifest has been merged, so its presence
        means the write finished. Readers skip `_`-prefixed files, so it never joins the data.

        Best-effort: a sink whose filesystem rejects the marker still has its data committed,
        so a marker failure must not fail an otherwise successful write.

        Transactional (lakehouse) sinks override this to commit the manifest's files
        atomically into a transaction log instead.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSink, WriteManifest
                >>> ParquetSink().commit(WriteManifest(), "out")

        Args:
            manifest: Every file the write produced, merged across shards.
            path: The write's destination root.
        """
        if not self._is_directory_write(manifest, path):
            return
        # Suppressed narrowly: a filesystem that rejects the marker (read-only mount, missing
        # PUT permission) must not fail a write whose data is already durable. A broader
        # `except Exception` here would also swallow a bug in this method itself.
        with contextlib.suppress(OSError):
            fs = self._resolve(path)
            # `atomic_writer` publishes via a temp file + rename, so a marker is never
            # observed half-written — a reader either sees a complete write or no marker.
            with fs.atomic_writer(f"{path.rstrip('/')}/_SUCCESS") as out:
                out.write(b"")

    @staticmethod
    def _is_directory_write(manifest: WriteManifest, path: str) -> bool:
        """Whether `path` is a directory of data files rather than a single output file.

        A plain single-node `write.parquet("out.parquet")` produces one file *at* `path`, and
        `path/_SUCCESS` would be a nonsense location inside it. A partitioned or sharded write
        produces files *under* `path`, which is where a completion marker belongs.
        """
        root = path.rstrip("/")
        return bool(manifest.files) and all(f.path.rstrip("/") != root for f in manifest.files)

    @staticmethod
    def _hive_partition(
        table: pa.Table, cols: list[str]
    ) -> Iterator[tuple[list[tuple[str, Any]], pa.Table]]:
        """Yield `(key_values, sub_table)` per distinct partition-key combo.

        Sorts by the partition columns **once** and slices the contiguous runs, rather
        than selecting each partition with its own mask. The mask form was O(partitions x
        rows): every distinct key rebuilt a full-table comparison per key column and then
        filtered the whole table, so a 10,000-partition write scanned the table 10,000
        times. Measured at 200k rows: 100 partitions 0.13 s, 500 0.48 s, 2,000 1.49 s —
        exactly linear in the partition count. Sorting is O(n log n) *once*.

        The sort is stable, so rows keep their relative order within a partition, as they
        did when each partition was `filter`ed out of the original table.

        Nulls and NaNs need care and are why the run detection is not a plain `equal`:
        `NULL == NULL` is NULL and `NaN == NaN` is False, so either would start a spurious
        run on every row and shatter that partition into one file per row. `group_by`
        placed both in a single group, and so does this.
        """
        import pyarrow.compute as pc

        if table.num_rows == 0:
            return
        ordered = table.take(pc.sort_indices(table, sort_keys=[(c, "ascending") for c in cols]))
        starts = _partition_run_starts(ordered, cols, pc)
        for begin, end in zip(starts, [*starts[1:], ordered.num_rows], strict=True):
            key_vals = [(c, ordered.column(c)[begin].as_py()) for c in cols]
            yield key_vals, ordered.slice(begin, end - begin).drop_columns(cols)

    @abstractmethod
    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        """Write the whole table to an open binary handle in this format."""


def _hive_str(value: Any) -> str:
    """The Hive path segment for a partition `value`, URL-encoded like Spark/Hive.

    A raw value containing ``/`` would spawn a spurious subdirectory (``c=x/y`` reads
    back as ``c=x``), and other reserved characters break directory discovery. pyarrow's
    Hive partitioning URI-decodes segment values on read, so the write must URI-encode
    them (``x/y`` → ``x%2Fy``) for the value to survive the round trip. NULL keeps its
    sentinel unencoded — the reader special-cases that exact string.
    """
    if value is None:
        return _HIVE_NULL
    from urllib.parse import quote

    return quote(str(value), safe="")


def _safe_size(fs: Any, path: str) -> int:
    try:
        return fs.size(path)
    except (OSError, ValueError):
        return 0
