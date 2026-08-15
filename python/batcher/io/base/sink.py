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

from batcher._internal.hardware import available_cpu_count, machine_memory_bytes
from batcher.io._backend import _scheme
from batcher.io.base._hive import (
    hive_partition_run_starts,
    hive_path_segment,
    warn_high_cardinality_partitioning,
)
from batcher.io.base._paths import normalize_path
from batcher.io.base._transient import with_retry
from batcher.io.filesystem import FileSystem, resolve_filesystem
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.plan.types import total_retained_bytes

__all__ = ["FileSink", "stream_part_concurrency"]


# How many files a write to an **object store** publishes at once. A PUT is tens of
# milliseconds of latency, so throughput tracks requests in flight, not cores available to
# encode them — the same asymmetry the read path already sizes for
# (`FileSource._REMOTE_READ_CONCURRENCY`). Bounded by the core count, a four-core worker
# published four of a 200-partition write at a time and spent the rest of its life blocked
# on sockets. Local disk keeps the core-count sizing: an NVMe write is bandwidth-bound, so
# oversubscribing it buys nothing and costs resident encoded buffers.
_REMOTE_WRITE_CONCURRENCY = max(2, int(os.environ.get("BATCHER_REMOTE_WRITE_CONCURRENCY", "32")))


def _write_concurrency(n_files: int, path: str) -> int:
    """How many of `n_files` outputs to publish at once, sized for where they are going."""
    by_core = available_cpu_count()
    remote = _scheme(path) not in ("", "file")
    return max(1, min(n_files, max(by_core, _REMOTE_WRITE_CONCURRENCY) if remote else by_core))


# Ceiling on the part files a *streaming* write keeps encoding at once, before the memory
# budget narrows it further (`FileSink._stream_part_concurrency`). A streaming write learns
# its file count only as it goes, so it has no `n_files` to size the pool from the way the
# collect path does — this stands in for it, and is the cores/requests sizing above once
# the parts are small enough for the budget not to bind.
_MAX_STREAM_PARTS_IN_FLIGHT = 64

# The share of machine memory a streaming write may hold in un-encoded parts. An eighth
# leaves the writer's real contract intact — resident bytes are a fixed multiple of one
# part, never a function of the result size — while being enough for the part sizes a row
# cap usually names.
_STREAM_PART_MEMORY_SHARE = 8

# Write-side retry, mirroring the read path's `_READ_RETRY_*` so a deployment tunes one
# idea rather than two. See `FileSink.write` for why retrying a write is safe.
_WRITE_RETRY_ATTEMPTS = max(1, int(os.environ.get("BATCHER_WRITE_RETRY_ATTEMPTS", "3")))
_WRITE_RETRY_BACKOFF_S = max(0.0, float(os.environ.get("BATCHER_WRITE_RETRY_BACKOFF_S", "0.5")))


def stream_part_concurrency(part_bytes: int, directory: str) -> int:
    """How many output files of `part_bytes` each to keep in flight, given where they go.

    Two bounds, and which one binds depends on the destination and the part size. An object
    store is latency-bound, so throughput tracks requests in flight rather than cores; local
    disk is bandwidth-bound, so oversubscribing it buys nothing and costs resident buffers.
    Either way the parts in flight are held live while their encoders run, so a share of
    machine memory caps the count for a large part.

    Shared by both streaming writers — `FileSink.write_stream_parts` and the training-shard
    writer — because they are making the same decision about the same storage. Two copies of
    this sizing would drift, and the one that drifted low would quietly serialize a write.

    Args:
        part_bytes: What one in-flight part pins in memory.
        directory: The destination, which decides whether concurrency is bounded by cores
            (local disk) or by requests in flight (an object store).

    Returns:
        The number of parts to write concurrently, at least 1.
    """
    by_place = _write_concurrency(_MAX_STREAM_PARTS_IN_FLIGHT, directory)
    if part_bytes <= 0:
        return 1
    budget = machine_memory_bytes() // _STREAM_PART_MEMORY_SHARE
    return max(1, min(by_place, budget // part_bytes))


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

    def write_stream_shard(
        self,
        batches: Iterator[pa.RecordBatch],
        directory: str,
        *,
        file_index: int,
        schema: pa.Schema | None = None,
        resume: bool = False,
    ) -> WrittenFile:
        """Stream one distributed shard into ``<directory>/part-{file_index}<suffix>``.

        The streaming counterpart of an unpartitioned, uncapped `write_partitioned`: same
        file, same name, but the shard's rows are consumed one batch at a time instead of
        being handed over as a table. A distributed write worker uses this so its memory is
        one batch rather than its whole share of the result.

        The name matches `write_partitioned`'s exactly — not `write_stream_parts`'
        ``part-{index}-{chunk}`` — because a shard that produces one file must be named the
        same way whether it streamed or materialized, or `resume` cannot recognize the work
        a previous run finished.

        Examples:
            .. doctest::

                >>> import tempfile
                >>> import pyarrow as pa
                >>> from batcher.io import ParquetSink
                >>> batches = pa.table({"x": [1, 2, 3]}).to_batches()
                >>> written = ParquetSink().write_stream_shard(
                ...     iter(batches), tempfile.mkdtemp(), file_index=3
                ... )
                >>> written.path.endswith("part-00003.parquet"), written.rows
                (True, 3)

        Args:
            batches: This shard's batches, consumed one at a time.
            directory: Destination directory URI, created if absent.
            file_index: This shard's index, so concurrent shards never collide.
            schema: Schema used to write a valid empty file when `batches` yields nothing.
            resume: Leave an already-present (hence complete) file untouched.

        Returns:
            The file this shard wrote, with its row count and size on storage.
        """
        directory = self._dest(directory)
        self._resolve(directory).mkdirs(directory, exist_ok=True)
        name = f"{directory.rstrip('/')}/part-{file_index:05d}{self.suffix}"
        return self.write_stream(batches, name, schema=schema, resume=resume)

    def write_stream_parts(
        self,
        batches: Iterator[pa.RecordBatch],
        directory: str,
        *,
        max_rows_per_file: int,
        schema: pa.Schema | None = None,
        file_index: int = 0,
        resume: bool = False,
    ) -> list[WrittenFile]:
        """Stream `batches` into a directory of files, each capped at `max_rows_per_file`.

        The bounded-memory form of a row-capped write. `write_partitioned` needs the whole
        table resident before it can slice it, so asking for a file size used to *cost* a
        full materialization on the driver — exactly backwards, since a caller who caps the
        file size is usually the caller whose result does not fit. Here the cap is a
        rollover point instead: the writer closes the current file and opens the next one
        when it fills, so memory stays at one batch no matter how large the output is.

        Files are named ``part-{file_index}-{chunk}`` throughout, including when only one
        is produced, because a stream cannot know it is the last one until it has already
        been written. Readers glob the directory, so the name is not load-bearing; what
        matters is that it is deterministic, which is what lets `resume` skip a finished
        file on a re-run.

        Examples:
            .. doctest::

                >>> import tempfile
                >>> import pyarrow as pa
                >>> from batcher.io import ParquetSink
                >>> batches = pa.table({"x": list(range(5))}).to_batches(max_chunksize=2)
                >>> written = ParquetSink().write_stream_parts(
                ...     iter(batches), tempfile.mkdtemp(), max_rows_per_file=2
                ... )
                >>> [f.rows for f in written]
                [2, 2, 1]

        Args:
            batches: The batches to persist, consumed one at a time.
            directory: Destination directory URI.
            max_rows_per_file: Cap on the rows in each output file.
            schema: Schema used to write a valid empty file when `batches` yields nothing.
            file_index: This shard's index, so concurrent shards never collide.
            resume: Skip any (atomically written, hence complete) file that exists.

        Returns:
            One entry per file written, in the order they were written.
        """
        directory = self._dest(directory)
        fs = self._resolve(directory)
        fs.mkdirs(directory, exist_ok=True)
        stream = _RollingStream(iter(batches))
        written: list[WrittenFile] = []
        pending: list[Any] = []
        # Encode the parts CONCURRENTLY. Reading the stream stays serial — one iterator, and
        # a part's rows must be the ones that follow the previous part's — but the encode is
        # where the time goes, and it was running on one core. Measured on 4M rows into 8
        # Parquet parts: the read is 128 ms, a single-threaded encode of the same rows is
        # 1,155 ms, and the whole write took 1,348 ms. The collect path already fans its
        # parts across a pool and finished the identical write in 312 ms, so streaming cost
        # 4.3x for the bounded memory it bought — a bad trade when the bound can be kept.
        #
        # It is kept by `_stream_part_concurrency`: a part is materialized, handed to the
        # pool, and the next one is read while it encodes, with the number in flight sized
        # from the first part's measured size so resident bytes stay bounded no matter how
        # large `max_rows_per_file` is.
        pool: ThreadPoolExecutor | None = None
        limit = 0
        try:
            # `not (written or pending)` is the "an empty stream still writes one empty
            # file" case. It must count the parts still *encoding* as well as the finished
            # ones, or an exhausted stream whose parts are all in flight looks like a
            # stream that produced nothing and the loop appends an empty file per turn.
            while (first := stream.peek()) is not None or not (written or pending):
                index = len(written) + len(pending)
                name = f"{directory}/part-{file_index:05d}-{index:05d}{self.suffix}"
                if resume and fs.exists(name):
                    # Drain this file's rows rather than letting `write_stream` skip the file
                    # without consuming them: leaving them in the stream would slide every
                    # later row into the wrong part file, so a resumed write would silently
                    # duplicate the skipped rows and drop an equal number at the end.
                    rows = sum(b.num_rows for b in stream.take(max_rows_per_file))
                    _drain(pending, written)
                    written.append(WrittenFile(path=name, rows=rows, bytes=_safe_size(fs, name)))
                    continue
                part = list(stream.take(max_rows_per_file))
                part_schema = schema if first is None else first.schema
                if pool is None:
                    limit = self._stream_part_concurrency(part, directory)
                    if limit > 1:
                        pool = ThreadPoolExecutor(max_workers=limit)
                if pool is None:
                    written.append(self.write_stream(iter(part), name, schema=part_schema))
                    continue
                pending.append(pool.submit(self.write_stream, iter(part), name, schema=part_schema))
                # Bound the parts resident at once: block on the oldest as soon as `limit`
                # are outstanding, so peak memory is `limit` parts and not the whole result.
                if len(pending) >= limit:
                    written.append(pending.pop(0).result())
            _drain(pending, written)
        finally:
            if pool is not None:
                # Cancel rather than wait on a failure: an exception above leaves parts
                # queued whose output nobody will read, and a `with` block would make the
                # caller wait for every one of them before seeing the error.
                pool.shutdown(wait=False, cancel_futures=True)
        return written

    def _stream_part_concurrency(self, part: list[pa.RecordBatch], directory: str) -> int:
        """How many part files to keep encoding at once, from the first part's real size.

        The row cap is the caller's, so a part can be anything from a few KB to several GB
        and the count cannot be a constant. Sizing it from the measured part keeps the
        resident bytes bounded — which is the property `write_stream_parts` exists for —
        while still using the cores a large write needs.

        The budget is an eighth of the machine's (cgroup-aware) memory. It is a share
        rather than a limit because this writer's contract is to be O(1) in the *result*
        size, not to fit in a particular envelope: whatever the budget, the parts in flight
        are a fixed number and the result may be a thousand times larger.

        Args:
            part: The first part's batches, used to measure a part's memory cost.
            directory: The destination, which decides whether concurrency is bounded by
                cores (local disk) or by requests in flight (an object store).

        Returns:
            The number of parts to encode concurrently, at least 1.
        """
        # What a part *pins*, not what it addresses: the parts in flight are held live
        # while their encoders run, and a part cut zero-copy out of a larger table keeps
        # the whole parent resident. `nbytes` also raises outright on the Arrow view
        # layouts, which would turn a memory estimate into a failed write.
        return stream_part_concurrency(total_retained_bytes(part), directory)

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
        warn_high_cardinality_partitioning(len(parts), partition_by, path)
        if not parts:
            # No rows means no partition values, so there are no `col=v` directories to
            # write — and writing *nothing* leaves the destination absent, where every
            # other write shape leaves a readable empty relation. A downstream read then
            # fails with "path does not exist" rather than returning no rows, which is a
            # difference the caller never asked for (a filter that matched nothing is not
            # an error). One empty part at the root, keeping the partition columns since
            # nothing was moved into a path.
            fs.mkdirs(path, exist_ok=True)
            return self._write_parts(table, path, file_index, resume, max_rows_per_file)

        def _write_partition(item: tuple[list[tuple[str, Any]], pa.Table]) -> list[WrittenFile]:
            key_vals, sub = item
            sub_dir = "/".join([path, *(f"{c}={hive_path_segment(v)}" for c, v in key_vals)])
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
        workers = _write_concurrency(len(parts), path)
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
        workers = _write_concurrency(len(chunks), directory)
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
        starts = hive_partition_run_starts(ordered, cols, pc)
        for begin, end in zip(starts, [*starts[1:], ordered.num_rows], strict=True):
            key_vals = [(c, ordered.column(c)[begin].as_py()) for c in cols]
            yield key_vals, ordered.slice(begin, end - begin).drop_columns(cols)

    @abstractmethod
    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        """Write the whole table to an open binary handle in this format."""


class _RollingStream:
    """A batch iterator that can be consumed in row-bounded runs, with one push-back slot.

    `take(n)` yields at most `n` rows, slicing the batch that straddles the boundary and
    holding its tail for the next run. That is what makes a row-capped write streamable:
    each run is handed to `write_stream` as its own lazy iterator, so a file's worth of
    rows is never buffered — only the one batch currently in flight.
    """

    __slots__ = ("_it", "_pending")

    def __init__(self, it: Iterator[pa.RecordBatch]) -> None:
        self._it = it
        self._pending: pa.RecordBatch | None = None

    def peek(self) -> pa.RecordBatch | None:
        """The next non-empty batch without consuming it, or None at end of stream."""
        while self._pending is None or self._pending.num_rows == 0:
            nxt = next(self._it, None)
            if nxt is None:
                return None
            self._pending = nxt
        return self._pending

    def take(self, rows: int) -> Iterator[pa.RecordBatch]:
        """Yield batches totalling at most `rows` rows, splitting one if it straddles."""
        remaining = rows
        while remaining > 0 and (batch := self.peek()) is not None:
            if batch.num_rows <= remaining:
                self._pending = None
                remaining -= batch.num_rows
                yield batch
            else:
                self._pending = batch.slice(remaining)
                yield batch.slice(0, remaining)
                return


def _drain(pending: list[Any], written: list[WrittenFile]) -> None:
    """Move every finished part future into `written`, in submission order.

    Order is what makes the manifest deterministic, and `resume` depends on that: a re-run
    must map the same rows onto the same ``part-NNNNN`` name. The futures complete in
    whatever order the encodes finish, so they are collected by position, never as-completed.
    """
    while pending:
        written.append(pending.pop(0).result())


def _safe_size(fs: Any, path: str) -> int:
    try:
        return fs.size(path)
    except (OSError, ValueError):
        return 0
