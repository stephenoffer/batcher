"""`FileSink` — the Template-Method base every file-format writer subclasses.

Owns filesystem resolution, atomic writes, Hive partitioning, the per-file manifest,
and the parallel-friendly `write_partitioned` (one worker calls it for its shard), so
a concrete format is a tiny subclass overriding only `_write_file`.

The `Sink` protocol itself lives in `io.sink`; this base structurally satisfies it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import IO, Any, ClassVar

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.io.filesystem import resolve_filesystem
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = ["FileSink"]

_HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"


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

    __slots__ = ()

    def write(self, table: pa.Table, path: str, *, resume: bool = False) -> WrittenFile:
        """Write the whole table to a single file at `path`, atomically.

        The bytes become visible at `path` only once the write completes — a crash
        mid-write leaves any prior file intact (no truncated output), closing Ray
        Data's overwrite data-loss (ray#62019). Local writes go via a temp file +
        atomic rename; object stores write directly (a single PUT is already atomic).

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
        fs = resolve_filesystem(path)
        if resume and fs.exists(path):
            return WrittenFile(path=path, rows=table.num_rows, bytes=_safe_size(fs, path))
        with fs.atomic_writer(path) as fh:
            self._write_file(table, fh)
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

        fs = resolve_filesystem(path)
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
            elif (writer := self._open_stream_writer(fh, first.schema)) is None:
                table = pa.Table.from_batches(list(chain([first], it)))
                self._write_file(table, fh)
                rows = table.num_rows
            else:
                for batch in chain([first], it):
                    if batch.num_rows:
                        self._write_batch(writer, batch)
                        rows += batch.num_rows
                self._close_stream_writer(writer)
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
        fs = resolve_filesystem(path)
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

    def commit(self, manifest: WriteManifest, path: str) -> None:  # noqa: B027 (intentional no-op default)
        """Finalize a write — a no-op here, since file sinks publish on write.

        Transactional (lakehouse) sinks override this to commit the manifest's files
        atomically into a transaction log.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSink, WriteManifest
                >>> ParquetSink().commit(WriteManifest(), "out")

        Args:
            manifest: Every file the write produced, merged across shards.
            path: The write's destination root.
        """

    @staticmethod
    def _hive_partition(
        table: pa.Table, cols: list[str]
    ) -> Iterator[tuple[list[tuple[str, Any]], pa.Table]]:
        """Yield `(key_values, sub_table)` per distinct partition-key combo.

        Vectorized: distinct combos via `group_by`, each group selected with a
        compute mask — no per-row Python.
        """
        import pyarrow.compute as pc

        keys = table.group_by(cols).aggregate([])
        for i in range(keys.num_rows):
            key_vals = [(c, keys.column(c)[i].as_py()) for c in cols]
            mask: Any = None
            for c, v in key_vals:
                col = table.column(c)
                # `col == NULL` is NULL for every row, not True, so a NULL partition key
                # would select zero rows and silently drop them (they land under
                # `__HIVE_DEFAULT_PARTITION__` with an empty file). Match nulls with
                # `is_null` so the NULL group keeps its rows. `NaN == NaN` is likewise
                # False for every row, so a NaN partition key (a float column with a NaN)
                # would select zero rows and drop them too — `group_by` puts NaN in its own
                # group, but `equal` can never re-select it. Match NaN with `is_nan`.
                if v is None:
                    eq = pc.is_null(col)
                elif isinstance(v, float) and v != v:  # NaN
                    eq = pc.is_nan(col)
                else:
                    eq = pc.equal(col, pa.scalar(v, table.schema.field(c).type))
                mask = eq if mask is None else pc.and_(mask, eq)
            yield key_vals, table.filter(mask).drop_columns(cols)

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
