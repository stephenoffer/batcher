"""Template-Method base classes for file-backed sources and sinks.

`FileSource` centralizes everything shared across file formats — path/glob/
filesystem resolution, schema caching, multi-file concatenation, projection
plumbing, streaming, and split generation — so a concrete format is a tiny
subclass overriding only its per-file read primitives. `FileSink` does the same
for writers. This is the shared-code spine that keeps each `io/formats/<fmt>.py`
small (the v2 antidote to v1's duplicated, mixin-heavy readers).

The `Source`/`Sink` protocols themselves live in `io.source`/`io.sink`; these
bases structurally satisfy them. `Split` lives in `io.splits`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import replace
from typing import IO, Any, ClassVar

import pyarrow as pa

from batcher.io.filesystem import resolve_filesystem
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.io.splits import FileSplit, RowGroupSplit, Split

__all__ = ["FileSink", "FileSource", "pack_row_groups"]


def pack_row_groups(
    num_row_groups: int, sizes: list[int], target_bytes: int | None
) -> list[tuple[int, ...]]:
    """Group row-group indices into contiguous runs of roughly `target_bytes`.

    With no target (or unknown sizes) each row-group is its own split — maximum
    parallelism. Otherwise adjacent row-groups are packed until their compressed
    size reaches the target, balancing task count against per-task overhead.
    """
    if target_bytes is None or not sizes:
        return [(i,) for i in range(num_row_groups)]
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    acc = 0
    for i in range(num_row_groups):
        current.append(i)
        acc += sizes[i] if i < len(sizes) else 0
        if acc >= target_bytes:
            runs.append(tuple(current))
            current, acc = [], 0
    if current:
        runs.append(tuple(current))
    return runs


class FileSource(ABC):
    """Base for a lazy, multi-file, projection-aware source over one format.

    Subclasses set `suffix` (for directory/glob expansion) and `format_name` (the
    registry key used to rebuild splits on a worker) and override `_read_schema`
    and `_read_file`. They may override `_iter_file` (streaming), `_file_row_count`
    (cheap counts), and `_file_splits` (sub-file split granularity).
    """

    suffix: ClassVar[str] = ""
    format_name: ClassVar[str] = ""

    __slots__ = ("_files_cache", "_fs", "_path", "_schema_cache", "_schema_mode")

    def __init__(self, path: str, *, schema_mode: str = "strict") -> None:
        self._path = path
        self._fs = resolve_filesystem(path)
        self._files_cache: list[str] | None = None
        self._schema_cache: pa.Schema | None = None
        # "strict" (default) keeps the historical behavior — file 0's schema is
        # assumed for all. "union"/"latest" reconcile differing per-file schemas
        # (`io.schema_evolution`); each file's batches are normalized to the result.
        self._schema_mode = schema_mode

    # ---- shared, do-not-override ------------------------------------------
    def _files(self) -> list[str]:
        if self._files_cache is None:
            self._files_cache = self._fs.expand(self._path, suffix=self.suffix)
        return self._files_cache

    def _file_schema(self, path: str) -> pa.Schema:
        with self._fs.open(path) as fh:
            return self._read_schema(fh)

    def schema(self) -> pa.Schema:
        if self._schema_cache is None:
            files = self._files()
            if self._schema_mode == "strict":
                self._schema_cache = self._file_schema(files[0])
            else:
                from batcher.io.schema import unify_schemas

                self._schema_cache = unify_schemas(
                    [self._file_schema(f) for f in files], self._schema_mode
                )
        return self._schema_cache

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        out: list[pa.RecordBatch] = []
        for f in self._files():
            with self._fs.open(f) as fh:
                out.extend(
                    self._normalize(self._read_file(fh, self._file_proj(f, projection)), projection)
                )
        return out

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for f in self._files():
            yield from self._normalize(
                self._iter_file(f, self._file_proj(f, projection)), projection
            )

    def _file_proj(self, path: str, projection: list[str] | None) -> list[str] | None:
        """The columns to actually request from `path`. In non-strict mode a file may
        lack some unified/projected columns; request only those it has (the rest are
        filled with nulls by `normalize_batch`)."""
        if self._schema_mode == "strict" or projection is None:
            return projection
        present = set(self._file_schema(path).names)
        return [c for c in projection if c in present]

    def _normalize(
        self,
        batches: Iterator[pa.RecordBatch] | list[pa.RecordBatch],
        projection: list[str] | None,
    ) -> Iterator[pa.RecordBatch]:
        """In non-strict mode, reshape each batch to the unified (optionally
        projected) schema — adding missing columns as nulls and casting promoted
        types — so files with differing schemas concatenate cleanly."""
        if self._schema_mode == "strict":
            yield from batches
            return
        from batcher.io.schema import normalize_batch

        target = self.schema()
        if projection is not None:
            target = pa.schema([target.field(c) for c in projection])
        for b in batches:
            yield normalize_batch(b, target)

    def row_count(self) -> int | None:
        counts = [self._file_row_count(f) for f in self._files()]
        return None if any(c is None for c in counts) else sum(counts)  # type: ignore[misc]

    def identity(self) -> str:
        return f"{self.format_name}:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:
        # Per-file splits each reconstruct a single-file reader with no knowledge of
        # the unified schema, so in a non-strict (schema-evolving) read they would
        # skip normalization and produce mismatched batches. Read such a source as a
        # single whole-source split — correct (the unification happens in `read`),
        # at the cost of per-file parallelism for evolving reads.
        if self._schema_mode != "strict":
            from batcher.io.splits import WholeSourceSplit

            return [WholeSourceSplit(self)]
        splits: list[Split] = []
        for f in self._files():
            splits.extend(self._file_splits(f, target_size))
        return splits

    # ---- override points --------------------------------------------------
    @abstractmethod
    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        """Read the schema from an open file handle (no data scan where possible)."""

    @abstractmethod
    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Read one file's batches from an open handle, honoring `projection`."""

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._fs.open(path) as fh:
            yield from self._read_file(fh, projection)

    def _file_row_count(self, path: str) -> int | None:  # noqa: ARG002 (default: unknown)
        return None

    def _file_splits(self, path: str, target_size: int | None) -> list[Split]:  # noqa: ARG002
        return [FileSplit(self.format_name, path)]


_HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"


class FileSink(ABC):
    """Base for a format writer: single-file, partitioned, or one shard of a
    distributed write.

    Subclasses set `suffix`/`format_name` and override `_write_file`. The base
    owns filesystem resolution, Hive partitioning, the per-file manifest, and the
    parallel-friendly `write_partitioned` (one worker calls it for its shard).
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

        With ``resume=True``, a file already present at `path` is left untouched and
        reported as-is: because writes are atomic, an existing file is necessarily a
        fully-committed one, so re-running a crashed job skips the work it finished.
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
        so any format stays correct. Atomic and `resume`-safe like `write`; `schema`
        only writes a valid empty file when the stream yields nothing.
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
        ``max_rows_per_file`` caps each file's row count — splitting a large
        (sub)table into multiple parts so a write never produces one giant file, and
        the cap is honored *per partition* even with `partition_by` (the bug where
        Ray Data ignores ``min_rows_per_file`` alongside ``partition_cols``).
        Returns one `WrittenFile` per file. ``resume=True`` skips any (atomically
        written, hence complete) file that already exists.
        """
        fs = resolve_filesystem(path)
        if not partition_by:
            fs.mkdirs(path, exist_ok=True)
            return self._write_parts(table, path, file_index, resume, max_rows_per_file)
        out: list[WrittenFile] = []
        for key_vals, sub in self._hive_partition(table, partition_by):
            sub_dir = "/".join([path, *(f"{c}={_hive_str(v)}" for c, v in key_vals)])
            fs.mkdirs(sub_dir, exist_ok=True)
            for written in self._write_parts(sub, sub_dir, file_index, resume, max_rows_per_file):
                out.append(replace(written, partition_values=dict(key_vals)))
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
        out: list[WrittenFile] = []
        for chunk_idx, start in enumerate(range(0, table.num_rows, max_rows_per_file)):
            chunk = table.slice(start, max_rows_per_file)
            name = f"{directory}/part-{file_index:05d}-{chunk_idx:05d}{self.suffix}"
            out.append(self.write(chunk, name, resume=resume))
        return out

    def commit(self, manifest: WriteManifest, path: str) -> None:  # noqa: B027 (intentional no-op default)
        """Finalize a write. File sinks make data visible on write, so the base
        is a no-op; transactional (lakehouse) sinks override to commit atomically.
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
                eq = pc.equal(table.column(c), pa.scalar(v, table.schema.field(c).type))
                mask = eq if mask is None else pc.and_(mask, eq)
            yield key_vals, table.filter(mask).drop_columns(cols)

    @abstractmethod
    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        """Write the whole table to an open binary handle in this format."""


def _hive_str(value: Any) -> str:
    return _HIVE_NULL if value is None else str(value)


def _safe_size(fs: Any, path: str) -> int:
    try:
        return fs.size(path)
    except (OSError, ValueError):
        return 0


def _parquet_row_group_splits(path: str, target_size: int | None) -> list[Split]:
    """Build `RowGroupSplit`s for a single Parquet file (used by ParquetSource)."""
    import pyarrow.parquet as pq

    fs = resolve_filesystem(path)
    with fs.open(path) as fh:
        meta = pq.ParquetFile(fh).metadata
        sizes = [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)]
        rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        runs = pack_row_groups(meta.num_row_groups, sizes, target_size)
    # Carry the footer-derived row count so balancing never re-opens the file.
    return [RowGroupSplit(path, run, sum(rows[i] for i in run)) for run in runs]
