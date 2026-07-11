"""Parquet — lazy projection/predicate read + write, plus the dataset reader.

`ParquetSource` reads one or more Parquet files with row-group-level splits.
`ParquetDatasetSource` is the PB-scale workhorse: a `pyarrow.dataset` over a
(possibly Hive-partitioned) directory tree, recovering partition columns from the
path and supporting partition + row-group pruning.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Any

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.io.base import FileSink, FileSource, _parquet_row_group_splits
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.structured import _parquet_native
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = ["ParquetDatasetSource", "ParquetFragmentSplit", "ParquetSink", "ParquetSource"]

# Process-wide cache of per-file Parquet row counts, keyed by full path/URI. A footer
# read is a ~80 ms object-store round trip; Parquet is write-once, so a file's row count
# is immutable and safe to cache by path — the same "never read a footer twice" guarantee
# the Rust reader's `meta_cache` gives the data plane. Without it, EVERY distributed
# `collect` re-reads every source file's footer just to size the worker fan-out
# (`learned_num_workers` → `total_source_rows`): measured ~0.9 s/collect on a 10-file sf10
# groupby, dwarfing the 0.26 s shuffle it was sizing.
_ROW_COUNT_CACHE: dict[str, int] = {}


@dataclass(frozen=True, slots=True)
class ParquetFragmentSplit:
    """One file of a partitioned Parquet dataset, read independently on a worker.

    Carries only locators (dataset root + partitioning + the fragment's path), so
    a worker reads just this file and recovers partition columns from the dataset
    schema — the whole dataset never materializes on the driver. Projection +
    predicate are pushed into the per-fragment read.
    """

    root: str
    partitioning: str
    file_path: str

    def _table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        import pyarrow.dataset as pads

        from batcher.io.splits import fragment_index

        # List the dataset once per worker (cached), then O(1) lookup — never
        # re-list per read (which would be O(files^2) over a per-file split).
        dataset, index = fragment_index(
            ("parquet", self.root, self.partitioning),
            lambda: pads.dataset(self.root, format="parquet", partitioning=self.partitioning),
        )
        flt = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            flt = to_pyarrow_expression(predicate)
        frag = index.get(self.file_path)
        if frag is not None:
            return frag.to_table(schema=dataset.schema, columns=projection, filter=flt)
        empty = dataset.schema.empty_table()
        return empty.select(projection) if projection is not None else empty

    def schema(self) -> pa.Schema:
        import pyarrow.dataset as pads

        return pads.dataset(self.root, format="parquet", partitioning=self.partitioning).schema

    def row_count(self) -> int | None:
        """Exact rows from this fragment's footer (no data scan)."""
        import pyarrow.parquet as pq

        from batcher.io.filesystem import resolve_filesystem

        try:
            fs = resolve_filesystem(self.file_path)
            with fs.open(self.file_path) as fh:
                return pq.ParquetFile(fh).metadata.num_rows
        except Exception:
            return None

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection, predicate).to_batches()

    def identity(self) -> str:
        return f"parquet_dataset:{self.root}:{self.file_path}"


@SOURCES.register("parquet")
class ParquetSource(FileSource):
    """One or more Parquet files (single file, directory, or glob)."""

    suffix = ".parquet"
    format_name = "parquet"
    # Predicate pushdown: Kyber's pushed predicate → pyarrow `filters`, giving
    # row-group + page pruning via the footer statistics.
    supports_predicate = True

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        import pyarrow.parquet as pq

        return pq.read_schema(fh)

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        import pyarrow.parquet as pq

        return pq.read_table(fh, columns=projection).to_batches()

    def _native_read_many(self, projection: list[str] | None) -> list[pa.RecordBatch] | None:
        """All files in one batched native pass (see `_parquet_native.read_many`), or ``None``.

        The unfiltered multi-file read: one native call overlaps every file's I/O, beating a
        per-file thread pool on a many-small-files scan. ``None`` (use base's per-file path)
        for a single file, in non-strict mode (base owns normalization), or with a byte cache.
        """
        files = self._files()
        if len(files) <= 1 or self._schema_mode != "strict":
            return None
        if self._fs.native_read_target(files[0]) is None:
            return None
        per_file = _parquet_native.read_many(files, projection)
        if per_file is None:
            return None
        return [b for file_batches in per_file for b in file_batches]

    def _read_by_path(self, path: str, projection: list[str] | None) -> list[pa.RecordBatch] | None:
        """The unfiltered read: native Rust reader when possible, else PyArrow.

        ``None`` (fall to the handle path) only when a read-through byte cache is active —
        the native reader and PyArrow's `filesystem=` read both bypass that cache, so its
        reads must go through `open`. Otherwise try the native Rust reader (3-4x faster on
        object storage) and fall back to PyArrow's native-filesystem read on anything it
        cannot handle.
        """
        if self._fs.native_read_target(path) is None:
            return None
        native = _parquet_native.read_one(path, projection)
        if native is not None:
            return native
        return self._read_table(path, projection).to_batches()

    def _read_table(
        self, path: str, projection: list[str] | None, pa_filter: Any = None
    ) -> pa.Table:
        """Read `path`, letting pyarrow do its own I/O when the backend allows it.

        Handed a Python file object, pyarrow's reader round-trips every read through the
        interpreter, and that serializes the decode threads it fans across column chunks
        — so the read gets *superlinearly* slower as the projection widens. Measured on
        TPC-H sf100 `lineitem` (one 16 GB file, 600 M rows): one column reads in 648 ms
        either way, four columns in 2,831 ms through a handle against 1,653 ms when
        pyarrow owns the I/O. Backends that cannot hand over a native target (fsspec
        behind a read-through cache) keep the handle; the result is identical.
        """
        import pyarrow.parquet as pq

        from batcher.io.filesystem import ensure_io_threads

        ensure_io_threads()  # lift the 8-thread IO cap so a wide S3 read isn't throttled
        target = self._fs.native_read_target(path)
        if target is not None:
            fs, in_path = target
            return pq.read_table(
                in_path, filesystem=fs, columns=projection, filters=pa_filter, pre_buffer=True
            )
        with self._fs.open(path) as fh:
            return pq.read_table(fh, columns=projection, filters=pa_filter)

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        import pyarrow.parquet as pq

        from batcher.io.filesystem import ensure_io_threads

        ensure_io_threads()
        # Prefer the native `(fs, path)` target with column-chunk pre-buffering: reading
        # through a Python handle serializes pyarrow's per-column decode threads (the same
        # anti-pattern `_read_table` avoids), and no `pre_buffer` leaves scattered column
        # chunks as many small GETs. Backends with no native target keep the handle.
        target = self._fs.native_read_target(path)
        if target is not None:
            fs, in_path = target
            pf = pq.ParquetFile(in_path, filesystem=fs, pre_buffer=True)
            yield from pf.iter_batches(columns=projection)
        else:
            with self._fs.open(path) as fh:
                yield from pq.ParquetFile(fh).iter_batches(columns=projection)

    @staticmethod
    def _pa_filter(predicate: dict | None) -> Any:
        if predicate is None:
            return None
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        pa_filter = self._pa_filter(predicate)
        if pa_filter is None:
            batched = self._native_read_many(projection)
            return batched if batched is not None else super().read(projection)
        files = self._files()

        def _read_one(f: str) -> list[pa.RecordBatch]:
            return self._read_table(f, projection, pa_filter).to_batches()

        try:
            if len(files) <= 1:
                return _read_one(files[0]) if files else []
            # Read files concurrently: Parquet decode + filtering run in the C++ layer
            # with the GIL released, so a 100-file source no longer opens and filters one
            # file at a time (the serial loop left 95 of 96 cores idle). Order preserved.
            from concurrent.futures import ThreadPoolExecutor

            workers = min(len(files), available_cpu_count() * 2)
            out: list[pa.RecordBatch] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for batches in pool.map(_read_one, files):  # order preserved
                    out.extend(batches)
            return out
        except Exception:
            # A filter the reader can't bind (e.g. a type it lacks a kernel for) must
            # never fail the query — the engine keeps the Filter operator, so an
            # unfiltered read is always correct, just reads more rows.
            return super().read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        pa_filter = self._pa_filter(predicate)
        if pa_filter is None:
            yield from super().iter_batches(projection)
            return
        # Stream with row-group pruning + filtering via pyarrow.dataset over the
        # local files; on any failure (e.g. a remote filesystem the dataset can't
        # resolve here) fall back to unfiltered streaming — the engine re-filters.
        try:
            import pyarrow.dataset as pads

            dataset = pads.dataset(self._files(), format="parquet")
        except Exception:
            yield from super().iter_batches(projection)
            return
        yield from dataset.to_batches(columns=projection, filter=pa_filter)

    def _file_row_count(self, path: str) -> int | None:
        import pyarrow.parquet as pq

        hit = _ROW_COUNT_CACHE.get(path)
        if hit is not None:
            return hit
        with self._fs.open(path) as fh:
            n = pq.ParquetFile(fh).metadata.num_rows
        _ROW_COUNT_CACHE[path] = n
        return n

    def _file_splits(self, path: str, target_size: int | None) -> list[Split]:
        return _parquet_row_group_splits(path, target_size)

    def statistics(self) -> SourceStatistics | None:
        """Footer-derived row count + per-column min/max/null, no data scan."""
        from batcher.io.stats import parquet_statistics

        try:
            return parquet_statistics(self._fs, self._files(), self.schema())
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class PartitionDirSplit:
    """One top-level partition directory of a Hive dataset, listed on the worker.

    Distributed listing: the driver enumerates only the top-level ``col=val`` dirs
    (one cheap, non-recursive list); each worker then lists *only its own subtree*
    and reads it, so the per-worker file-listing cost is O(subtree), not O(whole
    dataset). The top-level partition value lives in the dir name (not the data
    files), so it is appended, typed via the dataset schema carried on the split.
    """

    subdir: str
    partitioning: str
    part_name: str
    part_value: str
    dataset_schema: pa.Schema

    def _typed_value(self) -> Any:
        if self.part_value == "__HIVE_DEFAULT_PARTITION__":
            return None
        target = self.dataset_schema.field(self.part_name).type
        return pa.scalar(self.part_value, pa.string()).cast(target).as_py()

    def _table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        import pyarrow.dataset as pads

        from batcher.io.splits import fragment_index

        # List only this partition subtree (cached per worker), not the whole dataset.
        dataset, _index = fragment_index(
            ("pq_subdir", self.subdir, self.partitioning),
            lambda: pads.dataset(self.subdir, format="parquet", partitioning=self.partitioning),
        )
        want = list(projection) if projection is not None else list(self.dataset_schema.names)
        data_cols = [c for c in want if c != self.part_name]
        flt = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            flt = to_pyarrow_expression(predicate)
        # Push the filter into the subtree read (row-group pruning) when it doesn't
        # reference the top-level partition column the sub-dataset lacks.
        prefiltered = False
        try:
            table = dataset.to_table(columns=data_cols, filter=flt)
            prefiltered = flt is not None
        except Exception:
            table = dataset.to_table(columns=data_cols)
        if self.part_name in want:
            value = self._typed_value()
            target = self.dataset_schema.field(self.part_name).type
            table = table.append_column(
                self.part_name, pa.array([value] * table.num_rows, type=target)
            )
        table = table.select(want)
        if flt is not None and not prefiltered:
            table = table.filter(flt)
        return table

    def schema(self) -> pa.Schema:
        return self.dataset_schema

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection, predicate).to_batches()

    def row_count(self) -> int | None:
        """Exact rows in this partition subtree from footers (no data scan)."""
        import pyarrow.dataset as pads

        try:
            return pads.dataset(
                self.subdir, format="parquet", partitioning=self.partitioning
            ).count_rows()
        except Exception:
            return None

    def identity(self) -> str:
        return f"parquet_dataset:{self.subdir}"


def _hive_segment(name: str) -> tuple[str, str] | None:
    """Parse a ``col=val`` directory basename, or None if it isn't one."""
    base = name.rstrip("/").rsplit("/", 1)[-1]
    if "=" not in base:
        return None
    col, _, val = base.partition("=")
    return (col, val) if col else None


@SOURCES.register("parquet_dataset")
class ParquetDatasetSource:
    """A Hive-partitioned Parquet dataset read via `pyarrow.dataset`.

    Recursively discovers files under `path`, recovers partition columns from the
    directory layout (``col=val/``), and applies partition + row-group pruning.
    This reads the directories `write.parquet(..., partition_by=...)` produces.
    `splits()` emits one `ParquetFragmentSplit` per data file, so a distributed
    read fans the files across workers and never materializes the whole dataset on
    the driver.
    """

    # Predicate pushdown: Kyber's pushed predicate is translated to a pyarrow
    # dataset filter (partition + row-group + page pruning).
    supports_predicate = True

    __slots__ = ("_partitioning", "_path", "_schema_cache")

    def __init__(self, path: str, *, partitioning: str = "hive") -> None:
        self._path = path
        self._partitioning = partitioning
        self._schema_cache: pa.Schema | None = None

    def _dataset(self) -> Any:
        import pyarrow.dataset as ds

        return ds.dataset(self._path, format="parquet", partitioning=self._partitioning)

    @staticmethod
    def _pa_filter(predicate: dict | None) -> Any:
        if predicate is None:
            return None
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate)

    def schema(self) -> pa.Schema:
        if self._schema_cache is None:
            self._schema_cache = self._dataset().schema
        return self._schema_cache

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        table = self._dataset().to_table(columns=projection, filter=self._pa_filter(predicate))
        return table.to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._dataset().to_batches(columns=projection, filter=self._pa_filter(predicate))

    def row_count(self) -> int | None:
        return self._dataset().count_rows()

    def identity(self) -> str:
        return f"parquet_dataset:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """Distributed-listing splits — never list the whole tree on the driver.

        For a Hive-partitioned directory, the driver enumerates only the top-level
        ``col=val`` dirs (one cheap list) and emits one `PartitionDirSplit` per dir;
        each worker lists only its own subtree. For a non-partitioned dataset it
        falls back to per-file `ParquetFragmentSplit`s.
        """
        from batcher.io.filesystem import resolve_filesystem

        if not any(ch in self._path for ch in "*?["):
            try:
                dirs = resolve_filesystem(self._path).list_dirs(self._path)
            except Exception:
                dirs = []
            partition_dirs = [(d, seg) for d in dirs if (seg := _hive_segment(d)) is not None]
            if partition_dirs:
                schema = self.schema()
                return [
                    PartitionDirSplit(d, self._partitioning, name, value, schema)
                    for d, (name, value) in partition_dirs
                ]
        # Flat dataset (or non-listable): per-file splits read each file directly.
        try:
            paths = [frag.path for frag in self._dataset().get_fragments()]
        except Exception:
            return [WholeSourceSplit(self)]
        if not paths:
            return [WholeSourceSplit(self)]
        return [ParquetFragmentSplit(self._path, self._partitioning, p) for p in paths]


@SINKS.register("parquet")
class ParquetSink(FileSink):
    """Write a Parquet file."""

    suffix = ".parquet"
    format_name = "parquet"

    __slots__ = ("compression",)

    def __init__(self, compression: str = "zstd") -> None:
        self.compression = compression

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        import pyarrow.parquet as pq

        pq.write_table(table, fh, compression=self.compression)

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        import pyarrow.parquet as pq

        return pq.ParquetWriter(fh, schema, compression=self.compression)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write_batch(batch)

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
