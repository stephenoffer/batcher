"""`ParquetSource` — lazy projection/predicate read of one or more Parquet files.

The workhorse scan: schema and row counts come from the footer (never a data scan),
the pushed predicate becomes a pyarrow filter (row-group + page pruning), and splits
go down to row-group granularity so a distributed read slices *within* a file.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES
from batcher.io.formats.structured import _parquet_native
from batcher.io.splits import Split, parquet_row_group_splits
from batcher.plan.source_stats import SourceStatistics

__all__ = ["ParquetSource"]

# Process-wide cache of per-file Parquet row counts, keyed by full path/URI. A footer
# read is a ~80 ms object-store round trip; Parquet is write-once, so a file's row count
# is immutable and safe to cache by path — the same "never read a footer twice" guarantee
# the Rust reader's `meta_cache` gives the data plane. Without it, EVERY distributed
# `collect` re-reads every source file's footer just to size the worker fan-out
# (`learned_num_workers` → `total_source_rows`): measured ~0.9 s/collect on a 10-file sf10
# groupby, dwarfing the 0.26 s shuffle it was sizing.
_ROW_COUNT_CACHE: dict[str, int] = {}


@SOURCES.register("parquet")
class ParquetSource(FileSource):
    """One or more Parquet files (single file, directory, or glob).

    Examples:
        .. doctest::

            >>> from batcher.io import ParquetSource  # doctest: +SKIP
            >>> src = ParquetSource("s3://bucket/lineitem/*.parquet")  # doctest: +SKIP
            >>> src.row_count()  # from the footers, no data scan  # doctest: +SKIP
            600037902
    """

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
        """Read every file concurrently, pruning row-groups with `predicate`.

        A filter the reader cannot bind never fails the query: the read falls back to
        an unfiltered one, and the engine's `Filter` operator still produces the right
        rows — pushdown is a pure I/O optimization.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> src = ParquetSource("s3://bucket/lineitem/")  # doctest: +SKIP
                >>> batches = src.read(["l_orderkey"])  # doctest: +SKIP
                >>> batches[0].num_columns  # doctest: +SKIP
                1

        Args:
            projection: Columns to read. All columns when omitted; only these
                column chunks are fetched and decoded.
            predicate: A predicate-IR filter, translated to a pyarrow filter for
                row-group and page pruning.

        Returns:
            Every batch of every file, in file order.
        """
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
        """Stream the files with row-group pruning, in bounded memory.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> src = ParquetSource("s3://bucket/lineitem/")  # doctest: +SKIP
                >>> next(src.iter_batches(["l_orderkey"])).num_rows  # doctest: +SKIP
                16384

        Args:
            projection: Columns to read. All columns when omitted.
            predicate: A predicate-IR filter. On any failure to bind it (a remote
                filesystem the dataset API can't resolve here) the stream falls
                back to unfiltered — the engine re-filters.

        Returns:
            An iterator over the surviving batches, in file order.
        """
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
        return parquet_row_group_splits(path, target_size)

    def statistics(self) -> SourceStatistics | None:
        """Footer-derived row count + per-column min/max/null, no data scan.

        This is what lets Kyber prune partitions and answer an unfiltered
        ``MIN``/``MAX`` before a single row is read.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> ParquetSource("s3://bucket/lineitem/").statistics().row_count  # doctest: +SKIP
                600037902

        Returns:
            The source's row count and per-column bounds, or None if the footers
            cannot be read.
        """
        from batcher.io.stats import parquet_statistics

        try:
            return parquet_statistics(self._fs, self._files(), self.schema())
        except Exception:
            return None
