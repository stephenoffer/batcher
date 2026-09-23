"""The `ds.meta.storage` accessor — what a scan *would* read, without reading it.

These are questions about the bytes on disk rather than the rows a plan produces: how many
files, how many row groups, how many bytes, what the table is clustered and partitioned by.
A footer or a manifest already knows all of it, and a person deciding whether a query is
affordable needs exactly this and nothing else — "340 files, 12 GB, partitioned by day" is a
sentence you can act on, and it costs one metadata round trip to say.

Nothing here executes. A source that cannot describe itself makes the corresponding total
``None`` rather than a partial sum, because a byte count that silently omits one input is
worse than no byte count at all.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.api.dataset.meta._facts import MetaBase
from batcher.kyber.shortcuts import storage
from batcher.plan.stats import SortOrder

__all__ = ["StorageMeta"]


class StorageMeta(MetaBase):
    """Physical-layout shortcuts, reached as ``ds.meta.storage``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})
            >>> ds.meta.storage.num_sources()
            1
            >>> ds.meta.storage.row_count()
            3
    """

    __slots__ = ()

    def num_sources(self) -> int:
        """How many sources the query scans — one per table or dataset it is bound to.

        Returns:
            The number of bound sources.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.num_sources()
                1
        """
        return len(self._ds._sources)

    def row_count(self) -> int | None:
        """The exact number of rows the sources *hold*, or ``None`` unless every one is exact.

        The scanned row count, not the query's result count — a filter or a join changes the
        latter. ``ds.count()`` answers that.

        Returns:
            The total rows across every source, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).meta.storage.row_count()
                2
        """
        return storage.row_count(self.source_stats())

    def has_exact_row_count(self) -> bool:
        """Whether every source can state its row count without a scan.

        The one fact that decides whether ``ds.count()`` is free or is a query.

        Returns:
            ``True`` if the row count is known exactly from metadata.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.has_exact_row_count()
                True
        """
        return storage.has_exact_row_count(self.source_stats())

    def total_bytes(self) -> int | None:
        """The total size of the sources, in bytes, or ``None`` if any cannot say.

        A file source reports the size its metadata records: for Parquet that is the sum of
        the footers' per-row-group ``total_byte_size``, the column data *before* compression.
        An in-memory relation reports the retained size of its resident Arrow buffers, which
        it alone knows for free —
        without it every consumer sizing from this figure falls back to a coarse
        ``rows x type-width`` guess that under-sizes wide string columns badly.

        Returns:
            The byte size across every source, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.total_bytes() > 0
                True
        """
        return storage.total_bytes(self.source_stats())

    def row_group_count(self) -> int | None:
        """The number of physical blocks (Parquet row groups, ORC stripes), or ``None``.

        The granularity a zone-map prune actually skips at: pruning that removes 90% of the
        rows but still touches every row group saves nothing, and this is the number that
        says so.

        Returns:
            The total block count across every source, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.row_group_count() is None
                True
        """
        return storage.row_group_count(self.source_stats())

    def bytes_per_row(self) -> float | None:
        """The average stored bytes per row, or ``None`` when either total is unknown.

        ``total_bytes() / row_count()``, so it inherits that figure's meaning: for Parquet the
        recorded column-data width per row, for an in-memory relation the Arrow buffers'.
        ``ds.meta.approx.row_bytes()`` is the different question — an estimate of the
        *materialized* Arrow width, measured or type-derived.

        Returns:
            The average stored bytes per row, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.bytes_per_row() > 0
                True
        """
        return storage.bytes_per_row(self.source_stats())

    def partition_keys(self) -> tuple[str, ...]:
        """The partition keys every source agrees on, in order — empty if they disagree.

        A key only one input is partitioned by cannot prune the query, so the shared prefix is
        the honest answer.

        Returns:
            The common partition keys.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.partition_keys()
                ()
        """
        return storage.partition_keys(self.source_stats())

    def is_partitioned(self) -> bool:
        """Whether the data is physically partitioned on at least one column.

        Returns:
            ``True`` if a partition key is declared.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.is_partitioned()
                False
        """
        return storage.is_partitioned(self.source_stats())

    def sorted_by(self) -> tuple[SortOrder, ...]:
        """The ordering every source maintains, in order, direction included.

        Only a *recorded* ordering: empty means "not declared", not "unordered". A sort on
        this prefix is a no-op, which is what the optimizer uses it for. Each key is a
        `SortOrder` naming the column, whether it descends, and where its nulls sit.

        Returns:
            The common sort prefix.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.sorted_by()
                ()
        """
        return storage.sorted_by(self.source_stats())

    def files(self) -> list[str]:
        """The data files the query's sources hold, in scan order.

        Every file of every file-backed source: a single Parquet, CSV, or JSON file, each file
        of a directory or glob, and each leaf file of a hive-partitioned tree. It is the
        listing the scan plans from, taken before any filter prunes a file away, so it is the
        upper bound on what the query opens. Listing a directory reads no footer and no data.

        Empty for a source with no file backing (an in-memory relation, a streaming source),
        because the question does not apply to it. A source whose listing fails also
        contributes nothing rather than failing the call.

        Returns:
            The paths of the sources' data files.

        Examples:
            .. doctest::

                >>> import os, tempfile
                >>> import pyarrow as pa, pyarrow.parquet as pq
                >>> import batcher as bt
                >>> root = tempfile.mkdtemp()
                >>> for i in range(2):
                ...     pq.write_table(pa.table({"x": [i]}), os.path.join(root, f"{i}.parquet"))
                >>> sorted(os.path.basename(p) for p in bt.read.parquet(root).meta.storage.files())
                ['0.parquet', '1.parquet']
                >>> bt.from_pydict({"x": [1]}).meta.storage.files()
                []
        """
        found: list[str] = []
        for source in self._ds._sources:
            try:
                found.extend(_source_files(source))
            except Exception as exc:  # a source that cannot list itself contributes nothing
                note_suppressed("api", "list a source's files", exc)
        return found

    def num_files(self) -> int:
        """How many data files the query's sources hold — ``len(files())``.

        The small-files diagnosis, without a scan: a thousand files for a gigabyte means the
        query is about to spend its time on footers rather than on data. Zero for a source
        with no file backing, such as an in-memory relation.

        Returns:
            The number of data files.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.num_files()
                0
        """
        return len(self.files())


def _source_files(source: object) -> list[str]:
    """The data files one source reads, from the listing it already keeps, or ``[]``.

    No source exposes a public file list, so this asks each kind for the one it plans its
    scan from: `FileSource._files()` (Parquet, CSV, JSON, and every other `FileSource`
    format; memoized, so this is the same listing the read uses) and the fragments of a
    `ParquetDatasetSource`'s discovered hive dataset. Anything else has no files to name.
    """
    from batcher.io import FileSource
    from batcher.io.formats.structured.parquet import ParquetDatasetSource

    if isinstance(source, FileSource):
        return [str(path) for path in source._files()]
    if isinstance(source, ParquetDatasetSource):
        return [str(path) for path in source._file_paths(source._dataset())]
    return []
