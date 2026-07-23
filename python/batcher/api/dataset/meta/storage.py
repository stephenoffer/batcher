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

from batcher.api.dataset.meta._facts import MetaBase
from batcher.kyber.shortcuts import storage

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
        """The total on-disk size of the sources, in bytes, or ``None`` if any cannot say.

        Returns:
            The compressed byte size across every source, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.total_bytes() is None
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
        """The average on-disk bytes per row, or ``None`` when either total is unknown.

        *Compressed* width, so it is the number that predicts scan time.
        ``ds.meta.approx.row_bytes()`` estimates the in-memory width instead, which is wider.

        Returns:
            The average compressed bytes per row, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.bytes_per_row() is None
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

    def sorted_by(self) -> tuple[str, ...]:
        """The ascending, nulls-last ordering every source maintains, in order.

        Only a *recorded* ordering: empty means "not declared", not "unordered". A sort on
        this prefix is a no-op, which is what the optimizer uses it for.

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
        """The data files the query would open, in scan order.

        Empty for a source with no file backing (an in-memory relation, a streaming source) —
        the question does not apply to it, rather than being unknown.

        Returns:
            The paths of the files that would be read.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.files()
                []
        """
        found: list[str] = []
        for source in self._ds._sources:
            paths = getattr(source, "files", None)
            if callable(paths):
                try:
                    found.extend(str(p) for p in paths())
                except Exception:  # a source that cannot list itself contributes nothing
                    continue
        return found

    def num_files(self) -> int:
        """How many data files the query would open.

        The small-files diagnosis, without a scan: a thousand files for a gigabyte means the
        query is about to spend its time on footers rather than on data.

        Returns:
            The number of files that would be read.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.storage.num_files()
                0
        """
        return len(self.files())
