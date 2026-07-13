"""Parquet-dataset split locators — row groups, the footer cache, the fragment index.

`RowGroupSplit` is the finest distributed-read granularity Batcher has: a worker
reads only its assigned row-groups of one file (a single object-store range read).
The footer cache and the `pyarrow.dataset` fragment index live here because both
exist for the same reason — a worker reading many slices of one file (or one
dataset) must never re-read the metadata it already has. `fragment_index` is shared
with the lakehouse connectors, whose tables are Parquet datasets underneath.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pyarrow as pa

from batcher.io.splits.base import Split

__all__ = ["RowGroupSplit", "fragment_index", "pack_row_groups", "parquet_row_group_splits"]


# Per-process LRU of ``key -> (dataset, {fragment_path: fragment})``. A worker
# lists/opens a dataset ONCE and reuses the path→fragment index across all the
# splits it reads, instead of re-listing the whole dataset on every read (which
# would be O(files^2) over a per-file-split read — catastrophic at scale).
_FRAGMENT_INDEX_CACHE: OrderedDict[Any, tuple[Any, dict[str, Any]]] = OrderedDict()
_FRAGMENT_CACHE_MAX = 8


def fragment_index(key: Any, build_dataset: Any) -> tuple[Any, dict[str, Any]]:
    """Return ``(dataset, {fragment_path: fragment})`` for `key`, building once.

    `build_dataset` is a zero-arg callable returning a `pyarrow.dataset.Dataset`.
    The index is cached per process so each worker lists the dataset a single time
    regardless of how many of its fragments it reads. The cache is a bounded **LRU**:
    a hit marks the entry most-recently-used, and an insert past the bound evicts only
    the least-recently-used entry — so a worker cycling through more than
    `_FRAGMENT_CACHE_MAX` tables keeps its hot ones resident instead of dropping the
    whole cache (the old clear-all forced an O(files) re-list of every live table).
    """
    cached = _FRAGMENT_INDEX_CACHE.get(key)
    if cached is not None:
        _FRAGMENT_INDEX_CACHE.move_to_end(key)  # most-recently-used
        return cached
    dataset = build_dataset()
    index = {frag.path: frag for frag in dataset.get_fragments()}
    cached = (dataset, index)
    _FRAGMENT_INDEX_CACHE[key] = cached  # appended as most-recently-used
    while len(_FRAGMENT_INDEX_CACHE) > _FRAGMENT_CACHE_MAX:
        _FRAGMENT_INDEX_CACHE.popitem(last=False)  # evict least-recently-used
    return cached


@lru_cache(maxsize=1024)
def _parquet_footer(path: str):
    """The Parquet `FileMetaData` for `path`, read once and cached per process.

    Reading the footer (row-group offsets, schema) is a ~100ms object-store round trip;
    a worker reads many row-group splits of the same file, so caching the footer turns
    N footer reads into one. The metadata is immutable — safe to share across the
    threads of the scan prefetch pool. Bounded LRU so a long-lived process scanning many
    files stays memory-bounded.
    """
    import pyarrow.parquet as pq

    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(path)
    with fs.open(path) as fh:
        return pq.ParquetFile(fh).metadata


@dataclass(frozen=True, slots=True)
class RowGroupSplit:
    """A contiguous run of Parquet row-groups within one file.

    The finest distributed-read granularity for Parquet: a worker reads only its
    assigned row-groups (a single S3 range read), not the whole file. `rows` is the
    footer-derived row count captured when the split was built, so balancing the
    splits never re-opens the file just to count.

    Examples:
        .. doctest::

            >>> from batcher.io import RowGroupSplit  # doctest: +SKIP
            >>> split = RowGroupSplit("big.parquet", (0, 1), rows=2048)  # doctest: +SKIP
            >>> split.row_count()  # from the captured footer count  # doctest: +SKIP
            2048
    """

    path: str
    row_groups: tuple[int, ...]
    rows: int | None = None

    def _file(self) -> Any:
        import pyarrow.parquet as pq

        from batcher.io.filesystem import ensure_io_threads, resolve_filesystem

        ensure_io_threads()
        fs = resolve_filesystem(self.path)
        footer = _parquet_footer(self.path)
        # Pass the cached footer so opening this row-group reader does NOT re-read the
        # Parquet metadata from object storage. A worker reads several row-group splits
        # of one file; re-reading the footer per split is ~100ms each on S3 (the
        # dominant distributed-scan overhead). The footer is immutable, so sharing it
        # is safe; only a fresh data stream is opened per read.
        #
        # Prefer the native `(fs, path)` target: reading through a Python handle serializes
        # pyarrow's per-column decode threads (the `_read_table` finding — 2,831 vs 1,653 ms
        # for a 4-column read), and this per-split reader is the primary distributed fallback.
        # `pre_buffer` coalesces the column-chunk byte ranges into a few large asynchronous
        # object-store reads instead of one small GET per (column, row-group). Result-invariant.
        target = fs.native_read_target(self.path)
        if target is not None:
            pafs, in_path = target
            return pq.ParquetFile(in_path, filesystem=pafs, metadata=footer, pre_buffer=True)
        return pq.ParquetFile(fs.open(self.path), metadata=footer, pre_buffer=True)

    def schema(self) -> pa.Schema:
        """The file's schema, from the cached footer (no data scan).

        Examples:
            .. doctest::

                >>> from batcher.io import RowGroupSplit  # doctest: +SKIP
                >>> RowGroupSplit("big.parquet", (0,)).schema().names  # doctest: +SKIP
                ['id', 'ts']

        Returns:
            The Arrow schema of the file this split slices.
        """
        return self._file().schema_arrow

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        """Read only this split's row-groups, filtering before the shuffle.

        Examples:
            .. doctest::

                >>> from batcher.io import RowGroupSplit  # doctest: +SKIP
                >>> split = RowGroupSplit("big.parquet", (0, 1))  # doctest: +SKIP
                >>> len(split.read(["id"]))  # doctest: +SKIP
                2

        Args:
            projection: Columns to read. All columns when omitted.
            predicate: A filter applied to the decoded rows, so fewer rows cross
                the network. The engine re-checks it regardless.

        Returns:
            The batches of this split's row-groups.
        """
        table = self._file().read_row_groups(list(self.row_groups), columns=projection)
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            expr = to_pyarrow_expression(predicate)
            if expr is not None:
                table = table.filter(expr)  # reduce rows before the shuffle
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream this split's row-groups batch by batch.

        Examples:
            .. doctest::

                >>> from batcher.io import RowGroupSplit  # doctest: +SKIP
                >>> split = RowGroupSplit("big.parquet", (0,))  # doctest: +SKIP
                >>> next(split.iter_batches()).num_rows  # doctest: +SKIP
                1024

        Args:
            projection: Columns to read. All columns when omitted.

        Returns:
            An iterator over the split's batches.
        """
        yield from self._file().iter_batches(row_groups=list(self.row_groups), columns=projection)

    def row_count(self) -> int | None:
        """The split's exact rows — the captured count, else summed from the footer.

        Examples:
            .. doctest::

                >>> from batcher.io import RowGroupSplit  # doctest: +SKIP
                >>> RowGroupSplit("big.parquet", (0, 1), rows=2048).row_count()  # doctest: +SKIP
                2048

        Returns:
            The exact number of rows in this split's row-groups.
        """
        if self.rows is not None:
            return self.rows
        meta = self._file().metadata
        return sum(meta.row_group(i).num_rows for i in self.row_groups)

    def identity(self) -> str:
        """The ``parquet:path:rg<ids>`` key naming this run of row-groups.

        Examples:
            .. doctest::

                >>> from batcher.io import RowGroupSplit  # doctest: +SKIP
                >>> RowGroupSplit("big.parquet", (0, 1)).identity()  # doctest: +SKIP
                'parquet:big.parquet:rg0,1'

        Returns:
            A key that distinguishes this split from the file's other row-groups.
        """
        return f"parquet:{self.path}:rg{','.join(map(str, self.row_groups))}"


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


def parquet_row_group_splits(path: str, target_size: int | None) -> list[Split]:
    """Build `RowGroupSplit`s for a single Parquet file (used by ParquetSource)."""
    # The cached footer avoids re-reading the metadata here AND on the worker that later
    # reads these splits (a ~100ms object-store round trip per call); see `_parquet_footer`.
    meta = _parquet_footer(path)
    sizes = [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)]
    rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    runs = pack_row_groups(meta.num_row_groups, sizes, target_size)
    # Carry the footer-derived row count so balancing never re-opens the file.
    return [RowGroupSplit(path, run, sum(rows[i] for i in run)) for run in runs]
