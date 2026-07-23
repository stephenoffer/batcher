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
from batcher.io.stats.file_identity import file_identity

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


def _parquet_footer(path: str):
    """The Parquet `FileMetaData` for `path`, read once per *version* and cached.

    Reading the footer (row-group offsets, schema) is a ~100ms object-store round trip;
    a worker reads many row-group splits of the same file, so caching the footer turns
    N footer reads into one. Bounded LRU so a long-lived process scanning many files
    stays memory-bounded.

    Keyed on the file's identity — `(path, size, mtime)` — not on the path. The path
    alone was justified by "Parquet is write-once", which holds for an immutable lake and
    not for a pipeline re-run: `FileSink` writes deterministic names, so a job overwrites
    its own output, and a path-keyed footer then describes the *previous* file. That is
    not a stale file but stale metadata about a new one — row-group offsets from the old
    footer index into the middle of the new bytes (`RowGroupSplit` passes this metadata
    straight to `ParquetFile`), and a cached row count answers a `count()` that
    `collect()` then contradicts.

    A file that cannot be stat-ed is read uncached rather than cached under a token that
    could not detect its changing.
    """
    identity = file_identity(path)
    if identity is None:
        return _read_footer(path)
    return _parquet_footer_cached(identity)


@lru_cache(maxsize=1024)
def _parquet_footer_cached(identity: tuple[str, int, int]):
    """`_parquet_footer` keyed on the file identity (see there)."""
    return _read_footer(identity[0])


def _read_footer(path: str):
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


def parquet_row_group_splits(
    path: str, target_size: int | None, predicate: dict | None = None
) -> list[Split]:
    """Build `RowGroupSplit`s for a single Parquet file, dropping the ones that cannot match.

    `predicate` prunes at **plan** time, from the footer this function already reads. That is a
    different thing from the row-group pruning the reader does at read time, and a strictly
    better one: a row-group eliminated here never becomes a `Split`, so it is never balanced,
    never shipped to a worker, and never *opened*. A file whose every row-group is ruled out
    returns **no splits at all** and drops out of the query entirely — the task is never created.

    Sound by construction: `file_prune_mask` drops a row-group only when its recorded bounds
    *prove* it holds no matching row, and anything unknown (a missing statistic, an
    unrepresentable predicate) keeps it. The engine's `Filter` re-checks every surviving row
    regardless, so an over-broad survivor set only costs I/O; an over-narrow one would lose rows,
    and cannot happen.
    """
    # The cached footer avoids re-reading the metadata here AND on the worker that later
    # reads these splits (a ~100ms object-store round trip per call); see `_parquet_footer`.
    meta = _parquet_footer(path)
    targets = list(range(meta.num_row_groups))
    if predicate is not None:
        targets = _surviving_row_groups(meta, predicate)
        if not targets:
            return []  # provably no matching row in this file — do not create a task for it

    sizes = [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)]
    rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    runs = _pack(targets, sizes, target_size)
    # Carry the footer-derived row count so balancing never re-opens the file.
    return [RowGroupSplit(path, run, sum(rows[i] for i in run)) for run in runs]


def _pack(targets: list[int], sizes: list[int], target_size: int | None) -> list[list[int]]:
    """Group `targets` (already pruned, ascending) into runs of ~`target_size` bytes.

    Only *adjacent surviving* groups are coalesced. Pruning can leave gaps, and a run must stay
    a contiguous row-group range for the reader — so a gap ends the run.
    """
    if not target_size:
        return [[i] for i in targets]
    runs: list[list[int]] = []
    current: list[int] = []
    total = 0
    for i in targets:
        if current and (i != current[-1] + 1 or total + sizes[i] > target_size):
            runs.append(current)
            current, total = [], 0
        current.append(i)
        total += sizes[i]
    if current:
        runs.append(current)
    return runs


def _surviving_row_groups(meta: Any, predicate: dict) -> list[int]:
    """The row-groups whose footer statistics do not rule them out.

    Shapes the per-row-group stats into the very same **add-action layout** a lakehouse
    transaction log publishes (``path | num_records | min.<col> | max.<col> | null_count.<col>``)
    and hands them to `io.stats.file_skipping.file_prune_mask`. That evaluator is already
    vectorized over the file dimension, already three-valued-sound, and already tested — a
    Parquet footer is just another manifest, so it should not have a second zone-map
    implementation to keep in step with the first.
    """
    try:
        import pyarrow.compute as pc

        from batcher.io.stats.file_skipping import file_prune_mask

        columns = _columns_in(predicate)
        if not columns:
            return list(range(meta.num_row_groups))
        manifest = _row_group_manifest(meta, sorted(columns))
        if manifest is None:
            return list(range(meta.num_row_groups))
        mask = file_prune_mask(predicate, manifest)
        if mask is None:
            return list(range(meta.num_row_groups))
        keep = pc.fill_null(mask, True).to_pylist()
        return [i for i, k in enumerate(keep) if k]
    except Exception:
        return list(range(meta.num_row_groups))  # a footer we cannot read prunes nothing


def _row_group_manifest(meta: Any, columns: list[str]) -> pa.Table | None:
    """Per-row-group bounds for `columns`, in the add-action layout `file_skipping` consumes."""
    names = meta.schema.names
    wanted = {c: names.index(c) for c in columns if c in names}
    if not wanted:
        return None

    n = meta.num_row_groups
    data: dict[str, Any] = {
        "path": [str(i) for i in range(n)],
        "num_records": [meta.row_group(i).num_rows for i in range(n)],
    }
    for name, index in wanted.items():
        lows: list[Any] = []
        highs: list[Any] = []
        nulls: list[int | None] = []
        for i in range(n):
            stats = meta.row_group(i).column(index).statistics
            if stats is None or not getattr(stats, "has_min_max", False):
                lows.append(None)
                highs.append(None)
                nulls.append(None)
                continue
            low, high = stats.min, stats.max
            # A NaN bound is unordered; treat it as "unknown", which keeps the row-group.
            if _is_nan(low) or _is_nan(high):
                low = high = None
            lows.append(low)
            highs.append(high)
            nulls.append(stats.null_count if getattr(stats, "has_null_count", False) else None)
        data[f"min.{name}"] = lows
        data[f"max.{name}"] = highs
        data[f"null_count.{name}"] = nulls
    try:
        return pa.table(data)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return None  # bounds that will not unify into one column type prune nothing


def _columns_in(node: Any, out: set[str] | None = None) -> set[str]:
    """Every column name the predicate IR reads. Walks the dict, so no node kind is missed."""
    out = set() if out is None else out
    if isinstance(node, dict):
        if node.get("e") == "col" and isinstance(node.get("name"), str):
            out.add(node["name"])
        for value in node.values():
            _columns_in(value, out)
    elif isinstance(node, list):
        for value in node:
            _columns_in(value, out)
    return out


def _is_nan(value: Any) -> bool:
    import math

    return isinstance(value, float) and math.isnan(value)
