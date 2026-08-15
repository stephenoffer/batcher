"""Random access into a shard corpus by global row index, in bounded memory.

The read half of the training hot path. Two things decide whether it can feed an
accelerator: a batch of shuffled indices must be gathered with **vectorized** Arrow takes
rather than a slice per row, and a shard read that meets a throttle must be retried rather
than ending a multi-day training run.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import pyarrow as pa

from batcher.io.base._transient import with_retry
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.ml.shards.index import ShardIndex, read_shard_index

__all__ = ["ShardReader"]

#: Retry budget for one shard read. A training run reads its corpus for hours or days against
#: object storage, so it *will* meet a 503 or a dropped connection; without a retry that blip
#: ends the job. The read is against an immutable object, so repeating it returns the same
#: bytes — the precondition `with_retry` documents.
_READ_ATTEMPTS = 4
_READ_BACKOFF_S = 0.5


class ShardReader:
    """Random access into a shard directory by global row index, bounded memory.

    Holds at most `cache_size` decoded shards in an LRU cache, so a shuffled read that
    touches shards in any order stays bounded by ``cache_size`` shards resident — never the
    whole dataset. `take(global_indices)` gathers the requested rows (grouped by shard so
    each touched shard is read once per call).

    The cache is guarded by a lock, so a loader may read shards ahead on a background thread
    (`batcher._internal.prefetch`) without two threads racing to insert and evict the same
    entry.
    """

    __slots__ = ("_cache", "_cache_size", "_fs", "_index", "_lock", "_schema", "_starts")

    def __init__(self, directory: str, *, cache_size: int = 4) -> None:
        """Open the shard directory's index; shards are read lazily on demand."""
        self._index = read_shard_index(directory)
        self._fs = resolve_filesystem(directory)
        self._cache: OrderedDict[int, pa.Table] = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._lock = threading.Lock()
        self._schema: pa.Schema | None = self._index.schema
        self._starts: Any | None = None  # NumPy prefix sum, built only for a ragged corpus

    @property
    def total_rows(self) -> int:
        """Rows across every shard in the directory."""
        return self._index.total_rows

    @property
    def index(self) -> ShardIndex:
        """The directory's `ShardIndex`."""
        return self._index

    @property
    def schema(self) -> pa.Schema:
        """The dataset's Arrow schema, from the index where it records one.

        Falls back to reading the first shard's *header* (never its data) for a directory
        written before the index carried a schema. A directory with no shards and no
        recorded schema has none to report, which is an error rather than a silent
        zero-column table.
        """
        if self._schema is not None:
            return self._schema
        if not self._index.shard_count:
            raise ValueError(
                "shard directory has no shards and its index records no schema, "
                "so the dataset's columns are unknown; rewrite it with `write_shards`"
            )
        import pyarrow.ipc as ipc

        with self._fs.open(self._index.shard_path(0)) as fh:
            self._schema = ipc.open_file(fh).schema
        return self._schema

    def _shard(self, shard_idx: int) -> pa.Table:
        with self._lock:
            cached = self._cache.get(shard_idx)
            if cached is not None:
                self._cache.move_to_end(shard_idx)
                return cached
        import pyarrow.ipc as ipc

        path = self._index.shard_path(shard_idx)

        def _read() -> pa.Table:
            with self._fs.open(path) as fh:
                return ipc.open_file(fh).read_all()

        # Read outside the lock: a shard read is I/O, and holding the lock across it would
        # serialize every reader behind the slowest fetch (the point of prefetching is to
        # overlap them). A duplicate concurrent read of the same shard is wasted work at
        # worst, never a wrong answer.
        table = with_retry(_read, attempts=_READ_ATTEMPTS, backoff_base_s=_READ_BACKOFF_S)
        with self._lock:
            self._cache[shard_idx] = table
            self._cache.move_to_end(shard_idx)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)  # evict least-recently-used
        return table

    def _locate_many(self, indices: Any) -> tuple[Any, Any]:
        """Vectorized `ShardIndex.locate`: ``(shard_of_row, local_row)`` for every index.

        A uniform corpus resolves by integer division, so nothing per-shard is touched — the
        point of the uniform index. A corpus that names its shards falls back to a binary
        search over a materialized prefix sum, built once.
        """
        import numpy as np

        if indices.size:
            lo, hi = int(indices.min()), int(indices.max())
            if lo < 0 or hi >= self._index.total_rows:
                bad = lo if lo < 0 else hi
                raise IndexError(f"row {bad} out of range [0, {self._index.total_rows})")
        if self._index.uniform:
            width = np.int64(self._index.rows_per_shard)
            return indices // width, indices % width
        if self._starts is None:
            self._starts = np.asarray(self._index.starts, dtype=np.int64)
        shard = np.searchsorted(self._starts, indices, side="right") - 1
        return shard, indices - self._starts[shard]

    def take(self, global_indices: Iterable[int]) -> pa.Table:
        """Gather the given global row indices into one table, preserving their order.

        Indices are grouped by shard so each touched shard is read at most once here; the
        per-shard rows are then reassembled into the requested order.

        The reassembly is **two vectorized Arrow takes**, not a slice per row. Building the
        result as one single-row table per sample and `concat_tables`-ing them made a
        1,024-row batch cost ~25 ms of pure Python — around 40k rows/s, well under what one
        GPU consumes, and it handed the caller a table with 1,024 chunks per column so the
        tensor conversion downstream paid for the fragmentation a second time.

        Args:
            global_indices: Row indices into the whole dataset, in the order wanted.

        Returns:
            A `pyarrow.Table` of those rows, in that order, in one chunk per column.

        Raises:
            IndexError: If any index is outside ``[0, total_rows)``.
        """
        import numpy as np

        idx = np.asarray(list(global_indices), dtype=np.int64)
        if idx.size == 0:
            return self.schema.empty_table()
        shard_of, local = self._locate_many(idx)

        # Stable sort by shard: rows keep their request order *within* a shard, and every
        # shard's rows form one contiguous run, so each shard needs exactly one `take`.
        order = np.argsort(shard_of, kind="stable")
        sorted_shards = shard_of[order]
        boundaries = np.flatnonzero(np.diff(sorted_shards)) + 1
        parts: list[pa.Table] = []
        for run in np.split(order, boundaries):
            table = self._shard(int(shard_of[run[0]]))
            parts.append(table.take(pa.array(local[run])))
        gathered = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
        if len(parts) == 1 and boundaries.size == 0 and bool((order == np.arange(idx.size)).all()):
            return gathered.combine_chunks()  # already in request order — skip the reorder
        # `gathered` row j is request position `order[j]`; invert that to restore the order.
        inverse = np.empty(idx.size, dtype=np.int64)
        inverse[order] = np.arange(idx.size, dtype=np.int64)
        return gathered.take(pa.array(inverse)).combine_chunks()

    def take_batch(self, global_indices: Iterable[int]) -> pa.RecordBatch:
        """`take`, as a single `pyarrow.RecordBatch` rather than a `Table`.

        The shape a tensor conversion and a `collate_fn` actually want: one contiguous
        batch, so neither pays to walk a chunk list.

        Args:
            global_indices: Row indices into the whole dataset, in the order wanted.

        Returns:
            A `pyarrow.RecordBatch` of those rows, in that order.
        """
        table = self.take(global_indices)
        # `take` returns a combined table, so this is one batch — or none, when empty.
        batches = table.to_batches()
        return batches[0] if batches else pa.RecordBatch.from_pylist([], schema=table.schema)

    def clear_cache(self) -> None:
        """Drop every cached shard, releasing their memory immediately."""
        with self._lock:
            self._cache.clear()
