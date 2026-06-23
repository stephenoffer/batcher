"""Sharded training dataset — fixed-size Arrow-IPC shards + a JSON index.

The streaming-loader's storage layer (the MosaicML-Streaming role): a large training
corpus is written once as a directory of equal-size Arrow-IPC shards plus an
``index.json`` manifest, then read back with **random access by global row index**
through a bounded LRU shard cache. That is what lets `ds.ml.stream_loader` feed a
shuffled, sharded, resumable sample order to a trainer *without materializing the
whole dataset* — only the few shards a batch touches are resident.

Layout::

    <dir>/index.json            {"rows_per_shard", "total_rows", "shards":[{"path","rows"}]}
    <dir>/shard-00000.arrow     Arrow IPC file (one record batch)
    <dir>/shard-00001.arrow
    ...

The format is plain Arrow IPC, so a shard is readable by any Arrow consumer; the
index makes global-index → (shard, offset) resolution O(log n) without opening data.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher.io.filesystem import resolve_filesystem

__all__ = ["ShardIndex", "ShardReader", "read_shard_index", "write_shards"]

_INDEX = "index.json"


@dataclass(frozen=True, slots=True)
class ShardIndex:
    """The manifest for a shard directory: per-shard row counts and the running
    offsets that map a global row index to ``(shard, local offset)``."""

    rows_per_shard: int
    total_rows: int
    shard_paths: tuple[str, ...]
    shard_rows: tuple[int, ...]
    starts: tuple[int, ...]  # global start offset of each shard (prefix sum)

    def locate(self, global_index: int) -> tuple[int, int]:
        """Map a global row index to ``(shard_idx, local_idx)``."""
        if not 0 <= global_index < self.total_rows:
            raise IndexError(f"row {global_index} out of range [0, {self.total_rows})")
        shard = bisect_right(self.starts, global_index) - 1
        return shard, global_index - self.starts[shard]


def write_shards(
    batches: Iterable[pa.RecordBatch] | pa.Table,
    directory: str,
    *,
    rows_per_shard: int = 65_536,
) -> ShardIndex:
    """Write `batches` as equal-size Arrow-IPC shards + an index under `directory`.

    Rows are repacked to exactly `rows_per_shard` per shard (the last is the
    remainder), so shard boundaries are independent of the input batching — a
    prerequisite for deterministic global indexing. Returns the written `ShardIndex`.
    """
    if rows_per_shard < 1:
        raise ValueError(f"rows_per_shard must be >= 1, got {rows_per_shard}")
    import pyarrow.ipc as ipc

    table = batches if isinstance(batches, pa.Table) else pa.Table.from_batches(list(batches))
    fs = resolve_filesystem(directory)
    fs.mkdirs(directory, exist_ok=True)

    shard_paths: list[str] = []
    shard_rows: list[int] = []
    for shard_idx, start in enumerate(range(0, table.num_rows, rows_per_shard)):
        chunk = table.slice(start, rows_per_shard).combine_chunks()
        name = f"shard-{shard_idx:05d}.arrow"
        path = f"{directory}/{name}"
        with fs.atomic_writer(path) as fh:
            writer = ipc.new_file(fh, table.schema)
            writer.write_table(chunk)
            writer.close()
        shard_paths.append(name)
        shard_rows.append(chunk.num_rows)

    index = {
        "rows_per_shard": rows_per_shard,
        "total_rows": table.num_rows,
        "schema": table.schema.to_string(),
        "shards": [{"path": p, "rows": r} for p, r in zip(shard_paths, shard_rows, strict=True)],
    }
    with fs.atomic_writer(f"{directory}/{_INDEX}") as fh:
        fh.write(json.dumps(index, indent=2).encode())
    return _build_index(directory, index)


def read_shard_index(directory: str) -> ShardIndex:
    """Load the `ShardIndex` for a shard directory (reads only ``index.json``)."""
    fs = resolve_filesystem(directory)
    with fs.open(f"{directory}/{_INDEX}") as fh:
        doc = json.loads(fh.read())
    return _build_index(directory, doc)


def _build_index(directory: str, doc: dict[str, Any]) -> ShardIndex:
    shards = doc["shards"]
    paths = tuple(f"{directory}/{s['path']}" for s in shards)
    rows = tuple(int(s["rows"]) for s in shards)
    starts: list[int] = []
    acc = 0
    for r in rows:
        starts.append(acc)
        acc += r
    return ShardIndex(
        rows_per_shard=int(doc["rows_per_shard"]),
        total_rows=int(doc["total_rows"]),
        shard_paths=paths,
        shard_rows=rows,
        starts=tuple(starts),
    )


class ShardReader:
    """Random access into a shard directory by global row index, bounded memory.

    Holds at most `cache_size` decoded shards in an LRU cache, so a shuffled read
    that touches shards in any order stays bounded by ``cache_size`` shards resident
    — never the whole dataset. `take(global_indices)` gathers the requested rows
    (grouped by shard so each touched shard is read once per call).
    """

    __slots__ = ("_cache", "_cache_size", "_fs", "_index")

    def __init__(self, directory: str, *, cache_size: int = 4) -> None:
        self._index = read_shard_index(directory)
        self._fs = resolve_filesystem(directory)
        self._cache: OrderedDict[int, pa.Table] = OrderedDict()
        self._cache_size = max(1, cache_size)

    @property
    def total_rows(self) -> int:
        return self._index.total_rows

    @property
    def index(self) -> ShardIndex:
        return self._index

    def _shard(self, shard_idx: int) -> pa.Table:
        cached = self._cache.get(shard_idx)
        if cached is not None:
            self._cache.move_to_end(shard_idx)
            return cached
        import pyarrow.ipc as ipc

        with self._fs.open(self._index.shard_paths[shard_idx]) as fh:
            table = ipc.open_file(fh).read_all()
        self._cache[shard_idx] = table
        self._cache.move_to_end(shard_idx)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # evict least-recently-used
        return table

    def take(self, global_indices: list[int]) -> pa.Table:
        """Gather the given global row indices into one table, preserving their order.

        Indices are grouped by shard so each touched shard is read at most once here;
        the per-shard rows are then reassembled into the requested order.
        """
        if not global_indices:
            return self._index_empty_table()
        # Resolve to (shard, local) and remember each output position.
        per_shard: dict[int, list[tuple[int, int]]] = {}
        for out_pos, gi in enumerate(global_indices):
            shard, local = self._index.locate(gi)
            per_shard.setdefault(shard, []).append((local, out_pos))
        gathered: list[tuple[int, pa.Table]] = []  # (out_pos, single-row table)
        for shard_idx, pairs in per_shard.items():
            table = self._shard(shard_idx)
            locals_ = [lp[0] for lp in pairs]
            sub = table.take(locals_)
            for row_in_sub, (_, out_pos) in enumerate(pairs):
                gathered.append((out_pos, sub.slice(row_in_sub, 1)))
        gathered.sort(key=lambda t: t[0])
        return pa.concat_tables([t for _, t in gathered])

    def _index_empty_table(self) -> pa.Table:
        # Read the first shard's schema for an empty result (cheap: schema only).
        import pyarrow.ipc as ipc

        with self._fs.open(self._index.shard_paths[0]) as fh:
            schema = ipc.open_file(fh).schema
        return schema.empty_table()
