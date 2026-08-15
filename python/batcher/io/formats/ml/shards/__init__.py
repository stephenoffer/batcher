"""Sharded training dataset — fixed-size Arrow-IPC shards + a JSON index.

The streaming loader's storage layer (the MosaicML-Streaming role): a large training corpus
is written once as a directory of equal-size Arrow-IPC shards plus an ``index.json``
manifest, then read back with **random access by global row index** through a bounded LRU
shard cache. That is what lets `ds.ml.stream_loader` feed a shuffled, sharded, resumable
sample order to a trainer *without materializing the whole dataset* — only the few shards a
batch touches are resident.

Layout::

    <dir>/index.json               {"rows_per_shard", "total_rows", "shard_count", ...}
    <dir>/shard-00000000.arrow     Arrow IPC file
    <dir>/shard-00000001.arrow
    ...

The shards are plain Arrow IPC, so one is readable by any Arrow consumer and the whole
corpus reads back relationally through `TrainingShardsSource` (``bt.read.training_shards``).

The modules split on what each is responsible for at scale:

* `index` — the manifest. O(1) in the shard count for a corpus this package wrote, because a
  petabyte corpus is millions of shards and naming each one is a manifest nobody can afford
  to parse on every rank.
* `writer` — `write_shards`: streaming, crash-safe, resumable.
* `reader` — `ShardReader`: vectorized gathers, a bounded cache, and retries for the
  throttles a multi-day training read will meet.
* `source` — the relational view of the same directory.
"""

from __future__ import annotations

from batcher.io.formats.ml.shards.index import (
    ShardIndex,
    read_shard_index,
    shard_name,
    write_index,
)
from batcher.io.formats.ml.shards.reader import ShardReader
from batcher.io.formats.ml.shards.source import TrainingShardsSource
from batcher.io.formats.ml.shards.writer import write_shards

__all__ = [
    "ShardIndex",
    "ShardReader",
    "TrainingShardsSource",
    "read_shard_index",
    "shard_name",
    "write_index",
    "write_shards",
]
