"""Out-of-core execution on one node: scratch plumbing, and the spilling aggregate.

Split along the seam between *where the bytes go*, *what is done to a bucket*, and *what
the operator computes from one*:

- `scratch` — the scratch directory, the tiered (local then remote) store, the open-file
  cap, and the morsel iterator that keeps the input from bounding peak memory.
- `buckets` — the mechanics every breaker shares: the scratch lifecycle, the per-bucket
  writers, how big a bucket is, and the salted grace re-split. Shared with the ordering and
  binary breakers in `dist.spill_breakers` and with `dist.global_window`, which is the point
  — each of them used to state these for itself.
- `aggregate` — partition-and-spill aggregation and the `spill_collect` dispatcher. The
  bucket reduce and its recursion live there together, with the caller, because that is what
  keeps them one unit.

The import path `batcher.dist.spill` is unchanged: everything either half exposes is
re-exported here, including the underscore-prefixed names other modules and tests already
reach for.
"""

from __future__ import annotations

from batcher.dist.spill.aggregate import _MAX_SPILL_RECURSION as _MAX_SPILL_RECURSION
from batcher.dist.spill.aggregate import _SUB_BUCKETS as _SUB_BUCKETS
from batcher.dist.spill.aggregate import _empty_table as _empty_table
from batcher.dist.spill.aggregate import _peel_to_breaker as _peel_to_breaker
from batcher.dist.spill.aggregate import _reduce_agg_bucket as _reduce_agg_bucket
from batcher.dist.spill.aggregate import _split_salt as _split_salt
from batcher.dist.spill.aggregate import execute_spilling_aggregate, spill_collect
from batcher.dist.spill.buckets import (
    BucketWriters,
    regrace,
    resident_bytes,
    spill_scratch,
    split_salt,
)
from batcher.dist.spill.scratch import _FD_SAFE_PARTITIONS as _FD_SAFE_PARTITIONS
from batcher.dist.spill.scratch import _SPILL_INPUT_CHUNK_BYTES as _SPILL_INPUT_CHUNK_BYTES
from batcher.dist.spill.scratch import _fd_safe as _fd_safe
from batcher.dist.spill.scratch import _iter_spill_morsels as _iter_spill_morsels
from batcher.dist.spill.scratch import _make_store as _make_store
from batcher.dist.spill.scratch import _work_dir as _work_dir
from batcher.dist.spill.scratch import map_projection

__all__ = [
    "BucketWriters",
    "execute_spilling_aggregate",
    "map_projection",
    "regrace",
    "resident_bytes",
    "spill_collect",
    "spill_scratch",
    "split_salt",
]
