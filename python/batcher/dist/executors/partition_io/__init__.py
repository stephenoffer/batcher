"""Partitioning for the distributed operators — by *source split* and by *key range*.

`_sources` assigns a source's splits to per-worker partition files (lazily, so the driver never
materializes a splittable source) and re-applies post-breaker operators single-node. `ranges`
splits rows by key value into globally ordered buckets for the sorts.

Both were one module until the range half was re-derived, wrongly, elsewhere; they are kept
adjacent and re-exported here so the import path `batcher.dist.executors.partition_io` is
unchanged and there is exactly one place to find either.
"""

from __future__ import annotations

from batcher.dist.executors.partition_io._sources import (
    _apply_above,
    _balance,
    _eager_range_split,
    _partition_source,
    _projected_empty_batch,
    _slice_rows_evenly,
    descriptor_rows,
    iter_partition,
    iter_partition_descriptor,
    materialize_reduce_output,
    partition_descriptors,
    read_partition,
    read_partition_descriptor,
    source_pushdown,
    streaming_partial_aggregate,
)
from batcher.dist.executors.partition_io.ranges import (
    SAMPLE_PROBS,
    bucketize,
    merge_boundaries,
)

__all__ = [
    "SAMPLE_PROBS",
    "bucketize",
    "descriptor_rows",
    "iter_partition",
    "iter_partition_descriptor",
    "materialize_reduce_output",
    "merge_boundaries",
    "partition_descriptors",
    "read_partition",
    "read_partition_descriptor",
    "source_pushdown",
    "streaming_partial_aggregate",
]
