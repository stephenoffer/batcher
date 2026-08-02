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
    _apply_above as _apply_above,
)
from batcher.dist.executors.partition_io._sources import (
    _eager_range_split as _eager_range_split,
)
from batcher.dist.executors.partition_io._sources import (
    _partition_source as _partition_source,
)
from batcher.dist.executors.partition_io._sources import (
    _projected_empty_batch as _projected_empty_batch,
)
from batcher.dist.executors.partition_io._sources import (
    _slice_rows_evenly as _slice_rows_evenly,
)
from batcher.dist.executors.partition_io._sources import (
    consumer_pushdown,
    descriptor_rows,
    iter_partition,
    iter_partition_descriptor,
    materialize_reduce_output,
    partition_descriptors,
    read_partition,
    read_partition_descriptor,
    source_pushdown,
)
from batcher.dist.executors.partition_io.assignment import (
    _balance as _balance,
)
from batcher.dist.executors.partition_io.assignment import (
    assign_splits,
    balance_with_affinity,
    has_affinity,
)
from batcher.dist.executors.partition_io.folds import (
    streaming_map_buckets,
    streaming_partial_aggregate,
    streaming_topn,
)
from batcher.dist.executors.partition_io.ranges import (
    SAMPLE_PROBS,
    bucketize,
    merge_boundaries,
    sample_key_grid,
)

__all__ = [
    "SAMPLE_PROBS",
    "assign_splits",
    "balance_with_affinity",
    "bucketize",
    "consumer_pushdown",
    "descriptor_rows",
    "has_affinity",
    "iter_partition",
    "iter_partition_descriptor",
    "materialize_reduce_output",
    "merge_boundaries",
    "partition_descriptors",
    "read_partition",
    "read_partition_descriptor",
    "sample_key_grid",
    "source_pushdown",
    "streaming_map_buckets",
    "streaming_partial_aggregate",
    "streaming_topn",
]
