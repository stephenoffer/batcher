"""Learned execution-time sizing for the distributed executor (façade).

The distributed map/agg/shuffle path sizes several *scheduling* parameters — partition count,
per-task CPU share, shuffle fan-out, inference actor-pool size, straggler-speculation threshold —
from plan estimates. This subpackage learns them from the runtime feedback Core already records,
so the next run of a recurring shape starts tuned instead of cold. Every parameter is a pure
scheduling knob: none can change a result (see `sizing` for the invariance argument).
"""

from __future__ import annotations

from batcher.dist.adaptive_sizing.sizing import (
    aggregate_reducer_count,
    learned_actor_pool_size,
    learned_cpu_weight_factor,
    learned_partition_rows,
    learned_shuffle_fanout,
    learned_straggler_factor,
    record_actor_pool_reuse,
    record_aggregate_cardinality,
    record_partition_rows,
    row_shuffle_reducer_count,
)

__all__ = [
    "aggregate_reducer_count",
    "learned_actor_pool_size",
    "learned_cpu_weight_factor",
    "learned_partition_rows",
    "learned_shuffle_fanout",
    "learned_straggler_factor",
    "record_actor_pool_reuse",
    "record_aggregate_cardinality",
    "record_partition_rows",
    "row_shuffle_reducer_count",
]
