"""Per-operator distributed executor implementations.

The distributed dispatcher (`batcher.dist.executor`) routes a plan to one of
these operator implementations, each of which reuses the engine's mergeable
primitives so its distributed result is identical to single-node execution:

- `map` — distributed `map_batches` (batch inference), embarrassingly parallel.
- `aggregate` — distributed aggregation over a disk Arrow-IPC shuffle.
- `join` — distributed hash join, both sides co-partitioned by key.
- `sort` — distributed single-column sort via range partitioning.

`partition_io`, `plan_analysis`, and `ray_runtime` are the shared helpers
(partitioning / post-breaker re-apply, plan-shape inspection, Ray lifecycle +
single-node fallback) those operators build on. Internal to `batcher.dist`.
"""

from __future__ import annotations
