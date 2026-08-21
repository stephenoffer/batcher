"""Out-of-core streaming for the binary/ordering breakers: sort, join, window.

These reuse the same radix/range-partition-to-disk machinery as the aggregate spill
(`spill`), but each is shaped as a **generator** that yields its result one bounded bucket at
a time, so `iter_batches()` streams a sort / join / window in bounded memory instead of
materializing the whole result. The `execute_spilling_*` wrappers collect the same generator
for `collect()` — one implementation, two consumers.

Each breaker also exposes a `supports_spilling_*` predicate: the spill path declines rather
than raising when it cannot apply, and the caller falls back to the in-memory operator (which
costs memory, never correctness). Split into a package along those three seams; the import
path is unchanged.
"""

from __future__ import annotations

from batcher.dist.spill_breakers.join import (
    execute_spilling_join,
    iter_join_paths_spilling,
    reduce_join_paths_spilling,
    stream_spilling_join,
    supports_spilling_join,
)
from batcher.dist.spill_breakers.sort import (
    execute_spilling_sort,
    iter_ordered_buckets,
    stage_and_partition,
    stream_spilling_sort,
    supports_spilling_sort,
)
from batcher.dist.spill_breakers.window import (
    stream_spilling_window,
    supports_spilling_window,
)

__all__ = [
    "execute_spilling_join",
    "execute_spilling_sort",
    "iter_join_paths_spilling",
    "iter_ordered_buckets",
    "reduce_join_paths_spilling",
    "stage_and_partition",
    "stream_spilling_join",
    "stream_spilling_sort",
    "stream_spilling_window",
    "supports_spilling_join",
    "supports_spilling_sort",
    "supports_spilling_window",
]
