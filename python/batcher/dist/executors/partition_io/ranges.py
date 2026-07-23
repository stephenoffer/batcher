"""Range partitioning: split rows by *value* into globally ordered buckets.

The distributed and out-of-core sorts both work the same way — sample the leading key's
distribution, cut it into `n` ordered ranges, and route each row to its range — so the buckets
concatenate, in bucket order, into a globally sorted result with **no final merge**.

This lives in one module because every sort path must partition *identically*. When the
out-of-core sort re-derived this logic in NumPy instead of calling it, it put the null bucket
at the wrong end for a descending sort and silently returned unsorted data. The per-row work
belongs in Rust (`nat.range_partition_batches`); the only thing Python decides is where the
boundaries fall.
"""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.native import engine

__all__ = ["SAMPLE_PROBS", "bucketize", "merge_boundaries"]


# Per-sampler CDF granularity: a fine grid (33 probe points) so the merged boundaries
# balance the ranges well. Precision affects only balance, never the result. Shared by
# every range-partitioning path — the Ray sort (`executors/sort.py`), the Flight sort
# (`flight_sort.py`), and the out-of-core sort/window (`spill_breakers.py`) — so they
# all sample the same way.
SAMPLE_PROBS: list[float] = [i / 32 for i in range(33)]


def merge_boundaries(grids: list[tuple[list[float], int]], workers: int) -> list[float]:
    """Merge per-worker quantile grids into `workers-1` deduplicated range boundaries.

    Each grid is a `(sampled_cdf, row_count)` pair; empty/zero-row grids are dropped.
    Splits are roughly equal-size, so an unweighted concat of the sampled CDFs
    approximates the global distribution; the evenly spaced split points of that
    concat are the boundaries (dedup means equal keys never span a boundary). Returns
    `[]` for a single worker (one bucket).
    """
    import numpy as np

    samples = [np.asarray(grid, dtype=float) for grid, n in grids if grid and n]
    if not samples:
        return []
    qs = np.linspace(0, 1, workers + 1)[1:-1]
    if len(qs) == 0:
        return []
    return np.unique(np.quantile(np.concatenate(samples), qs)).tolist()


def bucketize(
    batches: list[pa.RecordBatch],
    key_name: str,
    boundaries: list[float],
    n_buckets: int,
    nulls_first: bool,
    descending: bool,
) -> list[list[pa.RecordBatch]]:
    """Split `batches` into `n_buckets` lists by the leading key's `boundaries`.

    Bucket `b` holds keys in the `b`-th open interval of `boundaries`, so the buckets
    are globally ordered and equal keys never span a boundary
    (`searchsorted(side="right")` keeps equal keys in one bucket). Nulls go to whichever
    end the caller's final concatenation places first/last: for a descending sort the
    driver concatenates buckets high→low, so the "front" bucket is `n_buckets-1` (else
    `0`); nulls land in the front bucket when `nulls_first`, else the opposite end —
    matching single-node null ordering exactly. Boundary precision affects only balance,
    never the result.

    The per-row bucketing + scatter runs in the Rust data plane
    (`nat.range_partition_batches`, the range counterpart of the hash
    `partition_batches`), so this stays off the per-row Python hot path.
    """
    if not batches:
        return [[] for _ in range(n_buckets)]
    nat = engine()
    key_index = batches[0].schema.get_field_index(key_name)
    return nat.range_partition_batches(
        list(batches), key_index, list(boundaries), n_buckets, nulls_first, descending
    )
