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

__all__ = ["SAMPLE_PROBS", "bucketize", "merge_boundaries", "sample_key_grid"]


# Per-sampler CDF granularity: a fine grid (33 probe points) so the merged boundaries
# balance the ranges well. Precision affects only balance, never the result. Shared by
# every range-partitioning path — the Ray sort (`executors/sort.py`), the Flight sort
# (`flight_sort.py`), and the out-of-core sort/window (`spill_breakers.py`) — so they
# all sample the same way.
SAMPLE_PROBS: list[float] = [i / 32 for i in range(33)]


def sample_key_grid(
    batches: list[pa.RecordBatch], key_name: str, probs: list[float]
) -> list[float] | list[str]:
    """Sample the leading sort key's distribution at `probs` — one grid, whatever its type.

    A numeric or temporal key is summarized by the mergeable KLL sketch; a **string** key
    has no numeric sketch, so it is sampled lexically instead. Every sort path samples
    through this one function for the reason this module exists: when the out-of-core sort
    re-derived the *bucketing* in NumPy it put the null bucket at the wrong end and silently
    returned unsorted data, and three separate copies of the *sampling* is the same shape of
    mistake waiting to happen — one of them would have kept refusing string keys after the
    others learned to route them.

    Args:
        batches: The mapped rows this sampler is describing.
        key_name: The leading sort key's column name.
        probs: Ascending probabilities to sample the distribution at.

    Returns:
        The sampled values in ascending order, empty when the key is absent or all null.
    """
    if not batches:
        return []
    nat = engine()
    index = batches[0].schema.get_field_index(key_name)
    if index < 0:
        return []
    dtype = batches[0].schema.field(index).type
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return nat.column_string_quantiles(key_name, list(batches), list(probs))
    return nat.column_quantiles([key_name], list(batches), list(probs)).get(key_name, [])


def merge_boundaries(
    grids: list[tuple[list[float], int]] | list[tuple[list[str], int]], workers: int
) -> list[float] | list[str]:
    """Merge per-worker quantile grids into `workers-1` deduplicated range boundaries.

    Each grid is a `(sampled_cdf, row_count)` pair; empty/zero-row grids are dropped. The
    global distribution is the **mixture** of the per-worker ones weighted by how many rows
    each describes, and the evenly spaced split points of that mixture are the boundaries
    (dedup means equal keys never span a boundary). Returns `[]` for a single worker.

    The row count is what makes this a mixture rather than an average, and it is not
    optional. Splits are *not* equal-size in general: the sample pass runs the mapped plan,
    so a pushed-down predicate that keeps 90% of one split and 1% of another leaves their
    post-filter counts orders of magnitude apart. Weighting every worker's samples equally
    then places the boundaries by the *number of splits* on each side rather than the number
    of rows, and one reducer receives most of the data. A string key made it worse than
    unweighted: its sampler caps at `MAX_BOUNDARY_SAMPLE` values, so a 100M-row split and a
    70K-row split contribute 65,536 and 70,000 samples — the smaller split dominating
    outright.

    A **string** key's grid is merged lexicographically instead, which is the same
    construction over a different order: the weighted union of the samples is sorted and cut
    at evenly spaced positions of cumulative weight. The grids say which they are — a float
    grid holds floats and a string grid holds strings — so no caller has to pass a flag that
    could drift from the data it describes.
    """
    import numpy as np

    kept = [(grid, n) for grid, n in grids if grid and n]
    if not kept:
        return []
    if isinstance(kept[0][0][0], str):
        return _merge_string_boundaries(kept, workers)
    qs = np.linspace(0, 1, workers + 1)[1:-1]
    if len(qs) == 0:
        return []
    values = np.concatenate([np.asarray(grid, dtype=float) for grid, _ in kept])
    # Each of a worker's samples stands for an equal share of that worker's rows, so the
    # per-sample weight is its split's row count spread across its own grid. This is what
    # makes a short grid over many rows outweigh a long grid over few.
    weights = np.concatenate([np.full(len(grid), n / len(grid), float) for grid, n in kept])
    return np.unique(_weighted_quantile(values, weights, qs)).tolist()


def _weighted_quantile(values, weights, qs):
    """Quantiles of the weighted empirical distribution of `values`.

    Places each sample at the centre of the cumulative-weight interval it accounts for, so a
    uniform weighting reproduces the unweighted quantile this replaced and no boundary is
    biased toward the low end by half a sample.
    """
    import numpy as np

    order = np.argsort(values, kind="stable")
    ordered, ordered_w = values[order], weights[order]
    cumulative = np.cumsum(ordered_w)
    total = cumulative[-1]
    if total <= 0:  # pragma: no cover - every kept grid has a positive row count
        return np.asarray([], dtype=float)
    return np.interp(qs, (cumulative - 0.5 * ordered_w) / total, ordered)


def _merge_string_boundaries(grids: list[tuple[list[str], int]], workers: int) -> list[str]:
    """The lexical counterpart of the numeric merge: sort the weighted union of every
    worker's sample and cut it at `workers-1` evenly spaced positions of cumulative weight.

    Weighted for the reason `merge_boundaries` describes, and by the same per-sample share of
    its split's rows. Lexical order admits no interpolation, so a cut takes the value whose
    weight interval spans the target rather than a point between two values.

    Deduplicated for the same reason the numeric path is: `partition_point(|b| b <= v)`
    routes a value equal to a boundary to the higher bucket, so a repeated boundary would
    leave an empty bucket rather than split equal keys — but dropping it keeps the bucket
    count honest against `n_buckets`.
    """
    if workers < 2:
        return []
    pooled = sorted((value, n / len(grid)) for grid, n in grids for value in grid)
    if not pooled:
        return []
    total = sum(weight for _, weight in pooled)
    cuts: list[str] = []
    index, below = 0, 0.0  # weight of every sample strictly before `index`
    for step in range(1, workers):
        target = step / workers * total
        while index < len(pooled) - 1 and below + pooled[index][1] < target:
            below += pooled[index][1]
            index += 1
        cuts.append(pooled[index][0])
    return sorted(set(cuts))


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
    # A string key routes by byte-lexical comparison and a numeric one by `f64`. The KEY
    # COLUMN decides, not the boundary list: a split whose key is entirely null samples an
    # empty grid, and an empty list of boundaries cannot say which partitioner it belongs
    # to. Routing a string key through the numeric one would order "12" before "9" and
    # disagree with the single-node sort — the reason the dispatcher used to refuse the
    # shape outright.
    if pa.types.is_string(batches[0].schema.field(key_index).type) or pa.types.is_large_string(
        batches[0].schema.field(key_index).type
    ):
        return nat.range_partition_batches_str(
            list(batches), key_index, list(boundaries), n_buckets, nulls_first, descending
        )
    return nat.range_partition_batches(
        list(batches), key_index, list(boundaries), n_buckets, nulls_first, descending
    )
