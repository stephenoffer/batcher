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

__all__ = [
    "SAMPLE_PROBS",
    "bucketize",
    "hot_key_share",
    "hot_sub_bucket",
    "isolate_hot_value",
    "merge_boundaries",
    "plan_hot_split",
    "sample_key_grid",
    "sample_probs",
    "split_hot_bucket",
]


# Per-sampler CDF granularity: a fine grid (33 probe points) so the merged boundaries
# balance the ranges well. Precision affects only balance, never the result. Shared by
# every range-partitioning path — the Ray sort (`executors/sort.py`), the Flight sort
# (`flight_sort.py`), and the out-of-core sort/window (`spill_breakers.py`) — so they
# all sample the same way. `sample_probs` derives the count from the bucket-to-sampler ratio
# instead, and every distributed path uses it; this constant remains the floor it clamps to,
# and the grid the out-of-core sort uses (where the samplers are morsels and vastly
# outnumber the buckets, so the floor is what the derivation would return anyway).
SAMPLE_PROBS: list[float] = [i / 32 for i in range(33)]

# Samples the pooled grids must place in each bucket. Imbalance is governed by this one
# number — not by the grid size and not by the bucket count alone — and it is a measured
# knee rather than a derivation: over a heavy-tailed key (lognormal, 2M rows), the busiest
# bucket runs 2.0-7.7x the mean at 4 samples per bucket, 1.09-1.78x at 16, 1.03-1.52x at 32,
# and 1.02-1.07x at 64, after which it barely moves. 64 is where the curve flattens.
_SAMPLES_PER_BUCKET = 64
# Floor on a sampler's probe count: the historical 32-interval grid, which is already inside
# a percent of even when there are no more buckets than samplers.
_MIN_PROBES = 32
# Upper bound on a sampler's probe count. The grid crosses the network once per sampler and
# is merged on the driver, so `samplers x probes` is a driver-side sort; 1024 keeps a
# thousand-worker shuffle's merge at a million values (a few MB, sub-second) while covering
# a 16x bucket-to-sampler ratio at the target accuracy.
_MAX_PROBES = 1024


def sample_probs(n_buckets: int, n_samplers: int) -> list[float]:
    """The probabilities one sampler should probe its key distribution at, to cut
    `n_buckets` ranges from `n_samplers` merged grids without imbalancing them.

    A fixed grid is the wrong shape here, and the reason is arithmetic rather than taste.
    Boundaries are cut from the *pooled* grids, so what decides how precisely one can be
    placed is how many pooled samples fall in the bucket it bounds — `g·S/P` for `g`
    intervals per sampler, `S` samplers and `P` buckets. That quantity, not `g`, is what the
    imbalance tracks, and a fixed `g` lets it fall as `P/S` rises. Solving it for a target
    gives ``g = samples_per_bucket · P / S``, floored so a narrow cut never samples more
    coarsely than the constant did.

    The reduce side is sized by volume and by the learned fan-out, so `P` routinely runs
    several times the sampler count, and that is exactly where the constant failed. Measured
    over a lognormal key at 2M rows, busiest bucket against the mean: at 8 samplers and 32
    buckets, **1.98x** with the fixed grid and 1.03x with this one; at 16 and 128, **3.95x**
    against 1.04x; at 32 and 256, **7.69x** against 1.07x; at 64 and 512, **15.2x** against
    1.18x. The failure grows with the cluster, which is the part that matters: a reducer
    carrying fifteen times its share *is* the reduce phase, while every other reducer sits
    finished — so it comes straight off the speedup that widening the shuffle bought.

    Balance only, never the result: buckets are globally ordered for *any* monotone boundary
    list, so a coarse grid costs a slow reducer and can never cost a row or an ordering.

    Args:
        n_buckets: The number of ranges the boundaries will cut.
        n_samplers: How many independent grids will be merged, one per source partition.

    Returns:
        Ascending probabilities from 0 to 1 inclusive, at least `_MIN_PROBES` intervals.

    Examples:
        .. doctest::

            >>> from batcher.dist.executors.partition_io import sample_probs
            >>> len(sample_probs(8, 8)) - 1        # one bucket per sampler
            64
            >>> len(sample_probs(64, 8)) - 1       # eight buckets per sampler: eight times finer
            512
            >>> len(sample_probs(2, 64)) - 1       # fewer buckets than samplers: the floor
            32
    """
    buckets = max(1, n_buckets)
    samplers = max(1, n_samplers)
    wanted = -(-_SAMPLES_PER_BUCKET * buckets // samplers)  # ceil
    g = min(_MAX_PROBES, max(_MIN_PROBES, wanted))
    return [i / g for i in range(g + 1)]


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


# A hot value is worth splitting once its bucket carries this many times the mean. Below
# 2x the imbalance is cheaper than the extra reducers, and the split is not free: it adds
# `subs - 1` buckets to the shuffle and forces a boundary the sample did not ask for.
_HOT_SPLIT_OVERLOAD = 2.0
# Cap on how many sub-buckets one value is spread across, matching `dist/skew.py`'s salt
# ceiling and bounded for the same reason: past this the fan-out costs more in stream count
# than the levelling is worth.
_MAX_HOT_SUBS = 64


def hot_key_share(grids) -> tuple[float, float] | None:
    """The most frequent sampled key and its share of the rows, or `None` if the sample is
    string-typed, empty, or shows no value repeating.

    The grids are already in hand — they are the quantile samples the boundaries are cut
    from — so this is a count over a few hundred numbers rather than a pass over the data.
    A value holding share `f` occupies `f` of the sampled positions in expectation, which is
    exactly what the caller needs to size the split.

    Numeric keys only. A string key has no cheap successor to isolate it with (see
    [`isolate_hot_value`]), so a skewed string sort keeps the unsplit bucket.

    Args:
        grids: The `(sampled_values, row_count)` pairs the samplers returned.

    Returns:
        `(value, share)` for the most frequent sampled value, or `None`.
    """
    import numpy as np

    kept = [(g, n) for g, n in grids if g and n]
    if not kept or isinstance(kept[0][0][0], str):
        return None
    values = np.concatenate([np.asarray(g, dtype=float) for g, _ in kept])
    weights = np.concatenate([np.full(len(g), n / len(g), float) for g, n in kept])
    finite = np.isfinite(values)
    if not finite.any():
        return None
    values, weights = values[finite], weights[finite]
    uniq, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    total = mass.sum()
    if total <= 0:
        return None
    i = int(mass.argmax())
    return float(uniq[i]), float(mass[i] / total)


def isolate_hot_value(boundaries: list[float], hot: float) -> tuple[list[float], int]:
    """Add the two boundaries that put `hot` in a bucket of its own, and say which bucket.

    `bucketize` sends a key to `#{b in boundaries : b <= key}`, so a bucket spans
    `[B[i-1], B[i])`. Giving `hot` a bucket containing nothing else therefore needs `hot`
    itself as a boundary and the *next representable value above it* as the one after —
    `nextafter` rather than `hot + 1`, so it is correct for a float key and still correct
    for an integer one, where every other key is at least a whole unit away and so lands
    beyond it.

    Args:
        boundaries: The merged, ascending, deduplicated boundaries.
        hot: The value to isolate.

    Returns:
        `(boundaries, hot_bucket)` — the new boundary list and the index of the bucket that
        now holds exactly `hot`.
    """
    import numpy as np

    above = float(np.nextafter(hot, np.inf))
    widened = sorted({*(float(b) for b in boundaries), float(hot), above})
    return widened, widened.index(float(hot)) + 1


def hot_sub_bucket(mapper_id: int, n_mappers: int, subs: int, descending: bool) -> int:
    """Which sub-bucket of a split hot value this mapper's rows belong in.

    Contiguous by mapper rather than round-robin, so the driver's ordered concatenation of
    the sub-buckets reads them in mapper order. That matters because every row in these
    sub-buckets ties on the sort key, and the order ties come out in is part of what a
    distributed sort has to reproduce from the single-node one.

    A descending sort needs the layout **reversed**, and the reason is that only one of the
    two orderings flips. The driver walks the buckets high to low, so the last sub-bucket is
    emitted first; but the engine's sort is *stable in both directions* — ties keep their
    input order whether the sort is ascending or descending (checked directly: sorting five
    equal keys descending returns them in input order). So the rows inside a sub-bucket stay
    in mapper order either way, and mapper 0 has to be placed in the sub-bucket the driver
    reads first. Getting this backwards is invisible to any assertion on keys or on the row
    multiset — both are unchanged — and shows up only as the payload arriving in the wrong
    order.

    Args:
        mapper_id: This mapper's index.
        n_mappers: How many mappers the shuffle has.
        subs: How many sub-buckets the hot value is spread across.
        descending: Whether the driver concatenates buckets high to low.

    Returns:
        The sub-bucket offset, in `[0, subs)`.
    """
    j = min(subs - 1, mapper_id * subs // max(1, n_mappers))
    return (subs - 1 - j) if descending else j


def split_hot_bucket(parts: list, hot_bucket: int, subs: int, sub: int) -> list:
    """Expand `parts` so the hot bucket occupies `subs` physical buckets, this mapper
    contributing only to `sub`.

    Buckets below the hot one keep their index and buckets above it shift up by `subs - 1`,
    so the physical order is still the key order and the driver concatenates it unchanged.

    Args:
        parts: The logical buckets `bucketize` produced.
        hot_bucket: Index of the bucket holding only the hot value.
        subs: How many physical buckets that logical bucket becomes.
        sub: Which of them this mapper writes to; the rest get nothing from it.

    Returns:
        The physical bucket list, `len(parts) + subs - 1` long.
    """
    out: list = list(parts[:hot_bucket])
    out.extend(parts[hot_bucket] if j == sub else [] for j in range(subs))
    out.extend(parts[hot_bucket + 1 :])
    return out


def plan_hot_split(grids, boundaries: list, n_buckets: int, nulls_first: bool, descending: bool):
    """Decide whether one dominant key should be spread across several buckets, and how.

    A range partition must keep equal keys together, because the result is the ordered
    concatenation of the buckets — so a value holding share `f` of the rows pins `f·N` of
    them on one reducer *however wide the shuffle is*. That is not a slow query, it is a
    query that stops scaling: measured on 600,000 rows with 40% on one key, the busiest
    bucket held ~244,000 rows at 4, 8, 16 and 32 buckets alike, while the even share fell
    from 150,000 to 18,750 — an overload of 1.6x rising to 13x purely by adding workers.

    The way out is that those rows all *tie*. Give the hot value a bucket of its own
    (`isolate_hot_value`), then let each mapper write its share of that bucket to a
    different physical sub-bucket, assigned contiguously by mapper id
    (`hot_sub_bucket`) so concatenating the sub-buckets in order still reproduces mapper
    order. Nothing about the relation changes: the same rows come back, in the same key
    order, with ties in the same order as before.

    Declined for a **descending** sort (see the comment on that branch — the layout is not
    yet right on the Flight reduce), on a key whose boundaries carry a NaN (no total order to
    isolate a value within), and when the hot value would share its bucket with the nulls,
    which the caller routes to whichever end its concatenation puts first — splitting that
    bucket would scatter the nulls through the result.

    Args:
        grids: The `(sampled_values, row_count)` pairs the samplers returned.
        boundaries: The merged boundaries the cut would otherwise use.
        n_buckets: The bucket count the caller sized the shuffle for.
        nulls_first: Whether nulls sort before non-nulls.
        descending: Whether the driver concatenates buckets high to low.

    Returns:
        `(boundaries, logical_buckets, hot_bucket, subs)`, or `None` to partition as usual.
        The caller passes `logical_buckets` to `bucketize` and reduces over
        `logical_buckets + subs - 1` physical buckets.
    """
    import math

    # Ascending only, and this is a real limitation rather than an oversight. A descending
    # sort has the driver read the buckets high to low while the engine's sort keeps ties in
    # input order either way, so the sub-buckets have to be laid out backwards — and laid out
    # backwards the Flight reduce still returns the hot value's rows in an order the unsplit
    # shuffle does not, which `test_diff_distributed_operator_matrix` catches and this author
    # could not explain. The disk path agrees under the same test; a split that were correct
    # on one transport and not the other would be worse than none. So a descending sort keeps
    # the unsplit partition and pays the imbalance, which costs time and never an answer.
    if descending:
        return None
    hot = hot_key_share(grids)
    if hot is None or hot[1] * max(1, n_buckets) < _HOT_SPLIT_OVERLOAD:
        return None
    # Decline on a key that carries NaN. Isolation has to re-sort the boundary list to place
    # the hot value, and NaN has no total order — every comparison against it is false, so
    # `sorted` returns a list whose arrangement depends on the input order rather than on
    # the values. `bucketize` then routes rows by `searchsorted` against a sequence that is
    # not the one it assumes, and rows land in the wrong bucket: the sort comes back with
    # the right keys in the right order and the payload of some ties rearranged, which is
    # exactly the kind of wrongness a key or multiset assertion cannot see. Skew tolerance
    # is a balance optimization and correctness outranks it, so a NaN-bearing key keeps the
    # unsplit partition.
    if not all(math.isfinite(b) for b in boundaries):
        return None
    value, share = hot
    widened, hot_bucket = isolate_hot_value(list(boundaries), value)
    logical = max(n_buckets, len(widened) + 1)
    front = logical - 1 if descending else 0
    null_bucket = front if nulls_first else (logical - 1 - front)
    if hot_bucket == null_bucket:
        return None
    subs = max(2, min(_MAX_HOT_SUBS, math.ceil(share * logical)))
    return widened, logical, hot_bucket, subs
