"""Join-key skew that Kyber already knows — no detection pass, no prior run of the shape.

A shuffle join sends every row with the same key to one reducer, so a single value holding
a large share of the rows overloads one worker while the rest idle. Salting fixes it (spread
the hot key's probe rows across `salt` reducers, replicate its build rows to each), and it
is result-preserving — it only changes *which* reducer does the work. The whole problem is
*knowing which values are hot*.

The distributed join has two ways to find out, and both arrive late:

  - a **detection pre-pass**, a full distributed Misra-Gries scan of both sides, which
    costs an extra pass over the data and only runs when the user has opted in; and
  - a **shape-keyed learned set**, free but keyed by the exact `(left_ir, right_ir, keys,
    type)` of the join — so it says nothing about a query shape that has not run before,
    even when the very same column was measured as skewed by a different query yesterday.

This module supplies the third, earliest answer: the column's **measured most-common
values**, which the metadata loop already records per `(source, column)` from the base data.
If `cust_id = 7` is 47% of the rows, that is a property of the *column*, not of the query —
so it is known before this join has ever run, for every query shape that joins on it, at no
cost. Kyber decides from the statistics it owns; `dist` schedules the salting.

The estimate is approximate (Misra-Gries) and only ever *engages* salting, which cannot
change the result — so a false positive costs a little extra fan-out and a false negative
just leaves the old behavior in place. That asymmetry is why an approximate statistic is
safe to act on here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.kyber.learning import load_learned_stats
from batcher.kyber.stats.distribution import mcv_join_rows
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.stats import RelStats

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import Join

__all__ = ["hot_join_values"]


def _frequencies(stats: RelStats, key: str) -> dict[str, float]:
    """`{value: measured share of rows}` for `key`.

    Read off the relation's *propagated* column statistics, so a key that reaches the join
    through filters and projections is still answered by the statistics of the column it
    actually came from — and by the right table's, not another table's column of the same
    name.
    """
    return dict(stats.column(key).mcv or {})


def _hot(stats: RelStats, key: str, min_fraction: float) -> set[str]:
    """The values of `key` measured to hold at least `min_fraction` of the relation's rows."""
    return {v for v, fraction in _frequencies(stats, key).items() if fraction >= min_fraction}


def _overloading(
    left_freq: dict[str, float],
    right_freq: dict[str, float],
    left_ndv: float | None,
    right_ndv: float | None,
    partitions: int,
) -> set[str]:
    """Values whose reducer would carry more than its share of the shuffle, at `partitions` wide.

    A fixed fraction is the wrong test for skew, twice over.

    **It ignores the shuffle's width.** Every reducer's fair share is `1/P` of the work, so a
    value at 5% is harmless across 4 reducers and a 10x straggler across 200. The threshold
    that matters is `f > 1/P`, and it moves with the cluster.

    **It ignores that a join multiplies.** The reducer owning `v` receives
    `f_L(v)|L| + f_R(v)|R|` input rows but *emits* `f_L(v)·f_R(v)·|L||R|`, and the join's whole
    output is `S·|L||R|` where `S = Σ_u f_L(u)f_R(u)` is the total match probability — a number
    far below 1 (about `1/d` for a uniform key). So `v`'s share of the output is
    `f_L(v)·f_R(v)/S`, which for a key with a million distinct values is amplified by six
    orders of magnitude relative to the raw product. Two frequencies that look unremarkable on
    their own inputs can therefore hand one reducer most of the join's output, and no test on
    either side alone can see it — which is exactly the straggler that only appears at scale.

    `S` is the same skew-plus-residual total the cardinality estimator computes for this join
    (`mcv_join_rows` at unit sizes), so the two cannot disagree about how big the join is.
    Without a frequency table on both sides there is no product to evaluate and only the input
    test applies.
    """
    if partitions <= 1:
        return set()
    share = 1.0 / partitions
    hot = {v for v, f in left_freq.items() if f > share}
    hot |= {v for v, f in right_freq.items() if f > share}
    total = mcv_join_rows(1.0, 1.0, left_freq, right_freq, left_ndv, right_ndv)
    if total is not None and total > 0.0:
        for value, f_left in left_freq.items():
            f_right = right_freq.get(value)
            if f_right is not None and (f_left * f_right) / total > share:
                hot.add(value)  # unremarkable inputs, an overloaded output
    return hot


def hot_join_values(
    join: Join,
    sources: list,
    hub: MetadataHub | None,
    min_fraction: float,
    partitions: int | None = None,
) -> list[str]:
    """The known-hot values of a single-key join's key, from measured column statistics.

    Values are returned as strings, the form the engine's salted partitioner keys on.
    Empty when the join is not single-key, when nothing has been measured, or when no value
    clears `min_fraction` — in every case the caller falls back to its existing behavior.

    Args:
        join: The join whose key is being checked for skew.
        sources: The bound inputs, indexed by a `Scan`'s `source_id`.
        hub: The metadata hub holding the measured column statistics.
        min_fraction: The share of rows a value must hold to count as hot.
        partitions: How many reduce partitions the shuffle fans into. When given, a value
            also counts as hot if it would overload one reducer relative to an even split —
            including through the *product* of two individually unremarkable frequencies
            (see `_overloading`). Omitting it keeps the plain fraction test.

    Returns:
        The hot key values, sorted, as strings; empty when none are known.
    """
    if len(join.left_keys) != 1 or len(join.right_keys) != 1:
        return []  # salting is defined for a single key
    try:
        estimator = StatsEstimator(sources, load_learned_stats(hub))
        left = estimator.estimate(join.left)
        right = estimator.estimate(join.right)
    except Exception:  # pragma: no cover - a statistics failure must never fail a join
        return []
    # Either side being skewed overloads the reducer that owns the key, so both count.
    lk, rk = join.left_keys[0], join.right_keys[0]
    hot = _hot(left, lk, min_fraction) | _hot(right, rk, min_fraction)
    if partitions is not None:
        hot |= _overloading(
            _frequencies(left, lk),
            _frequencies(right, rk),
            left.column(lk).ndv,
            right.column(rk).ndv,
            partitions,
        )
    return sorted(hot)
