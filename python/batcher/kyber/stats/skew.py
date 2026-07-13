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
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.stats import RelStats

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import Join

__all__ = ["hot_join_values"]


def _hot(stats: RelStats, key: str, min_fraction: float) -> set[str]:
    """The values of `key` measured to hold at least `min_fraction` of the relation's rows.

    Read off the relation's *propagated* column statistics, so a key that reaches the join
    through filters and projections is still answered by the statistics of the column it
    actually came from — and by the right table's, not another table's column of the same
    name.
    """
    mcv = stats.column(key).mcv or {}
    return {value for value, fraction in mcv.items() if fraction >= min_fraction}


def hot_join_values(
    join: Join,
    sources: list,
    hub: MetadataHub | None,
    min_fraction: float,
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
    hot = _hot(left, join.left_keys[0], min_fraction) | _hot(
        right, join.right_keys[0], min_fraction
    )
    return sorted(hot)
