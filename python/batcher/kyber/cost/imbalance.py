"""The straggler term — what a shuffle costs when one partition gets most of the rows.

`shuffle` charges an exchange by the bytes it moves, which implicitly assumes those bytes are
spread evenly over the reducers. They are, whenever the partitioning key is. When it is not,
the volume is unchanged and the *elapsed* cost is not: a shuffle finishes when its last reducer
does, so a key whose most common value holds 40% of the rows makes one worker do 40% of the
work while `W - 1` others idle. Charged by volume, that plan is priced identically to a
balanced one, and the optimizer has no reason to prefer the balanced alternative.

## Where this applies, and — more importantly — where it does not

Skew is only a cost where the engine cannot already remove it, and for two of the three
shuffling shapes it can:

* **Aggregate and Distinct** pre-reduce. The mergeable `partial → combine` form collapses a hot
  key to one partial row per worker *before* anything is shuffled, so the hot value arrives at
  its reducer as `W` rows however many rows produced it. These are immune, and charging them a
  skew penalty would push the optimizer away from the one operator shape that solves the
  problem.
* **Joins** are salted. `stats.skew.hot_join_values` already finds the hot values from measured
  most-common-value statistics and `dist` spreads them across reducers. A join whose skew is
  already being mitigated must not also be charged for it, or the cost model would penalize the
  fix.
* **A partitioned window** is neither. Its frame spans a whole partition, so a hot partition
  cannot be split across workers the way a join's hot key can be salted, and it cannot be
  pre-reduced the way an aggregate can — the operator's output is per input row. One worker
  receives the whole hot partition and the stage waits for it. This is the case the term exists
  for, and it is the one shape where a `PARTITION BY` on a low-cardinality skewed column turns
  a linear operator into a serial one.

## The multiplier

A balanced exchange gives each reducer `1/W` of the rows. A key whose largest value holds a
share `f` gives one reducer `max(f, 1/W)`. Since the stage's elapsed cost tracks its largest
reducer, the ratio between the skewed and balanced cases is `max(1, f x W)` — one at `f = 1/W`
and rising linearly with how far the hot value exceeds its fair share.

Measured statistics only. A column with no most-common-value record yields `1.0`, so a cold
metadata store ranks every plan exactly as it did before this existed. The statistic is
approximate (Misra-Gries), which is safe here for the same reason it is safe in `stats.skew`:
it only ever re-ranks plans that all return the same rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["MAX_IMBALANCE", "partition_imbalance"]

#: Ceiling on the multiplier. Past this the skewed plan has already lost to every alternative
#: that does not shuffle on this key, so a larger figure changes no ranking while making the
#: `net` axis unreadable in a decision log — and it would let one approximate statistic
#: dominate a plan total built from many.
MAX_IMBALANCE = 16.0

#: Share below which a value is not treated as hot at all. A most-common value is *always* the
#: largest, so without a floor every column with any statistics at all would carry some
#: multiplier, and the term would become a constant rather than a signal. Ten percent of a
#: relation in one value is the point at which a partition is visibly lopsided.
_MIN_HOT_FRACTION = 0.1


def partition_imbalance(
    frequencies: Sequence[Mapping[str, float]],
    workers: int,
) -> float:
    """How much longer a shuffle on these keys takes than a balanced one.

    Args:
        frequencies: One `{value: share of rows}` mapping per partitioning key, as
            `RelStats.column(key).mcv` records them. An empty mapping is an unmeasured key.
        workers: Reducers the exchange fans out to.

    Returns:
        A multiplier in `[1.0, MAX_IMBALANCE]`. Exactly `1.0` on a single worker, on unmeasured
        keys, and whenever no value exceeds its fair share by enough to matter — so an
        unmeasured column is priced exactly as it was before.
    """
    if workers <= 1 or not frequencies:
        return 1.0
    # The *combined* key is what a shuffle hashes, and a compound key is at least as balanced
    # as its most balanced component: `(hot_country, user_id)` splits finely even though
    # `country` alone does not. Taking the minimum across keys is what stops a compound
    # partitioning from being charged for skew its other column already removes.
    shares = [max(freq.values(), default=0.0) for freq in frequencies]
    hottest = min(shares) if shares else 0.0
    if hottest < _MIN_HOT_FRACTION:
        return 1.0
    fair_share = 1.0 / workers
    if hottest <= fair_share:
        return 1.0
    return min(MAX_IMBALANCE, hottest * workers)
