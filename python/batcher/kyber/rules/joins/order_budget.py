"""How much join-order search a query is worth — the budget the DP search spends against.

Join reordering is the one optimizer decision whose *search* is expensive enough to show up
in a query's wall clock, so the question "how hard should we look?" has to be answered per
query rather than by a constant. This module answers it, in the unit the DP search already
counts: **evaluated join pairs**, a candidate `(left half, right half)` split that gets built,
estimated and costed.

## Why leaf count is the wrong axis, measured

The search was budgeted on leaf count — the connected-subset DP up to 20 leaves, greedy beyond
— with a flat 200,000-pair cap under it. Leaf count does not predict search work, because the
*shape* of the join graph decides it. Measured on this machine against that search, over 1,000
rows, with the plan cache off:

| join graph      | leaves | pairs evaluated | planning time |
|-----------------|-------:|----------------:|--------------:|
| chain           |     14 |             455 |       0.067 s |
| star            |     14 |          53,248 |      10.916 s |
| star            |     15 |         114,688 |      25.458 s |

Same leaf count, 117x the work — so a leaf cap set where the chain is affordable lets the star
through, and one set where the star is affordable refuses the chain. The flat pair cap is the
right *unit* and the wrong *size*: 200,000 pairs is ~30 s of planning at the rate measured
below, and every pair of it is discarded when the cap trips and greedy answers anyway.

## Why the data size is the other axis

None of those figures move with the data. A 15-leaf star over a thousand rows executes in
milliseconds and planned for **25.5 seconds** — the whole of the small-query mandate spent
searching for an order whose worst case is microseconds away from its best. The same search
over a petabyte would be a bargain. So the budget is a share of what the query is *estimated
to cost*, which is the only quantity that separates those two cases.

Both axes together are the point: the budget adapts to the dataset (through the estimated
cost, which the cardinality loop sharpens across runs) and to the graph's density (through
pairs rather than leaves), with no user configuration in either direction.

It picks up the *cluster* for free, and that is worth noting because nothing here knows what
a worker is. The cost model already takes the `HardwareProfile`, so a plan that will shuffle
across a fleet is priced above the same plan on one node and is therefore worth searching
harder — measured at 14,773 pairs single-node against 29,227 on a 64-worker fleet, for one
identical two-billion-row join. Pricing the decision rather than encoding it is what makes
that come out right without a rule for it.

## The arithmetic, and where each constant was measured

`_PAIR_SECONDS` and `_UNIT_SECONDS` are the two conversions that turn an estimated *execution*
cost into an affordable *search* size. Both were measured here rather than assumed, which is
the difference that matters: the flat cap they replace permitted ~30 s of planning because it
was sized for a pair far cheaper than a pair measurably is.
"""

from __future__ import annotations

import math

from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.logical import LogicalPlan

__all__ = [
    "max_pairs",
    "predicted_pairs",
    "search_pair_budget",
]

#: Wall-clock seconds one evaluated join pair costs the planner. A pair builds a `Join` node,
#: estimates its cardinality and prices it, so it is far from free. Measured across eighteen
#: configurations spanning 6..14 leaves on star and chain graphs: 113 to 168 us, with no trend
#: in leaf count or graph density. That stability is what makes a pair the right unit to budget
#: in — the same count means the same time whatever the graph looks like.
_PAIR_SECONDS = 1.5e-4

#: Execution seconds one cost-model unit stands for. Measured by pricing three shapes (a
#: four-way join, a grouped aggregate, a filtered projection) at 100k / 1M / 4M rows and
#: dividing measured `collect()` time by `CostModel.cost(...).total()`. Small queries read
#: high (1.1e-9) because fixed overhead dominates and large ones converge to ~1.7e-10; the
#: asymptote is the honest figure, because it is the large queries whose budget this decides.
#: A small query lands under the floor below on any value in that range, so the choice within
#: it changes nothing that matters.
_UNIT_SECONDS = 1.7e-10

#: Share of a query's estimated execution time the optimizer may spend searching for a better
#: join order. A tenth is generous in the direction that pays: a better order routinely moves
#: a multi-way join by 10-100x, so a 10% planning premium buying even a 2x plan is a large
#: win, while the same premium wasted is bounded and small by construction.
_PLANNING_SHARE = 0.10

#: Pairs every query may spend regardless of how cheap it is estimated to be, so that a query
#: the cost model cannot justify searching still gets a real search rather than none.
#:
#: 512 pairs covers the full DP for a star up to 7 leaves and a chain up to 15 — measured, not
#: derived, since the reach depends on graph density. A *cheap* query outside that keeps its
#: result (join order is semantics-preserving) and takes the greedy order instead, which is
#: the intended trade and not a regression to be tuned away: the plans it gives up are worth
#: at most a share of a query the model estimates at under a second, while the search it gives
#: up cost 25.5 s on the 15-leaf star above. A query big enough for the difference to matter
#: is priced above this floor and searches accordingly.
#:
#: What it costs is now small in both directions. Clearing the floor is decided by the
#: prediction below *before* any pair is evaluated, so a graph too dense to search bails
#: having spent nothing at all; the worst fixed overhead left is the largest search that does
#: fit, a 15-leaf chain at ~77 ms. Sparse graphs — the shape real queries have — are therefore
#: unaffected: measured against the flat cap, a 12-to-15-leaf chain plans in the same time to
#: within noise, and keeps the same plan.
_MIN_PAIRS = 512

#: Pairs no query may exceed however large it is estimated to be. Deliberately the *same*
#: number as the flat cap this replaces, so that nothing which could afford a search before
#: gets a smaller one now: the change is a restriction on waste, not on search. What moves is
#: that the ceiling has to be earned — 200,000 pairs is ~30 s of planning, which the share
#: above grants only to a query estimated to run ~300 s. Every query used to be allowed it
#: unconditionally, which is how a 15-leaf star over a thousand rows came to spend 25.5 s
#: choosing between orders that were microseconds apart.
_MAX_PAIRS = 200_000

#: Evaluated pairs per connected subset, for predicting a search's size before running it.
#: The search evaluates a pair only where *both* halves of a split are themselves connected,
#: which is far fewer splits than a subset has: the closed form over-predicts a 12-leaf star
#: 15x (175,099 against 11,264 measured). The ratio to the connected-subset count, by
#: contrast, is stable across densities — 4.3 for a 14-leaf chain, 5.5 for a 12-leaf star,
#: 6.5 for a 14-leaf star.
#:
#: **This takes the bottom of that band rather than the top, because the two errors are not
#: symmetric.** Over-predicting sends a search that would have fit straight to greedy, and the
#: plan quality is gone with nothing spent to learn it was affordable. Under-predicting starts
#: a search that does not fit, and costs at most the budget before the incremental bail in the
#: search itself stops it — bounded, small, and self-correcting. Rounding up cost real plans:
#: at 7 the floor stopped searching an 8-leaf star whose actual 448 pairs fit inside it.
_PAIRS_PER_SUBSET = 4


def search_pair_budget(region: LogicalPlan, ctx: OptimizerContext) -> int:
    """Evaluated join pairs the search for `region`'s order may spend.

    A share of the region's own estimated execution cost, converted through the two measured
    rates at the top of this module and clamped to `[_MIN_PAIRS, _MAX_PAIRS]`. The region is
    priced *as written* — the order the API happened to build — which is the right reference
    in both directions: an as-written order that is already cheap has little for a search to
    win, and an expensive one is exactly where searching pays.

    A cost the model cannot produce yields the floor rather than the ceiling, so an
    unpriceable region searches like a small query instead of an unbounded one.

    Args:
        region: The join subtree whose order is being chosen.
        ctx: The optimizer context, for the shared cost model and estimator.

    Returns:
        A pair budget of at least `_MIN_PAIRS` and at most `_MAX_PAIRS`.
    """
    try:
        units = ctx.costs().cost(region).total()
    except Exception:
        return _MIN_PAIRS
    if not math.isfinite(units) or units <= 0.0:
        return _MIN_PAIRS
    pairs = units * _UNIT_SECONDS * _PLANNING_SHARE / _PAIR_SECONDS
    if not math.isfinite(pairs):
        return _MAX_PAIRS
    return max(_MIN_PAIRS, min(_MAX_PAIRS, int(pairs)))


def predicted_pairs(subsets: int) -> int:
    """Pairs a DP search over `subsets` connected subsets is expected to evaluate.

    The estimate a caller triages on before searching, so a graph too dense to search within
    budget goes straight to greedy instead of burning the budget to discover that. See
    `_PAIRS_PER_SUBSET` for why the connected-subset count predicts this and the closed form
    over the split space does not.

    Args:
        subsets: How many connected subsets the join graph has.

    Returns:
        The predicted evaluated-pair count.
    """
    return subsets * _PAIRS_PER_SUBSET


def max_pairs() -> int:
    """The largest search any query may be granted, whatever it is estimated to cost.

    The value a caller with no cost estimate falls back to — the oracle test, which wants the
    search unbounded, and any caller holding no priced region. It is a ceiling rather than an
    absence of one so that even an unpriced search stays bounded.

    Returns:
        The pair ceiling.
    """
    return _MAX_PAIRS
