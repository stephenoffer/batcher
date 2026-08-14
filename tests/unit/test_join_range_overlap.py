"""Only keys inside the intersection of the two key ranges can match.

That bound was applied to the inner-join estimate and to nothing else, and the omission
produced the worst single estimate in the join model. Containment asks only whether
`d_R >= d_L`, which is true of two key domains that barely overlap — so a left key spanning
`[0, 2000)` against a right key spanning `[1000, 3000)` had *every* left key "contained",
the semi-join was priced at the whole of `L`, and the anti-join at **exactly zero rows**
against 11,000 actual.

A zero is the worst answer to be wrong by. Build-side choice, join order, broadcast sizing
and the adaptive gate all read it as "this subtree is empty", and unlike a mis-sized estimate
it does not degrade gracefully. Hence the second rule here: containment and uniformity are
assumptions, and an assumption may not assert emptiness — only a proof may.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
_ROWS = 2000


def _estimate(dataset) -> float:
    stats = [s.statistics() for s in dataset._sources]
    return (
        StatsEstimator(dataset._sources, {}, _CFG, source_stats=stats).estimate(dataset._plan).rows
    )


def _sides(left_span, right_span):
    """Two relations whose keys span the given half-open ranges, one row per key."""
    left = bt.from_pydict({"k": [i % left_span[1] for i in range(_ROWS)]})
    right = bt.from_pydict({"k": list(range(*right_span))})
    return left, right


@pytest.fixture
def half_overlapping():
    # Left keys in [0, 2000), right keys in [1000, 3000): half of the left keys can match.
    return _sides((0, 2000), (1000, 3000))


def test_a_semi_join_cannot_keep_more_than_the_overlap(half_overlapping):
    left, right = half_overlapping
    semi = _estimate(left.join(right, on="k", how="semi"))
    assert semi < _ROWS, "containment alone claimed every left key was present"
    assert semi == pytest.approx(_ROWS * 0.5, rel=0.15)


def test_an_anti_join_is_not_estimated_empty_when_half_the_keys_are_unmatched(half_overlapping):
    left, right = half_overlapping
    anti = _estimate(left.join(right, on="k", how="anti"))
    executed = left.join(right, on="k", how="anti").count()
    assert anti > 0.0
    assert anti == pytest.approx(executed, rel=0.25)


def test_semi_and_anti_still_partition_the_left_side(half_overlapping):
    """The two must sum to `|L|`: every left row either matches or does not."""
    left, right = half_overlapping
    semi = _estimate(left.join(right, on="k", how="semi"))
    anti = _estimate(left.join(right, on="k", how="anti"))
    assert semi + anti == pytest.approx(_ROWS)


def test_an_anti_join_never_claims_emptiness_from_an_assumption():
    """Full containment gives `semi == |L|`, but zero is a proof and this is an estimate."""
    left = bt.from_pydict({"k": [i % 100 for i in range(_ROWS)]})
    right = bt.from_pydict({"k": list(range(100))})
    anti = _estimate(left.join(right, on="k", how="anti"))
    assert anti > 0.0, "a ratio asserted an empty subtree"
    assert anti <= 1.0, "the floor must stay negligible, not invent rows"


def test_an_empty_left_side_still_gives_an_empty_anti_join():
    """The floor applies to an assumption, never over a proof."""
    left = bt.from_pydict({"k": []})
    right = bt.from_pydict({"k": list(range(100))})
    assert _estimate(left.join(right, on="k", how="anti")) == pytest.approx(0.0)


def test_provably_disjoint_ranges_keep_their_certainties():
    """Disjoint keys: nothing matches, so a semi keeps none and an anti keeps everything."""
    left = bt.from_pydict({"k": list(range(500))})
    right = bt.from_pydict({"k": list(range(1000, 1500))})
    assert _estimate(left.join(right, on="k", how="semi")) == pytest.approx(0.0)
    anti = _estimate(left.join(right, on="k", how="anti"))
    assert anti == pytest.approx(500.0, rel=0.05)


def test_an_outer_join_counts_the_rows_the_overlap_leaves_unmatched(half_overlapping):
    """A LEFT JOIN emits the matched rows plus a null-extended row per unmatched left row."""
    left, right = half_overlapping
    outer = _estimate(left.join(right, on="k", how="left"))
    executed = left.join(right, on="k", how="left").count()
    assert outer == pytest.approx(executed, rel=0.25)
    assert outer >= _ROWS  # an outer join preserves its outer side


@pytest.mark.parametrize("how", ["semi", "anti", "left", "inner"])
def test_the_estimate_tracks_the_executed_row_count(how, half_overlapping):
    left, right = half_overlapping
    dataset = left.join(right, on="k", how=how)
    estimated, executed = _estimate(dataset), dataset.count()
    assert estimated == pytest.approx(executed, rel=0.3), (
        f"{how}: estimated {estimated:.1f} against {executed} executed rows"
    )
