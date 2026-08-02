"""Range boundaries must balance by *rows*, not by splits.

`merge_boundaries` receives a `(sampled_cdf, row_count)` pair per worker and used to drop the
row count on the floor, merging the sampled CDFs unweighted. That is only right when every
split holds about the same number of rows, and the sample pass runs the *mapped* plan — so a
pushed-down predicate that keeps 90% of one split and 1% of another leaves their post-filter
counts orders of magnitude apart. The boundaries then split by the number of splits on each
side instead of the number of rows, and one reducer receives nearly everything.

A string key was worse than merely unweighted. Its sampler caps at `MAX_BOUNDARY_SAMPLE`
(65,536) values, so a 100M-row split contributes 65,536 samples and a 70K-row split
contributes 70,000: the *smaller* split outvoted the larger one.

None of this is visible to a correctness suite. Every row still lands in exactly one bucket,
equal keys still co-locate, and the concatenation is still globally sorted — the answer is
right and one reducer does all the work. So these tests assert the *balance* of the resulting
buckets, which is the property that was broken, and they run with no cluster and no native
engine so the gate is reachable from CI.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from batcher.dist.executors.partition_io.ranges import SAMPLE_PROBS, merge_boundaries

pytestmark = pytest.mark.unit

#: Worst tolerated ratio between the heaviest and lightest bucket. Sampling is approximate, so
#: this is not 1.0 — but the defect it guards produced ratios in the hundreds.
MAX_IMBALANCE = 1.5


def _uniform_grid(low: float, high: float) -> list[float]:
    """The sampled inverse-CDF of a uniform distribution over `[low, high)`."""
    return [low + (high - low) * p for p in SAMPLE_PROBS]


def _bucket_loads(splits: list[tuple[float, float, int]], boundaries: list[float]) -> list[int]:
    """Rows landing in each bucket when `splits` are routed by `boundaries`.

    Each split is `(low, high, rows)` uniform over its range, which is exactly what
    `_uniform_grid` describes, so this models the routing the Rust partitioner performs.
    """
    edges = [-np.inf, *boundaries, np.inf]
    loads = []
    for lo_edge, hi_edge in itertools.pairwise(edges):
        total = 0.0
        for low, high, rows in splits:
            overlap = min(high, hi_edge) - max(low, lo_edge)
            total += rows * max(0.0, overlap) / (high - low)
        loads.append(round(total))
    return loads


def _imbalance(loads: list[int]) -> float:
    return max(loads) / max(1, min(loads))


def test_a_dominant_split_does_not_capture_a_whole_bucket():
    """1M rows in one split against 1K in another must still split by rows.

    Unweighted, the single boundary lands at the junction between the two splits' ranges —
    both contribute the same number of sample points — giving one reducer 1,000x the other's
    work. Weighted, it lands inside the dominant split and halves it.
    """
    splits = [(0.0, 1000.0, 1_000_000), (1000.0, 2000.0, 1_000)]
    grids = [(_uniform_grid(low, high), rows) for low, high, rows in splits]

    boundaries = merge_boundaries(grids, 2)

    assert len(boundaries) == 1
    assert 400.0 < boundaries[0] < 600.0, "the cut must fall inside the dominant split"
    assert _imbalance(_bucket_loads(splits, boundaries)) <= MAX_IMBALANCE


def test_many_unequal_splits_balance_across_many_buckets():
    """Eight splits spanning three orders of magnitude in size, into eight buckets."""
    sizes = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    splits = [(float(i) * 100.0, float(i + 1) * 100.0, n) for i, n in enumerate(sizes)]
    grids = [(_uniform_grid(low, high), rows) for low, high, rows in splits]

    boundaries = merge_boundaries(grids, 8)

    assert boundaries == sorted(boundaries), "boundaries must stay ascending"
    assert len(boundaries) == len(set(boundaries)), "boundaries must stay deduplicated"
    assert _imbalance(_bucket_loads(splits, boundaries)) <= MAX_IMBALANCE


def test_equal_splits_are_unchanged_by_weighting():
    """The uniform case must still cut evenly — weighting reduces to the old behavior."""
    splits = [(float(i) * 100.0, float(i + 1) * 100.0, 50_000) for i in range(4)]
    grids = [(_uniform_grid(low, high), rows) for low, high, rows in splits]

    boundaries = merge_boundaries(grids, 4)

    assert _imbalance(_bucket_loads(splits, boundaries)) <= 1.1
    assert boundaries == pytest.approx([100.0, 200.0, 300.0], abs=5.0)


def test_a_string_key_is_weighted_too():
    """The lexical merge weights by rows, where the sample cap otherwise inverts the vote.

    The big split contributes *fewer* samples than the small one here — the shape the
    `MAX_BOUNDARY_SAMPLE` cap produces on a real cluster — so an unweighted merge places the
    cut inside the small split and hands the big one to a single reducer.
    """
    big = ([f"a{i:04d}" for i in range(40)], 1_000_000)
    small = ([f"b{i:04d}" for i in range(80)], 1_000)

    boundaries = merge_boundaries([big, small], 2)

    assert len(boundaries) == 1
    assert boundaries[0].startswith("a"), "the cut must subdivide the dominant split"


def test_an_empty_or_zero_row_grid_is_still_dropped():
    """A split that sampled nothing says nothing about the distribution."""
    assert merge_boundaries([([], 0)], 4) == []
    assert merge_boundaries([([1.0, 2.0], 0), ([], 5)], 4) == []
    assert merge_boundaries([(_uniform_grid(0.0, 10.0), 100)], 1) == []
