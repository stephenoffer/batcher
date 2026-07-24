"""What a `UNION ALL` can and cannot claim about its columns.

Concatenation preserves more than most operators — several statistics merge exactly — but
the one it cannot know is how much the branches' *value sets* overlap. These tests pin both
halves: the exact merges, and the estimate that must never masquerade as one.
"""

from __future__ import annotations

import pytest

from batcher.kyber.stats.columns import union_columns
from batcher.plan.stats import ColumnStat, Provenance, RelStats

pytestmark = pytest.mark.unit


def _branch(rows: float, **col) -> RelStats:
    return RelStats(rows, Provenance.EXACT, {"x": ColumnStat(provenance=Provenance.EXACT, **col)})


def test_the_union_distinct_count_is_never_exact():
    """Two exact branch counts still give only an estimate of the union's.

    The branches may share every value or none, and nothing here measures which. Inheriting
    an EXACT bundle tag would let `count_distinct` answer a `UNION ALL` from a model — the
    exact failure the provenance discipline exists to prevent.
    """
    branches = [_branch(100.0, ndv=50), _branch(100.0, ndv=50)]
    merged = union_columns(branches, ["x"])["x"]
    assert merged.ndv is not None
    assert not merged.ndv_is_exact
    # ...and it respects the Fréchet bounds: at least the largest branch, at most the sum.
    assert 50 <= merged.ndv <= 100


def test_additive_and_weighted_statistics_merge_exactly():
    branches = [
        _branch(100.0, null_count=10, total_sum=1000.0, mean=10.0, avg_bytes=8.0),
        _branch(300.0, null_count=5, total_sum=6000.0, mean=20.0, avg_bytes=16.0),
    ]
    merged = union_columns(branches, ["x"])["x"]
    assert merged.null_count == 15  # additive
    assert merged.total_sum == pytest.approx(7000.0)  # additive
    # Row-weighted, not the unweighted average of 15.0: the second branch is three times
    # the size of the first.
    assert merged.mean == pytest.approx((100 * 10 + 300 * 20) / 400)
    assert merged.avg_bytes == pytest.approx((100 * 8 + 300 * 16) / 400)


def test_a_statistic_missing_from_one_branch_does_not_merge():
    """A mean over the branches that happen to have one is a different relation's mean."""
    branches = [_branch(100.0, mean=10.0), _branch(300.0)]
    merged = union_columns(branches, ["x"])["x"]
    assert merged.mean is None
    assert merged.total_sum is None
