"""Unit tests for Kyber's broadcast join selection (SELECTION phase).

The rule is a *physical* choice: it never changes the relation, only the planned
data movement. These tests assert the plan-shape decision (broadcast vs hash) from
estimated input sizes; the differential suite proves the result is still correct.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.rules.selection import adaptive_build_side
from batcher.plan.logical import Join, JoinOutputCol, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit


class _Source:
    """Minimal source stub exposing the row_count() the estimator reads."""

    def __init__(self, rows: int) -> None:
        self._rows = rows

    def row_count(self) -> int:
        return self._rows


def _scan(source_id: int, columns: list[str]) -> Scan:
    return Scan(source_id, SchemaRef(pa.schema([pa.field(c, pa.int64()) for c in columns])))


def _join(left: Scan, right: Scan, join_type: str = "inner") -> Join:
    return Join(
        left=left,
        right=right,
        left_keys=("k",),
        right_keys=("k",),
        join_type=join_type,
        output=(
            JoinOutputCol("left", "k", "k"),
            JoinOutputCol("right", "v", "v"),
        ),
    )


def _plan_with_sizes(
    left_rows: int,
    right_rows: int,
    join_type: str = "inner",
    learned: dict | None = None,
) -> tuple[Join, CardinalityEstimator]:
    left = _scan(0, ["k"])
    right = _scan(1, ["k", "v"])
    estimator = CardinalityEstimator([_Source(left_rows), _Source(right_rows)], learned)
    return _join(left, right, join_type), estimator


def test_small_build_side_picks_broadcast():
    plan, est = _plan_with_sizes(left_rows=10_000_000, right_rows=100)
    out, decisions = adaptive_build_side(plan, est)
    assert out.strategy == "broadcast"
    assert decisions[-1].broadcast is True


def test_large_build_side_stays_hash():
    # 500k rows × 64 B ≈ 32 MiB > the 10 MiB broadcast budget, and below the
    # 1M-row sort-merge floor → plain hash.
    big = 500_000
    plan, est = _plan_with_sizes(left_rows=big, right_rows=big)
    out, decisions = adaptive_build_side(plan, est)
    assert out.strategy == "hash"
    assert decisions[-1].broadcast is False


def test_wide_payload_build_side_not_broadcast():
    # Few rows but a wide measured payload: 100 rows × ~200 KB/row = ~20 MB, over
    # the broadcast budget, so byte-aware selection must NOT broadcast it (a
    # row-count check would have, wrongly).
    learned = {"__column_avg_bytes__": {"v": 200_000.0}}
    plan, est = _plan_with_sizes(left_rows=10_000_000, right_rows=100, learned=learned)
    out, decisions = adaptive_build_side(plan, est)
    assert decisions[-1].broadcast is False
    assert out.strategy == "hash"


def test_narrow_small_build_side_still_broadcasts():
    # The same 100-row build side with no wide column stays broadcast-eligible.
    plan, est = _plan_with_sizes(left_rows=10_000_000, right_rows=100)
    out, decisions = adaptive_build_side(plan, est)
    assert decisions[-1].broadcast is True
    assert out.strategy == "broadcast"


def test_inner_join_swaps_then_broadcasts_small_left():
    # Left is the small side: an inner join swaps it to the build (right) side,
    # then broadcasts it. The swapped result still broadcasts.
    plan, est = _plan_with_sizes(left_rows=50, right_rows=10_000_000)
    out, decisions = adaptive_build_side(plan, est)
    assert decisions[-1].swapped is True
    assert out.strategy == "broadcast"


def test_left_join_does_not_swap_but_broadcasts_small_right():
    # Outer joins never swap, but a small right side is still broadcast-eligible.
    plan, est = _plan_with_sizes(left_rows=10_000_000, right_rows=100, join_type="left")
    out, decisions = adaptive_build_side(plan, est)
    assert decisions[-1].swapped is False
    assert out.strategy == "broadcast"
    assert out.join_type == "left"


def test_two_large_sides_pick_sort_merge():
    from batcher.kyber.rules.selection import SORT_MERGE_MIN_ROWS

    big = int(SORT_MERGE_MIN_ROWS * 2)
    plan, est = _plan_with_sizes(left_rows=big, right_rows=big)
    out, _decisions = adaptive_build_side(plan, est)
    assert out.strategy == "sort_merge"


def test_medium_sides_stay_hash():
    # Above the broadcast cutoff (×64 B ≈ 33 MiB > 10 MiB) but below the
    # sort-merge floor → plain hash.
    from batcher.kyber.rules.selection import SORT_MERGE_MIN_ROWS

    medium = int((163_840 + SORT_MERGE_MIN_ROWS) / 2)
    plan, est = _plan_with_sizes(left_rows=medium, right_rows=medium)
    out, _decisions = adaptive_build_side(plan, est)
    assert out.strategy == "hash"


def test_already_sorted_medium_sides_stay_hash():
    # Two medium inputs (over the broadcast budget, under the 1M-row sort-merge floor)
    # that already arrive ordered on the join key still pick HASH, not sort-merge:
    # benchmarking showed SMJ's RowConverter encoding loses to hash even when its sort
    # is skipped, so preferring SMJ for sorted inputs was a regression and reverted.
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Sort, SortKeySpec

    left = Sort(_scan(0, ["k"]), (SortKeySpec(Col("k")),))
    right = Sort(_scan(1, ["k", "v"]), (SortKeySpec(Col("k")),))
    join = Join(
        left=left,
        right=right,
        left_keys=("k",),
        right_keys=("k",),
        join_type="inner",
        output=(JoinOutputCol("left", "k", "k"), JoinOutputCol("right", "v", "v")),
    )
    est = CardinalityEstimator([_Source(500_000), _Source(500_000)], None)
    out, _decisions = adaptive_build_side(join, est)
    assert out.strategy == "hash"
