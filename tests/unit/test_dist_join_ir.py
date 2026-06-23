"""Unit tests for the distributed join reducer IR construction.

Regression guard: the per-bucket reducer IR must carry the planner's physical
`strategy` through instead of silently dropping it (the bug that left the
distributed path always shuffling both sides even when Kyber chose broadcast).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.executors.join import _join_reducer_ir
from batcher.plan.logical import Join, JoinOutputCol, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit


def _scan(source_id: int, columns: list[str]) -> Scan:
    return Scan(source_id, SchemaRef(pa.schema([pa.field(c, pa.int64()) for c in columns])))


def _join(strategy: str) -> Join:
    return Join(
        left=_scan(0, ["k"]),
        right=_scan(1, ["k", "v"]),
        left_keys=("k",),
        right_keys=("k",),
        join_type="inner",
        output=(JoinOutputCol("left", "k", "k"), JoinOutputCol("right", "v", "v")),
        strategy=strategy,
    )


@pytest.mark.parametrize("strategy", ["hash", "broadcast", "sort_merge"])
def test_reducer_ir_carries_strategy(strategy):
    ir = _join_reducer_ir(_join(strategy))
    assert ir["strategy"] == strategy
    # Inputs are substituted with the co-partitioned bucket scans.
    assert ir["left"] == {"op": "scan", "source_id": 0}
    assert ir["right"] == {"op": "scan", "source_id": 1}
    assert ir["join_type"] == "inner"
