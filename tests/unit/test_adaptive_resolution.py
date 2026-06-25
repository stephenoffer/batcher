"""`adaptive="auto"` confidence gate: enable stage-by-stage re-opt only when it helps."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.api.adaptive import resolve_adaptive

pytestmark = pytest.mark.unit


def _hub():
    from batcher import core

    return core.default_hub()


def test_auto_enables_for_join_over_uncertain_breaker():
    # A join whose operand is an aggregate output (a breaker, size only guessed) is
    # exactly where measured cardinality flips a build-side / join-order choice.
    left = bt.from_arrow(pa.table({"k": [1, 2, 3, 1, 2], "v": [10, 20, 30, 40, 50]}))
    agg = left.group_by("k").agg(s=col("v").sum())
    right = bt.from_arrow(pa.table({"k": [1, 2, 3], "w": [100, 200, 300]}))
    joined = agg.join(right, on="k")
    assert resolve_adaptive("auto", joined._plan, joined._sources, _hub()) is True


def test_auto_stays_one_shot_without_join():
    # No join → re-optimization has no downstream decision to change.
    ds = bt.from_arrow(pa.table({"x": list(range(100))})).filter(col("x") > 5)
    assert resolve_adaptive("auto", ds._plan, ds._sources, _hub()) is False


def test_auto_stays_one_shot_for_scan_join_scan():
    # A join over two scans is sized from source statistics, not a guess — no benefit.
    sj = bt.from_arrow(pa.table({"k": [1, 2, 3], "a": [1, 2, 3]})).join(
        bt.from_arrow(pa.table({"k": [1, 2], "b": [9, 8]})), on="k"
    )
    assert resolve_adaptive("auto", sj._plan, sj._sources, _hub()) is False


def test_explicit_flag_always_wins():
    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]})).filter(col("x") > 0)
    assert resolve_adaptive(True, ds._plan, ds._sources, _hub()) is True
    assert resolve_adaptive(False, ds._plan, ds._sources, _hub()) is False


def test_auto_result_matches_one_shot():
    # The adaptive-triggering plan must produce the same result as the forced one-shot.
    left = bt.from_arrow(pa.table({"k": [1, 2, 3, 1, 2, 3], "v": [1, 2, 3, 4, 5, 6]}))
    right = bt.from_arrow(pa.table({"k": [1, 2, 3], "w": [10, 20, 30]}))

    def q():
        return left.group_by("k").agg(s=col("v").sum()).join(right, on="k")

    def norm(d):
        return sorted(zip(*[d[c] for c in sorted(d)], strict=True))

    auto = q().collect(adaptive="auto").to_pydict()
    one_shot = q().collect(adaptive=False).to_pydict()
    assert norm(auto) == norm(one_shot)
