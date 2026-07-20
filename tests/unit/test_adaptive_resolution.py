"""`adaptive="auto"`: the two gates that decide whether stage-by-stage re-opt runs.

`resolve_adaptive("auto", ...)` turns re-optimization on only when it could pay for itself,
and it asks two independent questions:

1. **Is the query big enough?** Re-opt trades a per-stage materialize and re-plan for a
   better downstream join choice. Below `_ADAPTIVE_MIN_INPUT_ROWS` the one-shot plan is
   already fast and the re-plan is pure overhead.
2. **Would measuring actually change anything?** Only a join whose operand comes out of a
   pipeline breaker with a *guessed* size (`Provenance.DEFAULT`) can have its build-side or
   join-order choice flipped by a real measurement. A join sized from source statistics
   gains nothing.

The size gate runs first and short-circuits. That matters for this file: with realistic
fixture tables (a handful of rows) *every* plan is below the threshold, so a test that just
asserts `False` passes without the confidence gate ever running — it would keep passing if
the confidence gate were deleted. So the confidence tests lower the threshold explicitly,
and one test pins the size gate on its own.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.api.adaptive import gating as adaptive_mod
from batcher.api.adaptive import resolve_adaptive

pytestmark = pytest.mark.unit


def _hub():
    from batcher import core

    return core.default_hub()


@pytest.fixture
def any_size(monkeypatch):
    """Lower the size gate so the *confidence* gate is what the test measures."""
    monkeypatch.setattr(adaptive_mod, "_ADAPTIVE_MIN_INPUT_ROWS", 1)


def _join_over_a_breaker():
    """A join whose left operand is an aggregate output: a breaker, size only guessed."""
    left = bt.from_arrow(pa.table({"k": [1, 2, 3, 1, 2], "v": [10, 20, 30, 40, 50]}))
    right = bt.from_arrow(pa.table({"k": [1, 2, 3], "w": [100, 200, 300]}))
    return left.group_by("k").agg(s=col("v").sum()).join(right, on="k")


def test_auto_enables_for_join_over_uncertain_breaker(any_size):
    # Measured cardinality here flips a build-side / join-order choice, so re-opt earns
    # its cost — provided the query is big enough, which `any_size` stands in for.
    joined = _join_over_a_breaker()
    assert resolve_adaptive("auto", joined._plan, joined._sources, _hub()) is True


def test_auto_stays_one_shot_below_the_size_threshold():
    # The same plan, without lowering the gate. A few rows is not worth a re-plan, however
    # uncertain the operand is. This is the gate that makes the test above need `any_size`.
    joined = _join_over_a_breaker()
    assert resolve_adaptive("auto", joined._plan, joined._sources, _hub()) is False


def test_auto_stays_one_shot_without_join(any_size):
    # No join → re-optimization has no downstream decision to change.
    ds = bt.from_arrow(pa.table({"x": list(range(100))})).filter(col("x") > 5)
    assert resolve_adaptive("auto", ds._plan, ds._sources, _hub()) is False


def test_auto_stays_one_shot_for_scan_join_scan(any_size):
    # A join over two scans is sized from source statistics, not a guess — no benefit.
    sj = bt.from_arrow(pa.table({"k": [1, 2, 3], "a": [1, 2, 3]})).join(
        bt.from_arrow(pa.table({"k": [1, 2], "b": [9, 8]})), on="k"
    )
    assert resolve_adaptive("auto", sj._plan, sj._sources, _hub()) is False


def test_explicit_flag_always_wins():
    # Neither gate is consulted: an explicit choice is the caller's to make.
    joined = _join_over_a_breaker()
    assert resolve_adaptive(True, joined._plan, joined._sources, _hub()) is True
    assert resolve_adaptive(False, joined._plan, joined._sources, _hub()) is False


def test_auto_result_matches_one_shot():
    # Gating adaptivity trades planning overhead, never the result.
    left = bt.from_arrow(pa.table({"k": [1, 2, 3, 1, 2, 3], "v": [1, 2, 3, 4, 5, 6]}))
    right = bt.from_arrow(pa.table({"k": [1, 2, 3], "w": [10, 20, 30]}))

    def q():
        return left.group_by("k").agg(s=col("v").sum()).join(right, on="k")

    def norm(d):
        return sorted(zip(*[d[c] for c in sorted(d)], strict=True))

    auto = q().collect(adaptive="auto").to_pydict()
    one_shot = q().collect(adaptive=False).to_pydict()
    assert norm(auto) == norm(one_shot)
