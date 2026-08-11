"""The adaptive size floor is charged per stage the loop would cut, not per query.

What staging costs is a constant of each *cut* — one materialize, one re-plan, and the
fusion and streaming width given up at that boundary — so a two-breaker plan pays it once
and a snowflake pays it six times. A single flat number has to be set for the worst shape it
will see, which is what made the loop unreachable below 20M rows for every shape, including
the cheap ones.

These tests pin both directions: the cheap shape reaches the loop at a size that used to be
refused, and the expensive shape is refused at a size that used to be accepted.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.adaptive.gating import (
    _ADAPTIVE_MIN_ROWS_PER_STAGE,
    _large_enough,
    _stage_count,
)

pytestmark = pytest.mark.unit

#: The flat floor this replaced. Kept as a literal rather than derived, because the point of
#: these tests is the relationship between the new rule and that specific old number.
_OLD_FLAT_FLOOR = 20_000_000


class _Estimator:
    """Reports `rows` for every scan, so a test can pick a size without building the data."""

    def __init__(self, rows: float):
        self.rows = rows

    def estimate(self, _node):
        return type("Stats", (), {"rows": self.rows})()

    def row_width(self, _node, default):
        return float(default)


def _sized(monkeypatch, rows_per_scan: float):
    """Make every scan in any plan report `rows_per_scan` rows of the default width."""
    from batcher.api.adaptive import gating

    monkeypatch.setattr(gating, "build_estimator", lambda *_a, **_k: _Estimator(rows_per_scan))


def _two_breaker():
    """`agg ⋈ dim` — one breaker producing an operand, plus the join. The cheap shape."""
    fact = bt.from_pydict({"k": [1, 2, 3], "v": [1, 2, 3]})
    dim = bt.from_pydict({"k": [1, 2, 3], "name": ["a", "b", "c"]})
    agg = fact.group_by("k").agg(s=bt.col("v").sum())
    return agg.join(dim, on="k")


def _six_breaker():
    """Two aggregates, two joins, a sort and a distinct — the many-cut shape that regressed."""
    fact = bt.from_pydict({"k": [1, 2, 3], "g": [1, 2, 3], "v": [1, 2, 3]})
    dim = bt.from_pydict({"k": [1, 2, 3], "name": ["a", "b", "c"]})
    a = fact.group_by("k").agg(s=bt.col("v").sum())
    b = fact.group_by("k").agg(m=bt.col("v").mean())
    return a.join(b, on="k").join(dim, on="k").distinct().sort("k")


def test_the_stage_count_is_the_plans_breaker_count():
    assert _stage_count(_two_breaker()._plan) == 2
    assert _stage_count(_six_breaker()._plan) >= 5


def test_a_joinless_plan_never_qualifies_however_large():
    """The size floor is one condition among several, and this one comes first."""
    ds = bt.from_pydict({"k": [1, 2, 3]})
    assert _large_enough(ds._plan, ds._sources, None) is False


def test_the_cheap_shape_now_reaches_the_loop_below_the_old_flat_floor(monkeypatch):
    """The defect being fixed: two cuts cost about a tenth of what six cost, and were
    priced as if they cost the same."""
    ds = _two_breaker()
    total = _ADAPTIVE_MIN_ROWS_PER_STAGE * 2  # exactly the two-stage floor
    assert total < _OLD_FLAT_FLOOR, "this test proves nothing unless the floor really moved"
    _sized(monkeypatch, total / 2)  # two scans
    assert _large_enough(ds._plan, ds._sources, None) is True


def test_the_cheap_shape_is_still_refused_below_its_own_floor(monkeypatch):
    ds = _two_breaker()
    _sized(monkeypatch, (_ADAPTIVE_MIN_ROWS_PER_STAGE * 2 - 2) / 2)
    assert _large_enough(ds._plan, ds._sources, None) is False


def test_the_expensive_shape_is_refused_where_the_flat_floor_admitted_it(monkeypatch):
    """The other direction, and the one with measured evidence behind it: q8/q17/q9/q3 at
    sf10 lost 3-6x to staging, and all four are many-breaker shapes."""
    ds = _six_breaker()
    stages = _stage_count(ds._plan)
    assert stages * _ADAPTIVE_MIN_ROWS_PER_STAGE > _OLD_FLAT_FLOOR
    _sized(monkeypatch, _OLD_FLAT_FLOOR / 3)  # three scans totalling the old flat floor
    assert _large_enough(ds._plan, ds._sources, None) is False


def test_the_expensive_shape_still_qualifies_once_it_is_big_enough(monkeypatch):
    ds = _six_breaker()
    stages = _stage_count(ds._plan)
    _sized(monkeypatch, _ADAPTIVE_MIN_ROWS_PER_STAGE * stages / 3 + 1)
    assert _large_enough(ds._plan, ds._sources, None) is True


def test_the_floor_is_never_zero_stages():
    """A degenerate walk must not divide the floor away and admit everything."""
    assert _stage_count(bt.from_pydict({"k": [1]})._plan) >= 1
