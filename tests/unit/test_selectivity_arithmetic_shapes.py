"""A predicate wrapped in arithmetic still describes the column underneath it.

`x + 1 < 100`, `-x > 0`, `price * 1.2 >= 60`, `id % 10 = 0` and `abs(delta) < 3` each name a
region of one column's distribution, and the measured histogram describes that region
exactly. None of them reached it: the comparison's operand was not a bare `Col`, so every one
took the Selinger constant (`1/3` for a range, `0.1` for an equality) however sharp the
statistics were.

`kyber.rules.extra.sargable` performs the same inversions as a *rewrite* and deliberately
refuses the ordered comparisons, because the engine's integer arithmetic wraps and
`x + 5 > 10` really does differ from `x > 5` at `INT64_MAX`. Nothing here rewrites anything,
so that argument does not apply: this covers exactly the gap the rewrite must leave open.

The same file covers the disjunctive twin of the interval rule — `x < 50 OR x < 70` is
`x < 70`, not an independent union of two coin flips.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.selectivity import predicate_selectivity as sel
from batcher.kyber.stats.selectivity.arithmetic import invert_arithmetic

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
_NDV = {"x": 100.0}
_BOUNDS = {"x": (0, 99)}


def _s(expr, bounds=_BOUNDS, ndv=_NDV, quantiles=None):
    return sel(expr, ndv, _CFG, quantiles, None, bounds)


# --- the inversion itself ---------------------------------------------------------


@pytest.mark.parametrize(
    ("built", "equivalent"),
    [
        ((bt.col("x") + 10) < 60, bt.col("x") < 50),
        ((bt.col("x") - 5) < 10, bt.col("x") < 15),
        ((10 + bt.col("x")) >= 60, bt.col("x") >= 50),
        ((bt.col("x") * 3) >= 9, bt.col("x") >= 3),
        ((bt.col("x") * -2) >= 9, bt.col("x") <= -4.5),
        (-bt.col("x") > 0, bt.col("x") < 0),
        ((100 - bt.col("x")) < 10, bt.col("x") > 90),
        ((bt.col("x") + 1) == 6, bt.col("x") == 5),
    ],
)
def test_a_monotone_wrapper_is_inverted_onto_the_column(built, equivalent):
    """The wrapped predicate estimates exactly as the equivalent bare one does."""
    assert _s(built) == pytest.approx(_s(equivalent))


def test_a_negative_multiplier_reverses_the_comparison():
    inverted = invert_arithmetic((bt.col("x") * -2) >= 9)
    assert inverted is not None
    assert inverted.op == "le"


def test_a_zero_multiplier_is_not_a_monotone_function():
    """`x * 0` is a constant, so nothing about `x` follows from it."""
    assert invert_arithmetic((bt.col("x") * 0) > 5) is None


def test_a_nested_wrapper_is_peeled_a_layer_at_a_time():
    """One level per call, and the caller re-estimates the result — so nesting resolves."""
    once = invert_arithmetic(((bt.col("x") + 1) * 2) < 10)
    assert once is not None
    twice = invert_arithmetic(once)
    assert twice is not None
    assert _s(((bt.col("x") + 1) * 2) < 10) == pytest.approx(_s(bt.col("x") < 4))


# --- modulo -----------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 10, 12])
def test_a_modulo_equality_is_one_of_k_residues(k):
    assert _s((bt.col("x") % k) == 0) == pytest.approx(1.0 / k)


def test_a_modulo_range_covers_its_share_of_the_residues():
    # `id % 10 < 3` keeps residues 0, 1, 2 — three tenths.
    assert _s((bt.col("x") % 10) < 3) == pytest.approx(0.3)
    assert _s((bt.col("x") % 10) >= 3) == pytest.approx(0.7)


def test_a_modulo_by_one_or_a_non_integer_is_declined():
    """`x % 1` is always 0 and carries no information; a float divisor is not a residue set."""
    assert _s((bt.col("x") % 1) == 0) == pytest.approx(_CFG.eq_selectivity)


# --- absolute value ---------------------------------------------------------------


def test_abs_reads_the_two_cdf_points_it_names():
    # A uniform grid over [-10, 10]: |x| < 5 is -5 < x < 5, half the range.
    grid = {"x": {"probs": [0.0, 1.0], "values": [-10.0, 10.0]}}
    assert sel(bt.col("x").abs() < 5, {}, _CFG, grid, None, {"x": (-10, 10)}) == pytest.approx(0.5)
    assert sel(bt.col("x").abs() >= 5, {}, _CFG, grid, None, {"x": (-10, 10)}) == pytest.approx(0.5)


def test_a_negative_bound_on_an_absolute_value_is_a_certainty():
    assert _s(bt.col("x").abs() < -1) == pytest.approx(0.0)
    assert _s(bt.col("x").abs() >= -1) == pytest.approx(1.0)


# --- same-column range disjunction ------------------------------------------------


def test_overlapping_range_disjuncts_unite_rather_than_multiply():
    """`x < 50 OR x < 70` is exactly `x < 70`, not an independent union."""
    united = _s((bt.col("x") < 50) | (bt.col("x") < 70))
    assert united == pytest.approx(_s(bt.col("x") < 70))
    # The independence rule would have invented survivors above the wider bound.
    assert united < 1.0 - (1.0 - _s(bt.col("x") < 50)) * (1.0 - _s(bt.col("x") < 70))


def test_disjoint_tails_sum():
    """`x < 10 OR x > 90` is the pair of tails, which do not overlap."""
    got = _s((bt.col("x") < 10) | (bt.col("x") > 90))
    assert got == pytest.approx(_s(bt.col("x") < 10) + _s(bt.col("x") > 90), rel=0.05)


def test_covering_range_disjuncts_keep_everything():
    """`x < 70 OR x > 30` covers the column; the union is capped at the whole relation."""
    assert _s((bt.col("x") < 70) | (bt.col("x") > 30)) == pytest.approx(1.0)


def test_a_disjunction_across_columns_is_still_independent():
    """The union rule must not reach across columns, where independence is the right model."""
    ndv = {"x": 100.0, "y": 100.0}
    bounds = {"x": (0, 99), "y": (0, 99)}
    got = sel((bt.col("x") < 50) | (bt.col("y") < 50), ndv, _CFG, None, None, bounds)
    one = sel(bt.col("x") < 50, ndv, _CFG, None, None, bounds)
    assert got == pytest.approx(1.0 - (1.0 - one) ** 2)


# --- against executed row counts --------------------------------------------------


@pytest.mark.parametrize(
    ("name", "build", "expected_rows"),
    [
        ("shift", lambda: (bt.col("x") + 10) < 60, 50),
        ("negate", lambda: -bt.col("x") > -50, 50),
        ("scale", lambda: (bt.col("x") * 2) < 100, 50),
        ("modulo", lambda: (bt.col("x") % 10) == 0, 10),
        ("or-overlap", lambda: (bt.col("x") < 50) | (bt.col("x") < 70), 70),
        ("or-tails", lambda: (bt.col("x") < 10) | (bt.col("x") > 90), 19),
    ],
)
def test_the_estimate_tracks_the_executed_row_count(name, build, expected_rows):
    """One row per value of `x` in [0, 100), so the true count is readable by hand."""
    values = list(range(100))
    dataset = bt.from_pydict({"x": values}).filter(build())
    stats = [s.statistics() for s in dataset._sources]
    estimated = (
        StatsEstimator(dataset._sources, {"__column_ndv__": {"x": 100.0}}, _CFG, source_stats=stats)
        .estimate(dataset._plan)
        .rows
    )
    executed = dataset.count()
    assert executed == expected_rows, f"{name}: fixture drifted"
    assert estimated == pytest.approx(executed, rel=0.12), (
        f"{name}: estimated {estimated:.1f} against {executed} executed rows"
    )
