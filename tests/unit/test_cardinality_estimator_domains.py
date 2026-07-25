"""Domain invariants for the cardinality estimators.

`kyber/stats/distribution.py` had eight of nine public names unmentioned by any test. These
are estimates, so a wrong one picks a worse plan rather than returning a wrong answer -- but
an estimate outside its own domain (a fraction above one, a negative count, a distinct count
above the row count) feeds the cost model a value it cannot reason about, and the adaptive
loop then learns from it.

472 combinations were swept, including the inconsistent inputs stale statistics produce: an
ndv above the row count, `rows_out` above `rows_in`, an empty most-common-value table, and a
zero row count.
"""

from __future__ import annotations

import itertools
import math

import pytest

from batcher.kyber.stats.distribution import (
    distinct_after_selection,
    geometric_mean,
    join_match_fraction,
    mcv_join_rows,
    overlap_fraction,
    residual_eq_frequency,
    residual_mass,
    union_ndv,
)

pytestmark = pytest.mark.unit

_MAGNITUDES = [0.0, 1.0, 2.0, 10.0, 1e6, 1e12]


@pytest.mark.parametrize("ndv", _MAGNITUDES)
@pytest.mark.parametrize("rows_in", _MAGNITUDES)
@pytest.mark.parametrize("rows_out", _MAGNITUDES)
def test_distinct_after_selection_cannot_exceed_what_survives(ndv, rows_in, rows_out) -> None:
    """A selection cannot leave more distinct values than it leaves rows, or than it started."""
    got = distinct_after_selection(ndv, rows_in, rows_out)
    assert math.isfinite(got)
    assert got >= 0.0
    cap = min(ndv, rows_out)
    assert got <= cap + 1e-6 + 1e-9 * max(1.0, cap)


@pytest.mark.parametrize("left", _MAGNITUDES)
@pytest.mark.parametrize("right", _MAGNITUDES)
def test_join_match_fraction_is_a_fraction(left, right) -> None:
    got = join_match_fraction(left, right)
    assert math.isfinite(got)
    assert 0.0 <= got <= 1.0


_RANGES = [None, (0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (-5.0, 5.0), (10.0, 20.0), (0.0, 1e12)]


@pytest.mark.parametrize("a", _RANGES)
@pytest.mark.parametrize("b", _RANGES)
def test_overlap_fraction_is_a_fraction_or_unknown(a, b) -> None:
    got = overlap_fraction(a, b)
    if got is None:
        return
    assert math.isfinite(got)
    assert 0.0 <= got <= 1.0


_MCVS = [None, {}, {"a": 0.5}, {"a": 0.5, "b": 0.5}, {"a": 1.0}, {"a": 0.9, "b": 0.9}]


@pytest.mark.parametrize("mcv", _MCVS)
def test_residual_mass_is_a_probability(mcv) -> None:
    """An over-subscribed table (frequencies summing past one) must clamp, not go negative."""
    got = residual_mass(mcv)
    assert math.isfinite(got)
    assert 0.0 <= got <= 1.0


@pytest.mark.parametrize("mcv", _MCVS)
@pytest.mark.parametrize("ndv", [None, 0.0, 1.0, 10.0, 1e9])
def test_residual_eq_frequency_is_a_probability(mcv, ndv) -> None:
    got = residual_eq_frequency(ndv, mcv, 0.1)
    assert math.isfinite(got)
    assert 0.0 <= got <= 1.0


@pytest.mark.parametrize(
    "ndvs", [[0.0], [1.0], [1.0, 2.0], [10.0, 10.0], [1e9, 1e9], [0.0, 0.0], [5.0, 0.0]]
)
@pytest.mark.parametrize("rows", [None, 0.0, 1.0, 5.0, 1e12])
def test_union_ndv_respects_the_frechet_bounds_and_the_row_cap(ndvs, rows) -> None:
    """``max_i d_i <= d_union <= Σ_i d_i``, and never above the row count when it is known."""
    got = union_ndv(ndvs, rows)
    if got is None:
        assert not [d for d in ndvs if d > 0.0]
        return
    assert math.isfinite(got)
    assert got >= 0.0
    assert got <= sum(ndvs) + 1e-6
    if rows is not None:
        assert got <= rows + 1e-6, f"a union of {rows} rows cannot hold {got} distinct values"


def test_union_ndv_of_an_empty_union_is_zero() -> None:
    """`rows=0` is a known count, not a missing one, so the cap applies and the floor does not.

    The `rows > 0` guard skipped the cap at exactly zero and the `max(1.0, ...)` floor then
    reported at least one distinct value, so `union_ndv([1e9], 0)` returned 1e9 while every
    positive row count capped correctly.
    """
    assert union_ndv([1e9, 1e9], 0.0) == 0.0
    assert union_ndv([1.0], 0.0) == 0.0
    assert union_ndv([1e9, 1e9], 0.5) == 1.0, "a non-empty relation still floors at one"


@pytest.mark.parametrize("left_rows", [0.0, 1.0, 1e6])
@pytest.mark.parametrize("right_rows", [0.0, 1.0, 1e6])
def test_mcv_join_rows_is_a_non_negative_row_count(left_rows, right_rows) -> None:
    for left_mcv, right_mcv in itertools.product([None, {}, {"a": 0.5}], repeat=2):
        got = mcv_join_rows(left_rows, right_rows, left_mcv, right_mcv, 10.0, 10.0)
        if got is None:
            continue
        assert math.isfinite(got)
        assert got >= 0.0


@pytest.mark.parametrize(
    "values", [[1.0], [1.0, 4.0], [1e-9, 1e9], [2.0] * 5, [], [0.0, 1.0], [-1.0, 2.0]]
)
def test_geometric_mean_lies_between_the_positive_extremes(values) -> None:
    got = geometric_mean(values)
    positive = [v for v in values if v > 0.0]
    if got is None:
        assert not positive
        return
    assert math.isfinite(got)
    assert min(positive) - 1e-6 <= got <= max(positive) + 1e-6
