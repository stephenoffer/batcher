"""Bounds survive a monotonic arithmetic projection.

`SELECT x + 10 AS y` followed by `WHERE y > 100` should still interpolate a range
selectivity from `x`'s footer bounds. A monotonic transform maps `[min, max]` onto a known
interval (reversed for a negative scale) and preserves the distinct count; anything else —
a two-column expression, a date column, a zero scale — drops to unknown, as before.
"""

from __future__ import annotations

import datetime

import pytest

from batcher.kyber.stats.derived import derived_projection_stat
from batcher.plan.expr_ir import col
from batcher.plan.stats import ColumnStat, Provenance, RelStats

pytestmark = pytest.mark.unit


def _child(cmin, cmax, ndv=101.0):
    return RelStats(
        100.0,
        Provenance.EXACT,
        {"x": ColumnStat(min=cmin, max=cmax, ndv=ndv, null_count=0, provenance=Provenance.EXACT)},
    )


@pytest.mark.parametrize(
    ("expr", "lo", "hi"),
    [
        (col("x") + 10, 10, 110),
        (10 + col("x"), 10, 110),
        (col("x") - 3, -3, 97),
        (10 - col("x"), -90, 10),  # c - x reverses order
        (col("x") * 2, 0, 200),
        (col("x") * -2, -200, 0),  # negative scale reverses order
        (-col("x"), -100, 0),  # lowered as 0 - x
    ],
)
def test_monotonic_transform_maps_the_interval(expr, lo, hi):
    stat = derived_projection_stat(expr, _child(0, 100))
    assert stat is not None
    assert (stat.min, stat.max) == (lo, hi)
    assert stat.ndv == 101.0  # injective → distinct count preserved
    assert stat.provenance is Provenance.DEFAULT  # never EXACT (overflow/rounding)


def test_two_column_expression_is_unknown():
    assert derived_projection_stat(col("x") + col("x"), _child(0, 100)) is None


def test_multiply_by_zero_defers_to_the_constant_folder():
    assert derived_projection_stat(col("x") * 0, _child(0, 100)) is None


def test_date_column_is_not_shifted_as_a_number():
    child = RelStats(
        10.0,
        Provenance.EXACT,
        {"x": ColumnStat(min=datetime.date(2020, 1, 1), max=datetime.date(2021, 1, 1))},
    )
    assert derived_projection_stat(col("x") + 10, child) is None


def test_missing_bounds_is_unknown():
    child = RelStats(100.0, Provenance.EXACT, {"x": ColumnStat(ndv=5.0)})
    assert derived_projection_stat(col("x") + 10, child) is None


def test_nullif_carries_the_column_bounds():
    # NULLIF(x, sentinel) keeps a subset of x's values, so bounds/ndv survive (downgraded);
    # only the null count is unknown. A common data-cleaning shape.
    from batcher.plan.expr_ir import Col, Lit, NullIf

    stat = derived_projection_stat(NullIf(left=Col("x"), right=Lit(-999)), _child(0, 100))
    assert stat is not None
    assert (stat.min, stat.max) == (0, 100)
    assert stat.provenance is Provenance.DEFAULT
    assert stat.null_count is None  # NULLIF adds nulls


def test_greatest_and_least_bounds_are_tight_not_a_union_box():
    """Which argument wins is known, so one end of each interval is much tighter than the box.

    The union box `[min(mins), max(maxes)]` is a valid superset for both, but it throws away
    the defining property: a `least` can never exceed the smallest of its arguments' maxima,
    and a `greatest` can never fall below the largest of their minima.
    """
    from batcher.plan.expr_ir import Col, Greatest, Least, Lit

    child = _child(0, 100)  # only column x ∈ [0, 100]
    # greatest(x, 200) is always 200 here: it cannot be below max(0, 200).
    g = derived_projection_stat(Greatest(inputs=[Col("x"), Lit(200)]), child)
    assert g is not None and (g.min, g.max) == (200, 200)
    # least(x, -5) is always -5, for the mirrored reason.
    least = derived_projection_stat(Least(inputs=[Col("x"), Lit(-5)]), child)
    assert least is not None and (least.min, least.max) == (-5, -5)

    # The clamping shape this exists for: `least(x, 50)` caps the column at 50, so a
    # downstream `> 60` is provably empty rather than estimated at a third of the table.
    capped = derived_projection_stat(Least(inputs=[Col("x"), Lit(50)]), child)
    assert capped is not None and (capped.min, capped.max) == (0, 50)
    floored = derived_projection_stat(Greatest(inputs=[Col("x"), Lit(50)]), child)
    assert floored is not None and (floored.min, floored.max) == (50, 100)


def test_greatest_of_unbounded_argument_is_unknown():
    from batcher.plan.expr_ir import Col, Greatest

    # `y` has no bounds in the child → cannot bound the greatest.
    assert derived_projection_stat(Greatest(inputs=[Col("x"), Col("y")]), _child(0, 100)) is None


def test_a_monotonic_projection_carries_the_quantile_grid_through():
    """`F_y(g(x)) = F_x(x)` for a strictly increasing `g`, so the values move and the
    probabilities do not — which is what keeps a range filter on a derived column sharp."""
    from batcher.plan.expr_ir import Binary, Col, Lit
    from batcher.plan.stats import ColumnStat, Provenance, RelStats

    grid = {"probs": [0.0, 0.5, 1.0], "values": [0.0, 10.0, 100.0]}
    child = RelStats(
        1000.0,
        Provenance.EXACT,
        {"x": ColumnStat(min=0, max=100, quantiles=grid, provenance=Provenance.EXACT)},
    )
    scaled = derived_projection_stat(Binary("mul", Col("x"), Lit(100)), child)
    assert scaled is not None and scaled.quantiles is not None
    assert scaled.quantiles["values"] == [0.0, 1000.0, 10000.0]
    assert scaled.quantiles["probs"] == [0.0, 0.5, 1.0]


def test_a_decreasing_projection_reverses_the_grid_and_complements_the_probabilities():
    from batcher.plan.expr_ir import Binary, Col, Lit
    from batcher.plan.stats import ColumnStat, Provenance, RelStats

    grid = {"probs": [0.0, 0.25, 1.0], "values": [0.0, 10.0, 100.0]}
    child = RelStats(
        1000.0,
        Provenance.EXACT,
        {"x": ColumnStat(min=0, max=100, quantiles=grid, provenance=Provenance.EXACT)},
    )
    negated = derived_projection_stat(Binary("mul", Col("x"), Lit(-1)), child)
    assert negated is not None and negated.quantiles is not None
    # The interpolator needs an ascending grid, so the values reverse and the probabilities
    # become their complements.
    assert negated.quantiles["values"] == [-100.0, -10.0, -0.0]
    assert negated.quantiles["probs"] == [0.0, 0.75, 1.0]


def test_a_monotonic_projection_carries_measured_skew_to_the_image_value():
    """The map is injective, so no two values collide and every frequency transfers."""
    from batcher.plan.expr_ir import Binary, Col, Lit
    from batcher.plan.stats import ColumnStat, Provenance, RelStats

    child = RelStats(
        1000.0,
        Provenance.EXACT,
        {"x": ColumnStat(min=0, max=100, mcv={"7": 0.4}, provenance=Provenance.EXACT)},
    )
    shifted = derived_projection_stat(Binary("add", Col("x"), Lit(10)), child)
    assert shifted is not None and shifted.mcv is not None
    assert shifted.mcv == {"17.0": 0.4}


def test_an_affine_projection_shifts_the_mean_exactly():
    from batcher.plan.expr_ir import Binary, Col, Lit
    from batcher.plan.stats import ColumnStat, Provenance, RelStats

    child = RelStats(
        1000.0,
        Provenance.EXACT,
        {"x": ColumnStat(min=0, max=100, mean=25.0, provenance=Provenance.EXACT)},
    )
    doubled = derived_projection_stat(Binary("mul", Col("x"), Lit(2)), child)
    assert doubled is not None and doubled.mean == pytest.approx(50.0)
    shifted = derived_projection_stat(Binary("sub", Col("x"), Lit(5)), child)
    assert shifted is not None and shifted.mean == pytest.approx(20.0)
