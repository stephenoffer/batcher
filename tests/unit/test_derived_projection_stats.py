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


def test_greatest_least_use_the_bounding_box():
    from batcher.plan.expr_ir import Col, Greatest, Least, Lit

    child = _child(0, 100)  # only column x ∈ [0, 100]
    # greatest(x, 200): every value is x or 200 → box [0, 200].
    g = derived_projection_stat(Greatest(inputs=[Col("x"), Lit(200)]), child)
    assert g is not None and (g.min, g.max) == (0, 200)
    # least(x, -5): box [-5, 100].
    least = derived_projection_stat(Least(inputs=[Col("x"), Lit(-5)]), child)
    assert least is not None and (least.min, least.max) == (-5, 100)


def test_greatest_of_unbounded_argument_is_unknown():
    from batcher.plan.expr_ir import Col, Greatest

    # `y` has no bounds in the child → cannot bound the greatest.
    assert derived_projection_stat(Greatest(inputs=[Col("x"), Col("y")]), _child(0, 100)) is None
