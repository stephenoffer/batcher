"""Plan-shape tests for the `kyber.rules.math_algebra` range families.

Each rule must produce the documented interval, must be reachable through the real
`Optimizer`, and must decline the shapes its correctness argument excludes: the
`INT64_MAX` magnitude where saturating integer `abs` disagrees, the zero-straddling
`sign` comparisons on a float (where `sign(NaN)` is `0.0`), a fractional bound for
`floor`/`ceil`, a `bit_count` over a float, and a non-positive or non-integral `//`
divisor.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Binary, Expr
from batcher.plan.expr_ir.core import MathExpr

_INT64_MAX = 2**63 - 1


def _ds():
    """Int64 `i`, Float64 `f`, String `s`."""
    return bt.from_pydict({"i": [1, 2, 3], "f": [1.0, 2.0, 3.0], "s": ["a", "b", "c"]})


def _proj(expr: Expr):
    return _ds().select(r=expr)._plan


def _filter(expr: Expr):
    return _ds().filter(expr)._plan


def _rule(name: str):
    for r in DEFAULT_REGISTRY.rules():
        if r.name == name:
            return r
    raise AssertionError(f"rule {name!r} is not registered")


def _fire(name: str, expr: Expr) -> dict:
    node = _proj(expr)
    out = _rule(name).apply(node, None)
    assert out.to_ir() != node.to_ir(), f"{name} did not fire"
    return out.items[0].expr.to_ir()


def _noop(name: str, expr: Expr) -> None:
    node = _proj(expr)
    assert _rule(name).apply(node, None).to_ir() == node.to_ir()


def _optimized(expr: Expr) -> dict:
    return optimize_logical(_proj(expr)).items[0].expr.to_ir()


# --- abs --------------------------------------------------------------------

_ABS_EXPECTED = {
    "lt": (col("f") > lit(-2)) & (col("f") < lit(2)),
    "le": (col("f") >= lit(-2)) & (col("f") <= lit(2)),
    "gt": (col("f") < lit(-2)) | (col("f") > lit(2)),
    "ge": (col("f") <= lit(-2)) | (col("f") >= lit(2)),
    "eq": (col("f") == lit(2)) | (col("f") == lit(-2)),
    "ne": (col("f") != lit(2)) & (col("f") != lit(-2)),
}


@pytest.mark.parametrize("op", list(_ABS_EXPECTED))
def test_abs_comparison_becomes_an_interval(op):
    expr = Binary(op, MathExpr("abs", col("f")), lit(2))
    assert _fire(f"abs_{op}_to_range", expr) == _ABS_EXPECTED[op].to_ir()


@pytest.mark.parametrize("op", list(_ABS_EXPECTED))
def test_abs_comparison_matches_with_the_literal_on_the_left(op):
    flip = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}[op]
    expr = Binary(flip, lit(2), MathExpr("abs", col("f")))
    assert _fire(f"abs_{op}_to_range", expr) == _ABS_EXPECTED[op].to_ir()


def test_abs_declines_a_non_positive_bound():
    _noop("abs_lt_to_range", MathExpr("abs", col("f")) < lit(0))
    _noop("abs_lt_to_range", MathExpr("abs", col("f")) < lit(-3))


def test_abs_declines_the_saturating_integer_boundary():
    _noop("abs_eq_to_range", MathExpr("abs", col("i")) == lit(_INT64_MAX))


def test_abs_declines_a_non_literal_bound():
    _noop("abs_lt_to_range", MathExpr("abs", col("f")) < col("i"))


# --- sign -------------------------------------------------------------------

_SIGN_INTEGER_ONLY = [
    ("sign_eq_one_to_positive", "eq", 1, "gt"),
    ("sign_eq_minus_one_to_negative", "eq", -1, "lt"),
    ("sign_gt_zero_to_positive", "gt", 0, "gt"),
    ("sign_lt_zero_to_negative", "lt", 0, "lt"),
    ("sign_ge_one_to_positive", "ge", 1, "gt"),
    ("sign_le_minus_one_to_negative", "le", -1, "lt"),
    ("sign_eq_zero_to_zero_integer", "eq", 0, "eq"),
    ("sign_ne_zero_to_nonzero_integer", "ne", 0, "ne"),
    ("sign_ge_zero_to_nonnegative_integer", "ge", 0, "ge"),
    ("sign_le_zero_to_nonpositive_integer", "le", 0, "le"),
    ("sign_gt_minus_one_to_nonnegative_integer", "gt", -1, "ge"),
    ("sign_lt_one_to_nonpositive_integer", "lt", 1, "le"),
]


@pytest.mark.parametrize(("name", "op", "value", "want"), _SIGN_INTEGER_ONLY)
def test_sign_comparison_fires_on_an_integer(name, op, value, want):
    expr = Binary(op, MathExpr("sign", col("i")), lit(value))
    assert _fire(name, expr) == Binary(want, col("i"), lit(0)).to_ir()


@pytest.mark.parametrize(("name", "op", "value", "_want"), _SIGN_INTEGER_ONLY)
def test_sign_comparison_declines_a_float(name, op, value, _want):
    # `sign(NaN)` is 0.0 while the engine's total order puts the NaN itself above every
    # finite value, so every one of these would reclassify a NaN row.
    _noop(name, Binary(op, MathExpr("sign", col("f")), lit(value)))


def test_sign_declines_an_out_of_range_literal():
    _noop("sign_eq_one_to_positive", MathExpr("sign", col("i")) == lit(2))


# --- floor / ceil -----------------------------------------------------------

_FLOOR_EXPECTED = {
    "lt": col("f") < lit(3),
    "le": col("f") < lit(4),
    "gt": col("f") >= lit(4),
    "ge": col("f") >= lit(3),
    "eq": (col("f") >= lit(3)) & (col("f") < lit(4)),
    "ne": (col("f") < lit(3)) | (col("f") >= lit(4)),
}

_CEIL_EXPECTED = {
    "lt": col("f") <= lit(2),
    "le": col("f") <= lit(3),
    "gt": col("f") > lit(3),
    "ge": col("f") > lit(2),
    "eq": (col("f") > lit(2)) & (col("f") <= lit(3)),
    "ne": (col("f") <= lit(2)) | (col("f") > lit(3)),
}


@pytest.mark.parametrize("op", list(_FLOOR_EXPECTED))
def test_floor_comparison_becomes_a_half_open_interval(op):
    expr = Binary(op, MathExpr("floor", col("f")), lit(3))
    assert _fire(f"floor_{op}_to_range", expr) == _FLOOR_EXPECTED[op].to_ir()


@pytest.mark.parametrize("op", list(_CEIL_EXPECTED))
def test_ceil_comparison_becomes_a_half_open_interval(op):
    expr = Binary(op, MathExpr("ceil", col("f")), lit(3))
    assert _fire(f"ceil_{op}_to_range", expr) == _CEIL_EXPECTED[op].to_ir()


def test_rounding_accepts_an_integral_float_bound():
    expr = MathExpr("floor", col("f")) >= lit(3.0)
    assert _fire("floor_ge_to_range", expr) == (col("f") >= lit(3)).to_ir()


def test_rounding_declines_a_fractional_bound():
    _noop("floor_eq_to_range", MathExpr("floor", col("f")) == lit(2.5))
    _noop("ceil_eq_to_range", MathExpr("ceil", col("f")) == lit(2.5))


# --- bit_count --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "op", "value", "want"),
    [
        ("bit_count_eq_zero_to_zero", "eq", 0, "eq"),
        ("bit_count_ne_zero_to_nonzero", "ne", 0, "ne"),
        ("bit_count_gt_zero_to_nonzero", "gt", 0, "ne"),
        ("bit_count_ge_one_to_nonzero", "ge", 1, "ne"),
    ],
)
def test_bit_count_comparison_becomes_a_zero_test(name, op, value, want):
    expr = Binary(op, MathExpr("bit_count", col("i")), lit(value))
    assert _fire(name, expr) == Binary(want, col("i"), lit(0)).to_ir()


def test_bit_count_declines_a_float_argument():
    _noop("bit_count_eq_zero_to_zero", MathExpr("bit_count", col("f")) == lit(0))


# --- integer floored division ----------------------------------------------

_DIV_EXPECTED = {
    "lt": col("i") < lit(12),
    "le": col("i") < lit(15),
    "gt": col("i") >= lit(15),
    "ge": col("i") >= lit(12),
    "eq": (col("i") >= lit(12)) & (col("i") < lit(15)),
    "ne": (col("i") < lit(12)) | (col("i") >= lit(15)),
}


@pytest.mark.parametrize("op", list(_DIV_EXPECTED))
def test_floor_div_comparison_becomes_a_bucket_interval(op):
    expr = Binary(op, col("i") // lit(3), lit(4))
    assert _fire(f"floor_div_{op}_to_range", expr) == _DIV_EXPECTED[op].to_ir()


def test_floor_div_declines_a_non_positive_divisor():
    _noop("floor_div_eq_to_range", (col("i") // lit(-3)) == lit(4))
    _noop("floor_div_eq_to_range", (col("i") // lit(0)) == lit(4))


def test_floor_div_declines_a_float_dividend():
    _noop("floor_div_eq_to_range", (col("f") // lit(3)) == lit(4))


def test_floor_div_declines_when_the_bucket_bound_overflows():
    _noop("floor_div_ge_to_range", (col("i") // lit(1000)) >= lit(_INT64_MAX))


# --- end to end -------------------------------------------------------------


def test_optimizer_reaches_the_interval_form():
    assert _optimized(MathExpr("abs", col("f")) < lit(2)) == _ABS_EXPECTED["lt"].to_ir()


@pytest.mark.parametrize(
    "expr",
    [
        MathExpr("abs", col("f")) < lit(2),
        MathExpr("sign", col("i")) == lit(0),
        MathExpr("floor", col("f")) == lit(3),
        (col("i") // lit(3)) == lit(4),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()


def test_interval_form_survives_into_a_filter():
    plan = optimize_logical(_filter(MathExpr("abs", col("f")) < lit(2)))
    assert "abs" not in str(plan.to_ir())
