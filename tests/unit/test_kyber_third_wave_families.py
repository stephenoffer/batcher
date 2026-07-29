"""Plan-shape tests for the third wave of rule families.

Covers the two half-integer rounding intervals (`round`, whose tie goes away from zero,
and `rint`, whose tie goes to the even neighbour), the `isnan`/`isinf` collapse through the
class-preserving rounding functions, the `IN`-list null-strictness family, the extended
`CASE` push families, and the `<>` complement added to the year/decade sargable family.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Binary, Case, Expr, InList
from batcher.plan.expr_ir.core import IsInf, IsNan, MathExpr
from batcher.plan.expr_ir.func_nodes import ConvertTimezone, DateOffset, ListSlice, StructField


def _ds():
    return bt.from_pydict(
        {
            "f": [1.0, 2.0, 3.0],
            "x": [1, 2, 3],
            "t": [dt.datetime(2020, 1, 1)] * 3,
            "l": [[1.0], [2.0], [3.0]],
        }
    )


def _proj(expr: Expr):
    return _ds().select(r=expr)._plan


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


# --- round: the tie goes away from zero -------------------------------------


@pytest.mark.parametrize(
    ("op", "k", "want"),
    [
        ("eq", 0, (col("f") > lit(-0.5)) & (col("f") < lit(0.5))),
        ("eq", 2, (col("f") >= lit(1.5)) & (col("f") < lit(2.5))),
        ("eq", -2, (col("f") > lit(-2.5)) & (col("f") <= lit(-1.5))),
        ("ge", 1, col("f") >= lit(0.5)),
        ("ge", 0, col("f") > lit(-0.5)),
        ("le", 0, col("f") < lit(0.5)),
        ("le", -1, col("f") <= lit(-0.5)),
        ("gt", 0, col("f") >= lit(0.5)),
        ("lt", 0, col("f") <= lit(-0.5)),
    ],
)
def test_round_comparison_becomes_a_half_integer_interval(op, k, want):
    expr = Binary(op, MathExpr("round", col("f")), lit(k))
    assert _fire(f"round_{op}_to_range", expr) == want.to_ir()


# --- rint: the tie goes to the even neighbour -------------------------------


@pytest.mark.parametrize(
    ("op", "k", "want"),
    [
        # An even bucket owns both of its endpoints…
        ("eq", 0, (col("f") >= lit(-0.5)) & (col("f") <= lit(0.5))),
        ("eq", 2, (col("f") >= lit(1.5)) & (col("f") <= lit(2.5))),
        # …and an odd one owns neither.
        ("eq", 1, (col("f") > lit(0.5)) & (col("f") < lit(1.5))),
        ("eq", -1, (col("f") > lit(-1.5)) & (col("f") < lit(-0.5))),
        ("ge", 0, col("f") >= lit(-0.5)),
        ("ge", 1, col("f") > lit(0.5)),
        ("le", 0, col("f") <= lit(0.5)),
        ("le", 1, col("f") < lit(1.5)),
    ],
)
def test_rint_comparison_respects_the_parity_of_the_bucket(op, k, want):
    expr = Binary(op, MathExpr("rint", col("f")), lit(k))
    assert _fire(f"rint_{op}_to_range", expr) == want.to_ir()


def test_round_and_rint_differ_at_the_endpoint():
    round_eq = _fire("round_eq_to_range", MathExpr("round", col("f")) == lit(2))
    rint_eq = _fire("rint_eq_to_range", MathExpr("rint", col("f")) == lit(2))
    assert round_eq != rint_eq


@pytest.mark.parametrize("fn", ["round", "rint"])
def test_half_integer_rules_decline_a_fractional_bound(fn):
    _noop(f"{fn}_eq_to_range", MathExpr(fn, col("f")) == lit(2.25))


# --- isnan / isinf through the class-preserving functions -------------------


@pytest.mark.parametrize("fn", ["abs", "ceil", "floor", "rint", "round", "trunc"])
def test_non_finite_checks_see_through_rounding(fn):
    assert (
        _fire("nan_check_through_rounding", IsNan(MathExpr(fn, col("f"))))
        == IsNan(col("f")).to_ir()
    )
    assert (
        _fire("inf_check_through_rounding", IsInf(MathExpr(fn, col("f"))))
        == IsInf(col("f")).to_ir()
    )


def test_non_finite_checks_do_not_see_through_sign():
    # `sign(NaN)` is 0.0, so `isnan(sign(x))` is false everywhere — not `isnan(x)`.
    _noop("nan_check_through_rounding", IsNan(MathExpr("sign", col("f"))))


@pytest.mark.parametrize("fn", ["ln", "exp", "sqrt", "sin"])
def test_non_finite_checks_do_not_see_through_a_function_that_can_create_one(fn):
    _noop("nan_check_through_rounding", IsNan(MathExpr(fn, col("f"))))


# --- IN-list null strictness ------------------------------------------------


def test_null_test_moves_through_an_in_list():
    assert _fire("is_null_through_in_list", InList(col("x"), (1, 2)).is_null()) == (
        col("x").is_null().to_ir()
    )
    assert _fire("is_not_null_through_in_list", InList(col("x"), (1, 2)).is_not_null()) == (
        col("x").is_not_null().to_ir()
    )


def test_null_test_declines_an_in_list_holding_a_null():
    # A null in the set makes a *non-null* `x` answer null for a value it does not
    # contain, so the test no longer tracks `x`'s own nullness. Asserted on the leaf
    # rewrite directly: an `IN` list holding a null has no wire form to build a plan from.
    from batcher.kyber.rules.nulls.strictness import _in_list_operand

    assert _in_list_operand(InList(col("x"), (1, 2))) is not None
    assert _in_list_operand(InList(col("x"), (1, None))) is None


# --- extended CASE pushes ---------------------------------------------------


def _case(then_value: Expr, else_value: Expr) -> Case:
    return Case([(col("x") > lit(1), then_value)], else_value)


@pytest.mark.parametrize(
    ("family", "wrap", "then_value", "else_value"),
    [
        ("convert_timezone", lambda i: ConvertTimezone(i, "UTC", "UTC"), col("t"), col("t")),
        ("date_offset", lambda i: DateOffset(i, 0, 1, 0), col("t"), col("t")),
        ("list_slice", lambda i: ListSlice(i, 0, 1), col("l"), col("l")),
        ("struct_field", lambda i: StructField(i, "a"), col("l"), col("l")),
        ("in_list", lambda i: InList(i, (1, 2)), lit(1), lit(3)),
    ],
)
def test_extended_case_push_moves_onto_every_branch(family, wrap, then_value, else_value):
    got = _fire(f"push_{family}_into_case_branches", wrap(_case(then_value, else_value)))
    want = Case([(col("x") > lit(1), wrap(then_value))], wrap(else_value))
    assert got == want.to_ir()


def test_in_list_push_folds_a_case_over_literals():
    # Pushing the membership test onto the branches turns each literal branch into a
    # boolean constant, which is what the surrounding CASE rules can then act on.
    got = optimize_logical(_proj(InList(_case(lit(1), lit(3)), (1, 2))))
    ir = got.items[0].expr.to_ir()
    assert ir["e"] == "case"
    assert [b["then"] for b in ir["branches"]] == [lit(True).to_ir()]
    assert ir["otherwise"] == lit(False).to_ir()


def test_membership_over_a_literal_input_folds():
    assert _fire("fold_in_list_of_literal_input", InList(lit(1), (1, 2))) == lit(True).to_ir()
    assert _fire("fold_in_list_of_literal_input", InList(lit(9), (1, 2))) == lit(False).to_ir()


# --- the <> complement of the year/decade sargable family -------------------


@pytest.mark.parametrize("fn", ["year", "decade"])
def test_temporal_inequality_becomes_the_complement_of_the_band(fn):
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert f"{fn}_ne_to_range" in names


def test_year_inequality_is_the_outside_of_the_year_band():
    plan = _ds().filter(col("t").dt.year() != lit(2020))._plan
    got = optimize_logical(plan).predicate
    want = (col("t") < lit(dt.datetime(2020, 1, 1))) | (col("t") >= lit(dt.datetime(2021, 1, 1)))
    assert got.to_ir() == want.to_ir()


def test_year_equality_is_still_the_band_itself():
    plan = _ds().filter(col("t").dt.year() == lit(2020))._plan
    got = optimize_logical(plan).predicate
    want = (col("t") >= lit(dt.datetime(2020, 1, 1))) & (col("t") < lit(dt.datetime(2021, 1, 1)))
    assert got.to_ir() == want.to_ir()


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        MathExpr("round", col("f")) == lit(2),
        MathExpr("rint", col("f")) == lit(2),
        IsNan(MathExpr("floor", col("f"))),
        InList(col("x"), (1, 2)).is_null(),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()


# --- rint joins the rounding family -----------------------------------------


@pytest.mark.parametrize(
    ("outer", "inner", "want"),
    [
        ("rint", "rint", "rint"),
        ("floor", "rint", "rint"),
        ("rint", "floor", "floor"),
        ("rint", "round", "round"),
        ("ceil", "rint", "rint"),
    ],
)
def test_nested_rounding_collapses_through_rint(outer, inner, want):
    # `rint` yields an integral float, on which every rounding function is the identity —
    # so it belongs in the same family as floor/ceil/trunc/round for the collapse rules.
    expr = MathExpr(outer, MathExpr(inner, col("f")))
    got = optimize_logical(_proj(expr)).items[0].expr
    assert got.to_ir() == MathExpr(want, col("f")).to_ir()


def test_rint_over_an_integer_column_is_a_cast():
    got = optimize_logical(_proj(MathExpr("rint", col("x")))).items[0].expr
    assert got.to_ir()["e"] == "cast"


def test_rint_over_an_integer_literal_folds_to_the_promotion():
    got = optimize_logical(_proj(MathExpr("rint", lit(5)))).items[0].expr
    assert got.to_ir() == lit(5.0).to_ir()


def test_even_is_not_in_the_rounding_family():
    # `even(5)` is 6, so it is neither the identity on an integral value nor a plain
    # int-to-float promotion — the two properties the collapse and fold rules rely on.
    from batcher.kyber.rules.extra.arith_extra import _IDEMPOTENT_MATH, _ROUNDING

    assert "even" not in _ROUNDING
    assert "even" not in _IDEMPOTENT_MATH
    assert "rint" in _ROUNDING
