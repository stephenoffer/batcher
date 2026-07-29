"""Plan-shape tests for the `kyber.rules.nulls` families.

Three things are proven per rule: it fires into the intended shape when applied on its
own, the full `Optimizer` reaches that shape end to end, and it declines the neighbouring
shape whose rewrite would be unsound — a non-strict function (`json_value`, `list.sum`),
one that can raise (`factorial`), or an operator that can abort on non-null operands
(`/`, `%`, the shifts).
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.nulls.strictness import (
    NULL_STRICTNESS_BINARY_RULES,
    NULL_STRICTNESS_UNARY_RULES,
)
from batcher.kyber.rules.nulls.three_valued import never_null
from batcher.plan.expr_ir import Array, Coalesce, Expr, Greatest, Least
from batcher.plan.expr_ir.core import Binary, IsInf, IsNan, Math2Expr, MathExpr
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    ListFunc,
    Strftime,
    StrFunc,
)


def _ds():
    """Int64 `a`/`b`, Float64 `f`, String `s`, Timestamp `t`, List `l`."""
    return bt.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "f": [1.0, 2.0, 3.0],
            "s": ["a", "b", "c"],
            "t": [dt.datetime(2020, 1, 1), dt.datetime(2021, 6, 2), dt.datetime(2022, 7, 3)],
            "l": [[1], [2, 3], []],
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
    """The IR of `expr` after the named rule — asserting it fired."""
    out = _rule(name).apply(_proj(expr), None)
    assert out.to_ir() != _proj(expr).to_ir(), f"{name} did not fire"
    return out.items[0].expr.to_ir()


def _noop(name: str, expr: Expr) -> None:
    """Assert the named rule leaves `expr` alone."""
    node = _proj(expr)
    assert _rule(name).apply(node, None).to_ir() == node.to_ir()


def _optimized(expr: Expr) -> dict:
    return optimize_logical(_proj(expr)).items[0].expr.to_ir()


# --- every rule the package claims to register is registered ----------------


def test_all_strictness_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    for r in (*NULL_STRICTNESS_UNARY_RULES, *NULL_STRICTNESS_BINARY_RULES):
        assert r.name in names


def test_strictness_registers_both_directions_per_family():
    names = {r.name for r in NULL_STRICTNESS_UNARY_RULES}
    for family in ("math_fn", "str_fn", "date_fn", "nan_check", "inf_check"):
        assert f"is_null_through_{family}" in names
        assert f"is_not_null_through_{family}" in names


# --- unary strictness -------------------------------------------------------

_UNARY_CASES = [
    ("math_fn", MathExpr("sqrt", col("f"))),
    ("str_fn", StrFunc("upper", col("s"))),
    ("date_fn", DateFunc("year", col("t"))),
    ("date_trunc", DateTrunc(col("t"), "month")),
    ("date_offset", DateOffset(col("t"), 1, 0, 0)),
    ("strftime", Strftime(col("t"), "%Y")),
    ("convert_timezone", ConvertTimezone(col("t"), "UTC", "UTC")),
    ("list_reduction", ListFunc("len", col("l"))),
    ("boolean_negation", ~(col("a") > lit(0))),
    ("nan_check", IsNan(col("f"))),
    ("inf_check", IsInf(col("f"))),
]


@pytest.mark.parametrize(("family", "inner"), _UNARY_CASES, ids=[c[0] for c in _UNARY_CASES])
def test_is_null_moves_onto_the_operand(family, inner):
    operand = inner.input
    assert _fire(f"is_null_through_{family}", inner.is_null()) == operand.is_null().to_ir()
    assert (
        _fire(f"is_not_null_through_{family}", inner.is_not_null()) == operand.is_not_null().to_ir()
    )


def test_strictness_chains_to_a_bare_column():
    # upper(trim(s)) IS NULL collapses through both calls in one bottom-up pass.
    expr = StrFunc("upper", StrFunc("trim", col("s"))).is_null()
    assert _optimized(expr) == col("s").is_null().to_ir()


@pytest.mark.parametrize("fn", ["json_value", "unhex", "from_base64"])
def test_declines_non_strict_string_functions(fn):
    _noop("is_null_through_str_fn", StrFunc(fn, col("s"), pattern="$.a").is_null())


@pytest.mark.parametrize("fn", ["sum", "min", "mean"])
def test_declines_list_reductions_that_are_null_on_an_empty_list(fn):
    _noop("is_null_through_list_reduction", ListFunc(fn, col("l")).is_null())


def test_declines_factorial_which_can_raise():
    _noop("is_null_through_math_fn", MathExpr("factorial", col("a")).is_null())


def test_declines_bit_count_which_is_null_on_a_nan():
    _noop("is_null_through_math_fn", MathExpr("bit_count", col("f")).is_null())


# --- binary strictness ------------------------------------------------------


def test_is_null_splits_over_arithmetic():
    got = _fire("is_null_through_arithmetic", (col("a") + col("b")).is_null())
    assert got == (col("a").is_null() | col("b").is_null()).to_ir()


def test_is_not_null_splits_over_arithmetic():
    got = _fire("is_not_null_through_arithmetic", (col("a") * col("b")).is_not_null())
    assert got == (col("a").is_not_null() & col("b").is_not_null()).to_ir()


def test_is_null_splits_over_comparison():
    got = _fire("is_null_through_comparison", (col("a") < col("b")).is_null())
    assert got == (col("a").is_null() | col("b").is_null()).to_ir()


def test_is_null_splits_over_two_argument_math():
    got = _fire("is_null_through_math2_fn", Math2Expr("pow", col("f"), col("a")).is_null())
    assert got == (col("f").is_null() | col("a").is_null()).to_ir()


@pytest.mark.parametrize("op", ["div", "mod", "floor_div", "shift_left", "bit_and"])
def test_declines_operators_that_can_abort(op):
    _noop("is_null_through_arithmetic", Binary(op, col("a"), col("b")).is_null())
    _noop("is_not_null_through_arithmetic", Binary(op, col("a"), col("b")).is_not_null())


# --- never-null / three-valued ---------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        lit(1),
        col("a").is_null(),
        col("a").is_not_null(),
        ~col("a").is_null(),
        Array([col("a"), col("b")]),
        Coalesce([col("a"), lit(0)]),
        Greatest([col("a"), lit(0)]),
        Least([lit(0), col("a")]),
    ],
)
def test_never_null_accepts(expr):
    assert never_null(expr)


@pytest.mark.parametrize("expr", [col("a"), lit(None), Coalesce([col("a"), col("b")])])
def test_never_null_declines(expr):
    assert not never_null(expr)


def test_is_null_of_never_null_folds_to_false():
    assert (
        _fire("is_null_of_never_null_to_false", col("a").is_null().is_null()) == lit(False).to_ir()
    )


def test_is_not_null_of_never_null_folds_to_true():
    assert (
        _fire("is_not_null_of_never_null_to_true", Coalesce([col("a"), lit(0)]).is_not_null())
        == lit(True).to_ir()
    )


def test_coalesce_drops_arguments_behind_a_never_null_one():
    got = _fire("drop_coalesce_args_after_never_null", Coalesce([col("a"), lit(0), col("b")]))
    assert got == Coalesce([col("a"), lit(0)]).to_ir()


def test_coalesce_of_never_null_first_argument_collapses():
    got = _fire("drop_coalesce_of_never_null_first_arg", Coalesce([lit(7), col("a")]))
    assert got == lit(7).to_ir()


def test_null_check_tautology():
    got = _fire("null_check_tautology_to_true", col("a").is_null() | col("a").is_not_null())
    assert got == lit(True).to_ir()


def test_null_check_contradiction():
    got = _fire("null_check_contradiction_to_false", col("a").is_null() & col("a").is_not_null())
    assert got == lit(False).to_ir()


def test_null_check_pair_on_different_operands_is_left_alone():
    _noop("null_check_tautology_to_true", col("a").is_null() | col("b").is_not_null())


def test_is_null_moves_inside_case_branches():
    case = bt.when(col("a") > lit(1)).then(col("a")).otherwise(col("b"))
    got = _fire("is_null_through_case_branches", case.is_null())
    assert got["e"] == case.to_ir()["e"]
    assert [b["then"] for b in got["branches"]] == [col("a").is_null().to_ir()]
    assert got["otherwise"] == col("b").is_null().to_ir()


def test_greatest_null_test_is_a_conjunction():
    got = _fire("is_null_of_greatest_to_all_null", Greatest([col("a"), col("b")]).is_null())
    assert got == (col("a").is_null() & col("b").is_null()).to_ir()


def test_greatest_not_null_test_is_a_disjunction():
    got = _fire(
        "is_not_null_of_greatest_to_any_not_null", Greatest([col("a"), col("b")]).is_not_null()
    )
    assert got == (col("a").is_not_null() | col("b").is_not_null()).to_ir()


def test_least_null_test_is_a_conjunction():
    got = _fire("is_null_of_least_to_all_null", Least([col("a"), col("b")]).is_null())
    assert got == (col("a").is_null() & col("b").is_null()).to_ir()


def test_least_not_null_test_is_a_disjunction():
    got = _fire("is_not_null_of_least_to_any_not_null", Least([col("a"), col("b")]).is_not_null())
    assert got == (col("a").is_not_null() | col("b").is_not_null()).to_ir()


# --- the optimizer as a whole stays idempotent ------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        StrFunc("upper", col("s")).is_null(),
        (col("a") + col("b")).is_not_null(),
        Greatest([col("a"), col("b")]).is_null(),
        col("a").is_null() | col("a").is_not_null(),
        Coalesce([col("a"), lit(0), col("b")]),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()


# --- the pure constructors are never-null and total -------------------------


def test_constructors_are_never_null():
    from batcher.plan.expr_ir.nodes import HashRows, MakeStruct

    for expr in (
        Array([col("a"), col("b")]),
        MakeStruct([("x", col("a"))]),
        HashRows([col("a")]),
    ):
        assert never_null(expr)


def test_constructors_are_safe_to_drop():
    # `safe_expr` is what lets a never-null constructor actually fold: a list, a struct
    # and a row hash are pure allocations that cannot raise on any element value.
    from batcher.kyber.rules.leaf_rewrite import safe_expr
    from batcher.plan.expr_ir.nodes import HashRows, MakeStruct

    assert safe_expr(Array([col("a")]))
    assert safe_expr(MakeStruct([("x", col("a"))]))
    assert safe_expr(HashRows([col("a")]))
    # …but only over safe children: a division can still abort inside one.
    assert not safe_expr(Array([col("a") / col("b")]))


def test_null_test_on_a_constructor_folds():
    assert (
        _fire("is_null_of_never_null_to_false", Array([col("a")]).is_null()) == lit(False).to_ir()
    )


def test_coalesce_of_a_constructor_declines_for_want_of_a_type():
    # A constructor is provably never null, so the *nullability* half of this rule is
    # satisfied. It still declines, because the type half cannot be settled: `infer_type`
    # has no answer for `array(...)`, and the guard refuses an unknown rather than guessing
    # that the join is unchanged. Declining costs one redundant COALESCE; guessing wrong
    # hands the user a differently-typed column.
    _noop("drop_coalesce_of_never_null_first_arg", Coalesce([Array([col("a")]), Array([col("b")])]))


def test_coalesce_keeps_an_argument_that_widens_the_result_type():
    # COALESCE takes its type from the *join* of its arguments, so an argument that widens
    # that join is not dead even when the first argument can never be null:
    # `coalesce(5, double_col)` is a DOUBLE column, and returning the bare `5` would hand
    # the user a BIGINT. That is a schema change, not an optimization. Regression for
    # `tests/differential/test_diff_kyber3_coalesce_type.py`, which caught it in the engine.
    _noop("drop_coalesce_of_never_null_first_arg", Coalesce([lit(5), col("f")]))
    _noop("drop_coalesce_args_after_never_null", Coalesce([col("a"), lit(5), col("f")]))


def test_coalesce_still_drops_a_same_typed_trailing_argument():
    # The type guard must not disable the rule outright: with every argument the same type
    # the join is unchanged and the rewrite still fires.
    got = _fire("drop_coalesce_of_never_null_first_arg", Coalesce([lit(5), col("a")]))
    assert got == lit(5).to_ir()


def test_coalesce_keeps_a_trailing_argument_that_can_raise():
    # The dropped arguments need the same totality guard the truncating sibling applies:
    # a columnwise COALESCE evaluates every branch, so discarding one that can abort
    # would delete an error the query would have hit.
    _noop(
        "drop_coalesce_of_never_null_first_arg",
        Coalesce([Array([col("a")]), col("a") / col("b")]),
    )
