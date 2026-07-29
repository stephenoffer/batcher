"""Plan-shape tests for the `kyber.rules.conditional_algebra` families.

The push rules must move a call onto every branch *and* the `ELSE`, must leave the
conditions untouched, and must decline a call that can raise. The branch rules must merge
only *adjacent* equal-valued branches (merging across an intervening branch would change
which value wins) and must drop only a nested condition the enclosing `CASE` has already
settled.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Case, Expr
from batcher.plan.expr_ir.core import Binary, IsNan, MathExpr
from batcher.plan.expr_ir.func_nodes import DateFunc, DateTrunc, Strftime, StrFunc


def _ds():
    import datetime as dt

    return bt.from_pydict(
        {
            "a": [1, 2, 3],
            "s": ["x", "y", "z"],
            "f": [1.0, 2.0, 3.0],
            "t": [dt.datetime(2020, 1, 1)] * 3,
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


def _case(then_value: Expr, else_value: Expr) -> Case:
    return Case([(col("a") > lit(1), then_value)], else_value)


# --- pushing a unary call ---------------------------------------------------

_UNARY_CASES = [
    ("math_fn", lambda inner: MathExpr("abs", inner), lit(-2), lit(3)),
    ("str_fn", lambda inner: StrFunc("upper", inner), lit("ab"), lit("cd")),
    ("date_fn", lambda inner: DateFunc("year", inner), col("t"), col("t")),
    ("date_trunc", lambda inner: DateTrunc(inner, "month"), col("t"), col("t")),
    ("strftime", lambda inner: Strftime(inner, "%Y"), col("t"), col("t")),
    ("nan_check", lambda inner: IsNan(inner), col("f"), lit(1.0)),
]


@pytest.mark.parametrize(
    ("family", "wrap", "then_value", "else_value"),
    _UNARY_CASES,
    ids=[c[0] for c in _UNARY_CASES],
)
def test_unary_call_moves_onto_every_branch(family, wrap, then_value, else_value):
    got = _fire(f"push_{family}_into_case_branches", wrap(_case(then_value, else_value)))
    want = Case([(col("a") > lit(1), wrap(then_value))], wrap(else_value))
    assert got == want.to_ir()


def test_push_preserves_the_string_function_slots():
    call = StrFunc("substr", _case(lit("abcd"), lit("efgh")), start=2, length=3)
    got = _fire("push_str_fn_into_case_branches", call)
    assert [b["then"]["start"] for b in got["branches"]] == [2]
    assert got["otherwise"]["length"] == 3


def test_push_declines_a_call_that_can_raise():
    _noop("push_math_fn_into_case_branches", MathExpr("factorial", _case(lit(1), lit(2))))


def test_push_declines_when_the_operand_is_not_a_case():
    _noop("push_math_fn_into_case_branches", MathExpr("abs", col("a")))


# --- pushing a binary operator ----------------------------------------------


@pytest.mark.parametrize("op", ["and", "or", "bit_and", "bit_or", "bit_xor"])
def test_binary_operator_moves_onto_every_branch(op):
    other = col("a") if op.startswith("bit") else col("a") > lit(0)
    then_value = lit(1) if op.startswith("bit") else lit(True)
    else_value = lit(2) if op.startswith("bit") else lit(False)
    expr = Binary(op, _case(then_value, else_value), other)
    got = _fire(f"push_{op}_into_case_branches", expr)
    want = Case([(col("a") > lit(1), Binary(op, then_value, other))], Binary(op, else_value, other))
    assert got == want.to_ir()


def test_binary_push_keeps_the_operand_order_when_the_case_is_on_the_right():
    expr = Binary("bit_xor", col("a"), _case(lit(1), lit(2)))
    got = _fire("push_bit_xor_into_case_branches", expr)
    want = Case(
        [(col("a") > lit(1), Binary("bit_xor", col("a"), lit(1)))],
        Binary("bit_xor", col("a"), lit(2)),
    )
    assert got == want.to_ir()


def test_binary_push_declines_when_both_sides_are_cases():
    _noop(
        "push_bit_and_into_case_branches",
        Binary("bit_and", _case(lit(1), lit(2)), _case(lit(3), lit(4))),
    )


def test_binary_push_declines_a_shift():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert "push_shift_left_into_case_branches" not in names


# --- merging branches -------------------------------------------------------


def test_adjacent_branches_with_the_same_value_merge():
    expr = Case(
        [(col("a") == lit(1), lit("x")), (col("a") == lit(2), lit("x"))],
        lit("y"),
    )
    got = _fire("merge_case_branches_with_equal_results", expr)
    want = Case(
        [((col("a") == lit(1)) | (col("a") == lit(2)), lit("x"))],
        lit("y"),
    )
    assert got == want.to_ir()


def test_non_adjacent_branches_with_the_same_value_do_not_merge():
    # Merging across the middle branch would let `a = 3` claim a row `a = 2` had won.
    expr = Case(
        [
            (col("a") == lit(1), lit("x")),
            (col("a") == lit(2), lit("z")),
            (col("a") == lit(3), lit("x")),
        ],
        lit("y"),
    )
    _noop("merge_case_branches_with_equal_results", expr)


def test_branches_with_different_values_do_not_merge():
    expr = Case(
        [(col("a") == lit(1), lit("x")), (col("a") == lit(2), lit("z"))],
        lit("y"),
    )
    _noop("merge_case_branches_with_equal_results", expr)


# --- nested conditions ------------------------------------------------------


def test_nested_case_repeating_an_outer_condition_collapses():
    inner = Case([(col("a") > lit(1), lit(2))], lit(3))
    expr = Case([(col("a") > lit(1), lit(1))], inner)
    got = _fire("drop_nested_case_on_settled_condition", expr)
    assert got == Case([(col("a") > lit(1), lit(1))], lit(3)).to_ir()


def test_nested_case_keeps_a_condition_the_outer_case_did_not_test():
    inner = Case([(col("a") > lit(1), lit(2)), (col("a") == lit(0), lit(4))], lit(3))
    expr = Case([(col("a") > lit(1), lit(1))], inner)
    got = _fire("drop_nested_case_on_settled_condition", expr)
    want = Case(
        [(col("a") > lit(1), lit(1))],
        Case([(col("a") == lit(0), lit(4))], lit(3)),
    )
    assert got == want.to_ir()


def test_nested_case_on_an_unrelated_condition_is_left_alone():
    inner = Case([(col("a") == lit(0), lit(2))], lit(3))
    _noop("drop_nested_case_on_settled_condition", Case([(col("a") > lit(1), lit(1))], inner))


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        StrFunc("upper", _case(lit("ab"), lit("cd"))),
        MathExpr("abs", _case(lit(-2), lit(3))),
        Case([(col("a") == lit(1), lit("x")), (col("a") == lit(2), lit("x"))], lit("y")),
        Case([(col("a") > lit(1), lit(1))], Case([(col("a") > lit(1), lit(2))], lit(3))),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()


def test_pushing_collapses_a_case_over_literals_to_a_constant():
    # abs() over two literal branches folds each branch, and the CASE itself then folds
    # away because both results are literals the surrounding rules can decide.
    got = optimize_logical(_proj(MathExpr("abs", _case(lit(-2), lit(2)))))
    assert got.items[0].expr.to_ir() == lit(2).to_ir()


def test_nested_case_keeps_a_dead_branch_that_widens_the_type():
    # A branch the enclosing CASE has already settled is dead by *value*, but not by
    # *type*: CASE takes its type from the join of every arm, so deleting one can narrow
    # the column even though no row could have taken it. Here the unreachable `2.5` is what
    # makes the whole expression a DOUBLE; dropping it would hand the user a BIGINT, which
    # is a schema change rather than an optimization.
    condition = col("a") > lit(1)
    inner = Case([(condition, lit(2.5))], lit(3))
    _noop("drop_nested_case_on_settled_condition", Case([(condition, lit(1))], inner))


def test_nested_case_still_drops_a_dead_branch_of_the_same_type():
    # The type guard must not disable the rule: with every arm the same type the join is
    # unchanged and the dead branch still goes.
    condition = col("a") > lit(1)
    inner = Case([(condition, lit(2))], lit(3))
    got = _fire("drop_nested_case_on_settled_condition", Case([(condition, lit(1))], inner))
    assert got == Case([(condition, lit(1))], lit(3)).to_ir()
