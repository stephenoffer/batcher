"""Plan-shape tests for the `kyber.rules.text_algebra` families.

Absorption must keep the stronger conjunct and the weaker disjunct and must decline two
patterns with no implication between them; the counting and slice predicates must land on
the direct membership/affix test and decline a slice whose length disagrees with the
literal; the length comparisons must become emptiness tests, and `>= 0` must have no rule
at all (its only faithful restatement flips under a negation).
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir.func_nodes import StrFunc


def _ds():
    return bt.from_pydict({"s": ["abc", "ab", ""], "t": ["x", "y", "z"], "i": [1, 2, 3]})


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


def _sw(pattern: str, column: str = "s"):
    return StrFunc("starts_with", col(column), pattern=pattern)


def _ew(pattern: str, column: str = "s"):
    return StrFunc("ends_with", col(column), pattern=pattern)


def _has(pattern: str, column: str = "s"):
    return StrFunc("contains", col(column), pattern=pattern)


# --- absorption -------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "weak", "strong"),
    [
        ("absorb_weaker_prefix_conjunct", _sw("a/"), _sw("a/b/")),
        ("absorb_weaker_suffix_conjunct", _ew(".gz"), _ew(".tar.gz")),
        ("absorb_weaker_substring_conjunct", _has("bc"), _has("abcd")),
    ],
)
def test_conjunction_keeps_the_stronger_predicate(name, weak, strong):
    assert _fire(name, weak & strong) == strong.to_ir()
    assert _fire(name, strong & weak) == strong.to_ir()


@pytest.mark.parametrize(
    ("name", "weak", "strong"),
    [
        ("absorb_stronger_prefix_disjunct", _sw("a/"), _sw("a/b/")),
        ("absorb_stronger_suffix_disjunct", _ew(".gz"), _ew(".tar.gz")),
        ("absorb_stronger_substring_disjunct", _has("bc"), _has("abcd")),
    ],
)
def test_disjunction_keeps_the_weaker_predicate(name, weak, strong):
    assert _fire(name, weak | strong) == weak.to_ir()
    assert _fire(name, strong | weak) == weak.to_ir()


@pytest.mark.parametrize(
    ("name", "left", "right"),
    [
        ("absorb_weaker_prefix_conjunct", _sw("ab"), _sw("cd")),
        ("absorb_weaker_suffix_conjunct", _ew("ab"), _ew("cd")),
        ("absorb_weaker_substring_conjunct", _has("ab"), _has("cd")),
    ],
)
def test_absorption_declines_unrelated_patterns(name, left, right):
    _noop(name, left & right)


def test_absorption_declines_different_columns():
    _noop("absorb_weaker_prefix_conjunct", _sw("a/") & _sw("a/b/", "t"))


@pytest.mark.parametrize("pattern", [_sw("ab"), _ew("bc"), _has("b")])
def test_equality_wins_a_conjunction(pattern):
    got = _fire("absorb_string_equality_conjunct", (col("s") == lit("abc")) & pattern)
    assert got == (col("s") == lit("abc")).to_ir()


@pytest.mark.parametrize("pattern", [_sw("ab"), _ew("bc"), _has("b")])
def test_pattern_wins_a_disjunction(pattern):
    got = _fire("absorb_string_equality_disjunct", (col("s") == lit("abc")) | pattern)
    assert got == pattern.to_ir()


def test_equality_absorption_declines_when_the_literal_fails_the_pattern():
    _noop("absorb_string_equality_conjunct", (col("s") == lit("abc")) & _sw("zz"))


# --- counting predicates ----------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "expr"),
    [
        ("gt_zero", StrFunc("position", col("s"), pattern="b") > lit(0)),
        ("ge_one", StrFunc("position", col("s"), pattern="b") >= lit(1)),
        ("ne_zero", StrFunc("position", col("s"), pattern="b") != lit(0)),
    ],
)
def test_position_positive_becomes_contains(suffix, expr):
    assert _fire(f"position_{suffix}_to_membership", expr) == _has("b").to_ir()


def test_position_zero_becomes_not_contains():
    expr = StrFunc("position", col("s"), pattern="b") == lit(0)
    assert _fire("position_eq_zero_to_membership", expr) == (~_has("b")).to_ir()


def test_regexp_count_positive_becomes_matches():
    expr = StrFunc("regexp_count", col("s"), pattern="a+") > lit(0)
    want = StrFunc("regexp_matches", col("s"), pattern="a+")
    assert _fire("regexp_count_gt_zero_to_membership", expr) == want.to_ir()


def test_counting_predicate_declines_a_non_zero_threshold():
    _noop(
        "position_gt_zero_to_membership",
        StrFunc("position", col("s"), pattern="b") > lit(2),
    )


# --- slice predicates -------------------------------------------------------


def test_leading_slice_equality_becomes_a_prefix_test():
    expr = StrFunc("substr", col("s"), start=1, length=2) == lit("ab")
    assert _fire("substr_eq_literal_to_affix_test", expr) == _sw("ab").to_ir()


def test_trailing_slice_equality_becomes_a_suffix_test():
    expr = StrFunc("right", col("s"), start=2) == lit("bc")
    assert _fire("right_eq_literal_to_affix_test", expr) == _ew("bc").to_ir()


def test_slice_inequality_becomes_a_negated_affix_test():
    expr = StrFunc("substr", col("s"), start=1, length=2) != lit("ab")
    assert _fire("substr_ne_literal_to_affix_test", expr) == (~_sw("ab")).to_ir()


def test_slice_declines_a_length_mismatch():
    _noop(
        "substr_eq_literal_to_affix_test",
        StrFunc("substr", col("s"), start=1, length=3) == lit("ab"),
    )


def test_slice_declines_an_offset_that_is_not_the_start():
    _noop(
        "substr_eq_literal_to_affix_test",
        StrFunc("substr", col("s"), start=2, length=2) == lit("ab"),
    )


def test_slice_declines_a_non_ascii_literal():
    _noop(
        "substr_eq_literal_to_affix_test",
        StrFunc("substr", col("s"), start=1, length=2) == lit("éé"),
    )


# --- reversal ---------------------------------------------------------------


def test_reverse_equality_moves_onto_the_column():
    expr = StrFunc("reverse", col("s")) == lit("cba")
    assert _fire("reverse_eq_literal_to_column_eq", expr) == (col("s") == lit("abc")).to_ir()


def test_reverse_inequality_moves_onto_the_column():
    expr = StrFunc("reverse", col("s")) != lit("cba")
    assert _fire("reverse_ne_literal_to_column_ne", expr) == (col("s") != lit("abc")).to_ir()


# --- lengths ----------------------------------------------------------------


@pytest.mark.parametrize("fn", ["len", "octet_length", "bit_length"])
@pytest.mark.parametrize(
    ("suffix", "op", "value", "want_op"),
    [
        ("gt_zero", "gt", 0, "ne"),
        ("ge_one", "ge", 1, "ne"),
        ("ne_zero", "ne", 0, "ne"),
        ("le_zero", "le", 0, "eq"),
        ("lt_one", "lt", 1, "eq"),
    ],
)
def test_length_comparison_becomes_an_emptiness_test(fn, suffix, op, value, want_op):
    from batcher.plan.expr_ir import Binary

    expr = Binary(op, StrFunc(fn, col("s")), lit(value))
    want = Binary(want_op, col("s"), lit(""))
    assert _fire(f"{fn}_{suffix}_to_emptiness_test", expr) == want.to_ir()


def test_length_ge_zero_has_no_rule():
    # `length(s) >= 0` is NULL for a null row; `s IS NOT NULL` is false. No rule may
    # claim that equivalence, so none is registered.
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert not any(name.startswith("len_ge_zero") for name in names)


def test_octet_length_zero_becomes_an_equality():
    expr = StrFunc("octet_length", col("s")) == lit(0)
    assert _fire("octet_length_eq_zero_to_emptiness_test", expr) == (col("s") == lit("")).to_ir()


# --- padding ----------------------------------------------------------------


@pytest.mark.parametrize("fn", ["lpad", "rpad"])
def test_repeated_padding_to_the_same_width_collapses(fn):
    inner = StrFunc(fn, col("s"), start=5, pattern="0")
    outer = StrFunc(fn, inner, start=5, pattern="0")
    assert _fire(f"collapse_idempotent_{fn}", outer) == inner.to_ir()


@pytest.mark.parametrize("fn", ["lpad", "rpad"])
def test_padding_to_a_different_width_is_left_alone(fn):
    inner = StrFunc(fn, col("s"), start=5, pattern="0")
    _noop(f"collapse_idempotent_{fn}", StrFunc(fn, inner, start=8, pattern="0"))


@pytest.mark.parametrize("fn", ["lpad", "rpad"])
def test_padding_with_a_different_fill_is_left_alone(fn):
    inner = StrFunc(fn, col("s"), start=5, pattern="0")
    _noop(f"collapse_idempotent_{fn}", StrFunc(fn, inner, start=5, pattern=" "))


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        _sw("a/") & _sw("a/b/"),
        StrFunc("position", col("s"), pattern="b") > lit(0),
        StrFunc("substr", col("s"), start=1, length=2) == lit("ab"),
        StrFunc("len", col("s")) > lit(0),
        StrFunc("reverse", col("s")) == lit("cba"),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()
