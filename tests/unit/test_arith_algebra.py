"""Plan-shape, idempotence, and does-not-fire tests for the arith_algebra rules.

These prove each rule fires into the intended shape, is idempotent (a second
application is a no-op — required for the fixpoint driver), and does NOT fire on
float columns (where the ring laws these rules rely on do not hold) — the
correctness spine, checked without the native engine.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import arith_algebra as aa
from batcher.plan.expr_ir import Binary, Col, Lit

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

_RULE_NAMES = [
    "fold_add_sub_constants",
    "fold_mul_constants",
    "fold_const_minus_sum",
    "fold_neg_sub",
    "factor_common_mul",
]


def _ds():
    return bt.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6], "f": [1.0, 2.0, 3.0]})


def _proj(expr):
    """A Project over int columns x/y and a float column f."""
    return _ds().select(r=expr)._plan


def _flt(pred):
    return _ds().filter(pred)._plan


def _expr_ir(node):
    return node.items[0].expr.to_ir()


# --- registration -----------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    for n in _RULE_NAMES:
        assert n in names, f"{n} not registered"


# --- fold_add_sub_constants -------------------------------------------------


def test_add_add_folds():
    out = aa.fold_add_sub_constants(_proj((col("x") + 5) + 2), None)
    assert _expr_ir(out) == Binary("add", Col("x"), Lit(7)).to_ir()


def test_add_sub_folds():
    out = aa.fold_add_sub_constants(_proj((col("x") + 5) - 2), None)
    assert _expr_ir(out) == Binary("add", Col("x"), Lit(3)).to_ir()


def test_sub_sub_folds_to_sub():
    out = aa.fold_add_sub_constants(_proj((col("x") - 5) - 2), None)
    assert _expr_ir(out) == Binary("sub", Col("x"), Lit(7)).to_ir()


def test_add_sub_cancels_to_value():
    out = aa.fold_add_sub_constants(_proj((col("x") + 5) - 5), None)
    assert _expr_ir(out) == Col("x").to_ir()


def test_add_sub_in_predicate_folds():
    out = aa.fold_add_sub_constants(_flt(((col("x") + 5) + 2) > 0), None)
    assert out.predicate.to_ir() == (Binary("add", Col("x"), Lit(7)) > 0).to_ir()


def test_add_sub_chain_collapses_one_pass():
    out = aa.fold_add_sub_constants(_proj(((col("x") + 1) + 2) + 3), None)
    assert _expr_ir(out) == Binary("add", Col("x"), Lit(6)).to_ir()


def test_add_sub_idempotent():
    once = aa.fold_add_sub_constants(_proj((col("x") + 5) + 2), None)
    assert aa.fold_add_sub_constants(once, None) is None


def test_add_sub_does_not_fire_on_float():
    # Float `+`/`-` is not associative — the rule must leave it alone.
    assert aa.fold_add_sub_constants(_proj((col("f") + 5) + 2), None) is None


def test_add_sub_does_not_fire_single_constant():
    # Only ONE constant — nothing to combine, and ExprSimplification owns `x + 0`.
    assert aa.fold_add_sub_constants(_proj(col("x") + 5), None) is None


def test_add_sub_wrapping_at_boundary():
    # (x + INT64_MAX) + 1 folds to x + wrap(INT64_MAX + 1) == x + INT64_MIN.
    out = aa.fold_add_sub_constants(_proj((col("x") + _INT64_MAX) + 1), None)
    assert _expr_ir(out) == Binary("add", Col("x"), Lit(_INT64_MIN)).to_ir()


# --- fold_mul_constants -----------------------------------------------------


def test_mul_mul_folds():
    out = aa.fold_mul_constants(_proj((col("x") * 3) * 4), None)
    assert _expr_ir(out) == Binary("mul", Col("x"), Lit(12)).to_ir()


def test_mul_mul_cancels_to_value_when_product_one():
    # (x * -1) * -1  →  x  (product ≡ 1).
    out = aa.fold_mul_constants(_proj((col("x") * -1) * -1), None)
    assert _expr_ir(out) == Col("x").to_ir()


def test_mul_mul_keeps_zero_product_as_times_zero():
    # product ≡ 0 stays `x * 0` (null-preserving), never the bare literal 0.
    out = aa.fold_mul_constants(_proj((col("x") * 0) * 4), None)
    assert _expr_ir(out) == Binary("mul", Col("x"), Lit(0)).to_ir()


def test_mul_idempotent():
    once = aa.fold_mul_constants(_proj((col("x") * 3) * 4), None)
    assert aa.fold_mul_constants(once, None) is None


def test_mul_does_not_fire_on_float():
    assert aa.fold_mul_constants(_proj((col("f") * 3) * 4), None) is None


# --- fold_const_minus_sum ---------------------------------------------------


def test_const_minus_sum_folds():
    out = aa.fold_const_minus_sum(_proj(10 - (col("x") + 3)), None)
    assert _expr_ir(out) == Binary("sub", Lit(7), Col("x")).to_ir()


def test_const_minus_diff_folds():
    out = aa.fold_const_minus_sum(_proj(10 - (col("x") - 3)), None)
    assert _expr_ir(out) == Binary("sub", Lit(13), Col("x")).to_ir()


def test_const_minus_sum_idempotent():
    once = aa.fold_const_minus_sum(_proj(10 - (col("x") + 3)), None)
    assert aa.fold_const_minus_sum(once, None) is None


def test_const_minus_sum_does_not_fire_on_float():
    assert aa.fold_const_minus_sum(_proj(10 - (col("f") + 3)), None) is None


# --- fold_neg_sub -----------------------------------------------------------


def test_neg_sub_swaps():
    out = aa.fold_neg_sub(_proj(0 - (col("x") - col("y"))), None)
    assert _expr_ir(out) == Binary("sub", Col("y"), Col("x")).to_ir()


def test_double_negation_reduces():
    # -(-x) desugars to 0 - (0 - x); the rule peels it to x - 0.
    neg_x = -col("x")
    out = aa.fold_neg_sub(_proj(-neg_x), None)
    assert _expr_ir(out) == Binary("sub", Col("x"), Lit(0)).to_ir()


def test_neg_sub_idempotent():
    once = aa.fold_neg_sub(_proj(0 - (col("x") - col("y"))), None)
    assert aa.fold_neg_sub(once, None) is None


def test_neg_sub_does_not_fire_on_float():
    assert aa.fold_neg_sub(_proj(0 - (col("f") - col("y"))), None) is None


def test_neg_sub_does_not_fire_nonzero_const():
    # 5 - (a - b) is not a negation; only 0 - (a - b) fires here.
    assert aa.fold_neg_sub(_proj(5 - (col("x") - col("y"))), None) is None


# --- factor_common_mul ------------------------------------------------------


def test_factor_common_add():
    out = aa.factor_common_mul(_proj(col("x") * 3 + col("x") * 4), None)
    assert _expr_ir(out) == Binary("mul", Col("x"), Lit(7)).to_ir()


def test_factor_common_sub():
    out = aa.factor_common_mul(_proj(col("x") * 5 - col("x") * 2), None)
    assert _expr_ir(out) == Binary("mul", Col("x"), Lit(3)).to_ir()


def test_factor_common_reduces_to_value_when_coeff_one():
    out = aa.factor_common_mul(_proj(col("x") * 3 - col("x") * 2), None)
    assert _expr_ir(out) == Col("x").to_ir()


def test_factor_idempotent():
    once = aa.factor_common_mul(_proj(col("x") * 3 + col("x") * 4), None)
    assert aa.factor_common_mul(once, None) is None


def test_factor_does_not_fire_different_multiplicand():
    assert aa.factor_common_mul(_proj(col("x") * 3 + col("y") * 4), None) is None


def test_factor_does_not_fire_on_float():
    assert aa.factor_common_mul(_proj(col("f") * 3 + col("f") * 4), None) is None
