"""Plan-shape, type-preservation and does-not-fire tests for the `arith_extra` rules.

Three things are proven for every rule: it fires into the intended shape (both when called
directly and end to end through the real `Optimizer`), it does **not** change the
expression's output type, and it declines every shape whose rewrite would be unsound —
NULL-destroying (`x * 0`, `x ^ x`), float-typed (where the bitwise kernels' Int64 coercion
would move the type), NaN/`-0.0`-sensitive, or overflowing (`abs(INT64_MIN)`).
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import arith_extra as ax
from batcher.plan.expr_ir import Binary, Cast, Col, Expr, Lit
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.types import infer_type

_INT64_MIN = -(2**63)

_RULE_NAMES = [
    "abs_of_negation",
    "bit_and_minus_one",
    "bit_and_self",
    "bit_or_self",
    "bit_or_zero",
    "bit_xor_zero",
    "collapse_idempotent_math_fn",
    "collapse_nested_rounding",
    "fold_bitwise_literals",
    "fold_math_of_int_literal",
    "rounding_of_int_is_cast",
    "shift_by_zero",
]


def _ds():
    """Int64 `x`/`y`, Float64 `f`, String `s`, Bool `b`."""
    return bt.from_pydict(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "f": [1.0, 2.0, 3.0],
            "s": ["a", "b", "c"],
            "b": [True, False, True],
        }
    )


def _proj(expr: Expr):
    return _ds().select(r=expr)._plan


def _schema():
    return _ds()._plan.available_schema()


def _fire(rule_fn, expr: Expr) -> dict:
    """The IR of `expr` after `rule_fn` — asserting the rule fired and kept the type."""
    node = _proj(expr)
    out = rule_fn(node, None)
    assert out is not None, "rule did not fire"
    rewritten = out.items[0].expr
    assert infer_type(rewritten, _schema()) == infer_type(expr, _schema()), "output type moved"
    return rewritten.to_ir()


def _noop(rule_fn, expr: Expr) -> None:
    """Assert `rule_fn` declines `expr` (the guard held)."""
    assert rule_fn(_proj(expr), None) is None


def _optimized(expr: Expr) -> dict:
    """The IR of `expr` after the *real* optimizer pipeline (every rule, to a fixpoint)."""
    return optimize_logical(_proj(expr)).items[0].expr.to_ir()


# --- registration -----------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert set(_RULE_NAMES) <= names


# --- abs_of_negation --------------------------------------------------------


def test_abs_of_negation_int():
    assert _fire(ax.abs_of_negation, abs(-col("x"))) == MathExpr("abs", Col("x")).to_ir()


def test_abs_of_negation_float():
    assert _fire(ax.abs_of_negation, abs(-col("f"))) == MathExpr("abs", Col("f")).to_ir()


def test_abs_of_negation_end_to_end():
    assert _optimized(abs(-col("x"))) == MathExpr("abs", Col("x")).to_ir()


def test_abs_of_negation_declines_narrow_float():
    # `0 - float32` promotes to Float64; dropping the subtraction would narrow the result
    # back to Float32, so the rule must not fire.
    _noop(ax.abs_of_negation, abs(-col("f").cast("float32")))


def test_abs_of_negation_declines_unknown_type():
    _noop(ax.abs_of_negation, abs(-col("s")))


# --- collapse_idempotent_math_fn / collapse_nested_rounding -----------------


def test_collapse_double_abs():
    assert _fire(ax.collapse_idempotent_math_fn, abs(abs(col("f")))) == abs(col("f")).to_ir()


def test_collapse_double_sign():
    expr = col("f").sign().sign()
    assert _fire(ax.collapse_idempotent_math_fn, expr) == col("f").sign().to_ir()


def test_collapse_double_floor_end_to_end():
    assert _optimized(col("f").floor().floor()) == col("f").floor().to_ir()


def test_collapse_nested_rounding_mixed():
    expr = col("f").floor().ceil()
    assert _fire(ax.collapse_nested_rounding, expr) == col("f").floor().to_ir()


def test_collapse_nested_rounding_round_of_trunc_end_to_end():
    assert _optimized(col("f").trunc().round()) == col("f").trunc().to_ir()


def test_collapse_nested_rounding_declines_abs_inner():
    # abs is not a rounding function: `floor(abs(x))` is real work.
    _noop(ax.collapse_nested_rounding, abs(col("f")).floor())


def test_collapse_idempotent_declines_different_fns():
    _noop(ax.collapse_idempotent_math_fn, col("f").floor().ceil())


def test_collapse_declines_sqrt_of_sqrt():
    # sqrt is NOT idempotent (sqrt(sqrt(16)) == 2, not 4).
    _noop(ax.collapse_idempotent_math_fn, col("f").sqrt().sqrt())
    assert _optimized(col("f").sqrt().sqrt()) == col("f").sqrt().sqrt().to_ir()


# --- rounding_of_int_is_cast ------------------------------------------------


def test_rounding_of_int_is_cast():
    assert _fire(ax.rounding_of_int_is_cast, col("x").floor()) == Cast(Col("x"), "float64").to_ir()


def test_rounding_of_int_is_cast_end_to_end():
    assert _optimized(col("x").floor()) == Cast(Col("x"), "float64").to_ir()


def test_round_of_int_is_not_cast_to_float():
    """`round` is the one member of the family the engine does NOT promote.

    `bc_expr::eval::math::eval_math` special-cases `(Round, Int64)` and returns the array
    untouched, because DuckDB answers BIGINT for `round(bigint)`. Rewriting it to
    `cast(i, float64)` therefore retyped the column *and* lost every value past 2^53 —
    which is precisely the corruption the engine's own special case exists to prevent.
    """
    _noop(ax.rounding_of_int_is_cast, col("x").round())
    assert _optimized(col("x").round()) == MathExpr("round", Col("x")).to_ir()


def test_round_of_large_int_keeps_every_bit():
    """The end-to-end regression: optimized and unoptimized must agree past 2^53."""
    big = 2**53 + 1
    ds = bt.from_pydict({"x": [big, 7]})
    out = ds.select(r=col("x").round(0)).collect()
    assert out.schema.field("r").type == pa.int64()
    assert out.column("r").to_pylist() == [big, 7]


def test_rounding_of_float_is_kept():
    _noop(ax.rounding_of_int_is_cast, col("f").floor())


def test_rounding_of_bool_is_kept():
    _noop(ax.rounding_of_int_is_cast, col("b").floor())


# --- fold_math_of_int_literal -----------------------------------------------


def test_fold_abs_of_int_literal():
    assert _fire(ax.fold_math_of_int_literal, abs(lit(-5))) == Lit(5).to_ir()


def test_fold_sign_of_int_literal():
    assert _fire(ax.fold_math_of_int_literal, lit(-7).sign()) == Lit(-1.0).to_ir()


def test_fold_rounding_of_int_literal():
    assert _fire(ax.fold_math_of_int_literal, lit(5).floor()) == Lit(5.0).to_ir()


def test_fold_sqrt_of_int_literal_end_to_end():
    assert _optimized(lit(4).sqrt()) == Lit(2.0).to_ir()


def test_fold_declines_abs_of_int64_min():
    # `i64::abs(INT64_MIN)` has no i64 result; Python's would silently produce 2**63.
    _noop(ax.fold_math_of_int_literal, abs(lit(_INT64_MIN)))
    assert _optimized(abs(lit(_INT64_MIN))) == abs(lit(_INT64_MIN)).to_ir()


def test_fold_declines_sqrt_of_negative():
    # The engine yields NaN; this IR has no NaN literal to fold to.
    _noop(ax.fold_math_of_int_literal, lit(-4).sqrt())


def test_fold_declines_float_literal():
    # -0.0 and NaN make a float fold's identity observable: refuse every float source.
    _noop(ax.fold_math_of_int_literal, abs(lit(-0.0)))
    _noop(ax.fold_math_of_int_literal, lit(2.5).floor())


def test_fold_declines_non_exact_math():
    # ln/exp/the trig family are not correctly rounded — Python's libm need not agree
    # bit-for-bit with Rust's.
    _noop(ax.fold_math_of_int_literal, lit(2).ln())
    _noop(ax.fold_math_of_int_literal, lit(2).exp())


# --- fold_bitwise_literals --------------------------------------------------


def test_fold_bitwise_and_or_xor():
    assert _fire(ax.fold_bitwise_literals, lit(6).bitwise_and(lit(3))) == Lit(2).to_ir()
    assert _fire(ax.fold_bitwise_literals, lit(6).bitwise_or(lit(3))) == Lit(7).to_ir()
    assert _fire(ax.fold_bitwise_literals, lit(6).bitwise_xor(lit(3))) == Lit(5).to_ir()


def test_fold_bitwise_negative_literals_end_to_end():
    assert _optimized(lit(-2).bitwise_and(lit(-3))) == Lit(-4).to_ir()


def test_fold_bitwise_declines_shift():
    # A shift count of 64+ is not behavior we can pin from the control plane.
    _noop(ax.fold_bitwise_literals, lit(1) << lit(70))
    assert _optimized(lit(1) << lit(70)) == (lit(1) << lit(70)).to_ir()


def test_fold_bitwise_declines_bool_literals():
    # bool is an int subclass in Python — `_is_int_lit` must reject it (the engine casts a
    # Boolean array to Int64, but folding here would change the *literal's* type).
    _noop(ax.fold_bitwise_literals, lit(True).bitwise_and(lit(False)))


# --- bitwise identity elements ----------------------------------------------


def test_bit_or_zero_both_sides():
    assert _fire(ax.bit_or_zero, col("x").bitwise_or(0)) == Col("x").to_ir()
    assert _fire(ax.bit_or_zero, lit(0).bitwise_or(col("x"))) == Col("x").to_ir()


def test_bit_xor_zero():
    assert _fire(ax.bit_xor_zero, col("x") ^ 0) == Col("x").to_ir()


def test_bit_and_minus_one():
    assert _fire(ax.bit_and_minus_one, col("x").bitwise_and(-1)) == Col("x").to_ir()


def test_shift_by_zero_both_ops():
    assert _fire(ax.shift_by_zero, col("x") << 0) == Col("x").to_ir()
    assert _fire(ax.shift_by_zero, col("x") >> 0) == Col("x").to_ir()


def test_shift_by_zero_is_not_commutative():
    # `0 << x` is a real computation, not an identity.
    _noop(ax.shift_by_zero, lit(0) << col("x"))


def test_bit_and_self_and_bit_or_self():
    assert _fire(ax.bit_and_self, col("x").bitwise_and(col("x"))) == Col("x").to_ir()
    assert _fire(ax.bit_or_self, col("x").bitwise_or(col("x"))) == Col("x").to_ir()


def test_bit_identities_end_to_end():
    assert _optimized(col("x").bitwise_or(0)) == Col("x").to_ir()
    assert _optimized(col("x").bitwise_and(-1)) == Col("x").to_ir()
    assert _optimized(col("x").bitwise_and(col("x"))) == Col("x").to_ir()


def test_bit_identity_declines_float_operand():
    # The bitwise kernels CAST to Int64: `f | 0` is an Int64 expression, so dropping the
    # `| 0` would hand back a Float64 one. The type guard must refuse.
    _noop(ax.bit_or_zero, col("f").bitwise_or(0))
    _noop(ax.bit_xor_zero, col("f") ^ 0)
    _noop(ax.bit_and_minus_one, col("f").bitwise_and(-1))
    _noop(ax.shift_by_zero, col("f") << 0)
    _noop(ax.bit_and_self, col("f").bitwise_and(col("f")))
    assert infer_type(col("f").bitwise_or(0), _schema()) == pa.int64()


def test_bit_identity_declines_narrow_int_operand():
    # `cast(x, 'int32') | 0` is Int64; dropping it would narrow the column to Int32.
    _noop(ax.bit_or_zero, col("x").cast("int32").bitwise_or(0))


def test_bit_identity_declines_bool_operand():
    _noop(ax.bit_or_zero, col("b").bitwise_or(0))


def test_bit_identity_declines_wrong_neutral():
    _noop(ax.bit_or_zero, col("x").bitwise_or(1))
    _noop(ax.bit_and_minus_one, col("x").bitwise_and(0))


# --- the unsound shapes that must survive the whole optimizer untouched ------


def test_mul_zero_is_not_folded():
    # NULL * 0 is NULL, not 0 — and NaN * 0 is NaN.
    expr = col("x") * 0
    assert _optimized(expr) == expr.to_ir()


def test_bit_and_zero_is_not_folded():
    expr = col("x").bitwise_and(0)
    assert _optimized(expr) == expr.to_ir()


def test_xor_self_is_not_folded():
    # NULL ^ NULL is NULL, not 0.
    expr = col("x") ^ col("x")
    assert _optimized(expr) == expr.to_ir()


def test_sub_self_and_div_self_are_not_folded():
    assert _optimized(col("x") - col("x")) == (col("x") - col("x")).to_ir()
    assert _optimized(col("x") / col("x")) == (col("x") / col("x")).to_ir()


def test_float_double_negation_is_not_folded():
    # 0.0 - (0.0 - (-0.0)) is +0.0, not -0.0 — the integer case is arith_algebra's.
    negated = -col("f")
    expr = -negated
    assert _optimized(expr) == expr.to_ir()


def test_pow_one_is_not_folded():
    # libm's `pow` is not correctly rounded, so `pow(x, 1) == x` is not provable here.
    expr = col("f") ** 1
    assert _optimized(expr) == expr.to_ir()


def test_abs_of_sqrt_is_not_collapsed():
    # sqrt(-0.0) is -0.0, and abs would flip its sign bit.
    expr = abs(col("f").sqrt())
    assert _optimized(expr) == expr.to_ir()


# --- idempotence (the fixpoint driver requires it) ---------------------------


def test_rules_are_idempotent():
    for rule_fn, expr in [
        (ax.abs_of_negation, abs(-col("x"))),
        (ax.collapse_idempotent_math_fn, abs(abs(col("f")))),
        (ax.collapse_nested_rounding, col("f").floor().ceil()),
        (ax.rounding_of_int_is_cast, col("x").floor()),
        (ax.fold_math_of_int_literal, abs(lit(-5))),
        (ax.fold_bitwise_literals, lit(6).bitwise_and(lit(3))),
        (ax.bit_or_zero, col("x").bitwise_or(0)),
        (ax.bit_xor_zero, col("x") ^ 0),
        (ax.bit_and_minus_one, col("x").bitwise_and(-1)),
        (ax.shift_by_zero, col("x") << 0),
        (ax.bit_and_self, col("x").bitwise_and(col("x"))),
        (ax.bit_or_self, col("x").bitwise_or(col("x"))),
    ]:
        once = rule_fn(_proj(expr), None)
        assert once is not None
        assert rule_fn(once, None) is None, f"{rule_fn.__name__} is not idempotent"


def test_rules_fire_inside_a_filter():
    node = _ds().filter(Binary("bit_or", Col("x"), Lit(0)) > 0)._plan
    out = ax.bit_or_zero(node, None)
    assert out is not None
    assert out.predicate.to_ir() == (col("x") > 0).to_ir()
