"""Plan-shape, type-preservation and does-not-fire tests for the `casts` rules.

The cast rules are the highest-risk family in the optimizer: a wrong one does not crash,
it silently returns a value of the wrong *type* (or hides an error, or manufactures a
NULL). So every rule is checked three ways — it fires into the intended shape (directly
*and* end to end through the real `Optimizer`), the expression's inferred output type is
unchanged, and it declines every shape it cannot prove: a fallible cast (string parse,
float→int overflow), a `TRY_CAST` that manufactures nulls, a float literal, and a narrow
target type that a Python literal cannot carry.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, lit, when
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import casts as cx
from batcher.plan.expr_ir import Binary, Cast, Col, Expr, IsNull, Lit
from batcher.plan.expr_ir.core import IsNan
from batcher.plan.types import infer_type

_RULE_NAMES = [
    "canonicalize_cast_dtype_alias",
    "drop_cast_to_inferred_type",
    "drop_infallible_cast_in_null_check",
    "drop_numeric_cast_in_float_predicate",
    "drop_string_cast_in_concat",
    "fold_cast_of_literal",
    "push_cast_into_case_literal_branches",
    "try_cast_to_strict_when_infallible",
]


def _ds():
    """Int64 `x`, Float64 `f`, String `s`, Bool `b`."""
    return bt.from_pydict(
        {"x": [1, 2, 3], "f": [1.0, 2.0, 3.0], "s": ["a", "b", "c"], "b": [True, False, True]}
    )


def _proj(expr: Expr):
    return _ds().select(r=expr)._plan


def _schema():
    return _ds()._plan.available_schema()


def _fire(rule_fn, expr: Expr) -> dict:
    """The IR of `expr` after `rule_fn` — asserting it fired and kept the output type."""
    out = rule_fn(_proj(expr), None)
    assert out is not None, "rule did not fire"
    rewritten = out.items[0].expr
    assert infer_type(rewritten, _schema()) == infer_type(expr, _schema()), "output type moved"
    return rewritten.to_ir()


def _noop(rule_fn, expr: Expr) -> None:
    assert rule_fn(_proj(expr), None) is None


def _optimized(expr: Expr) -> dict:
    return optimize_logical(_proj(expr)).items[0].expr.to_ir()


# --- registration -----------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert set(_RULE_NAMES) <= names


# --- drop_cast_to_inferred_type ---------------------------------------------


def test_drop_identity_cast_over_arithmetic():
    expr = (col("x") + col("x")).cast("int64")
    assert _fire(cx.drop_cast_to_inferred_type, expr) == (col("x") + col("x")).to_ir()


def test_drop_identity_cast_over_a_try_cast():
    # `normalize.simplify` leaves this alone (the try_cast flags differ); a strict cast of a
    # value that is *already* Int32 cannot fail, so the outer cast is the identity.
    expr = col("x").try_cast("int32").cast("int32")
    assert _fire(cx.drop_cast_to_inferred_type, expr) == col("x").try_cast("int32").to_ir()


def test_drop_identity_cast_end_to_end():
    assert _optimized((col("f") * col("f")).cast("float64")) == (col("f") * col("f")).to_ir()


def test_drop_identity_cast_declines_a_real_conversion():
    _noop(cx.drop_cast_to_inferred_type, col("x").cast("float64"))
    _noop(cx.drop_cast_to_inferred_type, col("s").cast("int64"))


# --- canonicalize_cast_dtype_alias ------------------------------------------


def test_canonicalize_alias():
    assert _fire(cx.canonicalize_cast_dtype_alias, col("f").cast("long")) == (
        Cast(Col("f"), "int64").to_ir()
    )
    assert _fire(cx.canonicalize_cast_dtype_alias, col("x").cast("double")) == (
        Cast(Col("x"), "float64").to_ir()
    )


def test_canonicalize_alias_keeps_try_cast_flag():
    out = cx.canonicalize_cast_dtype_alias(_proj(col("s").try_cast("utf8")), None)
    assert out.items[0].expr.to_ir() == Cast(Col("s"), "string", try_cast=True).to_ir()


def test_canonicalize_alias_declines_canonical_name():
    _noop(cx.canonicalize_cast_dtype_alias, col("f").cast("int64"))


def test_canonicalize_alias_end_to_end_exposes_the_identity_cast():
    # 'double' → 'float64', which `drop_cast_to_inferred_type` then removes entirely.
    assert _optimized(col("f").cast("double")) == Col("f").to_ir()


# --- fold_cast_of_literal ---------------------------------------------------


def test_fold_int_literal_to_string():
    assert _fire(cx.fold_cast_of_literal, lit(-5).cast("string")) == Lit("-5").to_ir()


def test_fold_bool_literal_to_int_and_string():
    assert _fire(cx.fold_cast_of_literal, lit(True).cast("int64")) == Lit(1).to_ir()
    assert _fire(cx.fold_cast_of_literal, lit(False).cast("string")) == Lit("false").to_ir()


def test_fold_int_literal_to_float():
    assert _fire(cx.fold_cast_of_literal, lit(5).cast("float64")) == Lit(5.0).to_ir()


def test_fold_try_cast_of_literal_is_the_same_fold():
    # None of the folded conversions can fail, so TRY_CAST computes the same value.
    assert _fire(cx.fold_cast_of_literal, lit(7).try_cast("string")) == Lit("7").to_ir()


def test_fold_end_to_end():
    assert _optimized(lit(True).cast("string")) == Lit("true").to_ir()


def test_fold_declines_int_beyond_exact_float_range():
    # Above 2^53 the int→double conversion rounds; leave it to the engine's kernel.
    _noop(cx.fold_cast_of_literal, lit(2**53 + 1).cast("float64"))


def test_fold_declines_float_source():
    # The engine rounds half-to-even to an integer; pyarrow's safe cast errors instead.
    _noop(cx.fold_cast_of_literal, lit(2.5).cast("int64"))
    _noop(cx.fold_cast_of_literal, lit(1.5).cast("string"))


def test_fold_declines_string_source():
    # A parse — its failure (or TRY_CAST's NULL) is the engine's to decide at run time.
    _noop(cx.fold_cast_of_literal, lit("12").cast("int64"))
    _noop(cx.fold_cast_of_literal, lit("x").try_cast("int64"))


def test_fold_declines_narrow_target():
    # A Python int literal is Int64; folding `cast(5, 'int32')` to `Lit(5)` would *widen*
    # the expression's type from Int32 to Int64.
    _noop(cx.fold_cast_of_literal, lit(5).cast("int32"))
    _noop(cx.fold_cast_of_literal, lit(5).cast("float32"))


# --- push_cast_into_case_literal_branches -----------------------------------


def test_push_cast_into_case():
    expr = when(col("x") > 1).then(1).otherwise(2).cast("float64")
    folded = when(col("x") > 1).then(lit(1.0)).otherwise(lit(2.0))
    assert _fire(cx.push_cast_into_case_literal_branches, expr) == folded.to_ir()


def test_push_cast_into_case_end_to_end():
    expr = when(col("x") > 1).then(1).otherwise(2).cast("string")
    out = _optimized(expr)
    assert out == when(col("x") > 1).then(lit("1")).otherwise(lit("2")).to_ir()


def test_push_cast_into_case_declines_column_arm():
    # Casting a column arm per row is work, not a saving — and under vectorized evaluation
    # a strict cast on an unselected arm can raise an error the original never would.
    _noop(
        cx.push_cast_into_case_literal_branches,
        when(col("x") > 1).then(col("x")).otherwise(2).cast("float64"),
    )


def test_push_cast_into_case_declines_float_arm():
    # `cast(2.5, 'int64')` rounds — refused, so the whole push is refused.
    _noop(
        cx.push_cast_into_case_literal_branches,
        when(col("x") > 1).then(1).otherwise(2.5).cast("int64"),
    )


def test_push_cast_into_case_declines_narrow_target():
    _noop(
        cx.push_cast_into_case_literal_branches,
        when(col("x") > 1).then(1).otherwise(2).cast("int32"),
    )


def test_push_cast_into_case_preserves_the_case_type():
    # The whole hazard: `CASE WHEN FALSE THEN 1 ELSE 2.5 END` is a DOUBLE. Casting it to
    # DOUBLE and pushing must leave a DOUBLE CASE, never an INT one.
    expr = when(col("x") > 1).then(1).otherwise(2).cast("float64")
    out = cx.push_cast_into_case_literal_branches(_proj(expr), None)
    assert infer_type(out.items[0].expr, _schema()) == pa.float64()


# --- try_cast_to_strict_when_infallible -------------------------------------


def test_try_cast_to_strict_int_to_float():
    assert _fire(cx.try_cast_to_strict_when_infallible, col("x").try_cast("float64")) == (
        Cast(Col("x"), "float64").to_ir()
    )


def test_try_cast_to_strict_numeric_to_string():
    assert _fire(cx.try_cast_to_strict_when_infallible, col("f").try_cast("string")) == (
        Cast(Col("f"), "string").to_ir()
    )


def test_try_cast_to_strict_end_to_end():
    assert _optimized(col("b").try_cast("int64")) == Cast(Col("b"), "int64").to_ir()


def test_try_cast_kept_for_a_string_parse():
    # THE case try_cast exists for: an unparseable value must become NULL, not an error.
    _noop(cx.try_cast_to_strict_when_infallible, col("s").try_cast("int64"))
    expr = col("s").try_cast("int64")
    assert _optimized(expr) == expr.to_ir()


def test_try_cast_kept_for_float_to_int():
    # Overflow is a real failure mode of float→int; TRY_CAST must keep NULLing it.
    _noop(cx.try_cast_to_strict_when_infallible, col("f").try_cast("int64"))


# --- drop_infallible_cast_in_null_check -------------------------------------


def test_drop_cast_in_is_null():
    assert _fire(cx.drop_infallible_cast_in_null_check, col("x").cast("float64").is_null()) == (
        IsNull(Col("x")).to_ir()
    )


def test_drop_cast_in_is_not_null_end_to_end():
    assert _optimized(col("x").cast("string").is_not_null()) == col("x").is_not_null().to_ir()


def test_drop_cast_in_null_check_declines_a_try_cast_that_makes_nulls():
    # `is_null(try_cast('a', 'int64'))` is TRUE where `is_null('a')` is FALSE — the cast
    # *manufactures* the null the predicate is asking about.
    expr = col("s").try_cast("int64").is_null()
    _noop(cx.drop_infallible_cast_in_null_check, expr)
    assert _optimized(expr) == expr.to_ir()


def test_drop_cast_in_null_check_declines_a_fallible_strict_cast():
    # A strict string→int cast aborts the query; dropping it would hide the error.
    _noop(cx.drop_infallible_cast_in_null_check, col("s").cast("int64").is_null())


def test_drop_cast_in_null_check_declines_float_to_int():
    _noop(cx.drop_infallible_cast_in_null_check, col("f").cast("int64").is_null())


# --- drop_string_cast_in_concat ---------------------------------------------


def test_drop_string_cast_in_concat():
    expr = Binary("concat", col("x").cast("string"), lit("!"))
    assert _fire(cx.drop_string_cast_in_concat, expr) == (
        Binary("concat", Col("x"), Lit("!")).to_ir()
    )


def test_drop_string_cast_in_concat_both_sides():
    expr = Binary("concat", col("f").cast("string"), col("b").cast("string"))
    assert _fire(cx.drop_string_cast_in_concat, expr) == (
        Binary("concat", Col("f"), Col("b")).to_ir()
    )


def test_drop_string_cast_in_concat_declines_a_temporal_source():
    # Date→string formatting is not pinned across the two Arrow implementations; the
    # explicit cast stays.
    ds = bt.from_pydict({"d": pa.array([1, 2], type=pa.date32())})
    node = ds.select(r=Binary("concat", col("d").cast("string"), lit("!")))._plan
    assert cx.drop_string_cast_in_concat(node, None) is None


def test_drop_string_cast_in_concat_declines_non_string_cast():
    _noop(cx.drop_string_cast_in_concat, Binary("concat", col("x").cast("float64"), lit("!")))


# --- drop_numeric_cast_in_float_predicate -----------------------------------


def test_drop_cast_in_is_nan():
    assert _fire(cx.drop_numeric_cast_in_float_predicate, col("f").cast("float64").is_nan()) == (
        IsNan(Col("f")).to_ir()
    )


def test_drop_cast_in_is_infinite_end_to_end():
    # (the Float64→Float64 cast here is also an identity cast, so either rule may fire
    # first; what matters is that the optimized shape is the bare predicate)
    assert _optimized(col("x").cast("float64").is_infinite()) == col("x").is_infinite().to_ir()


def test_drop_cast_in_is_nan_declines_a_string_source():
    # A string→float cast is a parse: it can fail or NULL, so the predicate would be
    # reading a different array than `is_nan`'s own internal cast produces.
    _noop(cx.drop_numeric_cast_in_float_predicate, col("s").cast("float64").is_nan())


def test_drop_cast_in_is_nan_declines_a_non_float_target():
    _noop(cx.drop_numeric_cast_in_float_predicate, col("f").cast("int64").is_nan())


# --- idempotence ------------------------------------------------------------


def test_rules_are_idempotent():
    for rule_fn, expr in [
        (cx.drop_cast_to_inferred_type, (col("x") + col("x")).cast("int64")),
        (cx.canonicalize_cast_dtype_alias, col("f").cast("long")),
        (cx.fold_cast_of_literal, lit(-5).cast("string")),
        (
            cx.push_cast_into_case_literal_branches,
            when(col("x") > 1).then(1).otherwise(2).cast("float64"),
        ),
        (cx.try_cast_to_strict_when_infallible, col("x").try_cast("float64")),
        (cx.drop_infallible_cast_in_null_check, col("x").cast("float64").is_null()),
        (cx.drop_string_cast_in_concat, Binary("concat", col("x").cast("string"), lit("!"))),
        (cx.drop_numeric_cast_in_float_predicate, col("f").cast("float64").is_nan()),
    ]:
        once = rule_fn(_proj(expr), None)
        assert once is not None
        assert rule_fn(once, None) is None, f"{rule_fn.__name__} is not idempotent"


def test_rules_fire_inside_a_filter():
    node = _ds().filter(col("x").cast("float64").is_null())._plan
    out = cx.drop_infallible_cast_in_null_check(node, None)
    assert out is not None
    assert out.predicate.to_ir() == col("x").is_null().to_ir()
