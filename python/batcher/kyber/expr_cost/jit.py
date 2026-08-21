"""Which expressions the Cranelift tier compiles — a conservative mirror of `analyze`.

`crates/bc-codegen/src/analyze.rs` decides, for a real `RecordBatch`, whether an `Expr`
compiles to native code. Kyber must make the same call one layer up, without dtypes and
without a batch, so this module answers **`False` whenever it cannot prove membership**
in the supported subset. Costing errs toward "interpreted", never toward a fast path
that does not exist.

The JIT still makes the real decision per batch — it falls back to the interpreter on
any batch with nulls in a referenced column — so `jit_compilable` means *eligible*, not
*guaranteed compiled*. That is exactly the granularity a cost model needs.

Keep this in lockstep with `analyze.rs`: a subset that grows there and not here only
costs accuracy (an expression priced as interpreted when it now compiles), never
correctness.
"""

from __future__ import annotations

import datetime

from batcher.plan.expr_ir import (
    Binary,
    Case,
    Cast,
    Col,
    Expr,
    IsNotNull,
    IsNull,
    Lit,
    Math2Expr,
    MathExpr,
    Not,
)
from batcher.plan.ir_tags import COMPARISON_OPS

__all__ = ["JIT_SPEEDUP", "jit_compilable"]

# How much cheaper a JIT-compiled expression is per row than the same expression run
# through Arrow kernels. Cranelift removes the per-kernel dispatch and the intermediate
# materialization of every sub-expression, and vectorizes the result. A prior only:
# `calibration` fits the real value from the tier each operator reported running on.
JIT_SPEEDUP = 4.0

# `analyze` accepts these on numeric operands. `concat` (string), the bitwise family,
# and `add_months` (calendar) are rejected there, so they fall back here too.
_JIT_ARITH = frozenset({"add", "sub", "mul", "div", "mod"})
_JIT_BOOL = frozenset({"and", "or"})
# Only Int64/Float64 targets compile, and only the exact widening conversions.
# `bc_arrow::dtype_from_name` resolves these aliases; anything else falls back.
_JIT_CAST_INT = frozenset({"int64", "long"})
_JIT_CAST_FLOAT = frozenset({"float64", "double"})

# Unary math the JIT lowers. `abs` keeps its operand's type; the rest yield f64 — either
# directly (`floor`/`ceil`/`sqrt`/`trunc`) or through a libm libcall. Everything absent
# stays on the interpreter to preserve bit-for-bit parity: `round` (different rounding
# mode), `sign` (select), `degrees`/`radians` (constant multiply), `cot` (reciprocal),
# and `cbrt` (Rust's software `cbrt` differs from the libm call by 1 ULP).
_JIT_MATH_ABS = "abs"
_JIT_MATH_F64 = frozenset(
    {
        "floor", "ceil", "sqrt", "trunc",  # direct lowerings
        # libm unary calls (`libm_unary_symbol`)
        "ln", "log10", "log2", "exp",
        "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
    }
)  # fmt: skip
# Two-arg math that is a single libm call (`libm_binary_symbol`). `round(x, digits)` and
# the integer-semantics `gcd`/`lcm`/`hypot` are not, so they fall back.
_JIT_MATH2 = frozenset({"pow", "atan2"})

# Result kinds `analyze` infers, plus two this layer needs because it cannot see dtypes.
#
# `_ANY` is a bare column: numeric or temporal, unknown. It unifies with either on demand,
# so the ubiquitous `date_col < DATE '1998-01-01'` (which the JIT *does* compile) is
# recognized, while `str_col == 'abc'` is still rejected via its string literal.
# `_NUM` is "numeric of unknown width" — the result of mixing an `_ANY` into arithmetic.
#
# `_I64` / `_F64` are tracked separately only because integer division and modulo can
# *trap* in cranelift while float division cannot, so the two must be distinguished to
# know whether `a / b` compiles.
_BOOL, _I64, _F64, _NUM, _TEMPORAL, _ANY = "bool", "i64", "f64", "num", "temporal", "any"

# Kinds a numeric context accepts: a known numeric, or an unresolved bare column.
_NUMERIC_KINDS = frozenset({_I64, _F64, _NUM, _ANY})


def jit_compilable(expr: Expr) -> bool:
    """Whether `bc-codegen` can compile `expr` to native code.

    Args:
        expr: The scalar expression to test.

    Returns:
        True if the expression is in the JIT's supported subset.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.expr_cost import jit_compilable
            >>> jit_compilable((bt.col("x") > 5) & (bt.col("y") < 2.0))
            True
            >>> jit_compilable(bt.col("s").str.contains("a"))
            False
            >>> jit_compilable(bt.col("x").is_null())
            True
    """
    return _kind(expr) is not None


def _kind(expr: Expr) -> str | None:
    """The result kind the JIT infers for `expr`, or `None` if unsupported.

    A bare temporal result is rejected at the top level by `compile_expr` (it is a
    comparison-only operand type); a bare column resolves to the numeric column clone
    `analyze` accepts.
    """
    kind = _kind_inner(expr)
    if kind == _TEMPORAL:
        return None
    # A bare column at the top level is the numeric column clone `analyze` accepts.
    return _NUM if kind == _ANY else kind


def _kind_inner(expr: Expr) -> str | None:
    if isinstance(expr, Col):
        return _ANY  # dtype not visible here; unified by the operator that uses it
    if isinstance(expr, Lit):
        value = expr.value
        # `bool` is checked first: it is a subclass of `int`, and a bool literal is
        # explicitly unsupported (as is a string literal).
        if isinstance(value, (bool, str)):
            return None
        if isinstance(value, int):
            return _I64
        if isinstance(value, float):
            return _F64
        # Only an actual date/datetime is the temporal operand the JIT accepts. Falling
        # through to `_TEMPORAL` for *everything* else priced a NULL literal (and `Decimal`,
        # `bytes`, …) as compiled — a fast path that does not exist, which is precisely the
        # direction this module's contract says it must never err in.
        if isinstance(value, (datetime.date, datetime.datetime)):
            return _TEMPORAL
        return None
    if isinstance(expr, Not):
        return _BOOL if _kind_inner(expr.input) == _BOOL else None
    if isinstance(expr, Cast):
        return _cast_kind(expr)
    if isinstance(expr, Binary):
        return _binary_kind(expr)
    if isinstance(expr, Case):
        return _case_kind(expr)
    if isinstance(expr, MathExpr):
        return _math_kind(expr)
    if isinstance(expr, Math2Expr):
        return _math2_kind(expr)
    if isinstance(expr, (IsNull, IsNotNull)):
        # `analyze` compiles both: it recurses into the operand and yields Bool, because
        # under the Kleene ABI the answer is just the operand's validity bit. So the
        # operand has to be in the subset, but its *kind* does not matter.
        return _BOOL if _kind_inner(expr.input) is not None else None
    # `is_nan` / `is_inf`, `in_list`, and every remaining function node fall back to the
    # interpreter.
    return None


def _is_safe_int_divisor(expr: Expr) -> bool:
    """Whether `expr` is an *integer* divisor the JIT will compile.

    Cranelift's integer `sdiv`/`srem` *trap* on a zero divisor and on `i64::MIN / -1`,
    which would abort the process, so `bc-codegen` compiles integer division only
    against a constant divisor that is neither 0 nor -1.
    """
    if not isinstance(expr, Lit):
        return False
    value = expr.value
    if isinstance(value, bool):
        return False
    return isinstance(value, int) and value not in (0, -1)


def _cast_kind(expr: Cast) -> str | None:
    """`analyze`'s cast rule: no `try_cast`; only exact widening to Int64/Float64."""
    if expr.try_cast:
        return None
    dtype = expr.dtype.lower()
    inner = _kind_inner(expr.input)
    if inner not in _NUMERIC_KINDS:
        return None
    if dtype in _JIT_CAST_FLOAT:
        return _F64  # int64 -> float64 is exact, and float64 -> float64 is a no-op
    if dtype in _JIT_CAST_INT:
        # float64 -> int64 is rejected by `analyze` (Arrow's rounding/saturation differs
        # from `fcvt`). A bare column is indistinguishable from either at this layer, so
        # the numeric case is accepted and the JIT rejects the F64 -> I64 pair itself.
        return None if inner == _F64 else _I64
    return None  # int32, bool, string, date, ...


def _arith_kind(left: str, right: str) -> str:
    """The result width of `left OP right` under Arrow's numeric promotion."""
    if _F64 in (left, right):
        return _F64
    if left == _I64 and right == _I64:
        return _I64
    return _NUM  # an unresolved bare column is in there somewhere


def _math_kind(expr: MathExpr) -> str | None:
    """`analyze`'s unary-math rule: `abs` preserves its operand's type, the lowered
    set yields f64, and everything else stays on the interpreter for exact parity."""
    inner = _kind_inner(expr.input)
    if inner not in _NUMERIC_KINDS:
        return None  # boolean/temporal operand, or an unsupported sub-expression
    if expr.fn == _JIT_MATH_ABS:
        return inner
    return _F64 if expr.fn in _JIT_MATH_F64 else None


def _math2_kind(expr: Math2Expr) -> str | None:
    """`analyze`'s two-arg-math rule: only the single-libm-call functions compile."""
    if expr.fn not in _JIT_MATH2:
        return None
    left = _kind_inner(expr.left)
    right = _kind_inner(expr.right)
    if left not in _NUMERIC_KINDS or right not in _NUMERIC_KINDS:
        return None
    return _F64  # both operands are promoted to f64 before the libm call


def _binary_kind(expr: Binary) -> str | None:
    """`analyze`'s binary rule: bool ops on bools, arithmetic/comparison on numerics,
    temporal comparison against the same temporal type only."""
    op = expr.op
    left = _kind_inner(expr.left)
    right = _kind_inner(expr.right)
    if left is None or right is None:
        return None
    if op in _JIT_BOOL:
        return _BOOL if left == _BOOL and right == _BOOL else None
    if _BOOL in (left, right):
        return None  # boolean operand to arithmetic/comparison
    if _TEMPORAL in (left, right):
        # Comparison only, and only against the same temporal type. A bare column
        # (`_ANY`) unifies with the temporal side, which is what makes the ubiquitous
        # `date_col < DATE '1998-01-01'` filter price as the compiled op it is.
        compatible = left == right or _ANY in (left, right)
        return _BOOL if op in COMPARISON_OPS and compatible else None
    if op in COMPARISON_OPS:
        return _BOOL
    if op in _JIT_ARITH:
        result = _arith_kind(left, right)
        if op in ("div", "mod") and result != _F64 and not _is_safe_int_divisor(expr.right):
            # Cranelift's *integer* `sdiv`/`srem` trap on a zero divisor (and on
            # `i64::MIN / -1`), so an integer division whose divisor is not a proven-safe
            # constant stays on the interpreter. Float division is IEEE — it yields
            # inf/nan and never traps — so it always compiles.
            return None
        return result
    return None  # concat, bitwise, add_months


def _case_kind(expr: Case) -> str | None:
    """`analyze`'s case rule: every `when` is boolean, every result is numeric.

    The result is f64 if any arm is, else i64 — the same promotion `analyze` applies.
    """
    otherwise = _kind_inner(expr.otherwise)
    if otherwise not in _NUMERIC_KINDS:
        return None
    result = otherwise
    for when, then in expr.branches:
        then_kind = _kind_inner(then)
        if _kind_inner(when) != _BOOL or then_kind not in _NUMERIC_KINDS:
            return None
        result = _arith_kind(result, then_kind)
    return result
