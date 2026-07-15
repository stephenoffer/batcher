"""Constant folding — evaluate constant sub-expressions at plan time.

`col("x") > 2 + 3` becomes `col("x") > 5`; `1 == 1` becomes `true`; and the SQL
date-interval bound `date '1998-12-01' - interval '90' day`, which lowers to a
`cast(cast(date, int64) + (-90), date)` chain, collapses to a single date `Lit`.

Correctness is anchored to the engine, not to Python: a fold only fires where the
result is **bit-identical** to what `bc-expr` would compute. Comparisons and boolean
ops fold freely; integer division/modulo and mixed int/float arithmetic are left
alone; and a `Cast` folds only across the conversions Arrow C++ and `arrow-rs` cannot
disagree on (see `_cast_is_exact`).
"""

from __future__ import annotations

import datetime as _dt

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.expr_ir import Binary, Cast, Expr, Lit, Not
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import LogicalPlan
from batcher.plan.types import DTYPE_REGISTRY
from batcher.plan.visitor import transform_up

__all__ = ["ConstantFolding", "fold_constants", "fold_expression"]

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_COMPARISONS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}


_COMPARISONS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}


# --- Constant folding -------------------------------------------------------


def fold_constants(plan: LogicalPlan) -> LogicalPlan:
    """Fold constant sub-expressions throughout the plan."""
    return transform_up(plan, lambda n: map_node_expressions(n, _fold_expr))


def fold_expression(expr: Expr) -> Expr:
    """Fold one expression's constant sub-trees, bottom-up — a `Lit` if it is fully constant.

    The plan-free entry point. The estimator uses it to decide whether a *projection* is
    constant: it substitutes each provably-constant column with its literal value and folds,
    and a `Lit` result means the output column is a known constant. Sharing this function is
    the point — an estimator that folded arithmetic its own way would be a second, drifting
    definition of what `a - b` means.

    Args:
        expr: The expression to fold.

    Returns:
        The expression with its constant sub-trees folded to literals.
    """
    return transform_expr_up(expr, _fold)


def _fold_expr(expr: Expr) -> Expr:
    return fold_expression(expr)


def _fold(expr: Expr) -> Expr:
    """Leaf rule: fold a node whose children are already folded."""
    if isinstance(expr, Not) and _is_bool(expr.input):
        return Lit(not expr.input.value)
    if isinstance(expr, Cast) and isinstance(expr.input, Lit):
        folded = _fold_cast(expr.input.value, expr.dtype)
        if folded is not None:
            return folded
    if isinstance(expr, Binary) and isinstance(expr.left, Lit) and isinstance(expr.right, Lit):
        folded = _fold_binary(expr.op, expr.left.value, expr.right.value)
        if folded is not None:
            return folded
    return expr


# Arrow types a `Lit` can carry, inferred from its Python value exactly as
# `Lit.to_ir` infers the wire kind (bool before int, datetime before date).
def _literal_type(value: object) -> pa.DataType | None:
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, str):
        return pa.string()
    if isinstance(value, _dt.datetime):
        return pa.timestamp("us")
    if isinstance(value, _dt.date):
        return pa.date32()
    return None


# Integer types, and each temporal type's integer *storage* width. A temporal↔integer
# cast is a pure reinterpretation of the stored epoch offset (days for `date32`,
# microseconds for `timestamp[us]`), so Arrow C++ and `arrow-rs` cannot disagree on it.
_INT_TYPES = frozenset({pa.int32(), pa.int64()})
_TEMPORAL_STORAGE: dict[pa.DataType, pa.DataType] = {
    pa.date32(): pa.int32(),
    pa.timestamp("us"): pa.int64(),
}


def _cast_is_exact(source: pa.DataType, target: pa.DataType) -> bool:
    """Whether `source → target` is a conversion Arrow folds *bit-identically* to what
    `bc-expr` computes, so plan-time folding cannot change a result.

    Admits the identity cast, integer↔integer (Arrow's safe cast raises on overflow
    rather than wrapping, so a folded value is always representable), and
    temporal↔integer (a reinterpretation of the epoch offset). Everything else is
    refused: float→int rounds, string→int parses, float→float narrows, and
    temporal→temporal rebases the unit (`date32`'s days vs `timestamp`'s microseconds) —
    four places where the two Arrow implementations are not guaranteed to agree
    bit-for-bit.
    """
    if source.equals(target):
        return True
    src_int, tgt_int = source in _INT_TYPES, target in _INT_TYPES
    src_temporal, tgt_temporal = source in _TEMPORAL_STORAGE, target in _TEMPORAL_STORAGE
    return (src_int and tgt_temporal) or (src_temporal and tgt_int) or (src_int and tgt_int)


def _cast_scalar(value: object, source: pa.DataType, target: pa.DataType) -> object:
    """`cast(value, target)` via Arrow, routing a temporal end through its storage width.

    Arrow implements `date32 ↔ int32` and `timestamp[us] ↔ int64` — the types that share
    a physical width — but not `date32 ↔ int64`, which is exactly the cast SQL date
    arithmetic produces. Hopping through the temporal type's storage integer makes the
    conversion available without changing its meaning. Raises (caught by `_fold_cast`) if
    a step overflows, since Arrow's default safe cast refuses to wrap.
    """
    scalar = pa.scalar(value, type=source)
    if source.equals(target):
        return scalar.as_py()
    src_storage = _TEMPORAL_STORAGE.get(source)
    if src_storage is not None:  # temporal → integer
        return scalar.cast(src_storage).cast(target).as_py()
    tgt_storage = _TEMPORAL_STORAGE.get(target)
    if tgt_storage is not None:  # integer → temporal
        return scalar.cast(tgt_storage).cast(target).as_py()
    return scalar.cast(target).as_py()  # integer → integer


def _fold_cast(value: object, dtype: str) -> Lit | None:
    """Evaluate `cast(<literal>, dtype)` at plan time, or `None` to leave it to the engine.

    Folds through Arrow's own cast kernel (never hand-rolled epoch arithmetic), gated by
    `_cast_is_exact`. This is what collapses the ubiquitous SQL date-interval bound —
    ``l_shipdate <= date '1998-12-01' - interval '90' day`` lowers to
    ``cast(cast(date, int64) + (-90), date)`` — into a single `Lit`. Left unfolded, that
    bound is opaque to zone-map pruning, source predicate pushdown, and the estimator's
    range selectivity (TPC-H Q1's filter estimated a flat 1/3 against an exactly-known
    date span).
    """
    target = DTYPE_REGISTRY.get(dtype)
    source = _literal_type(value)
    if target is None or source is None or not _cast_is_exact(source, target):
        return None
    try:
        folded = _cast_scalar(value, source, target)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OverflowError, ValueError):
        return None  # out of range / unsupported — the engine decides at run time
    if folded is None or _literal_type(folded) is None:
        return None
    return Lit(folded)


def _fold_binary(op: str, a: object, b: object) -> Lit | None:
    if op in _COMPARISONS:
        if not _comparable(a, b):
            return None
        return Lit(_compare(op, a, b))
    if op in ("and", "or"):
        if _is_bool_val(a) and _is_bool_val(b):
            return Lit(a and b) if op == "and" else Lit(a or b)
        return None
    return _fold_arith(op, a, b)


def _fold_arith(op: str, a: object, b: object) -> Lit | None:
    same_int = _is_int_val(a) and _is_int_val(b)
    same_float = isinstance(a, float) and isinstance(b, float)
    if not (same_int or same_float):
        return None  # mixed int/float: Arrow's promotion differs — don't fold
    if op == "add":
        r = a + b
    elif op == "sub":
        r = a - b
    elif op == "mul":
        r = a * b
    elif op == "div" and same_float:
        if b == 0.0:
            return None
        r = a / b
    else:
        # int div/mod (Arrow truncates ≠ Python), and float mod: leave alone.
        return None
    if same_int and not (_INT64_MIN <= r <= _INT64_MAX):
        return None  # would overflow int64 differently than Arrow
    return Lit(r)


def _compare(op: str, a: object, b: object) -> bool:
    return {
        "gt": a > b,
        "ge": a >= b,
        "lt": a < b,
        "le": a <= b,
        "eq": a == b,
        "ne": a != b,
    }[op]


def _comparable(a: object, b: object) -> bool:
    # Only same-kind comparisons (both numeric / both str / both bool) match the
    # engine; mixing kinds either errors there or has surprising semantics.
    if _is_bool_val(a) and _is_bool_val(b):
        return True
    if isinstance(a, str) and isinstance(b, str):
        return True
    return _is_number(a) and _is_number(b)


def _is_bool(expr: Expr) -> bool:
    return isinstance(expr, Lit) and isinstance(expr.value, bool)


def _is_bool_val(x: object) -> bool:
    return isinstance(x, bool)


def _is_int_val(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: object) -> bool:
    return (isinstance(x, (int, float))) and not isinstance(x, bool)


class ConstantFolding:
    """Pass: fold constant sub-expressions throughout the plan."""

    name = "constant_folding"

    def apply(self, plan: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan:
        return fold_constants(plan)
