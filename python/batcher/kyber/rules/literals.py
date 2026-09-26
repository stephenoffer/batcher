"""Literal-value predicates shared by the expression rules: the i64 range and boolean literals.

The engine does integer arithmetic in i64 and **wraps** on overflow (`bc_expr` uses the
wrapping operators, bit-for-bit with the Cranelift JIT), so a rule that folds a constant must
prove the folded value is still representable as i64 — otherwise it introduces a number the
engine would itself have wrapped. `INT64_MIN`/`INT64_MAX` and `in_int64` state that range once.

`is_true_lit`/`is_false_lit` recognize a boolean literal by identity (`is True`), so an integer
`1` or `0` — equal to `True`/`False` in Python — never passes for one.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Expr, Lit

__all__ = ["INT64_MAX", "INT64_MIN", "in_int64", "is_false_lit", "is_true_lit"]

#: The bounds of the engine's integer arithmetic.
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def in_int64(*values: int) -> bool:
    """Whether every value is representable as i64.

    Args:
        *values: Python integers, typically folded literals or range endpoints.

    Returns:
        True when each lies in `[INT64_MIN, INT64_MAX]`.
    """
    return all(INT64_MIN <= v <= INT64_MAX for v in values)


def is_true_lit(expr: Expr) -> bool:
    """Whether `expr` is the boolean literal `TRUE`.

    Args:
        expr: Any expression.

    Returns:
        True only for a `Lit` whose value is the `True` singleton.
    """
    return isinstance(expr, Lit) and expr.value is True


def is_false_lit(expr: Expr) -> bool:
    """Whether `expr` is the boolean literal `FALSE`.

    Args:
        expr: Any expression.

    Returns:
        True only for a `Lit` whose value is the `False` singleton.
    """
    return isinstance(expr, Lit) and expr.value is False
