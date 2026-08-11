"""Constraints between two columns of the same row.

`check()` can already express any of these — ``check(col("start") <= col("end"))``. What
this module adds is that the predicate it builds is **total**: a row where either side is
NULL is valid rather than NULL-valid, which is both the convention every other constraint
here follows and the property that lets the contract be discharged from metadata instead of
executed (`api.dataset.meta.prove`). A hand-written `check` is neither.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.api.dataset.dq.constraints import RowConstraint
from batcher.plan.expr_ir import Col, Expr

__all__ = ["OPERATORS", "compare_columns"]

OPERATORS: dict[str, str] = {
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
    "==": "eq",
    "!=": "ne",
}
"""The comparison operators `compare_columns` accepts, spelled as in Python."""


def _apply(left: Expr, op: str, right: Expr) -> Expr:
    """The comparison `op` between two expressions, dispatched from its spelling."""
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "==":
        return left == right
    return left != right


def compare_columns(left: str, op: str, right: str) -> RowConstraint:
    """`left op right` must hold for every row where both columns are present.

    Args:
        left: The left-hand column name.
        op: One of the keys of `OPERATORS`.
        right: The right-hand column name.

    Returns:
        The row constraint.
    """
    if op not in OPERATORS:
        known = ", ".join(sorted(OPERATORS))
        raise PlanError(f"compare_columns({left!r}, {op!r}, {right!r}): use one of {known}.")
    a, b = Col(left), Col(right)
    either_null = a.is_null() | b.is_null()
    return RowConstraint(f"compare_columns({left} {op} {right})", either_null | _apply(a, op, b))
