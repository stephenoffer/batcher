"""Recognizing a conjunction that no row can satisfy, from its shape alone.

Exponential backoff (`combine._exponential_backoff`) assumes the conjuncts of an `AND` are
positively correlated, which is the common case and the wrong one for predicates that exclude
each other. `x = 1 AND x = 2` estimated at 3% of the rows under backoff, and
`x > 5 AND x < 3` at 19%, when neither can keep a single row. Both are provable from the
literals, with no statistics at all, so this module proves them rather than estimating.

Only a bare column compared with a literal counts. A comparison through a cast is not the same
predicate on the column: `cast(x, int) = 1 AND x = 1.5` holds for `x = 1.5`, so reading both
sides as bounds on `x` would call a satisfiable filter empty. A NULL makes every comparison
NULL, and SQL keeps only TRUE, so a null row never satisfies any of them and cannot rescue a
contradiction.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Any

from batcher.plan.expr_ir import Binary, Col, Expr, Lit

__all__ = ["provably_unsatisfiable"]

_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


def _kind(value: Any) -> str | None:
    """The comparison family a literal belongs to, or None when it is not ordered here."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return "number" if not (isinstance(value, float) and math.isnan(value)) else None
    if isinstance(value, str):
        return "str"
    # `datetime` subclasses `date`, and comparing the two raises, so they are kept apart.
    if isinstance(value, datetime.datetime):
        return "datetime" if value.tzinfo is None else None
    if isinstance(value, datetime.date):
        return "date"
    return None


@dataclass
class _Constraints:
    """What the conjuncts on one column require of it."""

    kind: str
    eq: list[Any] = field(default_factory=list)
    ne: list[Any] = field(default_factory=list)
    lower: tuple[Any, bool] | None = None  # (bound, strict)
    upper: tuple[Any, bool] | None = None

    def add(self, op: str, value: Any) -> None:
        if op == "eq":
            self.eq.append(value)
        elif op == "ne":
            self.ne.append(value)
        elif op in ("gt", "ge"):
            strict = op == "gt"
            if self.lower is None or value > self.lower[0] or (value == self.lower[0] and strict):
                self.lower = (value, strict)
        else:
            strict = op == "lt"
            if self.upper is None or value < self.upper[0] or (value == self.upper[0] and strict):
                self.upper = (value, strict)

    def empty(self) -> bool:
        if any(v != self.eq[0] for v in self.eq[1:]):
            return True
        if self.eq and any(v == self.eq[0] for v in self.ne):
            return True
        points = self.eq[:1]
        if points:
            return not self._admits(points[0])
        if self.lower is not None and self.upper is not None:
            (lo, lo_strict), (hi, hi_strict) = self.lower, self.upper
            return lo > hi or (lo == hi and (lo_strict or hi_strict))
        return False

    def _admits(self, value: Any) -> bool:
        if self.lower is not None:
            lo, strict = self.lower
            if value < lo or (strict and value == lo):
                return False
        if self.upper is not None:
            hi, strict = self.upper
            if value > hi or (strict and value == hi):
                return False
        return True


def _column_comparison(expr: Expr) -> tuple[str, str, Any] | None:
    """`(column, op, literal)` for a bare `col OP literal` in either order, else None."""
    if not isinstance(expr, Binary) or expr.op not in _FLIP:
        return None
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left.name, expr.op, expr.right.value
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name, _FLIP[expr.op], expr.left.value
    return None


def provably_unsatisfiable(conjuncts: list[Expr]) -> bool:
    """Whether the `AND` of `conjuncts` provably keeps no row.

    True when the comparisons of one column against literals cannot all hold: two different
    equalities, an equality and its own inequality, an equality outside a range, or a range
    whose lower bound passes its upper one. A column compared with literals of different
    families (a number and a string) is left alone, since how the engine would coerce them
    is not decided here.

    Examples:
        .. doctest::

            >>> from batcher import col
            >>> from batcher.kyber.stats.selectivity.contradiction import (
            ...     provably_unsatisfiable,
            ... )
            >>> provably_unsatisfiable([col("x") > 5, col("x") < 3])
            True
            >>> provably_unsatisfiable([col("x") > 5, col("x") < 9])
            False

    Args:
        conjuncts: The flattened conjuncts of one `AND`.

    Returns:
        True only when no row can satisfy every conjunct.
    """
    by_column: dict[str, _Constraints | None] = {}
    for conjunct in conjuncts:
        found = _column_comparison(conjunct)
        if found is None:
            continue
        name, op, value = found
        kind = _kind(value)
        if name in by_column and by_column[name] is None:
            continue
        current = by_column.get(name)
        if kind is None or (current is not None and current.kind != kind):
            by_column[name] = None  # a family this module does not order: never a proof
            continue
        if current is None:
            current = by_column[name] = _Constraints(kind)
        current.add(op, value)
    return any(c is not None and c.empty() for c in by_column.values())
