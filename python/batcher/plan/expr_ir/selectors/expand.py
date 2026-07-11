"""Resolving a selector-bearing expression against a schema.

`expand_selectors` is the one place the relational layer turns a `Selector` into
concrete columns: it walks the surrounding expression, finds the single selector
leaf, and substitutes each matched column back in — so ``numeric().round(2)`` becomes
one rounded expression per numeric column. `has_selector` is the cheap predicate the
projection builders use to decide whether expansion is needed at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Aliased, Expr, InList, Lit
from batcher.plan.expr_ir.nodes import Col
from batcher.plan.expr_ir.selectors.core import Selector

__all__ = ["expand_selectors", "has_selector"]


def _walk(expr: Expr, visit: Callable[[Selector], None]) -> None:
    if isinstance(expr, Selector):
        visit(expr)
        return
    for child in _child_exprs(expr):
        _walk(child, visit)


def _child_exprs(expr: Expr) -> list[Expr]:
    """Every sub-expression of `expr`, whatever its node shape."""
    if isinstance(expr, Aliased):
        return [expr.inner]
    if isinstance(expr, InList):
        return [expr.input]
    if isinstance(expr, Lit):
        return []
    if is_dataclass(expr):
        # `IRNode` nodes are dataclasses; an `Expr`-valued field (possibly nested in a
        # list or a tuple, as in `Case.branches`) is a child. Discriminating on the
        # *value* rather than the field metadata is what makes this correct for the
        # irregular nodes (`Case`) whose children carry no wire metadata.
        return [e for f in fields(expr) for e in _exprs_in(getattr(expr, f.name))]
    return []


def _exprs_in(value: Any) -> list[Expr]:
    if isinstance(value, Expr):
        return [value]
    if isinstance(value, (list, tuple)):
        return [e for v in value for e in _exprs_in(v)]
    return []


def has_selector(expr: Any) -> bool:
    """Whether `expr` is, or contains, a column selector.

    Args:
        expr: Any value; non-expressions are never selectors.

    Returns:
        True when a `Selector` leaf is reachable from `expr`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import has_selector
            >>> has_selector(bt.numeric() + 1), has_selector(bt.col("x") + 1)
            (True, False)
    """
    if not isinstance(expr, Expr):
        return False
    found: list[Selector] = []
    _walk(expr, found.append)
    return bool(found)


def _substitute(expr: Expr, target: Selector, replacement: Expr) -> Expr:
    if expr is target:
        return replacement
    if isinstance(expr, Aliased):
        return Aliased(_substitute(expr.inner, target, replacement), expr.name)
    if isinstance(expr, InList):
        return InList(_substitute(expr.input, target, replacement), expr.values)
    if is_dataclass(expr) and not isinstance(expr, Selector):
        kwargs = {
            f.name: _substitute_value(getattr(expr, f.name), target, replacement)
            for f in fields(expr)
        }
        return type(expr)(**kwargs)
    return expr


def _substitute_value(value: Any, target: Selector, replacement: Expr) -> Any:
    if isinstance(value, Expr):
        return _substitute(value, target, replacement)
    if isinstance(value, list):
        return [_substitute_value(v, target, replacement) for v in value]
    if isinstance(value, tuple):
        return tuple(_substitute_value(v, target, replacement) for v in value)
    return value


def expand_selectors(expr: Expr, columns: list[str], schema: Any | None) -> list[tuple[str, Expr]]:
    """Expand a selector-bearing expression into one `(name, expr)` per matched column.

    The expression may contain at most one distinct selector; each matched column is
    substituted into the surrounding expression in turn, so ``numeric().round(2)``
    becomes one rounded expression per numeric column.

    Args:
        expr: An expression containing exactly one `Selector` leaf.
        columns: The input plan's column names, in order.
        schema: The input plan's `SchemaRef`, or None when it cannot be resolved.

    Returns:
        The expanded `(output_name, expression)` pairs, in input column order.

    Raises:
        PlanError: If the expression contains more than one distinct selector, or is
            wrapped in an `alias(...)` that would name several columns the same.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import expand_selectors
            >>> [n for n, _ in expand_selectors(bt.matches("^[ab]$") + 1, ["a", "b"], None)]
            ['a', 'b']
    """
    # Collect distinct selector *objects* by identity: `Expr.__eq__` builds an
    # expression rather than comparing, so `==`/`in`/`set` are unusable here.
    distinct: list[Selector] = []
    _walk(expr, lambda s: None if any(s is d for d in distinct) else distinct.append(s))
    if len(distinct) != 1:
        raise PlanError(
            f"an expression may reference at most one column selector, found {len(distinct)}: "
            f"{', '.join(repr(d) for d in distinct)}"
        )
    selector = distinct[0]
    matched = selector.matched_columns(columns, schema)
    if isinstance(expr, Aliased):
        # `alias(...)` names exactly one output, so it is only meaningful when the
        # selector narrowed to a single column. The alias then wins over the rename.
        if len(matched) > 1:
            raise PlanError(
                f"alias({expr.name!r}) names a single column but the selector "
                f"{selector!r} matched {len(matched)} columns: {matched}; rename them "
                "with .name.prefix(...) / .name.suffix(...) / .name.map(...) instead"
            )
        return [(expr.name, _substitute(expr.inner, selector, Col(c))) for c in matched]
    return [(selector.output_name(c), _substitute(expr, selector, Col(c))) for c in matched]
