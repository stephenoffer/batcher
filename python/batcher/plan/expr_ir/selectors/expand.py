"""Resolving a selector-bearing expression against a schema.

`expand_selectors` is the one place the relational layer turns a `Selector` into
concrete columns: it walks the surrounding expression, finds the single selector
leaf, and substitutes each matched column back in — so ``numeric().round(2)`` becomes
one rounded expression per numeric column. The walk sees through an aggregate
(``numeric().sum()``) and a window (``numeric().sum().over("g")``), so a selector
expands the same way in ``select``, ``with_columns`` and ``group_by().agg()``.
`has_selector` is the cheap predicate the projection builders use to decide whether
expansion is needed at all, and `resolve_names` is the form for the verbs that take
column *names* (``drop_nulls(subset=...)``, ``unpivot(on=...)``, ``group_by(...)``).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import AggExpr, Aliased, Expr, InList, Lit
from batcher.plan.expr_ir.nodes import Col, WindowExpr
from batcher.plan.expr_ir.selectors.core import Selector

__all__ = [
    "expand_one",
    "expand_selectors",
    "has_selector",
    "resolve_names",
    "single_selector",
    "substitute",
]

#: What a selector can sit inside: a scalar expression or an aggregate over one.
_Tree = Expr | AggExpr


def _walk(expr: _Tree, visit: Callable[[Selector], None]) -> None:
    if isinstance(expr, Selector):
        visit(expr)
        return
    for child in _child_exprs(expr):
        _walk(child, visit)


def _child_exprs(expr: _Tree) -> list[_Tree]:
    """Every sub-expression of `expr`, whatever its node shape."""
    if isinstance(expr, AggExpr):
        return list(expr.operands())
    if isinstance(expr, Aliased):
        return [expr.inner]
    if isinstance(expr, InList):
        return [expr.input]
    if isinstance(expr, WindowExpr):
        return _exprs_in([expr.input, expr.partition_by, expr.order_by])
    if isinstance(expr, Lit):
        return []
    if is_dataclass(expr):
        # `IRNode` nodes are dataclasses; an `Expr`-valued field (possibly nested in a
        # list or a tuple, as in `Case.branches`) is a child. Discriminating on the
        # *value* rather than the field metadata is what makes this correct for the
        # irregular nodes (`Case`) whose children carry no wire metadata.
        return [e for f in fields(expr) for e in _exprs_in(getattr(expr, f.name))]
    return []


def _exprs_in(value: Any) -> list[_Tree]:
    if isinstance(value, (Expr, AggExpr)):
        return [value]
    if isinstance(value, (list, tuple)):
        return [e for v in value for e in _exprs_in(v)]
    return []


def has_selector(expr: Any) -> bool:
    """Whether `expr` is, or contains, a column selector.

    Args:
        expr: Any value; non-expressions are never selectors.

    Returns:
        True when a `Selector` leaf is reachable from `expr`, including through an
        aggregate or a window.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import has_selector
            >>> has_selector(bt.numeric() + 1), has_selector(bt.col("x") + 1)
            (True, False)
            >>> has_selector(bt.numeric().sum())
            True
    """
    kind = type(expr)
    # A bare column or literal is a leaf and can never *contain* anything. `select`
    # asks this of every positional argument, so on a wide projection built from
    # `col(...)` references this is the difference between a type check and a
    # `dataclasses.fields` walk per column.
    if kind is Col or kind is Lit:
        return False
    if not isinstance(expr, (Expr, AggExpr)):
        return False
    found = False

    def mark(_selector: Selector) -> None:
        nonlocal found
        found = True

    _walk(expr, mark)
    return found


def single_selector(expr: _Tree) -> Selector:
    """The one distinct `Selector` leaf of `expr`, or a `PlanError` naming what was found.

    Args:
        expr: An expression or aggregate containing selector leaves.

    Returns:
        The selector, when exactly one distinct selector object is reachable.

    Raises:
        PlanError: If `expr` holds no selector, or more than one distinct one.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import single_selector
            >>> single_selector(bt.numeric().round(1))
            numeric()
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
    return distinct[0]


def substitute(expr: _Tree, target: Selector, replacement: Expr) -> _Tree:
    """`expr` with every occurrence of the `target` selector object replaced.

    Args:
        expr: The expression or aggregate to rebuild.
        target: The selector leaf to replace, matched by identity.
        replacement: What to put in its place (a `Col`, or a renamed selector).

    Returns:
        The rebuilt expression; nodes not on a path to `target` are shared, not copied.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import substitute
            >>> sel = bt.numeric()
            >>> substitute(sel.sum(), sel, bt.col("x"))
            col('x').sum()
    """
    if expr is target:
        return replacement
    if isinstance(expr, AggExpr):
        return expr.map_operands(lambda e: substitute(e, target, replacement))
    if isinstance(expr, Aliased):
        return Aliased(substitute(expr.inner, target, replacement), expr.name)
    if isinstance(expr, InList):
        return InList(substitute(expr.input, target, replacement), expr.values)
    if isinstance(expr, WindowExpr):
        out = copy.copy(expr)
        out.input = _substitute_value(expr.input, target, replacement)
        out.partition_by = _substitute_value(expr.partition_by, target, replacement)
        out.order_by = _substitute_value(expr.order_by, target, replacement)
        return out
    if is_dataclass(expr) and not isinstance(expr, Selector):
        kwargs = {
            f.name: _substitute_value(getattr(expr, f.name), target, replacement)
            for f in fields(expr)
        }
        return type(expr)(**kwargs)
    return expr


def _substitute_value(value: Any, target: Selector, replacement: Expr) -> Any:
    if isinstance(value, (Expr, AggExpr)):
        return substitute(value, target, replacement)
    if isinstance(value, list):
        return [_substitute_value(v, target, replacement) for v in value]
    if isinstance(value, tuple):
        return tuple(_substitute_value(v, target, replacement) for v in value)
    return value


def _alias_of(expr: _Tree) -> str | None:
    """The one output name `expr` was explicitly given, or None."""
    if isinstance(expr, Aliased):
        return expr.name
    if isinstance(expr, AggExpr):
        # An unaliased aggregate over a selector answers `.name` with the rename
        # namespace rather than a string, so only a string is an alias.
        name = expr.name
        return name if isinstance(name, str) else None
    return None


def expand_selectors(
    expr: _Tree,
    columns: list[str],
    schema: Any | None,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[tuple[str, _Tree]]:
    """Expand a selector-bearing expression into one `(name, expr)` per matched column.

    The expression may contain at most one distinct selector; each matched column is
    substituted into the surrounding expression in turn, so ``numeric().round(2)``
    becomes one rounded expression per numeric column and ``numeric().sum()`` one sum
    per numeric column. Each output is named by the selector's ``.name`` rename, else
    after its column.

    Args:
        expr: An expression or aggregate containing exactly one `Selector` leaf.
        columns: The input plan's column names, in order.
        schema: The input plan's `SchemaRef`, or None when it cannot be resolved.
        exclude: Columns the selector never matches here, such as the group keys
            of a `group_by().agg()`.

    Returns:
        The expanded `(output_name, expression)` pairs, in input column order.

    Raises:
        PlanError: If the expression contains more than one distinct selector, or is
            named by an ``alias(...)`` while the selector matched several columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import expand_selectors
            >>> [n for n, _ in expand_selectors(bt.matches("^[ab]$") + 1, ["a", "b"], None)]
            ['a', 'b']
    """
    selector = single_selector(expr)
    matched = [c for c in selector.matched_columns(columns, schema) if c not in exclude]
    alias = _alias_of(expr)
    if alias is None:
        return [(selector.output_name(c), substitute(expr, selector, Col(c))) for c in matched]
    # `alias(...)` names exactly one output, so it is only meaningful when the selector
    # narrowed to a single column. The alias then wins over the rename.
    if len(matched) > 1:
        raise PlanError(
            f"alias({alias!r}) names a single column but the selector "
            f"{selector!r} matched {len(matched)} columns: {matched}; rename them "
            "with .name.prefix(...) / .name.suffix(...) / .name.map(...) instead"
        )
    inner = expr.inner if isinstance(expr, Aliased) else expr
    return [(alias, substitute(inner, selector, Col(c))) for c in matched]


def expand_one(
    expr: _Tree,
    columns: list[str],
    schema: Any | None,
    *,
    exclude: frozenset[str] = frozenset(),
    where: str,
) -> _Tree:
    """Expand a selector-bearing expression that must resolve to exactly one column.

    For a position that names one output, such as a keyword argument
    (``agg(total=bt.numeric().sum())``): the selector has to narrow to one column, since
    several would all want the one name.

    Args:
        expr: An expression or aggregate containing exactly one `Selector` leaf.
        columns: The input plan's column names, in order.
        schema: The input plan's `SchemaRef`, or None when it cannot be resolved.
        exclude: Columns the selector never matches here.
        where: The call site, for the error message, e.g. ``"agg(total=...)"``.

    Returns:
        The expression with the selector replaced by its one matched column.

    Raises:
        PlanError: If the selector matched no column or several.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import expand_one
            >>> expand_one(bt.matches("^a$").sum(), ["a", "b"], None, where="agg(t=...)")
            col('a').sum()
    """
    expanded = expand_selectors(expr, columns, schema, exclude=exclude)
    if len(expanded) != 1:
        raise PlanError(
            f"{where} names a single column but its selector matched "
            f"{[name for name, _ in expanded]}; pass it positionally and rename with "
            ".name.prefix(...) / .name.suffix(...) instead"
        )
    return expanded[0][1]


def resolve_names(value: Any, columns: list[str], schema: Any | None, *, where: str) -> Any:
    """Replace every `Selector` in a column-name argument by the names it matches.

    For the verbs that take column *names* rather than expressions. A bare selector, or
    one inside a list, becomes its matched names in input order; plain names pass
    through; anything that is not a name list is returned unchanged for the verb to
    judge. A selector with a scalar expression composed onto it is refused, since a
    name argument has nowhere to put the computation.

    Args:
        value: The caller's argument: a name, a selector, a list of either, or None.
        columns: The input plan's column names, in order.
        schema: The input plan's `SchemaRef`, or None when it cannot be resolved.
        where: The verb and parameter, for the error message, e.g. ``"unpivot(on=...)"``.

    Returns:
        `value` with every selector expanded to column names.

    Raises:
        PlanError: If a selector carries an expression, which a name list cannot hold.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.selectors import resolve_names
            >>> resolve_names([bt.starts_with("a"), "z"], ["a1", "b", "a2"], None, where="t")
            ['a1', 'a2', 'z']
    """
    if isinstance(value, Selector):
        return value.matched_columns(columns, schema)
    if isinstance(value, Expr) and has_selector(value):
        raise PlanError(
            f"{where} takes column names or a bare column selector, got the expression "
            f"{value!r}; compute it with with_columns(...) first and pass the names"
        )
    if isinstance(value, (list, tuple)) and any(isinstance(v, Expr) for v in value):
        out: list[Any] = []
        for v in value:
            resolved = resolve_names(v, columns, schema, where=where)
            out.extend(resolved if isinstance(v, Selector) else [resolved])
        return out
    return value
