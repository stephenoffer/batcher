"""Constant list calls, and the two list calls that are the identity.

An `ARRAY[1, 2, 3]` built from literals is a constant, so every list call over it is one
too. `exprs/complex_types` already folds `list_len` and `list_get` on that shape; this
module adds the reductions, the reorderings, and the membership tests, which are the calls
a generated `IN`-list or a feature-vector literal produces.

Folding matters more here than the saved kernel pass suggests. A folded `list_contains`
becomes a boolean literal, which `filter_false_to_empty` can turn into an empty relation —
the plan stops reading the input at all. A folded reduction becomes a numeric literal that
the arithmetic folders then absorb into whatever surrounds it.

The two identity rules are the other half: `list_transform(x, element())` applies the
identity function to every element, and `list_filter(x, true)` keeps all of them. Both are
whole calls that produce their own input, and both are what a generated pipeline emits
when the user-supplied element expression turns out to be trivial.

A fold happens only where Python reproduces the engine's answer *exactly*. Three things
follow from that and are visible in the code: `mean`, `std`, `var` and the norms are
absent (their float answer depends on a summation order Python does not share); `sum` and
`product` fold over integers only, for the same reason; and an array holding a null
element declines outright, because where a null sorts and whether a reduction skips it are
engine decisions this module does not re-derive.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Array, Col, Expr, Lit
from batcher.plan.expr_ir.func_nodes import (
    ListContains,
    ListFilter,
    ListFunc,
    ListPosition,
    ListTransform,
)
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = [
    "LIST_LITERAL_FOLD_RULES",
    "drop_identity_list_filter",
    "drop_identity_list_transform",
]

_NODES = (Filter, Project, Aggregate, Sort, Window)

#: The name `element()` lowers to inside a `list_transform` / `list_filter` body.
_ELEMENT = "element"


def _literal_elements(expr: Expr) -> list[object] | None:
    """The Python values of an `ARRAY[...]` whose every element is a non-null literal.

    A null element makes the whole array decline rather than fold. Where a null sorts,
    whether a reduction skips it, and whether a membership test can match it are all
    engine decisions, and none of them is worth reproducing here for a shape a query
    almost never writes.
    """
    if not isinstance(expr, Array):
        return None
    values: list[object] = []
    for element in expr.elements:
        if not isinstance(element, Lit) or element.value is None:
            return None
        values.append(element.value)
    return values


def _product(values: list) -> object:
    total = 1
    for value in values:
        total *= value
    return total


#: Reductions this module folds, and the Python function that reproduces the engine's
#: answer exactly over a list of non-null literals.
_EXACT_REDUCTIONS: dict[str, Callable[[list], object]] = {
    "len": len,
    "n_unique": lambda values: len(set(values)),
    "min": min,
    "max": max,
    "sum": sum,
    "product": _product,
}

#: Reductions whose float answer depends on summation order, so they fold only over
#: integers. `min`/`max`/`len`/`n_unique` are exact for any comparable element type.
_INTEGER_ONLY = frozenset({"sum", "product"})


def _fold_reduction(fn: str) -> Callable[[Expr], Expr]:
    reduce = _EXACT_REDUCTIONS[fn]

    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, ListFunc) or expr.fn != fn:
            return expr
        values = _literal_elements(expr.input)
        if values is None or not values:
            return expr
        if fn in _INTEGER_ONLY and not all(
            isinstance(v, int) and not isinstance(v, bool) for v in values
        ):
            return expr
        try:
            return Lit(reduce(values))
        except TypeError:  # mixed element types the engine coerces and Python does not
            return expr

    return leaf


def _fold_reordering(fn: str) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, ListFunc) or expr.fn != fn:
            return expr
        values = _literal_elements(expr.input)
        if values is None:
            return expr
        if fn == "reverse":
            ordered = list(reversed(values))
        elif fn == "sort":
            try:
                ordered = sorted(values)
            except TypeError:
                return expr
        else:
            ordered = list(dict.fromkeys(values))
        return Array([Lit(v) for v in ordered])

    return leaf


def _fold_contains(expr: Expr) -> Expr:
    if not isinstance(expr, ListContains):
        return expr
    values = _literal_elements(expr.input)
    if values is None:
        return expr
    return Lit(expr.value in values)


def _fold_position(expr: Expr) -> Expr:
    if not isinstance(expr, ListPosition):
        return expr
    values = _literal_elements(expr.input)
    if values is None:
        return expr
    for index, value in enumerate(values, start=1):
        if value == expr.value:
            return Lit(index)
    return Lit(0)


def _register(name: str, leaf: Callable[[Expr], Expr], expr_matches: tuple[type, ...]):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=expr_matches,
        )
    )


#: Eleven folds over a literal `ARRAY[...]`: the six exact reductions, the three
#: reorderings, and the two membership tests.
LIST_LITERAL_FOLD_RULES = (
    [
        _register(f"fold_list_{fn}_of_literal_array", _fold_reduction(fn), (ListFunc,))
        for fn in _EXACT_REDUCTIONS
    ]
    + [
        _register(f"fold_list_{fn}_of_literal_array", _fold_reordering(fn), (ListFunc,))
        for fn in ("sort", "reverse", "unique")
    ]
    + [
        _register("fold_list_contains_of_literal_array", _fold_contains, (ListContains,)),
        _register("fold_list_position_of_literal_array", _fold_position, (ListPosition,)),
    ]
)


def _identity_transform(expr: Expr) -> Expr:
    if (
        isinstance(expr, ListTransform)
        and isinstance(expr.func, Col)
        and expr.func.name == _ELEMENT
    ):
        return expr.input
    return expr


def _identity_filter(expr: Expr) -> Expr:
    if isinstance(expr, ListFilter) and isinstance(expr.pred, Lit) and expr.pred.value is True:
        return expr.input
    return expr


drop_identity_list_transform = _register(
    "drop_identity_list_transform", _identity_transform, (ListTransform,)
)
"""`list_transform(x, element()) -> x`. Mapping the identity over every element rebuilds
the list it was given, at the cost of a full elementwise pass."""

drop_identity_list_filter = _register("drop_identity_list_filter", _identity_filter, (ListFilter,))
"""`list_filter(x, true) -> x`. A predicate that is the literal `true` keeps every element,
so the filter allocates a new list identical to its input."""
