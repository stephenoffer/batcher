"""Structural identity of an expression, and whole-subtree substitution.

`Expr.__eq__` is overloaded to *build a comparison expression*, so expressions cannot be
compared or dict-keyed directly. `expr_key` supplies the canonical identity that makes a
subexpression census possible, and `replace_subtrees` performs the top-down substitution
that common-subexpression elimination is built from.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

from batcher.plan.expr_ir import Expr
from batcher.plan.expr_rewrite.traverse import _EXPR_KIDS, _EXPR_REBUILD

__all__ = ["expr_key", "replace_subtrees", "subexpressions"]


def expr_key(expr: Expr) -> str:
    """A canonical structural key for `expr` — equal iff the expressions are equal.

    `Expr.__eq__` is overloaded to *build a comparison expression*, so expressions cannot
    be compared (or dict-keyed) directly. The lowered IR is the canonical form, and it is
    memoized per node, so this is cheap enough to key a subexpression census on.
    """
    return json.dumps(expr.to_ir(), sort_keys=True)


def subexpressions(expr: Expr) -> Iterator[Expr]:
    """Every node of `expr`, including `expr` itself, in post-order (children first)."""
    kids_of = _EXPR_KIDS.get(type(expr))
    if kids_of is not None:
        for kid in kids_of(expr):
            yield from subexpressions(kid)
    yield expr


def replace_subtrees(expr: Expr, by_key: Mapping[str, Expr]) -> Expr:
    """Replace each **maximal** subtree of `expr` whose `expr_key` is in `by_key`.

    Top-down, unlike `transform_expr_up`: a matched subtree is replaced whole and its
    interior is *not* descended into. That is what makes it usable for common-subexpression
    elimination, where a bottom-up pass would rewrite `a + b` inside `(a + b) * 2` first
    and thereby destroy the very key that identifies the larger candidate.
    """
    replacement = by_key.get(expr_key(expr))
    if replacement is not None:
        return replacement
    kids_of = _EXPR_KIDS.get(type(expr))
    if kids_of is None:
        return expr  # leaf
    kids = kids_of(expr)
    new = tuple(replace_subtrees(k, by_key) for k in kids)
    if all(a is b for a, b in zip(new, kids, strict=True)):
        return expr  # structural sharing: nothing below changed
    return _EXPR_REBUILD[type(expr)](expr, new)
