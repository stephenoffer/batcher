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

__all__ = ["contained_types", "expr_key", "replace_subtrees", "subexpressions"]

# Instance-`__dict__` memo slots, named like the `_c_` slots the plan and expression nodes
# already cache `to_ir` / `referenced_columns` in.
_KEY_SLOT = "_c_expr_key"
_TYPES_SLOT = "_c_node_types"


def expr_key(expr: Expr) -> str:
    """A canonical structural key for `expr` — equal iff the expressions are equal.

    `Expr.__eq__` is overloaded to *build a comparison expression*, so expressions cannot
    be compared (or dict-keyed) directly. The lowered IR is the canonical form.

    Memoized on the node, the way `to_ir` and `referenced_columns` are. The IR dict is
    already cached, but *serializing* it is not, and that is the expensive half: CSE keys
    every subexpression of every projection, then keys them again while replacing the
    matched subtrees, so the same node's whole subtree was re-serialized several times per
    rule pass. Nodes are immutable, and the result is an immutable string, so the cached
    value can be shared freely.
    """
    cached = expr.__dict__.get(_KEY_SLOT)
    if cached is not None:
        return cached
    key = json.dumps(expr.to_ir(), sort_keys=True)
    expr.__dict__[_KEY_SLOT] = key
    return key


def subexpressions(expr: Expr) -> Iterator[Expr]:
    """Every node of `expr`, including `expr` itself, in post-order (children first).

    Left recursive on purpose, and measured. `yield from` re-enters one generator per
    nesting level for every value it forwards, so the textbook fix is an explicit stack —
    but at the sizes that actually occur here it loses: an expression in a projection is
    typically a handful of nodes, and the ``(node, visited)`` bookkeeping a marker stack
    needs costs more per node than the re-entry it removes (measured on a 5-node
    expression: 1.31 us recursive against 1.73 us iterative; the iterative form only pulls
    ahead past ~40 nodes, and then by 6%). The recursion also stays lazy, which the
    `any(...)` callers in the CSE and guard rules exit early from.
    """
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
    # One- and two-child nodes are almost every node of almost every expression, and this
    # recurses over the whole tree once per candidate subexpression. Spelling those cases
    # out avoids a generator, a tuple, a `zip`, and an `all` generator per node; the
    # structural sharing is unchanged (unchanged children return `expr` itself).
    n = len(kids)
    if n == 2:
        left, right = kids
        new_left = replace_subtrees(left, by_key)
        new_right = replace_subtrees(right, by_key)
        if new_left is left and new_right is right:
            return expr
        return _EXPR_REBUILD[type(expr)](expr, (new_left, new_right))
    if n == 1:
        only = kids[0]
        new_only = replace_subtrees(only, by_key)
        if new_only is only:
            return expr
        return _EXPR_REBUILD[type(expr)](expr, (new_only,))
    new = tuple(replace_subtrees(k, by_key) for k in kids)
    if all(a is b for a, b in zip(new, kids, strict=True)):
        return expr  # structural sharing: nothing below changed
    return _EXPR_REBUILD[type(expr)](expr, new)


def contained_types(expr: Expr) -> frozenset[type]:
    """The distinct node types appearing anywhere in `expr`, including `expr` itself.

    Memoized on the node. The optimizer's schema-guarded rules each ask "does this
    expression contain a `Binary` / `StrFunc` / ...?" before doing any work, and answering
    it by walking every subexpression meant re-walking the same trees once per rule, per
    fixpoint pass — the single most repeated question the rule set asks. The answer is a
    property of the (immutable) expression, so it is computed once and the question becomes
    a lookup over a handful of types.

    Args:
        expr: The expression to summarize.

    Returns:
        Every ``type(node)`` reachable from `expr`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_rewrite import contained_types
            >>> from batcher.plan.expr_ir import Col
            >>> Col in contained_types(bt.col("x") + 1)
            True
    """
    cached = expr.__dict__.get(_TYPES_SLOT)
    if cached is None:
        cached = frozenset(type(sub) for sub in subexpressions(expr))
        expr.__dict__[_TYPES_SLOT] = cached
    return cached
