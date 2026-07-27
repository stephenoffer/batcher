"""The shared machinery every leaf-level expression rule is built from.

A *leaf rule* is the dominant shape in Kyber's rule set: a pure `Expr -> Expr`
function that recognizes one algebraic shape and rewrites it, lifted to a plan node
by applying it to every expression the node carries. This module owns the two pieces
all of them need, so the hundred-odd rule bodies stay one function each.

`rewrite_node` is the lifter. It is written around one performance fact: almost every
call is a no-op, because each rule matches a handful of shapes and passes over the
rest, and there are hundreds of rules times hundreds of nodes times the fixpoint
iterations. So it answers "nothing changed" by *object identity* first
(`map_node_expressions` and `transform_expr_up` both preserve structural sharing),
and only falls back to comparing the serialized IR on the path where the object
actually changed — which is needed because a rule may rebuild an equal-but-new tree,
and calling that a change would spin the fixpoint forever.

`safe_expr` is the soundness gate. Most algebraic identities are only valid if
dropping or duplicating a sub-expression preserves the query's *error behavior* as
well as its value. It answers whether an expression is deterministic and total: a
conservative whitelist of columns, literals, wrapping arithmetic, comparisons, the
boolean connectives, the null/NaN/inf predicates, and structural nodes over safe
children. It deliberately excludes division and modulo (a zero divisor aborts),
strict casts (which error on a bad value), and every opaque function call.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.plan.expr_ir import (
    Binary,
    Case,
    Coalesce,
    Col,
    Expr,
    Greatest,
    InList,
    IsNotNull,
    IsNull,
    Least,
    Lit,
    Not,
    NullIf,
)
from batcher.plan.expr_ir.core import IsInf, IsNan
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import LogicalPlan

__all__ = ["SAFE_BINARY_OPS", "rewrite_node", "safe_expr"]

#: Binary operators that are deterministic and cannot raise. Wrapping add/sub/mul,
#: the comparisons, and the Kleene boolean connectives are total. Division and modulo
#: are absent because a zero divisor aborts the query.
SAFE_BINARY_OPS = frozenset({"and", "or", "eq", "ne", "lt", "le", "gt", "ge", "add", "sub", "mul"})


def safe_expr(expr: Expr) -> bool:
    """Whether `expr` is deterministic and total, so dropping or duplicating it
    preserves both the query's value and whether the query errors.

    Conservative by construction: an expression this returns ``False`` for may still
    be safe, but everything it returns ``True`` for provably is. A rule that removes
    a sub-expression MUST gate on this.

    Args:
        expr: The expression to classify.

    Returns:
        ``True`` when the expression cannot raise and has no hidden state.
    """
    if isinstance(expr, (Lit, Col)):
        return True
    if isinstance(expr, Binary):
        return expr.op in SAFE_BINARY_OPS and safe_expr(expr.left) and safe_expr(expr.right)
    if isinstance(expr, (Not, IsNull, IsNotNull, IsNan, IsInf)):
        return safe_expr(expr.input)
    if isinstance(expr, InList):
        return safe_expr(expr.input)
    if isinstance(expr, (Coalesce, Greatest, Least)):
        return all(safe_expr(e) for e in expr.inputs)
    if isinstance(expr, NullIf):
        return safe_expr(expr.left) and safe_expr(expr.right)
    if isinstance(expr, Case):
        return all(safe_expr(c) and safe_expr(v) for c, v in expr.branches) and safe_expr(
            expr.otherwise
        )
    return False


def rewrite_node(node: LogicalPlan, leaf: Callable[[Expr], Expr]) -> LogicalPlan | None:
    """Apply a leaf `Expr -> Expr` rewrite to every expression `node` carries.

    Args:
        node: The plan node whose expressions should be rewritten.
        leaf: The leaf rewrite, applied bottom-up to every sub-expression.

    Returns:
        The rebuilt node, or ``None`` when nothing changed so the driver's fixpoint
        terminates.
    """
    new = map_node_expressions(node, lambda e: transform_expr_up(e, leaf))
    if new is node:  # structural sharing already proved it was a no-op
        return None
    return new if new.to_ir() != node.to_ir() else None
