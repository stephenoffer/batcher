"""The shared machinery every leaf-level expression rule is built from.

A *leaf rule* is the dominant shape in Kyber's rule set: a pure `Expr -> Expr`
function that recognizes one algebraic shape and rewrites it, lifted to a plan node
by applying it to every expression the node carries. This module owns the two pieces
all of them need, so the hundred-odd rule bodies stay one function each.

`rewrite_node` is the lifter, and `node_expr_rule` wraps it into the `(node, ctx)` body a
rule registers — the shape six `rules/exprs/` modules were each spelling out privately,
because a rule *family* builds its leaf from a parameter and so cannot use the
single-leaf `register_leaf_rule`.

`rewrite_node` is written around one performance fact: almost every
call is a no-op, because each rule matches a handful of shapes and passes over the
rest, and there are hundreds of rules times hundreds of nodes times the fixpoint
iterations. So it answers "nothing changed" by *object identity* first
(`map_node_expressions` and `transform_expr_up` both preserve structural sharing),
and only falls back to comparing the serialized IR on the path where the object
actually changed — which is needed because a rule may rebuild an equal-but-new tree,
and calling that a change would spin the fixpoint forever.

`EXPR_NODES` and `register_leaf_rule` are the other two. The node types that carry
expressions are one fact about the plan, and sixteen rule modules had each written the
tuple out; the registration call that turns a leaf into a registered rule is one shape,
and eleven had each written it out. Neither duplicate was reachable by a change: adding a
new expression-bearing plan node meant finding sixteen tuples, and none of them named the
others.

`safe_expr` is the soundness gate. Most algebraic identities are only valid if
dropping or duplicating a sub-expression preserves the query's *error behavior* as
well as its value. It answers whether an expression is deterministic and total: a
conservative whitelist of columns, literals, wrapping arithmetic, comparisons, the
boolean connectives, the null/NaN/inf predicates, the pure constructors (list, struct,
row hash), and structural nodes over safe children. It deliberately excludes division
and modulo (a zero divisor aborts), strict casts (which error on a bad value), and every
opaque function call.
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
from batcher.plan.expr_ir.nodes import Array, HashRows, MakeStruct
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.ir_tags import SAFE_BINARY_OPS
from batcher.plan.logical import Aggregate, Filter, LogicalPlan, Project, Sort, Window
from batcher.plan.visitor import transform_up

__all__ = [
    "EXPR_NODES",
    "SAFE_BINARY_OPS",
    "collapse_doubled_call",
    "collapse_involution",
    "node_expr_rule",
    "register_leaf_rule",
    "rewrite_node",
    "safe_expr",
    "whole_plan_expr_rule",
]

#: The plan nodes that carry expressions, and therefore the `matches` of every leaf rule.
#: One tuple, because it is one fact: a leaf rule rewrites expressions, and these are the
#: nodes that have any. It was written out identically in sixteen rule modules, so a new
#: expression-bearing node type would have had to find all sixteen — and nothing would have
#: failed if it missed one, the rules in that module would simply have stopped firing there.
EXPR_NODES: tuple[type, ...] = (Filter, Project, Aggregate, Sort, Window)

#: Binary operators that are deterministic and cannot raise. Wrapping add/sub/mul,
#: the comparisons, and the Kleene boolean connectives are total. Division and modulo
#: are absent because a zero divisor aborts the query.


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
    if isinstance(expr, Array):
        # A list literal is a pure constructor: it allocates a list per row and cannot
        # fail on any element value, so it is total exactly when its elements are.
        return all(safe_expr(e) for e in expr.elements)
    if isinstance(expr, MakeStruct):
        return all(safe_expr(e) for _, e in expr.fields)
    if isinstance(expr, HashRows):
        # A row hash is defined for every input, nulls included, and never raises.
        return all(safe_expr(e) for e in expr.inputs)
    if isinstance(expr, NullIf):
        return safe_expr(expr.left) and safe_expr(expr.right)
    if isinstance(expr, Case):
        return all(safe_expr(c) and safe_expr(v) for c, v in expr.branches) and safe_expr(
            expr.otherwise
        )
    return False


def whole_plan_expr_rule(leaf: Callable[[Expr], Expr]):
    """Lift a leaf `Expr -> Expr` rewrite into a whole-plan rule body.

    The `plan_rule` counterpart to `rewrite_node`: where that lifts a leaf over the
    expressions of *one* node, this lifts it over every expression of every node in the
    tree. Use it when a rewrite has no node-type it can be indexed on and must simply run
    everywhere.

    Args:
        leaf: The leaf rewrite, applied bottom-up to every sub-expression.

    Returns:
        A `f(plan, ctx) -> plan` suitable for `plan_rule`.
    """

    def apply(plan: LogicalPlan, _ctx) -> LogicalPlan:
        return transform_up(
            plan, lambda node: map_node_expressions(node, lambda e: transform_expr_up(e, leaf))
        )

    return apply


def collapse_doubled_call(node_type: type, fn: str) -> Callable[[Expr], Expr]:
    """The leaf rewrite for an **idempotent** unary function: ``f(f(x))`` -> ``f(x)``.

    `StrFunc`, `ListFunc` and `DateFunc` all carry the same two fields — a function name and
    one input — so "the outer call of a doubled application is redundant" is one rewrite over
    a parameter, not one rewrite per family. Three modules had written it out: `sort`, `unique`,
    `arg_sort`, `flatten` and `normalize` over lists, and `last_day` over dates.

    Not every doubled call qualifies, which is why this takes the function name rather than
    applying to all of them: it holds when `fn` maps its own output to itself. It does *not*
    hold for a function carrying extra arguments that must also agree — ``lpad(lpad(s, n, f),
    m, g)`` collapses only when the widths and fills match — so `text_algebra.lengths` keeps
    its own leaf, which compares them.

    Args:
        node_type: The expression node class, such as `ListFunc`.
        fn: The function name, matched on both the outer and the inner call.

    Returns:
        A leaf `Expr -> Expr` rewrite, unchanged on anything that does not match.
    """

    def leaf(expr: Expr) -> Expr:
        if (
            isinstance(expr, node_type)
            and expr.fn == fn
            and isinstance(expr.input, node_type)
            and expr.input.fn == fn
        ):
            return expr.input
        return expr

    return leaf


def collapse_involution(node_type: type, fn: str) -> Callable[[Expr], Expr]:
    """The leaf rewrite for an **involution**: ``f(f(x))`` -> ``x``, dropping *both* calls.

    The sibling of `collapse_doubled_call`, differing in one line and in the algebra behind it.
    An idempotent function keeps the inner call because a single application is not the
    identity; an involution is its own inverse, so the pair is. `reverse` is the case, over both
    lists and strings, and the two were written out identically.

    **Removing both calls can remove a coercion**, and a caller whose `fn` performs one must
    guard for it — `text_algebra.strings` wraps this and rechecks that the result is already
    `Utf8`, because `reverse` over a `Binary` column was also producing a string.

    Args:
        node_type: The expression node class, such as `StrFunc`.
        fn: The function name, matched on both calls.

    Returns:
        A leaf `Expr -> Expr` rewrite, unchanged on anything that does not match.
    """

    def leaf(expr: Expr) -> Expr:
        if (
            isinstance(expr, node_type)
            and expr.fn == fn
            and isinstance(expr.input, node_type)
            and expr.input.fn == fn
        ):
            return expr.input.input
        return expr

    return leaf


def node_expr_rule(leaf: Callable[[Expr], Expr]):
    """Lift a leaf `Expr -> Expr` rewrite into the `f(node, ctx)` body a node rule registers.

    The node-level counterpart of `whole_plan_expr_rule`, and the last piece of the leaf-rule
    shape that was still being written out by hand. Six modules in `rules/exprs/` each carried a
    private ``_make_<family>_rule`` factory whose whole body was this closure, and
    `register_leaf_rule` had a seventh copy inline as a default-argument lambda. The six existed
    only because a *family* of rules — one per date part, one per shift direction — builds its
    leaf from a parameter and so cannot use `register_leaf_rule`'s single-leaf shape.

    Args:
        leaf: The leaf rewrite, applied bottom-up to every sub-expression of the node.

    Returns:
        A `f(node, ctx) -> node | None` suitable for `node_rule`, returning `None` when nothing
        changed so the driver's fixpoint terminates.
    """

    def apply(node: LogicalPlan, _ctx) -> LogicalPlan | None:
        return rewrite_node(node, leaf)

    return apply


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


def register_leaf_rule(
    name: str,
    leaf: Callable[[Expr], Expr],
    *,
    expr_matches: tuple[type, ...],
    expr_ops: tuple[str, ...] | None = None,
    matches: tuple[type, ...] = EXPR_NODES,
    phase=None,
):
    """Register `leaf` as a normalize-phase rule over every expression-bearing node.

    The last step of writing a leaf rule, and it was the same eleven lines in eleven rule
    modules: wrap the leaf in `rewrite_node`, declare the plan nodes and the expression
    shapes it can act on, and add it to the registry. What actually varied between those
    copies was two arguments, so those are the arguments here.

    `expr_matches` and `expr_ops` are the driver's index, not a behavior: they let it skip a
    rule for an expression that cannot possibly match, and a rule that under-declares them
    silently stops firing. Declaring an operator means declaring its **mirror** too wherever
    the leaf normalizes the computed side to the left, which is why several callers pass
    `(op, COMPARISON_FLIP[op])`.

    Args:
        name: The rule's registry name, unique across the optimizer.
        leaf: The `Expr -> Expr` rewrite, applied bottom-up to every sub-expression.
        expr_matches: Expression node types the leaf can rewrite.
        expr_ops: Operator tags the leaf can rewrite, or `None` for every operator.
        matches: Plan node types to visit. Defaults to every expression-bearing node,
            which is what all but the schema-guarded rules want.
        phase: The phase to register in, defaulting to `Phase.NORMALIZE`.

    Returns:
        Whatever the registry's `add` returns, so a caller can keep using it as a decorator
        target or discard it as these all do.
    """
    from batcher.kyber.registry import DEFAULT_REGISTRY
    from batcher.kyber.rule import Phase, node_rule

    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE if phase is None else phase,
            node_expr_rule(leaf),
            matches=matches,
            expr_fn=leaf,
            expr_matches=expr_matches,
            expr_ops=expr_ops,
        )
    )
