"""Schema-aware helpers for expression rules that may only fire on a known type.

Many algebraic identities are sound for one Arrow type and unsound for its
neighbour. `x // 1 -> x` holds for an integer and changes the output type for a
float; `x * 0 -> 0` holds for a non-nullable integer, and is wrong for a nullable
one (`NULL * 0` is `NULL`) and for a float (`inf * 0` is `NaN`). Rules in this
package therefore ask two questions before rewriting, and this module is where both
are answered once.

`node_schema` resolves the schema an expression inside a node is evaluated against.
It is the *input* schema for every node type here: a `Filter` predicate and a
`Project` item both see the columns of the node's input, never the node's own
output. A `None` return means the schema could not be inferred, and every caller
treats that as "decline to fire" rather than "assume".

The `is_*` predicates answer the coarse type question the identities actually turn on,
and `nullable` reports whether a value can be null. Together they let a rule state its
precondition as a sentence -- "integer and non-nullable" -- and have the guard match it
exactly. `schema_rule` is the lifter that runs such a rule over a node, and it declines
before resolving a schema when the node carries nothing the rule could act on.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from batcher.kyber.rules.leaf_rewrite import EXPR_NODES
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_rewrite import (
    contained_types,
    map_node_expressions,
    transform_expr_up,
)
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type

__all__ = [
    "SchemaNode",
    "is_date",
    "is_float",
    "is_integer",
    "is_string",
    "is_timestamp",
    "node_schema",
    "nullable",
    "register_schema_leaf_rule",
    "schema_rule",
]


#: The node types that carry expressions, and so the ones whose expressions can be typed.
#: This is the same set `plan.expr_rewrite.map_node_expressions` rewrites; every one of them
#: evaluates its expressions against its *input's* schema, whether those are a predicate, a
#: projection, a group key, a sort key, or a window frame's partitioning. `Scan` has no
#: input, and `Join` carries no expressions through that path.


def node_schema(node: LogicalPlan) -> SchemaRef | None:
    """The schema the expressions carried by `node` are evaluated against.

    Args:
        node: A node that carries expressions and whose operands a rule wants to type.

    Returns:
        The input schema, or ``None`` when it cannot be inferred.
    """
    if not isinstance(node, EXPR_NODES):
        return None
    try:
        return node.input.available_schema()
    except Exception:
        return None


def _typed(expr: Expr, schema: SchemaRef | None) -> pa.DataType | None:
    if schema is None:
        return None
    try:
        return infer_type(expr, schema)
    except Exception:
        return None


def is_integer(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` provably has an integer type under `schema`."""
    t = _typed(expr, schema)
    return t is not None and pa.types.is_integer(t)


def is_float(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` provably has a floating-point type under `schema`."""
    t = _typed(expr, schema)
    return t is not None and pa.types.is_floating(t)


def is_string(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` provably has a UTF-8 string type under `schema`."""
    t = _typed(expr, schema)
    return t is not None and (pa.types.is_string(t) or pa.types.is_large_string(t))


def is_date(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` provably has a Date type under `schema`.

    The counterpart to `is_timestamp`, and needed for the same reason: a rule that folds a
    comparison onto a temporal literal must emit a *date* literal for a date column and a
    *timestamp* literal for a timestamp one, so it has to be able to tell them apart.
    """
    t = _typed(expr, schema)
    return t is not None and pa.types.is_date(t)


def is_timestamp(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` provably has a Timestamp type under `schema`.

    A `Date` answers ``False``. The distinction matters to any rule that folds a
    comparison onto a temporal literal: a `Date` column compared against a timestamp
    literal is a different comparison, so a rule that cannot tell them apart must not
    fire on either.
    """
    t = _typed(expr, schema)
    return t is not None and pa.types.is_timestamp(t)


def nullable(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` can evaluate to null under `schema`.

    Conservative: anything not provably non-nullable answers ``True``. Only a
    reference to a column the schema marks non-nullable, and a non-null literal, are
    known to be null-free.

    Args:
        expr: The expression to classify.
        schema: The schema it is evaluated against, or ``None``.

    Returns:
        ``True`` unless the expression provably never yields null.
    """
    from batcher.plan.expr_ir import Col, Lit

    if isinstance(expr, Lit):
        return expr.value is None
    if isinstance(expr, Col) and schema is not None and schema.has(expr.name):
        return bool(schema.field(expr.name).nullable)
    return True


def _carries_any(node: LogicalPlan, carries: tuple[type, ...]) -> bool:
    """Whether any expression `node` carries contains one of the `carries` node types.

    Uses `map_node_expressions` with an identity mapper purely as the node-agnostic way
    to enumerate a node's expressions; returning each one unchanged means the node is
    never rebuilt, so this is a read-only sweep.
    """
    found = False

    def probe(expr: Expr) -> Expr:
        nonlocal found
        if not found:
            # `contained_types` is memoized on the expression, so the "does this contain a
            # `Binary`?" question every schema-guarded rule opens with is answered once per
            # expression instead of once per rule per fixpoint pass. `issubclass` over the
            # cached types is exactly the `isinstance` test it replaces.
            found = any(issubclass(kind, carries) for kind in contained_types(expr))
        return expr

    map_node_expressions(node, probe)
    return found


def schema_rule(
    node: LogicalPlan,
    leaf: Callable[[Expr, SchemaRef], Expr],
    *,
    carries: tuple[type, ...],
) -> LogicalPlan | None:
    """Lift a schema-dependent leaf rewrite `leaf(expr, schema) -> Expr` over `node`.

    The counterpart to `leaf_rewrite.rewrite_node` for rules that must know an
    operand's type or nullability. It threads the node's input schema through, and
    declines outright when the schema is unknown -- a type-guarded rule must never
    fire on a guess.

    `carries` is the load-bearing argument, and it is about cost rather than
    correctness. Resolving a schema is not free: `available_schema` rebuilds a pyarrow
    schema up the plan, and this is called once per schema-dependent rule, per node,
    per fixpoint iteration -- while almost every one of those calls is a no-op, because
    a rule matches a handful of expression shapes and passes over the rest. So the node
    types the rule could possibly rewrite are declared up front and checked with a cheap
    `isinstance` sweep first; the schema is resolved only once something might match.
    Without that guard these rules pay their full price on every plan that does not
    contain them.

    Args:
        node: The plan node whose expressions should be rewritten.
        leaf: The leaf rewrite, receiving each sub-expression and the input schema.
        carries: Expression node types the rule can act on. When none of them appear in
            the node's expressions, the rule returns without resolving a schema.

    Returns:
        The rebuilt node, or ``None`` when nothing matched, the schema is unknown, or
        the rewrite changed nothing.
    """
    if not _carries_any(node, carries):
        return None
    schema = node_schema(node)
    if schema is None:
        return None
    new = map_node_expressions(node, lambda e: transform_expr_up(e, lambda x: leaf(x, schema)))
    if new is node:
        return None
    return new if new.to_ir() != node.to_ir() else None


#: The plan nodes a schema-guarded rule visits, as a type alias for annotating the `node`
#: parameter of one. The same fact as `register_schema_leaf_rule`'s `matches` default, and
#: stated once so the two cannot drift: a rule annotated for a node the helper does not visit
#: (or the reverse) type-checks and silently never fires. Three `extra` rule modules each
#: declared their own `_Node` for this.
SchemaNode = Filter | Project


def register_schema_leaf_rule(
    name: str,
    leaf,
    *,
    expr_matches: tuple[type, ...],
    expr_ops: tuple[str, ...] | None = None,
    carries: tuple[type, ...] | None = None,
    matches: tuple[type, ...] = (Filter, Project),
):
    """Register a schema-dependent `leaf(expr, schema) -> Expr` as a normalize-phase rule.

    The schema-guarded counterpart to `leaf_rewrite.register_leaf_rule`, and the same story:
    six rule modules each wrote out the same `node_rule(...)` call, differing only in the
    rule's name, its leaf, and the operator it declares.

    `matches` is `(Filter, Project)` rather than every expression-bearing node because a
    schema-guarded rule needs an input schema to resolve, and `carries` defaults to
    `expr_matches` because the types the leaf can rewrite are exactly the types worth
    resolving a schema for. Both were the value every caller passed.

    Args:
        name: The rule's registry name, unique across the optimizer.
        leaf: The rewrite, receiving each sub-expression and its input schema.
        expr_matches: Expression node types the leaf can rewrite.
        expr_ops: Operator tags the leaf can rewrite, or `None` for every operator. Declare
            an operator's **mirror** too wherever the leaf normalizes the computed side to
            the left, or the rule silently stops firing on the flipped spelling.
        carries: Types whose presence justifies resolving the schema; defaults to
            `expr_matches`. See `schema_rule` for why this guard is load-bearing.
        matches: Plan node types to visit.

    Returns:
        Whatever the registry's `add` returns.
    """
    from batcher.kyber.registry import DEFAULT_REGISTRY
    from batcher.kyber.rule import Phase, node_rule

    guard = expr_matches if carries is None else carries
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: schema_rule(node, _leaf, carries=guard),
            matches=matches,
            expr_schema_fn=leaf,
            expr_matches=expr_matches,
            expr_ops=expr_ops,
        )
    )
