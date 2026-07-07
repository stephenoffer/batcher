"""Shared traversal for scalar `Expr` trees and for the expressions inside a node.

Like `plan/visitor.py` but one level down: expression rewrites (constant folding,
simplification, and every future algebraic rule) should say *what* to do at a node
and never re-walk the `Binary`/`Not`/`Case`/… ladder. `referenced_columns` and
`remap_columns` in `expr_ir` predate this; new rules build on `transform_expr_up`.

`map_node_expressions` bridges the two levels: it applies an `Expr -> Expr`
rewrite to every expression a plan node carries (a `Filter`'s predicate, a
`Project`'s items, a `Sort`'s keys, …), so a pass is just
`transform_up(plan, lambda n: map_node_expressions(n, rule))`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from batcher.plan.expr_ir import (
    AggExpr,
    Array,
    Binary,
    Case,
    Cast,
    Coalesce,
    Col,
    DateFunc,
    DateTrunc,
    Expr,
    Greatest,
    IsNan,
    IsNotNull,
    IsNull,
    Least,
    ListContains,
    ListFunc,
    ListGet,
    ListJoin,
    ListSlice,
    Math2Expr,
    MathExpr,
    Not,
    NullIf,
    StrFunc,
    StructField,
)
from batcher.plan.expr_ir.core import IsInf
from batcher.plan.expr_ir.image import ImageFunc
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Filter,
    LogicalPlan,
    Project,
    Projection,
    Sort,
    SortKeySpec,
    Window,
    WindowFuncSpec,
)

__all__ = [
    "combine_conjuncts",
    "combine_disjuncts",
    "map_node_expressions",
    "split_conjuncts",
    "split_disjuncts",
    "substitute_columns",
    "transform_expr_up",
]

ExprRule = Callable[[Expr], Expr]


def split_conjuncts(expr: Expr) -> list[Expr]:
    """Flatten a top-level `AND` chain into its conjuncts (a non-AND yields `[expr]`).

    The inverse of `combine_conjuncts`. Used by predicate pushdown and predicate
    inference to reason about each conjunct independently."""
    if isinstance(expr, Binary) and expr.op == "and":
        return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return [expr]


def combine_conjuncts(exprs: list[Expr]) -> Expr:
    """Combine a non-empty list of expressions into a **balanced** `AND` tree.

    The inverse of `split_conjuncts`. The tree is balanced — depth O(log n) rather than
    the naive left-deep O(n) — so a long predicate (a fused chain of hundreds of filters,
    a large `IN` list, a generated boolean) never nests deep enough to exceed the engine's
    recursion limit when the IR is deserialized in the data plane, nor Python's own limit
    when `split_conjuncts` walks it back. `AND` is associative + commutative, so balancing
    preserves the predicate exactly (the conjuncts' left-to-right order is kept). Raises on
    an empty list (there is no neutral predicate to return without inventing a literal)."""
    if not exprs:
        raise ValueError("combine_conjuncts requires at least one expression")
    while len(exprs) > 1:
        # Pairwise-fold one level at a time (a bottom-up balanced tree); an odd tail
        # carries forward. log2(n) passes ⇒ a tree of depth ceil(log2(n)).
        exprs = [
            Binary("and", exprs[i], exprs[i + 1]) if i + 1 < len(exprs) else exprs[i]
            for i in range(0, len(exprs), 2)
        ]
    return exprs[0]


def split_disjuncts(expr: Expr) -> list[Expr]:
    """Flatten a top-level `OR` chain into its disjuncts (a non-OR yields `[expr]`).

    The inverse of `combine_disjuncts`; the `OR` analogue of `split_conjuncts`, used to
    factor a conjunct common to every branch of a disjunction out of the `OR`."""
    if isinstance(expr, Binary) and expr.op == "or":
        return split_disjuncts(expr.left) + split_disjuncts(expr.right)
    return [expr]


def combine_disjuncts(exprs: list[Expr]) -> Expr:
    """Combine a non-empty list of expressions into a left-deep `OR` chain.

    The inverse of `split_disjuncts`; raises on an empty list (no neutral disjunct
    exists without inventing a literal)."""
    if not exprs:
        raise ValueError("combine_disjuncts requires at least one expression")
    out = exprs[0]
    for e in exprs[1:]:
        out = Binary("or", out, e)
    return out


def substitute_columns(expr: Expr, mapping: dict[str, Expr]) -> Expr:
    """Replace every `Col(name)` in `expr` whose `name` is in `mapping` with the
    mapped expression. Used to rewrite a predicate/expression expressed over an
    operator's *output* columns into one over its *input* (e.g. inlining a
    projection's or a group key's defining expression when pushing a filter down)."""

    def sub(e: Expr) -> Expr:
        if isinstance(e, Col) and e.name in mapping:
            return mapping[e.name]
        return e

    return transform_expr_up(expr, sub)


def transform_expr_up(expr: Expr, rule: ExprRule) -> Expr:
    """Bottom-up expression rewrite: rebuild children first, then apply `rule` to
    the rebuilt node. A `rule` only has to handle one node given already-rewritten
    children — the structural recursion lives here, once.

    Dispatch is an O(1) exact-type lookup (`_EXPR_REBUILD`) rather than a long
    ``isinstance`` ladder: this runs for every node of every rule of every fixpoint
    iteration, and leaves (`Col`/`Lit` — the bulk of nodes) previously paid the full
    ladder before falling through. A node type absent from the table is a leaf with no
    sub-expressions (these concrete IR node types are never subclassed, so exact-type
    dispatch matches the old ``isinstance`` semantics)."""
    rebuild = _EXPR_REBUILD.get(type(expr))
    rebuilt = rebuild(expr, rule) if rebuild is not None else expr
    return rule(rebuilt)


# Exact-type → child-rebuild dispatch for `transform_expr_up`. Each entry rebuilds one
# node from its recursively-rewritten children, mirroring the former isinstance ladder
# one-for-one. Leaves (Col, Lit, AggExpr, …) are intentionally absent (rebuilt as-is).
_EXPR_REBUILD: dict[type, Callable[[Expr, ExprRule], Expr]] = {
    Binary: lambda e, r: Binary(e.op, transform_expr_up(e.left, r), transform_expr_up(e.right, r)),
    Not: lambda e, r: Not(transform_expr_up(e.input, r)),
    Cast: lambda e, r: Cast(transform_expr_up(e.input, r), e.dtype, try_cast=e.try_cast),
    IsNull: lambda e, r: IsNull(transform_expr_up(e.input, r)),
    IsNotNull: lambda e, r: IsNotNull(transform_expr_up(e.input, r)),
    IsNan: lambda e, r: IsNan(transform_expr_up(e.input, r)),
    IsInf: lambda e, r: IsInf(transform_expr_up(e.input, r)),
    MathExpr: lambda e, r: MathExpr(e.fn, transform_expr_up(e.input, r)),
    DateFunc: lambda e, r: DateFunc(e.fn, transform_expr_up(e.input, r)),
    DateTrunc: lambda e, r: DateTrunc(transform_expr_up(e.input, r), e.unit),
    ListFunc: lambda e, r: ListFunc(e.fn, transform_expr_up(e.input, r)),
    ListGet: lambda e, r: ListGet(transform_expr_up(e.input, r), e.index),
    ListContains: lambda e, r: ListContains(transform_expr_up(e.input, r), e.value),
    ListSlice: lambda e, r: ListSlice(transform_expr_up(e.input, r), e.offset, e.length),
    StructField: lambda e, r: StructField(transform_expr_up(e.input, r), e.field),
    ListJoin: lambda e, r: ListJoin(transform_expr_up(e.input, r), e.separator),
    StrFunc: lambda e, r: StrFunc(
        e.fn,
        transform_expr_up(e.input, r),
        pattern=e.pattern,
        replacement=e.replacement,
        start=e.start,
        length=e.length,
    ),
    ImageFunc: lambda e, r: ImageFunc(
        e.fn, transform_expr_up(e.input, r), width=e.width, height=e.height
    ),
    Coalesce: lambda e, r: Coalesce([transform_expr_up(x, r) for x in e.inputs]),
    Greatest: lambda e, r: Greatest([transform_expr_up(x, r) for x in e.inputs]),
    Least: lambda e, r: Least([transform_expr_up(x, r) for x in e.inputs]),
    Array: lambda e, r: Array([transform_expr_up(x, r) for x in e.elements]),
    NullIf: lambda e, r: NullIf(transform_expr_up(e.left, r), transform_expr_up(e.right, r)),
    Math2Expr: lambda e, r: Math2Expr(
        e.fn, transform_expr_up(e.left, r), transform_expr_up(e.right, r)
    ),
    Case: lambda e, r: Case(
        [(transform_expr_up(c, r), transform_expr_up(t, r)) for c, t in e.branches],
        transform_expr_up(e.otherwise, r),
    ),
}


def map_node_expressions(node: LogicalPlan, rule: ExprRule) -> LogicalPlan:
    """Apply `rule` to every expression carried directly by `node`, returning a
    rebuilt node (or `node` unchanged for nodes with no expressions: Scan, Join,
    Distinct, Union, Limit, MapBatches)."""
    if isinstance(node, Filter):
        return dataclasses.replace(node, predicate=rule(node.predicate))
    if isinstance(node, Project):
        return dataclasses.replace(
            node,
            items=tuple(Projection(it.alias, rule(it.expr)) for it in node.items),
        )
    if isinstance(node, Aggregate):
        return dataclasses.replace(
            node,
            group_keys=tuple(Projection(k.alias, rule(k.expr)) for k in node.group_keys),
            aggregates=tuple(_map_agg(spec, rule) for spec in node.aggregates),
        )
    if isinstance(node, Sort):
        return dataclasses.replace(node, keys=tuple(_map_sort_key(k, rule) for k in node.keys))
    if isinstance(node, Window):
        return dataclasses.replace(
            node,
            partition_keys=tuple(rule(e) for e in node.partition_keys),
            order_keys=tuple(_map_sort_key(k, rule) for k in node.order_keys),
            functions=tuple(_map_window_fn(f, rule) for f in node.functions),
        )
    return node


def _map_sort_key(key: SortKeySpec, rule: ExprRule) -> SortKeySpec:
    return dataclasses.replace(key, expr=rule(key.expr))


def _map_agg(spec: AggregateSpec, rule: ExprRule) -> AggregateSpec:
    # AggExpr is not a dataclass (custom __slots__ class), so rebuild it directly.
    if spec.agg.input is None:
        return spec
    # Carry the second input (arg_min/arg_max ordering key) through the rewrite too.
    input2 = rule(spec.agg.input2) if spec.agg.input2 is not None else None
    rebuilt = AggExpr(spec.agg.func, rule(spec.agg.input), param=spec.agg.param, input2=input2)
    return dataclasses.replace(spec, agg=rebuilt)


def _map_window_fn(fn: WindowFuncSpec, rule: ExprRule) -> WindowFuncSpec:
    if fn.input is None:
        return fn
    return dataclasses.replace(fn, input=rule(fn.input))
