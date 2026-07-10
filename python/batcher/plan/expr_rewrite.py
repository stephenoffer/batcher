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
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

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
    InList,
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
    WindowExpr,
)
from batcher.plan.expr_ir.core import IsInf
from batcher.plan.expr_ir.image import ImageFunc
from batcher.plan.expr_ir.nodes import HashRows
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
    "hoist_windows",
    "is_bare_window",
    "map_node_expressions",
    "split_conjuncts",
    "split_disjuncts",
    "substitute_columns",
    "transform_expr_up",
]

ExprRule = Callable[[Expr], Expr]

# Prefix of the synthetic column a hoisted window lands in. Chosen to be un-typeable
# as a user column (leading dunder-ish underscores) so a hoist can never shadow one.
WINDOW_TEMP_PREFIX = "__bt_win_"

# One element of a plan node's expression-carrying tuple (a Projection, SortKeySpec,
# AggregateSpec, WindowFuncSpec, or a bare Expr) — see `_map_tuple`.
_T = TypeVar("_T")


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

    Dispatch is an O(1) exact-type lookup (`_EXPR_KIDS`) rather than a long
    ``isinstance`` ladder: this runs for every node of every rule of every fixpoint
    iteration, and leaves (`Col`/`Lit` — the bulk of nodes) previously paid the full
    ladder before falling through. A node type absent from the table is a leaf with no
    sub-expressions (these concrete IR node types are never subclassed, so exact-type
    dispatch matches the old ``isinstance`` semantics).

    **Structural sharing:** a node whose rewritten children are all the *same objects*
    (`is`) is not rebuilt — `expr` itself is passed to `rule`. Most rules match a
    handful of nodes and leave the rest alone, so without this every rule reallocated
    the entire expression tree (discarding each node's memoized `to_ir`, and
    re-running the enclosing plan node's column validation) on every fixpoint pass.
    Nodes are immutable and value-typed, so reusing one is indistinguishable from
    rebuilding an equal copy — except that the optimizer's `is`-based change detection
    and per-node memo caches now hit."""
    kids_of = _EXPR_KIDS.get(type(expr))
    if kids_of is None:
        return rule(expr)  # leaf: no sub-expressions
    kids = kids_of(expr)
    new = tuple(transform_expr_up(k, rule) for k in kids)
    rebuilt = (
        expr
        if all(a is b for a, b in zip(new, kids, strict=True))
        else _EXPR_REBUILD[type(expr)](expr, new)
    )
    return rule(rebuilt)


def is_bare_window(expr: Expr) -> bool:
    """True when `expr` is a window whose argument holds no further window.

    Such a window needs no surrounding `Project`: it can be named directly by a
    `Window` node. `Dataset.with_columns` takes that shortcut; anything else goes
    through `hoist_windows`.
    """
    if not isinstance(expr, WindowExpr):
        return False
    return expr.input is None or not _contains_window(expr.input)


def _contains_window(expr: Expr) -> bool:
    found = False

    def probe(node: Expr) -> Expr:
        nonlocal found
        if isinstance(node, WindowExpr):
            found = True
        return node

    transform_expr_up(expr, probe)
    return found


def hoist_windows(exprs: Sequence[Expr]) -> tuple[list[Expr], list[tuple[str, WindowExpr]]]:
    """Lift every `WindowExpr` out of `exprs`, leaving a `Col` reference in its place.

    A window function has no scalar IR — the engine computes it in a relational
    `Window` operator. To let a window *compose* like a scalar
    (``col("x") - col("x").shift(1)``), the relational layer pulls each `WindowExpr`
    out into its own synthetic column and rewrites the surrounding tree to read that
    column. This function does the expression half; the caller builds one `Window`
    node per returned pair and projects the rewritten expressions on top.

    Windows nested inside a window's argument (``col("x").shift(1).cum_sum()``) are
    hoisted first, so the returned pairs are already in dependency order: building
    them front-to-back, each `Window` node sees the columns the next one reads.

    `WindowExpr` is a leaf to `transform_expr_up` (it carries no `_EXPR_KIDS` entry),
    so this recurses into its argument explicitly.

    One `WindowExpr` *object* reached from several places in the tree is hoisted once
    and shared. A builder that reuses a window — ``when(w >= n).then(w)`` — would
    otherwise emit two identical `Window` nodes and compute it twice. Identity is the
    right key (and safe): the nodes stay alive in `exprs` for the whole call, and
    without `__eq__`/`__hash__` on `Expr` there is no structural key to use.

    Args:
        exprs: The scalar expressions to rewrite.

    Returns:
        The rewritten expressions, and the ``(column_name, window)`` pairs to
        materialize before evaluating them — empty when `exprs` held no window.
    """
    hoisted: list[tuple[str, WindowExpr]] = []
    seen: dict[int, str] = {}  # id(WindowExpr) -> the column it was hoisted into

    def lift(expr: Expr) -> Expr:
        def rule(node: Expr) -> Expr:
            if not isinstance(node, WindowExpr):
                return node
            shared = seen.get(id(node))
            if shared is not None:
                return Col(shared)
            # Recurse into the argument first: an inner window must be materialized
            # before the outer one can read it.
            inner = node if node.input is None else node.with_input(lift(node.input))
            name = f"{WINDOW_TEMP_PREFIX}{len(hoisted)}"
            seen[id(node)] = name
            hoisted.append((name, inner))
            return Col(name)

        return transform_expr_up(expr, rule)

    return [lift(e) for e in exprs], hoisted


def _case_kids(e: Case) -> tuple[Expr, ...]:
    """A `Case`'s children, flattened: cond/then per branch, then `otherwise`."""
    return (*(x for branch in e.branches for x in branch), e.otherwise)


def _case_rebuild(_e: Case, kids: tuple[Expr, ...]) -> Expr:
    pairs = [(kids[i], kids[i + 1]) for i in range(0, len(kids) - 1, 2)]
    return Case(pairs, kids[-1])


# Exact-type → (children, rebuild) dispatch for `transform_expr_up`. `_EXPR_KIDS` yields
# a node's direct sub-expressions; `_EXPR_REBUILD` reconstructs the node from rewritten
# children (called only when a child actually changed). Leaves (Col, Lit, AggExpr, …)
# are intentionally absent from both.
_EXPR_KIDS: dict[type, Callable[[Any], tuple[Expr, ...]]] = {
    Binary: lambda e: (e.left, e.right),
    Not: lambda e: (e.input,),
    Cast: lambda e: (e.input,),
    IsNull: lambda e: (e.input,),
    IsNotNull: lambda e: (e.input,),
    IsNan: lambda e: (e.input,),
    IsInf: lambda e: (e.input,),
    InList: lambda e: (e.input,),
    MathExpr: lambda e: (e.input,),
    DateFunc: lambda e: (e.input,),
    DateTrunc: lambda e: (e.input,),
    ListFunc: lambda e: (e.input,),
    ListGet: lambda e: (e.input,),
    ListContains: lambda e: (e.input,),
    ListSlice: lambda e: (e.input,),
    StructField: lambda e: (e.input,),
    ListJoin: lambda e: (e.input,),
    StrFunc: lambda e: (e.input,),
    ImageFunc: lambda e: (e.input,),
    Coalesce: lambda e: tuple(e.inputs),
    Greatest: lambda e: tuple(e.inputs),
    HashRows: lambda e: tuple(e.inputs),
    Least: lambda e: tuple(e.inputs),
    Array: lambda e: tuple(e.elements),
    NullIf: lambda e: (e.left, e.right),
    Math2Expr: lambda e: (e.left, e.right),
    Case: _case_kids,
}

_EXPR_REBUILD: dict[type, Callable[[Any, tuple[Expr, ...]], Expr]] = {
    Binary: lambda e, k: Binary(e.op, k[0], k[1]),
    Not: lambda _e, k: Not(k[0]),
    Cast: lambda e, k: Cast(k[0], e.dtype, try_cast=e.try_cast),
    IsNull: lambda _e, k: IsNull(k[0]),
    IsNotNull: lambda _e, k: IsNotNull(k[0]),
    IsNan: lambda _e, k: IsNan(k[0]),
    IsInf: lambda _e, k: IsInf(k[0]),
    InList: lambda e, k: InList(k[0], e.values),
    MathExpr: lambda e, k: MathExpr(e.fn, k[0]),
    DateFunc: lambda e, k: DateFunc(e.fn, k[0]),
    DateTrunc: lambda e, k: DateTrunc(k[0], e.unit),
    ListFunc: lambda e, k: ListFunc(e.fn, k[0]),
    ListGet: lambda e, k: ListGet(k[0], e.index),
    ListContains: lambda e, k: ListContains(k[0], e.value),
    ListSlice: lambda e, k: ListSlice(k[0], e.offset, e.length),
    StructField: lambda e, k: StructField(k[0], e.field),
    ListJoin: lambda e, k: ListJoin(k[0], e.separator),
    StrFunc: lambda e, k: StrFunc(
        e.fn,
        k[0],
        pattern=e.pattern,
        replacement=e.replacement,
        start=e.start,
        length=e.length,
    ),
    ImageFunc: lambda e, k: ImageFunc(e.fn, k[0], width=e.width, height=e.height),
    Coalesce: lambda _e, k: Coalesce(list(k)),
    Greatest: lambda _e, k: Greatest(list(k)),
    HashRows: lambda e, k: HashRows(list(k), e.seed),
    Least: lambda _e, k: Least(list(k)),
    Array: lambda _e, k: Array(list(k)),
    NullIf: lambda _e, k: NullIf(k[0], k[1]),
    Math2Expr: lambda e, k: Math2Expr(e.fn, k[0], k[1]),
    Case: _case_rebuild,
}


def map_node_expressions(node: LogicalPlan, rule: ExprRule) -> LogicalPlan:
    """Apply `rule` to every expression carried directly by `node`, returning a
    rebuilt node (or `node` unchanged for nodes with no expressions: Scan, Join,
    Distinct, Union, Limit, MapBatches).

    Like `transform_expr_up`, this shares structure: when `rule` leaves every one of the
    node's expressions untouched — the common case, since a rule matches a few nodes and
    passes over the rest — `node` is returned as-is rather than `dataclasses.replace`d
    into an equal copy. That keeps the node's memoized `to_ir`/`available_schema`, skips
    its `__post_init__` column re-validation, and preserves the `is`-identity that the
    optimizer's fixpoint detection and the estimator's memo keys rely on."""
    if isinstance(node, Filter):
        predicate = rule(node.predicate)
        if predicate is node.predicate:
            return node
        return dataclasses.replace(node, predicate=predicate)
    if isinstance(node, Project):
        items = _map_tuple(node.items, rule, _map_projection)
        return node if items is None else dataclasses.replace(node, items=items)
    if isinstance(node, Aggregate):
        group_keys = _map_tuple(node.group_keys, rule, _map_projection)
        aggregates = _map_tuple(node.aggregates, rule, _map_agg)
        if group_keys is None and aggregates is None:
            return node
        return dataclasses.replace(
            node,
            group_keys=node.group_keys if group_keys is None else group_keys,
            aggregates=node.aggregates if aggregates is None else aggregates,
        )
    if isinstance(node, Sort):
        keys = _map_tuple(node.keys, rule, _map_sort_key)
        return node if keys is None else dataclasses.replace(node, keys=keys)
    if isinstance(node, Window):
        partition_keys = _map_tuple(node.partition_keys, rule, _apply_rule)
        order_keys = _map_tuple(node.order_keys, rule, _map_sort_key)
        functions = _map_tuple(node.functions, rule, _map_window_fn)
        if partition_keys is None and order_keys is None and functions is None:
            return node
        return dataclasses.replace(
            node,
            partition_keys=node.partition_keys if partition_keys is None else partition_keys,
            order_keys=node.order_keys if order_keys is None else order_keys,
            functions=node.functions if functions is None else functions,
        )
    return node


def _map_tuple(
    items: tuple[_T, ...], rule: ExprRule, fn: Callable[[_T, ExprRule], _T]
) -> tuple[_T, ...] | None:
    """Map `fn` over `items`, or return `None` when every element kept its identity."""
    mapped = tuple(fn(it, rule) for it in items)
    if all(a is b for a, b in zip(mapped, items, strict=True)):
        return None
    return mapped


def _apply_rule(expr: Expr, rule: ExprRule) -> Expr:
    return rule(expr)


def _map_projection(item: Projection, rule: ExprRule) -> Projection:
    expr = rule(item.expr)
    return item if expr is item.expr else Projection(item.alias, expr)


def _map_sort_key(key: SortKeySpec, rule: ExprRule) -> SortKeySpec:
    expr = rule(key.expr)
    return key if expr is key.expr else dataclasses.replace(key, expr=expr)


def _map_agg(spec: AggregateSpec, rule: ExprRule) -> AggregateSpec:
    # AggExpr is not a dataclass (custom __slots__ class), so rebuild it directly.
    if spec.agg.input is None:
        return spec
    # Carry the second input (arg_min/arg_max ordering key) through the rewrite too.
    input1 = rule(spec.agg.input)
    input2 = rule(spec.agg.input2) if spec.agg.input2 is not None else None
    if input1 is spec.agg.input and input2 is spec.agg.input2:
        return spec
    rebuilt = AggExpr(spec.agg.func, input1, param=spec.agg.param, input2=input2)
    return dataclasses.replace(spec, agg=rebuilt)


def _map_window_fn(fn: WindowFuncSpec, rule: ExprRule) -> WindowFuncSpec:
    if fn.input is None:
        return fn
    inp = rule(fn.input)
    return fn if inp is fn.input else dataclasses.replace(fn, input=inp)
