"""Apply an expression rewrite to every expression a *plan node* carries.

The bridge between the two levels: a pass is just
`transform_up(plan, lambda n: map_node_expressions(n, rule))`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TypeVar

from batcher.plan.expr_ir import AggExpr, Expr
from batcher.plan.expr_rewrite.traverse import ExprRule
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

__all__ = ["map_node_expressions"]

# One element of a plan node's expression-carrying tuple (a Projection, SortKeySpec,
# AggregateSpec, WindowFuncSpec, or a bare Expr) — see `_map_tuple`.
_T = TypeVar("_T")


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
    """Map `fn` over `items`, or return `None` when every element kept its identity.

    Written as one pass that allocates nothing until something actually changes. The
    unchanged answer is by far the common one — a rule matches a handful of nodes and
    passes over the rest — and this runs for every rule against every expression-carrying
    node of the plan, so building a full tuple and then walking a `zip` inside an `all`
    to discover "nothing moved" was the bulk of what the fixpoint spent here.
    """
    mapped: list[_T] | None = None
    for i, item in enumerate(items):
        new = fn(item, rule)
        if mapped is None:
            if new is item:
                continue
            mapped = list(items[:i])  # first change: catch up on the untouched prefix
        mapped.append(new)
    return None if mapped is None else tuple(mapped)


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
