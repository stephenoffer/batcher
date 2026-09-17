"""Transforms over sort and window keys, whose null placement differs per engine.

Batcher sorts nulls last unless told otherwise. PySpark sorts an ascending key nulls first and a
descending one nulls last, Polars places nulls first unless `nulls_last=True`, and Daft's
`nulls_first` follows `desc`. A sort rewrite therefore always passes `descending=` and
`nulls_first=` explicitly, so the migrated query orders rows exactly as the source did. A window
key cannot carry a null placement in Batcher, so an ascending window key keeps a marker.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.optional import require
from batcher.migrate.semantics.base import (
    Context,
    bool_list,
    bool_node,
    bools,
    callee_name,
    chain,
    expression_surfaces,
    flatten,
    is_none,
    keyword,
    list_node,
    transform,
    tuple_node,
)
from batcher.migrate.templates import Bound, Declined, literal, parens, simple_call

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__: list[str] = []

# PySpark's ordering functions and methods, as (descending, nulls_first).
_SPARK_ORDER = {
    "asc": (False, True),
    "asc_nulls_first": (False, True),
    "asc_nulls_last": (False, False),
    "desc": (True, False),
    "desc_nulls_first": (True, True),
    "desc_nulls_last": (True, False),
}
# Aggregates whose value over a whole partition does not depend on row order.
_AGGREGATES = frozenset({"sum", "mean", "min", "max", "count", "len", "median", "std", "var"})
_WINDOW_BOUNDS = {"unboundedPreceding": "None", "unboundedFollowing": "None", "currentRow": "0"}


def _sort_args(keys: list[tuple[Any, bool, bool]]) -> list[Any]:
    args = [cst.Arg(node) for node, _, _ in keys]
    for name, flags in (
        ("descending", [k[1] for k in keys]),
        ("nulls_first", [k[2] for k in keys]),
    ):
        value = bool_node(flags[0]) if len(set(flags)) == 1 else bool_list(flags)
        args.append(keyword(name, value))
    return args


def _spark_key(ctx: Context, key: Bound) -> tuple[Any, bool, bool, bool] | None:
    """`(node, descending, nulls_first, plain)`; a plain key follows `ascending=`."""
    original = key.original
    if isinstance(key.node, cst.SimpleString):
        return key.node, False, True, True
    name = callee_name(original.func) if isinstance(original, cst.Call) else None
    if name in _SPARK_ORDER:
        func = original.func
        if isinstance(func, cst.Attribute) and ctx.receiver(func.value) == "Column":
            inner = ctx.rewritten(func.value)
        elif len(original.args) == 1 and (
            not isinstance(func, cst.Attribute) or ctx.receiver(func.value) == "functions"
        ):
            arg = original.args[0].value
            inner = arg if isinstance(arg, cst.SimpleString) else ctx.rewritten(arg)
        else:
            return None
        ctx.consume(original)
        desc, nulls = _SPARK_ORDER[name]
        return inner, desc, nulls, False
    if ctx.receiver(original) in expression_surfaces(ctx):
        return key.node, False, True, True
    return None


@transform
def spark_sort(ctx: Context, cols: list[Bound], kwargs: dict[str, Bound]) -> list[Any] | None:
    """`orderBy(*cols, ascending=)` as `sort(*keys, descending=[...], nulls_first=[...])`."""
    if set(kwargs) - {"ascending"}:
        return None
    keys = [_spark_key(ctx, c) for c in flatten(cols)]
    if not keys or any(k is None for k in keys):
        return None
    ascending = kwargs.get("ascending")
    flags = [True] * len(keys) if ascending is None else bools(ascending, len(keys))
    if flags is None:
        return None
    resolved = []
    for (node, desc, nulls, plain), asc in zip(keys, flags, strict=True):  # type: ignore[misc]
        resolved.append((node, not asc, asc) if plain else (node, desc, nulls))
    return _sort_args(resolved)


def _typed_key(ctx: Context, key: Bound) -> bool:
    if isinstance(key.node, cst.SimpleString):
        return True
    return ctx.receiver(key.original) in expression_surfaces(ctx)


@transform
def polars_sort(
    ctx: Context, by: Bound, more: list[Bound], descending: Bound, nulls_last: Bound
) -> list[Any] | None:
    """Polars `sort` places nulls first unless `nulls_last=True`."""
    keys = flatten([by, *more])
    flags, last = bools(descending, len(keys)), bools(nulls_last, len(keys))
    if not all(_typed_key(ctx, k) for k in keys) or flags is None or last is None:
        return None
    return _sort_args([(k.node, d, not n) for k, d, n in zip(keys, flags, last, strict=True)])


@transform
def daft_sort(ctx: Context, by: Bound, desc: Bound, nulls_first: Bound) -> list[Any] | None:
    """Daft's `nulls_first` defaults to `desc`; Batcher's to False."""
    keys = flatten([by])
    flags = bools(desc, len(keys))
    if not all(_typed_key(ctx, k) for k in keys) or flags is None:
        return None
    firsts = flags if is_none(nulls_first.node) else bools(nulls_first, len(keys))
    if firsts is None:
        return None
    return _sort_args([(k.node, d, f) for k, d, f in zip(keys, flags, firsts, strict=True)])


def _window_bound(ctx: Context, node: Any) -> Any | None:
    if isinstance(node, cst.Attribute) and ctx.receiver(node.value) == "Window":
        found = _WINDOW_BOUNDS.get(node.attr.value)
        return None if found is None else cst.parse_expression(found)
    try:
        value = literal(node)
    except Declined:
        return None
    return node if isinstance(value, int) and not isinstance(value, bool) else None


def _window_step(ctx: Context, name: str, args: list[Any], spec: dict[str, Any]) -> bool:
    if name == "partitionBy":
        for arg in args:
            typed = isinstance(arg, cst.SimpleString) or ctx.receiver(arg) == "Column"
            if not typed:
                return False
            spec["partition_by"].append(ctx.rewritten(arg))
        return True
    if name == "orderBy":
        for arg in args:
            key = _spark_key(ctx, Bound(ctx.rewritten(arg), arg))
            if key is None:
                return False
            spec["order_by"].append(tuple_node([key[0], bool_node(True)]) if key[1] else key[0])
            if not key[1]:
                ctx.note("Spark orders nulls first in an ascending window key; Batcher last")
        return True
    if name == "rowsBetween" and len(args) == 2:
        bounds = [_window_bound(ctx, a) for a in args]
        spec["frame"] = None if None in bounds else tuple_node(bounds)
        return spec["frame"] is not None
    return False


@transform
def spark_window(ctx: Context, window: Bound) -> list[Any] | None:
    """A `Window.partitionBy(..).orderBy(..).rowsBetween(..)` spec as `over(...)` keywords."""
    spec_node = window.original
    if isinstance(spec_node, cst.Name):
        spec_node = ctx.definition(spec_node.value)
    steps = chain(spec_node, root="Window", ctx=ctx)
    if steps is None:
        return None
    spec: dict[str, Any] = {"partition_by": [], "order_by": [], "frame": None}
    for node, name in steps:
        if any(a.keyword is not None or a.star for a in node.args):
            return None
        if not _window_step(ctx, name, [a.value for a in node.args], spec):
            return None
    for node, _ in steps:
        ctx.consume(node)
    out = [keyword(k, list_node(spec[k])) for k in ("partition_by", "order_by") if spec[k]]
    if spec["frame"] is not None:
        out.append(keyword("frame", spec["frame"]))
    return out


@transform
def polars_over(
    ctx: Context,
    base: Bound,
    partition: Bound,
    more: list[Bound],
    order: Bound,
    desc: Bound,
    nulls_last: Bound,
    strategy: Bound,
) -> Any | None:
    """Polars `over(...)` as Batcher's keyword form.

    An `order_by` in Polars orders rows only for an order-dependent expression; an aggregate
    such as `sum` still sees the whole partition. Batcher's `over(order_by=)` is SQL's running
    frame, so an ordered aggregate is written with `frame=(None, None)`, which reads the whole
    partition again. Any other ordered expression declines.
    """
    if literal(strategy.node) != "group_to_rows" or literal(nulls_last.node) is not False:
        return None
    keys = [k for k in flatten([partition, *more]) if not is_none(k.node)]
    if not all(_typed_key(ctx, k) for k in keys):
        return None
    args = [keyword("partition_by", list_node([k.node for k in keys]))]
    if not is_none(order.node):
        callee = callee_name(base.original.func) if isinstance(base.original, cst.Call) else None
        if literal(desc.node) not in (True, False) or callee not in _AGGREGATES:
            return None
        whole = tuple_node([cst.Name("None"), cst.Name("None")])
        args.append(keyword("order_by", list_node([k.node for k in flatten([order])])))
        args.append(keyword("frame", whole))
    return simple_call(cst.Attribute(value=parens(base.node), attr=cst.Name("over")), args)
