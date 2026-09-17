"""Transforms over relational spellings: joins, writes, sessions, constructors, aggregates.

Join `how` values are spelled differently by every engine (`left_outer`, `leftsemi`, `outer`),
PySpark's save mode defaults to `errorifexists` where Batcher's is `overwrite`, and a Spark
session or reader/writer is a builder chain rather than one call. Each transform here maps one
such spelling onto Batcher's, for literal arguments, and declines anything else.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.optional import require
from batcher.migrate.semantics.base import (
    Context,
    call,
    callee_name,
    chain,
    flatten,
    is_none,
    keyword,
    list_node,
    python,
    string,
    transform,
)
from batcher.migrate.semantics.columns import column
from batcher.migrate.templates import Bound, Declined, literal, parens, simple_call

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__: list[str] = []

_JOIN_HOW = {
    **{h: h for h in ("inner", "cross", "left", "right", "full", "semi", "anti")},
    **dict.fromkeys(("leftouter", "left_outer"), "left"),
    **dict.fromkeys(("rightouter", "right_outer"), "right"),
    **dict.fromkeys(("outer", "fullouter", "full_outer"), "full"),
    **dict.fromkeys(("leftsemi", "left_semi"), "semi"),
    **dict.fromkeys(("leftanti", "left_anti"), "anti"),
}
_SPARK_MODES = {"error": "error", "errorifexists": "error", "overwrite": "overwrite"}
_SPARK_MODES |= {"append": "append", "ignore": "ignore"}
_BUILDER_STEPS = frozenset({"appName", "master", "getOrCreate", "enableHiveSupport", "config"})
_RAY_FORMATS = {"default": "numpy", "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow"}


@transform
def join_how(_ctx: Context, how: Bound) -> Any | None:
    """A join type spelled any engine's way, as Batcher's `how`; `None` is an inner join."""
    value = "inner" if is_none(how.node) else literal(how.node)
    normalized = _JOIN_HOW.get(value.lower()) if isinstance(value, str) else None
    return string(normalized) if normalized else None


@transform
def join_keys(_ctx: Context, on: Bound) -> Any | None:
    """Join keys that are column names; a `Column` join condition declines."""
    try:
        value = literal(on.node)
    except Declined:
        return None
    names = isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value)
    return on.node if isinstance(value, str) or names else None


@transform
def polars_how(ctx: Context, how: Bound, coalesce: Bound) -> Any | None:
    """A Polars full join keeps both key columns unless `coalesce=True`; Batcher coalesces."""
    found = join_how(ctx, how)
    if found is not None and literal(found) == "full" and literal(coalesce.node) is not True:
        return None
    return found


@transform
def spark_mode(_ctx: Context, mode: Bound) -> Any | None:
    """Spark's save mode, with its `errorifexists` default made explicit."""
    value = "error" if is_none(mode.node) else literal(mode.node)
    found = _SPARK_MODES.get(value.lower()) if isinstance(value, str) else None
    return string(found) if found else None


@transform
def spark_write(
    ctx: Context, writer: Bound, fmt: Bound, path: Bound, mode: Bound, partition: Bound
) -> Any | None:
    """`df.write.mode(m).partitionBy(..).parquet(p)` as `df.write.parquet(p, mode=...)`."""
    steps = chain(writer.original, root="DataFrameWriter", ctx=ctx)
    root = writer.original
    while isinstance(root, cst.Call):
        root = root.func.value
    if steps is None or not (isinstance(root, cst.Attribute) and root.attr.value == "write"):
        return None
    modes: list[Bound] = []
    partitions = [] if is_none(partition.node) else [partition]
    for node, name in steps:
        args = [Bound(a.value, a.value) for a in node.args]
        if name == "mode" and len(args) == 1:
            modes.append(args[0])
        elif name == "partitionBy":
            partitions.extend(args)
        else:
            return None
    if not is_none(mode.node):
        modes.append(mode)  # the writer method's own `mode=` wins over the chain's
    chosen = spark_mode(ctx, modes[-1] if modes else Bound(cst.Name("None")))
    cols = flatten(partitions)
    if chosen is None or not all(isinstance(c.node, cst.SimpleString) for c in cols):
        return None
    for node, _ in steps:
        ctx.consume(node)
    ctx.consume(root)
    args = [cst.Arg(path.node), keyword("mode", chosen)]
    if cols:
        args.append(keyword("partition_by", list_node([c.node for c in cols])))
    base = cst.Attribute(value=ctx.rewritten(root.value), attr=cst.Name("write"))
    return simple_call(cst.Attribute(value=base, attr=cst.Name(literal(fmt.node))), args)


def _truthy(node: Any) -> bool | None:
    value = literal(node)
    if isinstance(value, bool):
        return value
    return {"true": True, "false": False}.get(value.lower()) if isinstance(value, str) else None


@transform
def spark_read_csv(
    ctx: Context, reader: Bound, path: Bound, header: Bound, infer: Bound
) -> Any | None:
    """`spark.read.option("header", True).csv(p)` as `bt.read.csv(p)` when it reads the same.

    Spark reads a CSV headerless and untyped unless told otherwise; Batcher reads the header and
    infers types. Only a read that asks Spark for both is the same read, so anything else
    declines rather than silently changing the column names or types.
    """
    options: dict[str, Any] = {}
    steps = chain(reader.original, root="DataFrameReader", ctx=ctx)
    for node, name in steps or []:
        if name != "option" or len(node.args) != 2 or node.args[0].keyword is not None:
            return None
        options[str(literal(node.args[0].value))] = node.args[1].value
    for key, bound in (("header", header), ("inferSchema", infer)):
        if not is_none(bound.node):
            options[key] = bound.node
    if steps is None or set(options) != {"header", "inferSchema"}:
        return None
    if not all(_truthy(v) is True for v in options.values()):
        return None
    for node, _ in steps:
        ctx.consume(node)
    return call(f"{ctx.bt}.read.csv", [cst.Arg(path.node)])


@transform
def spark_session(ctx: Context, builder: Bound) -> Any | None:
    """`SparkSession.builder...getOrCreate()` as `bt.Session()`; dropped config is noted."""
    node = builder.original
    dropped = []
    while isinstance(node, cst.Call):
        name = callee_name(node.func)
        if name not in _BUILDER_STEPS:
            return None
        if name == "config":
            dropped.extend(cst.Module([]).code_for_node(a.value) for a in node.args[:1])
        if node is not builder.original:  # the call being rewritten keeps its own site
            ctx.consume(node)
        node = node.func.value
    if dropped:
        ctx.note(f"Spark session config not carried over: {', '.join(dropped)}")
    return call(f"{ctx.bt}.Session", [])


@transform
def spark_rows(ctx: Context, data: Bound, schema: Bound) -> Any | None:
    """`createDataFrame(rows, ["a", "b"])` over literal rows as `bt.from_pylist`."""
    try:
        rows, names = literal(data.node), literal(schema.node)
    except Declined:
        return None
    if not isinstance(rows, list):
        return None
    if names is None and all(isinstance(r, dict) for r in rows):
        return call(f"{ctx.bt}.from_pylist", [cst.Arg(data.node)])
    if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
        return None
    if not all(isinstance(r, (tuple, list)) and len(r) == len(names) for r in rows):
        return None
    dicts = cst.parse_expression(python([dict(zip(names, r, strict=True)) for r in rows]))
    return call(f"{ctx.bt}.from_pylist", [cst.Arg(dicts)])


@transform
def spark_count(ctx: Context, col: Bound) -> Any | None:
    """`F.count("*")` counts rows; `F.count(c)` counts a column's non-null values."""
    if isinstance(col.node, cst.SimpleString) and literal(col.node) == "*":
        return call(f"{ctx.bt}.count", [])
    found = column(ctx, col)
    if found is None:
        return None
    return simple_call(cst.Attribute(value=parens(found), attr=cst.Name("count")), [])


@transform
def polars_contains(
    _ctx: Context, namespace: Bound, pattern: Bound, is_literal: Bound
) -> Any | None:
    """Polars `str.contains` is a regex unless `literal=True`."""
    flag = literal(is_literal.node)
    if not isinstance(flag, bool):
        return None
    method = "contains" if flag else "regexp_matches"
    func = cst.Attribute(value=parens(namespace.node), attr=cst.Name(method))
    return simple_call(func, [cst.Arg(pattern.node)])


@transform
def ray_batch_format(_ctx: Context, fmt: Bound) -> Any | None:
    """Ray's `batch_format="default"` is numpy; Batcher's default is pyarrow."""
    value = literal(fmt.node)
    return string(_RAY_FORMATS[value]) if value in _RAY_FORMATS else None


@transform
def ray_agg(ctx: Context, base: Bound, fn: Bound, on: Bound, ignore_nulls: Bound) -> Any | None:
    """`groupby(k).sum("x")` as `agg(**{"sum(x)": bt.col("x").sum()})`, keeping Ray's name."""
    name, source = literal(fn.node), literal(on.node)
    if not isinstance(source, str) or literal(ignore_nulls.node) is not True:
        return None
    column_expr = call(f"{ctx.bt}.col", [cst.Arg(on.node)])
    expr = simple_call(cst.Attribute(value=column_expr, attr=cst.Name(name)), [])
    aggregates = cst.Dict([cst.DictElement(string(f"{name}({source})"), expr)])
    func = cst.Attribute(value=parens(base.node), attr=cst.Name("agg"))
    return simple_call(func, [cst.Arg(aggregates, star="**")])
