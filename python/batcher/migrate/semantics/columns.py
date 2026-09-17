"""Transforms over column references, argument checks, positions and date patterns.

A PySpark or Polars function takes a column *name* as a string where Batcher's method form needs
an expression (`F.upper("n")` is `bt.col("n").str.upper()`), and a verb that takes names in
Batcher may be handed a `col("k")` in the source. Positions move from 0-based to 1-based, and Java
`DateTimeFormatter` patterns become `strftime`, for literals only.
"""

from __future__ import annotations

import re
from typing import Any

from batcher._internal.optional import require
from batcher.migrate.semantics.base import (
    Context,
    bool_node,
    call,
    callee_name,
    expression_surfaces,
    flatten,
    is_none,
    keyword,
    string,
    transform,
)
from batcher.migrate.templates import Bound, Declined, literal

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["java_to_strftime"]

# Java DateTimeFormatter letter runs with an exact strftime equivalent. Anything else declines.
_JAVA = {
    "yyyy": "%Y",
    "yy": "%y",
    "MMMM": "%B",
    "MMM": "%b",
    "MM": "%m",
    "dd": "%d",
    "HH": "%H",
    "hh": "%I",
    "mm": "%M",
    "ss": "%S",
    "a": "%p",
    "EEEE": "%A",
    "EEE": "%a",
}
_JAVA_TOKEN = re.compile(r"'[^']*'|([A-Za-z])\1*|[^A-Za-z']+")


def java_to_strftime(pattern: str) -> str | None:
    """Translate a Java `DateTimeFormatter` pattern to `strftime`, or `None` if inexact.

    Args:
        pattern: A Java pattern such as `yyyy-MM-dd HH:mm:ss`.

    Returns:
        The `strftime` pattern, or `None` when a letter run has no exact equivalent.

    Examples:
        .. doctest::

            >>> from batcher.migrate.semantics import java_to_strftime
            >>> java_to_strftime("yyyy-MM-dd'T'HH:mm")
            '%Y-%m-%dT%H:%M'
            >>> print(java_to_strftime("yyyy-MM-dd SSS"))
            None
    """
    out = []
    for match in _JAVA_TOKEN.finditer(pattern):
        text = match.group(0)
        if text.startswith("'"):
            out.append(text[1:-1].replace("%", "%%") or "'")
        elif text[0].isalpha():
            if text not in _JAVA:
                return None
            out.append(_JAVA[text])
        else:
            out.append(text.replace("%", "%%"))
    return "".join(out)


@transform
def expression(ctx: Context, value: Bound) -> Any | None:
    """A value the inference proves is a column expression; a SQL string declines."""
    receiver = ctx.receiver(value.original) if value.original is not None else None
    return value.node if receiver in expression_surfaces(ctx) else None


@transform
def column(ctx: Context, value: Bound) -> Any | None:
    """A column argument as an expression: a string literal names a column."""
    if isinstance(value.node, cst.SimpleString):
        return call(f"{ctx.bt}.col", [cst.Arg(value.node)])
    return expression(ctx, value)


def name_of(ctx: Context, value: Bound) -> Any | None:
    """A column name as a string literal node, from `"k"` or a `col("k")` the rewrite absorbs."""
    if isinstance(value.node, cst.SimpleString):
        return value.node
    node = value.original
    if (
        isinstance(node, cst.Call)
        and callee_name(node.func) in ("col", "column")
        and len(node.args) == 1
        and isinstance(node.args[0].value, cst.SimpleString)
        and ctx.receiver(node) in expression_surfaces(ctx)
    ):
        ctx.consume(node)
        return node.args[0].value
    return None


@transform
def column_names(ctx: Context, values: list[Bound] | Bound) -> list[Any] | None:
    """Column names for a verb that takes names: `"k"`, `["a", "b"]` or `col("k")`."""
    out = []
    for value in flatten(values if isinstance(values, list) else [values]):
        name = name_of(ctx, value)
        if name is None:
            return None
        out.append(cst.Arg(name))
    return out or None


@transform
def spark_sql_exprs(_ctx: Context, values: list[Bound]) -> list[Any] | None:
    """Spark SQL expression strings (`selectExpr("size(xs) AS n")`) as `bt.sql_expr` calls.

    The dialect is passed explicitly: Spark's function vocabulary (`size`, `nvl`) is not the
    session default's, so the same text parsed without it fails or means something else.
    """
    out = []
    for value in flatten(values):
        if not isinstance(value.node, (cst.SimpleString, cst.ConcatenatedString)):
            return None
        out.append(
            cst.Arg(call("bt.sql_expr", [cst.Arg(value.node), keyword("dialect", string("spark"))]))
        )
    return out or None


@transform
def column_name(ctx: Context, value: Bound) -> Any | None:
    """One column name, as `column_names` accepts it."""
    return name_of(ctx, value)


@transform
def named(ctx: Context, values: list[Bound]) -> list[Any] | None:
    """Positional expressions, named as Polars names them.

    Batcher infers a positional output's name the way Polars does (an alias, else the
    leftmost column), so any expression carries over; a value that is not an expression
    (a bare Python scalar, which Batcher refuses positionally) declines.
    """
    out = []
    for value in values:
        if expression(ctx, value) is None:
            return None
        out.append(cst.Arg(value.node))
    return out


@transform
def selectable(ctx: Context, values: list[Bound]) -> list[Any] | None:
    """`select` arguments: column-name strings, or expressions carrying a name."""
    out = []
    for value in values:
        found = [cst.Arg(value.node)] if isinstance(value.node, cst.SimpleString) else None
        found = found or named(ctx, [value])
        if found is None:
            return None
        out.extend(found)
    return out


@transform
def mapping(_ctx: Context, value: Bound) -> Any | None:
    """A literal dict; a positional list of new names declines."""
    return value.node if isinstance(value.node, cst.Dict) else None


@transform
def negate(_ctx: Context, value: Bound) -> Any | None:
    """A literal boolean, negated."""
    flag = literal(value.node)
    return bool_node(not flag) if isinstance(flag, bool) else None


@transform
def expect(_ctx: Context, value: Bound, wanted: Bound) -> list[Any] | None:
    """Accept the call only when an argument holds the one value the rewrite assumes."""
    try:
        return [] if literal(value.node) == literal(wanted.node) else None
    except Declined:
        return None


@transform
def keep_note(ctx: Context, value: Bound) -> Any:
    """Rewrite, and keep the row's note on the call as a marker."""
    ctx.note("")
    return value.node


@transform
def frame(ctx: Context, data: Bound) -> Any | None:
    """A literal dict (columns) or list of dicts (rows) as the matching constructor."""
    if isinstance(data.node, cst.Dict):
        return call(f"{ctx.bt}.from_pydict", [cst.Arg(data.node)])
    elements = data.node.elements if isinstance(data.node, cst.List) else None
    if elements is not None and all(isinstance(e.value, cst.Dict) for e in elements):
        return call(f"{ctx.bt}.from_pylist", [cst.Arg(data.node)])
    return None


def _non_negative(node: Any) -> int | None:
    value = literal(node)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@transform
def plus_one(_ctx: Context, offset: Bound) -> Any | None:
    """A non-negative 0-based literal offset as the 1-based one."""
    value = _non_negative(offset.node)
    return None if value is None else cst.Integer(str(value + 1))


@transform
def spark_position(_ctx: Context, pos: Bound) -> Any | None:
    """Spark's `substring` start: 0 means 1, and a positive literal carries over."""
    value = _non_negative(pos.node)
    return None if value is None else cst.Integer(str(max(value, 1)))


@transform
def java_datetime(_ctx: Context, fmt: Bound) -> Any | None:
    """A literal Java `DateTimeFormatter` pattern as `strftime`."""
    if is_none(fmt.node):
        return None
    value = literal(fmt.node)
    found = java_to_strftime(value) if isinstance(value, str) else None
    return string(found) if found is not None else None
