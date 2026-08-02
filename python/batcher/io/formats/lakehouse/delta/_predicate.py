"""Rendering an expression for delta-rs: as partition filters, or as SQL.

delta-rs takes a predicate in two forms, and the difference decides how much a write costs.

* **Partition filters** — ``[(column, op, value)]`` over the table's *partition* columns.
  delta-rs can act on these from the log alone: a partition-scoped overwrite removes the
  matching partitions' add-actions and adds the new ones, moving no data at all. This is
  the fast path, and it is the one a backfill actually wants ("replace 2024-01-05").
* **SQL text** — an arbitrary predicate. delta-rs has to open the files that match it and
  rewrite them, which is bounded by the predicate but is real work.

So `to_partition_filters` is tried first and `to_sql` is the fallback. Both are total in
the same way the rest of the pushdown layer is: they return `None` for anything they cannot
express exactly, and the caller degrades to a slower-but-correct path. Approximating a
predicate here would silently overwrite the wrong rows.

The SQL renderer takes a `column` naming function because the two callers name their
columns differently: a MERGE clause has two sides to disambiguate (``source.x`` /
``target.x``), while a ``replace_where`` predicate has one (bare ``x``). Same renderer,
same guarantees — the alternative was a second copy of it, and a merge that renders a
condition differently from the way a write does is a bug waiting to be written.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from batcher.plan.ir_tags import COMPARISON_FLIP

__all__ = ["to_partition_filters", "to_sql"]

_BINARY_SQL = {
    "eq": "=",
    "ne": "<>",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "and": "AND",
    "or": "OR",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}
# The operators delta-rs accepts in a partition filter tuple.
_PARTITION_OPS = {"eq": "=", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def to_partition_filters(
    ir: dict[str, Any] | None, partition_columns: list[str]
) -> list[tuple[str, str, str]] | None:
    """`ir` as delta-rs partition filters, or None if it is not purely partition-scoped.

    Only an ``AND`` of ``partition_column OP literal`` qualifies. Anything else — an `OR`,
    a data column, a computed term — returns None, because delta-rs would then have to
    inspect the data and the commit could no longer be metadata-only.

    Args:
        ir: The predicate IR.
        partition_columns: The table's partition columns.

    Returns:
        The partition filters, or None if the predicate is not expressible as such.
    """
    if ir is None or not partition_columns:
        return None
    partitions = set(partition_columns)
    out: list[tuple[str, str, str]] = []

    def walk(node: dict[str, Any]) -> bool:
        if node.get("e") != "binary":
            return False
        op = node["op"]
        if op == "and":
            return walk(node["left"]) and walk(node["right"])
        if op not in _PARTITION_OPS:
            return False
        left, right = node["left"], node["right"]
        if left.get("e") == "col" and right.get("e") == "lit":
            column, value, operator = left["name"], right, op
        elif left.get("e") == "lit" and right.get("e") == "col":
            column, value, operator = right["name"], left, COMPARISON_FLIP[op]
        else:
            return False
        if column not in partitions:
            return False
        # A partition value is stored in the path, so delta-rs compares it as text.
        out.append((column, _PARTITION_OPS[operator], str(_unwrap(value))))
        return True

    return out if walk(ir) else None


def to_sql(
    ir: dict[str, Any] | None, column: Callable[[str], str] = lambda name: name
) -> str | None:
    """`ir` as a Delta SQL predicate, or None if it contains something unrenderable.

    `column` names a column reference — the identity for a plain predicate, an aliasing
    function for a MERGE clause's two sides.

    Returning None rather than approximating is the whole contract: a predicate that
    silently rendered to something *close* would overwrite or match the wrong rows, and the
    caller can always fall back to a path that does not need SQL.

    Args:
        ir: The predicate IR.
        column: Maps a column name to its SQL spelling.

    Returns:
        The SQL text, or None if the expression cannot be expressed exactly.
    """
    if ir is None:
        return None
    try:
        return _render(ir, column)
    except _Unrenderable:
        return None


class _Unrenderable(Exception):
    """An expression node delta-rs SQL cannot express."""


def _render(node: dict[str, Any], column: Callable[[str], str]) -> str:
    kind = node.get("e")
    if kind == "col":
        return column(node["name"])
    if kind == "lit":
        return _literal(node["value"])
    if kind == "binary":
        op = node.get("op")
        if op not in _BINARY_SQL:
            raise _Unrenderable(str(op))
        left = _render(node["left"], column)
        right = _render(node["right"], column)
        return f"({left} {_BINARY_SQL[op]} {right})"
    if kind == "is_null":
        return f"({_render(node['input'], column)} IS NULL)"
    if kind == "is_not_null":
        return f"({_render(node['input'], column)} IS NOT NULL)"
    if kind == "not":
        return f"(NOT {_render(node['input'], column)})"
    raise _Unrenderable(str(kind))


def _unwrap(node: dict[str, Any]) -> Any:
    """The Python value of a literal IR node."""
    ((_kind, value),) = node["value"].items()
    return value


def _literal(value: dict[str, Any]) -> str:
    """A literal IR value as a SQL literal."""
    ((kind, raw),) = value.items()
    if raw is None:
        return "NULL"
    if kind == "bool":
        return "true" if raw else "false"
    if kind in ("int", "float"):
        return repr(raw)
    if kind == "str":
        return "'" + str(raw).replace("'", "''") + "'"
    if kind == "date":
        import datetime as dt

        return f"'{(dt.date(1970, 1, 1) + dt.timedelta(days=int(raw))).isoformat()}'"
    if kind == "timestamp":
        import datetime as dt

        moment = dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=int(raw))
        return f"'{moment.isoformat(sep=' ')}'"
    raise _Unrenderable(kind)
