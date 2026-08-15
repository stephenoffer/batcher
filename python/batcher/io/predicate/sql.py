"""IR to a SQL ``WHERE`` fragment, for the warehouse and JDBC-style connectors.

The one translator whose output is *text*, which is why the caps and the `LIKE` guard
live here: a fragment has to survive a server's parser, and the wildcard-quoting rules
differ enough between them that the safe move is to decline. See the package docstring
for the widening/exactness contract every translator shares.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable
from typing import Any

from batcher.io.predicate._literals import _col_and_literal, _literal
from batcher.io.predicate._shapes import _combine, _const_bool, _in_list, _str_predicate
from batcher.plan.ir_tags import COMPARISON_FLIP, COMPARISON_OPS

__all__ = ["to_sql_where"]

_SQL_OP = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}

#: Wildcards (and the escape characters used to quote them) that make a `LIKE` pattern
#: ambiguous. Spelling them portably needs an `ESCAPE` clause that BigQuery does not
#: accept and whose backslash form ClickHouse and MySQL re-interpret inside the string
#: literal itself. A pattern containing one simply declines: the engine's `Filter` still
#: applies it, and the prefix filters worth pushing (``starts_with("US")``,
#: ``starts_with("2024-01")``) contain none of these.
_LIKE_UNSAFE = frozenset({"%", "_", "\\", "!"})

#: Longest `IN` list folded into SQL text. Oracle rejects a list past 1,000 expressions
#: outright and every server's parser slows on a long one, so a longer list declines and
#: the engine filters — which is what it did for every list before this pushed at all.
_SQL_IN_MAX = 1000


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if value is None:
        return "NULL"
    # Temporal literals MUST be emitted as typed, quoted SQL literals. A bare
    # ``str(date)`` renders ``2021-01-15``, which the server parses as the integer
    # arithmetic ``2021 - 1 - 15`` (→ 2005), and ``str(datetime)`` renders an
    # unquoted ``2021-01-15 00:00:00`` that is a syntax error. ANSI ``DATE '…'`` /
    # ``TIMESTAMP '…'`` / ``TIME '…'`` literals are accepted across the warehouses
    # these connectors target. (datetime is a subclass of date — check it first.)
    if isinstance(value, _dt.datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, _dt.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, _dt.time):
        return f"TIME '{value.isoformat()}'"
    return str(value)


def _identity(name: str) -> str:
    """The default identifier rendering: verbatim, as every caller had before quoting."""
    return name


def _sql_like(ir: dict[str, Any], quote: Callable[[str], str] = _identity) -> str | None:
    """A `LIKE` fragment for a pushable string predicate, or None.

    `LIKE`'s case sensitivity is a property of the column's *collation*, not of the
    operator, so this is a widening translation rather than an exact one: under MySQL's
    default case-insensitive collation ``name LIKE 'ab%'`` also matches ``AB…``. Widening
    is free here because the engine re-checks every row, which is why the caller only
    consults this outside exact mode.
    """
    parsed = _str_predicate(ir)
    if parsed is None:
        return None
    column, fn, pattern = parsed
    if any(char in pattern for char in _LIKE_UNSAFE):
        return None
    wrapped = {
        "starts_with": f"{pattern}%",
        "ends_with": f"%{pattern}",
        "contains": f"%{pattern}%",
    }[fn]
    return f"{quote(column)} LIKE {_sql_literal(wrapped)}"


def to_sql_where(
    ir: dict[str, Any], *, exact: bool = False, quote: Callable[[str], str] = _identity
) -> str | None:
    """Translate the pushable subset of `ir` to a SQL ``WHERE`` fragment, or None.

    An `AND` whose operands only partly translate yields the part that did, which is safe
    because the engine's own `Filter` re-checks every row a source returns. Do not reuse
    this for a ``DELETE``/``UPDATE`` predicate, where a widened filter would change rows
    the caller never named — pass ``exact=True`` for that, which is also what this passes
    itself under a `NOT`.

    Args:
        ir: The predicate's IR dictionary.
        exact: Decline any term that would only *widen* the result — a partly-translated
            `AND` and the collation-dependent `LIKE` forms — so the fragment matches the
            predicate exactly rather than a superset of it.
        quote: How to delimit a column name for the target dialect
            (`io.formats.sql.uri.quote_identifier`). Defaults to emitting it verbatim,
            which is what a caller that does not know its dialect must keep doing.

    Returns:
        A SQL boolean expression, or None when nothing pushable could be spelled.
    """
    e = ir.get("e")
    const = _const_bool(ir)
    if const is not None:
        return "1 = 1" if const else "1 = 0"
    if e == "is_null" and ir["input"].get("e") == "col":
        return f"{quote(ir['input']['name'])} IS NULL"
    if e == "is_not_null" and ir["input"].get("e") == "col":
        return f"{quote(ir['input']['name'])} IS NOT NULL"
    if e == "not":
        # Exact: negating a widened operand narrows the result and would drop rows.
        inner = to_sql_where(ir["input"], exact=True, quote=quote)
        return None if inner is None else f"NOT ({inner})"
    in_list = _in_list(ir)
    if in_list is not None:
        column, members = in_list
        if len(members) > _SQL_IN_MAX:
            return None
        rendered = ", ".join(_sql_literal(_literal({"value": m})) for m in members)
        return f"{quote(column)} IN ({rendered})"
    if not exact:
        like = _sql_like(ir, quote)
        if like is not None:
            return like
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = to_sql_where(ir["left"], exact=exact, quote=quote)
        right = to_sql_where(ir["right"], exact=exact, quote=quote)
        both = f"({left} {op.upper()} {right})" if left and right else None
        return both if exact else _combine(op, left, right, both)
    if op in COMPARISON_OPS:
        parsed = _col_and_literal(ir["left"], ir["right"])
        if parsed is None:
            return None
        col, value, flipped = parsed
        # NaN/Inf have no portable SQL literal spelling: ``col = nan`` / ``col < inf``
        # are rejected by every warehouse these connectors target (Snowflake,
        # BigQuery, ClickHouse, …). Leave the term unpushed — the engine's Filter
        # re-checks every row, so a non-pushed predicate is always correct.
        if isinstance(value, float) and not math.isfinite(value):
            return None
        effective = COMPARISON_FLIP[op] if flipped else op
        return f"{quote(col)} {_SQL_OP[effective]} {_sql_literal(value)}"
    return None
