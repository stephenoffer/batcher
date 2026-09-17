"""Translate one SQL *expression* (not a query) into an `Expr`.

`bt.sql_expr("a + 1 AS b")` and `bt.call_function("pmod", col("a"), 3)` reach the same
scalar translator `bt.sql` uses for a select list, so a function has one lowering whichever
front end names it. What differs is the context: there is no relation in scope, so column
references stay unresolved names, and an aggregate becomes an `AggExpr` for the caller to
place in `agg(...)` rather than a column a `GROUP BY` has already computed.

Registered Python functions (`bt.register_function`) are not reachable here. `bt.sql` runs
them as a `map_batches` stage over a relation, which an expression has no way to carry.
"""

from __future__ import annotations

import re
from typing import Any

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._internal.sql_errors import parse_sql
from batcher._sql.parser import grouping
from batcher._sql.parser.expressions.aggregates import is_agg_node
from batcher._sql.parser.translator import _Translator
from batcher.plan.expr_ir import Expr, Lit

__all__ = ["call_sql_function", "parse_sql_expression"]

# A function name `call_function` splices into SQL text, so it must be a bare identifier.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The statement kinds `sql_expr` refuses: a query or a command is not an expression.
_STATEMENTS = (
    exp.Query,
    exp.Values,
    exp.DDL,
    exp.DML,
    exp.Command,
    exp.Alter,
    exp.Cache,
    exp.Describe,
    exp.Drop,
    exp.Pragma,
    exp.Set,
    exp.Show,
    exp.TruncateTable,
    exp.Use,
)


class _ExpressionTranslator(_Translator):
    """The scalar translator with no relation in scope, plus bound placeholder arguments."""

    def __init__(self, bound: dict[str, Expr]) -> None:
        super().__init__({}, {})
        self._bound = bound

    def _scalar(self, node) -> Expr:
        if isinstance(node, exp.Column) and not node.table and node.name in self._bound:
            return self._bound[node.name]
        if is_agg_node(node):
            return grouping._agg(self, node)
        return super()._scalar(node)


def parse_sql_expression(text: str, *, dialect: str) -> Expr:
    """Parse and translate one SQL expression, honoring a trailing ``AS name``.

    Args:
        text: The SQL expression text.
        dialect: The sqlglot read dialect.

    Returns:
        The translated expression, aliased when the text carried ``AS name``.

    Raises:
        PlanError: If `text` does not parse, is a query or statement rather than an
            expression, or uses a construct the translator does not support.
    """
    if not isinstance(text, str):
        raise PlanError(f"sql_expr() expects a SQL string, got {type(text).__name__}")
    node = parse_sql(text, dialect=dialect)
    if isinstance(node, _STATEMENTS):
        raise PlanError(
            f"sql_expr() takes one expression, not a {type(node).__name__.upper()} statement; "
            "use bt.sql(...) to run a query"
        )
    if node.find(exp.Query) is not None:
        raise PlanError("sql_expr() cannot hold a subquery, which needs a relation; use bt.sql")
    return _translate(node, {}, where="sql_expr()")


def call_sql_function(name: str, args: tuple[Any, ...], *, dialect: str) -> Expr:
    """Build the call ``name(args...)`` through the SQL function table.

    A Python ``int``/``float``/``bool`` and a plain ``bt.lit`` constant are spliced in as SQL
    literals, so a function that needs a constant argument (``find_in_set``'s needle, a
    ``parse_url`` part) receives one. A string is a column name, and any other expression is
    bound in place of a placeholder column.

    Args:
        name: The SQL function name.
        args: The call's arguments.
        dialect: The sqlglot read dialect whose function names apply.

    Returns:
        The translated expression.

    Raises:
        PlanError: If `name` is not an identifier, or the call does not translate.
    """
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
        raise PlanError(f"call_function(): {name!r} is not a function name")
    bound: dict[str, Expr] = {}
    rendered = []
    for position, arg in enumerate(args):
        literal = _sql_literal(arg)
        if literal is not None:
            rendered.append(literal.sql(dialect=dialect))
            continue
        placeholder = f"__bt_call_arg_{position}"
        bound[placeholder] = arg if isinstance(arg, Expr) else _column(arg)
        rendered.append(placeholder)
    node = parse_sql(f"{name}({', '.join(rendered)})", dialect=dialect)
    return _translate(node, bound, where=f"call_function({name!r})")


def _sql_literal(arg: Any) -> exp.Expression | None:
    """The SQL literal a Python scalar or a plain `bt.lit` constant denotes, else None."""
    if type(arg) is Lit:
        return exp.convert(arg.value) if isinstance(arg.value, (bool, int, float, str)) else None
    if isinstance(arg, (bool, int, float)):
        return exp.convert(arg)
    return None


def _column(arg: Any) -> Expr:
    """A string argument as a column reference, as Spark's ``ColumnOrName`` reads it."""
    from batcher.plan.expr_ir import col

    if not isinstance(arg, str):
        raise PlanError(
            f"call_function(): an argument must be an expression, a column name or a "
            f"number, got {type(arg).__name__}"
        )
    return col(arg)


def _translate(node, bound: dict[str, Expr], *, where: str) -> Expr:
    """Translate `node`, applying an alias and turning a refusal into `PlanError`."""
    alias = node.alias if isinstance(node, exp.Alias) else None
    body = node.this if isinstance(node, exp.Alias) else node
    try:
        out = _ExpressionTranslator(bound)._scalar(body)
    except NotImplementedError as exc:
        raise PlanError(f"{where}: {exc}") from exc
    return out.alias(alias) if alias else out
