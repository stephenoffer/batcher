"""The output name a positional expression gets when nothing names it.

``ds.select(col("a") + 1)`` has no keyword and no ``.alias(...)``, so the projection has to
name the column itself. Batcher names it the way Polars does, because the migration
codemod carries Polars code across unchanged and the column names are part of what that
code reads back: the name is the name of the expression's **leftmost leaf**, walked in
input order. A column leaf gives its own name, a literal gives ``"literal"``, and an
``.alias(...)`` anywhere on that leftmost path wins over what is below it. A
``when(...).then(...)`` chain takes its first ``then`` value's name, which is what Polars
reads as its leftmost input.

The rule is deterministic and purely structural, so two positional expressions can infer
the same name. That is refused by the caller (`name_positionals`) rather than resolved by
suffixing, as Polars refuses it: a silent ``a_1`` is a column nobody asked for.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr, Aliased, Case, Col, Expr, Lit, WindowExpr
from batcher.plan.expr_rewrite.traverse import _EXPR_KIDS

__all__ = ["LITERAL_NAME", "name_positionals", "output_name"]

#: The name a projection with no column leaf gets, matching Polars.
LITERAL_NAME = "literal"

#: The name of an input-less aggregate (``count()``), after its own function.
_COUNT_NAME = "count"


def _key_expr(key: Any) -> Any:
    """The expression inside a window key, which may be a ``(key, descending, ...)`` tuple."""
    return key[0] if isinstance(key, tuple) else key


def _leaf_name(key: Any) -> str | None:
    """The name a window key would contribute: a string is a column name."""
    key = _key_expr(key)
    if isinstance(key, str):
        return key
    return output_name(key) if isinstance(key, (Expr, AggExpr)) else None


def output_name(expr: Expr | AggExpr) -> str:
    """The name Polars would give `expr` as an unnamed projection.

    Args:
        expr: The positional expression to name.

    Returns:
        The inferred output column name.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_rewrite.naming import output_name
            >>> output_name(bt.col("a") + 1), output_name(bt.lit(1) + bt.col("a"))
            ('a', 'literal')
            >>> output_name(bt.col("x").sum()), output_name(bt.count())
            ('x', 'count')
    """
    if isinstance(expr, Aliased):
        return expr.name
    if isinstance(expr, Col):
        return expr.name
    if isinstance(expr, Lit):
        return LITERAL_NAME
    if isinstance(expr, AggExpr):
        if expr.name is not None:
            return expr.name
        return output_name(expr.input) if expr.input is not None else _COUNT_NAME
    if isinstance(expr, WindowExpr):
        if expr.input is not None:
            return output_name(expr.input)
        # An input-less window (`rank()`, `row_number()`) is named after what it orders or
        # partitions by -- `col("x").rank()` is built as `rank() OVER (ORDER BY x)`, and
        # Polars names that `x`.
        for key in (*expr.order_by, *expr.partition_by):
            name = _leaf_name(key)
            if name is not None:
                return name
        return expr.func
    if isinstance(expr, Case):
        return output_name(expr.branches[0][1])
    kids_of = _EXPR_KIDS.get(type(expr))
    if kids_of is not None:
        kids = kids_of(expr)
        if kids:
            return output_name(kids[0])
    # A leaf that is neither a column nor a literal reads no column, so it is a constant
    # for naming purposes, as Polars names `int_range`/`lit` alike.
    return LITERAL_NAME


def name_positionals(
    exprs: Iterable[Expr | AggExpr], *, api: str, taken: Iterable[str] = ()
) -> dict[str, Expr | AggExpr]:
    """Name each positional expression, refusing two that would share a name.

    Args:
        exprs: The positional expressions, in call order. Selectors must already be
            expanded; a top-level ``.alias(...)`` is unwrapped here.
        api: The calling verb, named in the error.
        taken: Names already bound by the caller (earlier projections), checked too.

    Returns:
        An ordered mapping from inferred name to the expression it names.

    Raises:
        PlanError: If two expressions infer or declare the same output name.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_rewrite.naming import name_positionals
            >>> list(name_positionals([bt.col("a") + 1, bt.col("b")], api="select"))
            ['a', 'b']
    """
    seen = set(taken)
    out: dict[str, Expr | AggExpr] = {}
    for expr in exprs:
        name = output_name(expr)
        if name in seen:
            raise PlanError(
                f"{api}() would produce the output column {name!r} twice; an unnamed "
                "expression is named after its leftmost column (or 'literal'), so two of "
                f"them can collide -- rename one with .alias('...') or pass it as a keyword"
            )
        seen.add(name)
        out[name] = expr.inner if isinstance(expr, Aliased) else expr
    return out
