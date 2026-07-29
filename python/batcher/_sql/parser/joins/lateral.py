"""LATERAL handling for the SQL translator — `UNNEST` and no-FROM lateral subqueries.

Both are lateral in the SQL sense (evaluated per outer row) but neither is a join: an
`UNNEST` expands the row it belongs to, and a `LATERAL (SELECT <exprs>)` with no FROM
computes values from the outer row. They map to `Dataset.explode` and `with_columns`
respectively. Kept beside the join rewrites because `from_clause.py` is at its size limit.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.parser.core_utils import _alias_of, _unwrap_alias
from batcher.api.dataset import Dataset
from batcher.plan.schema import suggest_columns

__all__ = ["is_true_literal", "lateral_select", "lateral_unnest"]


def lateral_select(tr, ds: Dataset, lateral) -> Dataset:
    """`LATERAL (SELECT <exprs>)` with no FROM — per-row values from the outer row.

    A lateral subquery is evaluated once per outer row and may reference that row's
    columns. When it has no `FROM` of its own it produces exactly one row per outer row
    computed from those columns, which is precisely `with_columns` — no join, no
    correlation machinery.

    A lateral that *does* read a table is a correlated join (a different row count per
    outer row) and is rejected rather than approximated.

    Args:
        tr: The translator, used to lower the projected expressions.
        ds: The outer relation.
        lateral: The `Lateral` node.

    Returns:
        The relation with the lateral's expressions appended as columns.
    """
    inner = lateral.this
    inner = inner.this if isinstance(inner, exp.Subquery) else inner
    # sqlglot spells the FROM key `from_` in some versions and `from` in others; reading
    # only one let a lateral that DOES read a table fall through to the checks below, so a
    # correlated lateral without a WHERE would have been mistranslated into `with_columns`.
    has_from = inner.args.get("from_") or inner.args.get("from")
    if not isinstance(inner, exp.Select) or has_from:
        raise NotImplementedError(
            "LATERAL is supported only for a subquery with no FROM (per-row computed "
            "columns, e.g. `FROM t, LATERAL (SELECT t.a + 1 AS b)`); a lateral that reads "
            "a table is a correlated join, which is not supported"
        )
    if inner.args.get("group") or inner.args.get("where") or inner.args.get("having"):
        raise NotImplementedError(
            "a LATERAL with no FROM cannot have WHERE / GROUP BY / HAVING — there is no "
            "relation for them to apply to"
        )
    # An unaliased expression takes its SQL text as the column name, which is what DuckDB
    # does too (it parenthesises: `(v * 2)` where this gives `v * 2` — a pre-existing
    # auto-naming difference, not a value difference).
    named = {_alias_of(proj): tr._scalar(_unwrap_alias(proj)) for proj in inner.expressions}
    return ds.with_columns(**named)


def lateral_unnest(ds: Dataset, join) -> Dataset:
    """`UNNEST(<list column>) AS alias(name)` in the FROM clause → `Dataset.explode`.

    This is the one relational operation nested media needs most — chunks of a document,
    frames of a clip, segments of an audio file all arrive as a list column — and it is
    exactly what `explode` already does, so no new operator is involved.

    Only the column form is supported. `UNNEST` over a literal array, over several
    columns at once (which zips rather than explodes), and `WITH OFFSET` (which needs an
    ordinality column the operator does not emit) are each rejected explicitly rather than
    silently mistranslated.
    """
    unnest = join.this
    on = join.args.get("on")
    # `LEFT JOIN UNNEST(...) ON TRUE` is SQL's spelling of an outer unnest: the row
    # survives even when its list is empty. `ON TRUE` is the only predicate that means
    # anything here — an unnest expands the row it belongs to, so there is nothing to
    # join *against*.
    if join.args.get("using") or (on is not None and not is_true_literal(on)):
        raise NotImplementedError(
            "UNNEST takes no join predicate other than ON TRUE — it expands the row it "
            "belongs to; write `FROM t, UNNEST(t.col) AS u(x)`"
        )
    outer = (join.side or "").upper() == "LEFT"
    exprs = unnest.expressions
    if len(exprs) != 1:
        raise NotImplementedError(
            f"UNNEST of {len(exprs)} expressions is not supported — SQL zips them into "
            "one relation; unnest a single list column per UNNEST"
        )
    target = exprs[0]
    if not isinstance(target, exp.Column):
        raise NotImplementedError(
            "UNNEST is supported over a list *column* only, not over "
            f"{type(target).__name__.lower()}; project the value into a column first"
        )
    column = target.name
    if column not in ds.columns:
        known = list(ds.columns)
        raise PlanError(
            f"UNNEST: unknown column {column!r}; available: {known}{suggest_columns(column, known)}"
        )
    # `AS u(x)` names the element column `x`. Without the column list the element keeps
    # the source column's name, which is what DuckDB does for `UNNEST(xs)`.
    alias = unnest.args.get("alias")
    cols = getattr(alias, "columns", None) if alias is not None else None
    out_name = cols[0].name if cols else column
    # `WITH ORDINALITY` names an extra position column. sqlglot puts its name in
    # `offset` (or `True` for the bare form, whose conventional name is `ordinality`).
    ordinality = unnest.args.get("offset")
    index_name = None
    if ordinality:
        index_name = "ordinality" if ordinality is True else ordinality.name
        if index_name in ds.columns:
            raise PlanError(
                f"UNNEST ordinality column {index_name!r} collides with an existing "
                f"column: {list(ds.columns)}"
            )
    out = ds.explode(
        column,
        alias=out_name if out_name != column else None,
        outer=outer,
        index=index_name,
    )
    if index_name is not None:
        # SQL ordinality is 1-based; `explode(index=)` is 0-based (matching Batcher's own
        # `with_row_index`). Shift here so the SQL surface keeps SQL's convention.
        from batcher.plan.expr_ir import col

        out = out.with_columns(**{index_name: col(index_name) + 1})
    return out


def is_true_literal(node) -> bool:
    """Whether an ON predicate is the constant `TRUE`."""
    while isinstance(node, exp.Paren):
        node = node.this
    return isinstance(node, exp.Boolean) and bool(node.this)
