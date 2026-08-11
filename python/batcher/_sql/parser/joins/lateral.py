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
from batcher.plan.expr_ir import col
from batcher.plan.schema import suggest_columns

__all__ = ["is_true_literal", "lateral_select", "lateral_unnest", "select_unnest"]


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


#: What DuckDB names the element column of an `UNNEST` written without an `AS u(x)` column
#: list. Kept as a named constant because it is a *wire* name a user's `SELECT unnest` reads,
#: not an internal one.
_DUCKDB_UNNEST_COLUMN = "unnest"


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
    # `AS u(x)` names the element column `x`. **Without** a column list the element is
    # named `unnest`, which is what DuckDB calls it — and the list column stays in scope
    # either way.
    #
    # This used to give the element the *source* column's name and expand it in place, on
    # the stated belief that DuckDB did the same. It does not: `SELECT * FROM t,
    # UNNEST(arr)` returns `id, arr (the list), unnest (the element)` where Batcher
    # returned `id, arr (the element)` — one column fewer, with `arr` holding a different
    # type. Every existing test named its columns rather than starring, so a query that
    # asked for the shape SQL defines got a different relation and no error.
    alias = unnest.args.get("alias")
    cols = getattr(alias, "columns", None) if alias is not None else None
    out_name = cols[0].name if cols else _DUCKDB_UNNEST_COLUMN
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
    # An unnest in the FROM clause *adds* a relation; it does not consume the list column,
    # which stays in scope and repeats once per element. `explode(alias=...)` *renames* the
    # column it expands, so copy the list to the element name and expand the copy, leaving
    # the original where SQL says it is.
    if out_name in ds.columns:
        raise PlanError(
            f"UNNEST element column {out_name!r} collides with an existing column: "
            f"{list(ds.columns)}; name it with `AS u(<name>)`"
        )
    out = ds.with_columns(**{out_name: col(column)}).explode(
        out_name, outer=outer, index=index_name
    )
    if index_name is not None:
        # SQL ordinality is 1-based; `explode(index=)` is 0-based (matching Batcher's own
        # `with_row_index`). Shift here so the SQL surface keeps SQL's convention.
        out = out.with_columns(**{index_name: col(index_name) + 1})
    return out


def select_unnest(tr, ds: Dataset, projections) -> Dataset:
    """``SELECT unnest(xs)`` — an unnest written in the SELECT list rather than the FROM.

    DuckDB's shorthand for ``FROM t, UNNEST(t.xs) AS u(x)``, and the spelling most SQL in
    the wild uses. It reached the scalar path as an unhandled node and raised
    ``unsupported SQL expression: Explode``.

    It expands the *current* relation, so it is applied after WHERE (which SQL evaluates
    first) and before the projection, which then reads an ordinary column. Anything
    wrapped around it — ``unnest(xs) * 2`` — is evaluated per element by the projection,
    exactly as SQL specifies.

    Several unnests in one SELECT list *zip* rather than multiply in DuckDB, which
    `explode` cannot express, so that is rejected rather than answered with a cross
    product.

    Args:
        tr: The translator, for lowering a non-column argument.
        ds: The relation the SELECT reads.
        projections: The SELECT list; `Explode` nodes are replaced in place.

    Returns:
        `ds` expanded once per unnest, or `ds` unchanged when there is none.
    """
    found = [(p, e) for p in projections for e in p.find_all(exp.Explode)]
    if not found:
        return ds
    if len(found) > 1:
        raise NotImplementedError(
            f"{len(found)} UNNEST calls in one SELECT list are not supported — SQL zips "
            "them into one relation, which `explode` cannot express; unnest one list per "
            "query, or use FROM t, UNNEST(...) for each"
        )
    projection, explode = found[0]
    # Read the output name before the rewrite: an un-aliased `unnest(xs)` is named after
    # the expression as written, and replacing the node first would name it after the
    # internal column instead.
    # `_alias_of` renders an un-aliased item with sqlglot's default dialect, which spells
    # this node `EXPLODE(xs)`; the name every other engine (and DuckDB, the oracle) uses
    # is `unnest(xs)`.
    out_name = (
        explode.sql(dialect="duckdb").lower() if projection is explode else _alias_of(projection)
    )
    source = explode.this
    if isinstance(source, exp.Column) and source.name in ds.columns:
        column = source.name
    else:
        column = f"__bc_unnest{tr._win_arg_n}"
        tr._win_arg_n += 1
        ds = ds.with_columns(**{column: tr._scalar(source)})
    if projection is explode:
        # The whole select item is the unnest, so replacing the node *is* replacing the
        # item — and it has to carry the item's name with it. Re-wrapping afterwards would
        # act on a node already detached from the tree, and the column silently took the
        # source list's name (`xs`) instead of `unnest(xs)`.
        projection.replace(exp.alias_(exp.column(column), out_name))
    else:
        explode.replace(exp.column(column))
        if not isinstance(projection, exp.Alias):
            projection.replace(exp.alias_(projection.copy(), out_name))
    return ds.explode(column)


def is_true_literal(node) -> bool:
    """Whether an ON predicate is the constant `TRUE`."""
    while isinstance(node, exp.Paren):
        node = node.this
    return isinstance(node, exp.Boolean) and bool(node.this)
