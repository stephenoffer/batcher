"""Uncorrelated scalar subqueries, evaluated once each and inlined as literals.

Split from `expressions.scalar` (the scalar dispatch that reaches it) because evaluating one is
not expression translation: it runs a whole query while the SQL is being translated.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.expressions.lowering import typed_null
from batcher.plan.expr_ir import Expr, lit

__all__ = ["scalar_subquery"]


def scalar_subquery(tr, select_node) -> Expr:
    """Uncorrelated scalar subquery → a literal.

    Translate the inner SELECT, collect it **eagerly** (this executes the subquery now,
    while the SQL is being translated), check it is at most 1 row x 1 column, and inline the
    value as a literal. A lazy join against the subquery's one row is the alternative, and it
    measured ~60x slower than the literal filter over 4M rows. More than one row raises the
    typed `ExecutionError` DuckDB's message describes.

    The same subquery written twice in one statement is evaluated once (`_subquery_key`).
    """
    tr._reject_correlated(select_node)
    key = _subquery_key(tr, select_node)
    known = tr._scalar_values.get(key)
    if known is None:
        known = tr._scalar_values[key] = _evaluate_scalar_subquery(tr, select_node)
    return known


def _subquery_key(tr, select_node) -> tuple:
    """What makes two uncorrelated scalar subqueries in one statement the same value.

    Its text, and the relation each table name in it is bound to right now -- a CTE name is
    bound in the translator's registry, so the same text under a different `WITH` scope keys
    differently. Equal keys must give one literal and not merely equal ones: a float
    aggregate run twice can differ in its last bit (the summation order follows the plan the
    learning loop picks for each run), and TPC-DS q14 inlines the one `avg_sales` subquery in
    three `HAVING`s. One of three evaluations rounding differently made the ROLLUP levels'
    inputs structurally unequal, so `api.multi_group` declined to share them and the query ran
    18x slower.
    """
    names = sorted({t.name.lower() for t in select_node.find_all(exp.Table)})
    return (select_node.sql(), tuple((n, id(tr._registry.get(n))) for n in names))


def _evaluate_scalar_subquery(tr, select_node) -> Expr:
    """Run the subquery and return its single value as a literal (see `scalar_subquery`)."""
    # Detach from the outer AST so ancestor walks (e.g. _has_aggregate's
    # Subquery/Window checks) stay within the subquery's own scope.
    select_node = select_node.copy()
    # The subquery may itself aggregate, which resets the translator's aggregate
    # bookkeeping (``_agg_map`` / ``_agg_n``). Save and restore it so the enclosing
    # query's aggregate columns still resolve after the subquery is evaluated — e.g.
    # ``HAVING sum(x) > (SELECT sum(x) * k FROM ...)`` (TPC-H Q11).
    saved_agg_map, saved_agg_n = tr._agg_map, tr._agg_n
    try:
        inner_ds = tr.statement(select_node)
        if len(inner_ds.columns) != 1:
            raise NotImplementedError("scalar subquery must project exactly one column")
        table = inner_ds.collect()
    finally:
        tr._agg_map, tr._agg_n = saved_agg_map, saved_agg_n
    if table.num_rows == 0:
        # SQL: a scalar subquery with no rows is NULL (typed as its output column),
        # not an error — e.g. `(SELECT sal FROM emp WHERE id=999)` is NULL per row.
        return typed_null(table.schema.field(0).type)
    if table.num_rows > 1:
        from batcher._internal.errors import ExecutionError
        from batcher._sql.parser.subquery.scalar_sub import MULTIPLE_ROWS_MESSAGE

        raise ExecutionError(f"{MULTIPLE_ROWS_MESSAGE} (got {table.num_rows} rows)")
    value = table.column(0)[0].as_py()
    if value is None:
        # One row whose value *is* NULL — a different case from the no-rows one above, and
        # the one an ordinary threshold query hits: `WHERE x > (SELECT AVG(x) FROM t)` over
        # an empty or all-null column returns a single NULL row, not zero rows. `lit(None)`
        # has no wire form (the IR has no untyped null literal), so this raised a bare
        # `TypeError: unsupported literal type: NoneType` from deep inside `to_ir` where
        # DuckDB simply returns no rows.
        return typed_null(table.schema.field(0).type)
    return lit(value)
