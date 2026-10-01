"""A dimension-conditional aggregate input, split so its measure can be pre-aggregated.

`pre_aggregation_through_join` pushes a partial aggregate below an inner join whose other side
is unique on the join key, and it requires every aggregate input to read the pushed side
alone. One input shape reads both sides and still decomposes exactly: a `CASE` whose
condition reads only the unique side and whose value reads only the pushed side,

    SUM(CASE WHEN d_day_name = 'Sunday' THEN ws_ext_sales_price END)

Uniqueness puts exactly one dimension row behind each join key, so the condition is constant
per key, and the sum over a key's rows is the condition applied to the key's partial sum:

    SUM over keys of (CASE WHEN cond THEN SUM(value over the key's rows) END)

`MIN`/`MAX` decompose the same way. `COUNT` does not, with no `ELSE`: a group whose keys all
fail the condition counts 0, and the merged form would give NULL, so it is left alone.

This is the weekly-pivot shape of TPC-DS q2, q43 and q59 — seven such sums over a sales
table joined to `date_dim` — where it moves the condition from every sales row (millions of
string comparisons) to one evaluation per date.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Case, Col, Expr, Lit, NullIf, referenced_columns

__all__ = ["CONDITIONAL_AGGS", "conditional_merge_input", "split_dimension_conditional"]

#: Aggregates whose partials merge exactly under a per-key condition (see module doc).
CONDITIONAL_AGGS = frozenset({"sum", "min", "max"})


def _is_typed_null(otherwise: Expr, value: Expr) -> bool:
    """Whether `otherwise` is the NULL a `CASE` with no (or a NULL) `ELSE` carries.

    The SQL front-end types that NULL as the `THEN` value's own type by writing it as
    `NULLIF(value, value)`; a bare NULL literal is accepted too.
    """
    if isinstance(otherwise, Lit):
        return otherwise.value is None
    if isinstance(otherwise, NullIf):
        return otherwise.left.to_ir() == otherwise.right.to_ir() == value.to_ir()
    return False


def split_dimension_conditional(
    expr: Expr, pushed: set[str], unique: set[str]
) -> tuple[Expr, Expr] | None:
    """`(condition, value)` when `expr` is `CASE WHEN <unique side> THEN <pushed side> [ELSE NULL]`.

    Args:
        expr: An aggregate's input, in the join's output names.
        pushed: Output names of the side being pre-aggregated.
        unique: Output names of the side that is unique on the join key.

    Returns:
        The condition (reading `unique` only) and the value (reading `pushed` only), or None
        for any other shape.
    """
    if not isinstance(expr, Case) or len(expr.branches) != 1:
        return None
    cond, value = expr.branches[0]
    cond_cols, value_cols = referenced_columns(cond), referenced_columns(value)
    if not cond_cols or not cond_cols <= unique or not value_cols <= pushed:
        return None
    if not _is_typed_null(expr.otherwise, value):
        return None
    return cond, value


def conditional_merge_input(condition: Expr, partial: str) -> Expr:
    """The merged aggregate's input: the per-key condition applied to the partial column."""
    col = Col(partial)
    return Case([(condition, col)], NullIf(col, col))
