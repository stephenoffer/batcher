"""Correlated **inequality** EXISTS/NOT EXISTS decorrelation — a range semi/anti join.

`EXISTS (SELECT 1 FROM b WHERE a.x < b.y)` is exactly ``a SEMI JOIN b ON a.x < b.y``, and
`NOT EXISTS` is the anti join. Before this the translator raised
`NotImplementedError: correlated subqueries not supported` for the shape: an inequality is
neither a correlation *key* (only equalities become one) nor a *local* predicate, so it fell
through to `_reject_correlated`.

It is a thin front-end helper, like its `<>` sibling in `subquery_neq`: the engine side was
already done. `bc_ir::RelOp::RangeJoin` carries a `join_type`, and
`bc_runtime::join::range_join_indices` implements `Semi` and `Anti` and is fuzzed against a
brute-force cross-product oracle for both — this is simply the first caller to emit a
`RangeJoin` that is not an inner join.

The obvious alternative does not work: a cross join, the predicate as a filter, then
`DISTINCT` collapses duplicate qualifying outer rows, which a semi join must keep.
`test_correlated_exists_preserves_outer_duplicates` pins that.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.api._join_helpers import range_semi_join

__all__ = ["decorrelate_inequality_exists"]


# sqlglot comparison -> the wire op, oriented `outer OP inner`, plus the reading when the
# operands arrive the other way round (`b.y > a.x` is `a.x < b.y`).
_INEQUALITIES: dict[str, tuple[str, str]] = {
    "LT": ("lt", "gt"),
    "LTE": ("le", "ge"),
    "GT": ("gt", "lt"),
    "GTE": ("ge", "le"),
}


def _inequality_correlation(leaves: list, local: set[str], local_cols: set[str] | None):
    """The single `outer.col <op> inner.col` inequality among `leaves`, or `None`.

    Returns a list of `(leaf, outer_col, inner_col, op)`, each `op` oriented
    ``outer OP inner`` — the orientation `RangeCondition` stores. `None` when there are none
    (not this shape) or more than two (past the engine's two-axis ceiling).
    """
    found = []
    for leaf in leaves:
        cls = type(leaf).__name__.upper()
        ops = _INEQUALITIES.get(cls)
        if ops is None:
            continue
        lhs, rhs = leaf.this, leaf.expression
        if not (isinstance(lhs, exp.Column) and isinstance(rhs, exp.Column)):
            continue

        def _is_outer(c) -> bool:
            if c.table:
                return c.table not in local
            return local_cols is not None and c.name not in local_cols

        lhs_outer, rhs_outer = _is_outer(lhs), _is_outer(rhs)
        if lhs_outer == rhs_outer:
            continue
        candidate = (
            (leaf, lhs.name, rhs.name, ops[0]) if lhs_outer else (leaf, rhs.name, lhs.name, ops[1])
        )
        found.append(candidate)
    # Two is the engine's ceiling (IEJoin sorts on two axes). A third inequality would have to
    # stay a post-filter, and a *semi* join has no way to express one — it emits no right
    # columns to filter on — so the whole shape is declined rather than silently narrowed.
    return found if 1 <= len(found) <= 2 else None


def decorrelate_inequality_exists(tr, ds, inner, local_preds, local, local_cols, negate):
    """The range semi/anti join for an inequality-correlated EXISTS, or `None`.

    `None` means "not this shape" — no inequality correlation, or more than one, which would
    need a two-condition range *semi* join that the planner does not emit yet. The caller
    then continues down its existing paths unchanged.

    Everything except the correlation stays as the inner relation's own filter, and the
    subquery's select list becomes just the joined column — the same treatment the equality
    path gives its correlation keys. `_reject_correlated` still runs afterwards, so any
    *other* outer reference is declined as before rather than silently mis-planned.

    Args:
        tr: The translator, for recursing into the subquery.
        ds: The outer dataset.
        inner: The subquery's SELECT node (already detached from the outer AST).
        local_preds: The subquery's WHERE leaves that are not correlation equalities.
        local: Table names and aliases the subquery introduces.
        local_cols: The local tables' column names, or `None` when unresolvable.
        negate: `True` for `NOT EXISTS`.

    Returns:
        The joined `Dataset`, or `None` when the shape does not apply.
    """

    from batcher._sql.parser.core_utils import _join_and
    from batcher._sql.parser.subquery.core import _reject_correlated

    found = _inequality_correlation(local_preds, local, local_cols)
    if found is None:
        return None
    used = {id(leaf) for leaf, _o, _i, _op in found}
    rest = [p for p in local_preds if id(p) not in used]
    inner.set("where", exp.Where(this=_join_and(rest)) if rest else None)
    inner.set("group", None)
    # Project exactly the columns the join reads, deduplicated in first-seen order — the same
    # treatment the equality path gives its correlation keys.
    inner_cols: list[str] = []
    for _leaf, _outer, inner_col, _op in found:
        if inner_col not in inner_cols:
            inner_cols.append(inner_col)
    inner.set("expressions", [exp.column(c) for c in inner_cols])
    _reject_correlated(inner)
    conditions = [(outer, inner_col, op) for _leaf, outer, inner_col, op in found]
    return range_semi_join(ds, tr.statement(inner), conditions, negate=negate)
