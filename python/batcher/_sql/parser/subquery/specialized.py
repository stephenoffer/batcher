"""The correlated `EXISTS` shapes that are not a plain equi-semi-join, and their order.

Two of them, and this module is where `core._apply_exists` asks about both at once rather
than testing each in line: a **pure inequality** correlation, which `range` turns into a range
semi join, and an equality carrying an inequality alongside it, which is implemented here.
They are tried together because they are the same step in the same decision -- "is this a
shape with its own plan, before the general semi join?" -- and splitting that question across
two blocks in the caller is what let the second shape go unnoticed while the first was built.

# The mixed correlation

`EXISTS (SELECT 1 FROM b WHERE b.k = a.k AND b.v > a.v)` is the one correlation shape the
other three decorrelations leave: `core` turns equalities into semi-join keys, `range` turns a
lone inequality into a range semi join, and `neq` handles a `<>` residual — but a predicate
carrying an equality *and* an inequality matched none of them, so the inequality stayed in the
inner relation's `WHERE` still referencing the outer table and `_reject_correlated` raised
`NotImplementedError: correlated subqueries not supported`. DuckDB answers the query.

# Why this is a row-tag rewrite and not a new join

The natural plan is a semi join on `k` with the inequality as a residual, and the engine has no
such operator: `RangeJoin` deliberately excludes `=` (`bc_ir::RangeOp` says so — an equality is
a hash join), and a semi join emits no right columns, so there is nothing for a residual filter
to read. Adding an equi-prefix to `RangeJoin` would be a two-sided IR change for one SQL shape.

So the correlation is decorrelated the general way instead, with operators that already exist:
tag each outer row, inner-join on the equality keys, apply the inequality as an ordinary filter
on the joined rows, and reduce the survivors to the set of outer tags that matched.

**The tag is what makes it correct, and it is not incidental.** Without it the rewrite would be
a join, a filter and a `DISTINCT` over the outer columns — which collapses two identical outer
rows into one, where `EXISTS` must keep both. That is the same trap
`test_correlated_exists_preserves_outer_duplicates` pins for the `range` path, and a row index
is what lets the final `DISTINCT` be taken over an identity rather than over values.

The cost is that the inner join materializes every matching pair before the filter discards
them, where a semi join would stop at the first match per outer row. That is the price of the
shape currently having no plan at all, and it is bounded by the equality keys — this is a
hash join on `k`, not a cross product.

Nulls need no special care: a comparison against `NULL` is `NULL`, the filter drops it, so the
outer row finds no match and is excluded from `EXISTS` and kept by `NOT EXISTS`, which is what
SQL says.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import col

__all__ = ["decorrelate_correlated_exists"]

# `outer OP inner`, the orientation `_inequality_correlation` returns.
_APPLY = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def _fresh(taken: set[str], stem: str) -> str:
    """A column name `taken` does not already hold.

    The joined relation carries the outer columns beside the projected inner ones, so an inner
    column sharing a name with an outer one would make the filter below ambiguous about which
    it means. Renaming the inner side on the way in removes the question.
    """
    name = f"__bx_{stem}"
    n = 0
    while name in taken:
        n += 1
        name = f"__bx_{stem}_{n}"
    return name


def _decorrelate_mixed(tr, ds, inner, corr, local_preds, local, local_cols, negate):
    """The join-filter-reduce rewrite for a mixed correlation, or `None`.

    `None` means "not this shape" — no inequality among the local predicates — and the caller
    continues down its existing paths unchanged, so the equality-only correlation keeps the
    plain semi join it already had.

    Args:
        tr: The translator, for recursing into the subquery.
        ds: The outer dataset.
        inner: The subquery's SELECT node, already detached from the outer AST.
        corr: The `(outer_col, inner_col)` equality pairs that correlate the subquery.
        local_preds: The subquery's WHERE leaves that are not correlation equalities.
        local: Table names and aliases the subquery introduces.
        local_cols: The local tables' column names, or `None` when unresolvable.
        negate: `True` for `NOT EXISTS`.

    Returns:
        The rewritten `Dataset`, or `None` when the shape does not apply.
    """
    from batcher._sql.parser.core_utils import _join_and
    from batcher._sql.parser.subquery.correlation import _reject_correlated
    from batcher._sql.parser.subquery.range import _inequality_correlation

    found = _inequality_correlation(local_preds, local, local_cols)
    if not corr or not found:
        return None

    outer_cols = list(ds.columns)
    taken = set(outer_cols)

    # Rename every projected inner column to a name the outer relation cannot also have, so
    # the filter and the join keys below are unambiguous.
    key_names, projections = [], []
    for _outer, inner_col in corr:
        name = _fresh(taken, f"k{len(key_names)}")
        taken.add(name)
        key_names.append(name)
        projections.append(exp.alias_(exp.column(inner_col), name))

    residuals = []
    for _leaf, outer_col, inner_col, op in found:
        name = _fresh(taken, f"v{len(residuals)}")
        taken.add(name)
        residuals.append((outer_col, name, op))
        projections.append(exp.alias_(exp.column(inner_col), name))

    # Everything that is neither a correlation equality nor the inequality stays the inner
    # relation's own filter, exactly as the other decorrelations leave it.
    used = {id(leaf) for leaf, _o, _i, _op in found}
    rest = [p for p in local_preds if id(p) not in used]
    inner.set("where", exp.Where(this=_join_and(rest)) if rest else None)
    inner.set("group", None)
    inner.set("expressions", projections)
    _reject_correlated(inner)  # any *other* outer reference is still declined

    rid = _fresh(taken, "rid")
    tagged = ds.with_row_index(rid)
    matched = tagged.join(
        tr.statement(inner),
        left_on=[outer for outer, _inner in corr],
        right_on=key_names,
        how="inner",
    )
    for outer_col, name, op in residuals:
        matched = matched.filter(_APPLY[op](col(outer_col), col(name)))

    # The set of outer rows that matched, as identities rather than as values — which is what
    # keeps two identical outer rows two rows.
    qualifying = matched.select(rid).distinct()
    kept = tagged.join(
        qualifying,
        left_on=[rid],
        right_on=[rid],
        how="anti" if negate else "semi",
    )
    return kept.select(*outer_cols)


def decorrelate_correlated_exists(tr, ds, inner, corr, local_preds, local, local_cols, negate):
    """The specialized decorrelation for this correlation shape, or `None` for neither.

    `None` means the correlation is a plain set of equalities, and the caller's general
    semi/anti join on those keys is the right plan.

    Args:
        tr: The translator, for recursing into the subquery.
        ds: The outer dataset.
        inner: The subquery's SELECT node, already detached from the outer AST.
        corr: The `(outer_col, inner_col)` equality pairs that correlate the subquery.
        local_preds: The subquery's WHERE leaves that are not correlation equalities.
        local: Table names and aliases the subquery introduces.
        local_cols: The local tables' column names, or `None` when unresolvable.
        negate: `True` for `NOT EXISTS`.

    Returns:
        The rewritten `Dataset`, or `None` when neither shape applies.
    """
    from batcher._sql.parser.subquery.range import decorrelate_inequality_exists

    if not corr:
        return decorrelate_inequality_exists(tr, ds, inner, local_preds, local, local_cols, negate)
    return _decorrelate_mixed(tr, ds, inner, corr, local_preds, local, local_cols, negate)
