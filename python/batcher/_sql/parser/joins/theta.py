"""Theta (non-equi) join lowering for the SQL translator.

A join whose ``ON`` predicate has no equality conjunct has no hash keys, so it is lowered
to the existing algebra: INNER is `cross join + filter` — the definition of a nested-loop
join — and LEFT/RIGHT/FULL add each preserved side's unmatched rows back, null-extended. Kept
here rather than in `from_clause.py` because it is a self-contained rewrite with its own
correctness argument, and that module is at its size limit.
"""

from __future__ import annotations

from batcher.api.dataset import Dataset

__all__ = ["and_conjuncts", "outer_theta_join", "swap_on_sides"]


def and_conjuncts(node) -> list:
    """Flatten an ``AND`` tree (and parentheses) into its conjunct list."""
    from sqlglot import expressions as exp

    if isinstance(node, exp.And):
        return and_conjuncts(node.this) + and_conjuncts(node.expression)
    if isinstance(node, exp.Paren):
        return and_conjuncts(node.this)
    return [node]


def _null_extend(ds: Dataset, other: Dataset, order: list[str]) -> Dataset:
    """Append `other`'s columns to `ds` as all-null, correctly typed, in `order`.

    There is no null *literal* in the expression IR, so rather than synthesize one per
    type, this left-joins against an EMPTY copy of `other`: a left join with nothing to
    match appends exactly that side's columns, correctly typed and all null — which is the
    null-extension being implemented.
    """
    from batcher.plan.expr_ir import lit

    ck = "__theta_null_key"
    return (
        ds.with_columns(**{ck: lit(1)})
        .join(other.limit(0).with_columns(**{ck: lit(1)}), on=ck, how="left")
        .drop(ck)
        .select(*order)
    )


def outer_theta_join(tr, ds: Dataset, right: Dataset, on, how: str) -> Dataset:
    """A LEFT/RIGHT/FULL join on a pure theta predicate, via matched plus unmatched.

    An outer join is the inner join plus each preserved side's rows that matched
    *nothing*, null-extended. With no equality conjunct the inner half is
    `cross join + filter`, and an unmatched half is found by tagging that side with a row
    index, collecting the indices that survived the filter, and anti-joining to get the
    rest. Those anti-joins are on the index — an equality — so they are ordinary hash
    joins, not further cross products.

    This is a nested-loop outer join expressed in the existing algebra: correct, but
    O(left x right) like any theta join, and the cross product materializes.

    Args:
        tr: The translator, used to lower the ON predicate.
        ds: The left relation.
        right: The right relation.
        on: The theta predicate.
        how: ``"left"``, ``"right"`` or ``"full"`` — which side(s) are preserved.

    Returns:
        The joined dataset, with the original left-then-right column order.
    """
    # A RIGHT join preserves the right side, which is the mirror image of a LEFT join, so
    # run it as one over swapped operands and restore the column order at the end. FULL is
    # symmetric, so it needs no swap.
    out_order = [*ds.columns, *right.columns]
    if how == "right":
        ds, right = right, ds
        on = swap_on_sides(on)

    lidx, ridx = "__theta_lidx", "__theta_ridx"
    left_tagged = ds.with_row_index(lidx)
    # FULL needs to identify unmatched rows on BOTH sides, so the right side is tagged too.
    right_tagged = right.with_row_index(ridx) if how == "full" else right
    matched = left_tagged.cross_join(right_tagged).filter(tr._scalar(on))
    cols = matched.columns

    # Left rows that matched nothing, null-extended with the right side's columns.
    left_missing = left_tagged.join(matched.select(lidx).distinct(), on=lidx, how="anti")
    parts = [matched, _null_extend(left_missing, right_tagged, cols)]

    if how == "full":
        # And symmetrically: right rows that matched nothing, null-extended with the left's.
        right_missing = right_tagged.join(matched.select(ridx).distinct(), on=ridx, how="anti")
        parts.append(_null_extend(right_missing, left_tagged, cols))

    out = parts[0]
    for part in parts[1:]:
        out = out.union(part, distinct=False)
    return out.drop(*[c for c in (lidx, ridx) if c in out.columns]).select(*out_order)


def swap_on_sides(on):
    """Mirror every equality in an ``ON`` predicate so its two sides trade places.

    `_split_join_on` decides which relation a key belongs to from the equality's operand
    *position*, so an ON predicate outlives an operand swap only if its equalities are
    mirrored with it. Non-equality conjuncts are left alone — they are filters, and a
    filter does not care which relation is on the left.

    Args:
        on: The `ON` predicate node.

    Returns:
        The predicate with each column-to-column equality's operands exchanged.
    """
    from sqlglot import expressions as exp

    swapped = []
    for conj in and_conjuncts(on):
        if (
            isinstance(conj, exp.EQ)
            and isinstance(conj.this, exp.Column)
            and isinstance(conj.expression, exp.Column)
        ):
            swapped.append(exp.EQ(this=conj.expression.copy(), expression=conj.this.copy()))
        else:
            swapped.append(conj)
    out = swapped[0]
    for term in swapped[1:]:
        out = exp.And(this=out, expression=term)
    return out
