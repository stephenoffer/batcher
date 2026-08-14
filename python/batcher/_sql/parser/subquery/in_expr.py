"""`IN (subquery)` where the left-hand side is an *expression* rather than a bare column.

`WHERE v + 1 IN (SELECT k FROM u)` raised
`NotImplementedError: IN (subquery) supports a plain column or a row value of columns`.
The restriction was in the key resolution rather than in the plan: `_apply_in_subquery` reads
the left side as a *column name* so it can hand it to a semi/anti join, and an expression has
no name to hand over. Everything past that point — the correlation split, the three-valued
`NOT IN` handling, the multi-column row value — was already general.

So this gives the expression a name and lets the existing path do the work: evaluate it into a
synthetic column, rewrite the predicate to name that column, run the ordinary decorrelation,
then project the outer columns back so the synthetic one never reaches the caller. The
recursion terminates because the rewritten predicate's left side is a plain column, which is
the case that does not come back here.

Naming the value rather than special-casing it is what keeps `NOT IN` correct. `x NOT IN (S)`
is not an anti join when `S` can yield NULL (`_not_in_antijoin` implements the three-valued
answer), and a second implementation of `IN` for expressions would have had to restate that
rule and could have restated it wrongly.
"""

from __future__ import annotations

__all__ = ["in_over_expression"]


def in_over_expression(tr, ds, node, *, negate: bool):
    """Decorrelate `<expr> IN (subquery)` by naming `<expr>` first.

    Args:
        tr: The translator, for evaluating the left-hand expression and the subquery.
        ds: The outer dataset.
        node: The `IN` node, whose `this` is the left-hand expression.
        negate: `True` for `NOT IN`.

    Returns:
        A new `Dataset` carrying exactly `ds`'s columns.
    """
    from sqlglot import expressions as exp

    from batcher._sql.parser.subquery.core import _apply_in_subquery

    outer_cols = list(ds.columns)
    name = "__bx_in"
    n = 0
    while name in outer_cols:
        n += 1
        name = f"__bx_in_{n}"

    extended = ds.with_columns(**{name: tr._scalar(node.this)})
    probe = node.copy()
    probe.set("this", exp.column(name))
    return _apply_in_subquery(tr, extended, probe, negate=negate).select(*outer_cols)
