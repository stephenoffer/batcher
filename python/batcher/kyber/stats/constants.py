"""When a *computed* column is provably a constant — the one projection that keeps EXACT.

`project_columns` carries a column's statistics through a projection, and drops them for
anything computed: the output distribution of `a * b` is unknown, so claiming anything about it
would be a guess. There is one exception, and it is worth its own module because it is the seam
where a whole family of terminals stops being a query.

A column whose EXACT statistics say `min == max` with no nulls holds **one value and no other**.
An expression over such columns therefore has one value too — and the control plane can compute
it once, instead of the engine computing it a billion times.

That is not an exotic shape. `ds.null_count()` lowers to `count(*) - count(col)` over a *global
aggregate*, whose outputs are exactly such constants. Because the subtraction was not folded,
the one-row answer that the footer fully determined was obtained by scanning the table.

The folding itself is *borrowed*, not restated: `fold_expression` is the optimizer's constant
folder, and an estimator with its own private idea of what `a - b` means would be a second
definition of the language's arithmetic, free to drift from the first. Substituting the
constants and handing the result to the existing folder keeps exactly one.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir import Col, Expr, Lit, referenced_columns
from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = ["constant_projection_stat", "constant_value"]


def constant_projection_stat(expr: Expr, child: RelStats) -> ColumnStat | None:
    """The EXACT constant `expr` evaluates to over `child`, or None if it is not one.

    Every column the expression reads must be a *proven* constant (see `constant_value`); the
    substituted expression is then folded, and a `Lit` result is the output column's single
    value. Anything that does not fold — a non-constant input, an operator the folder does not
    handle, an int64 overflow it refuses — returns None, and the column is simply unknown, as
    it was before.
    """
    from batcher.kyber.rules.normalize.fold import fold_expression
    from batcher.plan.expr_rewrite import transform_expr_up

    names = referenced_columns(expr)
    if not names:
        return None  # a column-free expression; a bare `Lit` is the caller's business
    constants: dict[str, Any] = {}
    for name in names:
        value = constant_value(child.columns.get(name))
        if value is None:
            return None  # an input that is not a proven constant → the output is not one either
        constants[name] = value

    def substitute(node: Expr) -> Expr:
        return Lit(constants[node.name]) if isinstance(node, Col) and node.name in names else node

    folded = fold_expression(transform_expr_up(expr, substitute))
    if not isinstance(folded, Lit):
        return None
    return ColumnStat(
        min=folded.value, max=folded.value, null_count=0, ndv=1, provenance=Provenance.EXACT
    )


def constant_value(stat: ColumnStat | None) -> Any | None:
    """The single value a column provably holds, or None when it is not a proven constant.

    Requires `EXACT` provenance, a present `min == max`, and no nulls. A NaN bound fails the
    `min == max` test by construction — NaN is equal to nothing, itself included — which is
    exactly right: a column of NaN is not a constant anything may be computed from.
    """
    if stat is None or stat.provenance is not Provenance.EXACT or stat.null_count != 0:
        return None
    if stat.min is None or stat.min != stat.max:
        return None
    return stat.min
