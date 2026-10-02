"""An upper bound on a group-by's output from where its keys come from, not what they hold.

`combine_ndv` estimates the distinct combinations of a key set from the keys' own distinct
counts, capped by the aggregate's input rows. When the keys are dimension attributes reached
through a join, both halves of that are weak: the damped combination of several dimension
columns lands in the millions, and the input it is capped by is the join's own estimate. On
TPC-DS q22 (`GROUP BY i_product_name, i_brand, i_class, i_category` over inventory joined to
item) that estimate was **2,030,411** groups against **8,999** actual.

The keys say more than their counts do: every one of them is a column of `item`, which has
18,000 rows at that scale. Distinct combinations of columns drawn from one relation cannot
outnumber that relation's rows, however many times a join repeats each row. So the bound is
the smallest row estimate on the path from the aggregate's input down to the relation the
keys come from, following them through filters, renaming projections and join sides.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.plan.expr_ir import Col
from batcher.plan.logical import Filter, Join, LogicalPlan, Project

__all__ = ["key_origin_rows"]


def key_origin_rows(
    plan: LogicalPlan, keys: list[str], rows: Callable[[LogicalPlan], float]
) -> float | None:
    """The fewest rows of any relation every one of `keys` is drawn from, or None.

    Descends while all the keys can be followed together: through a `Filter` (a subset of the
    rows), a `Project` that renames them (the same values), and a `Join` whose output takes
    every key from one side (each output combination is a combination of that side, and the
    null-supplied side of an outer join adds at most the one all-null combination). It stops
    at anything else -- a computed key, keys split across both sides, an aggregate -- because
    past that point the bound would no longer be about these keys.

    Args:
        plan: The aggregate's input.
        keys: The group-key column names, as `plan` outputs them.
        rows: The estimator's row estimate for a subtree.

    Returns:
        An upper bound on the distinct combinations of `keys`; None when `keys` is empty.
    """
    if not keys:
        return None
    bound = rows(plan)
    node, names, extra = plan, list(keys), 0.0
    while True:
        if isinstance(node, Filter):
            node = node.input
        elif isinstance(node, Project):
            sources = {item.alias: item.expr for item in node.items}
            renamed = [sources.get(n) for n in names]
            if not all(isinstance(e, Col) for e in renamed):
                break
            names = [e.name for e in renamed if isinstance(e, Col)]
            node = node.input
        elif isinstance(node, Join):
            step = _join_side(node, names)
            if step is None:
                break
            node, names, null_extended = step
            extra += 1.0 if null_extended else 0.0
        else:
            break
        bound = min(bound, rows(node) + extra)
    return bound


def _join_side(join: Join, names: list[str]) -> tuple[LogicalPlan, list[str], bool] | None:
    """The side of `join` all `names` come from, their names there, and if it is null-supplied."""
    outputs = {o.alias: o for o in join.output}
    picked = [outputs.get(n) for n in names]
    if any(o is None for o in picked):
        return None
    sides = {o.side for o in picked if o is not None}
    if len(sides) != 1:
        return None
    side = sides.pop()
    null_extended = (side == "right" and join.join_type in ("left", "full")) or (
        side == "left" and join.join_type in ("right", "full")
    )
    child = join.left if side == "left" else join.right
    return child, [o.name for o in picked if o is not None], null_extended
