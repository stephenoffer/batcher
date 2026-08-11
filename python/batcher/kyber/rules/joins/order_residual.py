"""Non-equi predicates inside a join region: hoist them, then re-attach where they fit.

A `Filter` between two joins is a wall to join reordering. `order._collect_leaves` stops at
anything that is not an inner `Join`, so everything below such a filter becomes a single
opaque leaf and no alternative order for it is ever considered — the pair of tables under it
is frozen exactly as the SQL was written.

That is not a rare shape, because **the optimizer creates it**. Predicate pushdown runs in
an earlier phase than join reordering, and a predicate over two relations (`a.x < b.y`) is
pushed down to the lowest join whose output has both columns. So the pass that improves the
plan is the same pass that decides, irrevocably, which two tables get joined first.

TPC-DS **q72** is the case in point. `inv_quantity_on_hand < cs_quantity` reads `inventory`
and `catalog_sales`, so pushdown parks it directly on `inventory ⋈ catalog_sales`, and the
reorderer then sees one leaf worth 1.18 **billion** rows that it cannot take apart. The
order that query wants — join each fact to its date dimension first, then join the two on
`(item_sk, d_week_seq)` — was never a candidate.

This module makes such a predicate a *property of a leaf subset* rather than a wall.
`order.py` hoists it out on the way down, recording which leaves it reads; the search in
`order_search.py` re-attaches it above the first join whose subset covers those leaves. That
is exactly once along any path from the leaves to the root, so the rebuilt region computes
the same relation — and the estimator now sees the filter at the place the plan will apply
it, so its selectivity is priced into every candidate order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from batcher.plan.expr_ir import Expr, referenced_columns, remap_columns
from batcher.plan.logical import Filter, Join, LogicalPlan, Project

__all__ = [
    "Residual",
    "attach_residuals",
    "bind_residuals",
    "hoistable_filter",
    "residual_refs",
]

# A logical column: `(leaf index, originating column name)`. Duplicated from
# `order_search.ColRef` by *value*, not imported, because importing it here and the
# residual types there would close a cycle between the two halves of the rule.
_ColRef = tuple[int, str]


@dataclass(frozen=True)
class Residual:
    """One hoisted predicate, and the leaves it needs before it can be applied.

    `by_name` maps each column name the predicate reads to the logical column it resolves
    to, so the search can rewrite the predicate into whatever aliases the subplan it lands
    on happens to carry. `leaves` is the set of leaf indices those columns come from — the
    subset that must be joined before the predicate is evaluable.
    """

    predicate: Expr
    by_name: dict[str, _ColRef]
    leaves: frozenset[int]


def hoistable_filter(node: LogicalPlan, is_transparent: Callable[[LogicalPlan], bool]) -> bool:
    """Whether `node` is a `Filter` sitting *between* joins, and so part of the region.

    A filter directly above a leaf is that leaf's own predicate — pushdown put it there and
    it belongs there, since evaluating it earlier is always at least as good. Only a filter
    whose input continues the inner-join region is a wall worth removing, and only those are
    hoisted, so a leaf's own selectivity keeps being estimated as part of the leaf.

    Args:
        node: The plan node to classify.
        is_transparent: `order._is_transparent` — whether a `Project` only selects and
            renames, which is the same seam the leaf walk sees through. Passed in rather
            than imported so the two halves of the rule do not import each other.

    Returns:
        True when `node` is a hoistable region filter.
    """
    if not isinstance(node, Filter):
        return False
    if not referenced_columns(node.predicate):
        return False  # a constant predicate reads no leaf; leave it where it is
    return _continues_region(node.input, is_transparent)


def _continues_region(node: LogicalPlan, is_transparent: Callable[[LogicalPlan], bool]) -> bool:
    """Whether `node` is (or leads to) another inner join of the same region."""
    if isinstance(node, Join):
        return node.join_type == "inner"
    if isinstance(node, Project):
        return is_transparent(node) and _continues_region(node.input, is_transparent)
    if isinstance(node, Filter):
        return _continues_region(node.input, is_transparent)
    return False


def bind_residuals(
    hoisted: list[tuple[Expr, LogicalPlan]],
    resolve,
    index: dict[int, int],
) -> list[Residual] | None:
    """Resolve each hoisted predicate's columns to leaf columns, or `None` to decline.

    Args:
        hoisted: Each hoisted predicate paired with the node that was directly below its
            filter — the plan whose output names the predicate is phrased in.
        resolve: `order._resolve`, tracing a column name down to its originating leaf.
        index: Leaf object id to leaf position, as `order._try_reorder` built it.

    Returns:
        One `Residual` per hoisted predicate, or `None` when any column fails to resolve to
        a leaf of this region — in which case reordering must not proceed, because
        re-attaching a predicate whose provenance is unknown could evaluate it in the wrong
        place.
    """
    out: list[Residual] = []
    for predicate, below in hoisted:
        by_name: dict[str, _ColRef] = {}
        for name in referenced_columns(predicate):
            resolved = resolve(below, name)
            if resolved is None or id(resolved[0]) not in index:
                return None
            by_name[name] = (index[id(resolved[0])], resolved[1])
        out.append(Residual(predicate, by_name, frozenset(ref[0] for ref in by_name.values())))
    return out


def residual_refs(residuals: list[Residual]) -> set[_ColRef]:
    """Every logical column the residual predicates read.

    These must be carried through the rebuilt joins alongside the required output and the
    join keys, or the predicate lands on a subplan that no longer has its inputs.
    """
    return {ref for r in residuals for ref in r.by_name.values()}


def attach_residuals(
    plan: LogicalPlan,
    schema: list[tuple[str, _ColRef]],
    residuals: list[Residual],
    subset: frozenset[int],
    left: frozenset[int],
    right: frozenset[int],
) -> LogicalPlan | None:
    """Wrap `plan` in the residual predicates that this join makes evaluable, if any.

    A residual applies here when its leaves are covered by `subset` but by *neither* half —
    if a half covered them, that half's plan already applied it. Along any root-to-leaf path
    there is therefore exactly one node where each residual attaches, which is what makes
    the rebuilt region compute the original relation.

    Args:
        plan: The join just built for `subset`.
        schema: `plan`'s carried columns as `(alias, logical column)`.
        residuals: Every residual of the region.
        subset: The leaves `plan` covers.
        left: The leaves the left half covers.
        right: The leaves the right half covers.

    Returns:
        `plan`, wrapped in one `Filter` per newly-evaluable residual (unwrapped when none),
        or `None` if a predicate's column was not carried this far. `None` is the one answer
        that must never be replaced by a best effort: skipping the predicate would return
        *more rows than the query asked for*, silently and with every test green, so the
        caller abandons the candidate instead. `residual_refs` feeds the carried-column set
        precisely so this cannot happen.
    """
    alias_of = {ref: alias for alias, ref in schema}
    for r in residuals:
        if not r.leaves <= subset or r.leaves <= left or r.leaves <= right:
            continue
        mapping = {name: alias_of[ref] for name, ref in r.by_name.items() if ref in alias_of}
        if len(mapping) != len(r.by_name):
            return None
        plan = Filter(plan, remap_columns(r.predicate, mapping))
    return plan
