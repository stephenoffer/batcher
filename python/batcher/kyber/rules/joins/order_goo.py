"""Greedy operator ordering: the bushy fallback when the join-order DP cannot afford a graph.

The left-deep greedy builder (`order_search._rebuild_greedy`) starts from the smallest leaf and
only ever grows that one tree, so the smallest leaf decides the whole order. In a two-fact
query that is routinely the wrong choice: TPC-DS q72's smallest leaf is `warehouse` (5 rows),
whose only neighbour is `inventory` (11.7M), so the greedy order walked the whole inventory
spine -- 16.4M rows before anything selective joined -- while every selective dimension sat
on `catalog_sales`, the other fact. The DP finds the plan that joins those first, and runs the
query in ~70 ms against ~620 ms, but at nine leaves its search did not fit the budget the
query's estimated cost grants (`order_budget`), so the greedy order is what ran.

GOO (Fegaras, 1998) keeps a *forest* instead: every leaf starts as its own tree, and each step
joins the connected pair of trees whose join is cheapest. It therefore builds the selective
`catalog_sales` side and the `inventory` side independently and joins them last, which is the
bushy shape the DP picks, in O(n^2) pair evaluations per step rather than a pass over every
connected subset. `order._try_reorder` runs it beside the left-deep builder and keeps the
cheaper tree by the same cost model, so adding it can only lower the estimated cost of the
fallback.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.joins.order_residual import Residual, attach_residuals, residual_refs
from batcher.kyber.rules.joins.order_search import (
    ColRef,
    SrcRef,
    _base_leaf,
    _bits,
    _final_projection,
    _join_plans,
    _needed_cols,
)
from batcher.plan.logical import LogicalPlan

__all__ = ["rebuild_goo"]

# A tree of the forest: its plan and the (alias, logical column) pairs it carries.
_Tree = tuple[LogicalPlan, list[tuple[str, ColRef]]]


class _Unplaceable(Exception):
    """A residual predicate could not be attached to a join; the whole search is abandoned."""


def rebuild_goo(
    leaves: list[LogicalPlan],
    edges: list[tuple[ColRef, ColRef]],
    required: list[tuple[str, SrcRef]],
    ctx: OptimizerContext,
    residuals: list[Residual] | None = None,
) -> LogicalPlan | None:
    """The join tree greedy operator ordering builds, or None when it cannot build one.

    Args:
        leaves: The join region's leaf relations.
        edges: Equi-join edges between logical columns of the leaves.
        required: The region's output columns and where each comes from.
        ctx: The optimizer context, for the shared cost model.
        residuals: Hoisted predicates to attach once their columns are all joined.

    Returns:
        The rebuilt region, or None on a disconnected graph (which would need a cross join)
        or a residual that cannot be placed.
    """
    residuals = residuals or []
    needed = _needed_cols(required, edges) | residual_refs(residuals)
    cost = ctx.costs()
    forest: dict[int, _Tree] = {}
    for i, leaf in enumerate(leaves):
        schema = [(c, (i, c)) for c in leaf.available_columns() if (i, c) in needed]
        based = _base_leaf(leaf, i, schema, residuals)
        if based is None:
            return None
        forest[1 << i] = (based, schema)
    # Candidate joins between two trees, keyed by the pair of masks. A step invalidates only
    # the pairs touching the two trees it merged, so each is costed once per tree it joins.
    scored: dict[tuple[int, int], tuple[float, _Tree] | None] = {}
    try:
        return _grow(forest, scored, edges, required, residuals, cost)
    except _Unplaceable:
        return None  # never drop a residual: the region keeps its order as written


def _grow(forest, scored, edges, required, residuals, cost) -> LogicalPlan | None:
    """Join the cheapest connected pair of trees until one tree remains."""
    while len(forest) > 1:
        best: tuple[float, int, int, _Tree] | None = None
        masks = sorted(forest)
        for a_i, a in enumerate(masks):
            for b in masks[a_i + 1 :]:
                if (a, b) not in scored:
                    scored[(a, b)] = _candidate(forest[a], forest[b], a, b, edges, residuals, cost)
                entry = scored[(a, b)]
                if entry is not None and (best is None or entry[0] < best[0]):
                    best = (entry[0], a, b, entry[1])
        if best is None:
            return None  # no two trees share an edge: the graph is disconnected
        _, a, b, tree = best
        del forest[a], forest[b]
        scored = {k: v for k, v in scored.items() if a not in k and b not in k}
        forest[a | b] = tree
    ((plan, schema),) = forest.values()
    return _final_projection(plan, schema, required)


def _candidate(
    left: _Tree,
    right: _Tree,
    left_mask: int,
    right_mask: int,
    edges: list[tuple[ColRef, ColRef]],
    residuals: list[Residual],
    cost,
) -> tuple[float, _Tree] | None:
    """`(this join's own cost, the joined tree)`, or None when no edge connects the two.

    The incremental term, as `_rebuild_dphyp` scores a split: each tree's own cost is fixed
    whichever pair joins next, so only the join being added distinguishes the candidates.
    A residual that cannot be placed raises `_Unplaceable`, as the other two searches
    abandon the reorder rather than build a tree with the predicate dropped.
    """
    built = _join_plans(left[0], left[1], right[0], right[1], edges)
    if built is None:
        return None
    jplan, jschema = built
    union = left_mask | right_mask
    placed = attach_residuals(
        jplan, jschema, residuals, _bits(union), _bits(left_mask), _bits(right_mask)
    )
    if placed is None:
        raise _Unplaceable
    return cost.join_op_cost(jplan).total(), (placed, jschema)
