"""Cost-based join reordering — the JOIN_REORDER phase.

A multi-table inner-join query is built by the API as a fixed (usually left-deep)
tree, but inner equi-joins are associative and commutative, so *any* order produces
the same result. Order matters enormously for cost: joining the most selective
relations first keeps intermediate results small. This rule extracts a maximal
connected subtree of inner joins, costs candidate orders with the shared
`CardinalityEstimator` (which learns across executions), and rebuilds the subtree
in a greedy size-minimizing order.

Correctness is guaranteed structurally rather than by replicating the join's column
bookkeeping:

  1. Every key and output column is traced back to the leaf relation it originates
     from (following `JoinOutputCol` provenance), giving a language of *logical
     columns* `(leaf, column)` and a graph of equi-join edges between them.
  2. The reordered tree is rebuilt carrying **all** columns of both sides at every
     step (suffixing only on name collisions) — no coalescing or key-dropping to
     get wrong.
  3. A final `Project` selects exactly the original output columns, by logical
     identity, in the original order — so the output schema and values are
     identical to the original join no matter how the interior was reshaped.

Only inner joins are reordered (outer joins are neither associative nor commutative
in general); non-inner joins and other operators are treated as opaque leaves. The
rule engages only for ≥3 connected leaves (two-way is the build-side rule's job) and
never produces a cross join (a disconnected graph is left untouched). Any
unexpected shape causes a safe no-op.

This module reads the graph; `order_search` searches it and builds the chosen tree.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, RuleCategory, plan_rule
from batcher.kyber.rules.joins.order_residual import (
    bind_residuals,
    hoistable_filter,
    residual_refs,
)
from batcher.kyber.rules.joins.order_search import (
    _MAX_EXHAUSTIVE_LEAVES,
    ColRef,
    SrcRef,
    _needed_cols,
    _rebuild_dp,
    _rebuild_dphyp,
    _rebuild_greedy,
)
from batcher.plan.expr_ir import Col, Expr, Lit
from batcher.plan.logical import (
    Join,
    LogicalPlan,
    Project,
    Projection,
    constant_column_literal,
    is_cartesian_key_pair,
)
from batcher.plan.visitor import children, with_children

__all__ = ["reorder_joins"]


def reorder_joins(plan: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan:
    """Reorder maximal inner-join subtrees by estimated cost (top-down, once)."""
    if not ctx.sources:  # estimation needs source sizes; nothing to cost without them
        return plan

    def visit(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Join) and node.join_type == "inner":
            reordered = _try_reorder(node, ctx, visit)
            if reordered is not None:
                return reordered
        return with_children(node, [visit(c) for c in children(node)])

    return visit(plan)


def _try_reorder(top: Join, ctx: OptimizerContext, visit) -> LogicalPlan | None:
    """Reorder the maximal inner-join subtree rooted at `top`, or None to skip."""
    leaves: list[LogicalPlan] = []
    hoisted: list[tuple[Expr, LogicalPlan]] | None = []
    _collect_leaves(top, leaves, hoisted)
    if hoisted and len(leaves) > _MAX_EXHAUSTIVE_LEAVES:
        # Hoisting a region filter merges the regions it separated, and a merged region can
        # be far wider than either half: TPC-DS q64's two 9-leaf halves become one 18-leaf
        # graph. Past the exhaustive DP's reach the search is a *heuristic* over a space it
        # cannot read, and on q64 the plan it chose ran 130 s against the 30 ms the walled-off
        # regions produced. So the merge is taken only when the exhaustive DP can still cost
        # it; beyond that the filter stays where pushdown put it and the plan is exactly the
        # one this rule produced before residuals existed.
        leaves, hoisted = [], None
        _collect_leaves(top, leaves, hoisted)
    if len(leaves) < 3:
        return None  # two-way: leave it to build-side selection
    # Leaves must be distinct objects so identity indexing is unambiguous.
    index: dict[int, int] = {}
    for i, leaf in enumerate(leaves):
        if id(leaf) in index:
            return None
        index[id(leaf)] = i

    hoist = hoisted is not None
    edges = _collect_edges(top, index, hoist)
    required = _required_output(top, index, hoist)
    residuals = bind_residuals(hoisted or [], lambda n, c: _resolve(n, c, hoist), index)
    if edges is None or required is None or residuals is None:
        return None

    # Reorder nested subtrees inside each leaf first, then prune each leaf to just the
    # columns the rebuilt subtree needs (its keys + the required output). Seeing through
    # transparent projections (above) discards the column-pruning projects the builder
    # placed between joins, so without re-pruning here the rebuilt joins would read
    # full-width leaves — re-materializing every dropped column (large strings, blobs)
    # the projection pushdown had already eliminated. Projection pushdown runs before
    # this phase and does not run again, so reorder must carry the pruning itself.
    needed = _needed_cols(required, edges) | residual_refs(residuals)
    leaves = [_prune_leaf(visit(leaf), i, needed) for i, leaf in enumerate(leaves)]
    # Bushy-tree DP: exhaustive up to `_MAX_EXHAUSTIVE_LEAVES`, connected-subset DP
    # for larger sparse graphs, greedy fallback.
    if len(leaves) <= _MAX_EXHAUSTIVE_LEAVES:
        dp = _rebuild_dp(leaves, edges, required, ctx, residuals)
    else:
        dp = _rebuild_dphyp(leaves, edges, required, ctx, residuals)
    return dp if dp is not None else _rebuild_greedy(leaves, edges, required, ctx, residuals)


def _is_transparent(node: LogicalPlan) -> bool:
    """Whether `node` is a pass-through projection: bare columns and synthetic constants.

    Such a `Project` only renames/selects columns and tags on constants — it computes
    nothing *from the data* — so an inner-join subtree split across one is still a single
    reorderable subtree. The join builder emits these between joins, and without seeing
    through them every join looks two-leaved and reordering never engages. Column
    provenance is followed through the renames, and reorder rebuilds the columns it needs
    from the leaves directly, so dropping the intermediate projection is safe.

    **The `Lit` arm is what makes a comma join reorderable at all**, and its absence was a
    process-killer rather than a missed optimization. A cross join lowers to
    ``with_columns(__cross_key=lit(1)).join(..., on=__cross_key).drop(__cross_key)``, so
    the projection *above* each cross join is bare columns (transparent) while the one
    *below* it binds a literal — and treating that as opaque walled off everything under
    it. A ``FROM a, b, c, ...`` chain was therefore never one region: each cross join was
    seen alone, as two leaves with no edge between them, which is below the three-leaf
    floor and a disconnected graph besides. Reordering declined every time, so a table
    whose join partner appears later in the `FROM` list kept the cartesian product the
    left-deep lowering gave it. JOB q7c is the case in point: `info_type` (11 rows, whose
    only edge is to `person_info`) sat beside a 36 M-row `cast_info ⋈ aka_name`, and the
    400 M-row product materialized until the kernel killed the process.
    """
    return isinstance(node, Project) and all(isinstance(it.expr, (Col, Lit)) for it in node.items)


def _prune_leaf(leaf: LogicalPlan, leaf_idx: int, needed: set[ColRef]) -> LogicalPlan:
    """Project `leaf` down to just the columns the rebuilt subtree needs from it.

    Reordering carries only `needed` columns through the joins, but the *leaf inputs*
    are otherwise read at full width (every scan column), re-materializing the columns
    projection pushdown already pruned. Wrapping the leaf in a select-only projection
    restores that pruning so a reordered join reads no more than an un-reordered one.
    A no-op when the leaf already exposes exactly the needed columns.
    """
    keep = [c for c in leaf.available_columns() if (leaf_idx, c) in needed]
    if len(keep) == len(leaf.available_columns()):
        return leaf
    return Project(leaf, tuple(Projection(c, Col(c)) for c in keep))


def _collect_leaves(
    node: LogicalPlan, out: list[LogicalPlan], hoisted: list[tuple[Expr, LogicalPlan]] | None
) -> None:
    """Walk the region's leaves, lifting out any `Filter` that sits between two joins.

    Such a filter is a predicate over several leaves — pushdown put it at the lowest join
    whose output has all its columns — and leaving it in place makes everything under it one
    opaque leaf, freezing that join. Recording it in `hoisted` (with the node below it, whose
    names it is phrased in) lets the search re-attach it where it fits; see `order_residual`.
    """
    if isinstance(node, Join) and node.join_type == "inner":
        _collect_leaves(node.left, out, hoisted)
        _collect_leaves(node.right, out, hoisted)
    elif _is_transparent(node):
        _collect_leaves(node.input, out, hoisted)
    elif hoisted is not None and hoistable_filter(node, _is_transparent):
        hoisted.append((node.predicate, node.input))
        _collect_leaves(node.input, out, hoisted)
    else:
        out.append(node)


def _resolve(node: LogicalPlan, colname: str, hoist: bool = True) -> tuple[LogicalPlan, str] | None:
    """Trace `colname` in `node`'s output down to the leaf (subplan, column) it
    originates from, following inner-join output provenance and transparent renames.

    None when the trace runs out — including at a transparent projection's *constant*
    item, which originates in no leaf at all. Callers handle that case by value
    (`_required_output` re-emits the literal) rather than by tracing it."""
    if isinstance(node, Join) and node.join_type == "inner":
        for o in node.output:
            if o.alias == colname:
                child = node.left if o.side == "left" else node.right
                return _resolve(child, o.name, hoist)
        return None  # column not found in this join's output (unexpected)
    if _is_transparent(node):
        for it in node.items:
            if it.alias == colname:
                if not isinstance(it.expr, Col):
                    return None  # a synthetic constant — no originating leaf
                return _resolve(node.input, it.expr.name, hoist)
        return None  # column not produced by this projection (unexpected)
    if hoist and hoistable_filter(node, _is_transparent):
        # A hoisted filter is not a leaf boundary and renames nothing, so provenance
        # passes straight through it — the same view `_collect_leaves` takes.
        return _resolve(node.input, colname, hoist)
    return (node, colname)


def _collect_edges(
    node: LogicalPlan, index: dict[int, int], hoist: bool = True
) -> list[tuple[ColRef, ColRef]] | None:
    """Equi-join edges between logical columns, gathered over the whole subtree."""
    if _is_transparent(node):
        return _collect_edges(node.input, index, hoist)
    if hoist and hoistable_filter(node, _is_transparent):
        return _collect_edges(node.input, index, hoist)
    if not (isinstance(node, Join) and node.join_type == "inner"):
        return []
    left = _collect_edges(node.left, index, hoist)
    right = _collect_edges(node.right, index, hoist)
    if left is None or right is None:
        return None
    edges = left + right
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        # A cartesian pseudo-key (the `__cross_key` a comma/cross join lowers to) is the
        # same constant on both sides — it connects nothing. Skipping it keeps the join
        # graph honest, so reordering reflects real connectivity and never builds a cross
        # product across two relations that only the pseudo-key "joined". Asked *before*
        # resolving, because the pseudo-key is exactly the column that resolves to no leaf
        # — tracing it first would abandon the whole subtree over a non-edge.
        if is_cartesian_key_pair(node.left, lk, node.right, rk):
            continue
        la = _resolve(node.left, lk, hoist)
        ra = _resolve(node.right, rk, hoist)
        if la is None or ra is None:
            return None
        edges.append(((index[id(la[0])], la[1]), (index[id(ra[0])], ra[1])))
    return edges


def _required_output(
    top: Join, index: dict[int, int], hoist: bool = True
) -> list[tuple[str, SrcRef]] | None:
    """The original output as (alias, source), preserving order.

    A column that traces to no leaf is kept only when it is *provably* a constant — the
    `__cross_key` a comma join carries in its output until the projection above drops it.
    Reproducing it from its literal is what lets the region be rebuilt without it: there
    is no leaf holding it, and refusing the whole reorder over a column that is the same
    value in every row would forfeit the reordering the cross join most needs.
    """
    out: list[tuple[str, SrcRef]] = []
    for o in top.output:
        resolved = _resolve(top, o.alias, hoist)
        if resolved is not None:
            out.append((o.alias, (index[id(resolved[0])], resolved[1])))
            continue
        const = constant_column_literal(top, o.alias)
        if const is None:
            return None
        out.append((o.alias, const))
    return out


DEFAULT_REGISTRY.add(
    plan_rule(
        "join_reorder",
        Phase.JOIN_REORDER,
        reorder_joins,
        matches=(Join,),
        category=RuleCategory.SELECTION,
    )
)
