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
"""

from __future__ import annotations

from itertools import combinations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, RuleCategory, plan_rule
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Join,
    JoinOutputCol,
    LogicalPlan,
    Project,
    Projection,
    is_cartesian_key_pair,
)
from batcher.plan.visitor import children, with_children

__all__ = ["reorder_joins"]

# A logical column: the (leaf index, originating column name) it ultimately comes
# from. Two physical columns with the same logical column hold the same values.
ColRef = tuple[int, str]


def _needed_cols(
    required: list[tuple[str, ColRef]], edges: list[tuple[ColRef, ColRef]]
) -> set[ColRef]:
    """Logical columns a rebuilt subtree must carry: the required output plus every
    join-key endpoint (so projection-pushed-away columns are not re-introduced)."""
    needed: set[ColRef] = {ref for _, ref in required}
    for a, b in edges:
        needed.add(a)
        needed.add(b)
    return needed


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
    _collect_leaves(top, leaves)
    if len(leaves) < 3:
        return None  # two-way: leave it to build-side selection
    # Leaves must be distinct objects so identity indexing is unambiguous.
    index: dict[int, int] = {}
    for i, leaf in enumerate(leaves):
        if id(leaf) in index:
            return None
        index[id(leaf)] = i

    edges = _collect_edges(top, index)
    required = _required_output(top, index)
    if edges is None or required is None:
        return None

    # Reorder nested subtrees inside each leaf first, then prune each leaf to just the
    # columns the rebuilt subtree needs (its keys + the required output). Seeing through
    # transparent projections (above) discards the column-pruning projects the builder
    # placed between joins, so without re-pruning here the rebuilt joins would read
    # full-width leaves — re-materializing every dropped column (large strings, blobs)
    # the projection pushdown had already eliminated. Projection pushdown runs before
    # this phase and does not run again, so reorder must carry the pruning itself.
    needed = _needed_cols(required, edges)
    leaves = [_prune_leaf(visit(leaf), i, needed) for i, leaf in enumerate(leaves)]
    # Bushy-tree DP: exhaustive up to `_MAX_EXHAUSTIVE_LEAVES`, connected-subset DP
    # for larger sparse graphs, greedy fallback.
    if len(leaves) <= _MAX_EXHAUSTIVE_LEAVES:
        dp = _rebuild_dp(leaves, edges, required, ctx)
    else:
        dp = _rebuild_dphyp(leaves, edges, required, ctx)
    return dp if dp is not None else _rebuild_greedy(leaves, edges, required, ctx)


def _is_transparent(node: LogicalPlan) -> bool:
    """Whether `node` is a pure pass-through projection (every output is a bare column).

    Such a `Project` only renames/selects columns — it computes nothing — so an
    inner-join subtree split across one is still a single reorderable subtree. The
    join builder emits these between joins (e.g. the `.drop(__cross_key)` after each
    comma join), and without seeing through them every join looks two-leaved and
    reordering never engages. Column provenance is followed through the renames, and
    reorder rebuilds the columns it needs from the leaves directly, so dropping the
    intermediate projection is safe.
    """
    return isinstance(node, Project) and all(isinstance(it.expr, Col) for it in node.items)


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


def _collect_leaves(node: LogicalPlan, out: list[LogicalPlan]) -> None:
    if isinstance(node, Join) and node.join_type == "inner":
        _collect_leaves(node.left, out)
        _collect_leaves(node.right, out)
    elif _is_transparent(node):
        _collect_leaves(node.input, out)
    else:
        out.append(node)


def _resolve(node: LogicalPlan, colname: str) -> tuple[LogicalPlan, str] | None:
    """Trace `colname` in `node`'s output down to the leaf (subplan, column) it
    originates from, following inner-join output provenance and transparent renames."""
    if isinstance(node, Join) and node.join_type == "inner":
        for o in node.output:
            if o.alias == colname:
                child = node.left if o.side == "left" else node.right
                return _resolve(child, o.name)
        return None  # column not found in this join's output (unexpected)
    if _is_transparent(node):
        for it in node.items:
            if it.alias == colname:
                return _resolve(node.input, it.expr.name)
        return None  # column not produced by this projection (unexpected)
    return (node, colname)


def _collect_edges(node: LogicalPlan, index: dict[int, int]) -> list[tuple[ColRef, ColRef]] | None:
    """Equi-join edges between logical columns, gathered over the whole subtree."""
    if _is_transparent(node):
        return _collect_edges(node.input, index)
    if not (isinstance(node, Join) and node.join_type == "inner"):
        return []
    left = _collect_edges(node.left, index)
    right = _collect_edges(node.right, index)
    if left is None or right is None:
        return None
    edges = left + right
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        la = _resolve(node.left, lk)
        ra = _resolve(node.right, rk)
        if la is None or ra is None:
            return None
        # A cartesian pseudo-key (the `__cross_key` a comma/cross join lowers to) is the
        # same constant on both sides — it connects nothing. Skipping it keeps the join
        # graph honest, so reordering reflects real connectivity and never builds a cross
        # product across two relations that only the pseudo-key "joined".
        if is_cartesian_key_pair(la[0], la[1], ra[0], ra[1]):
            continue
        edges.append(((index[id(la[0])], la[1]), (index[id(ra[0])], ra[1])))
    return edges


def _required_output(top: Join, index: dict[int, int]) -> list[tuple[str, ColRef]] | None:
    """The original output as (alias, logical column), preserving order."""
    out: list[tuple[str, ColRef]] = []
    for o in top.output:
        resolved = _resolve(top, o.alias)
        if resolved is None:
            return None
        out.append((o.alias, (index[id(resolved[0])], resolved[1])))
    return out


def _rebuild_greedy(
    leaves: list[LogicalPlan],
    edges: list[tuple[ColRef, ColRef]],
    required: list[tuple[str, ColRef]],
    ctx: OptimizerContext,
) -> LogicalPlan | None:
    est = ctx.estimator
    cost = ctx.costs()
    n = len(leaves)
    # Carry only the columns that are actually used: those in the final required
    # output plus those that appear as a join key (an edge endpoint). Carrying
    # truly-unused columns would re-introduce them into join output lists after
    # projection pushdown already pruned them from the scans, leaving the join
    # output referencing a column its (pruned) input no longer provides.
    needed = _needed_cols(required, edges)

    # Start from the smallest leaf, then repeatedly add the connected leaf that
    # yields the smallest estimated intermediate result.
    sizes = [est.estimate(leaf).rows for leaf in leaves]
    start = min(range(n), key=lambda i: sizes[i])

    current = leaves[start]
    # current schema: list of (alias in `current`, logical column) — needed cols only.
    schema: list[tuple[str, ColRef]] = [
        (c, (start, c)) for c in leaves[start].available_columns() if (start, c) in needed
    ]
    joined = {start}

    while len(joined) < n:
        best: tuple[float, int, Join, list[tuple[str, ColRef]]] | None = None
        for j in range(n):
            if j in joined:
                continue
            built = _make_join(current, schema, leaves[j], j, edges, needed)
            if built is None:
                continue  # not connected to the joined set yet
            cand_join, cand_schema = built
            # Rank by estimated *cost*, not raw output rows: cost folds the
            # build/probe asymmetry and accumulated work, so a calibrated model can
            # prefer an order that pure row-minimization gets wrong.
            score = cost.cost(cand_join).total()
            if best is None or score < best[0]:
                best = (score, j, cand_join, cand_schema)
        if best is None:
            return None  # disconnected graph → would be a cross join; skip reorder
        _, j, current, schema = best  # type: ignore[assignment]
        joined.add(j)

    return _final_projection(current, schema, required)


# Exhaustive O(3ⁿ) subset DP up to `_MAX_EXHAUSTIVE_LEAVES`; the connected-subset DP
# (`_rebuild_dphyp`) up to `_MAX_DP_LEAVES`; greedy beyond. `_MAX_DP_PAIRS` caps the
# connected-subset DP's work so a dense large graph bails to greedy (small-query
# mandate) instead of blowing up. Keeping the exhaustive DP for the small case leaves
# its plans unchanged.
_MAX_EXHAUSTIVE_LEAVES = 12
_MAX_DP_LEAVES = 20
_MAX_DP_PAIRS = 200_000


def _rebuild_dp(
    leaves: list[LogicalPlan],
    edges: list[tuple[ColRef, ColRef]],
    required: list[tuple[str, ColRef]],
    ctx: OptimizerContext,
) -> LogicalPlan | None:
    """Cost-optimal join order via DP over connected leaf subsets (DPccp-style).

    For each subset of leaves, keep the minimum-cost sub-plan; a subset's plan is
    the cheapest join of two of its sub-partitions that share an edge. Unlike the
    greedy left-deep builder this considers **bushy** trees (e.g. two fact tables
    each pre-joined to a dimension), which win on star/snowflake schemas. Returns
    `None` to defer to greedy when there are too many leaves or the graph is
    disconnected (a cross join, which this rule never introduces).
    """
    n = len(leaves)
    if n > _MAX_EXHAUSTIVE_LEAVES:
        return None
    needed = _needed_cols(required, edges)
    cost = ctx.costs()

    # best[subset] = (plan, schema, accumulated_cost). Base case: each singleton leaf.
    best: dict[frozenset[int], tuple[LogicalPlan, list[tuple[str, ColRef]], float]] = {}
    for i, leaf in enumerate(leaves):
        schema = [(c, (i, c)) for c in leaf.available_columns() if (i, c) in needed]
        best[frozenset({i})] = (leaf, schema, 0.0)

    for size in range(2, n + 1):
        for subset_t in combinations(range(n), size):
            subset = frozenset(subset_t)
            chosen: tuple[LogicalPlan, list[tuple[str, ColRef]], float] | None = None
            for s1, s2 in _splits(subset):
                left = best.get(s1)
                right = best.get(s2)
                if left is None or right is None:
                    continue
                built = _join_plans(left[0], left[1], right[0], right[1], edges)
                if built is None:
                    continue  # the two sides share no edge under this split
                jplan, jschema = built
                # Add only *this join's* operator cost to the two halves' already-
                # accumulated costs (`cost.cost(jplan)` would re-walk and double-count the
                # children that `left[2]`/`right[2]` already paid for — which penalizes
                # deep subtrees super-linearly and can flip the optimum to a plan with a
                # huge many-to-many intermediate). This is the standard additive DP
                # recurrence: cost(S) = cost(S1) + cost(S2) + op_cost(join).
                total = left[2] + right[2] + cost.op_cost(jplan).total()
                if chosen is None or total < chosen[2]:
                    chosen = (jplan, jschema, total)
            if chosen is not None:
                best[subset] = chosen

    full = best.get(frozenset(range(n)))
    if full is None:
        return None  # disconnected graph → would be a cross join; skip reorder
    return _final_projection(full[0], full[1], required)


def _rebuild_dphyp(
    leaves: list[LogicalPlan],
    edges: list[tuple[ColRef, ColRef]],
    required: list[tuple[str, ColRef]],
    ctx: OptimizerContext,
) -> LogicalPlan | None:
    """Cost-optimal bushy join order over **connected subgraphs only**, by size.

    Where the exhaustive `_rebuild_dp` keeps a plan for *every* subset (O(3ⁿ), capping
    near 12 leaves), this enumerates only the graph's connected subsets (a sparse
    star/snowflake/chain has far fewer than 2ⁿ) smallest-first, so both halves of any
    split are already final — the identical global optimum (the oracle test pins this)
    at more tables. Bails to greedy (`None`) on a dense/too-large or disconnected graph.
    """
    n = len(leaves)
    if n > _MAX_DP_LEAVES:
        return None
    needed = _needed_cols(required, edges)
    cost = ctx.costs()

    # Adjacency between leaf indices as bitmasks (edge endpoints carry their leaf id).
    adj = [0] * n
    for a, b in edges:
        i, j = a[0], b[0]
        if i != j:
            adj[i] |= 1 << j
            adj[j] |= 1 << i

    def neighbors(mask: int) -> int:
        nb = 0
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            nb |= adj[v]
            m &= m - 1
        return nb & ~mask

    # All connected subsets, grown from singletons one neighbor at a time (so a
    # sparse graph stays far under 2ⁿ — disconnected subsets are never created).
    connected: set[int] = {1 << i for i in range(n)}
    frontier = list(connected)
    while frontier:
        nxt: list[int] = []
        for s in frontier:
            ext = neighbors(s)
            while ext:
                bit = ext & -ext
                ext &= ext - 1
                t = s | bit
                if t not in connected:
                    connected.add(t)
                    nxt.append(t)
                    if len(connected) > _MAX_DP_PAIRS:
                        return None  # too many connected subsets → defer to greedy
        frontier = nxt

    # dp[mask] = (plan, schema, accumulated_cost); base case = each singleton leaf.
    dp: dict[int, tuple[LogicalPlan, list[tuple[str, ColRef]], float]] = {}
    for i, leaf in enumerate(leaves):
        schema = [(c, (i, c)) for c in leaf.available_columns() if (i, c) in needed]
        dp[1 << i] = (leaf, schema, 0.0)

    # Smallest-first so both halves of every split are already final. For each subset
    # the cheapest split into two connected halves wins, min-element on the left to
    # match the exhaustive DP's single orientation.
    pairs = 0
    for subset in sorted(connected, key=int.bit_count):
        if subset.bit_count() < 2:
            continue
        pivot = subset & -subset  # lowest set bit = the subset's min element
        rest = subset & ~pivot
        chosen: tuple[LogicalPlan, list[tuple[str, ColRef]], float] | None = None
        # Every submask of `rest` including empty (so `s1 = pivot` alone is tried) and
        # excluding `rest` itself (`s2` empty); the left half always carries the pivot.
        sub = rest
        while True:
            s2 = rest & ~sub
            if s2 != 0:
                s1 = pivot | sub
                left = dp.get(s1)
                right = dp.get(s2)
                if left is not None and right is not None:  # both halves connected
                    pairs += 1
                    if pairs > _MAX_DP_PAIRS:
                        return None
                    built = _join_plans(left[0], left[1], right[0], right[1], edges)
                    if built is not None:
                        jplan, jschema = built
                        # Just this join's op cost; the halves already carry their own
                        # (see `_rebuild_dp` — `cost.cost` would double-count children).
                        total = left[2] + right[2] + cost.op_cost(jplan).total()
                        if chosen is None or total < chosen[2]:
                            chosen = (jplan, jschema, total)
            if sub == 0:
                break
            sub = (sub - 1) & rest
        if chosen is not None:
            dp[subset] = chosen

    full = dp.get((1 << n) - 1)
    if full is None:
        return None  # disconnected graph → would be a cross join; skip reorder
    return _final_projection(full[0], full[1], required)


def _splits(subset: frozenset[int]):
    """Yield each unordered partition of `subset` into two non-empty parts once, by
    pinning the smallest element to the left part (so `(s1, s2)` and `(s2, s1)` are
    not both emitted — build-side orientation is the build-side rule's job)."""
    elems = sorted(subset)
    pivot, rest = elems[0], elems[1:]
    for r in range(len(rest)):
        for combo in combinations(rest, r):
            s1 = frozenset((pivot, *combo))
            s2 = subset - s1
            if s2:
                yield s1, s2


def _join_plans(
    left: LogicalPlan,
    left_schema: list[tuple[str, ColRef]],
    right: LogicalPlan,
    right_schema: list[tuple[str, ColRef]],
    edges: list[tuple[ColRef, ColRef]],
) -> tuple[Join, list[tuple[str, ColRef]]] | None:
    """Build `left ⋈ right` from two sub-plans (each carrying its needed columns),
    or `None` when no edge connects them. The bushy generalization of `_make_join`."""
    left_alias = {ref: alias for alias, ref in left_schema}
    right_alias = {ref: alias for alias, ref in right_schema}

    left_keys: list[str] = []
    right_keys: list[str] = []
    seen: set[tuple[str, str]] = set()
    for a, b in edges:
        if a in left_alias and b in right_alias:
            pair = (left_alias[a], right_alias[b])
        elif b in left_alias and a in right_alias:
            pair = (left_alias[b], right_alias[a])
        else:
            continue
        if pair not in seen:
            seen.add(pair)
            left_keys.append(pair[0])
            right_keys.append(pair[1])
    if not left_keys:
        return None

    output: list[JoinOutputCol] = []
    new_schema: list[tuple[str, ColRef]] = []
    used: set[str] = set()
    for alias, ref in left_schema:
        output.append(JoinOutputCol("left", alias, alias))
        used.add(alias)
        new_schema.append((alias, ref))
    for alias, ref in right_schema:
        out_alias = alias
        while out_alias in used:
            out_alias = f"{out_alias}_r"
        output.append(JoinOutputCol("right", alias, out_alias))
        used.add(out_alias)
        new_schema.append((out_alias, ref))
    join = Join(left, right, tuple(left_keys), tuple(right_keys), "inner", tuple(output))
    return join, new_schema


def _make_join(
    current: LogicalPlan,
    schema: list[tuple[str, ColRef]],
    leaf: LogicalPlan,
    leaf_idx: int,
    edges: list[tuple[ColRef, ColRef]],
    needed: set[ColRef],
) -> tuple[Join, list[tuple[str, ColRef]]] | None:
    """Build `current ⋈ leaf`, carrying only `needed` columns, or None if the leaf
    is not connected to the already-joined set."""
    alias_of = {ref: alias for alias, ref in schema}
    leaf_cols = leaf.available_columns()
    leaf_refs = {(leaf_idx, c) for c in leaf_cols}

    left_keys: list[str] = []
    right_keys: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for a, b in edges:
        # Orient the edge so one endpoint is in `current` and the other in `leaf`.
        if a in alias_of and b in leaf_refs:
            pair = (alias_of[a], b[1])
        elif b in alias_of and a in leaf_refs:
            pair = (alias_of[b], a[1])
        else:
            continue
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            left_keys.append(pair[0])
            right_keys.append(pair[1])
    if not left_keys:
        return None  # no join condition connects them

    output: list[JoinOutputCol] = []
    new_schema: list[tuple[str, ColRef]] = []
    used: set[str] = set()
    for alias, ref in schema:
        output.append(JoinOutputCol("left", alias, alias))
        used.add(alias)
        new_schema.append((alias, ref))
    for c in leaf_cols:
        if (leaf_idx, c) not in needed:
            continue  # carry only used columns (see `needed` in _rebuild_greedy)
        alias = c
        while alias in used:
            alias = f"{alias}_r"
        output.append(JoinOutputCol("right", c, alias))
        used.add(alias)
        new_schema.append((alias, (leaf_idx, c)))

    join = Join(current, leaf, tuple(left_keys), tuple(right_keys), "inner", tuple(output))
    return join, new_schema


def _final_projection(
    current: LogicalPlan,
    schema: list[tuple[str, ColRef]],
    required: list[tuple[str, ColRef]],
) -> LogicalPlan | None:
    """Select the original output columns (by logical identity) in original order."""
    alias_of = {ref: alias for alias, ref in schema}
    items: list[Projection] = []
    for out_alias, ref in required:
        src = alias_of.get(ref)
        if src is None:
            return None  # a required column wasn't carried (unexpected) → skip
        items.append(Projection(out_alias, Col(src)))
    return Project(current, tuple(items))


DEFAULT_REGISTRY.add(
    plan_rule(
        "join_reorder",
        Phase.JOIN_REORDER,
        reorder_joins,
        matches=(Join,),
        category=RuleCategory.SELECTION,
    )
)
