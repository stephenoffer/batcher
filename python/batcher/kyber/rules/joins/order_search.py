"""Join-order search: pick a tree over an extracted join graph, and build it.

The second half of the JOIN_REORDER rule. `order.py` reads a *join graph* out of a plan —
leaves, equi-join edges between logical columns, and the output the region must reproduce —
and this module answers the question that graph poses: which shape of tree is cheapest, and
what does that tree look like as `LogicalPlan` nodes.

Three searches, tried widest-first by `order._try_reorder`: an exhaustive subset DP up to
`_MAX_EXHAUSTIVE_LEAVES`, a connected-subset (DPhyp-style) DP within a per-query search
budget (`order_budget`) up to `_MAX_DP_LEAVES`, and a
greedy size-minimizing builder beyond. All three return the *same* logical relation and are
interchangeable; they differ only in how much of the search space they can afford to read.

Split from `order.py` on the seam between reading a graph and searching it: the two halves
share only the graph vocabulary below, and keeping them in one module put it past the size
limit with no room to explain either half.
"""

from __future__ import annotations

from itertools import combinations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.joins import order_budget
from batcher.kyber.rules.joins.order_residual import Residual, attach_residuals, residual_refs
from batcher.plan.expr_ir import Col, Lit
from batcher.plan.logical import Join, JoinOutputCol, LogicalPlan, Project, Projection

__all__ = ["ColRef", "SrcRef"]

# A logical column: the (leaf index, originating column name) it ultimately comes
# from. Two physical columns with the same logical column hold the same values.
ColRef = tuple[int, str]

# Where a required output column originates: a leaf column, or a synthetic constant
# with no leaf at all (a comma join's `__cross_key`), reproduced from its literal.
SrcRef = ColRef | Lit


def _needed_cols(
    required: list[tuple[str, SrcRef]], edges: list[tuple[ColRef, ColRef]]
) -> set[ColRef]:
    """Logical columns a rebuilt subtree must carry: the required output plus every
    join-key endpoint (so projection-pushed-away columns are not re-introduced).

    A constant-valued output carries nothing — it has no leaf to be read from and is
    re-emitted as its literal by `_final_projection`."""
    needed: set[ColRef] = {ref for _, ref in required if not isinstance(ref, Lit)}
    for a, b in edges:
        needed.add(a)
        needed.add(b)
    return needed


def _bits(mask: int) -> frozenset[int]:
    """The leaf indices a `_rebuild_dphyp` bitmask stands for."""
    return frozenset(i for i in range(mask.bit_length()) if mask >> i & 1)


def _base_leaf(
    leaf: LogicalPlan,
    idx: int,
    schema: list[tuple[str, ColRef]],
    residuals: list[Residual],
) -> LogicalPlan | None:
    """A leaf with any residual that reads only *this* leaf already applied.

    Almost every hoisted predicate spans two leaves — that is why it was hoisted — but one
    can read a single leaf and still have been sitting above a join, and `attach_residuals`
    only ever fires at a join. Applying single-leaf residuals here is what makes "exactly
    once along every root-to-leaf path" true rather than nearly true: without it such a
    predicate is covered by both halves of the first join above it, skipped as
    already-applied, and silently lost from the query.
    """
    return attach_residuals(leaf, schema, residuals, frozenset({idx}), frozenset(), frozenset())


def _rebuild_greedy(
    leaves: list[LogicalPlan],
    edges: list[tuple[ColRef, ColRef]],
    required: list[tuple[str, SrcRef]],
    ctx: OptimizerContext,
    residuals: list[Residual] | None = None,
) -> LogicalPlan | None:
    est = ctx.estimator
    cost = ctx.costs()
    n = len(leaves)
    # Carry only the columns that are actually used: those in the final required
    # output plus those that appear as a join key (an edge endpoint). Carrying
    # truly-unused columns would re-introduce them into join output lists after
    # projection pushdown already pruned them from the scans, leaving the join
    # output referencing a column its (pruned) input no longer provides.
    residuals = residuals or []
    needed = _needed_cols(required, edges) | residual_refs(residuals)

    # Start from the smallest leaf, then repeatedly add the connected leaf that
    # yields the smallest estimated intermediate result.
    sizes = [est.estimate(leaf).rows for leaf in leaves]
    start = min(range(n), key=lambda i: sizes[i])

    # current schema: list of (alias in `current`, logical column) — needed cols only.
    schema: list[tuple[str, ColRef]] = [
        (c, (start, c)) for c in leaves[start].available_columns() if (start, c) in needed
    ]
    current = _base_leaf(leaves[start], start, schema, residuals)
    if current is None:
        return None
    joined = {start}

    while len(joined) < n:
        best: tuple[float, int, LogicalPlan, list[tuple[str, ColRef]]] | None = None
        for j in range(n):
            if j in joined:
                continue
            built = _make_join(current, schema, leaves[j], j, edges, needed)
            if built is None:
                continue  # not connected to the joined set yet
            cand_join, cand_schema = built
            with_res = attach_residuals(
                cand_join,
                cand_schema,
                residuals,
                frozenset(joined | {j}),
                frozenset(joined),
                frozenset({j}),
            )
            if with_res is None:
                return None  # a residual could not be placed — never drop it (see below)
            # Rank by estimated *cost*, not raw output rows: cost folds the build/probe
            # asymmetry, so a calibrated model can prefer an order that pure
            # row-minimization gets wrong. Score the same **incremental** term the DP
            # recurrence uses — the new leaf's subtree plus this join's own cost, at the
            # orientation SELECTION will pick. The accumulated cost of `current` is common
            # to every candidate, so including it would not change the argmin; it would
            # only re-walk the whole subtree once per candidate per step.
            score = cost.cost(leaves[j]).total() + cost.join_op_cost(cand_join).total()
            if best is None or score < best[0]:
                best = (score, j, with_res, cand_schema)
        if best is None:
            return None  # disconnected graph → would be a cross join; skip reorder
        _, j, current, schema = best
        joined.add(j)

    return _final_projection(current, schema, required)


# The connected-subset DP (`_rebuild_dphyp`) runs up to `_MAX_DP_LEAVES` leaves; greedy
# beyond. How much *work* it may do inside that reach is not a constant -- it is the
# per-query budget `order_budget.search_pair_budget` derives from what the query is estimated
# to cost, because leaf count does not predict search work (a 14-leaf chain evaluates 455
# pairs and a 14-leaf star 53,248) and neither predicts what a better order is worth. The flat
# 200,000-pair cap this replaces was a bound on neither: it permitted ~30 s of planning for a
# query that might run in a millisecond. `_MAX_DP_LEAVES` survives as a *memory* guard -- the
# DP holds a plan per connected subset -- rather than as a proxy for time.
#
# `_MAX_EXHAUSTIVE_LEAVES` no longer selects a search -- it bounds how far `order.py` will
# merge join regions to hoist a residual filter, which is a plan-quality decision (see the
# comment there) rather than a cost-of-search one.
_MAX_EXHAUSTIVE_LEAVES = 12
_MAX_DP_LEAVES = 20


def _rebuild_dp(
    leaves: list[LogicalPlan],
    edges: list[tuple[ColRef, ColRef]],
    required: list[tuple[str, SrcRef]],
    ctx: OptimizerContext,
    residuals: list[Residual] | None = None,
) -> LogicalPlan | None:
    """Cost-optimal join order by exhaustive DP over *every* leaf subset.

    **This is the test oracle, not a production path.** It is the obviously-correct
    statement of the recurrence -- enumerate all 2ⁿ subsets and all 3ⁿ splits, keep the
    cheapest -- against which `_rebuild_dphyp` is checked on 120 random graphs by
    `tests/unit/test_dphyp_join_order.py`. It enumerates disconnected subsets too, and
    only learns they are disconnected by failing to build a join for each of their
    splits, so on a sparse graph it is exponentially slower than the DP that replaced it
    on the live path. `order.py` calls `_rebuild_dphyp` at every size now. Keep this one:
    an oracle that shared the optimization under test would not be an oracle.

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
    residuals = residuals or []
    needed = _needed_cols(required, edges) | residual_refs(residuals)
    cost = ctx.costs()

    # best[subset] = (plan, schema, accumulated_cost). Base case: each singleton leaf.
    best: dict[frozenset[int], tuple[LogicalPlan, list[tuple[str, ColRef]], float]] = {}
    for i, leaf in enumerate(leaves):
        schema = [(c, (i, c)) for c in leaf.available_columns() if (i, c) in needed]
        based = _base_leaf(leaf, i, schema, residuals)
        if based is None:
            return None
        best[frozenset({i})] = (based, schema, 0.0)

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
                filtered = attach_residuals(jplan, jschema, residuals, subset, s1, s2)
                if filtered is None:
                    return None  # a residual could not be placed; abandon the rewrite
                # Add only *this join's* operator cost to the two halves' already-
                # accumulated costs (`cost.cost(jplan)` would re-walk and double-count the
                # children that `left[2]`/`right[2]` already paid for — which penalizes
                # deep subtrees super-linearly and can flip the optimum to a plan with a
                # huge many-to-many intermediate). This is the standard additive DP
                # recurrence: cost(S) = cost(S1) + cost(S2) + op_cost(join).
                total = left[2] + right[2] + cost.join_op_cost(jplan).total()
                if chosen is None or total < chosen[2]:
                    chosen = (filtered, jschema, total)
            if chosen is not None:
                best[subset] = chosen

    full = best.get(frozenset(range(n)))
    if full is None:
        return None  # disconnected graph → would be a cross join; skip reorder
    return _final_projection(full[0], full[1], required)


def _rebuild_dphyp(
    leaves: list[LogicalPlan],
    edges: list[tuple[ColRef, ColRef]],
    required: list[tuple[str, SrcRef]],
    ctx: OptimizerContext,
    residuals: list[Residual] | None = None,
    budget: int | None = None,
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
    # `None` means "search as hard as anything is ever allowed to" -- what the oracle test
    # wants, and what a caller with no cost model in hand gets. A real query always arrives
    # with a budget priced from its own estimated cost.
    if budget is None:
        budget = order_budget.max_pairs()
    residuals = residuals or []
    needed = _needed_cols(required, edges) | residual_refs(residuals)
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
                    if order_budget.predicted_pairs(len(connected)) > budget:
                        # Too dense to search within budget. Bailing *here* -- before a single
                        # cardinality estimate or cost -- is the point of predicting from the
                        # subset count: the alternative is to spend the whole budget
                        # discovering it and hand the answer to greedy anyway.
                        return None
        frontier = nxt

    # dp[mask] = (plan, schema, accumulated_cost); base case = each singleton leaf.
    dp: dict[int, tuple[LogicalPlan, list[tuple[str, ColRef]], float]] = {}
    for i, leaf in enumerate(leaves):
        schema = [(c, (i, c)) for c in leaf.available_columns() if (i, c) in needed]
        based = _base_leaf(leaf, i, schema, residuals)
        if based is None:
            return None
        dp[1 << i] = (based, schema, 0.0)

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
                    if pairs > budget:
                        # The exact guarantee behind the prediction above, for a graph whose
                        # pairs-per-subset ratio runs above the measured band.
                        return None
                    built = _join_plans(left[0], left[1], right[0], right[1], edges)
                    if built is not None:
                        jplan, jschema = built
                        filtered = attach_residuals(
                            jplan, jschema, residuals, _bits(subset), _bits(s1), _bits(s2)
                        )
                        if filtered is None:
                            return None  # a residual could not be placed; abandon it
                        # Just this join's op cost; the halves already carry their own
                        # (see `_rebuild_dp` — `cost.cost` would double-count children).
                        total = left[2] + right[2] + cost.join_op_cost(jplan).total()
                        if chosen is None or total < chosen[2]:
                            chosen = (filtered, jschema, total)
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
    left_keys, right_keys = _crossing_keys(left_alias, right_alias, edges)
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


def _crossing_keys(
    left_alias: dict[ColRef, str],
    right_alias: dict[ColRef, str],
    edges: list[tuple[ColRef, ColRef]],
) -> tuple[list[str], list[str]]:
    """The key pairs joining two sub-plans: one per equality the two sides do not already imply.

    Every edge with both endpoints inside one side was applied when that side was built --
    each builder here joins on all the edges its two halves share, so by induction a sub-plan
    already enforces every equality among its own leaves. Equality is transitive, so a crossing
    edge whose endpoints are *already* connected, through edges inside either side plus the
    crossing edges kept so far, adds no condition a row could fail: it is implied, exactly. Only
    a spanning forest of the crossing edges is kept.

    The redundant pairs were not free. JOB writes every pairwise equality among the tables
    joined on a movie, so the fourth table to join met its partner on four copies of one key:
    the hash table hashed and compared four columns where one decides the match, and the
    estimator read those four columns as a composite key whose distinct count combines four
    copies of one value domain. That combination saturates the side's row count, which the
    composite primary-key test reads as a unique key, and the join was priced at the larger
    input's size -- `hash_join [inner on id, id, id, id, id]` estimated at 160,809 rows against
    743 on q27c. With the implied pairs gone the same join is a single-key join, priced by the
    single-key containment estimate and its unique-key cap.

    Inner equi-joins only (the region `order.py` extracts): a NULL key matches nothing, so every
    row that survives an applied edge holds equal, non-null values on both of its endpoints, and
    the transitive step is sound for every row the join can see.

    Args:
        left_alias: Each logical column the left sub-plan carries, to its name there.
        right_alias: The same for the right sub-plan.
        edges: The region's equi-join edges.

    Returns:
        The left and right key names, in edge order, with implied pairs dropped. Empty when
        no edge crosses between the two sides.
    """
    parent: dict[ColRef, ColRef] = {}

    def find(ref: ColRef) -> ColRef:
        while parent.get(ref, ref) != ref:
            ref = parent[ref]
        return ref

    def union(a: ColRef, b: ColRef) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    for a, b in edges:
        if (a in left_alias and b in left_alias) or (a in right_alias and b in right_alias):
            union(a, b)  # already applied inside one side
    left_keys: list[str] = []
    right_keys: list[str] = []
    for a, b in edges:
        if a in left_alias and b in right_alias:
            lref, rref = a, b
        elif b in left_alias and a in right_alias:
            lref, rref = b, a
        else:
            continue
        if union(lref, rref):
            left_keys.append(left_alias[lref])
            right_keys.append(right_alias[rref])
    return left_keys, right_keys


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
    left_keys, right_keys = _crossing_keys(alias_of, {(leaf_idx, c): c for c in leaf_cols}, edges)
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
    required: list[tuple[str, SrcRef]],
) -> LogicalPlan | None:
    """Select the original output columns (by logical identity) in original order.

    A constant-valued output is re-emitted as its literal — it holds the same value in
    every row the region produces, so the rebuilt tree need not carry it through the
    joins to reproduce it exactly."""
    alias_of = {ref: alias for alias, ref in schema}
    items: list[Projection] = []
    for out_alias, ref in required:
        if isinstance(ref, Lit):
            items.append(Projection(out_alias, ref))
            continue
        src = alias_of.get(ref)
        if src is None:
            return None  # a required column wasn't carried (unexpected) → skip
        items.append(Projection(out_alias, Col(src)))
    return Project(current, tuple(items))
