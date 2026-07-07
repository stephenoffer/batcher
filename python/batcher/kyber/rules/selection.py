"""SELECTION-phase rules — cost-based physical algorithm choice.

Today this is adaptive join build-side selection. The runtime hash join builds its
hash table on the *right* input and probes with the left. Building the smaller input
is faster and uses less memory, so this rule compares the **cost** of each orientation
of every inner join and swaps the sides when the swapped orientation is cheaper
(semantics-preserving for inner joins — the output columns carry the same values, just
sourced from the swapped side). The cost model folds the build-vs-probe asymmetry
(`hash_build_row` ≠ `hash_probe_row`) and the memory axis, and is calibrated from
measured `op_stats`, so this decision reflects the real engine and *learns* across
executions (its cardinalities sharpen via the MetadataHub).

`adaptive_build_side` is the pure rewrite (returning its decisions for telemetry);
`build_side_rule` is the `plan_rule` body that pulls the estimator/cost model from the
`OptimizerContext` and records its decisions on `ctx.notes`. It is registered as the
`adaptive_build_side` rule in `Phase.SELECTION` (see
`kyber.registry.register_builtin_rules`).
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.learned_tuning import (
    learned_broadcast_max_bytes,
    learned_join_strategy,
    learned_sort_merge_min_rows,
)
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.signature import plan_signature
from batcher.plan.logical import (
    Join,
    JoinOutputCol,
    LogicalPlan,
    Scan,
    Union,
)
from batcher.plan.visitor import children, with_children

__all__ = ["BuildSideDecision", "adaptive_build_side", "build_side_rule"]


# The build-side broadcast byte threshold lives in `OptimizerConfig.broadcast_max_bytes`
# (the single source of truth shared with the distributed executor's runtime guard):
# below it, replicating the build side is cheaper than shuffling the probe side
# (cf. Spark's autoBroadcastJoinThreshold). Byte-based, not row-based, so a few rows
# of wide payloads (embeddings, blobs) aren't mistakenly broadcast. The choice only
# affects data movement, never the result — a performance knob, not a correctness one.

# When neither side is broadcast-small but both exceed this, prefer a sort-merge
# join (no hash table over a huge build side). Also a performance knob — every
# strategy yields the same relation.
# Build-side row floor above which a hash table is large enough that sort-merge's
# bounded-memory merge is worth its encoding cost. Set high because hash beats SMJ
# until the build genuinely strains memory: a build of this many narrow-key rows is a
# multi-GB hash table. Below it (the common case, including selective joins whose small
# side is still > 1M rows) a hash join over the smaller side wins.
SORT_MERGE_MIN_ROWS = 50_000_000.0


@dataclasses.dataclass(frozen=True, slots=True)
class BuildSideDecision:
    left_rows: float
    right_rows: float
    swapped: bool
    provenance: str
    broadcast: bool = False
    cost_delta: float = 0.0  # cost(current) − cost(swapped); > 0 means the swap saves


def adaptive_build_side(
    plan: LogicalPlan,
    estimator: CardinalityEstimator,
    cost_model: CostModel | None = None,
    *,
    broadcast_max_bytes: int | None = None,
    sort_merge_min_rows: float | None = None,
) -> tuple[LogicalPlan, list[BuildSideDecision]]:
    """Rewrite inner joins so the cheaper-to-build input is the build side.

    `cost_model` defaults to a model over the same estimator with the configured
    coefficients, so cost always drives the decision even when called outside the
    optimizer context (e.g. the adaptive re-optimization loop).

    `broadcast_max_bytes` / `sort_merge_min_rows` override the strategy thresholds —
    `build_side_rule` passes the values *learned* from measured timings (see
    `kyber.learned_tuning`); when `None` the static config default and module floor
    stand, so every existing caller is unchanged. Thresholds only pick among
    equivalent physical algorithms, so learning them never changes the result."""
    decisions: list[BuildSideDecision] = []
    cost = cost_model or CostModel(estimator)
    max_bytes = broadcast_max_bytes if broadcast_max_bytes is not None else _broadcast_max_bytes()
    smr = sort_merge_min_rows if sort_merge_min_rows is not None else SORT_MERGE_MIN_ROWS
    return _rewrite(plan, estimator, cost, decisions, max_bytes, smr), decisions


def _broadcast_max_bytes() -> int:
    """Build-side broadcast threshold in bytes — `OptimizerConfig.broadcast_max_bytes`.

    The single source of truth shared with the distributed executor's runtime guard;
    a function (not an inlined read) so tests can patch the planner's threshold.
    """
    from batcher.config import active_config

    return active_config().optimizer.broadcast_max_bytes


def build_side_rule(plan: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan:
    """Cost-based join build-side selection. Needs estimated input sizes (sources),
    and records its decisions on the context for explain/telemetry.

    The broadcast byte threshold and the sort-merge row floor are taken from what the
    hub has *learned* from measured broadcast-vs-shuffle and hash-vs-sort-merge timings
    (falling back to the static defaults cold); then any per-signature learned
    join-strategy arm overrides the cost-model choice. All three are choices among
    equivalent physical algorithms, so the result is invariant."""
    if not ctx.sources:
        return plan
    learned_bmax = learned_broadcast_max_bytes(ctx.hub)
    max_bytes = learned_bmax if learned_bmax is not None else _broadcast_max_bytes()
    learned_smr = learned_sort_merge_min_rows(ctx.hub, SORT_MERGE_MIN_ROWS)
    smr = learned_smr if learned_smr is not None else SORT_MERGE_MIN_ROWS
    plan, decisions = adaptive_build_side(
        plan, ctx.estimator, ctx.costs(), broadcast_max_bytes=max_bytes, sort_merge_min_rows=smr
    )
    plan = _apply_learned_strategies(plan, ctx.hub)
    ctx.notes["build_side_decisions"] = decisions
    return plan


def _apply_learned_strategies(node: LogicalPlan, hub) -> LogicalPlan:
    """Override each join's strategy with the per-signature bandit arm the hub learned.

    A regret-minimizing bandit over `{hash, broadcast, sort_merge}` converges to the
    algorithm measured fastest for a join of this shape on *this* hardware, correcting a
    mis-ranked static cost guess. Every arm emits the identical relation (the engine
    falls back to hash for any it cannot honor), so this is a pure performance override.
    A cold signature yields `None` and the cost-model choice stands, so the plan is
    unchanged until there is evidence."""
    if hub is None:  # no learned store → skip the per-join signature work entirely
        return node
    node = with_children(node, [_apply_learned_strategies(c, hub) for c in children(node)])
    if isinstance(node, Join):
        arm = learned_join_strategy(hub, plan_signature(node))
        if arm is not None and arm != node.strategy:
            return dataclasses.replace(node, strategy=arm)
    return node


def _rewrite(
    node: LogicalPlan,
    est: CardinalityEstimator,
    cost: CostModel,
    decisions: list,
    max_bytes: int,
    smr: float,
) -> LogicalPlan:
    if isinstance(node, Scan):
        return node
    if isinstance(node, Union):
        return Union(
            tuple(_rewrite(i, est, cost, decisions, max_bytes, smr) for i in node.inputs),
            node.distinct,
        )
    if isinstance(node, Join):
        left = _rewrite(node.left, est, cost, decisions, max_bytes, smr)
        right = _rewrite(node.right, est, cost, decisions, max_bytes, smr)
        node = Join(
            left, right, node.left_keys, node.right_keys, node.join_type, node.output, node.strategy
        )
        l_est, r_est = est.estimate(node.left), est.estimate(node.right)
        # Build-side swap is only valid for inner joins (associative/commutative).
        # Compare the cost of this orientation against the swapped one; children are
        # identical between them, so the per-join `op_cost` is the deciding term.
        # Build-side swap is only valid for inner joins (associative/commutative).
        # Compare the cost of this orientation against the swapped one; children are
        # identical between them, so the per-join `op_cost` is the deciding term.
        # (A left/right join's `A LEFT JOIN B == B RIGHT JOIN A` rename was tried to
        # build the smaller *preserved* side, but it regressed: building the small side
        # forces probing the large one, and the scattered probe lookups cost more than
        # the larger but cache-friendlier build. The current cost model's build:probe
        # ratio mis-ranks that, so the rename is withheld until the model is calibrated.)
        cost_delta = 0.0
        swap = False
        broadcast = False
        # Size *both* sides in bytes (rows × measured per-row width) up front, so wide
        # payloads aren't broadcast on a misleadingly small row count.
        left_bytes = l_est.rows * cost.row_bytes(node.left)
        right_bytes = r_est.rows * cost.row_bytes(node.right)
        if node.join_type == "inner":
            # Broadcast-first: replicating the *smaller* side as the build and probing
            # the larger in parallel (no shuffle of the big side) is the dominant
            # strategy whenever the small side fits the byte threshold — what DuckDB /
            # Spark do. Decide it from the two sides' bytes directly, independent of the
            # marginal cpu-delta swap below: a mis-ranked cost delta must never forfeit a
            # clearly-beneficial broadcast. The old code only ever checked the *right*
            # side, so when the cost model failed to swap and the small side was the
            # left/probe, broadcast was missed and the join fell back to shuffling the
            # 6M-row build (TPC-H Q5's orders⋈lineitem: 419 ms shuffle vs 78 ms broadcast).
            if min(left_bytes, right_bytes) <= max_bytes:
                # Build the smaller side (the runtime builds the right); swap when it is
                # the left. Inner joins are associative/commutative, so this is safe.
                swap = left_bytes < right_bytes
                if swap:
                    node = _swap(node)
                broadcast = True
            else:
                # Neither side is broadcast-small: pick the cheaper build orientation by
                # cost (build the smaller of two large sides; shuffle either way).
                swapped = _swap(node)
                here = cost.op_cost(node).total()
                there = cost.op_cost(swapped).total()
                cost_delta = here - there
                swap = there < here
                if swap:
                    node = swapped
        else:
            # Non-inner joins are not commutative — the build is always the right input.
            # Broadcast it when it is small enough to replicate (the engine probes left).
            broadcast = right_bytes <= max_bytes
        # After any swap, the right input is the build side.
        build_rows = min(l_est.rows, r_est.rows) if swap else r_est.rows
        if broadcast:
            node = dataclasses.replace(node, strategy="broadcast")
        elif build_rows >= smr:
            # Sort-merge only when the *build* side (the one hashed, after the swap) is
            # itself so large that a hash table over it is memory-prohibitive — then
            # SMJ's bounded-memory merge wins despite its RowConverter encoding cost.
            # Gating on the build side (not both inputs) is the key: a hash join builds
            # only the smaller side and streams the larger one, so a 6M ⋈ 1.5M join
            # hashes 1.5M (fits easily) and beats sorting *both* 6M and 1.5M. The old
            # "both sides large" gate mis-chose SMJ for selective joins whose small side
            # was merely over a million rows (TPC-H Q18's top join: 219ms SMJ sorting 6M
            # lineitem to emit 399 rows, where a hash probe is ~30ms).
            # NOTE: preferring SMJ for *already-ordered* inputs was tried and reverted —
            # SMJ's encoding overhead loses to hash even when its sort is skipped; only
            # the genuinely-too-big-to-hash build keeps it.
            node = dataclasses.replace(node, strategy="sort_merge")
        decisions.append(
            BuildSideDecision(
                l_est.rows, r_est.rows, swap, _prov(l_est, r_est), broadcast, cost_delta
            )
        )
        return node
    # Single-input nodes: rewrite the child in place.
    if hasattr(node, "input"):
        return dataclasses.replace(
            node, input=_rewrite(node.input, est, cost, decisions, max_bytes, smr)
        )
    return node


def _swap(join: Join) -> Join:
    """Swap an inner join's sides so the old left becomes the (build) right side."""
    return Join(
        left=join.right,
        right=join.left,
        left_keys=join.right_keys,
        right_keys=join.left_keys,
        join_type="inner",
        output=tuple(JoinOutputCol(_flip(o.side), o.name, o.alias) for o in join.output),
    )


def _flip(side: str) -> str:
    return "right" if side == "left" else "left"


def _prov(l_est, r_est) -> str:
    from batcher.plan.stats import weakest

    return str(weakest(l_est.provenance, r_est.provenance))
