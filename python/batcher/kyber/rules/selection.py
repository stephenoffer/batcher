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
from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.logical import (
    Join,
    JoinOutputCol,
    LogicalPlan,
    Scan,
    Union,
)

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
SORT_MERGE_MIN_ROWS = 1_000_000.0


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
) -> tuple[LogicalPlan, list[BuildSideDecision]]:
    """Rewrite inner joins so the cheaper-to-build input is the build side.

    `cost_model` defaults to a model over the same estimator with the configured
    coefficients, so cost always drives the decision even when called outside the
    optimizer context (e.g. the adaptive re-optimization loop)."""
    decisions: list[BuildSideDecision] = []
    cost = cost_model or CostModel(estimator)
    return _rewrite(plan, estimator, cost, decisions, _broadcast_max_bytes()), decisions


def _broadcast_max_bytes() -> int:
    """Build-side broadcast threshold in bytes — `OptimizerConfig.broadcast_max_bytes`.

    The single source of truth shared with the distributed executor's runtime guard;
    a function (not an inlined read) so tests can patch the planner's threshold.
    """
    from batcher.config import active_config

    return active_config().optimizer.broadcast_max_bytes


def build_side_rule(plan: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan:
    """Cost-based join build-side selection. Needs estimated input sizes (sources),
    and records its decisions on the context for explain/telemetry."""
    if not ctx.sources:
        return plan
    plan, decisions = adaptive_build_side(plan, ctx.estimator, ctx.costs())
    ctx.notes["build_side_decisions"] = decisions
    return plan


def _rewrite(
    node: LogicalPlan,
    est: CardinalityEstimator,
    cost: CostModel,
    decisions: list,
    max_bytes: int,
) -> LogicalPlan:
    if isinstance(node, Scan):
        return node
    if isinstance(node, Union):
        return Union(
            tuple(_rewrite(i, est, cost, decisions, max_bytes) for i in node.inputs),
            node.distinct,
        )
    if isinstance(node, Join):
        left = _rewrite(node.left, est, cost, decisions, max_bytes)
        right = _rewrite(node.right, est, cost, decisions, max_bytes)
        node = Join(
            left, right, node.left_keys, node.right_keys, node.join_type, node.output, node.strategy
        )
        l_est, r_est = est.estimate(node.left), est.estimate(node.right)
        # Build-side swap is only valid for inner joins (associative/commutative).
        # Compare the cost of this orientation against the swapped one; children are
        # identical between them, so the per-join `op_cost` is the deciding term.
        cost_delta = 0.0
        swap = False
        if node.join_type == "inner":
            swapped = _swap(node)
            here = cost.op_cost(node).total()
            there = cost.op_cost(swapped).total()
            cost_delta = here - there
            swap = there < here
            if swap:
                node = swapped
        # After any swap, the right input is the build side. Broadcast it when it is
        # small enough to replicate — cheaper than shuffling the probe side. Valid
        # for every join type (the engine probes the left, replicating the right).
        build_rows = min(l_est.rows, r_est.rows) if swap else r_est.rows
        # Size the build side in bytes (rows × measured per-row width), so wide
        # payloads aren't broadcast on a misleadingly small row count.
        build_bytes = build_rows * cost.row_bytes(node.right)
        broadcast = build_bytes <= max_bytes
        if broadcast:
            node = dataclasses.replace(node, strategy="broadcast")
        elif l_est.rows >= SORT_MERGE_MIN_ROWS and r_est.rows >= SORT_MERGE_MIN_ROWS:
            # Two large inputs, neither small enough to broadcast: sort-merge avoids
            # building a hash table over a huge side (Spark's default large-join).
            # NOTE: preferring sort-merge for *already-ordered* inputs was tried and
            # reverted — benchmarking showed SMJ's RowConverter encoding overhead loses
            # to the hash join even when its sort is skipped, so it was a regression.
            # The engine's skip-sort fast-path (`bc-runtime` `sort_indices_if_unsorted`)
            # still makes this size-driven SMJ cheaper when its inputs happen to arrive
            # sorted; only the planner *preference* was withdrawn.
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
            node, input=_rewrite(node.input, est, cost, decisions, max_bytes)
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
