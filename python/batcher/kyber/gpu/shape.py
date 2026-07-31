"""What Kyber can tell the GPU backend about a plan's *shape*, as opposed to its cost.

`policy` answers "should this run on a device, and how many" from estimated size. These answer
a different kind of question: whether the plan's structure admits a fan-out at all, and whether
its join is one the planner would replicate. Both are read by the routing decision next door
and by `dist.gpu` when it builds the fan-out, so they are kept apart from the cost model that
consumes them.

Both are asked of the plan rather than read off it. The GPU backend is offered a plan *before*
the optimizer runs, so a join's `strategy` field is still whatever the plan builder put there;
reading it found `hash` on every join and the join fan-out never ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = ["broadcast_join", "is_shardable"]


def broadcast_join(plan: LogicalPlan, sources: list[Source]) -> bool:
    """Whether Kyber would run this plan's join by replicating its build side.

    The GPU backend splits a join's probe side across devices and gives every device the whole
    build side, which is only worth doing — and only fits — when the build side is small. That
    is a cost decision Kyber already makes, through the same `adaptive_build_side` the CPU join
    path uses; asking it here means the two backends cannot disagree about which joins are
    broadcast, and a disagreement is an out-of-memory on every device at once.

    It has to be *asked* rather than read off the plan, because the GPU backend is offered the
    plan before the optimizer runs, so the join's `strategy` is still whatever the plan builder
    put there. Reading it found `hash` on every join and the fan-out never ran.

    A decision that also **swaps** the join's sides reports False. The swap is correct and the
    fan-out could honor it, but the probe side would then be the plan's right input, and a
    fan-out that split the wrong side would be wrong rather than slow.

    Never raises: an unanswerable *question* is answered "no", and the join runs on one device.
    That tolerance covers an estimator that cannot size the inputs — not a moved symbol, which
    is why the imports sit outside it. Swallowing one of those is how this fan-out was
    unreachable in the first place.
    """
    from batcher._internal.logging import note_suppressed
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.rules.selection import adaptive_build_side
    from batcher.plan.logical import Join

    try:
        joins = [n for n in _walk(plan) if isinstance(n, Join)]
        if len(joins) != 1:
            return False
        est = CardinalityEstimator(sources=sources, learned=load_learned_stats(None))
        _rewritten, decisions = adaptive_build_side(joins[0], est)
        return len(decisions) == 1 and decisions[0].broadcast and not decisions[0].swapped
    except Exception as exc:  # pragma: no cover - routing must never break a plan
        note_suppressed("kyber", "ask whether the join broadcasts", exc)
        return False


def _walk(node):
    """Every node of a plan, parents before children."""
    yield node
    for attr in ("input", "left", "right"):
        child = getattr(node, attr, None)
        if child is not None:
            yield from _walk(child)
    for child in getattr(node, "inputs", ()) or ():
        yield from _walk(child)


def is_shardable(plan: LogicalPlan) -> bool:
    """Whether `plan` divides across devices, so its per-device memory is one shard's.

    Two shapes do: one with a mergeable reducer, whose shards fold, and a row-local one, whose
    shards concatenate. Answered from the plan's own IR through the shared algebra in
    `plan.distribution` rather than re-derived here — the optimizer routing a plan to the
    fan-out and the backend building it must agree about which plans divide, and two statements
    of that rule are the one way they could ever disagree.

    Never raises: a plan that cannot be lowered (a `map_batches` UDF) simply is not shardable.
    """
    from batcher._internal.logging import note_suppressed
    from batcher.plan.distribution import flatten_ops, shard_plan

    try:
        ops = flatten_ops(plan.to_ir())
        return ops is not None and shard_plan(ops) is not None
    except Exception as exc:  # pragma: no cover - routing must never break a plan
        note_suppressed("kyber", "test the plan for a mergeable reducer", exc)
        return False
