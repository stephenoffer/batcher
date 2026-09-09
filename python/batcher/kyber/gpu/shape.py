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
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = ["broadcast_join", "is_shardable"]


def broadcast_join(
    plan: LogicalPlan,
    sources: list[Source],
    hub: MetadataHub | None = None,
    device_bytes: float = 0.0,
    device_count: int = 0,
) -> bool:
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

    `hub` supplies the learned statistics the estimator sizes the two sides with, and it is
    the *same* hub the routing decision next door estimates against. Passing `None` here while
    the router passed a hub is how the two came to disagree: the router sized the build side
    from measured cardinalities and this sized it from footer estimates, so on any fleet that
    had learned anything the router could route a join to the fan-out that this had refused to
    call a broadcast — or, worse, the reverse, which puts a build side nobody measured onto
    every device at once. `None` is still accepted, and still means "no learned statistics".

    `device_bytes` is what one device may hold of a replicated build side
    (`DistributedConfig.device_replication_bytes`). `0.0` means the caller cannot say, and the
    CPU threshold stands — which is what this always did. `device_count` is how many devices
    would each read a copy; `0` or `1` skips the ratio test below, which has nothing to weigh.

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
        est = CardinalityEstimator(sources=sources, learned=load_learned_stats(hub))
        # The **device's** ceiling, not the CPU's. Left unset, `adaptive_build_side` resolves
        # `resolved_broadcast_max_bytes(l3_cache_bytes=0, workers=1)` — the 4 MiB fallback,
        # which is a share of one CPU's last-level cache. That is the right question for a
        # single-node CPU join and the wrong one by three orders of magnitude for a 15 GB
        # board: measured on a six-T4 fleet at TPC-H sf10, q4 and q12 have build sides of
        # roughly 240 MB, were refused the fan-out, and ran the whole join on one device —
        # 8.7 s and 8.5 s against CPU answers of 0.33 s and 1.28 s.
        #
        # `0.0` keeps the previous behaviour exactly, for a caller that cannot say what device
        # this would run on.
        ceiling = int(device_bytes) if device_bytes and device_bytes > 0 else None
        _rewritten, decisions = adaptive_build_side(joins[0], est, broadcast_max_bytes=ceiling)
        if len(decisions) != 1 or not decisions[0].broadcast or decisions[0].swapped:
            return False
        return device_count < 2 or _replicating_pays(decisions[0], device_count)
    except Exception as exc:  # pragma: no cover - routing must never break a plan
        note_suppressed("kyber", "ask whether the join broadcasts", exc)
        return False


def _walk(node, seen: set[int] | None = None):
    """Every *distinct* node of a plan, parents before children.

    Distinct by identity, because a logical plan is a DAG rather than a tree: a subtree bound
    to a variable and used twice (`d = ...; d.join(d, ...)`) is one object reachable by two
    paths. Yielding it twice made a single self-join look like two joins, so `len(joins) != 1`
    refused the fan-out for the one plan shape most obviously worth fanning out — and on a
    deep shared subtree the duplicate traversal is exponential rather than merely wrong.
    """
    seen = set() if seen is None else seen
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for attr in ("input", "left", "right"):
        child = getattr(node, attr, None)
        if child is not None:
            yield from _walk(child, seen)
    for child in getattr(node, "inputs", ()) or ():
        yield from _walk(child, seen)


def is_shardable(plan: LogicalPlan) -> bool:
    """Whether `plan` divides across devices, so its per-device memory is one shard's.

    Three shapes do: one with a mergeable reducer, whose shards fold; a row-local one, whose
    shards concatenate; and a **join tree** with a splittable leaf, whose fan-out splits that
    leaf and replicates the rest. Answered from the plan's own IR through the shared algebra in
    `plan.distribution` rather than re-derived here — the optimizer routing a plan to the
    fan-out and the backend building it must agree about which plans divide, and two statements
    of that rule are the one way they could ever disagree.

    The third shape was missing, and it was not a small omission: `flatten_ops` cannot flatten a
    branch, so **every join plan answered False** and the router sized it for a single device.
    Measured on a six-T4 fleet at TPC-H sf10, that ran a 60 M x 15 M join on one board — q4 and
    q12 at 8.7 s and 8.5 s against CPU-engine answers of 0.33 s and 1.28 s, five devices idle.

    Never raises: a plan that cannot be lowered (a `map_batches` UDF) simply is not shardable.
    """
    from batcher._internal.logging import note_suppressed
    from batcher.plan.distribution import ir_divides

    try:
        return ir_divides(plan.to_ir())
    except Exception as exc:  # pragma: no cover - routing must never break a plan
        note_suppressed("kyber", "test the plan for a mergeable reducer", exc)
        return False


def _replicating_pays(decision, device_count: int) -> bool:
    """Whether splitting the probe side buys anything worth replicating the build side for.

    "Does it fit" and "is it worth it" are different questions, and a byte ceiling only answers
    the first. Per device, a broadcast fan-out does `build + probe/N` where a single device does
    `build + probe` — so it is never slower, and what it *gains* is the probe side it divides.
    When the probe is small next to the build, it divides almost nothing while every device
    reads a whole copy: N devices do N times the work of one and finish at the same time.

    TPC-H q4 at sf10 is that in its clearest form. `orders SEMI lineitem` has a build side of
    30 M rows and a probe of 573 K, so replicating gives each of six T4s the whole thirty
    million rows to answer a query whose entire probe side is half a million. Measured: **23.1
    s**, against 0.35 s for the CPU engine and 11.4 s for the same join on one device — six
    devices spent to reproduce one device's time.

    The rule is therefore that the fan-out must **divide more than it replicates**: the probe
    side has to be the larger of the two. Rows stand in for bytes because that is what the
    estimator carries, and the comparison only needs to be right by an order of magnitude — the
    losing cases miss it by three.

    Deliberately *not* the stronger `probe > build x devices`. That is the right rule for
    aggregate fleet-seconds and the wrong one for wall time, because the devices read
    concurrently: measured on the same fleet, it refused q14 (0.48 GB replicated against a
    1.68 GB probe) and gave up a **12.4x** speedup to save bytes nobody was waiting on.

    Args:
        decision: The build-side decision `adaptive_build_side` returned.
        device_count: Devices that would each read a copy.

    Returns:
        True when splitting the probe divides more than replicating the build costs, and
        whenever the sizes cannot be compared — the ceiling has already established the build
        side fits, and refusing on an unreadable estimate would put the join back on one device.
    """
    build_rows = float(decision.right_rows)
    probe_rows = float(decision.left_rows)
    if build_rows <= 0 or probe_rows <= 0:
        return True
    del device_count  # the comparison is per device, and both terms scale with it
    return probe_rows > build_rows
