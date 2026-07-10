"""Conductor-level adaptive-tuning decisions — activate the learned choices, close the loops.

`api` is the only layer allowed to import `kyber`, `carbonite`, `core`, `metadata`, `dist`,
`config` and `plan` together, so it is where the learned decisions each subsystem *exposed but
left dormant* get ACTIVATED, and where the measured outcomes get RECORDED back so the learning
actually happens. Kyber ships the read side of a decision (e.g. ``learned_join_strategy``); this
module supplies the missing write side (``record_join_strategy`` after the join runs) and the
few conductor-only decisions (learned worker fan-out, learned distribute size, the learned spill
codec / partition seed).

Everything here is **best-effort** (a malformed hub, a cold store, or any read/write error yields
the default and never raises into the hot path) and **result-invariant**: it tunes performance
and scheduling only — a partition count only shards, a credit window only buffers, a compression
codec is lossless, a strategy arm picks a semantically-equivalent algorithm, a distribute / worker
choice only decides *where* the identical result is produced. A first run over a cold hub is
byte-for-byte the pre-tuning behavior.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import TYPE_CHECKING

from batcher.config import active_config, config_context
from batcher.plan.logical import Aggregate, Distinct, Join, Sort, Union, Window

if TYPE_CHECKING:
    from batcher.core import ExecutionContext
    from batcher.io.source import Source
    from batcher.kyber.rules.selection import BuildSideDecision
    from batcher.metadata.hub import MetadataHub
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan
    from batcher.plan.resource import SchedulingEnvelope

__all__ = [
    "auto_num_partitions",
    "distributed_grant",
    "learned_num_workers",
    "learned_output_rows",
    "learned_partition_seed",
    "record_distributed",
    "record_join_outcomes",
    "record_run_feedback",
    "record_shuffle_outcome",
    "spill_compression_scope",
    "total_source_rows",
]

# A stateful pipeline breaker whose measured output volume is worth learning for spill /
# shuffle partition sizing (and whose presence means partition sizing even applies).
_BREAKERS = (Aggregate, Sort, Distinct, Window, Join, Union)


# --- learned reads (conductor decisions) -----------------------------------------------------
def learned_output_rows(hub: MetadataHub | None, plan: LogicalPlan) -> float | None:
    """The measured output-row size learned for `plan`'s signature, or `None` cold.

    Reads the same ``kyber.stats`` feedback ``record_execution`` writes, so a recurring query's
    routing decision can start from what it *actually* produced last time rather than a first-run
    estimate. Size-only — it steers a where-to-run choice, never a result.
    """
    if hub is None:
        return None
    try:
        from batcher.kyber.learned_tuning import learned_signature_rows
        from batcher.kyber.signature import plan_signature

        return learned_signature_rows(hub, plan_signature(plan))
    except Exception:  # pragma: no cover - learning must never break routing
        return None


def learned_num_workers(
    hub: MetadataHub | None, plan: LogicalPlan, sources: list[Source], cluster_nodes: int
) -> int | None:
    """Worker fan-out sized from the learned/estimated data volume, or `None` cold.

    Targets ``target_rows_per_task`` rows per worker from the measured size this shape produced
    (falling back to the sources' row counts), clamped to ``[1, cluster_nodes]`` so it never
    over-asks the cluster. Returns ``None`` when nothing is known, so the caller keeps its own
    default. Worker count only shards the work — the mergeable algebra makes the result identical
    for any count.
    """
    if hub is None or cluster_nodes <= 0:
        return None
    try:
        rows = learned_output_rows(hub, plan)
        if rows is None:
            rows = total_source_rows(sources)
        if rows is None or rows <= 0:
            return None
        target = active_config().optimizer.target_rows_per_task
        if target <= 0:
            return None
        return min(max(1, math.ceil(rows / target)), cluster_nodes)
    except Exception:  # pragma: no cover - sizing must never break a query
        return None


def auto_num_partitions(plan: LogicalPlan, sources: list[Source], hub: MetadataHub | None) -> int:
    """Data-sized spill/shuffle partition count for `plan` (used when the user gives none).

    Prefers the count implied by the *measured* rows this breaker last shuffled (``learned_
    partition_seed``); else estimates the plan's input cardinality with Kyber's estimator and
    targets ~``target_rows_per_task`` rows (and ``target_bytes_per_task`` bytes) per partition —
    the same sizing rule Kyber uses for breaker parallelism. Falls back to ``DEFAULT_PARTITIONS``
    when the size is unknown. A partition count only shards — the merged result is identical.
    """
    from batcher.api.orchestration import DEFAULT_PARTITIONS, _clamp_partitions
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator

    seed = learned_partition_seed(hub, plan)
    if seed is not None:
        return _clamp_partitions(seed)
    try:
        learned = load_learned_stats(hub) if hub is not None else None
        est = CardinalityEstimator(sources=sources, learned=learned)
        rows = est.estimate(plan).rows
        opt = active_config().optimizer
        target = opt.target_rows_per_task
        if rows <= 0 or target <= 0:
            return DEFAULT_PARTITIONS
        row_parts = math.ceil(rows / target)
        # Also shard by bytes so a few wide rows (GB blobs/embeddings) don't land a huge
        # partition on one task: take the larger of the row- and byte-derived counts.
        width = est.row_width(plan, opt.row_bytes)
        byte_parts = math.ceil(rows * width / max(1, opt.target_bytes_per_task))
        return _clamp_partitions(max(row_parts, byte_parts))
    except Exception:  # pragma: no cover - sizing must never break a query
        return DEFAULT_PARTITIONS


def learned_partition_seed(hub: MetadataHub | None, plan: LogicalPlan) -> int | None:
    """Spill/shuffle partition count implied by the measured rows this breaker last shuffled.

    Seeds ``auto_num_partitions`` from the *measured* volume (recorded by ``record_run_feedback``)
    so a recurring stage shards to fit memory on its first re-run instead of from a cold estimate.
    A partition count only shards — any value yields the identical result. `None` cold.
    """
    if hub is None:
        return None
    try:
        from batcher.kyber.learned_tuning import learned_partition_count
        from batcher.kyber.signature import plan_signature

        target = active_config().optimizer.target_rows_per_task
        return learned_partition_count(hub, plan_signature(plan), target)
    except Exception:  # pragma: no cover - sizing must never break a query
        return None


def spill_compression_scope(rm, opt: PhysicalPlan):
    """Config scope forcing the learned spill codec for `opt`, or a no-op when unlearned.

    Carbonite's ``recommend_spill_compression`` reads the LEARNED peak memory: a large, IO-bound
    out-of-core state compresses (trade CPU for less disk/network), a small one does not. IPC
    self-describes its codec, so the un-spilled result is byte-identical whichever way this falls —
    a pure throughput lever. `None` (un-sized plan) keeps the configured default.
    """
    try:
        compress = rm.recommend_spill_compression(opt)
    except Exception:  # pragma: no cover - a learned read must never break spilling
        return contextlib.nullcontext()
    if compress is None:
        return contextlib.nullcontext()
    cfg = active_config()
    codec = "zstd" if compress else None
    if cfg.memory.spill_compression == codec:
        return contextlib.nullcontext()
    memory = dataclasses.replace(cfg.memory, spill_compression=codec)
    return config_context(dataclasses.replace(cfg, memory=memory))


def distributed_grant(
    rm, opt: PhysicalPlan, plan: LogicalPlan, sources: list[Source], ctx: ExecutionContext
) -> tuple[int | None, SchedulingEnvelope]:
    """Resolve the distributed run's worker fan-out and shuffle-credit envelope from learning.

    Sizes ``num_workers`` from the measured data volume (when the user gave none) and warm-starts
    the shuffle credit window from the window this signature converged on last time. Both are pure
    scheduling levers — AIMD still governs the window actually used and the mergeable algebra keeps
    the result identical for any worker count — so a cold hub reproduces the default grant exactly.
    """
    from batcher import dist
    from batcher.kyber.signature import plan_signature

    workers = ctx.num_workers
    if workers is None:
        try:
            nodes = int(dist.cluster_topology().get("nodes", 0))
        except Exception:  # pragma: no cover - topology probe must never break a query
            nodes = 0
        workers = learned_num_workers(ctx.hub, plan, sources, nodes)
    envelope = rm.scheduling_envelope(opt, workers)
    max_credits = max((op.bounds.c_max_credits for op in opt.ops), default=0)
    if max_credits > 0:
        window = rm.grant_credits(max_credits, signature=plan_signature(plan))
        envelope = dataclasses.replace(envelope, credits=window)
    return workers, envelope


# --- feedback writes (close the learning loops) ----------------------------------------------
def record_run_feedback(
    hub: MetadataHub | None,
    plan: LogicalPlan,
    logical_opt: LogicalPlan,
    decisions: list[BuildSideDecision],
    *,
    out_rows: int,
    input_rows: int | None,
    wall_ms: float,
) -> None:
    """Fold one relational run's measured outcomes into the learned-decision stores.

    Closes three loops the read side already consumes: the breaker's shuffled volume
    (``learned_partition_count`` → spill/shuffle fan-out), a group-by's cardinality reduction
    (``learned_partial_agg``), and — for an unambiguous single-join plan — the join-strategy bandit
    and its side sizes (``learned_join_strategy`` / ``learned_build_sides``). Every write is
    best-effort; each recorded signal only steers a later *performance* choice, never a result.
    """
    if hub is None:
        return
    try:
        from batcher.kyber.learned_tuning import record_group_reduction, record_partition_rows
        from batcher.kyber.signature import plan_signature

        if isinstance(plan, _BREAKERS):
            record_partition_rows(hub, plan_signature(plan), float(out_rows))
        if isinstance(plan, Aggregate) and plan.group_keys and input_rows:
            record_group_reduction(hub, plan_signature(plan), float(out_rows), float(input_rows))
        record_join_outcomes(hub, logical_opt, decisions, wall_ms)
    except Exception:  # pragma: no cover - recording must never break a query
        return


def record_join_outcomes(
    hub: MetadataHub | None,
    logical_opt: LogicalPlan,
    decisions: list[BuildSideDecision],
    wall_ms: float,
    *,
    distributed: bool = False,
) -> None:
    """Record an executed join's strategy, side sizes and timing so the bandit learns.

    Handled only for a plan with exactly one join (so the whole-query wall time is unambiguously
    that join's, and the single decision maps to the single join). The executed strategy (``None``
    → the engine's default hash) and the measured wall time feed the UCB1 strategy bandit and the
    hash-vs-sort-merge crossover; the measured side sizes seed build-side selection. Each is a
    choice among equivalent algorithms, so the learning changes throughput only.
    """
    if hub is None or wall_ms <= 0.0:
        return
    try:
        from batcher.kyber.learned_tuning import (
            record_broadcast_timing,
            record_join_sides,
            record_join_strategy,
            record_sort_merge_timing,
        )
        from batcher.kyber.signature import plan_signature

        joins = _all_joins(logical_opt)
        if len(joins) != 1 or len(decisions) != 1:
            return
        join, dec = joins[0], decisions[0]
        sig = plan_signature(join)
        strategy = join.strategy or "hash"
        record_join_sides(hub, sig, float(dec.left_rows), float(dec.right_rows))
        record_join_strategy(hub, sig, strategy, wall_ms)
        if strategy in ("hash", "sort_merge"):
            # `BuildSideDecision` records the *pre-swap* orientation, so a swap moves the
            # left side into the build position. This is the x-axis of the hash-vs-sort_merge
            # crossover fit; feeding it `min(l, r)` mislabels every byte-driven swap.
            build_rows = dec.left_rows if dec.swapped else dec.right_rows
            record_sort_merge_timing(hub, strategy, float(build_rows), wall_ms)
        # Feed the broadcast-vs-shuffle crossover, whose recorder had no caller at all —
        # so `learned_broadcast_max_bytes` always returned `None` and the threshold never
        # moved off its static default. Only on the distributed path: broadcasting is
        # replication *across workers*, so a single-node hash join is not a "shuffle" and
        # would poison the fit.
        if distributed and dec.build_bytes > 0.0:
            arm = "broadcast" if dec.broadcast else "shuffle"
            record_broadcast_timing(hub, arm, float(dec.build_bytes), wall_ms)
    except Exception:  # pragma: no cover - recording must never break a query
        return


def record_shuffle_outcome(hub: MetadataHub | None, plan: LogicalPlan, credits: int) -> None:
    """Persist the credit window a distributed shuffle converged on, keyed by signature.

    Warm-starts the next run of this shape via ``ResourceManager.grant_credits(signature=)`` (the
    AIMD law still governs the window it actually uses, so only the *starting* point moves — the
    result is unchanged). Best-effort; a non-positive window records nothing.
    """
    if hub is None or credits <= 0:
        return
    try:
        from batcher.carbonite.policies import record_shuffle_window
        from batcher.kyber.signature import plan_signature

        record_shuffle_window(hub, plan_signature(plan), int(credits))
    except Exception:  # pragma: no cover - recording must never break a query
        return


def record_distributed(
    hub: MetadataHub | None,
    plan: LogicalPlan,
    logical_opt: LogicalPlan,
    decisions: list[BuildSideDecision],
    credits: int,
    wall_ms: float,
) -> None:
    """Close a distributed run's loops: persist its shuffle window and feed the join bandit."""
    record_shuffle_outcome(hub, plan, credits)
    record_join_outcomes(hub, logical_opt, decisions, wall_ms, distributed=True)


# --- small shared helpers --------------------------------------------------------------------
def _all_joins(node: LogicalPlan) -> list[LogicalPlan]:
    """Every `Join` node in `node` (pre-order)."""
    from batcher.plan.visitor import children

    out: list[LogicalPlan] = [node] if isinstance(node, Join) else []
    for child in children(node):
        out.extend(_all_joins(child))
    return out


def total_source_rows(sources: list[Source]) -> int | None:
    """Total row count across `sources` (cheap footer reads), or `None` if any is unknown.

    Feeds the group-reduction learner an input size without touching a single row.
    """
    total = 0
    for s in sources:
        rc = s.row_count() if hasattr(s, "row_count") else None
        if rc is None:
            return None
        total += rc
    return total
