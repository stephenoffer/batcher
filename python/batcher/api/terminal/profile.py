"""Profiled terminal execution — the `explain(analyze=True)` / `stats()` engine.

Assembles a `QueryProfile` by running a plan through the real execution path (single
node, out-of-core spill, or distributed) with a `ProfileCollector` attached, or — for
the planned-only view — by optimizing without executing. This is the conductor stitching
Kyber's estimates to Core's measurements; it lives in `api` because it touches all three
subsystems, and is split out of `terminal.core` to keep that module within size limits.
"""

from __future__ import annotations

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan
from batcher.plan.profile import (
    Decision,
    OpProfile,
    ProfileCollector,
    QueryProfile,
    merge_metric_ops,
)

__all__ = [
    "admission_decision",
    "build_side_decisions",
    "explain",
    "planned_profile",
    "record_plan",
    "record_spill",
    "resource_decision",
    "run_profiled",
    "verdict_summary",
]


def record_spill(prof, partitions: int, reason: str | None = None) -> None:
    """Record that the out-of-core spill path was taken into the profile.

    `reason` names *why* Carbonite routed the query out of core. A partition count alone
    says what happened; the reason separates "this plan is too big" (reshape the plan)
    from "the box is under pressure" (find what else is holding memory), which are
    indistinguishable from the outside and call for opposite responses.

    Args:
        prof: The profile collector.
        partitions: How many out-of-core buckets the spilled state was sharded into.
        reason: Carbonite's spill reason, or `None` when it was not recorded.
    """
    because = f" — {reason}" if reason else ""
    prof.carbonite_summary = f"out-of-core spill ({partitions} partitions){because}"
    prof.decisions.append(
        Decision(
            subsystem="carbonite",
            category="spill",
            summary=(
                f"executed out-of-core under bounded memory ({partitions} partitions){because}"
            ),
            detail={"partitions": partitions, "reason": reason},
        )
    )


def build_side_decisions(decisions: list) -> list[Decision]:
    """Convert Kyber's per-join build-side notes into neutral `Decision` records."""
    out: list[Decision] = []
    for d in decisions:
        parts = []
        if d.swapped:
            parts.append("swap build→left")
        if d.broadcast:
            parts.append("broadcast")
        action = " + ".join(parts) if parts else "keep"
        out.append(
            Decision(
                subsystem="kyber",
                category="selection",
                summary=(
                    f"join build side: left≈{d.left_rows:,.0f} right≈{d.right_rows:,.0f} "
                    f"[{d.provenance}] → {action}"
                ),
                detail={
                    "left_rows": d.left_rows,
                    "right_rows": d.right_rows,
                    "swapped": d.swapped,
                    "broadcast": d.broadcast,
                    "provenance": d.provenance,
                    "cost_delta": d.cost_delta,
                },
            )
        )
    return out


def record_plan(prof, opt, plan, distributed: bool, decisions: list) -> None:
    """Record the optimized plan + its join decisions into the profile collector."""
    prof.optimized_ir = opt.ir
    prof.logical_ir = plan.to_ir()
    prof.physical_ops = opt.ops
    prof.distributed = distributed
    prof.decisions.extend(build_side_decisions(decisions))


def verdict_summary(verdict) -> str:
    """One-line human summary of Carbonite's admission verdict.

    Names the *operator* that binds the constraint when Carbonite identified one. Which
    resource ran out is rarely actionable on its own; which step ran it out is.
    """
    if verdict.feasible:
        return "feasible"
    at = f" at {verdict.binding_op}" if verdict.binding_op else ""
    return f"infeasible (binding: {verdict.binding_constraint}{at}) → out-of-core / counter-offer"


def admission_decision(verdict) -> Decision:
    """A neutral `Decision` capturing Carbonite's feasibility verdict for the event log."""
    return Decision(
        subsystem="carbonite",
        category="admission",
        summary=verdict_summary(verdict),
        detail={
            "feasible": verdict.feasible,
            "binding_constraint": verdict.binding_constraint,
            "binding_op": verdict.binding_op,
        },
    )


def resource_decision(rm) -> Decision:
    """A neutral `Decision` carrying Carbonite's live resource reading for the profile.

    `explain(analyze=True)` reports what the query *did* and what the optimizer *chose*,
    and said nothing about the envelope it ran in. That is the missing half of a slow-query
    diagnosis: the same plan is fast with headroom and slow at the edge of the budget, and
    from the plan alone the two look identical.

    Args:
        rm: The Carbonite `ResourceManager` the query ran under.

    Returns:
        A `Decision` whose detail is the manager's `stats()` snapshot.
    """
    stats = rm.stats()
    pool = stats.get("pool") or {}
    summary = f"memory pressure {stats.get('pressure_level', 'UNKNOWN')}"
    if pool:
        summary += f", envelope {pool.get('peak_utilization', 0.0):.0%} used at peak"
    return Decision(subsystem="carbonite", category="resources", summary=summary, detail=stats)


def explain(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    *,
    analyze: bool = False,
    fmt: str = "text",
) -> str:
    """Render the plan as a tree, optionally with the measured execution profile.

    `analyze=False` builds a planned-only `QueryProfile` (estimates + provenance +
    decisions) without executing; `analyze=True` runs the query and joins the measured
    per-operator metrics in. `fmt` selects text or a JSON document.
    """
    profile = run_profiled(plan, sources, columns) if analyze else planned_profile(plan, sources)
    if fmt == "json":
        import json

        return json.dumps(profile.to_dict(), indent=2, default=str)
    if fmt != "text":
        from batcher._internal.errors import PlanError

        raise PlanError(f"explain(format={fmt!r}) is not supported; use 'text' or 'json'.")
    return profile.render(analyze=analyze)


def _io_throughput_decisions(sources: list[Source], hub) -> list:
    """One `Decision` per source whose read throughput has been *measured* on a prior run —
    surfacing the small-files scan pathology (a slow source shows a low MB/s) directly in
    `explain()`, from the learned `io_stats` metadata. Empty on a cold store."""
    from batcher.api.source_stats import _source_identity
    from batcher.metadata.io_stats import load_source_throughput_mbps, predicted_read_seconds
    from batcher.metadata.source_stats_store import load_source_stats
    from batcher.plan.profile import Decision

    out = []
    for src in sources:
        ident = _source_identity(src)
        mbps = load_source_throughput_mbps(hub, ident)
        if mbps is None:
            continue
        detail = {"identity": ident, "throughput_mbps": round(mbps, 1)}
        summary = f"source read at {mbps:.0f} MB/s (learned)"
        # When the source's byte size is known, turn the rate into a *predicted read cost* —
        # the "predict" the optimizer/user can act on before running (a slow source stands out).
        stats = load_source_stats(hub, ident) if hub is not None else None
        byte_size = getattr(stats, "byte_size", None) if stats is not None else None
        secs = predicted_read_seconds(hub, ident, byte_size) if byte_size else None
        if secs is not None:
            detail["predicted_read_seconds"] = round(secs, 2)
            summary += f" — ~{secs:.1f}s to read"
        out.append(Decision("core", "io", summary, detail))
    return out


def _streaming_decisions(plan: LogicalPlan, sources: list[Source]) -> list:
    """What `explain()` should say about a plan whose input never ends.

    Every number in an `explain()` of a streaming plan is a placeholder: the row estimate is
    the `unknown_rows` sentinel with `DEFAULT` provenance, which a stream shares with any
    bounded source whose size merely could not be measured. So the rendering said
    ``est≈1,000,000,000,000 (default)`` and nothing at all about the two things that decide
    whether the query works — whether it can emit before its input ends, and whether its
    state is bounded by something that advances. Both are pure functions of the plan and both
    were already computed (`kyber.streaming`), and neither reached the reader.

    Silent on a wholly bounded plan, where the questions do not arise.

    Args:
        plan: The logical plan being explained.
        sources: Its bound sources, to tell a stream from a bounded relation.

    Returns:
        Zero to three `Decision`s: the unbounded inputs, the blocking operators, and the
        operators whose state nothing releases.
    """
    from batcher.kyber.streaming import (
        blocking_operators,
        unbounded_scan_ids,
        unbounded_state_operators,
    )
    from batcher.plan.profile import Decision

    unbounded = unbounded_scan_ids(plan, sources)
    if not unbounded:
        return []
    out = [
        Decision(
            "kyber",
            "streaming",
            f"{len(unbounded)} unbounded source(s) — row estimates are placeholders, not sizes",
            {"unbounded_source_ids": sorted(unbounded)},
        )
    ]
    blocking = sorted({type(n).__name__.lower() for n in blocking_operators(plan)})
    if blocking:
        out.append(
            Decision(
                "kyber",
                "streaming",
                f"cannot emit until the input ends: {', '.join(blocking)} — this plan will not "
                "stream",
                {"blocking_operators": blocking},
            )
        )
    else:
        out.append(Decision("kyber", "streaming", "emits incrementally — no blocking operator", {}))
    leaking = sorted({type(n).__name__.lower() for n in unbounded_state_operators(plan)})
    if leaking:
        out.append(
            Decision(
                "kyber",
                "streaming",
                f"retains state nothing releases: {', '.join(leaking)} — bounded only by "
                "memory.streaming_state_max_bytes",
                {"unbounded_state_operators": leaking},
            )
        )
    return out


def _logical_op_profiles(
    plan: LogicalPlan, metric_ops: list[dict] | None = None
) -> list[OpProfile]:
    """`OpProfile`s from the un-lowered LOGICAL plan tree, pre-order.

    The seam a UDF plan takes: a `map_batches`/inference pipeline has no engine IR (its
    `to_ir()` deliberately raises), so the estimate/measure join `build_op_profiles`
    performs off the lowered IR cannot run. This walks the logical tree via
    `logical_preorder`, naming each operator by its node type, so `explain()` renders a
    readable operator tree — `MapBatches` included — instead of crashing.

    `metric_ops` are the `StageRecorder`'s measurements for the same tree, numbered by the
    same walk, so `stats()` on an ML pipeline shows measured rows and time per stage rather
    than refusing. `None` leaves every row planned-only, which is `explain()` without
    `analyze`.
    """
    from batcher.plan.profile import logical_preorder

    measured = {int(m.get("op_id", -1)): m for m in (metric_ops or [])}
    nodes = list(logical_preorder(plan))
    out: list[OpProfile] = []
    for op_id, (depth, node) in enumerate(nodes):
        m = measured.get(op_id)
        # Prefer the measured operator name, as `build_op_profiles` does: it is the only
        # thing that can tell a per-row `map` from a vectorized `map_batches`, which are the
        # same node type but 10-100x apart in cost.
        kind = str(m.get("kind")) if m and m.get("kind") else type(node).__name__
        if m is None:
            out.append(OpProfile(op_id=op_id, kind=kind, depth=depth))
            continue
        out.append(
            OpProfile(
                op_id=op_id,
                kind=kind,
                depth=depth,
                measured=True,
                rows_in=_rows_in(m, op_id, nodes, measured),
                rows_out=int(m.get("rows_out", 0)),
                elapsed_ms=float(m.get("elapsed_ns", 0)) / 1e6,
                result_bytes=int(m.get("result_bytes", 0)),
                threads=int(m.get("threads", 0)),
                backend=str(m.get("backend", "")),
            )
        )
    return out


def _rows_in(metric: dict, op_id: int, nodes: list, measured: dict) -> int:
    """A stage's input rows, read off the stage below it when it could not count them.

    The streaming path meters a stage by wrapping its *output* generator, which sees no
    input — so it reports `rows_in=0` and the tree supplies it instead. In a linear chain
    (which is the only shape that path takes) a stage's input is exactly the output of the
    node directly beneath it, i.e. the next entry in the pre-order walk. Without this the
    table shows `0` for every streamed stage, which reads as "this stage consumed nothing"
    rather than "this seam could not observe it".
    """
    rows_in = int(metric.get("rows_in", 0))
    if rows_in:
        return rows_in
    child_id = op_id + 1
    if child_id >= len(nodes):
        return 0
    depth, _node = nodes[op_id]
    child_depth, _child = nodes[child_id]
    if child_depth != depth + 1:  # not this node's child — a sibling or an ancestor's
        return 0
    return int(measured.get(child_id, {}).get("rows_out", 0))


def _udf_planned_profile(plan: LogicalPlan, sources: list[Source], hub) -> QueryProfile:
    """A planned-only `QueryProfile` for a plan carrying a Python UDF (`map_batches`).

    Such a plan cannot be lowered to engine IR, so it is shown un-optimized: the logical
    tree, named by node type, with a `Decision` header stating the plan is un-lowered and
    thus unestimated. An honest partial plan beats the `NotImplementedError` the lowering
    path would raise — reading the plan is step one of debugging a batch-inference query.
    """
    note = Decision(
        subsystem="core",
        category="explain",
        summary=(
            "plan contains a map_batches/UDF stage shown un-lowered "
            "(no engine IR, so no optimized tree or row estimates)"
        ),
    )
    return QueryProfile(
        ops=tuple(_logical_op_profiles(plan)),
        decisions=(
            note,
            *_io_throughput_decisions(sources, hub),
            # A `map_batches` pipeline over a topic is the shape most likely to be a stream,
            # so the branch that cannot lower to IR is the one that needs this most.
            *_streaming_decisions(plan, sources),
        ),
        logical_ir=None,
        optimized_ir=None,
    )


def planned_profile(plan: LogicalPlan, sources: list[Source]) -> QueryProfile:
    """A planned-only `QueryProfile`: Kyber's optimized tree, estimates, and decisions.

    A plan carrying a `map_batches`/inference UDF has no engine IR to lower or optimize, so
    it renders the un-lowered logical tree (`_udf_planned_profile`) instead of crashing.

    Collects the per-source statistics first, exactly as the execution path does. Without
    them this showed a *different* plan than `collect()` runs, silently and in the direction
    that matters: every rewrite that reasons about a column's recorded min/max — zone-map
    pruning, the join-disjointness proof, ordered sargable transposition — has nothing to read
    without a footer, so `explain()` reported the un-pruned plan for a query that prunes. The
    plan memo is keyed on the statistics too, so supplying them is also what lets `explain()`
    and `collect()` share one cache entry instead of planning the same query twice.
    """
    from batcher import core, kyber
    from batcher.api.source_stats import collect_source_stats, column_bounds_needed
    from batcher.plan.profile import build_op_profiles

    hub = core.default_hub()
    if core.has_map_batches(plan):
        return _udf_planned_profile(plan, sources, hub)
    source_stats = collect_source_stats(sources, hub, need_columns=column_bounds_needed(plan))
    opt, decisions = kyber.optimize_traced(
        plan, sources=sources, hub=hub, source_stats=source_stats
    )
    return QueryProfile(
        ops=build_op_profiles(opt.ir, opt.ops, None),
        decisions=(
            *build_side_decisions(decisions),
            *_io_throughput_decisions(sources, hub),
            *_streaming_decisions(plan, sources),
        ),
        logical_ir=plan.to_ir(),
        optimized_ir=opt.ir,
    )


def run_profiled(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    query_id: str = "",
) -> QueryProfile:
    """Execute the plan through the real (single-node/spill/distributed) path, profiled.

    Always executes (no metadata short-circuit — the point is to measure) with a
    `ProfileCollector` attached, then assembles a `QueryProfile`. Runs the *same* path the
    query would (`distributed="auto"` resolves to the live cluster), under the sensed
    config, so the profile reflects reality: single-node gives every operator a measured
    row in the driver tree; a distributed aggregate adds a measured *worker map sub-plan*
    section (its own op-id space, kept separate rather than joined). Raises `PlanError` for
    an unbounded source.

    A `map_batches`/ML pipeline has no engine IR to number operators against, so its stages
    are measured by the orchestrator instead (`StageRecorder`) and rendered against the
    logical tree. That path used to refuse outright, which left the batch-inference shape —
    the one the ML surface exists for — with no answer to "which stage is the bottleneck",
    the first question every tuning guide asks.
    """
    import time

    from batcher import core
    from batcher.api import executors
    from batcher.api.terminal.core import _resolve_distributed
    from batcher.io.source import is_bounded
    from batcher.plan.profile import ProfileCollector

    if any(not is_bounded(s) for s in sources):
        from batcher._internal.errors import PlanError

        raise PlanError(
            "explain(analyze=True)/stats() materializes the result, but the dataset "
            "has an unbounded source."
        )
    collector = ProfileCollector()
    # Pass plan + sources so the size-aware "auto" decision matches `collect()`. Resolving
    # with neither hit `resolve_distributed`'s `sources is None -> True` fall-through, forcing
    # every profiled run to distribute on a multi-node cluster — measuring a path a small
    # query runs single-node, the opposite of "reflects reality".
    distributed = _resolve_distributed("auto", plan, sources)
    ctx = core.ExecutionContext(columns=columns, hub=core.default_hub(), profile=collector)
    t0 = time.perf_counter()
    table = executors.select(plan, distributed=distributed).execute(plan, sources, ctx)
    total_ms = (time.perf_counter() - t0) * 1000.0
    # The soft memory envelope the run was admitted against, so the profile can report peak
    # memory as a fraction of budget (the >80% memory-utilization target). Best-effort.
    budget = 0
    try:
        from batcher.carbonite.memory.pressure import PressureMonitor

        budget = PressureMonitor().budget_bytes()
    except Exception:  # pragma: no cover - a missing budget just omits the memory-% line
        budget = 0
    if core.has_map_batches(plan):
        return _udf_measured_profile(
            plan, sources, collector, total_ms=total_ms, rows=table.num_rows, query_id=query_id
        )
    return collector.to_profile(
        total_ms=total_ms, rows=table.num_rows, query_id=query_id, memory_budget_bytes=budget
    )


def _udf_measured_profile(
    plan: LogicalPlan,
    sources: list[Source],
    collector: ProfileCollector,
    *,
    total_ms: float,
    rows: int,
    query_id: str,
) -> QueryProfile:
    """A measured `QueryProfile` for a `map_batches`/ML pipeline, off the logical tree.

    The engine numbers operators against the lowered IR, which this plan shape does not have,
    so the stages are numbered against the logical plan instead and the two never mix. Row
    estimates stay absent: Kyber cannot size past an opaque UDF, and inventing a number here
    would make `est_error` compare a measurement against a guess.

    **Distributed is the one case this cannot measure.** The stages then run inside Ray
    workers, in other processes, while the recorder lives on the driver — so it collects
    nothing. The profile says so rather than rendering an empty table that looks like a run
    which did no work, and it still carries any `worker_ops` the workers shipped back.
    """
    from batcher import core
    from batcher.plan.profile import worker_op_profiles

    metric_ops = collector.stage_recorder.metric_ops()
    workers = (
        worker_op_profiles(merge_metric_ops(collector.worker_metrics))
        if collector.worker_metrics
        else ()
    )
    summary = (
        "map_batches/UDF pipeline measured per stage against the logical tree "
        "(no engine IR, so no row estimates to compare against)"
    )
    if not metric_ops:
        summary = (
            "map_batches/UDF pipeline ran in worker processes, so the driver measured no "
            "stages; per-stage timings are available on a single-node run"
        )
    note = Decision(subsystem="core", category="explain", summary=summary)
    return QueryProfile(
        ops=tuple(_logical_op_profiles(plan, metric_ops)),
        total_ms=total_ms,
        rows=rows,
        query_id=query_id,
        measured=bool(metric_ops) or bool(workers),
        distributed=collector.distributed,
        decisions=(note, *_io_throughput_decisions(sources, core.default_hub())),
        worker_ops=workers,
    )
