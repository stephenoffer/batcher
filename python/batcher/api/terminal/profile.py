"""Profiled terminal execution — the `explain(analyze=True)` / `stats()` engine.

Assembles a `QueryProfile` by running a plan through the real execution path (single
node, out-of-core spill, or distributed) with a `ProfileCollector` attached, or — for
the planned-only view — by optimizing without executing. This is the conductor stitching
Kyber's estimates to Core's measurements; it lives in `api` because it touches all three
subsystems, and is split out of `terminal.core` to keep that module within size limits.
"""

from __future__ import annotations

from batcher._internal.errors import BackendError
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan
from batcher.plan.profile import Decision, QueryProfile

__all__ = [
    "admission_decision",
    "build_side_decisions",
    "explain",
    "planned_profile",
    "record_plan",
    "record_spill",
    "run_profiled",
    "verdict_summary",
]


def record_spill(prof, partitions: int) -> None:
    """Record that the out-of-core spill path was taken into the profile."""
    prof.carbonite_summary = f"out-of-core spill ({partitions} partitions)"
    prof.decisions.append(
        Decision(
            subsystem="carbonite",
            category="spill",
            summary=f"executed out-of-core under bounded memory ({partitions} partitions)",
            detail={"partitions": partitions},
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
    """One-line human summary of Carbonite's admission verdict."""
    if verdict.feasible:
        return "feasible"
    return f"infeasible (binding: {verdict.binding_constraint}) → out-of-core / counter-offer"


def admission_decision(verdict) -> Decision:
    """A neutral `Decision` capturing Carbonite's feasibility verdict for the event log."""
    return Decision(
        subsystem="carbonite",
        category="admission",
        summary=verdict_summary(verdict),
        detail={"feasible": verdict.feasible, "binding_constraint": verdict.binding_constraint},
    )


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


def planned_profile(plan: LogicalPlan, sources: list[Source]) -> QueryProfile:
    """A planned-only `QueryProfile`: Kyber's optimized tree, estimates, and decisions."""
    from batcher import core, kyber
    from batcher.plan.profile import build_op_profiles

    hub = core.default_hub()
    opt, decisions = kyber.optimize_traced(plan, sources=sources, hub=hub)
    return QueryProfile(
        ops=build_op_profiles(opt.ir, opt.ops, None),
        decisions=(*build_side_decisions(decisions), *_io_throughput_decisions(sources, hub)),
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
    an unbounded source and `BackendError` for a `map_batches`/ML pipeline (the opaque UDF
    path emits no per-operator metrics — profile the relational portion instead).
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
    if core.has_map_batches(plan):
        raise BackendError(
            "explain(analyze=True)/stats() is not available for map_batches/ML pipelines "
            "(the opaque UDF path emits no per-operator metrics); profile the relational "
            "portion instead."
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
    return collector.to_profile(
        total_ms=total_ms, rows=table.num_rows, query_id=query_id, memory_budget_bytes=budget
    )
