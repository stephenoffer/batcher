"""The contract loop: Kyber optimizes, Carbonite admits, Core executes, metadata flows back."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher._internal.hardware import available_cpu_count
from batcher._internal.logging import get_logger, log_kv
from batcher.api._join_helpers import _empty_result_schema
from batcher.api.source_stats import (
    collect_source_stats,
    column_bounds_needed,
)
from batcher.config import active_config, config_context
from batcher.io.source import Source, read_source

if TYPE_CHECKING:
    from batcher.core import ExecutionContext
    from batcher.kyber.rules.selection import BuildSideDecision
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan


from batcher.api.orchestration.autoconfig import resolve_auto_config, with_auto_config  # noqa: F401

# --- Zero-config sizing -----------------------------------------------------
# When the user leaves a knob unset, fill it from the same analyses Kyber/Carbonite
# already produce rather than a blind constant — composing their decisions, never
# re-deriving them. The fallback is the historical default, used only when nothing
# about the data size is known.
DEFAULT_PARTITIONS = 16
_MIN_PARTITIONS = 4
_MAX_PARTITIONS = 4096

_log = get_logger("api.run")


def _phase(name: str, seconds: float) -> None:
    """Record one contract-loop phase timing (stats → Kyber → Carbonite → Core) as DEBUG.

    Formerly `print`s behind a `BATCHER_SORT_PROFILE` env var — stdout noise that no log
    file, log shipper, or dashboard ever saw, and that a user could not turn on per
    subsystem. On `batcher.api.run` at DEBUG they follow the one `log_level`, and the
    phase/duration are structured fields, so "where did the control plane spend its time"
    is answerable without re-parsing text.
    """
    log_kv(_log, logging.DEBUG, "run phase", phase=name, seconds=round(seconds, 3))


def _log_decisions(opt, decisions, verdict, *, distributed: bool) -> None:
    """Record the plan verdict at INFO and each join's build-side choice beside it.

    This is what makes ``verbosity="verbose"`` mean what it says. The optimizer's choices
    already existed as `Decision` objects on the profile, but the profile is an artifact you
    read *afterwards* — so without these records the verbose rung showed nothing a user
    could not already see at `normal`, and the level was a promise the engine did not keep.

    INFO, not DEBUG: a join order flipping or an admission verdict turning infeasible is the
    kind of thing an operator wants to see without opting into per-phase timing noise.
    """
    log_kv(
        _log,
        logging.INFO,
        "plan admitted" if verdict.feasible else "plan infeasible",
        ops=len(opt.ops),
        feasible=verdict.feasible,
        binding=verdict.binding_constraint or "none",
        distributed=distributed,
    )
    for i, decision in enumerate(decisions):
        log_kv(
            _log,
            logging.INFO,
            "join build side",
            join=i,
            build=("right" if decision.swapped else "left"),
            broadcast=decision.broadcast,
            left_rows=round(decision.left_rows),
            right_rows=round(decision.right_rows),
            why=decision.provenance,
        )


def _clamp_partitions(n: int) -> int:
    return max(_MIN_PARTITIONS, min(_MAX_PARTITIONS, n))


def _distributed_hardware():
    """The cluster's `HardwareProfile` for planning, or `None` if the topology is unreadable.

    Isolated so the `dist` import stays lazy — a single-node run never touches Ray — and so a
    topology read that fails (Ray down, no worker nodes) degrades to `None`, leaving the
    Optimizer to plan against the local machine rather than a fabricated cluster.
    """
    try:
        from batcher.dist.executors.ray_runtime.scaling import cluster_hardware_profile

        return cluster_hardware_profile()
    except Exception:  # pragma: no cover - Ray optional / topology unreadable
        return None


def partitions_from_physical(opt: PhysicalPlan) -> int | None:
    """Spill partition count implied by the optimized plan, or `None` if unsized.

    Reuses the per-breaker ``n_max_parallelism`` Kyber already computed (input rows
    / `target_rows_per_task`) — the same data-sized fan-out the distributed path
    uses — so out-of-core spilling shards by data volume instead of a blind 16.

    Floored at the machine's usable core count, because data volume alone answers only half
    the question. Kyber sizes this purely from rows, which is right for the distributed path
    where `clamp_workers` refits it to the cluster afterwards — but nothing refits it here.
    A 40M-row aggregate at the default 4M rows/task is 10 partitions whether the box has 4
    cores or 128, and a spilled merge cannot use more cores than it has partitions, so the
    other 118 sit idle for the whole out-of-core phase. Partitions are the unit of work; there
    must be at least one per core for the phase to fill the machine.
    """
    widths = [op.bounds.n_max_parallelism for op in opt.ops if op.bounds.n_max_parallelism > 0]
    if not widths:
        return None
    return _clamp_partitions(max(max(widths), available_cpu_count()))


# `auto_num_partitions` (the data-sized spill/shuffle partition count, learned-seeded) is an
# adaptive-sizing decision, so it lives in `api.tuning`; re-exported here for its callers.


def run_relational(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    *,
    distributed: bool = False,
    materialize: bool = True,
) -> tuple[pa.Table | Source, list[BuildSideDecision]]:
    """Run one relational (non-UDF) plan through Kyber → Carbonite → Core.

    Returns the materialized result and the optimizer's per-join build-side
    decisions (telemetry the adaptive executor reports; ignored by the one-shot
    executors). Raises `PlanError` if Carbonite's admission policy rejects the
    plan. `distributed` fans the plan out across Ray workers, using Carbonite's
    scheduling envelope; the distributed executor makes its own shape/partition
    decisions, so the *logical* plan is shipped and single-node rewrites are not
    overlaid (the mergeable algebra guarantees the result equals single-node).

    When `execution.adaptive_morsel_sizing` is on (the default) and memory is under
    pressure, Carbonite's pressure-scaled morsel target is activated for the execution
    scope (reaching both the in-process engine and the shipped worker config) — a
    smaller streaming working set when memory is tight. Result-invariant, and a no-op
    when memory is unpressured (the target is returned unchanged), so an unpressured
    query stays byte-identical on every path.
    """
    import contextlib

    from batcher import carbonite

    scope: contextlib.AbstractContextManager = contextlib.nullcontext()
    if active_config().execution.adaptive_morsel_sizing:
        # Pass the hub so the morsel target reflects the *learned* per-family peak memory
        # (Carbonite's `LearnedMemoryModel` over recorded `m_peak_bytes`), not just live
        # pressure — result-invariant (a morsel only batches data). Restrict the learned
        # width to *this plan's* operator families so a narrow plan is not throttled by a
        # wide aggregate measured in an unrelated earlier query.
        from batcher.plan.visitor import walk

        families = {type(node).__name__ for node in walk(plan)}
        adapted = carbonite.ResourceManager(hub=ctx.hub).recommended_config(families)
        if adapted is not None:
            scope = config_context(adapted)
    with scope:
        return _run_relational(plan, sources, ctx, distributed=distributed, materialize=materialize)


def _projected_input_bytes(sources: list[Source], projections: dict[int, list[str]]) -> int:
    """Bytes the sources would occupy if resolved whole, from metadata alone.

    The in-memory path materializes every source before the engine starts, so this is
    the resident cost of *reading*, independent of what the query then computes. It is
    a declared `row_count()` times the projected schema's per-row width — no I/O, no
    scan, so it is available early enough to choose the streaming path instead.

    Returns `0` when any source cannot declare its size. A partial sum would understate
    the total, and understating here is the direction that OOMs, so an unknown makes the
    whole figure unknown rather than optimistic.
    """
    from batcher.plan.types import schema_row_bytes

    total = 0.0
    for i, src in enumerate(sources):
        rows = _declared_row_count(src)
        if rows is None or rows < 0:
            return 0
        try:
            schema = src.schema()
            projection = projections.get(i)
            if projection:
                schema = pa.schema([schema.field(schema.get_field_index(c)) for c in projection])
        except Exception:  # pragma: no cover - a source that cannot describe itself
            return 0
        total += rows * schema_row_bytes(schema)
    return int(total)


def _declared_row_count(src: Source) -> int | None:
    """The exact row count a source declares without a scan, or None if it cannot.

    Used to decide whether a read saw the source *whole* (so its distinct count may be
    learned). A source with no `row_count`, or one that raises, is treated as unknown — the
    safe side, since an unverifiable "did I see everything?" must answer no.
    """
    fn = getattr(src, "row_count", None)
    if not callable(fn):
        return None
    try:
        n = fn()
    except Exception:  # pragma: no cover - a source that cannot count itself
        return None
    return int(n) if n is not None else None


def _proven_empty_table(logical_opt: LogicalPlan, plan: LogicalPlan) -> pa.Table | None:
    """A typed, zero-row result when the optimizer proved the plan yields no rows.

    Kyber signals that proof by rewriting the root to a `Limit(input, 0)` — the only way
    the plan algebra can say "provably empty". The result is then fully determined by
    the output schema, so no source is read and the engine never runs. Returns None
    (execute normally) when the root is not that shape or the schema cannot be inferred
    without executing.
    """
    from batcher.plan.logical import Limit

    if not (isinstance(logical_opt, Limit) and logical_opt.n == 0):
        return None
    inferred = plan.available_schema()
    return None if inferred is None else inferred.arrow.empty_table()


def _run_relational(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    *,
    distributed: bool = False,
    materialize: bool = True,
) -> tuple[pa.Table | Source, list[BuildSideDecision]]:
    """The Kyber → Carbonite → Core body, run under the (possibly adapted) config."""
    import time

    from batcher import carbonite, core, kyber
    from batcher._internal.logging import ensure_configured

    ensure_configured()
    _t0 = time.perf_counter()  # wall clock for the join-strategy bandit's per-run reward
    # Per-source statistics (footer/manifest/catalog) let the optimizer's zone-map
    # and null-driven rules prune predicates and skip files before execution. Reuse
    # the conductor's already-collected stats when present (the metadata-answer
    # attempt for a missed count()/is_empty() collected them), so a terminal op reads
    # each source's footer once across both passes.
    _rpt = time.perf_counter()
    source_stats = (
        ctx.source_stats
        if ctx.source_stats is not None
        else collect_source_stats(sources, ctx.hub, need_columns=column_bounds_needed(plan))
    )
    _phase("collect_source_stats", time.perf_counter() - _rpt)
    _rpt = time.perf_counter()
    # Seed the distinct counts the optimizer is about to read: no file footer carries
    # `ndv`, so without this a query's *first* run orders its joins blind.
    from batcher.api.terminal._metadata import seed_column_ndv

    seed_column_ndv(ctx.hub, sources, plan)
    # The hardware Kyber plans for. Single-node: the Optimizer detects this machine. Distributed:
    # the cluster's *binding* worker (smallest cores/RAM), so a cache/memory-sized threshold
    # tracks the workers rather than a possibly-fat driver. Falls back to the local profile when
    # the topology is unreadable, so a distributed run on a down cluster plans exactly as before.
    hardware = _distributed_hardware() if distributed else None
    # One optimizer run yields both the physical plan (admission/costing) and the
    # optimized *logical* plan (the distributed / out-of-core executors read its derived
    # join keys + pushed predicates). Computing both here avoids re-running the entire
    # pipeline a second time via `optimize_logical` on those paths.
    opt, logical_opt, decisions = kyber.optimize_full(
        plan, sources=sources, hub=ctx.hub, source_stats=source_stats, hardware=hardware
    )
    _phase("kyber.optimize_full", time.perf_counter() - _rpt)
    _rpt = time.perf_counter()
    prof = ctx.profile
    if prof is not None:
        from batcher.api.terminal.profile import record_plan

        record_plan(prof, opt, plan, distributed, decisions)

    # Kyber's zone-map rules can *prove* a plan empty — a predicate its per-source
    # bounds rule out entirely — and record that by rewriting the root to `Limit(x, 0)`.
    # Executing that reads every source in full and throws every row away: the maximum
    # possible I/O for an answer already known, and worse than not having proved it at
    # all (the proof drops the `Filter`, and with it the predicate the source would have
    # used to skip its files). The proof only exists *after* optimization, which is why
    # the raw-plan pre-gate in `metadata_answer` cannot see it. Answer it here instead.
    if materialize:
        empty = _proven_empty_table(logical_opt, plan)
        if empty is not None:
            kyber.record_execution(ctx.hub, plan, 0)
            return empty, decisions

    # Hub-backed so admission/spill/reservation size from the learned per-family peak
    # memory (measured `m_peak_bytes`), not the plan estimate alone — closing the "peak
    # measured but never consumed" gap. Result-invariant: a spill/reservation choice only
    # changes where data lives, never the answer.
    rm = carbonite.ResourceManager(hub=ctx.hub)
    verdict = rm.validate(opt)
    _log_decisions(opt, decisions, verdict, distributed=distributed)
    if prof is not None:
        from batcher.api.terminal.profile import admission_decision, verdict_summary

        prof.carbonite_summary = verdict_summary(verdict)
        prof.decisions.append(admission_decision(verdict))
    # A memory-binding "infeasible" verdict is Carbonite's spill-friendly
    # counter-offer, not a hard stop: the plan won't fit memory, so route it
    # out-of-core (below) rather than failing. Any *other* binding constraint
    # (e.g. parallelism) has no spill remedy here, so it is a real failure.
    must_spill = not verdict.feasible and verdict.binding_constraint == "memory"
    if not verdict.feasible and not must_spill:
        raise PlanError(f"plan is infeasible (binding constraint: {verdict.binding_constraint})")

    if distributed:
        from batcher import dist
        from batcher.api.tuning import distributed_grant, record_distributed

        # Learned scheduling: size worker fan-out from the measured data volume (when the user
        # gave none) and warm-start the shuffle credit window from what this signature converged
        # on last time. Both are pure scheduling levers (AIMD still governs the window used, the
        # mergeable algebra makes any worker count identical), so a cold hub grants the default.
        _phase("carbonite.validate", time.perf_counter() - _rpt)
        _rpt = time.perf_counter()
        workers, envelope = distributed_grant(rm, opt, plan, sources, ctx)
        _phase("distributed_grant", time.perf_counter() - _rpt)
        _rpt = time.perf_counter()
        # Distribute the OPTIMIZED logical plan, not the raw one: the distributed executor
        # reads join keys / pushed predicates straight off the LogicalPlan, and a comma
        # join (`FROM a, b WHERE a.k=b.k`) is raw-lowered as a cartesian inner join on a
        # constant `__cross_key` with the equality stranded in a Filter above it. Run raw,
        # every row hashes to one bucket (a cross product) and the shuffle collapses onto a
        # single reducer; the optimized logical plan derives the real `a.k=b.k` join keys
        # first (the same structure the adaptive path already distributes). Single-node was
        # unaffected because it executes `opt`'s IR, which carries the derived keys.
        logical = logical_opt
        # Profiling: collect the workers' map sub-plan metrics (their own profile section).
        wm: list = []
        result = dist.execute_distributed(
            logical,
            sources,
            workers,
            transport=ctx.transport,
            envelope=envelope,
            hub=ctx.hub,
            materialize=materialize,
            metrics_out=wm if prof is not None else None,
        )
        _phase("execute_distributed", time.perf_counter() - _rpt)
        _rpt = time.perf_counter()
        if prof is not None:
            prof.worker_metrics = wm
        # Core collects metadata on every path so later plans improve with use.
        from batcher.api.terminal._metadata import collect_source_metadata

        collect_source_metadata(ctx.hub, sources)
        _phase("collect_source_metadata", time.perf_counter() - _rpt)
        # Close the loops: persist the shuffle window used and feed the join-strategy bandit.
        record_distributed(
            ctx.hub,
            plan,
            logical_opt,
            decisions,
            envelope.credits,
            (time.perf_counter() - _t0) * 1000.0,
        )
        return result, decisions

    # Carbonite decides out-of-core: if the estimated working set won't fit the
    # memory envelope (admission counter-offer or the spill estimate), run the
    # partition-and-spill executor so the query completes under bounded memory
    # instead of OOMing. Shapes with no spilling path fall through to in-memory —
    # unless admission already proved it won't fit, in which case that is a real
    # infeasibility rather than a silent OOM.
    # The input is the third signal, and on a scan-heavy query the dominant one: the
    # in-memory path below resolves every source to Arrow *before* the engine runs, so a
    # 600M-row scan is resident in full even when the query returns four rows. Sized from
    # source metadata (no I/O), so it is known here rather than discovered by the OOM.
    input_bytes = _projected_input_bytes(sources, opt.source_projections)
    if must_spill or rm.should_spill(opt) or rm.input_exceeds_budget(input_bytes):
        from batcher.dist.spill import spill_collect

        # Shard the out-of-core spill by data volume: prefer the learned recommendation
        # (from measured per-family peak memory), then Kyber's per-breaker fan-out, then a
        # constant — so a bigger group-by/join uses more, smaller buckets. Partition count
        # only shards the spill; the merged result is identical (mergeable algebra).
        partitions = (
            rm.recommend_spill_partitions(opt)
            or partitions_from_physical(opt)
            or DEFAULT_PARTITIONS
        )
        # Close the loop's return leg: when admission refused this plan it attached a
        # `suggested_bounds` counter-offer naming the envelope the plan *would* fit in.
        # Honor it as a floor, so each bucket actually fits the budget that forced the
        # spill — the count above is sized from a fixed bytes-per-partition constant that
        # knows nothing about this machine's memory. Result-invariant (mergeable algebra).
        partitions = max(partitions, rm.partitions_for_bounds(opt, verdict.suggested_bounds))
        if prof is not None:
            from batcher.api.terminal.profile import record_spill

            record_spill(prof, partitions)
        # Spill the *optimized* logical plan, not the raw one: the optimizer derives real
        # join keys (a comma join, else a cartesian blow-up out-of-core) and lowers
        # `COUNT(DISTINCT x)` to `COUNT(*)` over a `DISTINCT` — so the spilling executor
        # dedups efficiently (hash-partitioned) instead of spilling a giant value list.
        # Reuse the logical plan already optimized above rather than re-running the pipeline.
        # Force the learned spill codec (large IO-bound state compresses; small state does not);
        # IPC self-describes its codec, so the un-spilled result is byte-identical either way.
        from batcher.api.tuning import spill_compression_scope

        with spill_compression_scope(rm, opt):
            spilled = spill_collect(logical_opt, sources, partitions)
        if spilled is not None:
            kyber.record_execution(ctx.hub, plan, spilled.num_rows)
            return spilled, decisions
        # An *advisory* infeasibility rests on a `Provenance.DEFAULT` guess: worth routing
        # out-of-core, but a guess must never fail a legitimate query (the admission
        # contract), so fall through to the in-memory path instead of raising.
        if must_spill and not verdict.advisory:
            raise PlanError(
                "plan does not fit the memory envelope and has no out-of-core path "
                f"(binding constraint: {verdict.binding_constraint})"
            )

    # Resolve lazy sources to Arrow batches (reads happen here, not earlier).
    # Projection + predicate pushdown tell each source what to read. Each read is timed and
    # its observed throughput captured per source identity — Core measures the I/O the
    # hardware actually delivered so a later read of the same source can predict its cost
    # (the small-files scan pathology). Best-effort, negligible overhead.
    import time as _time

    from batcher.api.source_stats import _source_identity
    from batcher.metadata.io_stats import record_source_io, scanned_byte_count

    resolved = []
    complete_scan: list[bool] = []
    for i, src in enumerate(sources):
        _t0 = _time.perf_counter()
        predicate = opt.source_predicates.get(i)
        batches = read_source(src, opt.source_projections.get(i), predicate)
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
        resolved.append(batches)
        # Whether this read saw the source *whole* — no predicate filtered it, and the row
        # count read matches what the source declares. Only a whole scan may teach the learner
        # a source-level distinct count (a partial scan's ndv is an under-count, not an
        # estimate; see `learn_column_stats`). Unknown row count → treat as partial, the safe
        # side: a distinct count we might be wrong about is one we decline to record.
        declared = _declared_row_count(src)
        scanned = sum(b.num_rows for b in batches)
        complete_scan.append(predicate is None and declared is not None and scanned == declared)
        identity = _source_identity(src)
        record_source_io(
            ctx.hub,
            identity,
            scanned_byte_count(identity, opt.source_projections.get(i), scanned, batches),
            _elapsed_ms,
        )
    # Reserve the estimated envelope against the process-wide buffer pool for the
    # duration of execution, so concurrent queries draw on one budget. If the
    # reservation does not fit (concurrent queries already over budget), prefer the
    # out-of-core path over racing them into an OOM — reserve-before-allocate is only
    # real if a `False` actually changes behavior (C30/C31).
    with rm.reserve(rm.estimated_bytes(opt)) as granted:
        if not granted:
            from batcher.dist.spill import spill_collect

            parts = partitions_from_physical(opt) or DEFAULT_PARTITIONS
            # The *optimized* plan, for the reason the primary spill site above states: the
            # raw plan spills a comma join as a cartesian product.
            spilled = spill_collect(logical_opt, sources, parts)
            if spilled is not None:
                kyber.record_execution(ctx.hub, plan, spilled.num_rows)
                return spilled, decisions
        # A bare `Scan` is already done: the reader decoded the files and applied the pushed
        # projection, so its batches *are* the result. Handing them back to the engine only to
        # pass them through a no-op operator costs a full round trip of the data across the FFI
        # boundary — ~25% of the wall clock of `read_parquet(...).collect()`, the most common
        # query there is. Recognize the shape and skip it (`core.scan_only_result`).
        table = (
            None
            if prof is not None  # profiling wants the per-operator metrics; take the real path
            else core.scan_only_result(logical_opt, resolved, opt.source_predicates)
        )
        if table is None:
            # When profiling, take the metered path (still feeding the hub) so the per-operator
            # `ExecMetrics` reach the conductor's `QueryProfile`; otherwise the plain path,
            # which skips even the tiny metrics serialization — keeping an ordinary run intact.
            if prof is not None:
                batches, metric_ops = core.execute_local_metered(opt, resolved, feedback=ctx.hub)
                prof.metric_ops = metric_ops
            else:
                batches = core.execute_local(opt, resolved, feedback=ctx.hub)
            table = pa.Table.from_batches(
                batches,
                schema=batches[0].schema if batches else _empty_result_schema(plan, ctx.columns),
            )
    # Feed the measured output size back to the learner for next time, learn
    # per-column distinct counts / quantiles from the scanned input, and record the
    # filter's measured selectivity (a ratio that generalizes across input sizes) —
    # so later plans get sketch- and feedback-driven cardinality.
    kyber.record_execution(ctx.hub, plan, table.num_rows)
    from batcher.api.terminal._metadata import learn_column_stats

    learn_column_stats(ctx.hub, resolved, sources, plan, complete_scan)
    kyber.record_selectivity(ctx.hub, plan, sources, table.num_rows)
    # Close the conductor's tuning loops from this run's measured outcomes (breaker volume →
    # partition count, group reduction → pre-aggregation, join wall time → the bandit). Each
    # only steers a later performance choice, so recording is result-invariant.
    from batcher.api.tuning import record_run_feedback, total_source_rows

    record_run_feedback(
        ctx.hub,
        plan,
        logical_opt,
        decisions,
        out_rows=table.num_rows,
        input_rows=total_source_rows(sources),
        wall_ms=(time.perf_counter() - _t0) * 1000.0,
    )
    # Close the pressure-hysteresis loop too: the manager reads a past run's flap rate at
    # construction to stiffen de-escalation for an oscillating workload, but nothing had ever
    # written one, so that read was permanently cold and the mechanism never engaged.
    _record_flap_rate(ctx.hub, rm)
    return table, decisions


def _record_flap_rate(hub: object, rm: object) -> None:
    """Persist this run's measured pressure-flap rate. Best-effort; never breaks a query."""
    try:
        from batcher.carbonite.memory.pressure import record_flap_rate

        rate = rm.flap_rate()  # type: ignore[attr-defined]
        if rate is not None:
            record_flap_rate(hub, rate)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - a learned write must never fail a run
        return
