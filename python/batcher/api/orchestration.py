"""The shared Kyber → Carbonite → Core contract loop for relational plans.

This is the single implementation of the conductor's terminal-op orchestration:
optimize the plan (full Kyber, with per-operator `ResourceBounds`), let Carbonite
govern it (admission, out-of-core spill, buffer reservation / scheduling
envelope), execute via Core with the metadata feedback sink, and record what was
measured so later plans improve. Every relational (non-UDF) terminal path —
single-node, distributed, and each adaptive stage — routes through
`run_relational`, so the contract loop is applied in exactly one place and the
paths cannot drift out of sync.

It lives in `api` because it imports all three subsystems (plus `dist`); the
independence contract forbids any of them from importing the others, so the
conductor is the one layer allowed to assemble them.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api._join_helpers import _empty_result_schema
from batcher.api.source_stats import (
    collect_source_stats,
    column_bounds_needed,
    invalidate_source_stats,
    persist_written_source_stats,
)
from batcher.config import Config, active_config, config_context
from batcher.io.source import Source, read_source

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcher.core import ExecutionContext
    from batcher.kyber.rules.selection import BuildSideDecision
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "DEFAULT_PARTITIONS",
    "approx_quantile",
    "auto_num_partitions",
    "collect_source_stats",
    "invalidate_source_stats",
    "partitions_from_physical",
    "persist_written_source_stats",
    "resolve_auto_config",
    "run_relational",
    "with_auto_config",
]

_R = TypeVar("_R")


def resolve_auto_config(config: Config | None = None) -> Config:
    """Return `config` with auto-sensed tunables filled in (a no-op `config` if none).

    When `memory.max_memory_bytes` is unset and `memory.unbounded_memory` is off, a
    concrete cap is sensed from the live envelope (host RAM / cgroup, via Carbonite's
    `PressureMonitor`) and frozen in — driving both the data plane's spill budget and
    the control plane's admission envelope, so a large query spills instead of OOMing
    with zero config. An explicit cap or `unbounded_memory=True` is returned untouched
    (the same object, so a caller can detect the no-op with ``is``).
    """
    cfg = config if config is not None else active_config()
    mem = cfg.memory
    if mem.max_memory_bytes is not None or mem.unbounded_memory:
        return cfg
    # `api` may consult Carbonite (it is the conductor); `config` may not.
    from batcher.carbonite.memory.pressure import PressureMonitor

    sensed = PressureMonitor(cfg).envelope_bytes()
    if sensed <= 0:
        return cfg  # could not sense — keep the safe unbounded fallback
    return dataclasses.replace(cfg, memory=dataclasses.replace(mem, max_memory_bytes=sensed))


def with_auto_config(fn: Callable[..., _R]) -> Callable[..., _R]:
    """Decorate a terminal entry point to run under the auto-resolved config.

    Fixes a query's sensed memory envelope once, at the materializing-terminal
    boundary (collect / write / stats and what delegates to them) — not per stage,
    where adaptive re-planning and the growing working set would drift it. A no-op
    when the user pinned the memory config or sensing is unavailable.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> _R:
        resolved = resolve_auto_config()
        if resolved is active_config():
            return fn(*args, **kwargs)
        with config_context(resolved):
            return fn(*args, **kwargs)

    return wrapper


def approx_quantile(batches: Iterable[pa.RecordBatch], column: str, q: float) -> float | None:
    """Approximate quantile `q` of `column` from a streamed, merged TDigest.

    Opt-in and explicitly approximate: tail-accurate (p99/p999) and far cheaper than
    an exact sort. Consumes `batches` one at a time — building a per-batch TDigest and
    merging the (tiny) sketches — so the column is never held whole on the driver; the
    caller projects to just `column` and streams it (single-node or distributed).
    Returns None if the column is non-numeric or empty.
    """
    from batcher import core

    sketches = [sk for b in batches if (sk := core.tdigest_partial([b], column)) is not None]
    return core.tdigest_quantile(sketches, q)


# --- Zero-config sizing -----------------------------------------------------
# When the user leaves a knob unset, fill it from the same analyses Kyber/Carbonite
# already produce rather than a blind constant — composing their decisions, never
# re-deriving them. The fallback is the historical default, used only when nothing
# about the data size is known.
DEFAULT_PARTITIONS = 16
_MIN_PARTITIONS = 4
_MAX_PARTITIONS = 4096


def _clamp_partitions(n: int) -> int:
    return max(_MIN_PARTITIONS, min(_MAX_PARTITIONS, n))


def partitions_from_physical(opt: PhysicalPlan) -> int | None:
    """Spill partition count implied by the optimized plan, or `None` if unsized.

    Reuses the per-breaker ``n_max_parallelism`` Kyber already computed (input rows
    / `target_rows_per_task`) — the same data-sized fan-out the distributed path
    uses — so out-of-core spilling shards by data volume instead of a blind 16.
    """
    widths = [op.bounds.n_max_parallelism for op in opt.ops if op.bounds.n_max_parallelism > 0]
    if not widths:
        return None
    return _clamp_partitions(max(widths))


# `auto_num_partitions` (the data-sized spill/shuffle partition count, learned-seeded) is an
# adaptive-sizing decision, so it lives in `api.tuning`; re-exported here for its callers.
from batcher.api.tuning import auto_num_partitions  # noqa: E402


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
    from batcher.config import active_config, config_context

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
    from batcher._internal.logging import ensure_configured, get_logger

    ensure_configured()
    _t0 = time.perf_counter()  # wall clock for the join-strategy bandit's per-run reward
    # Per-source statistics (footer/manifest/catalog) let the optimizer's zone-map
    # and null-driven rules prune predicates and skip files before execution. Reuse
    # the conductor's already-collected stats when present (the metadata-answer
    # attempt for a missed count()/is_empty() collected them), so a terminal op reads
    # each source's footer once across both passes.
    import os as _rp_os

    _rpp = _rp_os.environ.get("BATCHER_SORT_PROFILE")
    _rpt = time.perf_counter()
    source_stats = (
        ctx.source_stats
        if ctx.source_stats is not None
        else collect_source_stats(sources, ctx.hub, need_columns=column_bounds_needed(plan))
    )
    if _rpp:
        print(f"[rr] collect_source_stats {time.perf_counter() - _rpt:.1f}s", flush=True)
        _rpt = time.perf_counter()
    # Seed the distinct counts the optimizer is about to read: no file footer carries
    # `ndv`, so without this a query's *first* run orders its joins blind.
    from batcher.api.terminal._metadata import seed_column_ndv

    seed_column_ndv(ctx.hub, sources, plan)
    # One optimizer run yields both the physical plan (admission/costing) and the
    # optimized *logical* plan (the distributed / out-of-core executors read its derived
    # join keys + pushed predicates). Computing both here avoids re-running the entire
    # pipeline a second time via `optimize_logical` on those paths.
    opt, logical_opt, decisions = kyber.optimize_full(
        plan, sources=sources, hub=ctx.hub, source_stats=source_stats
    )
    if _rpp:
        print(f"[rr] kyber.optimize_full {time.perf_counter() - _rpt:.1f}s", flush=True)
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
    get_logger("api").debug("optimized %d ops; feasible=%s", len(opt.ops), verdict.feasible)
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
        if _rpp:
            print(f"[rr] carbonite.validate {time.perf_counter() - _rpt:.1f}s", flush=True)
            _rpt = time.perf_counter()
        workers, envelope = distributed_grant(rm, opt, plan, sources, ctx)
        if _rpp:
            print(f"[rr] distributed_grant {time.perf_counter() - _rpt:.1f}s", flush=True)
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
        if _rpp:
            print(f"[rr] execute_distributed {time.perf_counter() - _rpt:.1f}s", flush=True)
            _rpt = time.perf_counter()
        if prof is not None:
            prof.worker_metrics = wm
        # Core collects metadata on every path so later plans improve with use.
        from batcher.api.terminal._metadata import collect_source_metadata

        collect_source_metadata(ctx.hub, sources)
        if _rpp:
            print(f"[rr] collect_source_metadata {time.perf_counter() - _rpt:.1f}s", flush=True)
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
    if must_spill or rm.should_spill(opt):
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
    from batcher.metadata.io_stats import record_source_io

    resolved = []
    for i, src in enumerate(sources):
        _t0 = _time.perf_counter()
        batches = read_source(src, opt.source_projections.get(i), opt.source_predicates.get(i))
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
        resolved.append(batches)
        record_source_io(
            ctx.hub, _source_identity(src), sum(b.nbytes for b in batches), _elapsed_ms
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

    learn_column_stats(ctx.hub, resolved, sources, plan)
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
    return table, decisions
