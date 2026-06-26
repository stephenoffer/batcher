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
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api._join_helpers import _empty_schema
from batcher.config import Config, active_config, config_context
from batcher.io.source import Source, read_source

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcher.core import ExecutionContext
    from batcher.kyber.rules.selection import BuildSideDecision
    from batcher.metadata.hub import MetadataHub
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "DEFAULT_PARTITIONS",
    "approx_quantile",
    "auto_num_partitions",
    "collect_source_stats",
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


def collect_source_stats(sources: list[Source], hub: MetadataHub | None) -> list:
    """Per-source `SourceStatistics`, from the source itself or the metadata cache.

    A source's own `statistics()` (footer/manifest/catalog) is authoritative for
    the file as it exists now. When a source declares none (a footerless CSV/JSON),
    fall back to statistics Batcher persisted when it *wrote* that path — but marked
    advisory (`exact_rows=False`), since the file may have changed since: cached
    stats sharpen cost and cardinality, they never answer an exact `count()`.
    """
    from dataclasses import replace

    from batcher.io.source import source_statistics
    from batcher.metadata.source_stats_store import load_source_stats

    out = []
    for s in sources:
        stats = source_statistics(s)
        if stats is None and hub is not None:
            cached = load_source_stats(hub, _source_identity(s))
            stats = replace(cached, exact_rows=False) if cached is not None else None
        out.append(stats)
    return out


def _source_identity(source: Source) -> str:
    identity_fn = getattr(source, "identity", None)
    return identity_fn() if callable(identity_fn) else ""


def persist_written_source_stats(table: pa.Table, path: str, fmt: str) -> None:
    """Persist a freshly-written result's statistics for a future read of `path`.

    Keyed by the read-side identity (`<fmt>:<path>`), so a later `read.<fmt>(path)`
    over a footerless format still finds an exact row count and per-column distinct
    estimates. Best-effort; never breaks a write.
    """
    from batcher import core
    from batcher.metadata.source_stats_store import save_source_stats
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    try:
        from batcher.config import active_config

        cols = table.schema.names
        ndv, _quants, _bytes = core.column_statistics(table.to_batches(), cols)
        index_on = active_config().optimizer.build_bloom_index
        blooms = _build_bloom_index(table, cols) if index_on else {}
        columns = {
            name: ColumnStat(
                ndv=float(ndv[name]) if ndv.get(name) else None,
                provenance=Provenance.SKETCH,
                bloom=blooms.get(name),
            )
            for name in cols
            if ndv.get(name) or blooms.get(name)
        }
        stats = SourceStatistics(
            row_count=table.num_rows, byte_size=table.nbytes, columns=columns, exact_rows=True
        )
        save_source_stats(core.default_hub(), f"{fmt}:{path}", stats)
    except Exception:  # pragma: no cover - persistence must never break a write
        pass


def _build_bloom_index(table: pa.Table, cols: list[str]) -> dict[str, bytes]:
    """A per-column membership bloom for each indexable (int/text) column — the
    data-skipping index `zonemap_prune_filter` consults for equality/`IN`. Built in
    Rust over the result already in memory; unindexable columns yield no entry."""
    import batcher._native as nat

    batches = table.to_batches()
    out: dict[str, bytes] = {}
    for i, name in enumerate(cols):
        bloom = nat.build_column_bloom(batches, i, max(1, table.num_rows))
        if bloom is not None:
            out[name] = bloom
    return out


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


def auto_num_partitions(plan: LogicalPlan, sources: list[Source], hub: MetadataHub | None) -> int:
    """Data-sized spill partition count for `plan` (used when the user gives none).

    Estimates the plan's input cardinality with Kyber's `CardinalityEstimator`
    (sharpened by any learned stats in `hub`) and targets ~`target_rows_per_task`
    rows per partition — the same sizing rule Kyber uses for breaker parallelism.
    Falls back to `DEFAULT_PARTITIONS` when the size is unknown.
    """
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator

    try:
        learned = load_learned_stats(hub) if hub is not None else None
        est = CardinalityEstimator(sources=sources, learned=learned)
        rows = est.estimate(plan).rows
        opt = active_config().optimizer
        target = opt.target_rows_per_task
        if rows <= 0 or target <= 0:
            return DEFAULT_PARTITIONS
        row_parts = math.ceil(rows / target)
        # Also shard by bytes so a few wide rows (GB blobs/embeddings) don't land a
        # huge partition on one task: take the larger of the row- and byte-derived
        # counts. Width is the flat default until measured, so narrow data is
        # unaffected (byte_parts <= row_parts there).
        width = est.row_width(plan, opt.row_bytes)
        byte_target = max(1, opt.target_bytes_per_task)
        byte_parts = math.ceil(rows * width / byte_target)
        return _clamp_partitions(max(row_parts, byte_parts))
    except Exception:  # pragma: no cover - sizing must never break a query
        return DEFAULT_PARTITIONS


# The estimator's reserved key for per-column distinct counts (see
# `kyber.cardinality` / `kyber.learning`). Used here only to skip already-measured
# columns so the sketch build never repeats.
_NDV_KEY = "__column_ndv__"


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
        adapted = carbonite.ResourceManager().recommended_config()
        if adapted is not None:
            scope = config_context(adapted)
    with scope:
        return _run_relational(plan, sources, ctx, distributed=distributed, materialize=materialize)


def _run_relational(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    *,
    distributed: bool = False,
    materialize: bool = True,
) -> tuple[pa.Table | Source, list[BuildSideDecision]]:
    """The Kyber → Carbonite → Core body, run under the (possibly adapted) config."""
    from batcher import carbonite, core, kyber
    from batcher._internal.logging import ensure_configured, get_logger

    ensure_configured()
    # Per-source statistics (footer/manifest/catalog) let the optimizer's zone-map
    # and null-driven rules prune predicates and skip files before execution. Reuse
    # the conductor's already-collected stats when present (the metadata-answer
    # attempt for a missed count()/is_empty() collected them), so a terminal op reads
    # each source's footer once across both passes.
    source_stats = (
        ctx.source_stats if ctx.source_stats is not None else collect_source_stats(sources, ctx.hub)
    )
    opt, decisions = kyber.optimize_traced(
        plan, sources=sources, hub=ctx.hub, source_stats=source_stats
    )
    prof = ctx.profile
    if prof is not None:
        from batcher.api.terminal.profile import record_plan

        record_plan(prof, opt, plan, distributed, decisions)

    rm = carbonite.ResourceManager()
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

        envelope = rm.scheduling_envelope(opt, ctx.num_workers)
        # Profiling: collect the workers' map sub-plan metrics (their own profile section).
        wm: list = []
        result = dist.execute_distributed(
            plan,
            sources,
            ctx.num_workers,
            transport=ctx.transport,
            envelope=envelope,
            hub=ctx.hub,
            materialize=materialize,
            metrics_out=wm if prof is not None else None,
        )
        if prof is not None:
            prof.worker_metrics = wm
        # Core collects metadata on every path so later plans improve with use.
        _collect_source_metadata(ctx.hub, sources)
        return result, decisions

    # Carbonite decides out-of-core: if the estimated working set won't fit the
    # memory envelope (admission counter-offer or the spill estimate), run the
    # partition-and-spill executor so the query completes under bounded memory
    # instead of OOMing. Shapes with no spilling path fall through to in-memory —
    # unless admission already proved it won't fit, in which case that is a real
    # infeasibility rather than a silent OOM.
    if must_spill or rm.should_spill(opt):
        from batcher.dist.spill import spill_collect

        # Shard the out-of-core spill by data volume (Kyber's per-breaker fan-out),
        # not a blind constant, so a bigger group-by/join uses more, smaller buckets.
        partitions = partitions_from_physical(opt) or DEFAULT_PARTITIONS
        if prof is not None:
            from batcher.api.terminal.profile import record_spill

            record_spill(prof, partitions)
        spilled = spill_collect(plan, sources, partitions)
        if spilled is not None:
            kyber.record_execution(ctx.hub, plan, spilled.num_rows)
            return spilled, decisions
        if must_spill:
            raise PlanError(
                "plan does not fit the memory envelope and has no out-of-core path "
                f"(binding constraint: {verdict.binding_constraint})"
            )

    # Resolve lazy sources to Arrow batches (reads happen here, not earlier).
    # Projection + predicate pushdown tell each source what to read.
    resolved = [
        read_source(
            src,
            opt.source_projections.get(i),
            opt.source_predicates.get(i),
        )
        for i, src in enumerate(sources)
    ]
    # Reserve the estimated envelope against the process-wide buffer pool for the
    # duration of execution, so concurrent queries draw on one budget. If the
    # reservation does not fit (concurrent queries already over budget), prefer the
    # out-of-core path over racing them into an OOM — reserve-before-allocate is only
    # real if a `False` actually changes behavior (C30/C31).
    with rm.reserve(rm.estimated_bytes(opt)) as granted:
        if not granted:
            from batcher.dist.spill import spill_collect

            parts = partitions_from_physical(opt) or DEFAULT_PARTITIONS
            spilled = spill_collect(plan, sources, parts)
            if spilled is not None:
                kyber.record_execution(ctx.hub, plan, spilled.num_rows)
                return spilled, decisions
        # When profiling, take the metered path (still feeding the hub) so the per-operator
        # `ExecMetrics` reach the conductor's `QueryProfile`; otherwise the plain path,
        # which skips even the tiny metrics serialization — keeping an ordinary run intact.
        if prof is not None:
            batches, metric_ops = core.execute_local_metered(opt, resolved, feedback=ctx.hub)
            prof.metric_ops = metric_ops
        else:
            batches = core.execute_local(opt, resolved, feedback=ctx.hub)
    table = pa.Table.from_batches(
        batches, schema=batches[0].schema if batches else _empty_schema(ctx.columns)
    )
    # Feed the measured output size back to the learner for next time, learn
    # per-column distinct counts / quantiles from the scanned input, and record the
    # filter's measured selectivity (a ratio that generalizes across input sizes) —
    # so later plans get sketch- and feedback-driven cardinality.
    kyber.record_execution(ctx.hub, plan, table.num_rows)
    _learn_column_stats(ctx.hub, resolved)
    kyber.record_selectivity(ctx.hub, plan, sources, table.num_rows)
    return table, decisions


def _collect_source_metadata(hub, sources: list[Source]) -> None:
    """Record per-column ndv/quantiles from the base sources (Core collects).

    The UDF and distributed paths don't surface their scanned batches the way the
    native path hands `resolved` to `_learn_column_stats`, so this reads the base
    sources directly. It is gated on the cheap `Source.schema` — a source is only
    read when it has a not-yet-measured column — so a file is never re-scanned once
    its columns are learned. Best-effort: learning never breaks a query.
    """
    if hub is None:
        return
    from batcher import kyber

    try:
        known = set(kyber.load_learned_stats(hub).get(_NDV_KEY, {}))
        resolved = [
            read_source(src, None, None)
            for src in sources
            if any(c not in known for c in src.schema().names)
        ]
        if resolved:
            _learn_column_stats(hub, resolved)
    except Exception:  # pragma: no cover - learning must never break execution
        pass


def _learn_column_stats(hub, resolved: list[list[pa.RecordBatch]]) -> None:
    """Measure per-column ndv/quantiles from the just-scanned input and record them.

    Gated to columns not already known, so the O(rows) sketch build happens at most
    once per column — a bounded, one-time cost that sharpens every later plan. Core
    measures (`core.column_statistics`); Kyber persists/consumes. Best-effort: a
    failure here never affects the query result.
    """
    if hub is None:
        return
    from batcher import core, kyber

    try:
        known = set(kyber.load_learned_stats(hub).get(_NDV_KEY, {}))
        min_frac = active_config().optimizer.cardinality.mcv_min_fraction
        ndv_all: dict[str, float] = {}
        quant_all: dict[str, dict[str, list[float]]] = {}
        bytes_all: dict[str, float] = {}
        mcv_all: dict[str, dict[str, float]] = {}
        for batches in resolved:
            if not batches:
                continue
            cols = [c for c in batches[0].schema.names if c not in known]
            if not cols:
                continue
            ndv, quants, avg_bytes = core.column_statistics(batches, cols)
            ndv_all.update(ndv)
            quant_all.update(quants)
            bytes_all.update(avg_bytes)
            # MCV: a skew value's measured frequency sharpens `col = value` past 1/ndv.
            total = sum(b.num_rows for b in batches)
            for col_name, hits in core.heavy_hitters(batches, cols, min_frac).items():
                if total > 0 and hits:
                    mcv_all[col_name] = {str(v): n / total for v, n in hits}
        if ndv_all or quant_all or bytes_all or mcv_all:
            kyber.record_column_stats(hub, ndv_all, quant_all, bytes_all, mcv_all)
    except Exception:  # pragma: no cover - learning must never break execution
        pass
