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
        # Footer/manifest statistics are stable for a source's (immutable) file set, but a
        # source's `statistics()` re-reads + re-processes every row-group footer on each
        # call — ~9s for a 100-file TPC-H sf100 read, paid PER QUERY and dwarfing the actual
        # distributed run. Memoize by source identity for the session: stats only feed the
        # optimizer's cost/cardinality (never a result), so a stale entry can at worst pick
        # a slightly worse plan, never a wrong answer. Sources without an identity are not
        # cached (computed each time, as before).
        ident = _source_identity(s)
        if ident and ident in _SOURCE_STATS_CACHE:
            out.append(_SOURCE_STATS_CACHE[ident])
            continue
        stats = source_statistics(s)
        if stats is None and hub is not None:
            cached = load_source_stats(hub, ident)
            stats = replace(cached, exact_rows=False) if cached is not None else None
        if ident:
            _SOURCE_STATS_CACHE[ident] = stats
        out.append(stats)
    return out


# Session cache of per-source statistics, keyed by source identity (see collect_source_stats).
_SOURCE_STATS_CACHE: dict[str, object] = {}


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
        # pressure — result-invariant (a morsel only batches data).
        adapted = carbonite.ResourceManager(hub=ctx.hub).recommended_config()
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
        ctx.source_stats if ctx.source_stats is not None else collect_source_stats(sources, ctx.hub)
    )
    if _rpp:
        print(f"[rr] collect_source_stats {time.perf_counter() - _rpt:.1f}s", flush=True)
        _rpt = time.perf_counter()
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
            ctx.hub, plan, logical_opt, decisions, envelope.credits,
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
    from batcher.api.terminal._metadata import learn_column_stats

    learn_column_stats(ctx.hub, resolved)
    kyber.record_selectivity(ctx.hub, plan, sources, table.num_rows)
    # Close the conductor's tuning loops from this run's measured outcomes (breaker volume →
    # partition count, group reduction → pre-aggregation, join wall time → the bandit). Each
    # only steers a later performance choice, so recording is result-invariant.
    from batcher.api.tuning import record_run_feedback, total_source_rows

    record_run_feedback(
        ctx.hub, plan, logical_opt, decisions,
        out_rows=table.num_rows,
        input_rows=total_source_rows(sources),
        wall_ms=(time.perf_counter() - _t0) * 1000.0,
    )
    return table, decisions
