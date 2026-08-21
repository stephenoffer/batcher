"""Terminal/materialization operations for `Dataset`.

These free functions own the orchestration of the three layers (Kyber → Carbonite
→ Core) for terminal operations. `Dataset`'s terminal methods are thin wrappers
that forward their state (`self._plan`, `self._sources`, `self.columns`) here.
"""

from __future__ import annotations

import time
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher._internal.optional import require
from batcher.api.orchestration import with_auto_config
from batcher.api.terminal.metadata_answer import (
    metadata_aggregate_table,
    metadata_count,
    metadata_empty_table,
    metadata_is_empty,
)
from batcher.api.terminal.routing import resolve_distributed as _resolve_distributed
from batcher.config import active_config
from batcher.io.base._layout import FileLayout
from batcher.io.manifest import WriteManifest
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan
from batcher.plan.types import logical_bytes

__all__ = [
    "_collect",
    "_count",
    "_explain",
    "_is_empty",
    "_resolve_distributed",
    "_schema",
    "_show",
    "_stats",
    "_to_pandas",
    "_to_polars",
    "_to_pydict",
    "_to_pylist",
    "_write",
]


@with_auto_config
def _collect(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    distributed: bool | str = "auto",
    num_workers: int | None = None,
    spill: bool = False,
    num_partitions: int | None = None,
    adaptive: bool | str = "auto",
    transport: str = "auto",
    cache: bool = False,
    source_stats: list | None = None,
    backend: str = "cpu",
) -> pa.Table:
    """Execute the plan and materialize the result as a `pyarrow.Table`.

    Zero-config by default: `distributed`, spill, worker fan-out, and partition
    count are all decided from the plan's estimated size and the live cluster —
    each argument is an optional override. `num_partitions=None` derives a
    data-sized count for forced spills.

    Raises `PlanError` if any source is unbounded (streaming) — there is no finite
    result to materialize. This guards every materializing terminal (`collect`,
    `count`, `to_*`, `show`), so they fail fast instead of hanging.

    The exception is a top-level `LIMIT n` over a breaker-free pipeline. That result *is*
    finite -- n rows -- and the streaming router already stops reading once it has them, so
    `bt.read.kafka(...).head(10).to_pydict()` is bounded in both memory and time. It is
    also the first thing anyone types against an unfamiliar topic, and it used to be
    answered by advice to restructure the query. A limit over a *sort* is still refused:
    top-N is finite too, but only after the stream ends, which it does not.
    """
    from batcher.io.source import is_bounded

    # A prepared hit answers before any of the routing below runs, which is the point: for a
    # re-issued small query every one of those guards recomputes an answer the entry already
    # holds. The key carries the arguments, and the entry re-proves its sources and config on
    # every hit, so a hit is only ever the same derivation replayed. Miss (the default, and
    # always when `execution.fast_path` is off) costs one dict lookup.
    prepared_key = None
    if source_stats is None and active_config().execution.fast_path:
        from batcher.api.orchestration import prepared as _prepared

        prepared_key = _prepared.entry_key(
            plan,
            sources,
            (
                distributed,
                num_workers,
                spill,
                num_partitions,
                adaptive,
                transport,
                cache,
                backend,
                tuple(columns),
            ),
        )
        hit = _prepared.lookup(prepared_key, sources, active_config())
        if hit is not None:
            return hit.execute(sources)

    if any(not is_bounded(s) for s in sources) and not _is_bounded_peek(plan):
        raise PlanError(
            "this operation materializes the full result, but the dataset has an "
            "unbounded (streaming) source. Consume it with iter_batches(), take a bounded "
            "peek with head(n), or write it to a sink instead."
        )
    if any(not is_bounded(s) for s in sources):
        from batcher.api.terminal.stream import _iter_batches

        peeked = list(_iter_batches(plan, sources, columns))
        schema = peeked[0].schema if peeked else _peek_schema(plan)
        return pa.Table.from_batches(peeked, schema=schema)
    # Collect source statistics once when the metadata-aggregate attempt could use
    # them (a keyless aggregate), so a *missed* attempt doesn't re-read every footer
    # during execution. A non-aggregate collect skips this entirely (the attempt
    # returns at its cheap structural guard). `count()`/`is_empty()` pass theirs in.
    if source_stats is None:
        from batcher.api.terminal.metadata_answer import is_global_aggregate

        if is_global_aggregate(plan):
            from batcher import core
            from batcher.api.orchestration import collect_source_stats

            # Only MIN/MAX read a source's column bounds (and only for the aggregated
            # column); COUNT answers from the row count, and SUM/MEAN/COUNT DISTINCT from
            # the source's lazy per-column methods — none of which need the O(rows)
            # zone-map scan. So a keyless SUM/COUNT over a fresh in-memory source no longer
            # pays to build bounds it never reads, and a MIN(x) scans only column x.
            source_stats = collect_source_stats(
                sources, core.default_hub(), need_columns=_global_agg_bound_columns(plan)
            )
    metadata = metadata_aggregate_table(plan, sources, source_stats)
    if metadata is not None:
        return metadata
    # Provably-empty short-circuit: a scan-free empty table (contradiction / limit(0) /
    # empty-side join) when metadata proves zero rows (see `metadata_empty_table`).
    if (empty := metadata_empty_table(plan, sources, source_stats)) is not None:
        return empty
    # GPU backend. `backend="gpu"` forces the GPU for any supported shape (honoring the user past
    # the small-input threshold, but Kyber still routes single-device vs sharded by working-set
    # size); `backend="auto"` lets Kyber's cost policy decide GPU vs CPU fully. Anything else — an
    # unsupported shape, a GPU-less cluster, or data Kyber routes to the CPU — silently uses the
    # CPU engine, so both are always safe. Same result, different *where*.
    if backend not in ("cpu", "gpu", "auto"):
        raise PlanError(f"backend must be 'cpu', 'gpu', or 'auto', got {backend!r}")
    if backend in ("gpu", "auto"):
        from batcher import core
        from batcher.api.terminal.gpu_backend import try_gpu_collect

        gpu_result = try_gpu_collect(
            plan, sources, core.default_hub(), force=(backend == "gpu"), columns=columns
        )
        if gpu_result is not None:
            return gpu_result
    # Opt-in: offload large-payload columns out of line around breakers (the blobs ride
    # through as tiny handles). Inserted before execution routing so the resulting
    # `map_batches` stages take the same mixed-executor path as an explicit offload.
    from batcher.api.terminal.blob_offload import maybe_insert_blob_offload

    plan = maybe_insert_blob_offload(plan)
    distributed = _resolve_distributed(distributed, plan, sources)
    # Resolve `adaptive="auto"` to a concrete decision before the fast-path checks
    # below ("auto" is a truthy string). Join-less plans short-circuit to False without
    # touching source stats, so the common path pays nothing. `distributed` is passed so
    # a shape the one-shot dispatcher can't route (a 3+-table join) takes the staged path
    # rather than raising.
    from batcher import core
    from batcher.api.adaptive import resolve_adaptive

    adaptive = resolve_adaptive(
        adaptive, plan, sources, core.default_hub(), distributed=distributed
    )

    # `head(n)` / `limit(n)` over a breaker-free pipeline reads the source only until
    # `n` rows are produced, then stops — no whole-source scan (Ray's `limit` does not
    # short-circuit). Only on the plain single-node path; the distributed / adaptive /
    # spill paths keep their own routing.
    if not distributed and not adaptive and not spill and len(sources) == 1:
        from batcher import core
        from batcher.plan.logical import Limit, is_streamable

        if (
            isinstance(plan, Limit)
            and is_streamable(plan.input)
            and not core.has_map_batches(plan.input)
        ):
            from batcher.api._join_helpers import _empty_result_schema
            from batcher.api.terminal.stream.dispatch import _pushdown
            from batcher.core.streaming import stream_limit

            batches = list(stream_limit(plan, sources[0], projection=_pushdown(plan)))
            schema = batches[0].schema if batches else _empty_result_schema(plan, columns)
            return pa.Table.from_batches(batches, schema=schema)

    if adaptive:
        from batcher import core
        from batcher.api.adaptive import execute_adaptive, record_adaptive_route

        # Adaptive re-optimization now works distributed too: each breaker stage
        # fans out across workers and its measured cardinality re-plans the rest.
        _t0 = time.perf_counter()
        table = execute_adaptive(
            plan,
            sources,
            core.default_hub(),
            distributed=distributed,
            num_workers=num_workers,
            transport=transport,
        ).table
        # Feed the staged arm of the route bandit. This is the *only* place either arm is
        # measured, and it has to be the whole query rather than a per-stage total, because
        # what the two routes differ in is precisely the work that is not inside a stage:
        # the per-stage materialize, re-plan, and the fusion and parallel width the pipeline
        # gives up by being cut at every breaker.
        record_adaptive_route(core.default_hub(), plan, True, (time.perf_counter() - _t0) * 1000.0)
        return table

    if spill and not distributed:
        from batcher import core, kyber
        from batcher.api.orchestration import auto_num_partitions
        from batcher.api.orchestration.run import record_cardinality_outcome
        from batcher.dist.spill import spill_collect

        hub = core.default_hub()
        partitions = num_partitions or auto_num_partitions(plan, sources, hub)
        # Spill the optimized plan (COUNT(DISTINCT)→COUNT over DISTINCT; derived join keys).
        opt_lp = kyber.optimize_logical(plan, sources=sources, hub=hub)
        spilled = spill_collect(opt_lp, sources, partitions)
        if spilled is not None:
            # This route bypasses `run_relational`, so it must close its own loops — it
            # recorded nothing at all, which made an explicit `spill=True` the one way to run
            # a query that taught the optimizer literally nothing. The whole-input measures
            # (`learn_column_stats`) still cannot run: an out-of-core scan never holds the
            # batches to measure.
            record_cardinality_outcome(hub, plan, sources, spilled.num_rows)
            return spilled
        # Other plan shapes have no spilling path → fall through to in-memory.

    # Opt-in small-query fast path (`execution.fast_path`, off by default). Placed here
    # because every routing decision above it is now resolved: a plan that reaches this line
    # is single-node, non-adaptive, CPU, and did not take the spill or streaming-limit route,
    # which is most of what `eligible` would otherwise have to re-derive. Same optimized
    # plan, same engine call, same rows — it skips the orchestration below, not the work.
    from batcher.api.orchestration.fast_path import eligible, run_fast

    if eligible(
        plan,
        sources,
        distributed=distributed,
        adaptive=bool(adaptive),
        spill=spill,
        backend=backend,
        cache=cache,
    ):
        return run_fast(plan, sources, columns, remember_as=prepared_key)

    # Imported here (not at module load) to keep the layer-import contract
    # simple and avoid importing the engine for pure-Python tooling. `time` is not one of
    # those: it is stdlib, already loaded, and re-importing it on every terminal op bought
    # nothing.
    from batcher import core
    from batcher._internal import events
    from batcher.api import executors
    from batcher.api.terminal.event_log import (
        event_log_collector,
        pipeline_signature,
        query_label,
        report_failure,
        start_query_report,
        write_event_log,
    )
    from batcher.observe import ensure_sinks

    ensure_sinks()  # attach the progress bar / dashboard the config asks for (idempotent)
    query_id = start_query_report(query_label(plan), pipeline_signature(plan))

    ctx = core.ExecutionContext(
        columns=columns,
        hub=core.default_hub(),
        num_workers=num_workers,
        transport=transport,
        cache=cache,
        source_stats=source_stats,
        profile=event_log_collector(),
    )
    t0 = time.perf_counter()
    try:
        # Make the id ambient for the whole execution. The subsystems with the most to say
        # about a distributed query — `dist` deciding a fan-out, a placement, a transport —
        # are the furthest from where the id is minted and must not import `api` to ask for
        # it, so their events used to reach the bus with an empty id and be dropped by
        # `observe.store` (which discards any event naming no live query, by design).
        with events.query_scope(query_id):
            table = executors.select(plan, distributed=distributed).execute(plan, sources, ctx)
    except BaseException as exc:
        # A failed query must close out on the bus too, or its progress bar spins forever
        # and the dashboard shows it as still running. Re-raised unchanged — reporting the
        # failure must not alter it.
        report_failure(query_id, total_ms=(time.perf_counter() - t0) * 1000.0, exc=exc)
        raise
    total_ms = (time.perf_counter() - t0) * 1000.0
    # `plan`/`sources` are what let a `map_batches`/ML pipeline report at all: it has no
    # engine IR, so the profile has to be assembled against the logical tree instead.
    write_event_log(
        ctx.profile,
        total_ms=total_ms,
        rows=table.num_rows,
        query_id=query_id,
        plan=plan,
        sources=sources,
    )
    from batcher.api.adaptive import record_adaptive_route
    from batcher.api.terminal.gpu_backend import record_cpu_crossover  # adaptive-crossover sample

    record_cpu_crossover(plan, sources, ctx.hub, total_ms)  # gated to a GPU cluster; else no-op
    record_adaptive_route(ctx.hub, plan, False, total_ms)  # the one-shot arm of the route bandit
    return table


@with_auto_config
def _explain(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    *,
    analyze: bool = False,
    fmt: str = "text",
) -> str:
    """Render the plan as a tree (planned, or measured when `analyze`); `with_auto_config`
    so an analyzed run profiles under the same sensed config `collect()`/`stats()` use."""
    from batcher.api.terminal.profile import explain

    return explain(plan, sources, columns, analyze=analyze, fmt=fmt)


def _global_agg_bound_columns(plan: LogicalPlan) -> set[str] | None:
    """The columns whose min/max bounds a keyless aggregate's metadata answer reads.

    Only ``MIN``/``MAX`` read bounds (from `SourceStatistics.columns`), and only of their
    own input column; ``COUNT`` answers from the row count and ``SUM``/``MEAN``/``COUNT
    DISTINCT`` from the source's lazy per-column methods. So the needed set is the plain
    input columns of the min/max aggregates — empty when there are none (skip the scan),
    and `None` (collect everything, be safe) when the shape isn't the expected keyless
    aggregate or a min/max input is a computed expression rather than a bare column.
    """
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Aggregate, Project

    node = plan
    while isinstance(node, Project):
        node = node.input
    if not isinstance(node, Aggregate):
        return None  # not the expected shape — be safe and collect full bounds
    needed: set[str] = set()
    for spec in node.aggregates:
        if spec.agg.func in ("min", "max"):
            if isinstance(spec.agg.input, Col):
                needed.add(spec.agg.input.name)
            else:
                return None  # min/max over a computed expr — fall back to full bounds
    return needed


def _shared_source_stats(plan: LogicalPlan, sources: list[Source]) -> list | None:
    """Source statistics to share across a metadata attempt and its execution fallback.

    Collected once, only when a metadata answer may even be attempted (a bounded,
    non-UDF plan) — exactly the case where the relational execution fallback would
    also read them. Returns `None` otherwise, leaving each path to collect its own,
    so an opaque UDF/streaming terminal pays nothing extra.
    """
    from batcher.api.terminal.metadata_answer._core import _metadata_answerable

    if not _metadata_answerable(plan, sources):
        return None
    from batcher import core
    from batcher.api.orchestration import collect_source_stats
    from batcher.api.source_stats import column_bounds_needed

    # `count()`/`is_empty()` answer from the row count and, when filtered, from the
    # predicate columns' bounds (comparison-empty detection). So only those columns need
    # a bounds scan — a `count()` over a wide unfiltered relation stays a cheap row count.
    return collect_source_stats(
        sources, core.default_hub(), need_columns=column_bounds_needed(plan)
    )


def _count(plan: LogicalPlan, sources: list[Source], _columns: list[str]) -> int:
    """Return the number of result rows — from metadata when provable, else a `COUNT(*)`.

    Metadata answers a derivable count (`limit(n)`, a global aggregate, empty source,
    row-preserving operators) with no execution. Otherwise a global `COUNT(*)` runs (via
    `global_count_plan`): projection pushdown reads only the filter/key columns and a count
    over a `Filter` fuses into one `count_if` pass, so no result rows are materialized.
    """
    from batcher.api.terminal.metadata_answer import global_count_plan, pushed_count

    source_stats = _shared_source_stats(plan, sources)
    answer = metadata_count(plan, sources, source_stats)
    if answer is not None:
        _record_count_selectivity(plan, sources, answer)
        return answer
    # A source that can count itself, asked only now: the free answers have declined, and
    # the remaining option is to read the relation to count it. One `COUNT(*)` round trip
    # beats transferring every column of every row to produce an integer.
    answer = pushed_count(plan, sources)
    if answer is not None:
        _record_count_selectivity(plan, sources, answer)
        return answer
    if _is_bounded_peek(plan):
        # `global_count_plan` wraps the peek in an aggregate, and an aggregate over an
        # endless source is exactly what `_collect` refuses. Counting the peek's own rows
        # is bounded by `n` and is what the caller asked; without this, `head(10)` could be
        # materialized but not counted, which is an inconsistency nobody could explain.
        return _collect(plan, sources, _columns).num_rows
    table = _collect(global_count_plan(plan), sources, ["n"], source_stats=source_stats)
    count = int(table.column("n")[0].as_py()) if table.num_rows else 0
    _record_count_selectivity(plan, sources, count)
    return count


def _record_count_selectivity(plan: LogicalPlan, sources: list[Source], count: int) -> None:
    """Warm the learned-selectivity cache from a measured `count()`.

    A `count()` over a `Filter` is an *exact* measurement of the filter's surviving-row
    count — which `record_selectivity` turns into a per-signature selectivity ratio
    (`count / scan rows`) that sharpens later plans of the same shape. The execution path
    records feedback off the aggregate-*wrapped* count plan (whose Filter is hidden under
    the `Aggregate`), so this closes that gap directly with the original filter plan.
    Best-effort; never raises into the terminal."""
    from batcher import core, kyber

    kyber.record_selectivity(core.default_hub(), plan, sources, count)


def _is_empty(plan: LogicalPlan, sources: list[Source], columns: list[str]) -> bool:
    """Whether the result has no rows, from metadata when provable, else execute.

    Falls back to a single-row probe (`limit(1)`), which the streaming early-stop
    reads without scanning the whole source.
    """
    from batcher.plan.logical import Limit

    source_stats = _shared_source_stats(plan, sources)
    answer = metadata_is_empty(plan, sources, source_stats)
    if answer is not None:
        return answer
    # The `limit(1)` probe runs over the same sources, so their stats still apply.
    return _collect(Limit(plan, 1), sources, columns, source_stats=source_stats).num_rows == 0


def _schema(plan: LogicalPlan, sources: list[Source], columns: list[str]) -> pa.Schema:
    """The output Arrow schema without scanning rows.

    A bare scan returns its source schema, normalized the way the FFI boundary will
    normalize it. Otherwise the plan's type-carrying `available_schema()` analysis answers
    without touching the engine when it can infer every output type; anything it leaves
    uncertain falls back to a zero-row execution (`limit(0)`), which the engine answers
    without materializing data.

    The `widen` on the scan arm is what keeps all three arms agreeing. The other two
    already predict the boundary's normalization — `available_schema()` through
    `plan.types`, and the `limit(0)` fallback by actually executing — while this one
    handed back the source's own types. A dictionary-encoded column (what Parquet emits
    natively for a low-cardinality string) was therefore reported as
    `dictionary<values=string, ...>` by `Dataset.schema` when `collect()` returns plain
    `string`, so the cheapest arm was the only one that lied.
    """
    from batcher.plan.logical import Limit, Scan
    from batcher.plan.types import widen

    if isinstance(plan, Scan) and len(sources) == 1:
        source_schema = sources[0].schema()
        return pa.schema(
            [f.with_type(widen(f.type)) for f in source_schema],
            metadata=source_schema.metadata,
        )
    inferred = plan.available_schema()
    if inferred is not None:
        return inferred.arrow
    return _collect(Limit(plan, 0), sources, columns).schema


@with_auto_config
def _stats(plan: LogicalPlan, sources: list[Source], columns: list[str]):
    """Execute through the real path (single-node/spill/distributed) and return `RunStats`.

    Raises `PlanError` for an unbounded source. A `map_batches`/ML pipeline is measured per
    stage against its logical tree (see `run_profiled`), so it reports rows and time per
    stage rather than refusing.
    """
    from batcher.api.stats import RunStats
    from batcher.api.terminal.profile import run_profiled

    profile = run_profiled(plan, sources, columns)
    return RunStats.from_profile(profile)


def _sink_owns_its_layout(sink: object, path: str) -> bool:
    """Whether this sink's file layout is a property of the *target*, not of this write.

    `partitions_itself` is the flat form: a sink whose layout never comes from the call site
    (a partitioned Iceberg table declares its spec in the catalog). `requires_partitioned_write`
    is the form that has to look: a Delta table is partitioned or not depending on how it was
    created, so only the table can answer, and the answer decides whether this write may take
    the single-file path.

    It matters because Delta keeps a partition value in the file's *directory name*. A write
    that takes the single-file path into a partitioned table does not merely lay the file out
    differently — the value is nowhere, and the rows read back null in the column the table is
    organised by.

    Args:
        sink: The format sink about to be written through.
        path: The write's destination root.

    Returns:
        True when this write must go through the partitioned path.
    """
    ask = getattr(sink, "requires_partitioned_write", None)
    if ask is not None:
        return bool(ask(path))
    return bool(getattr(sink, "partitions_itself", False))


def _destination_is_a_directory(sink: object, path: str) -> bool:
    """Whether `path` already exists as a directory this file sink would have to replace.

    Only asked of a `FileSink`: a warehouse/table sink's "path" is an identifier, not a
    location, and stat-ing it is meaningless. Best-effort — a filesystem that cannot answer
    leaves the write on the path it would have taken anyway.

    Args:
        sink: The sink about to be written through.
        path: The write's destination.

    Returns:
        True when the destination is an existing directory.
    """
    from batcher.io.base.sink import FileSink

    if not isinstance(sink, FileSink):
        return False
    try:
        from batcher.io.filesystem import resolve_filesystem

        return resolve_filesystem(path).is_dir(path)
    except Exception:
        return False


def _streaming_write_eligible(
    plan: LogicalPlan,
    sources: list[Source],
    distributed: bool,
    partition_by: list[str] | None,
    max_rows_per_file: int | None,
    num_files: int | None,
    target_bytes_per_file: int | None,
    sink: object = None,
    path: str = "",
) -> bool:
    """Whether `_write` can stream the result to the sink instead of collecting it.

    Eligible when a single-node, breaker-free plan reads a *lazy* source (file /
    iterator): the batches stream straight to the sink, bounding driver memory to one
    batch. A fully-resident in-memory source gains nothing (its data is already in RAM),
    so it keeps the collect path — which also persists per-column sketch statistics (a
    full pass the streaming path can't do) for a later read.

    A `max_rows_per_file` cap streams too, as a rollover point (`write_stream_parts`).
    That combination used to be excluded, which had it exactly backwards: a caller who
    caps the file size is usually a caller whose result does not fit on the driver, and
    the cap was what forced the whole thing to materialize there first.

    `num_files` and `target_bytes_per_file` genuinely need the whole result — both are
    computed from its total size — so they stay on the collect path, as does
    `partition_by`, which has to fan rows out by key.

    A plan with a **breaker** (a sort, a join, a group-by) streams too, but only under a
    `max_rows_per_file` cap. The router (`_iter_batches`) yields such plans in bounded
    memory — a top-level aggregate or top-N folds, a sort / join / window comes off the
    out-of-core bucket pipeline, a UNION ALL goes branch by branch — so the read side is not
    what stops it. The *write* side is: a single-file write ends by persisting the result's
    statistics for a later read of this path, and computing those needs the whole table. The
    cap is exactly the case where it would not have persisted them anyway (that write goes
    to the parts path), so nothing is given up there — and it is the caller most likely to
    need this, since capping the file size usually means the result does not fit on the
    driver, which is what the cap used to force it to do.
    """
    from batcher.io.source import InMemorySource, MaterializedSource
    from batcher.plan.logical import is_streamable

    if distributed or partition_by:
        return False
    if sink is not None and _sink_owns_its_layout(sink, path):
        return False  # a partitioned target needs the whole table to fan out by key
    if num_files is not None or target_bytes_per_file is not None:
        return False
    if max_rows_per_file is not None and not hasattr(sink, "write_stream_parts"):
        return False
    if not is_streamable(plan) and max_rows_per_file is None:
        return False
    return not all(isinstance(s, InMemorySource | MaterializedSource) for s in sources)


def _report_write(manifest: WriteManifest, fmt: str) -> None:
    """Announce a committed write on the event bus.

    The read side of a job has always been countable and the write side never was, which is
    backwards for an ETL job: the thing it exists to produce is the thing nothing measured.
    Every sink already returns a `WrittenFile` per file carrying its rows and its size on
    storage, and `WriteManifest` already rolls them up — the numbers existed and stopped at
    whoever held the manifest.

    Published here because this is the one funnel every write branch already routes through,
    which is also why it is one event per commit rather than one per file: a partitioned
    write producing ten thousand files must not produce ten thousand events.

    A no-op when nothing is listening, and never able to fail a write that has already
    committed — the data is on storage by this point, and no counter is worth losing it over.
    """
    from batcher._internal import events

    if not events.listening():
        return
    try:
        events.publish(
            events.WRITE,
            name=fmt or "unknown",
            files=manifest.num_files,
            rows=manifest.total_rows,
            bytes=manifest.total_bytes,
        )
    except Exception as exc:  # pragma: no cover - telemetry must never fail a commit
        from batcher._internal.logging import note_suppressed

        note_suppressed("api", "report the write on the event bus", exc)


def _commit(
    sink,
    manifest: WriteManifest,
    path: str,
    fmt: str,
    schema: pa.Schema | None = None,
    *,
    auto_compact: bool = False,
) -> WriteManifest:
    """Commit the write, then drop any statistics this session cached for `path`.

    Those statistics are a zone map the optimizer prunes predicates and join sides with,
    so serving the *previous* version's after a rewrite produces a wrong answer rather
    than a slow plan. Every copy-on-write pattern — `write.merge`, `ds.scd.*`,
    `ds.scd.apply_changes` — rewrites a path it also reads, so this is the common case,
    not a corner one. Routed through one helper because `_write` commits from three
    branches and a missed one is a silent wrong answer.

    `schema` is the write's output schema. A transactional sink creating a table needs
    it and cannot recover it from the data files alone — a partitioned write stores its
    partition columns in the path, not the file — so the driver, which knows the plan's
    output type, attaches it here for every branch.
    """
    import dataclasses

    from batcher.api.orchestration import invalidate_source_stats

    if schema is not None and manifest.schema is None:
        manifest = dataclasses.replace(manifest, schema=schema)
    sink.commit(manifest, path)
    _report_write(manifest, fmt)
    if auto_compact:
        # After the commit, never before: the data is already durable, so a compaction that
        # fails costs nothing but the compaction.
        from batcher.io.formats.lakehouse.maintenance import auto_compact as _auto_compact

        _auto_compact(path, fmt)
    invalidate_source_stats(path, fmt)
    return manifest


def _write_distributed_breaker(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    path: str,
    fmt: str,
    sink: Any,
    sink_kwargs: dict[str, Any] | None,
    partition_by: list[str] | None,
    num_workers: int | None,
    *,
    layout: FileLayout,
    resume: bool,
    auto_compact: bool,
) -> WriteManifest:
    """Write a distributed plan whose shape needs a breaker, without collecting it.

    The breaker runs with ``materialize=False``, so each reducer keeps its bucket where it
    computed it — Arrow-IPC files on shared scratch, or buckets resident on the Flight
    fleet. The workers then write those buckets directly to the sink and only `WrittenFile`
    locators come back, which is the same two-phase shape `_distributed_write_plan` uses
    for a breaker-free plan.

    A shape that cannot keep its result partitioned hands back a collected table instead,
    and that table is re-sharded across the workers to write, exactly as before. Callers
    therefore get a manifest either way; what changes is whether the rows passed through
    the driver on the way.

    The whole run is held inside one `query_shuffle_scope`, for the reason
    `terminal.distributed_stream.iter_distributed` documents at length: scope exit evicts
    the query's shuffle buckets on the premise that leaving the scope means the query is
    over, which is false while the driver still holds handles to buckets nobody has read.
    An unregistered ticket reads back as an *empty bucket rather than an error*, so without
    the enclosing scope a Flight-transport write would commit a valid, empty table.

    Args:
        plan: The logical plan to run.
        sources: Its bound sources, in scan order.
        columns: The requested output columns.
        path: Destination path or table identifier.
        fmt: Sink format name.
        sink: The driver-side sink that performs the commit.
        sink_kwargs: Constructor arguments for the worker-side sinks.
        partition_by: Hive partition columns, or None.
        num_workers: Explicit worker fan-out, or None to derive one.
        layout: The caller's file-sizing request.
        resume: Skip shards whose part file already exists.
        auto_compact: Compact the destination after the commit.

    Returns:
        The committed `WriteManifest`.
    """
    from batcher import core
    from batcher.api.orchestration import run_relational
    from batcher.dist import resolve_worker_fanout
    from batcher.dist.executors.plan_analysis import requires_staging
    from batcher.dist.executors.write import _distributed_write, _distributed_write_partitioned
    from batcher.dist.fleet.plan_id import query_shuffle_scope

    workers = resolve_worker_fanout(num_workers)

    # Two shapes keep the collect, both deliberately.
    #
    # A **Hive-partitioned** write: `_write_shards` cuts a collected result *by partition
    # key*, so each key lands wholly in one shard and the write emits one file per
    # partition. A breaker's buckets are partitioned by its own key (the group key, the
    # sort range, the join key), which is not the partition column, so writing them where
    # they lie scatters every key across every worker and emits workers x partitions files.
    # That is the small-files problem the by-key reshard exists to avoid, and it is what
    # makes the *next* query slow. The trade is stated rather than made silently — see the
    # "Distributed writes" section of the writing-data guide.
    #
    # A **multi-stage** plan (a join whose operand spans two sources, or a breaker beneath
    # another breaker): the one-shot dispatcher has no path for it and refuses rather than
    # computing a wrong answer, so it must go through `_collect`, which routes it to the
    # adaptive executor that stages it. `requires_staging` is the same predicate
    # `dist.executor._unsupported` consults before raising.
    if partition_by is not None or requires_staging(plan):
        table = _collect(plan, sources, columns, distributed=True, num_workers=num_workers)
        manifest = _distributed_write(
            sink, table, path, partition_by, workers, layout=layout, resume=resume
        )
        return _commit(sink, manifest, path, fmt, table.schema, auto_compact=auto_compact)

    out_schema = _schema(plan, sources, columns)
    ctx = core.ExecutionContext(columns=columns, hub=core.default_hub(), num_workers=num_workers)
    with query_shuffle_scope():
        result, _ = run_relational(plan, sources, ctx, distributed=True, materialize=False)
        if isinstance(result, pa.Table):
            # This shape could not keep its result partitioned (an operator above the
            # breaker, or a sort carrying a limit). Re-shard the collected table across the
            # workers to write it, exactly as before.
            manifest = _distributed_write(
                sink, result, path, partition_by, workers, layout=layout, resume=resume
            )
            return _commit(sink, manifest, path, fmt, result.schema, auto_compact=auto_compact)
        try:
            manifest = _distributed_write_partitioned(
                result,
                path,
                fmt,
                sink_kwargs,
                partition_by,
                workers,
                layout=layout,
                resume=resume,
            )
        finally:
            if callable(getattr(result, "cleanup", None)):
                result.cleanup()
    # Every bucket was empty (a filter that matched nothing, or a join with no matches).
    # Each worker correctly wrote no file, but the single-node path writes ONE empty file,
    # so without this the distributed result is an absent path where single-node leaves a
    # readable empty table — a `distributed != single-node` divergence that surfaces
    # downstream as "path does not exist" rather than an empty read. The streaming
    # breaker-free write above repairs it the same way.
    if not manifest.files and out_schema is not None:
        empty = pa.Table.from_batches([], schema=out_schema)
        manifest = WriteManifest(tuple(sink.write_partitioned(empty, path, file_index=0)))
    return _commit(sink, manifest, path, fmt, out_schema, auto_compact=auto_compact)


@with_auto_config
def _write(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    path: str,
    fmt: str,
    *,
    partition_by: list[str] | None = None,
    distributed: bool | str = "auto",
    num_workers: int | None = None,
    resume: bool = False,
    max_rows_per_file: int | None = None,
    num_files: int | None = None,
    target_bytes_per_file: int | None = None,
    auto_compact: bool = False,
    sink_kwargs: dict[str, Any] | None = None,
    sink: Any | None = None,
    directory: bool = False,
) -> WriteManifest:
    """Execute the plan and write the result via the `fmt` sink.

    A plain path with no `partition_by` writes a single file (and, distributed, a
    directory of ``part-*`` files). `partition_by` writes a Hive-layout directory.
    Workers write their own data files in parallel; the driver then performs one
    atomic `commit` over the merged manifest.

    `distributed` resolves through the SAME `"auto"` routing as `collect()`. It used to
    default to `False`, so on a cluster every write — the one terminal whose output size
    is the whole result — ran the read, the transform, and the write on the driver alone.
    """
    from batcher.io.sink import SINKS, table_sink_kwargs
    from batcher.plan.logical import is_streamable

    distributed = _resolve_distributed(distributed, plan, sources)

    # Validate the row cap up front: 0 would raise an opaque `range() step zero`, and
    # a negative value would silently produce an *empty* range — writing no files at
    # all (total data loss). Fail clearly instead.
    if max_rows_per_file is not None and max_rows_per_file < 1:
        from batcher._internal.errors import PlanError

        raise PlanError(f"max_rows_per_file must be >= 1, got {max_rows_per_file}")

    # The transactional (lakehouse) sinks commit on the DRIVER from a manifest of
    # worker-staged files, so the driver's sink — a separate instance from the workers'
    # — must carry `partition_by` (the workers get it via `write_partitioned`, but the
    # driver only sees it here). Thread it through the constructor so the commit
    # partitions correctly. File sinks take partition_by per-call and ignore this.
    sink_kwargs = dict(sink_kwargs or {})
    if fmt == "delta" and partition_by is not None:
        sink_kwargs.setdefault("partition_by", partition_by)
    # A table format may need its destination at construction rather than per call — for
    # Iceberg the write `path` IS the table identifier, and one write token is shared by
    # every worker's sink so all shards of this write name their staged files under the
    # same token (a later write uses a different one, so `add_files` never lets it clobber
    # a file a prior snapshot still references). Shared with the streaming table sink,
    # which needs the identical treatment.
    for key, value in table_sink_kwargs(fmt, path).items():
        sink_kwargs.setdefault(key, value)
    # `sink` lets a caller supply the writer instead of taking the registry's default. The
    # copy-on-write MERGE needs it: its output files land *beside* files it deliberately did
    # not read, so they must carry a unique token (`TokenizedParquetSink`) rather than the
    # registry sink's deterministic `part-{index}` name, which would overwrite a survivor.
    if sink is None:
        sink = SINKS.get(fmt)(**sink_kwargs)

    # Streaming distributed write: for a breaker-free single-source *relational* plan, each
    # worker writes its own output files and only manifests return — the full
    # result never materializes on the driver (no OOM on tables bigger than one
    # node). Plans with a breaker (aggregate/join/sort) reduce the result first,
    # so the collect-then-write path below is fine for them.
    #
    # `map_batches`/UDF pipelines go through here TOO. `_distributed_write_plan` routes them to
    # the UDF-aware distributed map with the sink bound into each worker, so a batch-inference
    # job writes its output from the worker that produced it. Excluding them here (as this path
    # used to) sent them down `_collect(distributed=True)`, which lands the whole post-inference
    # result on the driver — an unconditional OOM on any job whose output exceeds one node.
    # The file-layout request travels WITH the write rather than being resolved here: a
    # streaming distributed write never materializes its result on the driver, so only the
    # worker holding a shard can turn `num_files`/`target_bytes_per_file` into a row cap.
    layout = FileLayout(
        max_rows_per_file=max_rows_per_file,
        num_files=num_files,
        target_bytes_per_file=target_bytes_per_file,
    )

    if distributed and len(sources) == 1 and is_streamable(plan):
        from batcher.dist import resolve_worker_fanout
        from batcher.dist.executors.write import _distributed_write_plan

        out_schema = _schema(plan, sources, columns)
        manifest = _distributed_write_plan(
            plan,
            sources,
            path,
            fmt,
            sink_kwargs,
            partition_by,
            resolve_worker_fanout(num_workers),
            layout=layout,
            resume=resume,
        )
        # Every shard produced zero rows (a filter that matched nothing). Each worker
        # correctly wrote no file, but the single-node path writes ONE empty file, so
        # without this the distributed result is an absent path where single-node leaves a
        # readable empty table — a `distributed != single-node` divergence that surfaces
        # downstream as "path does not exist" rather than an empty read.
        if not manifest.files and out_schema is not None:
            empty = pa.Table.from_batches([], schema=out_schema)
            manifest = WriteManifest(tuple(sink.write_partitioned(empty, path, file_index=0)))
        return _commit(sink, manifest, path, fmt, out_schema, auto_compact=auto_compact)

    # An unbounded source reaching the materialize path below would never finish.
    # The streaming distributed write above handles breaker-free shapes; otherwise
    # refuse with an actionable message instead of hanging on _collect.
    from batcher.io.source import is_bounded

    if any(not is_bounded(s) for s in sources):
        from batcher._internal.errors import PlanError

        if is_streamable(plan):
            raise PlanError(
                "writing an unbounded (streaming) source needs the streaming write "
                "path — pass distributed=True so each worker writes its shards "
                "incrementally without materializing the whole stream."
            )
        raise PlanError(
            "cannot write an unbounded (streaming) source through a plan that must "
            "materialize (sort / join / window / multi-source). Restructure to a "
            "streamable shape, or consume it with iter_batches()."
        )

    # A plan WITH a breaker (aggregate / sort / join / distinct / window), distributed.
    # The breaker runs with `materialize=False`, so each reducer keeps its bucket where it
    # computed it, and the workers then write those buckets straight out — the result never
    # lands on the driver at all.
    #
    # The collect-then-reshard path this replaces was justified on the premise that a
    # breaker reduces. An aggregate does. A **sort** and a **window** emit one row per input
    # row, a join can emit far more than either input, and a high-cardinality `distinct`
    # removes almost nothing — so on exactly the shapes where a distributed write matters
    # most, the one terminal whose output is the size of the result was also the one that
    # funnelled every row of it through a single process.
    if distributed:
        return _write_distributed_breaker(
            plan,
            sources,
            columns,
            path,
            fmt,
            sink,
            sink_kwargs,
            partition_by,
            num_workers,
            layout=layout,
            resume=resume,
            auto_compact=auto_compact,
        )

    # Streaming single-node write: a breaker-free plan over a lazy source
    # (read→filter→project→write) streams batch-by-batch into one file, so the driver
    # holds one batch — never the whole result.
    if hasattr(sink, "write_stream") and _streaming_write_eligible(
        plan,
        sources,
        distributed,
        partition_by,
        max_rows_per_file,
        num_files,
        target_bytes_per_file,
        sink,
        path,
    ):
        from batcher._internal.prefetch import prefetch
        from batcher.api.terminal.map_stream import peek_stream
        from batcher.api.terminal.stream import _iter_batches

        # Peek the first batch for the schema (else an opaque `map_batches` forces an
        # extra zero-row pass), then overlap read→transform with the write off-thread.
        schema, stream = peek_stream(
            _iter_batches(plan, sources, columns), lambda: _schema(plan, sources, columns)
        )
        if max_rows_per_file is not None:
            # A row cap makes the destination a directory of parts, rolled over as the
            # stream fills each one. Driver memory stays at one batch however many rows
            # (or files) the write produces.
            files = sink.write_stream_parts(
                prefetch(stream),
                path,
                max_rows_per_file=max_rows_per_file,
                schema=schema,
                resume=resume,
            )
            return _commit(
                sink, WriteManifest(tuple(files)), path, fmt, schema, auto_compact=auto_compact
            )
        written = sink.write_stream(prefetch(stream), path, schema=schema, resume=resume)
        return _commit(
            sink, WriteManifest((written,)), path, fmt, schema, auto_compact=auto_compact
        )

    # Single-node from here: every distributed shape returned above.
    table = _collect(plan, sources, columns, distributed=False, num_workers=num_workers)
    # Resolve the layout to a per-file row cap now that the size is known (no extra
    # counting pass): split into `num_files`, or size files to `target_bytes_per_file`
    # using the materialized byte size.
    max_rows_per_file = layout.rows_per_file(table.num_rows, logical_bytes(table))
    if max_rows_per_file is not None or (
        directory
        or partition_by
        or _sink_owns_its_layout(sink, path)
        or _destination_is_a_directory(sink, path)
    ):
        # A row cap (or partitioning) writes a directory of `part-*` files; the cap
        # bounds each file's size (no single giant file; tiny files coalesce upstream).
        # `directory` forces that layout with neither: a copy-on-write MERGE writes *into*
        # an existing directory of data files, so a single-file write at `path` would try
        # to replace the directory itself.
        #
        # `_destination_is_a_directory` is the same situation arrived at without the flag:
        # any overwrite of a path that *already is* a directory of part files. Writing one
        # file there means renaming a temp file over a directory, which fails outright on
        # local disk (`IsADirectoryError`) and leaves the old files in place on an object
        # store. It is reached by `replace_where` on an unpartitioned directory, and by any
        # plain rewrite of a sharded output.
        #
        # `partitions_itself` is how a sink says its layout is a property of the *table*,
        # not of the write. A partitioned Iceberg table declares its partitioning in the
        # catalog's spec, so no `partition_by` argument ever appears at the call site — and
        # this branch used to be skipped, the shard written as one flat file, and the commit
        # rejected it ("more than one partition values"). A partitioned Iceberg table was
        # simply unwritable.
        #
        # `resume`/`max_rows_per_file` are file-sink concepts; a warehouse/DB sink ignores
        # them (each shard appends to one table), so pass them only when accepted.
        written = _sink_write_partitioned(
            sink,
            table,
            path,
            partition_by=partition_by,
            resume=resume,
            max_rows_per_file=max_rows_per_file,
        )
        manifest = WriteManifest(tuple(written))
    else:
        manifest = WriteManifest((_sink_write(sink, table, path, resume=resume),))
        # Single-file write: remember the result's stats so a later read of this
        # exact path can be answered from metadata even for a footerless format.
        from batcher.api.orchestration import persist_written_source_stats

        persist_written_source_stats(table, path, fmt)
    return _commit(sink, manifest, path, fmt, table.schema, auto_compact=auto_compact)


def _accepts_keyword(fn, name: str) -> bool:
    """Whether `fn` declares (or absorbs) a keyword called `name`.

    Asked of a sink's `write`, this replaces a `try/except TypeError` that could not tell
    the two meanings of a `TypeError` apart: "this function has no such parameter", and
    "this function raised while running". The second is the dangerous one — a sink that
    applied half its rows and *then* raised a `TypeError` was retried without the keyword,
    so a database append wrote those rows a second time. Nothing about that is visible: the
    call succeeds, and the table has duplicates.

    `io.source.read._accepts` answers the mirror question for a source's `splits`, and for
    the same reason. Signature inspection is the only way to ask it that cannot be
    confused by what the callee does.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - a builtin/C callable
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _sink_write(sink, table: pa.Table, path: str, *, resume: bool):
    """`sink.write`, tolerating sinks whose `write` has no `resume` parameter.

    The `Sink` protocol's `write(table, path)` does not include `resume`; only the file
    sinks (which write atomically to an idempotent path) added it. A warehouse/DB sink
    (Snowflake/Mongo/ADBC/DB-API/Lance) implements the bare protocol signature, so passing
    `resume=` crashed. Resume is meaningless for an append-to-table sink, so drop it for
    those — but a *requested* `resume=True` the sink cannot honor is surfaced, never
    silently ignored (silently ignoring it would risk duplicate ingest on a re-run).

    Which sinks take it is decided by `_accepts_keyword`, not by calling and catching. See
    there for why the difference matters.
    """
    if _accepts_keyword(sink.write, "resume"):
        return sink.write(table, path, resume=resume)
    if resume:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"write(resume=True) is not supported by the {type(sink).__name__} sink "
            "(resume is exactly-once only for the atomic file sinks). Write without "
            "resume, or land the data in a file format first."
        )
    return sink.write(table, path)


def _sink_write_partitioned(
    sink,
    table: pa.Table,
    path: str,
    *,
    partition_by: list[str] | None,
    resume: bool,
    max_rows_per_file: int | None,
):
    """`sink.write_partitioned`, tolerating sinks whose signature omits the file-sink-only
    `resume`/`max_rows_per_file` (warehouse/DB sinks — each shard just appends to one table).

    A *requested* `resume=True` a sink cannot honor is surfaced rather than silently dropped
    (dropping it risks duplicate ingest on a re-run), mirroring `_sink_write` — and, like it,
    the capability is read off the signature rather than discovered by catching a
    `TypeError` a running sink could equally well have raised itself."""
    if _accepts_keyword(sink.write_partitioned, "resume"):
        return sink.write_partitioned(
            table,
            path,
            partition_by=partition_by,
            resume=resume,
            max_rows_per_file=max_rows_per_file,
        )
    if resume:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"write(resume=True) is not supported by the {type(sink).__name__} sink "
            "(resume is exactly-once only for the atomic file sinks)."
        )
    return sink.write_partitioned(table, path, partition_by=partition_by)


def _to_pydict(
    plan: LogicalPlan, sources: list[Source], columns: list[str]
) -> dict[str, list[Any]]:
    """Execute and return the result as a column-oriented dict."""
    return _collect(plan, sources, columns).to_pydict()


def _to_pylist(
    plan: LogicalPlan, sources: list[Source], columns: list[str]
) -> list[dict[str, Any]]:
    """Execute and return the result as a list of row dicts."""
    return _collect(plan, sources, columns).to_pylist()


def _to_pandas(plan: LogicalPlan, sources: list[Source], columns: list[str]) -> Any:
    """Execute and return the result as a pandas `DataFrame` (via Arrow)."""
    require("pandas", feature="Dataset.to_pandas()", provides="pandas", extra="pandas")
    return _collect(plan, sources, columns).to_pandas()


def _to_polars(plan: LogicalPlan, sources: list[Source], columns: list[str]) -> Any:
    """Execute and return the result as a Polars `DataFrame` (zero-copy from Arrow)."""
    polars = require("polars", feature="Dataset.to_polars()", provides="polars", extra="polars")
    return polars.from_arrow(_collect(plan, sources, columns))


def _show(plan: LogicalPlan, sources: list[Source], columns: list[str], limit: int) -> None:
    """Print a preview of the result.

    The `limit` is pushed into the PLAN, not applied to a materialized table: `show()`
    on a billion-row dataset must read only enough of the source to produce `limit`
    rows (the streaming early-stop / distributed top-N paths), never collect the whole
    result to the driver just to slice ten rows off it.

    What gets printed is a row-oriented table (`terminal.preview`), not pyarrow's own
    column-oriented `Table` repr — see that module for why.
    """
    from batcher.api.terminal.preview import render

    print(render(_collect(_narrowed_limit(plan, limit), sources, columns), limit=limit))


def _narrowed_limit(plan: LogicalPlan, limit: int) -> LogicalPlan:
    """`plan` capped at `limit` rows, folding into an existing top-level `LIMIT`.

    Wrapping one limit in another is correct but unstreamable -- the router recognizes a
    limit over a breaker-free pipeline, and a limit is not one -- so `head(10).show()` over
    an endless source refused while `show()` on the same source did not. Folding keeps it
    one node, and the arithmetic is exact: the outer limit takes its rows from the front of
    what the inner one already skipped to.
    """
    from batcher.plan.logical import Limit

    if isinstance(plan, Limit):
        return Limit(plan.input, min(plan.n, limit), plan.offset)
    return Limit(plan, limit)


def _is_bounded_peek(plan) -> bool:
    """Whether `plan` is a `LIMIT n` whose result is finite even over an endless source.

    Two inputs qualify, and both for the same reason: the router has a driver that stops
    reading the moment the answer is settled.

    - A **breaker-free** input — filter / select / map_batches — where the first `n` rows
      out are the answer.
    - A whole-column **`DISTINCT`**, where the first `n` distinct rows are. Once `n` of
      them have been seen, every later row is either a duplicate or arrives too late to
      displace one, so the read stops there (`core.streaming.stream_distinct_limit`).
      `distinct().head(20)` is how anyone finds out what values a topic carries, and it
      was refused while `filter().head(20)` beside it was not — a difference the caller had
      no way to predict, since both terminate and both hold `n` rows.

    Over a *sort* the answer is finite too and still unreachable, because top-N is not
    known until the last row has arrived. A **keyed** `DISTINCT ON` likewise stays out: its
    survivor per key can be replaced by a later row, so no prefix settles it.
    """
    from batcher.plan.logical import Distinct, Limit, is_streamable

    if not isinstance(plan, Limit):
        return False
    inner = plan.input
    if is_streamable(inner):
        return True
    return isinstance(inner, Distinct) and not inner.keys and is_streamable(inner.input)


def _peek_schema(plan) -> pa.Schema:
    """The declared schema of an empty peek, so `to_pydict()` still names the columns."""
    declared = plan.available_schema()
    if declared is not None:
        return declared.arrow
    return pa.schema([pa.field(name, pa.null()) for name in plan.available_columns()])
