"""Terminal/materialization operations for `Dataset`.

These free functions own the orchestration of the three layers (Kyber → Carbonite
→ Core) for terminal operations. `Dataset`'s terminal methods are thin wrappers
that forward their state (`self._plan`, `self._sources`, `self.columns`) here.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher._internal.mathx import ceil_div
from batcher.api.orchestration import with_auto_config
from batcher.api.terminal.metadata_answer import (
    metadata_aggregate_table,
    metadata_count,
    metadata_empty_table,
    metadata_is_empty,
)
from batcher.api.terminal.routing import resolve_distributed as _resolve_distributed
from batcher.io.manifest import WriteManifest
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

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
    """
    from batcher.io.source import is_bounded

    if any(not is_bounded(s) for s in sources):
        raise PlanError(
            "this operation materializes the full result, but the dataset has an "
            "unbounded (streaming) source. Consume it with iter_batches() or write "
            "it to a sink instead."
        )
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

        gpu_result = try_gpu_collect(plan, sources, core.default_hub(), force=(backend == "gpu"))
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
            from batcher.core.streaming import stream_limit

            batches = list(stream_limit(plan, sources[0]))
            schema = batches[0].schema if batches else _empty_result_schema(plan, columns)
            return pa.Table.from_batches(batches, schema=schema)

    if adaptive:
        from batcher import core
        from batcher.api.adaptive import execute_adaptive

        # Adaptive re-optimization now works distributed too: each breaker stage
        # fans out across workers and its measured cardinality re-plans the rest.
        return execute_adaptive(
            plan,
            sources,
            core.default_hub(),
            distributed=distributed,
            num_workers=num_workers,
            transport=transport,
        ).table

    if spill and not distributed:
        from batcher import core, kyber
        from batcher.api.orchestration import auto_num_partitions
        from batcher.dist.spill import spill_collect

        hub = core.default_hub()
        partitions = num_partitions or auto_num_partitions(plan, sources, hub)
        # Spill the optimized plan (COUNT(DISTINCT)→COUNT over DISTINCT; derived join keys).
        opt_lp = kyber.optimize_logical(plan, sources=sources, hub=hub)
        spilled = spill_collect(opt_lp, sources, partitions)
        if spilled is not None:
            return spilled
        # Other plan shapes have no spilling path → fall through to in-memory.

    # Imported here (not at module load) to keep the layer-import contract
    # simple and avoid importing the engine for pure-Python tooling.
    import time

    from batcher import core
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
        table = executors.select(plan, distributed=distributed).execute(plan, sources, ctx)
    except BaseException as exc:
        # A failed query must close out on the bus too, or its progress bar spins forever
        # and the dashboard shows it as still running. Re-raised unchanged — reporting the
        # failure must not alter it.
        report_failure(query_id, total_ms=(time.perf_counter() - t0) * 1000.0, exc=exc)
        raise
    total_ms = (time.perf_counter() - t0) * 1000.0
    write_event_log(ctx.profile, total_ms=total_ms, rows=table.num_rows, query_id=query_id)
    from batcher.api.terminal.gpu_backend import record_cpu_crossover  # adaptive-crossover sample

    record_cpu_crossover(plan, sources, ctx.hub, total_ms)  # gated to a GPU cluster; else no-op
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
    from batcher.api.terminal.metadata_answer import global_count_plan

    source_stats = _shared_source_stats(plan, sources)
    answer = metadata_count(plan, sources, source_stats)
    if answer is not None:
        _record_count_selectivity(plan, sources, answer)
        return answer
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

    A bare scan returns its source schema directly. Otherwise the plan's
    type-carrying `available_schema()` analysis answers without touching the engine
    when it can infer every output type; anything it leaves uncertain falls back to
    a zero-row execution (`limit(0)`), which the engine answers without
    materializing data.
    """
    from batcher.plan.logical import Limit, Scan

    if isinstance(plan, Scan) and len(sources) == 1:
        return sources[0].schema()
    inferred = plan.available_schema()
    if inferred is not None:
        return inferred.arrow
    return _collect(Limit(plan, 0), sources, columns).schema


@with_auto_config
def _stats(plan: LogicalPlan, sources: list[Source], columns: list[str]):
    """Execute through the real path (single-node/spill/distributed) and return `RunStats`.

    Raises `PlanError` for an unbounded source and `BackendError` for a `map_batches`/ML
    pipeline (the opaque UDF path emits no per-operator metrics).
    """
    from batcher.api.stats import RunStats
    from batcher.api.terminal.profile import run_profiled

    profile = run_profiled(plan, sources, columns)
    return RunStats.from_profile(profile)


def _streaming_write_eligible(
    plan: LogicalPlan,
    sources: list[Source],
    distributed: bool,
    partition_by: list[str] | None,
    max_rows_per_file: int | None,
    num_files: int | None,
    target_bytes_per_file: int | None,
) -> bool:
    """Whether `_write` can stream the result to one file instead of collecting it.

    Eligible when a single-node, breaker-free plan reads a *lazy* source (file /
    iterator) into a plain single-file write: the batches stream straight to the sink,
    bounding driver memory to one batch. A fully-resident in-memory source gains
    nothing (its data is already in RAM), so it keeps the collect path — which also
    persists per-column sketch statistics (a full pass the streaming path can't do)
    for a later read. Partitioning or a per-file row/file/byte layout also needs the
    whole table first, so those stay on the collect path too.
    """
    from batcher.io.source import InMemorySource, MaterializedSource
    from batcher.plan.logical import is_streamable

    if distributed or partition_by:
        return False
    if max_rows_per_file is not None or num_files is not None or target_bytes_per_file is not None:
        return False
    if not is_streamable(plan):
        return False
    return not all(isinstance(s, InMemorySource | MaterializedSource) for s in sources)


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
    if auto_compact:
        # After the commit, never before: the data is already durable, so a compaction that
        # fails costs nothing but the compaction.
        from batcher.io.formats.lakehouse.maintenance import auto_compact as _auto_compact

        _auto_compact(path, fmt)
    invalidate_source_stats(path, fmt)
    return manifest


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
    from batcher.io.sink import SINKS
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
    if fmt == "iceberg":
        # For Iceberg the write `path` IS the table identifier (the sink needs it at
        # construction for staging + commit, on the driver and every worker).
        sink_kwargs.setdefault("identifier", path)
        # One write token shared by every worker's sink so all shards of this write name
        # their staged files under the same token (and a later write uses a different one,
        # so `add_files` never lets it clobber a file a prior snapshot still references).
        import uuid

        sink_kwargs.setdefault("write_token", uuid.uuid4().hex[:12])
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
    if distributed and len(sources) == 1 and is_streamable(plan):
        from batcher.dist import resolve_worker_fanout
        from batcher.dist.executors.write import _distributed_write_plan

        out_schema = _schema(plan, sources, columns)
        manifest = _distributed_write_plan(
            plan, sources, path, fmt, sink_kwargs, partition_by, resolve_worker_fanout(num_workers)
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
    ):
        from batcher._internal.prefetch import prefetch
        from batcher.api.terminal.map_stream import peek_stream
        from batcher.api.terminal.stream import _iter_batches

        # Peek the first batch for the schema (else an opaque `map_batches` forces an
        # extra zero-row pass), then overlap read→transform with the write off-thread.
        schema, stream = peek_stream(
            _iter_batches(plan, sources, columns), lambda: _schema(plan, sources, columns)
        )
        written = sink.write_stream(prefetch(stream), path, schema=schema, resume=resume)
        return _commit(
            sink, WriteManifest((written,)), path, fmt, schema, auto_compact=auto_compact
        )

    table = _collect(plan, sources, columns, distributed=distributed, num_workers=num_workers)
    # Resolve a `repartition` layout to a per-file row cap now that the size is known
    # (no extra counting pass): split into `num_files`, or size files to
    # `target_bytes_per_file` using the materialized byte size.
    if max_rows_per_file is None and table.num_rows:
        if num_files is not None:
            max_rows_per_file = ceil_div(table.num_rows, num_files)
        elif target_bytes_per_file is not None and table.nbytes:
            rows = table.num_rows * target_bytes_per_file // table.nbytes
            max_rows_per_file = max(1, int(rows))
    if distributed:
        from batcher.dist import resolve_worker_fanout
        from batcher.dist.executors.write import _distributed_write

        manifest = _distributed_write(
            sink, table, path, partition_by, resolve_worker_fanout(num_workers)
        )
    elif (
        directory
        or partition_by
        or max_rows_per_file is not None
        or getattr(sink, "partitions_itself", False)
    ):
        # A row cap (or partitioning) writes a directory of `part-*` files; the cap
        # bounds each file's size (no single giant file; tiny files coalesce upstream).
        # `directory` forces that layout with neither: a copy-on-write MERGE writes *into*
        # an existing directory of data files, so a single-file write at `path` would try
        # to replace the directory itself.
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


def _sink_write(sink, table: pa.Table, path: str, *, resume: bool):
    """`sink.write`, tolerating sinks whose `write` has no `resume` parameter.

    The `Sink` protocol's `write(table, path)` does not include `resume`; only the file
    sinks (which write atomically to an idempotent path) added it. A warehouse/DB sink
    (Snowflake/Mongo/ADBC/Lance) implements the bare protocol signature, so passing
    `resume=` crashed. Resume is meaningless for an append-to-table sink, so drop it for
    those — but a *requested* `resume=True` the sink cannot honor is surfaced, never
    silently ignored (silently ignoring it would risk duplicate ingest on a re-run).
    """
    try:
        return sink.write(table, path, resume=resume)
    except TypeError:
        if resume:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"write(resume=True) is not supported by the {type(sink).__name__} sink "
                "(resume is exactly-once only for the atomic file sinks). Write without "
                "resume, or land the data in a file format first."
            ) from None
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
    (dropping it risks duplicate ingest on a re-run), mirroring `_sink_write`."""
    try:
        return sink.write_partitioned(
            table,
            path,
            partition_by=partition_by,
            resume=resume,
            max_rows_per_file=max_rows_per_file,
        )
    except TypeError:
        if resume:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"write(resume=True) is not supported by the {type(sink).__name__} sink "
                "(resume is exactly-once only for the atomic file sinks)."
            ) from None
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
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise BackendError("to_pandas() needs: pip install 'batcher-engine[pandas]'") from exc
    return _collect(plan, sources, columns).to_pandas()


def _to_polars(plan: LogicalPlan, sources: list[Source], columns: list[str]) -> Any:
    """Execute and return the result as a Polars `DataFrame` (zero-copy from Arrow)."""
    try:
        import polars
    except ImportError as exc:
        raise BackendError("to_polars() needs: pip install 'batcher-engine[polars]'") from exc
    return polars.from_arrow(_collect(plan, sources, columns))


def _show(plan: LogicalPlan, sources: list[Source], columns: list[str], limit: int) -> None:
    """Print a preview of the result.

    The `limit` is pushed into the PLAN, not applied to a materialized table: `show()`
    on a billion-row dataset must read only enough of the source to produce `limit`
    rows (the streaming early-stop / distributed top-N paths), never collect the whole
    result to the driver just to slice ten rows off it.
    """
    from batcher.plan.logical import Limit

    print(_collect(Limit(plan, limit), sources, columns))
