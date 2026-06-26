"""Terminal/materialization operations for `Dataset`.

These free functions own the orchestration of the three layers (Kyber → Carbonite
→ Core) for terminal operations. `Dataset`'s terminal methods are thin wrappers
that forward their state (`self._plan`, `self._sources`, `self.columns`) here.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.api.orchestration import with_auto_config
from batcher.api.terminal.metadata_answer import (
    metadata_aggregate_table,
    metadata_count,
    metadata_is_empty,
)
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


def _resolve_distributed(distributed: bool | str) -> bool:
    """Resolve ``distributed="auto"``: use the cluster iff already connected to a
    multi-node one, else stay single-node.

    Never forces a Ray init for a local query — if Ray isn't up, "auto" is local.
    An explicit ``True``/``False`` always wins (the user's override).
    """
    if distributed != "auto":
        return bool(distributed)
    try:
        import ray

        if not ray.is_initialized():
            return False
        from batcher import dist

        return dist.cluster_topology()["nodes"] > 1
    except Exception:
        return False


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
        from batcher._internal.errors import PlanError

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
        from batcher.plan.logical import Aggregate

        if isinstance(plan, Aggregate) and not plan.group_keys:
            from batcher import core
            from batcher.api.orchestration import collect_source_stats

            source_stats = collect_source_stats(sources, core.default_hub())
    metadata = metadata_aggregate_table(plan, sources, source_stats)
    if metadata is not None:
        return metadata
    # Opt-in: offload large-payload columns out of line around breakers (the blobs ride
    # through as tiny handles). Inserted before execution routing so the resulting
    # `map_batches` stages take the same mixed-executor path as an explicit offload.
    from batcher.api.terminal.blob_offload import maybe_insert_blob_offload

    plan = maybe_insert_blob_offload(plan)
    distributed = _resolve_distributed(distributed)
    # Resolve `adaptive="auto"` to a concrete decision before the fast-path checks
    # below ("auto" is a truthy string). Join-less plans short-circuit to False without
    # touching source stats, so the common path pays nothing.
    from batcher import core
    from batcher.api.adaptive import resolve_adaptive

    adaptive = resolve_adaptive(adaptive, plan, sources, core.default_hub())

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
            from batcher.api._join_helpers import _empty_schema
            from batcher.core.streaming import stream_limit

            batches = list(stream_limit(plan, sources[0]))
            schema = batches[0].schema if batches else _empty_schema(columns)
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
        from batcher import core
        from batcher.api.orchestration import auto_num_partitions
        from batcher.dist.spill import spill_collect

        partitions = (
            num_partitions
            if num_partitions is not None
            else auto_num_partitions(plan, sources, core.default_hub())
        )
        spilled = spill_collect(plan, sources, partitions)
        if spilled is not None:
            return spilled
        # Other plan shapes have no spilling path → fall through to in-memory.

    # Imported here (not at module load) to keep the layer-import contract
    # simple and avoid importing the engine for pure-Python tooling.
    import time

    from batcher import core
    from batcher.api import executors
    from batcher.api.terminal.event_log import event_log_collector, write_event_log

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
    table = executors.select(plan, distributed=distributed).execute(plan, sources, ctx)
    write_event_log(ctx.profile, total_ms=(time.perf_counter() - t0) * 1000.0, rows=table.num_rows)
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


def _shared_source_stats(plan: LogicalPlan, sources: list[Source]) -> list | None:
    """Source statistics to share across a metadata attempt and its execution fallback.

    Collected once, only when a metadata answer may even be attempted (a bounded,
    non-UDF plan) — exactly the case where the relational execution fallback would
    also read them. Returns `None` otherwise, leaving each path to collect its own,
    so an opaque UDF/streaming terminal pays nothing extra.
    """
    from batcher.api.terminal.metadata_answer import _metadata_answerable

    if not _metadata_answerable(plan, sources):
        return None
    from batcher import core
    from batcher.api.orchestration import collect_source_stats

    return collect_source_stats(sources, core.default_hub())


def _count(plan: LogicalPlan, sources: list[Source], columns: list[str]) -> int:
    """Return the number of result rows, from metadata when provable, else execute.

    Metadata-first: a plan whose row count is exactly derivable without execution
    (`limit(n)`, a global aggregate, an empty source, counts through row-preserving
    operators) is answered from `SourceStatistics` alone. Otherwise it falls back to
    a full `_collect`, which is always correct — and, crucially, executes the *same*
    plan shape, so its measured per-operator cardinalities feed the learning loop
    (e.g. a filter's selectivity) under the plan's own signature.
    """
    source_stats = _shared_source_stats(plan, sources)
    answer = metadata_count(plan, sources, source_stats)
    if answer is not None:
        return answer
    return _collect(plan, sources, columns, source_stats=source_stats).num_rows


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


@with_auto_config
def _write(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    path: str,
    fmt: str,
    *,
    partition_by: list[str] | None = None,
    distributed: bool = False,
    num_workers: int | None = None,
    resume: bool = False,
    max_rows_per_file: int | None = None,
    num_files: int | None = None,
    target_bytes_per_file: int | None = None,
    sink_kwargs: dict[str, Any] | None = None,
) -> WriteManifest:
    """Execute the plan and write the result via the `fmt` sink.

    A plain path with no `partition_by` writes a single file (and, distributed, a
    directory of ``part-*`` files). `partition_by` writes a Hive-layout directory.
    Workers write their own data files in parallel; the driver then performs one
    atomic `commit` over the merged manifest.
    """
    from batcher.io.sink import SINKS
    from batcher.plan.logical import is_streamable

    # Validate the row cap up front: 0 would raise an opaque `range() step zero`, and
    # a negative value would silently produce an *empty* range — writing no files at
    # all (total data loss). Fail clearly instead.
    if max_rows_per_file is not None and max_rows_per_file < 1:
        from batcher._internal.errors import PlanError

        raise PlanError(f"max_rows_per_file must be >= 1, got {max_rows_per_file}")

    sink = SINKS.get(fmt)(**(sink_kwargs or {}))

    # Streaming distributed write: for a breaker-free single-source plan, each
    # worker writes its own output files and only manifests return — the full
    # result never materializes on the driver (no OOM on tables bigger than one
    # node). Plans with a breaker (aggregate/join/sort) reduce the result first,
    # so the collect-then-write path below is fine for them.
    if distributed and len(sources) == 1 and is_streamable(plan):
        from batcher.dist.executors.write import _distributed_write_plan

        manifest = _distributed_write_plan(
            plan, sources, path, fmt, sink_kwargs, partition_by, num_workers or 4
        )
        sink.commit(manifest, path)
        return manifest

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
    if _streaming_write_eligible(
        plan,
        sources,
        distributed,
        partition_by,
        max_rows_per_file,
        num_files,
        target_bytes_per_file,
    ):
        from batcher.api.terminal.stream import _iter_batches

        written = sink.write_stream(
            _iter_batches(plan, sources, columns),
            path,
            schema=_schema(plan, sources, columns),
            resume=resume,
        )
        manifest = WriteManifest((written,))
        sink.commit(manifest, path)
        return manifest

    table = _collect(plan, sources, columns, distributed=distributed, num_workers=num_workers)
    # Resolve a `repartition` layout to a per-file row cap now that the size is known
    # (no extra counting pass): split into `num_files`, or size files to
    # `target_bytes_per_file` using the materialized byte size.
    if max_rows_per_file is None and table.num_rows:
        if num_files is not None:
            max_rows_per_file = -(-table.num_rows // num_files)  # ceil-divide
        elif target_bytes_per_file is not None and table.nbytes:
            rows = table.num_rows * target_bytes_per_file // table.nbytes
            max_rows_per_file = max(1, int(rows))
    if distributed:
        from batcher.dist.executors.write import _distributed_write

        manifest = _distributed_write(sink, table, path, partition_by, num_workers or 4)
    elif partition_by or max_rows_per_file is not None:
        # A row cap (or partitioning) writes a directory of `part-*` files; the cap
        # bounds each file's size (no single giant file; tiny files coalesce upstream).
        written = sink.write_partitioned(
            table,
            path,
            partition_by=partition_by,
            resume=resume,
            max_rows_per_file=max_rows_per_file,
        )
        manifest = WriteManifest(tuple(written))
    else:
        manifest = WriteManifest((sink.write(table, path, resume=resume),))
        # Single-file write: remember the result's stats so a later read of this
        # exact path can be answered from metadata even for a footerless format.
        from batcher.api.orchestration import persist_written_source_stats

        persist_written_source_stats(table, path, fmt)
    sink.commit(manifest, path)
    return manifest


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
    """Print a preview of the result."""
    table = _collect(plan, sources, columns)
    print(table.slice(0, limit))
