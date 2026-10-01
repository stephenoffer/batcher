"""Run a scan-heavy plan with its largest input streamed into the engine, not read up front.

The in-memory path resolves every source to Arrow before the engine starts
(`stages.resolve_sources`), so its peak memory is the whole projected input, and reading and
computing strictly alternate. On a table larger than memory that routed the query out of core —
TPC-H sf100 q5 ran in 3 s on a 240 GB box and in 233 s on a 92 GB one, the difference being
entirely the spill path — and on one that fits, the read and the engine could never overlap.

Here the plan's driving source is handed to the engine as an iterator of chunks
(`core.execute_local_chunked`): each is a group of Parquet files decoded by the concurrent native
reader while the engine folds the previous one (`dist.spill.iter_spill_chunks`). Every other
source is resolved exactly as the in-memory path resolves it. The engine decides whether the plan
has a shape it can serve this way (`bc_interp::stream::chunked::chunkable`) and the result is the
same rows the resident path returns; a plan it cannot serve, or an aggregate whose state outgrows
the envelope, returns `None` here and takes the path it would have taken anyway.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher import core
from batcher.api.orchestration.chunked_sideways import run_staged_sideways
from batcher.api.orchestration.sizing import projected_input_bytes
from batcher.api.orchestration.stages import read_scanned

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "CHUNKED_MIN_INPUT_BYTES",
    "UNITS_MIN_INPUT_BYTES",
    "chunk_worthy",
    "execute_chunked",
    "run_chunked",
    "units_worthy",
]

#: Projected bytes of the driving source above which chunked streaming is taken even when the
#: whole input would fit in memory. Below it the resident path's read is short enough that
#: overlapping it buys nothing and the chunked path's per-chunk parallel steps are pure overhead.
CHUNKED_MIN_INPUT_BYTES = 8 << 30


#: Projected input bytes above which the engine reads a Parquet driving source itself, row group
#: by row group (`io.formats.structured.parquet.units`), rather than having it decoded whole
#: first. Below it the read is small enough that how it is scheduled does not show.
#:
#: This was 64 MiB, on the reading that a short read gains nothing from overlapping with the
#: compute. The overlap was never the point at that size; the resident read's *fixed* costs
#: are -- a Python thread per file, each batch conformed by a Python loop (`_normalize`), the
#: whole relation crossing the FFI before the engine starts. ClickBench's
#: ``COUNT(*) WHERE AdvEngineID <> 0`` over 10M rows projects one `Int16` column, sat under
#: the old bar, and spent 17 ms reading resident plus 30 ms executing for 54 ms in all; read
#: by the engine's workers it took 16 ms, against DuckDB's 25. One-file reads of 1,000 to
#: 1,000,000 rows measured the same or faster on the unit path (2.7 vs 2.7 ms, 3.6 vs 4.1 ms,
#: 8.5 vs 12.4 ms grouped). The floor stays at 1 MiB only so that a query over in-memory
#: sources -- which never take this path -- does not pay the eligibility checks to find out.
UNITS_MIN_INPUT_BYTES = 1 << 20


def units_worthy(input_bytes: int) -> bool:
    """Whether an input this large is worth letting the engine read row group by row group."""
    return input_bytes >= UNITS_MIN_INPUT_BYTES


def chunk_worthy(input_bytes: int) -> bool:
    """Whether an input this large is worth streaming even when it would fit in memory."""
    return input_bytes >= CHUNKED_MIN_INPUT_BYTES


def run_chunked(
    plan, opt: PhysicalPlan, ctx, sources: list[Source], *, input_bytes: int, spill: bool
) -> pa.Table | None:
    """`execute_chunked` for the conductor: timed as a phase, returned as a table, or `None`.

    Tried when the query is going out of core (`spill`) or its projected input is large enough
    that streaming it pays (`chunk_worthy`, `units_worthy`); `None` without trying otherwise.
    """
    from batcher.api._join_helpers import _empty_result_schema
    from batcher.api.orchestration import phases

    python_chunks = spill or chunk_worthy(input_bytes)
    if not (python_chunks or units_worthy(input_bytes)):
        return None

    phases.begin("core.execute.chunked")
    mark = time.perf_counter()
    batches = execute_chunked(
        sources,
        opt,
        lambda i: projected_input_bytes(sources, opt.source_projections, [i]),
        python_chunks=python_chunks,
        feedback=ctx.hub,
        profile=ctx.profile,
    )
    phases.record("core.execute.chunked", time.perf_counter() - mark)
    if batches is None:
        return None
    return pa.Table.from_batches(
        batches,
        schema=batches[0].schema if batches else _empty_result_schema(plan, ctx.columns),
    )


def execute_chunked(
    sources: list[Source],
    opt: PhysicalPlan,
    input_bytes_of,
    *,
    python_chunks: bool = True,
    feedback=None,
    profile=None,
) -> list[pa.RecordBatch] | None:
    """The plan's result with its driving source streamed in chunks, or `None` when it cannot be.

    Args:
        sources: The plan's bound sources.
        opt: The optimized physical plan, with its pushed projections and predicates.
        input_bytes_of: Maps a source id to its projected input bytes; picks the driving one.
        python_chunks: Whether a source the engine cannot read itself may be streamed from the
            control plane in chunks. Without it only the engine's own row-group read is tried.
        feedback: Where the engine's row-group read records its per-operator metrics.
        profile: The query's profile, which the row-group read hands those metrics to as well.

    Returns:
        The result batches, or `None` when no source can stream itself, the engine cannot run
        the plan's shape chunked, or the aggregate state outgrew the envelope.
    """
    from batcher.dist.spill.scratch import iter_spill_chunks, spill_chunk_bytes

    # Always the largest source, never the one that would need no join swapped: the choice has
    # to be the same for a query and every subquery the control plane evaluates first, because
    # TPC-H q15 compares a grouped `sum` for *equality* with a `max` of the same sums, and two
    # executors summing in different orders disagree in the last bit. Preferring an unswapped
    # source was measured doing exactly that (q15 returned no row), and on q4/q10/q18 it
    # streamed a small table to hash `lineitem`, 1.4-1.7x slower.
    if opt.prefer_sideways:
        # A decorrelated aggregate over a relation the chunks do not drive: run the outer side
        # first and stream the aggregate restricted to its keys (`chunked_sideways`). Tried
        # before the plan is judged chunkable, because its stages can be when the plan is not:
        # a semi join whose build side is the largest relation (`chunked_sideways._sideways_join`).
        staged = run_staged_sideways(
            sources,
            opt,
            input_bytes_of,
            lambda srcs, stage: execute_chunked(
                srcs,
                stage,
                lambda i: projected_input_bytes(srcs, stage.source_projections, [i]),
                python_chunks=python_chunks,
            ),
        )
        if staged is not None:
            return staged
    driving = _driving_source(sources, opt, input_bytes_of)
    if driving is None or not core.plan_chunkable(opt, driving):
        return None
    # Kyber's sideways verdict on a plan that also reads the driving table through another binding
    # means the resident, materializing route restricts that binding's aggregate to the probe
    # side's keys, which the row-group read would bypass (TPC-H q21: ~660 ms routed, ~1,030 ms
    # not). Only then: with the table read once, the verdict is about an aggregate *over* the
    # streamed rows, and keeping the row-group read keeps a query and the subqueries evaluated
    # before it on one executor — TPC-H q15 compares their float sums for equality, and taking
    # this branch on the verdict alone returned no row.
    sideways = opt.prefer_sideways and _shared(sources, opt, driving)
    units = None if sideways else _unit_read(sources[driving], *_pushed(opt, driving))
    if units is None and not python_chunks:
        return None
    projection, predicate = _pushed(opt, driving)
    carrier = _schema_carrier(sources[driving], projection)
    if carrier is None:
        return None
    scanned = opt.scanned_source_ids()
    reads = read_scanned(sources, opt, set(scanned) - {driving})
    resident: list[list[pa.RecordBatch]] = [
        [carrier] if i == driving else reads[i][0] if i in reads else []
        for i in range(len(sources))
    ]
    try:
        if units is not None:
            out, ops, usage = core.execute_local_parquet(
                opt, resident, driving, units, _held_budget(), feedback
            )
            if profile is not None:
                profile.metric_ops = ops
                profile.record_usage(usage)
            return out
        chunks = iter_spill_chunks(sources[driving], projection, predicate, spill_chunk_bytes())
        return core.execute_local_chunked(opt, resident, driving, chunks, _held_budget())
    except Exception as exc:
        if type(exc).__name__ == "MemoryBudgetExceededError":
            return None  # the aggregate needs to spill: the out-of-core path will
        raise


def _pushed(opt: PhysicalPlan, i: int) -> tuple[list[str] | None, dict | None]:
    """The projection and predicate the plan pushed into source `i`'s scan."""
    return opt.source_projections.get(i), opt.source_predicates.get(i)


def _shared(sources: list[Source], opt: PhysicalPlan, i: int) -> bool:
    """Whether another binding the plan scans reads the same source object as binding `i`."""
    return any(j != i and sources[j] is sources[i] for j in opt.scanned_source_ids())


def _unit_read(source: Source, projection: list[str] | None, predicate: dict | None):
    """The engine's own row-group read of `source` (`parquet.units`), or None."""
    from batcher.io.formats.structured.parquet.source import ParquetSource
    from batcher.io.formats.structured.parquet.units import unit_read

    if type(source) is not ParquetSource:
        return None
    return unit_read(source, projection, predicate)


def _held_budget() -> int:
    """Bytes the chunked path may hold: half the memory envelope.

    What it holds grows with the answer, not the input — a grouped aggregate's state, or every
    output row of a plan with no aggregate (an adaptive stage whose root is a join). TPC-H sf100
    q9 stages `lineitem JOIN orders` whole, 600M rows, and held without a bound that was an OOM
    kill at 71 GB. Half leaves the resident build sides and the two in-flight chunks room.
    """
    from batcher._internal.hardware.memory import machine_memory_bytes
    from batcher.config import active_config

    return (active_config().memory.max_memory_bytes or machine_memory_bytes()) // 2


def _driving_source(sources: list[Source], opt: PhysicalPlan, input_bytes_of) -> int | None:
    """The scanned source with the most projected bytes, if it can read itself in chunks."""
    scanned = [i for i in opt.scanned_source_ids() if i < len(sources)]
    if not scanned:
        return None
    if any(opt.source_limits.get(i) is not None for i in scanned):
        return None  # a pushed row cap is a property of the whole read, not of each chunk
    driving = max(scanned, key=input_bytes_of)
    return driving if callable(getattr(sources[driving], "iter_chunks", None)) else None


def _schema_carrier(source: Source, projection: list[str] | None) -> pa.RecordBatch | None:
    """A zero-row batch with the columns the driving scan reads, in the order it reads them."""
    schema = source.schema()
    if projection is not None:
        try:
            schema = pa.schema([schema.field(c) for c in projection])
        except KeyError:
            return None
    return pa.RecordBatch.from_pylist([], schema=schema)
