"""Spill scratch: where an out-of-core query's bytes go, and how its input is fed in.

The plumbing every spilling breaker stands on, independent of *which* breaker is spilling:
resolving the scratch directory, building the tiered (local -> remote) store, capping the
open-file fan-out, and streaming a source into morsels small enough that the input never
bounds peak memory. `aggregate` and `dist.spill_breakers` are the operators that use it.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.carbonite.spill.scratch import make_store, scratch_dir
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan
from batcher.plan.types import logical_bytes, one_batch

#: The scratch-directory and store construction both this path and the result cache's
#: disk tier need, defined once in `carbonite.spill.scratch`. Aliased under the private
#: names this module's callers already import rather than re-implemented here: `dist` may
#: import `carbonite`, and a second copy of the volume-selection rule is exactly the
#: copy-paste `.claude/rules/architecture.md` forbids between subsystems.
_work_dir = scratch_dir
_make_store = make_store


# Byte target the out-of-core partition phase feeds the engine at once. A source's
# batch size is not the engine's to trust: it can be far too large (`from_arrow` of a
# whole table, a fat parquet row group) or far too small (a streaming reader, a
# per-file scan, an exploded/filtered upstream emitting thousands of tiny batches).
# Both hurt.
#
#   * Too large: the parallel partial-aggregate builds per-thread hash tables over the
#     entire batch's cardinality, so peak memory scales with the batch, not the morsel
#     — a high-cardinality group-by peaked ~2.6x higher on one 20M-row batch than on
#     the same rows normalized here.
#   * Too small: the partition phase makes one engine dispatch per batch, and a batch
#     far under a morsel-group can't fill the cores — 256-row batches ran ~30x slower
#     than 256K-row batches through the identical spill.
#
# Normalizing every source to ~this target (split the over-large, coalesce runs of the
# under-large) caps the partition phase's working set *and* keeps each chunk wide
# enough to fan across all cores — so out-of-core throughput no longer depends on how
# the source happened to chunk its output.
_SPILL_INPUT_CHUNK_BYTES = 8 << 20  # 8 MiB


def map_projection(plan: LogicalPlan, source_id: int) -> list[str] | None:
    """The columns `source_id` must produce for `plan` — Kyber's answer, for a spill phase.

    Every out-of-core partition phase reads its source through `_iter_spill_morsels`, and
    every one of them read it *whole*: the `projection` parameter existed and no call site
    ever passed one. Out-of-core is exactly where that costs the most, because the source
    read is the dominant IO of a spilling aggregate, join, sort, or window, and a column the
    plan never touches is decoded, chunked, hash-partitioned, compressed, written to disk,
    and read back again.

    Asked of the **breaker**, not of its map sub-plan. The sub-plan a partition phase
    executes per morsel is often a bare `Scan`, which requires every column by definition —
    what narrows the read is the operator above it (the aggregate's keys and arguments, the
    sort's key and output, the join's keys and projection). Asking Kyber keeps the decision
    in Kyber's lane.

    Args:
        plan: The breaker being spilled, rooted above the scan.
        source_id: The scan whose projection is wanted.

    Returns:
        The projection for that source, or ``None`` when the plan does not narrow it.
    """
    from batcher import kyber

    return kyber.required_columns_per_source(plan).get(source_id)


def _iter_spill_morsels(source: Source, projection: list[str] | None = None):
    """Yield `source`'s batches normalized to ~``_SPILL_INPUT_CHUNK_BYTES``.

    Over-large batches are split into zero-copy `slice` views (bounded without a
    copy); runs of small batches are coalesced into one chunk so the partition phase
    always processes an efficiently-sized, all-cores-wide morsel-group regardless of
    the source's batching. This is the single input tap every out-of-core partition
    phase (aggregate/join/sort/window) reads through; coalescing/splitting only
    reshapes the row stream, so every spill result is byte-identical.
    """
    pending: list[pa.RecordBatch] = []
    pending_bytes = 0

    def _flush() -> pa.RecordBatch | None:
        nonlocal pending_bytes
        if not pending:
            return None
        # One buffered batch needs no copy; a run is compacted into a single 0-offset
        # batch so the engine sees one contiguous chunk, not a chain of tiny ones.
        # `one_batch` is the shared compaction: it keeps every row and raises rather than
        # returning a prefix when a flush genuinely exceeds the 32-bit offset limit, which
        # a spilled multimodal or embedding column can.
        out = one_batch(pending)
        pending.clear()
        pending_bytes = 0
        return out

    for batch in source.iter_batches(projection):
        n = batch.num_rows
        if n == 0:
            continue
        # `logical_bytes`, not `batch.nbytes`: this is the single tap every out-of-core
        # phase reads through, and a bare `nbytes` *raises* on the Arrow view layouts
        # (`string_view`/`binary_view`/`list_view`) that a Parquet reader with view types
        # on, DuckDB, Polars, or any Velox-backed producer hands over. A spilling query
        # over such a column died in the sizing arithmetic — before a byte was spilled —
        # with an ArrowTypeError naming a layout the user never chose.
        nbytes = logical_bytes(batch)
        if nbytes >= _SPILL_INPUT_CHUNK_BYTES:
            # Emit any buffered small batches first (order-preserving), then split.
            buffered = _flush()
            if buffered is not None:
                yield buffered
            if n == 1:
                yield batch
            else:
                rows = max(1, (_SPILL_INPUT_CHUNK_BYTES * n) // nbytes)
                for off in range(0, n, rows):
                    yield batch.slice(off, min(rows, n - off))
        else:
            pending.append(batch)
            pending_bytes += nbytes
            if pending_bytes >= _SPILL_INPUT_CHUNK_BYTES:
                yield _flush()
    tail = _flush()
    if tail is not None:
        yield tail


# Cap on simultaneously-open spill files: the partition phase holds one writer per
# non-empty bucket open at once, so an unbounded `num_partitions` would exhaust the
# process file-descriptor limit at scale. Capping keeps FDs bounded; a bigger
# data volume is then absorbed by grace recursion into larger-then-split
# buckets rather than more files.
_FD_SAFE_PARTITIONS = 1024


def _fd_safe(n_buckets: int) -> int:
    return max(1, min(n_buckets, _FD_SAFE_PARTITIONS))
