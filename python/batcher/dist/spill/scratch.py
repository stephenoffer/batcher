"""Spill scratch: where an out-of-core query's bytes go, and how its input is fed in.

The plumbing every spilling breaker stands on, independent of *which* breaker is spilling:
resolving the scratch directory, building the tiered (local -> remote) store, capping the
open-file fan-out, and streaming a source into morsels small enough that the input never
bounds peak memory. `aggregate` and `dist.spill_breakers` are the operators that use it.
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa

from batcher.carbonite.spill import TieredSpillStore
from batcher.config import active_config
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan


def _work_dir(spill_dir: str | None, prefix: str) -> tuple[str, bool]:
    """Resolve the local scratch dir for a spill, and whether we own it (rmtree it).

    An explicit `spill_dir` is caller-owned (not removed). Otherwise, if the config
    sets `MemoryConfig.spill_dir`, create a unique per-query subdir *under* that root
    (so striping onto fast/large disks is honored and rmtree only ever removes our
    own subdir — never a shared root). With neither, fall back to the node's measured
    local scratch volume, and to a system tempdir only when there is none.

    That last step matters on a GPU node, where a system tempdir is an overlay on the
    container root — commonly under 100 GB and shared with the image and every other
    tenant — while the several terabytes of local NVMe the node ships with are mounted
    under a provider-specific name. Spilling to the tempdir there fails with `ENOSPC`
    beside unused storage, and the failure looks like an undersized query rather than a
    misplaced directory.
    """
    from batcher._internal.site import local_scratch_root

    if spill_dir is not None:
        return spill_dir, False
    root = active_config().memory.spill_dir or local_scratch_root()
    if root:
        os.makedirs(root, exist_ok=True)
        return tempfile.mkdtemp(prefix=prefix, dir=root), True
    return tempfile.mkdtemp(prefix=prefix), True


def _make_store(work_dir: str) -> TieredSpillStore:
    """A tiered spill store for `work_dir`, configured from the active `Config`.

    Local NVMe by default; overflows to `MemoryConfig.spill_remote_uri` once the
    local budget is exhausted, so an out-of-core query survives a full local disk.
    Spilled batches are compressed with the configured codec.
    """
    mem = active_config().memory
    return TieredSpillStore(
        work_dir,
        remote_uri=mem.spill_remote_uri,
        local_budget_bytes=mem.spill_local_budget_bytes,
        compression=mem.spill_compression,
    )


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
        out = (
            pending[0]
            if len(pending) == 1
            else pa.Table.from_batches(pending).combine_chunks().to_batches()[0]
        )
        pending.clear()
        pending_bytes = 0
        return out

    for batch in source.iter_batches(projection):
        n = batch.num_rows
        if n == 0:
            continue
        nbytes = batch.nbytes
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
