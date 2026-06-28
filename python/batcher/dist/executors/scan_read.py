"""Worker-side scan read primitives — how a distributed worker reads its split slice.

Split out of `partition_io` (which owns *partitioning* — assigning splits to workers)
because *reading* them is a distinct, throughput-critical concern. The dominant cost of a
distributed scan is object-store read throughput, so the reader is chosen for speed:

* `_read_split_batches` — the entry point. For uniform Parquet row-group splits it runs an
  async, coalesced **pyarrow dataset scan** (every assigned row-group read concurrently in
  C++ with column-chunk coalescing + readahead — ~5x a Python per-split read on a
  high-latency worker→S3 path, and it *streams* so a worker never materializes its whole
  partition). Anything else (or any failure building the scan) falls back to a bounded
  thread-pool prefetch over per-split reads.

`_SCAN_PREFETCH` / `_SPLIT_TARGET_BYTES` are read tuning the partitioner also consults, so
they live here as the single source of truth.
"""

from __future__ import annotations

import os
from inspect import signature

from batcher.io.splits import Split

# Splits a worker reads ahead concurrently while folding the current one. A distributed
# scan is I/O-LATENCY-bound on object storage (a single connection is far below a node's
# bandwidth, and each request waits ~tens of ms), so the number of in-flight reads — not
# bandwidth — caps throughput. Reading ahead overlaps I/O with compute and keeps many
# connections busy at once: on TPC-H sf100 raising this 8 → 32 cut the distributed agg
# ~53s → ~31s (it plateaus past 32). Bounded: peak memory is ≤ `depth` in-flight splits.
# Module-level (not `config`) so it applies on a worker without shipping the driver's
# config_context; env-overridable for wider tables / tighter RAM.
_SCAN_PREFETCH = max(1, int(os.environ.get("BATCHER_SCAN_PREFETCH", "32")))

# Target compressed bytes per scan split — coalesces a source's native chunks. Parquet
# files with many small row-groups (sf100 lineitem: 49/file → 4,900 one-row-group splits)
# make per-request latency the bottleneck; packing adjacent row-groups to this size turns
# hundreds of tiny GETs per worker into a few dozen large reads. `_scan_splits` applies it
# only while enough splits remain to keep the fan-out busy. Env-overridable.
_SPLIT_TARGET_BYTES = max(1 << 20, int(os.environ.get("BATCHER_SPLIT_TARGET_BYTES", str(64 << 20))))

# Object-store read concurrency for the dataset scan. The scan is S3-LATENCY-bound, so
# throughput tracks the number of in-flight range requests, which pyarrow caps at the
# global IO thread pool — whose default of 8 throttles a 16-core worker to ~120 MB/s.
# Raising it to 32 (with matching fragment/batch readahead) measured ~6x on a TPC-H sf100
# worker (121 → 716 MB/s); it plateaus past ~32 threads. Set once per process (idempotent,
# global); `fragment_readahead` is how many files a worker reads at once, `batch_readahead`
# how far it reads into each. All env-overridable.
_IO_THREADS = max(8, int(os.environ.get("BATCHER_IO_THREADS", "32")))
_FRAGMENT_READAHEAD = max(2, int(os.environ.get("BATCHER_FRAGMENT_READAHEAD", "32")))
_BATCH_READAHEAD = max(2, int(os.environ.get("BATCHER_BATCH_READAHEAD", "64")))


def _read_split_batches(splits, projection, predicate):
    """Stream `splits`' batches (projection/predicate pushed) via the fastest reader.

    The coalesced dataset scan for Parquet row-group splits, else the prefetch pool. Both
    stream and push projection/predicate, so the result is identical and a worker never
    holds its whole partition.
    """
    scan = _dataset_scan_batches(splits, projection, predicate)
    if scan is not None:
        yield from scan
    else:
        yield from _prefetch_split_reads(splits, projection, predicate, _SCAN_PREFETCH)


def _dataset_scan_batches(splits, projection, predicate):
    """A streaming pyarrow dataset scanner over `splits` as Parquet row-group fragments,
    or `None` when they aren't all uniform Parquet row-group splits OR the scan can't be
    built (caller then falls back). Reads the worker's row-groups concurrently in C++
    (`pre_buffer` coalesces the projected column-chunk byte ranges; fragment/batch
    readahead overlap I/O) — no Python read loop, no whole-partition materialization.
    Result-invariant: same rows/columns as the per-split read."""
    from batcher.io.splits import RowGroupSplit

    if not splits or not all(isinstance(s, RowGroupSplit) for s in splits):
        return None
    try:
        import pyarrow as pa
        import pyarrow.dataset as pads

        from batcher.io.filesystem import resolve_filesystem

        pa.set_io_thread_count(_IO_THREADS)  # idempotent; lifts the 8-thread S3 read cap
        fsw = resolve_filesystem(splits[0].path)
        pafs = getattr(fsw, "_fs", None)
        if pafs is None:
            return None
        fmt = pads.ParquetFileFormat(
            default_fragment_scan_options=pads.ParquetFragmentScanOptions(pre_buffer=True)
        )
        frags = [
            fmt.make_fragment(
                fsw._p(s.path).rstrip("/"), filesystem=pafs, row_groups=list(s.row_groups)
            )
            for s in splits
        ]
        dset = pads.FileSystemDataset(frags, frags[0].physical_schema, fmt, pafs)
        expr = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            expr = to_pyarrow_expression(predicate)
        return dset.scanner(
            columns=projection,
            filter=expr,
            use_threads=True,
            batch_readahead=_BATCH_READAHEAD,
            fragment_readahead=_FRAGMENT_READAHEAD,
        ).to_batches()
    except Exception:  # any scan-build failure → fall back to the per-split pool
        return None


def _split_read(split: Split, projection: list[str] | None, predicate: dict | None) -> list:
    """Read a split, passing `predicate` only if its `read` accepts one."""
    if predicate is not None and "predicate" in signature(split.read).parameters:
        return split.read(projection, predicate=predicate)
    return split.read(projection)


def _prefetch_split_reads(splits, projection, predicate, depth: int):
    """Yield each split's batches **in order**, reading up to `depth` splits ahead on a
    thread pool so object-store I/O overlaps the caller's per-split compute and several
    reads run at once. `depth <= 1` (or a single split) is the plain sequential read.

    Memory is bounded to at most `depth` in-flight split reads — the map-side fold consumes
    each before the window advances, so a wide partition never materializes whole. Read
    order is preserved (a FIFO of futures), so a downstream that assumes file order is
    unaffected.
    """
    if depth <= 1 or len(splits) <= 1:
        for s in splits:
            yield from _split_read(s, projection, predicate)
        return

    import collections
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=depth) as pool:
        pending: collections.deque = collections.deque()
        it = iter(splits)
        for s in _take(it, depth):
            pending.append(pool.submit(_split_read, s, projection, predicate))
        while pending:
            batches = pending.popleft().result()  # raises if the read failed
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(_split_read, nxt, projection, predicate))
            yield from batches


def _take(it, n: int):
    """The next ≤`n` items of `it` (priming the prefetch window)."""
    out = []
    for _ in range(n):
        x = next(it, None)
        if x is None:
            break
        out.append(x)
    return out
