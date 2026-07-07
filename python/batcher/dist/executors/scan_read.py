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

import collections
import os
import threading
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


# Native Rust parquet reader (bc-io via bc_py): decodes parquet over object_store
# (S3/GCS/Azure/HTTP/local), fetching the projected row-groups concurrently. It MATCHES
# pyarrow single-node, but under concurrent distributed load (all workers reading at once)
# object_store's HTTP client trails pyarrow's AWS C++ SDK (~3x on the cluster), so it is
# OPT-IN for the distributed S3 path (`BATCHER_NATIVE_READER=1`) until that concurrency
# gap is closed. It still serves direct reads and non-S3 backends; the pyarrow dataset
# scan (well-tuned: 32 IO threads + readahead) remains the default distributed reader.
_NATIVE_READER = os.environ.get("BATCHER_NATIVE_READER", "0") not in ("0", "false", "")

# Row-groups per native read call. The native reader returns a *materialized* batch list,
# so reading a whole file at once would buffer the worker's entire partition — defeating
# the streaming partial-aggregate's bounded memory (and its read/compute overlap). Reading
# in windows of this many row-groups bounds the in-flight memory to ~one window while
# still fetching that window's row-groups concurrently. Env-overridable.
_NATIVE_RG_WINDOW = max(1, int(os.environ.get("BATCHER_NATIVE_RG_WINDOW", "8")))


# --- Worker scan cache: decoded batches kept on the persistent worker between queries ---
# Distributed scans are object-store-read-bound; caching the decoded Arrow batches on the
# (session-fleet) worker lets a warm/repeated query skip both the S3 fetch and the decode
# and run at compute speed. Bounded LRU by total cached bytes — defaults to a fraction of
# the node's RAM so it never crowds out the working set; `BATCHER_SCAN_CACHE_BYTES=0`
# disables it. Lives on the worker process, so it persists exactly as long as the fleet.
def _default_scan_cache_cap() -> int:
    frac = max(0.0, float(os.environ.get("BATCHER_SCAN_CACHE_FRACTION", "0.3")))
    try:
        import psutil

        total = psutil.virtual_memory().total
    except Exception:
        total = 8 * 1024**3
    return int(total * frac)


_SCAN_CACHE_CAP = int(os.environ.get("BATCHER_SCAN_CACHE_BYTES", str(_default_scan_cache_cap())))
_SCAN_CACHE: collections.OrderedDict = collections.OrderedDict()  # key -> (bytes, [batches])
_SCAN_CACHE_BYTES = 0
_SCAN_CACHE_LOCK = threading.Lock()


def _all_rowgroup(splits) -> bool:
    from batcher.io.splits import RowGroupSplit

    return all(isinstance(s, RowGroupSplit) for s in splits)


def _scan_cache_key(splits, projection, predicate) -> tuple:
    """A stable key for a partition's decoded batches: which row-groups, which columns,
    which pushed predicate. `identity()` encodes file + row-groups; the projection and
    predicate are part of the result, so they must be part of the key."""
    ids = tuple(sorted(s.identity() for s in splits))
    proj = tuple(projection) if projection is not None else ()
    return (ids, proj, repr(predicate))


def _scan_cache_get(key):
    with _SCAN_CACHE_LOCK:
        hit = _SCAN_CACHE.get(key)
        if hit is None:
            return None
        _SCAN_CACHE.move_to_end(key)  # LRU: mark most-recently-used
        return hit[1]


def _scan_cache_put(key, batches, nbytes) -> None:
    global _SCAN_CACHE_BYTES
    if nbytes > _SCAN_CACHE_CAP:
        return
    with _SCAN_CACHE_LOCK:
        if key in _SCAN_CACHE:
            return
        _SCAN_CACHE[key] = (nbytes, batches)
        _SCAN_CACHE_BYTES += nbytes
        while _SCAN_CACHE_BYTES > _SCAN_CACHE_CAP and _SCAN_CACHE:
            _evk, (evb, _ev) = _SCAN_CACHE.popitem(last=False)  # evict least-recently-used
            _SCAN_CACHE_BYTES -= evb


def _read_split_batches(splits, projection, predicate):
    """Stream `splits`' decoded batches, serving from the worker's scan cache when warm.

    On a persistent (session-fleet) worker, the SAME splits route here on every query
    (the split→worker assignment is deterministic), so the first read decodes from object
    storage and caches the result and later reads of the same data + projection +
    predicate skip S3 and decode entirely — the warm path runs at compute speed, where
    Batcher beats the read-bound competition. Parquet files are immutable, so a cached
    decode is byte-identical to a fresh one. Falls through to a normal (uncached) read
    when caching is off, the splits aren't cacheable row-groups, or the partition exceeds
    the cache budget (then it just streams)."""
    if not (_SCAN_CACHE_CAP > 0 and splits and _all_rowgroup(splits)):
        yield from _read_split_batches_uncached(splits, projection, predicate)
        return
    key = _scan_cache_key(splits, projection, predicate)
    cached = _scan_cache_get(key)
    if cached is not None:
        yield from cached
        return
    # Miss: stream the fresh read while accumulating it for the cache. If the partition
    # outgrows the budget, stop accumulating and just stream (never balloon memory).
    acc: list | None = []
    acc_bytes = 0
    for batch in _read_split_batches_uncached(splits, projection, predicate):
        yield batch
        if acc is not None:
            acc.append(batch)
            acc_bytes += batch.nbytes
            if acc_bytes > _SCAN_CACHE_CAP:
                acc = None  # too large to cache; keep streaming
    if acc is not None:
        _scan_cache_put(key, acc, acc_bytes)


def _read_split_batches_uncached(splits, projection, predicate):
    """The reader itself (no cache): native Rust for predicate-free row-group splits, else
    the coalesced pyarrow dataset scan, else the prefetch pool. All stream and push
    projection, so the result is identical and a worker never holds its whole partition."""
    if _NATIVE_READER and predicate is None:
        native = _native_scan_batches(splits, projection)
        if native is not None:
            yield from native
            return
    scan = _dataset_scan_batches(splits, projection, predicate)
    if scan is not None:
        yield from scan
    else:
        yield from _prefetch_split_reads(splits, projection, predicate, _SCAN_PREFETCH)


def _native_scan_batches(splits, projection):
    """Read uniform Parquet row-group splits with the native Rust reader, or `None`.

    Groups the splits by file and reads each file's requested row-groups in one native
    call (which fetches them concurrently). Returns `None` (caller falls back to pyarrow)
    when the splits aren't all `RowGroupSplit`s or the native extension/read is
    unavailable — so an unsupported scheme or any read error never fails the scan.
    """
    from batcher.io.splits import RowGroupSplit

    if not splits or not all(isinstance(s, RowGroupSplit) for s in splits):
        return None
    try:
        import batcher._native as nat
        from batcher.config import active_config

        batch_rows = active_config().execution.morsel_rows
    except Exception:
        return None
    # Preserve file order; union each file's requested row-groups.
    by_file: dict[str, list[int]] = {}
    for s in splits:
        by_file.setdefault(s.path, []).extend(s.row_groups)
    cols = list(projection) if projection is not None else None

    def _gen():
        for path, rgs in by_file.items():
            uri = _native_uri(path)
            ordered = sorted(set(rgs))
            # Window the row-groups so the worker streams ~one window at a time (bounded
            # memory + read/compute overlap) instead of materializing its whole partition.
            for i in range(0, len(ordered), _NATIVE_RG_WINDOW):
                window = ordered[i : i + _NATIVE_RG_WINDOW]
                yield from nat.read_parquet(uri, window, cols, batch_rows)

    # Probe the first read eagerly so a failure falls back to pyarrow instead of yielding
    # a half-stream; on success, chain the probed batches back in.
    gen = _gen()
    try:
        first = next(gen, _SENTINEL)
    except Exception:
        return None
    if first is _SENTINEL:
        return iter(())
    return _chain_first(first, gen)


_SENTINEL = object()


def _chain_first(first, rest):
    yield first
    yield from rest


# Resolved S3 bucket → region, cached. Worker nodes often lack AWS_REGION in their env,
# and object_store needs the region to address the bucket; resolve it once per bucket
# (pyarrow's GetBucketLocation) and pass it on the URI so the native reader is region-
# correct without relying on worker environment.
_S3_REGION: dict[str, str] = {}


def _native_uri(path: str) -> str:
    """The URI to hand the native reader, with the S3 region appended for `s3://` paths.

    A no-op for non-S3 schemes / local paths, and when the URI already carries a region.
    """
    if not path.startswith(("s3://", "s3a://")) or "region=" in path:
        return path
    bucket = path.split("://", 1)[1].split("/", 1)[0]
    region = _S3_REGION.get(bucket)
    if region is None:
        try:
            import pyarrow.fs as pafs

            region = pafs.resolve_s3_region(bucket)
        except Exception:
            region = ""
        _S3_REGION[bucket] = region
    if not region:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}region={region}"


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
