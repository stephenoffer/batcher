"""Read a shard onto the device, instead of onto the host and then across the bus.

A GPU shard's read is the half of the query nobody accelerated. The worker asked pyarrow for
the rows, which decoded Parquet on the CPU into host memory, and only then did the frame cross
PCIe onto the device. For a scan-heavy query — the shape a GPU is worth using for — that decode
is most of the wall clock, and the device spends it idle waiting for a core to hand it
something. cuDF reads Parquet on the device: the compressed bytes cross, and the decode happens
on the thing that was going to compute anyway.

The gain is only real if the two readers produce the same rows, so the conditions are narrow
and every one of them is checked before the read rather than hoped for afterwards:

* the shard's splits must be plain Parquet locators of types both readers agree on, which is
  `io.splits.device`'s question, not this module's;
* **a pushed predicate must already have done its skipping.** The concern is bytes, not rows:
  a device read that ignored a predicate the host reader would have used to skip row groups
  would move far more of them than the read it replaced. A row-group locator was produced by
  applying that predicate to the footer at *plan* time, so the skipping has happened and both
  readers move the same bytes; a whole-file locator was never pruned, so a predicate still
  disqualifies it. `io.splits.device` draws that line;
* the frame that comes back must present the schema the host path would have. It is compared,
  not assumed — a device Parquet reader is a second implementation, and the one failure this
  cannot tolerate is a shard whose schema differs from its neighbours' in a concatenation.

Any of those failing returns `None`, which the caller reads as "use the host reader". That is
the same contract every other fallback in the GPU backend follows, and it is why turning this
on cannot change an answer.
"""

from __future__ import annotations

import collections
import threading
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    from batcher.core.gpu_plan import DfBackend

__all__ = [
    "cached_device_frame",
    "clear_device_frame_cache",
    "device_frame_cache_stats",
    "read_descriptor_on_device",
    "remember_source_dates",
    "reset_device_frame_cache",
]

#: Decoded device frames kept between queries: key -> (bytes, frame). An `OrderedDict` used as
#: an LRU, exactly as the host worker scan cache is.
_FRAME_CACHE: collections.OrderedDict = collections.OrderedDict()
_FRAME_CACHE_BYTES = 0
_FRAME_CACHE_LOCK = threading.Lock()
_FRAME_CACHE_HITS = 0
_FRAME_CACHE_MISSES = 0


def device_frame_cache_stats() -> dict[str, int]:
    """This worker's device frame cache: entries, bytes held, hits and misses."""
    with _FRAME_CACHE_LOCK:
        return {
            "entries": len(_FRAME_CACHE),
            "bytes": _FRAME_CACHE_BYTES,
            "hits": _FRAME_CACHE_HITS,
            "misses": _FRAME_CACHE_MISSES,
        }


def clear_device_frame_cache() -> int:
    """Drop every cached frame, returning the bytes released.

    Called before the subdivision ladder on a device out-of-memory: the cache is the one part
    of a worker's device footprint that is pure optimization, so it is the first thing to give
    back. A shard that then fits has cost one uncached read; a shard that still does not is
    subdivided exactly as it was before the cache existed.
    """
    global _FRAME_CACHE_BYTES
    with _FRAME_CACHE_LOCK:
        released = _FRAME_CACHE_BYTES
        _FRAME_CACHE.clear()
        _FRAME_CACHE_BYTES = 0
    return released


def reset_device_frame_cache() -> None:
    """Forget the cache *and* the memoized budget — for tests, and for a changed device."""
    clear_device_frame_cache()
    _BUDGET.clear()


#: The budget this process settled on, keyed by the inputs that can legitimately change it.
#: **Memoized because pricing the device is expensive, not because it is slow to compute.**
#: `visible_device_usable_bytes` reads the local inventory, this process's physical index, the
#: live telemetry and its own resident bytes — the same NVML sequence `prepare_device_memory`
#: measures at **414 ms**, and it was being paid once per shard. On a fan-out whose shards are a
#: fifth of a second each, sizing the cache cost more than the read it exists to avoid.
_BUDGET: dict[tuple[float, int], int] = {}


def _cache_budget_bytes() -> int:
    """How many bytes of cached frames this process may hold on its device.

    A share of the device's **usable** memory — the cache sits beside the shard being computed,
    the intermediates built over it and the CUDA context — net of the co-tenants Ray packed onto
    this board, since the cache is per process and the device is not.

    Memoized per `(fraction, tenants)`. Those are the only two inputs that differ between two
    calls in one worker: a stage packed four to a device gets a different answer than one that
    had the board to itself, and memoizing across that would hand the second stage the first
    one's budget.
    """
    from batcher.config import active_config
    from batcher.dist.gpu.resources import task_device_tenants

    cfg = active_config()
    fraction = min(0.9, max(0.0, float(cfg.distributed.gpu_frame_cache_fraction)))
    if fraction <= 0.0:
        return 0
    tenants = max(1, task_device_tenants())
    settled = _BUDGET.get((fraction, tenants))
    if settled is not None:
        return settled
    from batcher.carbonite.accel import visible_device_usable_bytes

    try:
        # Already net of the CUDA context, of what a co-tenant has resident, and of the
        # headroom — the same figure the RMM pool is planned against, so the cache and the
        # pool cannot each believe they own the same bytes.
        usable = float(visible_device_usable_bytes(cfg.accelerator.vram_headroom, tenants))
    except Exception as exc:
        note_suppressed("dist", "size the device frame cache", exc)
        usable = 0.0
    budget = int(usable * fraction)
    _BUDGET[(fraction, tenants)] = budget
    return budget


def _frame_key(descriptor: dict) -> tuple | None:
    """The cache key for a shard: which row-groups, which columns, which pushed predicate.

    The same three things the host worker scan cache keys on, and for the same reason: all
    three change the rows that come back. `identity()` encodes file and row-groups; a split
    that cannot identify itself makes the shard uncacheable rather than mis-keyed.

    **It carries no version, and that is inherited rather than overlooked.** A
    `RowGroupSplit`'s identity is `parquet:<path>:rg<ids>` — no mtime, no size — so a file
    *rewritten in place* would be served from either cache as it was before. Both rest on the
    same assumption, stated on `dist/executors/scan_read.py`: Parquet data files are immutable,
    which is how every lakehouse format writes (Delta, Iceberg and Hudi add files and rewrite
    manifests; they do not edit a data file). A deployment that rewrites data files under a live
    path must set `gpu_frame_cache_fraction` to `0` and `BATCHER_SCAN_CACHE_BYTES=0`, and the
    reason is the same for both.
    """
    splits = descriptor.get("splits")
    if not splits:
        return None
    try:
        ids = tuple(sorted(split.identity() for split in splits))
    except Exception as exc:
        note_suppressed("dist", "identify a shard for the device frame cache", exc)
        return None
    projection = descriptor.get("projection")
    columns = tuple(projection) if projection is not None else ()
    return (ids, columns, repr(descriptor.get("predicate")))


def _frame_bytes(frame) -> int:
    """Device bytes a frame occupies, or `0` when it will not say (then it is not cached)."""
    try:
        return int(frame.memory_usage(deep=True).sum())
    except Exception as exc:
        note_suppressed("dist", "measure a device frame for the cache", exc)
        return 0


def cached_device_frame(descriptor: dict, be: DfBackend):
    """The shard as a device frame, served from this worker's cache when it is warm.

    The device counterpart of `dist/executors/scan_read.py`'s worker scan cache, and the same
    argument for it: a GPU worker persists between tasks (`gpu_worker_reuse`), the split-to-
    worker assignment is deterministic, and Parquet is immutable — so the second query over the
    same shard, columns and predicate need not read anything at all. Measured on a T4 against a
    10 M-row shard of TPC-H `lineitem` projected to six columns, that is 0.12 s of a 0.22 s
    shard.

    A **shallow copy** is handed out, never the cached frame itself. The copy shares every
    column's buffer, so it costs a Python object rather than a device allocation, and it means
    an operator that adds a private column to what it was given — which `aggregate` and `sort`
    both do — cannot grow the cached entry.

    Returns `None` exactly where `read_descriptor_on_device` does, so the caller's host-reader
    fallback is unchanged.
    """
    global _FRAME_CACHE_BYTES, _FRAME_CACHE_HITS, _FRAME_CACHE_MISSES
    budget = _cache_budget_bytes()
    key = _frame_key(descriptor) if budget else None
    if key is not None:
        with _FRAME_CACHE_LOCK:
            hit = _FRAME_CACHE.get(key)
            if hit is not None:
                _FRAME_CACHE.move_to_end(key)
                _FRAME_CACHE_HITS += 1
            else:
                _FRAME_CACHE_MISSES += 1
        if hit is not None:
            # A hit skips the read, and the read is where the backend learns which columns are
            # calendar days. Without this the shard's dates come back as timestamps — the right
            # values in the wrong column, which the schema contract then refuses.
            remember_source_dates(descriptor, be)
            return hit[1].copy(deep=False)
    frame = read_descriptor_on_device(descriptor, be)
    if frame is None or key is None:
        return frame
    nbytes = _frame_bytes(frame)
    # A shard larger than the whole budget is served uncached rather than evicting everything
    # to hold one entry that the next shard would evict again.
    if nbytes <= 0 or nbytes > budget:
        return frame
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[key] = (nbytes, frame)
        _FRAME_CACHE_BYTES += nbytes
        while budget < _FRAME_CACHE_BYTES and len(_FRAME_CACHE) > 1:
            _evicted, (evicted_bytes, _f) = _FRAME_CACHE.popitem(last=False)
            _FRAME_CACHE_BYTES -= evicted_bytes
    return frame.copy(deep=False)


def read_descriptor_on_device(descriptor: dict, be: DfBackend):
    """The shard as a device frame read straight from storage, or `None` to use the host path.

    Args:
        descriptor: A descriptor from `partition_descriptors`.
        be: The dataframe backend that will compute on the result. A host backend declines
            immediately: reading "on the device" through pandas would be a second Parquet
            reader with none of the benefit, and the verification path must exercise the same
            code the engine's own tests cover.

    Returns:
        A device dataframe holding the shard's rows, or `None` when the shard is not
        device-readable, when a pushed predicate would still have skipped bytes this read
        cannot, or when the device reader produced a schema the host reader would not have.

    Examples:
        .. doctest::

            >>> import pandas as pd
            >>> from batcher.core.gpu_plan import DfBackend
            >>> from batcher.dist.gpu.device_read import read_descriptor_on_device
            >>> read_descriptor_on_device({"batches": []}, DfBackend(pd)) is None
            True
    """
    if not be.is_gpu:
        return None
    specs = _specs(descriptor)
    if specs is None:
        return None
    projection = descriptor.get("projection")
    _publish_transfer_path(specs)
    try:
        frame = _widen(_read_parquet(specs, projection), descriptor, projection, be)
    except Exception as exc:
        # A device read that fails is a slow shard, never a failed query: the host reader is
        # still there and still correct.
        note_suppressed("dist", "read a gpu shard on the device", exc)
        return None
    return frame if _schema_agrees(frame, descriptor, projection) else None


def remember_source_dates(descriptor: dict, be: DfBackend) -> None:
    """Tell `be` which of this shard's columns are calendar days.

    Neither the device reader nor the frame cache goes through `from_arrow`, which is where a
    backend normally learns this — and a DATE that entered a frame comes back out of `to_arrow`
    as a **timestamp** unless the backend was told. The result is a column of the right values
    and the wrong type, which is this tier's characteristic defect and the one its pandas-backed
    tests are structurally unable to see.

    It is a separate function because there are now two doors into a device frame and both have
    to walk through it. Measured on TPC-H q3 at sf10, the second one — a cache hit, which by
    construction skips the read — produced `o_orderdate` as `timestamp[s]` where the engine
    declares `date32[day]`; the schema contract refused the result and the CPU engine answered,
    so the query was correct and lost its device.
    """
    splits = descriptor.get("splits")
    if not splits:
        return
    try:
        be.remember_dates(splits[0].schema())
    except Exception as exc:
        note_suppressed("dist", "register a shard's date columns", exc)


def _widen(frame, descriptor: dict, projection: list[str] | None, be: DfBackend):
    """Widen the frame's narrow numeric columns, as the host path's `from_arrow` would.

    This reader never goes through `from_arrow`, so it does not get that widening for free —
    and without it a device-read shard contributes an `int32` column beside a host-read one's
    `int64`, which is exactly the concatenation a fan-out then has to make sense of. The source
    schema decides rather than the frame's own dtypes, so both readers reach the same answer
    from the same fact.
    """
    from batcher.core.gpu_plan.backend import widened_type

    schema = descriptor["splits"][0].schema()
    remember_source_dates(descriptor, be)
    names = list(projection) if projection is not None else list(frame.columns)
    for name in names:
        target = widened_type(schema.field(name).type)
        if target is not None:
            frame[name] = frame[name].astype(be.dtype(target))
    return frame


def _specs(descriptor: dict):
    """The device locators for this descriptor, or `None` when it must go through the host.

    The predicate goes to `device_read_specs` rather than disqualifying the read here. It once
    did disqualify it, which cost the device path exactly the queries it was built for: a
    scan-heavy query is a filtered one, so the decode this exists to move onto the device stayed
    on a CPU core for every TPC-H and ClickBench shape. What the predicate would have skipped,
    a `RowGroupSplit` has already skipped at plan time — see that module for which splits carry
    the pruning and which do not.
    """
    from batcher.io.splits.device import device_read_specs

    splits = descriptor.get("splits")
    if not splits:
        return None
    return device_read_specs(splits, descriptor.get("projection"), descriptor.get("predicate"))


def _publish_transfer_path(specs: list) -> None:
    """Report whether these files reach the device by DMA or through a host bounce buffer.

    A device-native read is two wins stacked: the decode happens on the device, and — only
    when GPUDirect Storage applies — the bytes travel storage-to-device without the host
    touching them. The second one silently does not happen on a container overlay, on a FUSE
    -mounted object store, and in an image without the cuFile library, and the read then adds
    a host copy to a path whose whole argument was avoiding one. Nothing about the result
    changes either way, which is exactly why it needs to be visible: a scan that is half its
    expected rate looks identical to one that is not.

    Best-effort and skipped entirely when nothing is listening, so an unobserved run pays
    neither the mount-table read nor the library probe.
    """
    from batcher._internal import events

    if not events.listening():
        return
    try:
        from batcher.io.splits.gds import gds_summary

        events.publish(
            events.GPU,
            name="device_read",
            event="transfer_path",
            **gds_summary(tuple(spec.path for spec in specs)),
        )
    except Exception as exc:  # pragma: no cover - observability must never fail a read
        note_suppressed("dist", "report the gpu read's transfer path", exc)


def _read_parquet(specs: list, projection: list[str] | None):
    """Read every locator in one cuDF call, so the files are fetched concurrently."""
    import cudf

    paths = [spec.path for spec in specs]
    kwargs: dict = {}
    if projection is not None:
        kwargs["columns"] = list(projection)
    # cuDF takes row groups as one list per path, and only when every path names some. A
    # whole-file split alongside a row-group one has no such list, so the pair is read whole
    # and the operator chain above sees the same rows either way.
    if all(spec.row_groups is not None for spec in specs):
        kwargs["row_groups"] = [list(spec.row_groups) for spec in specs]
    return cudf.read_parquet(paths, **kwargs)


def _schema_agrees(frame, descriptor: dict, projection: list[str] | None) -> bool:
    """Whether the device read produced the columns, in the order, the host read would have.

    Compared rather than trusted. Two Parquet readers agreeing on a file's *types* is what
    `io.splits.device` gates on; agreeing on its *columns and their order* is separate, because
    a projection is applied by two different mechanisms on the two paths, and a shard whose
    column order differs from its neighbours' silently corrupts the concatenation that follows.
    """
    try:
        expected = _expected_names(descriptor, projection)
        return expected is None or list(frame.columns) == expected
    except Exception as exc:
        note_suppressed("dist", "compare the device read's schema", exc)
        return False


def _expected_names(descriptor: dict, projection: list[str] | None) -> list[str] | None:
    """The column names the host reader would have returned, or `None` when it cannot say."""
    if projection is not None:
        return list(projection)
    splits = descriptor.get("splits")
    return list(splits[0].schema().names) if splits else None
