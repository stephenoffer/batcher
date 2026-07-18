"""Tiered spill storage — keep large state alive under bounded memory, at any scale.

When an operator's state won't fit memory, Carbonite spills it to disk and, when
the local tier fills, overflows to object storage — so an out-of-core query keeps
running even past the local disk's capacity (the PB-scale case). Two tiers:

- `LOCAL` — Arrow IPC files on local disk (NVMe): fast, capacity-bounded.
- `REMOTE` — any `fsspec` URL (`s3://`, `gs://`, `memory://`): effectively
  unbounded, slower; used only once the local budget is exhausted.

Writes **stream** batch-by-batch straight to the tier's IPC stream — the store
never buffers a whole partition in memory (that would reintroduce the very OOM
spilling exists to avoid). Reads stream back the same way (a memory-mapped local
file, a chunked remote read). `fsspec` is an optional dependency — the remote tier
raises a clear error if it (or the `cloud` extra) is absent, while the local tier
always works.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from batcher._internal.errors import IOError as BatcherIOError
from batcher._internal.errors import ResourceError

__all__ = ["SpillHandle", "SpillTier", "TieredSpillStore"]

# Never let the local spill tier fill the scratch filesystem past this fraction of its
# measured free space: the last sliver keeps room for other tenants, log/tmp writes, and
# the filesystem metadata that a 100%-full disk starves — a full scratch disk is a hard
# query failure, not a slow one.
_SPILL_DISK_FRACTION = 0.9


def _clamp_to_free_disk(local_dir: str, budget: int | None) -> int | None:
    """Clamp a configured local spill budget to a safe fraction of *measured* free disk.

    The static config budget is a guess about a disk it has never seen: on a
    smaller-than-configured scratch volume it lets the local tier fill the filesystem
    before overflow to the remote tier ever triggers (a hard OOM-on-disk); on a larger one
    it overflows to slow object storage prematurely. Measuring the scratch filesystem's
    free space makes overflow track the disk that actually exists — and derives a budget at
    all when none was configured. Best-effort: if the volume can't be stat'd, the configured
    value stands unchanged, so behavior only ever gets safer, never more fragile.
    """
    # The spill dir is created lazily (on first spill), so stat the nearest *existing*
    # ancestor — the same filesystem the dir will live on — instead of losing the disk-aware
    # clamp entirely on the common first-use path (which would silently keep the too-large
    # configured budget, the OOM-on-disk this guards against).
    free = None
    path = os.path.abspath(local_dir)
    while True:
        try:
            free = shutil.disk_usage(path).free
            break
        except OSError:
            parent = os.path.dirname(path)
            if parent == path:  # reached the root and still couldn't stat
                break
            path = parent
    if free is None:  # pragma: no cover - even the root failed to stat
        return budget
    safe = int(free * _SPILL_DISK_FRACTION)
    return safe if budget is None else min(budget, safe)


def _open_local_map(path: str) -> pa.MemoryMappedFile:
    """Memory-map a local spill file, mapping a *missing* file to a retryable error.

    A spot/preemptible node's local NVMe is ephemeral: it can be reclaimed mid-query,
    vanishing the spilled partition. Reading it then fails with a cryptic `OSError`; we
    surface a clear, **retryable** `ResourceError` instead so the distributed recovery
    path recomputes the partition (the recovery loop already treats `ResourceError` as
    worker/disk loss) rather than crashing the query. Set `memory.spill_remote_uri` for
    a durable overflow tier that survives the node.
    """
    try:
        return pa.memory_map(path, "r")
    except OSError as exc:
        raise ResourceError(
            f"spilled partition is unreadable at {path!r} — its local disk was likely "
            "reclaimed (an ephemeral/spot node). The partition must be recomputed; set "
            "memory.spill_remote_uri for a durable overflow tier that survives node loss."
        ) from exc


class SpillTier(Enum):
    """Which storage tier a spilled partition lives on."""

    LOCAL = "local"  # Arrow IPC on local disk (NVMe)
    REMOTE = "remote"  # object storage via fsspec


@dataclass(frozen=True, slots=True)
class SpillHandle:
    """An opaque reference to one spilled partition (tier + path + sizes).

    `nbytes` is the **compressed, on-disk** size (what the local-budget accounting charges).
    `logical_nbytes` is the **uncompressed, in-memory** size of everything written — what a
    reducer must budget against before reading the bucket back into RAM, since a compressible
    bucket's on-disk size can be many times smaller than its resident footprint.
    """

    tier: SpillTier
    path: str
    nbytes: int
    logical_nbytes: int = 0


def _fsspec_open(path: str, mode: str):
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BatcherIOError(
            f"spilling to object storage ({path!r}) needs fsspec — install the 'cloud' extra"
        ) from exc
    return fsspec.open(path, mode)


def _ipc_options(compression: str | None) -> pa.ipc.IpcWriteOptions | None:
    """Arrow-IPC write options for the configured codec, or `None` if unavailable.

    Spilled data is transient, so a cheap-fast codec (LZ4) trades CPU for disk I/O
    and footprint. ``"auto"`` (the datatype-aware default) is uncompressed here: this
    Python tier spills to fast local disk, where compressing numeric/string state
    costs more CPU than the I/O it saves (the native Rust path's per-batch classifier
    compresses only blob payloads, where it pays). Set ``"lz4"``/``"zstd"`` explicitly
    to force compression on this tier (worthwhile for the slow remote-overflow path).
    Degrades silently to uncompressed if the codec isn't built into this pyarrow, so
    spilling never fails on a missing optional codec.
    """
    if not compression or compression == "auto":
        return None
    try:
        return pa.ipc.IpcWriteOptions(compression=compression)
    except (ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return None


def _remote_ipc_options(compression: str | None) -> pa.ipc.IpcWriteOptions | None:
    """IPC write options for the REMOTE tier — **always compressed**.

    Object storage is slow and priced by bytes transferred, so the CPU of a cheap codec
    always pays there even when the fast local NVMe tier stays uncompressed. An unset or
    ``"auto"`` codec is upgraded to LZ4; an explicit ``"lz4"``/``"zstd"`` is honored.
    Degrades to uncompressed only if the codec isn't built into this pyarrow.
    """
    codec = compression if compression in ("lz4", "zstd") else "lz4"
    return _ipc_options(codec)


class _BucketWriter:
    """Streams batches for one spill bucket to its tier, chosen at first write.

    The tier is decided lazily on the first batch (so an empty bucket opens no file
    and costs nothing): REMOTE when a remote URI is configured and the local budget
    is already exhausted, else LOCAL. Batches stream straight to the IPC writer — the
    partition is never held whole in memory.
    """

    def __init__(self, store: TieredSpillStore, name: str) -> None:
        self._store = store
        self._name = name
        self._tier: SpillTier | None = None
        self._path: str | None = None
        self._fh = None
        self._writer: pa.ipc.RecordBatchStreamWriter | None = None
        # Bytes this writer has streamed to the LOCAL tier but not yet finalized. Counted
        # live against the store's local budget so a *sibling* bucket opened later routes to
        # the remote tier once the buckets already streaming have exhausted the local budget
        # — the on-close accounting alone cannot see an open bucket's growth (see the store).
        self._pending_bytes = 0

    def write(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        if self._writer is None:
            self._open(batch.schema)
        self._writer.write_batch(batch)
        if self._tier is SpillTier.LOCAL:
            # Charge the batch's (uncompressed, in-memory) size to the store's live local
            # usage as it lands, not just at close. A slight over-estimate vs the compressed
            # on-disk size only makes overflow trigger a touch early — safe (never over-fills
            # the disk), and result-invariant (the remote tier reads back identically).
            self._pending_bytes += batch.nbytes
            self._store._local_pending += batch.nbytes

    def _open(self, schema: pa.Schema) -> None:
        store = self._store
        overflow = (
            store._remote_uri is not None
            and store._local_budget is not None
            and store._local_used + store._local_pending >= store._local_budget
        )
        if overflow:
            self._tier = SpillTier.REMOTE
            self._path = f"{store._remote_uri}/{self._name}.arrow"
            self._fh = _fsspec_open(self._path, "wb").open()
            # The remote tier is slow object storage that charges for bytes transferred, so
            # LZ4 always pays there even when the local NVMe tier stays uncompressed.
            opts = _remote_ipc_options(store._compression)
            self._writer = pa.ipc.new_stream(self._fh, schema, options=opts)
        else:
            self._tier = SpillTier.LOCAL
            self._path = os.path.join(store._local_dir, f"{self._name}.arrow")
            self._fh = pa.OSFile(self._path, "wb")
            opts = _ipc_options(store._compression)
            self._writer = pa.ipc.new_stream(self._fh, schema, options=opts)

    def close(self) -> SpillHandle | None:
        """Finalize the bucket. Returns its handle, or `None` if it got no rows."""
        if self._writer is None:
            return None
        self._writer.close()
        self._fh.close()
        # The uncompressed (in-memory) size of everything written — captured before the LOCAL
        # branch zeroes the pending estimate. The reducer budgets against this, not the
        # compressed on-disk `nbytes` below (which for a compressible bucket can be far smaller
        # and would let an over-large bucket skip re-spill recursion and OOM the finalize).
        logical_nbytes = self._pending_bytes
        if self._tier is SpillTier.LOCAL:
            # This bucket is finalized: hand its bytes from the live "pending" estimate to
            # the store's confirmed `_local_used` (`_on_closed` adds the real file size), so
            # the two are never double-counted.
            self._store._local_pending -= self._pending_bytes
            self._pending_bytes = 0
        nbytes = self._store._on_closed(self._tier, self._path)
        return SpillHandle(self._tier, self._path, nbytes, logical_nbytes)


class TieredSpillStore:
    """Streams `RecordBatch` buckets to local disk, overflowing to object storage.

    Open a streaming `writer(name)` per bucket and `write` batches to it as they are
    produced (never materializing the whole partition); `close()` returns a
    `SpillHandle`. New buckets overflow to the remote tier once the cumulative local
    bytes reach `local_budget_bytes` (and a `remote_uri` is configured). `read`
    streams a bucket back from whichever tier holds it. `cleanup` removes the local
    files this store created — only those, so a shared scratch dir is safe.
    """

    def __init__(
        self,
        local_dir: str,
        *,
        remote_uri: str | None = None,
        local_budget_bytes: int | None = None,
        compression: str | None = "lz4",
    ) -> None:
        self._local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)
        self._remote_uri = remote_uri.rstrip("/") if remote_uri else None
        # Overflow to the remote tier tracks the disk that actually exists, not a static
        # guess — so a smaller-than-configured scratch volume overflows before it fills
        # rather than failing the query on a full filesystem.
        self._local_budget = _clamp_to_free_disk(local_dir, local_budget_bytes)
        self._compression = compression
        self._local_used = 0
        # Bytes streamed to the local tier by writers that are still OPEN. The tier decision
        # is made once, on a bucket's first write, from the local budget — but `_local_used`
        # only grows when a bucket CLOSES, so with several buckets streaming at once (the
        # partition phase's pattern) none of them ever observes the others' growth and the
        # remote overflow tier never engages, letting the local disk fill past its budget.
        # Tracking in-flight bytes here lets a later-opened bucket overflow correctly.
        self._local_pending = 0
        self._local_paths: list[str] = []

    @property
    def local_bytes(self) -> int:
        """Total bytes currently held on the local tier."""
        return self._local_used

    def writer(self, name: str) -> _BucketWriter:
        """A streaming writer for bucket `name` (tier chosen on first batch)."""
        return _BucketWriter(self, name)

    def _on_closed(self, tier: SpillTier | None, path: str | None) -> int:
        """Record a finished bucket's bytes; return its size."""
        if tier is SpillTier.LOCAL and path is not None:
            nbytes = os.path.getsize(path)
            self._local_used += nbytes
            self._local_paths.append(path)
            return nbytes
        if path is not None:
            with _fsspec_open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                return fh.tell()
        return 0

    def spill(self, batches: list[pa.RecordBatch], name: str = "partition") -> SpillHandle | None:
        """Convenience: stream a whole batch list into one bucket and close it.

        Returns the handle, or `None` for an empty/all-empty partition (an empty
        bucket is intrinsic to a shuffle, not an error). Prefer `writer()` when the
        batches are produced incrementally so they never co-reside in memory.
        """
        w = self.writer(name)
        for batch in batches:
            w.write(batch)
        return w.close()

    def read(self, handle: SpillHandle) -> list[pa.RecordBatch]:
        """Stream the partition referenced by `handle` back from its tier."""
        if handle.tier is SpillTier.LOCAL:
            with _open_local_map(handle.path) as mm:
                return pa.ipc.open_stream(mm).read_all().to_batches()
        with _fsspec_open(handle.path, "rb") as fh:
            reader = pa.ipc.open_stream(fh)
            return reader.read_all().to_batches()

    def read_stream(self, handle: SpillHandle):
        """Yield the partition's batches one at a time (never materializing it whole).

        The reader that grace recursion uses to re-partition an over-large bucket
        without first loading the entire bucket into memory.
        """
        if handle.tier is SpillTier.LOCAL:
            with _open_local_map(handle.path) as mm:
                yield from pa.ipc.open_stream(mm)
        else:
            with _fsspec_open(handle.path, "rb") as fh:
                yield from pa.ipc.open_stream(fh)

    def cleanup(self) -> None:
        """Remove the local files this store created and reset local accounting."""
        import contextlib

        for path in self._local_paths:
            with contextlib.suppress(OSError):
                os.remove(path)
        self._local_paths.clear()
        self._local_used = 0
        self._local_pending = 0
