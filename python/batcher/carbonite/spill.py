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
from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from batcher._internal.errors import IOError as BatcherIOError

__all__ = ["SpillHandle", "SpillTier", "TieredSpillStore"]


class SpillTier(Enum):
    """Which storage tier a spilled partition lives on."""

    LOCAL = "local"  # Arrow IPC on local disk (NVMe)
    REMOTE = "remote"  # object storage via fsspec


@dataclass(frozen=True, slots=True)
class SpillHandle:
    """An opaque reference to one spilled partition (tier + path + size)."""

    tier: SpillTier
    path: str
    nbytes: int


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
    and footprint. Degrades silently to uncompressed if the codec isn't built into
    this pyarrow, so spilling never fails on a missing optional codec.
    """
    if not compression:
        return None
    try:
        return pa.ipc.IpcWriteOptions(compression=compression)
    except (ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return None


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

    def write(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        if self._writer is None:
            self._open(batch.schema)
        self._writer.write_batch(batch)

    def _open(self, schema: pa.Schema) -> None:
        store = self._store
        overflow = (
            store._remote_uri is not None
            and store._local_budget is not None
            and store._local_used >= store._local_budget
        )
        opts = _ipc_options(store._compression)
        if overflow:
            self._tier = SpillTier.REMOTE
            self._path = f"{store._remote_uri}/{self._name}.arrow"
            self._fh = _fsspec_open(self._path, "wb").open()
            self._writer = pa.ipc.new_stream(self._fh, schema, options=opts)
        else:
            self._tier = SpillTier.LOCAL
            self._path = os.path.join(store._local_dir, f"{self._name}.arrow")
            self._fh = pa.OSFile(self._path, "wb")
            self._writer = pa.ipc.new_stream(self._fh, schema, options=opts)

    def close(self) -> SpillHandle | None:
        """Finalize the bucket. Returns its handle, or `None` if it got no rows."""
        if self._writer is None:
            return None
        self._writer.close()
        self._fh.close()
        nbytes = self._store._on_closed(self._tier, self._path)
        return SpillHandle(self._tier, self._path, nbytes)


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
        self._local_budget = local_budget_bytes
        self._compression = compression
        self._local_used = 0
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
            with pa.memory_map(handle.path, "r") as mm:
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
            with pa.memory_map(handle.path, "r") as mm:
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
