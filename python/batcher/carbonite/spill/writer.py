"""One spill bucket, streamed to whichever tier its first batch can afford.

A bucket writer never holds the partition whole: batches go straight to an Arrow IPC
stream as they are produced, which is the property that makes spilling an answer to
running out of memory rather than another way to run out of it.

The lifecycle has three exits, and all three matter:

- `close()` finalizes the bucket and returns its `SpillHandle`. It is **idempotent** —
  a second call returns the same handle instead of double-charging the store's local
  accounting and re-appending the path to the cleanup list.
- `abort()` gives up on a bucket, releasing its in-flight byte charge and removing the
  partial file. Without it, a writer abandoned on an exception left its bytes charged to
  the store's live local budget *forever*, so every later bucket in the query saw a full
  local tier and overflowed to object storage.
- The context-manager form does the right one automatically: `close` on success, `abort`
  on an exception in the block.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import ResourceError
from batcher._internal.paths import open_private
from batcher.carbonite.spill import disk
from batcher.carbonite.spill.handle import SpillHandle, SpillTier

if TYPE_CHECKING:
    from types import TracebackType

    from batcher.carbonite.spill.store import TieredSpillStore

__all__ = ["BucketWriter"]


class BucketWriter:
    """Streams batches for one spill bucket to its tier, chosen at first write.

    The tier is decided lazily on the first batch (so an empty bucket opens no file
    and costs nothing): REMOTE when a remote URI is configured and the local budget
    is already exhausted, else LOCAL. Batches stream straight to the IPC writer — the
    partition is never held whole in memory.
    """

    __slots__ = (
        "_fh",
        "_handle",
        "_name",
        "_num_rows",
        "_path",
        "_pending_bytes",
        "_store",
        "_tier",
        "_writer",
    )

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
        self._num_rows = 0
        # The finalized handle, so `close()` is idempotent. A double close previously
        # re-entered `writer.close()` on a closed stream and, worse, charged the bucket's
        # bytes to the store's confirmed usage twice while appending its path to the
        # cleanup list twice.
        self._handle: SpillHandle | None = None

    @property
    def num_rows(self) -> int:
        """Rows streamed into this bucket so far."""
        return self._num_rows

    @property
    def tier(self) -> SpillTier | None:
        """The tier this bucket landed on, or `None` before its first batch."""
        return self._tier

    @property
    def path(self) -> str | None:
        """This bucket's path, or `None` before its first batch."""
        return self._path

    def write(self, batch: pa.RecordBatch) -> None:
        """Stream one batch into the bucket, opening its tier on the first call.

        Args:
            batch: The batch to append. An empty batch is skipped, so a bucket that
                only ever receives empty batches opens no file at all.

        Raises:
            ResourceError: If the local scratch volume is full or over quota.
        """
        if batch.num_rows == 0:
            return
        if self._writer is None:
            self._open(batch.schema)
        try:
            self._writer.write_batch(batch)
        except OSError as exc:
            # A full scratch disk otherwise surfaces as a bare `OSError: [Errno 28]` from
            # deep inside the Arrow writer, which names neither the spill tier nor the way
            # out. It cannot be recovered in place — the batches already streamed to this
            # bucket are not retained, so failing over to the remote tier mid-stream would
            # silently drop them — so the honest move is to fail with the fix in the message.
            if disk.is_out_of_space(exc):
                self._raise_out_of_space(exc)
            raise
        self._num_rows += batch.num_rows
        if self._tier is SpillTier.LOCAL:
            # Charge the batch's (uncompressed, in-memory) size to the store's live local
            # usage as it lands, not just at close. A slight over-estimate vs the compressed
            # on-disk size only makes overflow trigger a touch early — safe (never over-fills
            # the disk), and result-invariant (the remote tier reads back identically).
            self._pending_bytes += batch.nbytes
            self._store._local_pending += batch.nbytes
        else:
            self._pending_bytes += batch.nbytes

    def _raise_out_of_space(self, exc: OSError) -> None:
        """Turn a space/quota refusal into an error that names the tier and the fix."""
        if self._tier is SpillTier.REMOTE:
            raise ResourceError(
                f"the remote spill tier rejected a write to {self._path!r} for lack of "
                "space or quota. Raise the bucket/prefix quota, or point "
                "memory.spill_remote_uri at a location with room."
            ) from exc
        raise ResourceError(
            f"the local spill disk is full while writing {self._path!r}. Free space "
            f"on the scratch volume, point memory.spill_dir at a larger one, or set "
            f"memory.spill_remote_uri so buckets overflow to object storage instead."
        ) from exc

    def _open(self, schema: pa.Schema) -> None:
        store = self._store
        # Two independent reasons to overflow, and the second is why the first is not enough:
        # the budget accounts only for bytes *this store* wrote, so a volume filled by anyone
        # else (a co-tenant, another query's scratch, a growing log) stays invisible to it and
        # the local tier keeps writing until the filesystem returns ENOSPC — a hard query
        # failure. The clamp at construction is a single sample taken before any of that
        # happened. Re-measuring per bucket is the only way overflow can track a disk that
        # fills *during* the query; a bucket's tier is fixed at open, so this is also the last
        # point where the choice is still free (mid-stream there are already-written batches
        # that would have to be rewritten).
        over_budget = (
            store._local_budget is not None
            and store._local_used + store._local_pending >= store._local_budget
        )
        overflow = store._remote_uri is not None and (over_budget or store._local_disk_low())
        if overflow:
            self._open_remote(store, schema)
        else:
            self._open_local(store, schema)

    def _open_remote(self, store: TieredSpillStore, schema: pa.Schema) -> None:
        self._tier = SpillTier.REMOTE
        self._path = f"{store._remote_uri}/{self._name}.arrow"
        self._fh = disk.fsspec_open(self._path, "wb").open()
        # The remote tier is slow object storage that charges for bytes transferred, so
        # LZ4 always pays there even when the local NVMe tier stays uncompressed.
        self._writer = pa.ipc.new_stream(
            self._fh, schema, options=disk.remote_ipc_options(store._compression)
        )
        store._overflowed += 1

    def _open_local(self, store: TieredSpillStore, schema: pa.Schema) -> None:
        self._tier = SpillTier.LOCAL
        self._path = os.path.join(store._local_dir, f"{self._name}.arrow")
        # 0600 at `open`, not by a later chmod: a chmod leaves a window in which the
        # rows are world-readable, and the private directory above is best-effort.
        self._fh = pa.PythonFile(open_private(self._path), mode="w")
        self._writer = pa.ipc.new_stream(
            self._fh, schema, options=disk.ipc_options(store._compression)
        )

    def close(self) -> SpillHandle | None:
        """Finalize the bucket. Returns its handle, or `None` if it got no rows.

        Idempotent: calling it again returns the same handle without re-charging the
        store's accounting.

        Returns:
            The `SpillHandle` for the finalized bucket, or `None` for an empty one.
        """
        if self._handle is not None:
            return self._handle
        if self._writer is None:
            self._store._open_writers.discard(self)
            return None
        try:
            self._writer.close()
            # The byte offset the IPC stream ended at *is* the file size, and reading it here
            # costs nothing. The alternative for the remote tier was a second `fsspec` open
            # plus a seek-to-end per bucket — a full object-storage round trip, per bucket,
            # purely to learn a number the handle already had.
            written = self._file_position()
            self._fh.close()
        except BaseException:
            # A finalize that fails — the flush that finally hits a full disk, a remote
            # upload that is refused — left the writer's bytes charged against the store's
            # live local budget forever and its partial file on disk. That is exactly the
            # "permanent, silent throughput cliff" `abort()` exists to prevent, and `close()`
            # was the one exit that could not reach it.
            self.abort()
            raise
        self._writer = None
        self._store._open_writers.discard(self)
        # The uncompressed (in-memory) size of everything written — captured before the LOCAL
        # branch zeroes the pending estimate. The reducer budgets against this, not the
        # compressed on-disk `nbytes` below (which for a compressible bucket can be far smaller
        # and would let an over-large bucket skip re-spill recursion and OOM the finalize).
        logical_nbytes = self._pending_bytes
        self._release_pending()
        nbytes = self._store._on_closed(self._tier, self._path, written)
        self._handle = SpillHandle(self._tier, self._path, nbytes, logical_nbytes, self._num_rows)
        return self._handle

    def abort(self) -> None:
        """Give up on this bucket: release its byte charge and remove the partial file.

        The failure path `close()` cannot serve. A writer abandoned mid-stream (an
        exception in the operator producing the batches, a cancelled query) otherwise
        leaves `_pending_bytes` charged against the store's live local budget for the rest
        of the process's life, so every subsequent bucket reads the local tier as full and
        overflows to object storage — a permanent, silent throughput cliff triggered by one
        unrelated error. Idempotent, and never raises: it runs on the way out of a failure.
        """
        if self._handle is not None:
            return
        self._release_pending()
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
            self._writer = None
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()
            self._fh = None
        if self._path is not None:
            self._store._remove_partial(self._tier, self._path)
        self._num_rows = 0
        self._store._open_writers.discard(self)

    def _release_pending(self) -> None:
        """Hand this writer's in-flight local byte charge back to the store."""
        if self._tier is SpillTier.LOCAL and self._pending_bytes:
            # This bucket is finalized (or abandoned): hand its bytes from the live
            # "pending" estimate back, so the confirmed `_local_used` accounting added by
            # `_on_closed` is never double-counted with it.
            self._store._local_pending -= self._pending_bytes
        self._pending_bytes = 0

    def _file_position(self) -> int | None:
        """Bytes written so far, from the handle's own offset, or `None` if it won't say."""
        try:
            return int(self._fh.tell())
        except Exception:  # pragma: no cover - a handle that cannot report its offset
            return None

    def __enter__(self) -> BucketWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
