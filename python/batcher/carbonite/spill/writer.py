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
from batcher.plan.types import logical_bytes

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
        "_seen_dictionaries",
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
        # Buffer addresses of dictionaries already charged to this bucket. Batches of a
        # dictionary-encoded column all point at the *same* values array, but `nbytes`
        # includes that whole dictionary in every one — see `_charged_bytes`.
        self._seen_dictionaries: set[int] = set()
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
            # out. With a remote tier configured the bucket can be carried over to it
            # instead of failing the query (see `_failover_to_remote`); without one — or if
            # the carry-over cannot account for every row — the honest move is to fail with
            # the fix in the message.
            if not disk.is_out_of_space(exc):
                raise
            if not self._failover_to_remote(batch.schema):
                self._raise_out_of_space(exc)
            self._writer.write_batch(batch)
        self._num_rows += batch.num_rows
        charged = self._charged_bytes(batch)
        if self._tier is SpillTier.LOCAL:
            # Charge the batch's (uncompressed, in-memory) size to the store's live local
            # usage as it lands, not just at close. A slight over-estimate vs the compressed
            # on-disk size only makes overflow trigger a touch early — safe (never over-fills
            # the disk), and result-invariant (the remote tier reads back identically).
            self._pending_bytes += charged
            self._store._local_pending += charged
        else:
            self._pending_bytes += charged

    def _charged_bytes(self, batch: pa.RecordBatch) -> int:
        """`batch.nbytes`, charging each shared dictionary only the first time it appears.

        Batches of a dictionary-encoded column all point at the *same* values array, but
        `nbytes` includes that whole dictionary in every one. Measured: a 20,000-entry string
        dictionary reports 256 KB per batch for 16 KB of indices — so a bucket of 100 such
        batches is charged 25 MB for 1.8 MB of content, a ~14x over-count.

        Three decisions run on this figure, and all three go wrong the same way. The local
        budget overflows to object storage far earlier than it needs to; `read_reserved`
        reserves ~14x the memory the read actually takes, squeezing every concurrent query;
        and the bucket looks over `spill_bucket_max_bytes`, so the re-split recursion fires
        and pays a full extra write and read to split a bucket that already fitted. None of
        them is a wrong answer, which is why an over-count here survives — it only ever
        shows up as the engine being slow.

        The count matches what reading the bucket back costs, which is what `logical_nbytes`
        promises: the IPC stream carries the dictionary once, and the reader reconstructs one
        values array shared by every batch.
        """
        total = logical_bytes(batch)
        for column in batch.columns:
            total -= _recounted_dictionary_bytes(column, self._seen_dictionaries)
        return max(total, 0)

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

    def _failover_to_remote(self, schema: pa.Schema) -> bool:
        """Carry a bucket whose local volume filled mid-stream over to the remote tier.

        `_open` picks a tier per bucket by re-measuring free disk, which handles a volume
        that is *already* low. It cannot handle one that fills **during** a bucket, and that
        is the case at scale: a bucket is a whole partition, so a 16-way spill of a terabyte
        writes ~60 GB per bucket, and the check at open is a single sample taken before any
        of it was written. `memory.spill_remote_uri` is documented to keep an out-of-core
        query alive when local disk fills, and until now it only did so at bucket boundaries.

        Recovery is possible because the batches already streamed are **on disk**, not
        discarded — they are read back and re-written to the remote tier **one at a time**,
        so the carry-over is bounded by a single batch and does not undo the spilling it is
        rescuing. The local stream is abandoned rather than finished: writing its
        end-of-stream marker would need the disk that just refused.

        Returns `False` — leaving the caller to fail with the actionable error — when there
        is no remote tier, or when the rows recovered do not match the rows this writer
        knows it wrote. That second condition is the important one: a partial carry-over
        would turn a loud out-of-space failure into a silently short bucket, which is
        strictly worse than the failure it replaces.

        Args:
            schema: The bucket's schema, for the remote stream.

        Returns:
            Whether the bucket now has a live remote writer ready for the failed batch.
        """
        store = self._store
        if self._tier is not SpillTier.LOCAL or store._remote_uri is None:
            return False
        local_path = self._path
        expected_rows = self._num_rows

        # Abandon, do not finish: `close()` would flush and write the EOS marker, which
        # needs the disk that just refused.
        for closeable in (self._writer, self._fh):
            if closeable is not None:
                with contextlib.suppress(Exception):
                    closeable.close()
        self._writer = None
        self._fh = None
        # Hand the local byte charge back — the file is about to be deleted — while keeping
        # `_pending_bytes` as this bucket's running *logical* size, which `close()` reports
        # as `logical_nbytes`. Switching the tier below also stops `_release_pending` from
        # subtracting the same bytes a second time.
        store._local_pending -= self._pending_bytes

        self._open_remote(store, schema)
        recovered = self._copy_stream_to_current(local_path)
        if recovered != expected_rows:
            # Cannot prove the whole bucket came across. Undo the half-made remote bucket and
            # let the caller raise, rather than continue with a bucket that is short.
            for closeable in (self._writer, self._fh):
                if closeable is not None:
                    with contextlib.suppress(Exception):
                        closeable.close()
            self._writer = None
            self._fh = None
            store._remove_partial(SpillTier.REMOTE, self._path)
            self._tier, self._path = SpillTier.LOCAL, local_path
            store._local_pending += self._pending_bytes
            return False

        with contextlib.suppress(OSError):
            os.unlink(local_path)
        return True

    def _copy_stream_to_current(self, path: str) -> int:
        """Re-write every complete batch in the local IPC stream at `path` into the writer
        now open, one batch at a time, and return the rows carried over.

        The file may end mid-message, because the write that filled the disk was partway
        through one. Everything before that point is intact, so the read is allowed to stop
        there — but only the *read* is: a failure of the re-write propagates, because that is
        the remote tier refusing and not something to absorb.
        """
        rows = 0
        with open(path, "rb") as fh:
            batches = iter(pa.ipc.open_stream(fh))
            while True:
                try:
                    batch = next(batches)
                except StopIteration:
                    break
                except Exception:
                    # The partial tail message the failed write left behind.
                    break
                self._writer.write_batch(batch)
                rows += batch.num_rows
        return rows

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


def _dictionary_id(values: pa.Array) -> int | None:
    """An identity for a dictionary's values array: the address of its first real buffer.

    Unambiguous while the arrays being measured are alive, which they are — the caller is
    charging batches it currently holds. `None` when there is no buffer to identify it by, in
    which case the caller charges it in full rather than guessing.
    """
    for buf in values.buffers():
        if buf is not None:
            return buf.address
    return None


def _recounted_dictionary_bytes(array: pa.Array, seen: set[int]) -> int:
    """Bytes of `array` that `nbytes` counted for a dictionary already charged.

    On a dictionary's first sighting its values are kept and the walk descends *into* them,
    so a dictionary nested inside another's values is handled too. On a later sighting the
    whole values array is subtracted — exactly what `nbytes` added for it.

    The walk covers structs and lists, because a dictionary column in a semi-structured or
    multimodal schema is usually one level down rather than at the top.
    """
    if isinstance(array, pa.DictionaryArray):
        values = array.dictionary
        ident = _dictionary_id(values)
        if ident is None:
            return 0
        if ident in seen:
            return logical_bytes(values)
        seen.add(ident)
        return _recounted_dictionary_bytes(values, seen)
    if isinstance(array, pa.StructArray):
        return sum(
            _recounted_dictionary_bytes(array.field(i), seen) for i in range(array.type.num_fields)
        )
    if isinstance(array, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
        return _recounted_dictionary_bytes(array.values, seen)
    return 0
