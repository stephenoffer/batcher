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

Cleanup removes **both** tiers. A remote bucket is scratch too: it is written under a
query-scoped prefix and is dead the moment the query ends, so leaving it behind bills
the operator monthly for bytes nothing will ever read again.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import PlanError, ResourceError
from batcher._internal.paths import private_dir
from batcher.carbonite.spill import disk
from batcher.carbonite.spill.handle import SpillHandle, SpillTier
from batcher.carbonite.spill.writer import BucketWriter

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import TracebackType

__all__ = ["TieredSpillStore"]

# A bucket name becomes a path component, so it must not be able to reach outside the
# scratch directory. Every real caller passes something like `part-17`; anything with a
# separator, a `..`, or a NUL is a bug in the caller (or, in a distributed setting, an
# attacker-influenced identifier) and is rejected rather than quietly writing through it.
_SAFE_BUCKET_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._@=+-]*\Z")


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


class TieredSpillStore:
    """Streams `RecordBatch` buckets to local disk, overflowing to object storage.

    Open a streaming `writer(name)` per bucket and `write` batches to it as they are
    produced (never materializing the whole partition); `close()` returns a
    `SpillHandle`. New buckets overflow to the remote tier once the cumulative local
    bytes reach `local_budget_bytes` (and a `remote_uri` is configured). `read`
    streams a bucket back from whichever tier holds it. `cleanup` removes the files
    this store created — only those, so a shared scratch dir is safe.

    Usable as a context manager, which calls `cleanup` on the way out so a query that
    raises does not leave its scratch behind.
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
        # Owner-only. A spill file holds the query's actual rows on a shared scratch path
        # such as /tmp, so the default 0755/0644 makes them readable by every local user.
        private_dir(local_dir)
        self._remote_uri = remote_uri.rstrip("/") if remote_uri else None
        # Overflow to the remote tier tracks the disk that actually exists, not a static
        # guess — so a smaller-than-configured scratch volume overflows before it fills
        # rather than failing the query on a full filesystem.
        self._local_budget = disk.clamp_to_free_disk(local_dir, local_budget_bytes)
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
        # Remote buckets this store wrote. Tracked for the same reason as the local ones:
        # they are query scratch, and an untracked one is a bucket nobody ever deletes.
        self._remote_paths: list[str] = []
        self._remote_used = 0
        # How many buckets were pushed to the remote tier. The single number that explains
        # a spill phase that suddenly got slow: it means the local tier filled.
        self._overflowed = 0
        # Lifetime totals, which the live `_local_used`/`_remote_used` cannot supply: a
        # bucket is dropped as soon as it has been read back, so by the end of a reduce phase
        # a store that moved gigabytes reports zero bytes held. Without these, "how much did
        # this query spill" had no answer at the only moment anyone asks it.
        self._bytes_written = 0
        self._buckets_written = 0
        self._peak_local_used = 0
        self._peak_remote_used = 0
        # Writers handed out and not yet finalized. `cleanup` aborts whatever is left, which
        # is the only way an abandoned *remote* bucket ever gets deleted.
        self._open_writers: set[BucketWriter] = set()

    def _local_disk_low(self) -> bool:
        """Whether the scratch volume is too full to start another local bucket.

        Measured, not accounted: this is what catches a disk filled by something other
        than this store. An unstat-able volume reads as "not low", so the budget alone
        decides and behavior is exactly as before.

        The threshold is `ELEVATED`, not `FULL`. Waiting for the reserve floor means the
        *first* bucket to notice is one that already has nowhere to go, and a bucket's tier
        is fixed at open — so by the time the floor is crossed there can be several buckets
        already streaming to a volume that cannot hold them, and an out-of-space write
        cannot be recovered in place. Routing earlier costs those buckets object-storage
        latency and costs nothing else, since the remote tier reads back identically.
        """
        return self.disk_pressure() >= disk.DiskPressure.ELEVATED

    def disk_pressure(self) -> disk.DiskPressure:
        """How full this store's scratch volume is (`NORMAL`/`ELEVATED`/`FULL`).

        Returns:
            The measured level. `NORMAL` for a volume that cannot be stat'd — a
            measurement that could not be taken is not evidence of a problem.
        """
        return disk.disk_pressure(self._local_dir)

    @property
    def local_bytes(self) -> int:
        """Total bytes currently held on the local tier."""
        return self._local_used

    @property
    def remote_bytes(self) -> int:
        """Total bytes this store has written to the remote overflow tier."""
        return self._remote_used

    @property
    def total_bytes(self) -> int:
        """Bytes this store holds across both tiers — its whole out-of-core footprint."""
        return self._local_used + self._remote_used

    @property
    def bucket_count(self) -> int:
        """Finalized buckets this store is holding across both tiers."""
        return len(self._local_paths) + len(self._remote_paths)

    @property
    def overflowed(self) -> int:
        """Buckets that went to the remote tier because the local one was exhausted.

        Non-zero means the spill phase is paying object-storage latency, which is the
        usual explanation for an out-of-core query that got dramatically slower without
        any change to the query itself.
        """
        return self._overflowed

    def stats(self) -> dict[str, int | str]:
        """A snapshot of this store's accounting, for telemetry and tests.

        Returns:
            Bytes and bucket counts *held* per tier and their high-water marks, the lifetime
            volume written, how many buckets overflowed, the local budget in force (`-1` when
            the local tier is unbounded), and the volume's measured pressure and free space
            (`-1` when it cannot be stat'd).

            The held figures and the lifetime ones answer different questions and both are
            needed. A reduce phase drops each bucket as it reads it back, so a store that
            moved gigabytes reports `local_bytes` of zero by the time anyone asks — while
            `bytes_written` is the volume that went to disk and `peak_local_bytes` is the
            most that was ever resident on it at once.
        """
        free = disk.free_disk_bytes(self._local_dir)
        return {
            "local_bytes": self._local_used,
            "remote_bytes": self._remote_used,
            "local_buckets": len(self._local_paths),
            "remote_buckets": len(self._remote_paths),
            "peak_local_bytes": self._peak_local_used,
            "peak_remote_bytes": self._peak_remote_used,
            "bytes_written": self._bytes_written,
            "buckets_written": self._buckets_written,
            "overflowed": self._overflowed,
            "local_budget_bytes": -1 if self._local_budget is None else self._local_budget,
            "local_pending_bytes": self._local_pending,
            "disk_pressure": self.disk_pressure().name,
            "free_disk_bytes": -1 if free is None else free,
        }

    def writer(self, name: str) -> BucketWriter:
        """A streaming writer for bucket `name` (tier chosen on first batch).

        Args:
            name: The bucket's identifier. It becomes a path component, so it must be a
                plain token — letters, digits, and ``. _ - + = @`` — with no separators.

        Returns:
            The open `BucketWriter`.

        Raises:
            PlanError: If `name` is empty or is not a safe path component.
        """
        if not _SAFE_BUCKET_NAME.match(name):
            raise PlanError(
                f"spill bucket name {name!r} is not a safe path component",
                hint="use letters, digits, and `. _ - + = @` only — no separators or `..`.",
            )
        writer = BucketWriter(self, name)
        self._open_writers.add(writer)
        return writer

    def _on_closed(self, tier: SpillTier | None, path: str | None, written: int | None) -> int:
        """Record a finished bucket's bytes; return its size.

        `written` is the byte offset the IPC stream ended at, which *is* the file size and
        which the writer already knew. Using it avoids a `stat` per local bucket and, on
        the remote tier, a whole extra object-storage open-and-seek per bucket. It falls
        back to asking the filesystem when a handle would not report its offset.
        """
        if path is None or tier is None:
            return 0
        nbytes = written if written is not None and written >= 0 else self._measure(tier, path)
        if tier is SpillTier.LOCAL:
            self._local_used += nbytes
            self._local_paths.append(path)
            self._peak_local_used = max(self._peak_local_used, self._local_used)
        else:
            self._remote_used += nbytes
            self._remote_paths.append(path)
            self._peak_remote_used = max(self._peak_remote_used, self._remote_used)
        self._bytes_written += nbytes
        self._buckets_written += 1
        return nbytes

    @staticmethod
    def _measure(tier: SpillTier, path: str) -> int:
        """The on-disk size of a finalized bucket, asked of the filesystem."""
        if tier is SpillTier.LOCAL:
            return os.path.getsize(path)
        with disk.fsspec_open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            return fh.tell()

    def _remove_partial(self, tier: SpillTier | None, path: str) -> None:
        """Delete a bucket that was abandoned mid-stream. Best-effort, never raises."""
        with contextlib.suppress(Exception):
            if tier is SpillTier.LOCAL:
                os.remove(path)
            elif tier is SpillTier.REMOTE:
                self._remove_remote(path)

    @staticmethod
    def _remove_remote(path: str) -> None:
        """Delete one remote bucket through `fsspec`."""
        import fsspec

        fs, _, paths = fsspec.get_fs_token_paths(path)
        fs.rm(paths[0])

    def spill(
        self, batches: Iterable[pa.RecordBatch], name: str = "partition"
    ) -> SpillHandle | None:
        """Convenience: stream a whole batch list into one bucket and close it.

        Returns the handle, or `None` for an empty/all-empty partition (an empty
        bucket is intrinsic to a shuffle, not an error). Prefer `writer()` when the
        batches are produced incrementally so they never co-reside in memory.

        Args:
            batches: The batches to write, consumed in order.
            name: The bucket name (see `writer`).

        Returns:
            The bucket's handle, or `None` if nothing was written.
        """
        # The context-manager form matters here: if `batches` is a generator that raises
        # part-way (an operator failing mid-partition), the writer is aborted rather than
        # abandoned, so its in-flight byte charge does not strand the local budget.
        with self.writer(name) as w:
            for batch in batches:
                w.write(batch)
            return w.close()

    def read(self, handle: SpillHandle) -> list[pa.RecordBatch]:
        """Stream the partition referenced by `handle` back from its tier.

        Args:
            handle: The bucket to read.

        Returns:
            Its batches, materialized.
        """
        return list(self.read_stream(handle))

    @contextlib.contextmanager
    def read_reserved(self, handle: SpillHandle) -> Iterator[Iterator[pa.RecordBatch]]:
        """Stream a bucket back with its resident footprint reserved against the pool.

        Reading a bucket puts it *back* in memory, which is the one step of spilling that
        can undo it: the state was written out because it did not fit, and nothing checked
        the budget before pulling it in again. The figure to reserve is the handle's
        `logical_nbytes` — the uncompressed size — because the on-disk size of a
        compressible bucket can be many times smaller than what it occupies once read, so
        budgeting against the file size under-reserves by exactly the compression ratio.

        This is deliberately advisory rather than blocking: it accounts the read so
        concurrent readers and the running query see one envelope, and a reservation that
        does not fit still proceeds (the caller is already out of core, and refusing here
        would strand the query with no way to make progress). What it buys is that the
        *next* decision — another reader, an operator sizing its state — sees the memory
        this read is holding instead of allocating on top of it blindly.

        Args:
            handle: The bucket to read.

        Yields:
            The bucket's batches, one at a time.
        """
        from batcher.carbonite.memory.pool import current_process_pool

        pool = current_process_pool()
        if pool is None:
            yield self.read_stream(handle)
            return
        with pool.reserve(handle.logical_nbytes or handle.nbytes):
            yield self.read_stream(handle)

    def read_stream(self, handle: SpillHandle) -> Iterator[pa.RecordBatch]:
        """Yield the partition's batches one at a time (never materializing it whole).

        The reader that grace recursion uses to re-partition an over-large bucket
        without first loading the entire bucket into memory.

        Every batch is counted and the total checked against the handle's `num_rows` when the
        stream ends. That check is not paranoia about I/O in general: an Arrow IPC **stream**
        truncated at a message boundary -- the last complete batch present, the end-of-stream
        marker gone -- is byte-for-byte a shorter *valid* stream, so `open_stream` returns the
        batches it finds and reports success. Measured: five batches of 1,000 rows cut after
        the third read back as 3,000 rows with no error. The reducer then computes a correct
        answer over the wrong rows, and nothing anywhere records that it happened. The remote
        tier makes it likelier still, since a partially-written object is an ordinary outcome
        of an interrupted upload.

        `SpillHandle.num_rows` has always carried the count for exactly this ("a caller can
        detect a truncated bucket"), and no caller did.

        Args:
            handle: The bucket to read.

        Yields:
            Each `RecordBatch` in write order.

        Raises:
            ResourceError: If the bucket reads back short of the rows written to it.
        """
        seen = 0
        if handle.tier is SpillTier.LOCAL:
            with _open_local_map(handle.path) as mm:
                for batch in pa.ipc.open_stream(mm):
                    seen += batch.num_rows
                    yield batch
        else:
            with disk.fsspec_open(handle.path, "rb") as fh:
                for batch in pa.ipc.open_stream(fh):
                    seen += batch.num_rows
                    yield batch
        _verify_complete(handle, seen)

    def release(self, handle: SpillHandle) -> None:
        """Delete one bucket now that its reader is done with it.

        The incremental counterpart to `cleanup`: a reduce that consumes buckets one at a
        time can give each one's disk back as it finishes instead of holding the whole
        spilled state until teardown, which is what keeps peak *disk* bounded the way the
        credit window keeps peak memory bounded.

        Args:
            handle: The bucket to delete. Unknown or already-released handles are ignored.
        """
        paths = self._local_paths if handle.tier is SpillTier.LOCAL else self._remote_paths
        if handle.path not in paths:
            return
        paths.remove(handle.path)
        if handle.tier is SpillTier.LOCAL:
            self._local_used = max(0, self._local_used - handle.nbytes)
        else:
            self._remote_used = max(0, self._remote_used - handle.nbytes)
        self._remove_partial(handle.tier, handle.path)

    def cleanup(self) -> None:
        """Remove the files this store created on *both* tiers and reset accounting.

        Writers still open are aborted first. A bucket abandoned mid-stream — an exception
        in the operator producing its batches, a cancelled query — has no closed path, so it
        is in neither tier's path list and `cleanup` could not see it. Locally that partial
        file was swept up by the caller's `rmtree`; on the **remote** tier nothing removed
        it at all, so every failed out-of-core query that had overflowed left orphaned
        objects in the bucket, accumulating and billable, with no record that they existed.
        """
        for writer in list(self._open_writers):
            with contextlib.suppress(Exception):
                writer.abort()
        self._open_writers.clear()
        for path in self._local_paths:
            with contextlib.suppress(OSError):
                os.remove(path)
        self._local_paths.clear()
        for path in self._remote_paths:
            with contextlib.suppress(Exception):
                self._remove_remote(path)
        self._remote_paths.clear()
        self._local_used = 0
        self._local_pending = 0
        self._remote_used = 0

    def __enter__(self) -> TieredSpillStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Read *before* cleanup, which zeroes the accounting: the interesting figures are
        # what the store held at its high-water, and after the buckets are removed there is
        # nothing left to report. An out-of-core query that is otherwise invisible in the
        # counters — it returns the right answer, slowly — shows up here as tier volumes and
        # an overflow count.
        self.publish_stats()
        self.cleanup()

    def publish_stats(self) -> None:
        """Put this store's accounting on the event bus as a `RESOURCE` reading.

        A no-op when nothing is listening, which is the default. `free_disk_bytes` costs a
        `statvfs`, so a process exporting no metrics must not pay for it.

        Returns:
            None.
        """
        from batcher._internal import events

        if not events.listening():
            return
        events.publish(events.RESOURCE, name="spill", stats=self.stats())


def _verify_complete(handle: SpillHandle, seen: int) -> None:
    """Fail if a bucket read back fewer rows than were written to it.

    Only a *short* read is an error. Reading more cannot come from truncation and would mean
    the recorded count is itself wrong, which is not worth failing a query over. A handle
    written before the count existed reports `0`, which this correctly treats as "nothing to
    check" rather than as an empty bucket.
    """
    if not handle.num_rows or seen >= handle.num_rows:
        return
    from batcher._internal.errors import ResourceError

    raise ResourceError(
        f"spilled bucket {handle.path} read back {seen} rows but {handle.num_rows} were "
        f"written to it, so it is truncated or was modified underneath the query. This "
        f"would otherwise have silently dropped {handle.num_rows - seen} rows from the "
        f"result. Check for a full or evicted spill volume, or an interrupted upload to "
        f"memory.spill_remote_uri."
    )
