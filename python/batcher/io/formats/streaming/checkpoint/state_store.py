"""Durable snapshots of a streaming query's running aggregation state.

The running state in `core.streaming._AggFold` is one Arrow ``RecordBatch`` (the
output of the native ``combine``), so it serializes with ``pyarrow.ipc`` exactly
like the ML shard writer — no FFI addition needed. Snapshots are written atomically
per micro-batch and reloaded on recovery to resume a stateful query without
recomputing from the start of the stream.

Local and remote are written differently, and `location.py` says why: a local write is
fsynced and renamed and the directory fsynced, because the engine snapshots state and
*then* records the commit; on an object store a PUT is durable when it returns and there is
no rename to make durable separately. Reading, listing, and deleting are the same code on
both, because none of them has a durability ordering to preserve.
"""

from __future__ import annotations

import contextlib
import os

import pyarrow as pa
from pyarrow import ipc

from batcher._internal.paths import open_private, private_dir
from batcher.io.formats.streaming.checkpoint.location import CheckpointDir, is_local_location

__all__ = ["StateStore"]


def _sync_dir(directory: str) -> None:
    """Flush a directory entry so a completed rename survives a crash.

    A rename is durable only once the *directory* is synced; syncing the file alone leaves
    a window where the data is on disk under a name nothing points to. Best-effort: some
    filesystems (and Windows) refuse to open a directory for syncing, and a checkpoint that
    cannot sync its directory is still better than one that refuses to run.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform-dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform-dependent
        pass
    finally:
        os.close(fd)


def _serialize(state: pa.RecordBatch) -> bytes:
    """One state batch as an Arrow IPC file, in memory.

    Buffering costs nothing that is not already paid: this state is capped by
    `memory.streaming_state_max_bytes` and is a live in-process `RecordBatch` at the moment
    it is asked for, so the serialized copy is bounded by the same budget. Having the bytes
    in hand is what lets the local and remote writers differ only in *how they land them*.
    """
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, state.schema) as writer:
        writer.write_batch(state)
    return sink.getvalue().to_pybytes()


class StateStore:
    """Per-micro-batch Arrow-IPC snapshots of the running state under a directory."""

    __slots__ = ("_dir", "_local")

    def __init__(self, directory: str) -> None:
        from batcher.io.filesystem import local_path

        self._local = local_path(directory) if is_local_location(directory) else None
        if self._local is not None:
            # A snapshot holds the running aggregate's *actual* group keys and values, and
            # unlike spill or shuffle scratch it is durable — it outlives the query by
            # design. `state/` is Batcher's own subdirectory of the checkpoint location, so
            # tightening it protects the rows without touching the location the user named.
            private_dir(self._local)
        # Built for both, because reads, listing and deletes go through it either way.
        self._dir = CheckpointDir(self._local if self._local is not None else directory)

    @staticmethod
    def _name(batch_id: int) -> str:
        return f"batch-{batch_id:08d}.arrow"

    def snapshot(self, batch_id: int, state: pa.RecordBatch) -> None:
        """Atomically write the running `state` for `batch_id`.

        Any scalar that must ride with the state (the windowed fold's watermark and its
        per-partition maxima) travels in the batch's Arrow schema metadata, which IPC
        persists — so there is no separate sidecar to keep consistent with the ``.arrow``
        file.

        On local disk the write is made **durable** before the rename, and the rename itself
        durable by syncing the directory. The rename alone is atomic, not durable: it
        guarantees a reader never sees a half-written snapshot, and guarantees nothing about
        an OS-level crash. The engine snapshots the state and *then* records the commit, so
        without these syncs a crash could leave the commit on disk and the state not — and
        recovery would resume past data it had consumed with an empty running aggregate,
        which is silent wrong output rather than a lost query. Two fsyncs per stateful
        micro-batch is what that ordering costs to actually hold. On an object store the
        same ordering holds without them, because the PUT does not return until the object
        is durable.

        Args:
            batch_id: The micro-batch this snapshot belongs to.
            state: The running state to persist.
        """
        payload = _serialize(state)
        if self._local is None:
            self._dir.write(self._name(batch_id), payload)
            return
        path = os.path.join(self._local, self._name(batch_id))
        tmp = f"{path}.tmp"
        with open_private(tmp) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _sync_dir(self._local)

    def restore(self, batch_id: int) -> pa.RecordBatch | None:
        """Reload the running state snapshot for `batch_id`, or None if absent.

        **A snapshot with no rows is still a snapshot.** `Table.to_batches()` returns an
        empty list for a table with no rows — and for the zero-*column* table the windowed
        fold writes when its watermark has advanced past every open window, which is the
        ordinary state of a windowed query between windows. Returning ``None`` there made
        the engine skip `restore_state` entirely, so the watermark silently rewound to
        whatever the next batch happened to carry: rows the old watermark had correctly
        ruled late were re-admitted, and folded into windows that had already been emitted.
        The snapshot side goes out of its way to persist that case; this is the side that
        was dropping it.

        Rebuilding from the reader's schema rather than the table's batches keeps the
        payload that lives in the schema *metadata* — which is where the watermark rides,
        precisely so it needs no sidecar file.

        Args:
            batch_id: The micro-batch whose snapshot to reload.

        Returns:
            The snapshotted state, or None when no snapshot exists for `batch_id`.
        """
        name = self._name(batch_id)
        if not self._dir.exists(name):
            return None
        with self._dir.open_reader(name) as fh:
            reader = ipc.open_file(fh)
            schema = reader.schema
            table = reader.read_all()
        batches = table.to_batches()
        return batches[0] if batches else pa.RecordBatch.from_pylist([], schema=schema)

    def prune(self, keep_through: int) -> None:
        """Delete snapshots for batch ids below `keep_through` (state retention).

        Stale ``.tmp`` files go too. A crash between the write and the rename leaves one
        behind, and nothing else ever removes it — so the directory this method exists to
        bound would grow one orphan per crash, forever, and the *only* symptom would be a
        checkpoint location that slowly fills a disk. Only the local writer produces them;
        an object store's PUT has no temp sibling to leave.
        """
        if self._local is not None:
            for name in os.listdir(self._local):
                if name.endswith(".tmp"):
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(self._local, name))
        for name in self._dir.names(".arrow"):
            if not name.startswith("batch-"):
                continue
            try:
                bid = int(name[len("batch-") : -len(".arrow")])
            except ValueError:
                continue
            if bid < keep_through:
                self._dir.remove(name)
