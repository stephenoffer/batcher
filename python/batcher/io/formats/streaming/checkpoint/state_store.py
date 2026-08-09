"""Durable snapshots of a streaming query's running aggregation state.

The running state in `core.streaming._AggFold` is one Arrow ``RecordBatch`` (the
output of the native ``combine``), so it serializes with ``pyarrow.ipc`` exactly
like the ML shard writer — no FFI addition needed. Snapshots are written atomically
(temp file + rename) per micro-batch and reloaded on recovery to resume a stateful
query without recomputing from the start of the stream.
"""

from __future__ import annotations

import contextlib
import os

import pyarrow as pa
from pyarrow import ipc

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


class StateStore:
    """Per-micro-batch Arrow-IPC snapshots of the running state under a directory."""

    __slots__ = ("_dir",)

    def __init__(self, directory: str) -> None:
        self._dir = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, batch_id: int) -> str:
        return os.path.join(self._dir, f"batch-{batch_id:08d}.arrow")

    def snapshot(self, batch_id: int, state: pa.RecordBatch) -> None:
        """Atomically write the running `state` for `batch_id` (temp file + rename).

        Any scalar that must ride with the state (the windowed fold's watermark) travels in
        the batch's Arrow schema metadata, which IPC persists — so there is no separate
        sidecar to keep consistent with the ``.arrow`` file.

        The write is made **durable** before the rename, and the rename itself is made
        durable by syncing the directory. The rename alone is atomic, not durable: it
        guarantees a reader never sees a half-written snapshot, and guarantees nothing about
        an OS-level crash. The engine snapshots the state and *then* records the commit, so
        without these syncs a crash could leave the commit on disk and the state not — and
        recovery would resume past data it had consumed with an empty running aggregate,
        which is silent wrong output rather than a lost query. Two fsyncs per stateful
        micro-batch is what that ordering costs to actually hold.
        """
        path = self._path(batch_id)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            with ipc.new_file(fh, state.schema) as writer:
                writer.write_batch(state)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _sync_dir(self._dir)

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
        path = self._path(batch_id)
        if not os.path.exists(path):
            return None
        with ipc.open_file(path) as reader:
            schema = reader.schema
            table = reader.read_all()
        batches = table.to_batches()
        return batches[0] if batches else pa.RecordBatch.from_pylist([], schema=schema)

    def prune(self, keep_through: int) -> None:
        """Delete snapshots for batch ids below `keep_through` (state retention).

        Stale ``.tmp`` files go too. A crash between the write and the rename leaves one
        behind, and nothing else ever removes it — so the directory this method exists to
        bound would grow one orphan per crash, forever, and the *only* symptom would be a
        checkpoint location that slowly fills a disk.
        """
        for name in os.listdir(self._dir):
            if name.endswith(".tmp"):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(self._dir, name))
                continue
            if not name.startswith("batch-") or not name.endswith(".arrow"):
                continue
            try:
                bid = int(name[len("batch-") : -len(".arrow")])
            except ValueError:
                continue
            if bid < keep_through:
                os.remove(os.path.join(self._dir, name))
