"""Durable snapshots of a streaming query's running aggregation state.

The running state in `core.streaming._AggFold` is one Arrow ``RecordBatch`` (the
output of the native ``combine``), so it serializes with ``pyarrow.ipc`` exactly
like the ML shard writer — no FFI addition needed. Snapshots are written atomically
(temp file + rename) per micro-batch and reloaded on recovery to resume a stateful
query without recomputing from the start of the stream.
"""

from __future__ import annotations

import json
import os

import pyarrow as pa
from pyarrow import ipc

__all__ = ["StateStore"]


class StateStore:
    """Per-micro-batch Arrow-IPC snapshots of the running state under a directory."""

    __slots__ = ("_dir",)

    def __init__(self, directory: str) -> None:
        self._dir = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, batch_id: int) -> str:
        return os.path.join(self._dir, f"batch-{batch_id:08d}.arrow")

    def snapshot(self, batch_id: int, state: pa.RecordBatch, meta: dict | None = None) -> None:
        """Atomically write the running `state` for `batch_id` (temp file + rename)."""
        path = self._path(batch_id)
        tmp = f"{path}.tmp"
        with ipc.new_file(tmp, state.schema) as writer:
            writer.write_batch(state)
        os.replace(tmp, path)
        if meta is not None:
            with open(f"{path}.meta.json", "w") as fh:
                json.dump(meta, fh)

    def restore(self, batch_id: int) -> pa.RecordBatch | None:
        """Reload the running state snapshot for `batch_id`, or None if absent."""
        path = self._path(batch_id)
        if not os.path.exists(path):
            return None
        with ipc.open_file(path) as reader:
            table = reader.read_all()
        batches = table.to_batches()
        return batches[0] if batches else None

    def prune(self, keep_through: int) -> None:
        """Delete snapshots for batch ids below `keep_through` (state retention)."""
        for name in os.listdir(self._dir):
            if not name.startswith("batch-") or not name.endswith(".arrow"):
                continue
            try:
                bid = int(name[len("batch-") : -len(".arrow")])
            except ValueError:
                continue
            if bid < keep_through:
                os.remove(os.path.join(self._dir, name))
                meta = os.path.join(self._dir, f"{name}.meta.json")
                if os.path.exists(meta):
                    os.remove(meta)
