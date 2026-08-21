"""Durable snapshots of a streaming query's running aggregation state.

The running state in `core.streaming._AggFold` is one Arrow ``RecordBatch`` (the
output of the native ``combine``), so it serializes with ``pyarrow.ipc`` exactly
like the ML shard writer — no FFI addition needed. Snapshots are written atomically
per micro-batch and reloaded on recovery to resume a stateful query without
recomputing from the start of the stream.

Two kinds of file live here. A **snapshot** (``batch-<id>.arrow``) is the whole running
state; a **delta** (``batch-<id>.delta.arrow``) is one micro-batch's partial. Rewriting the
whole state every epoch makes the checkpoint cost grow with the state it protects, which for
an unwatermarked aggregate — the one that never evicts and only grows — means the per-epoch
cost rises for the life of the query. A delta costs the *batch's* distinct group count
instead. Recovery combines the newest snapshot with every delta recorded after it, which is
sound because `combine` is associative and commutative (invariant #7).

Only a fold that never *removes* rows may use deltas: a delta chain has no way to express an
eviction, so replaying it would resurrect a closed window. `core.streaming.folds._AggFold`
qualifies and offers a delta; `_WindowedAggFold` does not and keeps whole snapshots.

Local and remote are written differently, and `location.py` says why: a local write is
fsynced and renamed and the directory fsynced, because the engine snapshots state and
*then* records the commit; on an object store a PUT is durable when it returns and there is
no rename to make durable separately. Reading, listing, and deleting are the same code on
both, because none of them has a durability ordering to preserve.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from pyarrow import ipc

from batcher._internal.paths import open_private, private_dir
from batcher.io.formats.streaming.checkpoint.location import CheckpointDir, is_local_location

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

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


def _parts(state: pa.RecordBatch | Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
    """Normalize a snapshot argument to a stream of non-empty batches."""
    if isinstance(state, pa.RecordBatch):
        yield state
        return
    for batch in state:
        if batch.num_rows or batch.num_columns == 0:
            yield batch


def _write_parts(sink: Any, parts: Iterator[pa.RecordBatch]) -> bool:
    """Write every part into one Arrow IPC file on `sink`; False when there was nothing.

    The file's schema is the **first** part's, and later parts are rebound to it. That is not
    cosmetic: the resident half of a windowed fold's state carries the watermark in its schema
    metadata and the spilled halves do not, so writing them as they come would have IPC reject
    the second batch for a schema that differs only in metadata — losing the spilled state to
    an error that names none of this.

    Streaming rather than concatenating is the whole point. A spilling fold's state is larger
    than `memory.streaming_state_max_bytes` by construction, so building one batch to serialize
    would undo the bound the spill exists to hold, on exactly the queries big enough to spill.
    """
    writer = None
    schema: pa.Schema | None = None
    try:
        for batch in parts:
            if writer is None:
                schema = batch.schema
                writer = ipc.new_file(sink, schema)
                writer.write_batch(batch)
                continue
            writer.write_batch(
                batch
                if batch.schema.equals(schema)
                else pa.RecordBatch.from_arrays(list(batch.columns), schema=schema)
            )
    finally:
        if writer is not None:
            writer.close()
    return writer is not None


def _serialize(state: pa.RecordBatch | Iterable[pa.RecordBatch]) -> bytes:
    """A snapshot as an Arrow IPC file, in memory — the object-store path.

    A PUT needs the whole object, so the remote tier buffers where the local one streams. A
    spilled state therefore costs its full size in memory at snapshot time on an object-store
    checkpoint; see `StateStore._write` for the local path that does not.
    """
    sink = pa.BufferOutputStream()
    _write_parts(sink, _parts(state))
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

    #: Zero-padded so a lexical listing is also a numeric one, which is what lets the
    #: object-store backend order files without parsing every name twice.
    _SNAPSHOT = "batch-{:08d}.arrow"
    _DELTA = "batch-{:08d}.delta.arrow"

    @staticmethod
    def _name(batch_id: int) -> str:
        return StateStore._SNAPSHOT.format(batch_id)

    @staticmethod
    def _delta_name(batch_id: int) -> str:
        return StateStore._DELTA.format(batch_id)

    @staticmethod
    def _parse(name: str) -> tuple[int, bool] | None:
        """``(batch_id, is_delta)`` for a state file name, or None if it is not one."""
        if not name.startswith("batch-") or not name.endswith(".arrow"):
            return None
        stem = name[len("batch-") : -len(".arrow")]
        is_delta = stem.endswith(".delta")
        if is_delta:
            stem = stem[: -len(".delta")]
        try:
            return int(stem), is_delta
        except ValueError:
            return None

    def _index(self) -> list[tuple[int, bool]]:
        """Every state file as ``(batch_id, is_delta)``, oldest first.

        One listing, shared by restore and prune. On an object store a listing is an API
        call, and doing it twice per commit — which is what a separate scan in each method
        cost — doubles the request count of a checkpoint that is already the epoch's
        slowest step.
        """
        found = [self._parse(name) for name in self._dir.names(".arrow")]
        return sorted(entry for entry in found if entry is not None)

    def snapshot(self, batch_id: int, state: pa.RecordBatch | Iterable[pa.RecordBatch]) -> None:
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
        self._write(self._name(batch_id), state)
        # A restart can leave the *other* kind of file for this id behind: run 1 recorded a
        # delta for batch 58, crashed before committing it, and run 2 reprocessed 58 and
        # snapshotted instead. Both would then sit under one batch id, and which one a chain
        # picked would depend on the file naming rather than on anything meaningful. One
        # file per id, always — the write that lands is the one that counts.
        self._dir.remove(self._delta_name(batch_id))

    def snapshot_delta(self, batch_id: int, delta: pa.RecordBatch) -> None:
        """Atomically record one micro-batch's changelog entry for `batch_id`.

        Written with the same durability ordering as a whole snapshot, because it carries
        the same obligation: the engine records the delta and *then* commits the batch, so a
        delta that is not durable when the commit is leaves recovery rebuilding state from a
        chain with a hole in it — silently short totals rather than a failed query.

        Args:
            batch_id: The micro-batch this delta belongs to.
            delta: The partial that micro-batch folded in.
        """
        self._write(self._delta_name(batch_id), delta)
        self._dir.remove(self._name(batch_id))  # see `snapshot` — one file per batch id

    def _write(self, name: str, state: pa.RecordBatch | Iterable[pa.RecordBatch]) -> None:
        """Land one state file atomically, and durably where the filesystem needs it.

        The local path writes the parts **straight into the temp file**, so peak memory is one
        part rather than the whole snapshot. That matters now that a windowed fold can spill:
        its state is deliberately larger than the memory cap, and serializing it into a buffer
        first would put all of it back in memory at every checkpoint.
        """
        if self._local is None:
            self._dir.write(name, _serialize(state))
            return
        path = os.path.join(self._local, name)
        tmp = f"{path}.tmp"
        with open_private(tmp) as fh:
            wrote = _write_parts(fh, _parts(state))
            fh.flush()
            os.fsync(fh.fileno())
        if not wrote:
            os.remove(tmp)
            return
        os.replace(tmp, path)
        _sync_dir(self._local)

    def restore(self, batch_id: int) -> pa.RecordBatch | None:
        """Reload the whole running state as of `batch_id`, or None if there is none.

        Answers from the delta chain when there is one: the newest snapshot at or before
        `batch_id` plus every delta after it. A caller that cannot combine partials should
        use `restore_chain` and do so itself — this convenience exists for the single-file
        case and for tests, and it returns only the **base** when deltas are present, which
        is why the engine does not use it. See `restore_chain`.

        **A snapshot with no rows is still a snapshot.** `Table.to_batches()` returns an
        empty list for a table with no rows — and for the zero-*column* table the windowed
        fold writes when its watermark has advanced past every open window, which is the
        ordinary state of a windowed query between windows. Returning ``None`` there made
        the engine skip `restore_state` entirely, so the watermark silently rewound to
        whatever the next batch happened to carry: rows the old watermark had correctly
        ruled late were re-admitted, and folded into windows that had already been emitted.

        Args:
            batch_id: The micro-batch whose state to reload.

        Returns:
            The base snapshot at or before `batch_id`, or None when none exists.
        """
        chain = self.restore_chain(batch_id)
        return chain[0] if chain else None

    def restore_chain(self, batch_id: int) -> list[pa.RecordBatch]:
        """The base snapshot for `batch_id` and every delta recorded after it, oldest first.

        The base is the newest snapshot **at or before** `batch_id`, not the one named by it.
        Two things make the distinction load-bearing. A delta-checkpointed epoch writes no
        snapshot at all, so an exact-name lookup would find nothing and the engine would
        resume a stateful query with empty state — silent wrong output, not a failure. And a
        micro-batch that folded nothing in writes neither file, which is the ordinary state
        of an idle trigger.

        Deltas *after* `batch_id` are excluded. They belong to an epoch that was recorded and
        never committed, so replaying them would fold in a micro-batch the query is about to
        re-read — double-counting every row of it.

        Args:
            batch_id: The last committed micro-batch.

        Returns:
            The batches to combine, base first. Empty when no state was ever written.
        """
        index = self._index()
        base_id: int | None = None
        for bid, is_delta in index:
            if not is_delta and bid <= batch_id:
                base_id = bid
        chain: list[pa.RecordBatch] = []
        if base_id is not None:
            chain.extend(self._read(self._name(base_id)))
        floor = -1 if base_id is None else base_id
        for bid, is_delta in index:
            if is_delta and floor < bid <= batch_id:
                chain.extend(self._read(self._delta_name(bid)))
        return chain

    def _read(self, name: str) -> list[pa.RecordBatch]:
        """Every batch in one state file, oldest part first.

        A list rather than a single batch because a snapshot is now multi-part: a windowed
        fold that has spilled writes its resident state and each spilled run into one file.
        Returning only the first — which is what this did — would have silently restored the
        resident half and dropped everything the spill had moved to disk.

        Rebuilding from the reader's schema when the file holds no batches keeps the payload
        that lives in the schema *metadata* — where the windowed fold's watermark rides,
        precisely so it needs no sidecar file — and keeps a zero-row snapshot a snapshot.
        """
        if not self._dir.exists(name):
            return []
        with self._dir.open_reader(name) as fh:
            reader = ipc.open_file(fh)
            schema = reader.schema
            table = reader.read_all()
        batches = table.to_batches()
        return batches if batches else [pa.RecordBatch.from_pylist([], schema=schema)]

    def prune(self, keep_through: int) -> None:
        """Delete state files no recovery can still need (bounded ``state/`` directory).

        Recovery rebuilds from the newest snapshot at or before the last committed batch plus
        the deltas after it, so that snapshot is the **floor**: everything strictly older is
        dead weight, and everything from the floor up is live. Pruning by batch id alone —
        which is what this did before deltas existed — would delete the base a chain of
        deltas depends on and leave the deltas behind, so recovery would combine partials
        with no state under them and silently resume with a fraction of the aggregate.

        Stale ``.tmp`` files go too. A crash between the write and the rename leaves one
        behind, and nothing else ever removes it — so the directory this method exists to
        bound would grow one orphan per crash, forever, and the *only* symptom would be a
        checkpoint location that slowly fills a disk. Only the local writer produces them;
        an object store's PUT has no temp sibling to leave.

        Args:
            keep_through: The last committed micro-batch; nothing recovery could need to
                reach it is removed.
        """
        if self._local is not None:
            for name in os.listdir(self._local):
                if name.endswith(".tmp"):
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(self._local, name))
        index = self._index()
        floor = keep_through
        for bid, is_delta in index:
            if not is_delta and bid <= keep_through:
                floor = bid
        for bid, is_delta in index:
            if bid < floor:
                self._dir.remove(self._delta_name(bid) if is_delta else self._name(bid))
