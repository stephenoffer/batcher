"""Who a checkpoint belongs to: its stream id, the plan that wrote it, and its one owner.

A checkpoint location records *where* a streaming query had got to. Three questions about
it are not answered by the offset and commit logs, and each one, left unanswered, loses or
corrupts data while every log entry looks correct:

**Which stream is this?** The Delta sink makes a replayed micro-batch a no-op by checking
the table's log for an ``(app_id, batch_id)`` transaction, and a transaction counts as
committed when the recorded version is *at or past* the batch id. The default ``app_id``
used to be derived from the destination table. Two independent queries writing one table
therefore shared it, and so did two runs of one query without a checkpoint, whose batch
counter restarts at 0: the second run found batches ``0..n`` "already committed" and wrote
nothing, returning success. `stream_app_id` gives a checkpointed query an id persisted in
the checkpoint itself (Spark keeps its query id in the same place, for the same reason), and
a checkpoint-less query an id unique to the run, because without a checkpoint there is no
earlier run to be idempotent against.

**Was it written by this plan?** Restoring a running aggregate folded by a different
aggregation resumes from state that means something else. `CheckpointOwner.bind_plan`
records the plan's content fingerprint and refuses a *stateful* restart under a different
one. A stateless plan may change between runs, as in Spark: its checkpoint holds only
source positions, which a changed filter or projection reads the same way.

**Who is writing it?** Two drivers on one checkpoint race its offsets, state and commits.
`CheckpointOwner` takes a lease when the store opens. On local disk it is an exclusive
``flock``, so a second driver is refused at start and a crashed one releases it with its
process. An object store has no lock, so there the newest driver's token wins and every
older driver is fenced: its next offset record or commit reads the owner document, finds a
token that is not its own, and raises before touching the logs.

Layer: io (neutral). The fingerprint is computed by the caller, which owns the plan.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from typing import Any

from batcher.io.formats.streaming.checkpoint.location import CheckpointDir, is_local_location

__all__ = ["CheckpointOwner", "stream_app_id"]

_METADATA = "metadata.json"
_PLAN = "plan.json"
_OWNER = "owner.json"
_LOCK = "owner.lock"


def _root(location: str) -> str:
    if is_local_location(location):
        from batcher.io.filesystem import local_path

        return local_path(location)
    return location.rstrip("/")


def _dir(location: str) -> CheckpointDir:
    return CheckpointDir(_root(location))


def _load(directory: CheckpointDir, name: str) -> dict[str, Any] | None:
    raw = directory.read(name)
    if raw is None:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def _has_history(location: str) -> bool:
    """Whether the checkpoint already holds a recorded micro-batch (written before ids)."""
    root = _root(location)
    if is_local_location(location):
        return os.path.exists(os.path.join(root, "offsets.sqlite"))
    return bool(CheckpointDir(f"{root}/offsets").names(".json"))


def stream_app_id(query_name: str | None, checkpoint: str | None, destination: str) -> str:
    """The transaction application id a streaming write stamps on each micro-batch.

    Args:
        query_name: The caller's query name. With a checkpoint it is used verbatim, so a
            named query keeps its id across restarts and is responsible for its uniqueness.
        checkpoint: The query's checkpoint location, or ``None``.
        destination: The table being written. Used only for a checkpoint written before
            stream ids existed, whose earlier commits carry the old destination-derived id.

    Returns:
        With a checkpoint, `query_name` or ``batcher-stream:<id>`` for the id persisted in
        the checkpoint (created on first use). Without one, an id unique to this run: the
        batch counter restarts at 0, so any id shared with an earlier run would make this
        run's batches look already committed.
    """
    if checkpoint is None:
        return f"{query_name or 'batcher-stream'}:run-{uuid.uuid4().hex}"
    if query_name:
        return query_name
    directory = _dir(checkpoint)
    doc = _load(directory, _METADATA)
    if doc is not None and isinstance(doc.get("app_id"), str):
        return doc["app_id"]
    # A checkpoint with history but no identity predates this module. Its committed batches
    # carry the destination-derived id, and adopting it keeps the in-flight replay
    # idempotent; a fresh id would re-append that batch.
    legacy = _has_history(checkpoint)
    app_id = (
        f"batcher-stream:{destination.rstrip('/')}"
        if legacy
        else f"batcher-stream:{uuid.uuid4().hex}"
    )
    directory.write(_METADATA, json.dumps({"app_id": app_id}).encode())
    return app_id


class CheckpointOwner:
    """The lease one streaming driver holds on its checkpoint, and the plan it bound."""

    __slots__ = ("_dir", "_local", "_lock_fd", "_token")

    def __init__(self, location: str) -> None:
        from batcher._internal.errors import CommitError

        self._dir = _dir(location)
        self._local = is_local_location(location)
        self._token = uuid.uuid4().hex
        self._lock_fd: int | None = None
        if self._local:
            self._lock_fd, locked = _try_lock(os.path.join(_root(location), _LOCK))
            # Without flock on this platform the owner token below is the only fence.
            self._local = locked
            if self._lock_fd is None:
                raise CommitError(
                    f"checkpoint {location!r} is in use by another running streaming query. "
                    "Two drivers on one checkpoint race its offsets, state and commits; stop "
                    "the other query, or give this one its own checkpoint location."
                )
        self._dir.write(_OWNER, json.dumps({"token": self._token}).encode())

    def verify(self) -> None:
        """Raise `CommitError` when a newer driver has taken this checkpoint over.

        Local ownership is an exclusive lock held for the query's life, so there is nothing
        to re-check. Remote ownership is last-writer-wins, and this is the fence.
        """
        if self._local:
            return
        from batcher._internal.errors import CommitError

        doc = _load(self._dir, _OWNER)
        if doc is None or doc.get("token") != self._token:
            raise CommitError(
                "this streaming query no longer owns its checkpoint: another driver started "
                "on the same checkpoint location and took it over. This driver stops "
                "without recording offsets or commits, so the two cannot interleave."
            )

    def bind_plan(self, fingerprint: str | None, *, stateful: bool, has_history: bool) -> None:
        """Record the plan writing this checkpoint, refusing an incompatible stateful restart.

        Args:
            fingerprint: The plan's content fingerprint, or ``None`` for a plan with an
                opaque node (a Python UDF) that cannot be fingerprinted across processes.
            stateful: Whether the checkpoint carries running state for this plan.
            has_history: Whether the checkpoint already committed a micro-batch.

        Raises:
            PlanError: When a stateful plan restarts on a checkpoint whose state was folded
                by a different plan.
        """
        previous = _load(self._dir, _PLAN)
        if (
            has_history
            and stateful
            and previous is not None
            and previous.get("stateful")
            and fingerprint is not None
            and previous.get("fingerprint") not in (None, fingerprint)
        ):
            from batcher._internal.errors import PlanError

            raise PlanError(
                "this checkpoint holds running state written by a different streaming plan: "
                "restoring it would resume an aggregation, window or dedup from state that a "
                "different computation produced. Keep the original plan to resume, or start "
                "the changed plan on a new checkpoint location (it then reprocesses from the "
                "source's starting position)."
            )
        doc = {"fingerprint": fingerprint, "stateful": stateful}
        if previous != doc:
            self._dir.write(_PLAN, json.dumps(doc).encode())

    def release(self) -> None:
        """Give the lease up (the owner document stays; the next owner overwrites it)."""
        if self._lock_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._lock_fd)
            self._lock_fd = None


def _try_lock(path: str) -> tuple[int | None, bool]:
    """``(fd, locked)`` for an exclusive lock on `path`; ``fd`` is ``None`` when it is held.

    ``locked`` is False only on a platform without ``flock``, where the fd is returned
    unlocked and the owner token is left to do the fencing.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        import fcntl
    except ImportError:
        return fd, False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None, True
    return fd, True
