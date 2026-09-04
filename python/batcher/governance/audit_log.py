"""The durable audit sink: governance decisions appended to a file that outlives the query.

`audit.GovernanceEvent` is the record and `api.security._binding._emit` is the emitter. This
module is the third piece: the append-only JSONL file `GovernanceConfig.audit_path` names.
It is deliberately **not** re-exported from `batcher.governance`: a user configures
`audit_path` and the engine writes, so exporting the writer would add a public name nobody
calls and commit the project to keeping it.
Without it the only durable trace of an authorization decision was a log line, which is a
debugging aid rather than an audit trail -- it interleaves with everything else the process
says, it is formatted for a human, and a deployment that ships logs to a search index has
already decided how long to keep them for reasons that have nothing to do with compliance.

Three properties this file has to have, and the reasons they are not merely style:

**Fail-closed.** A write that fails raises, and the read fails with it. That is the rule
`_emit` already states for a caller-supplied sink -- "a compliance pipeline that cannot record
an access should stop the access" -- and a configured sink is a stronger commitment than a
callback, not a weaker one. The alternative is a full disk quietly turning a governed
deployment into an ungoverned one, which is the failure an auditor is asking about.

**Reopened per record rather than held open.** A cached handle is faster and it is the reason
audit logs go missing: `logrotate` renames the file, and a process holding the old descriptor
appends to an unlinked inode for the rest of its life, writing to a file nobody can find.
`O_APPEND` per record is atomic for a record of this size on a POSIX filesystem and lets the
file be rotated underneath a running engine.

**Owner-only from the moment it exists.** Via `open_private`, so there is no window in which
the file is world-readable. The events name principals, tables and policies -- never values,
which `GovernanceEvent` guarantees -- but who read what is itself worth protecting.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

from batcher._internal.paths import open_private
from batcher.governance.audit import GovernanceEvent

__all__ = ["event_payload", "record_governance_event"]

# Serializes the append so two concurrent queries cannot interleave a record. `O_APPEND` already
# makes the write atomic at the syscall level; the lock is what keeps one *record* to one
# `write` call, since a JSON object split across two calls is a corrupt line either way.
_write_lock = threading.Lock()


def event_payload(event: GovernanceEvent, *, at: datetime | None = None) -> dict[str, object]:
    """The JSON-safe form of one governance decision, with the time it was recorded.

    The timestamp is stamped here rather than inside the event because `enforce` is a pure
    function of its inputs: a clock read inside the plan rewrite would make the rewrite
    non-deterministic, and the plan cache keys on it.

    Args:
        event: The decision to serialize.
        at: The moment to record, defaulting to now in UTC.

    Returns:
        A dict of JSON-safe scalars and string lists.
    """
    when = at or datetime.now(UTC)
    return {
        "at": when.isoformat(),
        "principal": event.principal,
        "roles": list(event.roles),
        "table": event.table,
        # Which privilege was decided. Without it a write decision and a read decision are
        # the same record in the file an auditor actually reads: both name a principal, a
        # table and a column list, and "who wrote to this table" is unanswerable.
        "privilege": event.privilege,
        "allowed": event.allowed,
        "visible": list(event.visible),
        "denied": list(event.denied),
        "masked": list(event.masked),
        "row_filters": list(event.row_filters),
    }


def record_governance_event(event: GovernanceEvent, path: str | None) -> None:
    """Append one decision to the configured audit file, as a single JSON line.

    A no-op when `path` is None, which is the default and every existing deployment.

    Examples:
        .. doctest::

            >>> import json, os, tempfile
            >>> from batcher.governance import GovernanceEvent
            >>> from batcher.governance.audit_log import record_governance_event
            >>> event = GovernanceEvent(
            ...     principal="ana",
            ...     roles=("analyst",),
            ...     table="/data/customers.parquet",
            ...     visible=("id",),
            ...     denied=("ssn",),
            ...     masked=(),
            ...     row_filters=(),
            ... )
            >>> path = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
            >>> record_governance_event(event, path)
            >>> with open(path) as fh:
            ...     json.loads(fh.read())["denied"]
            ['ssn']

    Args:
        event: The decision to record.
        path: The audit file, or None to record nothing.

    Raises:
        OSError: If the record cannot be written. Deliberately not suppressed -- see the
            module docstring on failing closed.
    """
    if not path:
        return
    line = (json.dumps(event_payload(event), separators=(",", ":")) + "\n").encode()
    with _write_lock, open_private(path, "ab") as fh:
        fh.write(line)
