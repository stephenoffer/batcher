"""`GovernanceEvent` — the record of one authorization decision.

Emitted once per governed table per read: who asked, what they were allowed to see,
what was withheld, what was masked, and which row filters were applied. It is the
artifact a compliance review actually wants — not "a query ran", but "this principal
read these columns of this table, under these policies".

A **write** decision is the same record with `privilege` naming the write
(``INSERT``/``UPDATE``/``DELETE``) and `visible` naming the columns written. One shape
for both, so an audit reader does not have to join two logs to answer "who touched this
table", and so `allowed` keeps its one meaning: something came through.

An event names *columns and policies*, never **values** and never key material. It is
designed to be safe to write to a log, so nothing in it may be sensitive: the whole
point is that it survives in a place the data may not.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GovernanceEvent"]


@dataclass(frozen=True, slots=True)
class GovernanceEvent:
    """One principal's authorized view of one table, as resolved by the catalog.

    Carries no timestamp: `enforce` is a pure function, and stamping a clock inside it
    would make the plan rewrite non-deterministic. The emitter stamps the event.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ev = bt.GovernanceEvent(
            ...     principal="ana",
            ...     roles=("analyst",),
            ...     table="/data/customers.parquet",
            ...     visible=("id", "email"),
            ...     denied=("ssn",),
            ...     masked=("email",),
            ...     row_filters=(),
            ... )
            >>> ev.allowed
            True
            >>> ev.visible
            ('id', 'email')
    """

    principal: str
    roles: tuple[str, ...]
    table: str
    visible: tuple[str, ...]
    denied: tuple[str, ...]
    masked: tuple[str, ...]
    row_filters: tuple[str, ...]
    #: The privilege this decision was about, one of `batcher.governance.PRIVILEGES`.
    #: Defaults to ``"SELECT"`` so every event built before writes were governed reads
    #: as exactly what it was.
    privilege: str = "SELECT"

    @property
    def allowed(self) -> bool:
        """Whether the principal may exercise `privilege` on the table at all.

        False exactly when no column is visible — the case that raises
        `AccessDeniedError`, and the event a security review most wants to find. A
        refused write carries no columns for the same reason a refused read does not:
        nothing came through.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.GovernanceEvent("ana", (), "t", ("id",), (), (), ()).allowed
                True
                >>> bt.GovernanceEvent("ana", (), "t", (), ("id",), (), ()).allowed
                False

        Returns:
            True unless no column is visible.
        """
        return bool(self.visible)

    def __str__(self) -> str:
        """A single-line, log-friendly rendering of the decision."""
        verb = "" if self.privilege == "SELECT" else f" {self.privilege}"
        if not self.allowed:
            return f"governance: DENY{verb} {self.principal}@{sorted(self.roles)} -> {self.table}"
        parts = [f"visible={list(self.visible)}"]
        if self.denied:
            parts.append(f"denied={list(self.denied)}")
        if self.masked:
            parts.append(f"masked={list(self.masked)}")
        if self.row_filters:
            parts.append(f"row_filters={list(self.row_filters)}")
        return (
            f"governance: ALLOW{verb} {self.principal}@{sorted(self.roles)} -> "
            f"{self.table} ({', '.join(parts)})"
        )
