"""`GovernanceEvent` — the record of one authorization decision.

Emitted once per governed table per read: who asked, what they were allowed to see,
what was withheld, what was masked, and which row filters were applied. It is the
artifact a compliance review actually wants — not "a query ran", but "this principal
read these columns of this table, under these policies".

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
    """

    principal: str
    roles: tuple[str, ...]
    table: str
    visible: tuple[str, ...]
    denied: tuple[str, ...]
    masked: tuple[str, ...]
    row_filters: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Whether the principal may read the table at all.

        False exactly when no column is visible — the case that raises
        `AccessDeniedError`, and the event a security review most wants to find.
        """
        return bool(self.visible)

    def __str__(self) -> str:
        """A single-line, log-friendly rendering of the decision."""
        if not self.allowed:
            return f"governance: DENY {self.principal}@{sorted(self.roles)} -> {self.table}"
        parts = [f"visible={list(self.visible)}"]
        if self.denied:
            parts.append(f"denied={list(self.denied)}")
        if self.masked:
            parts.append(f"masked={list(self.masked)}")
        if self.row_filters:
            parts.append(f"row_filters={list(self.row_filters)}")
        return (
            f"governance: ALLOW {self.principal}@{sorted(self.roles)} -> "
            f"{self.table} ({', '.join(parts)})"
        )
