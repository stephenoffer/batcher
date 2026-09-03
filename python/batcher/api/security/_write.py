"""Where governance meets a write: which privilege it needs, and whether the principal has it.

The read side (`_binding.govern_scan`) governs at the *scan*, because a `Dataset` is lazy
and enforcing at the terminal operation would mean enforcing at a dozen of them. A write
has the opposite shape: it is a terminal operation, there is exactly one of them, and the
destination is not known until it is called. So this runs at the write itself — at
`api.terminal.core._write`, the single function every batch, streaming, and MERGE write
passes through.

**Why a write needs its own check at all.** The read rewrite governs the tables a query
*reads*. It says nothing about the table a query *writes*, which is usually a different
table and often a more sensitive one: the whole point of a masked read is undone if the
principal can write the unmasked join result somewhere it can read back freely. Until
this existed, a `SELECT` grant was the only privilege the engine knew, so every governed
deployment had an ungoverned write path.

**One call site, and it is not the earliest possible one.** The check runs inside
`_write` rather than in each of the three callers, because an authorization check with
three call sites has three ways to be forgotten. The cost is that `mode="error"` raises
its "path already exists" refusal *before* the privilege check, so an unauthorized
principal can still learn whether a path it named exists. That is one bit about a path
the caller supplied, and buying it back would mean either duplicating the check or
introducing an "already authorized" flag — both of which trade the property that actually
matters (there is exactly one place a write can be authorized) for a marginal one.

**Privileges follow the save mode, not the format.** ``append`` adds rows and needs
``INSERT``; ``overwrite`` destroys the rows that were there and needs ``DELETE`` as well,
which is the distinction that lets a load job be granted ``INSERT`` alone and be unable to
drop yesterday's data. The map below is the whole of that reasoning and is the only place
it is written down.
"""

from __future__ import annotations

from collections.abc import Sequence

from batcher._internal.errors import AccessDeniedError, PlanError, unknown_value
from batcher.api.security._binding import emit_event
from batcher.api.security._context import current_security
from batcher.governance import PRIVILEGES, GovernanceEvent
from batcher.io.filesystem import canonical_path

__all__ = [
    "authorize_write",
    "merge_privileges",
    "refuse_governed_rewrite",
    "required_privileges",
]

#: What each write mode does to the rows already in the destination, as privileges.
#:
#: Every save mode and every row-level DML verb the sinks accept is listed, because a mode
#: missing from this map would fall through to a default and be governed as something it
#: is not. `required_privileges` raises on an unknown mode rather than guessing — on an
#: authorization boundary, a mode nobody has classified must not be silently permitted.
_MODE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    # Save modes. `error` and `ignore` only ever create, so they add rows and nothing else.
    "append": ("INSERT",),
    "error": ("INSERT",),
    "ignore": ("INSERT",),
    # Overwrite replaces: the rows that were there are gone. A role holding INSERT alone
    # can load new data and cannot destroy what is already loaded.
    "overwrite": ("INSERT", "DELETE"),
    "overwrite_partitions": ("INSERT", "DELETE"),
    # Row-level DML, as the database and document-store sinks accept it.
    "upsert": ("INSERT", "UPDATE"),
    "update": ("UPDATE",),
    "delete": ("DELETE",),
    "delete_insert": ("INSERT", "DELETE"),
}


def required_privileges(mode: str) -> tuple[str, ...]:
    """The privileges a write in `mode` needs, in `PRIVILEGES` order.

    Examples:
        .. doctest::

            >>> from batcher.api.security._write import required_privileges
            >>> required_privileges("append")
            ('INSERT',)
            >>> required_privileges("overwrite")
            ('INSERT', 'DELETE')

    Args:
        mode: A save mode or a row-level DML verb, already normalized by the writer.

    Returns:
        The privileges, ordered as `PRIVILEGES` orders them so a message reads the same
        way every time.

    Raises:
        PlanError: If `mode` names no known write. A mode nobody has classified is not
            assumed harmless.
    """
    needed = _MODE_PRIVILEGES.get(mode) if isinstance(mode, str) else None
    if needed is None:
        raise unknown_value(
            PlanError,
            "write mode",
            mode,
            sorted(_MODE_PRIVILEGES),
            label="Governed write modes",
            hint=(
                "Every write mode must be classified as the privileges it needs before "
                "it can be authorized."
            ),
        )
    return tuple(p for p in PRIVILEGES if p in needed)


def authorize_write(path: str, columns: Sequence[str], privileges: Sequence[str]) -> None:
    """Refuse the write unless the active principal holds every one of `privileges`.

    Returns silently when no `security()` block is active, or when the catalog declares no
    policy naming this destination — the same "a table nobody wrote a policy about is left
    alone" rule the read path follows, so installing a catalog does not break every
    pipeline that writes somewhere unmentioned.

    The decision is audited either way, through the same sinks and in the same
    `GovernanceEvent` shape a read produces, so "who wrote to this table" and "who read
    it" are one query over one log rather than two.

    Args:
        path: The write destination, in any spelling. Canonicalized before it is matched,
            so ``s3a://`` cannot walk past a policy written about ``s3://``.
        columns: The columns being written, recorded in the audit event.
        privileges: What this write needs, from `required_privileges`.

    Raises:
        AccessDeniedError: If the principal is missing any of `privileges` on `path`.
    """
    ctx = current_security()
    if ctx is None:
        return
    table = canonical_path(path)
    if not table or not ctx.catalog.governs(table):
        return
    missing = tuple(p for p in privileges if not ctx.catalog.holds(table, ctx.principal, p))
    principal = ctx.principal
    granted = tuple(p for p in privileges if p not in missing)
    # One event per write, naming the privilege actually at issue: the first one refused,
    # or — when nothing was refused — the strongest one exercised. `PRIVILEGES` order puts
    # SELECT first and the destructive verbs last, so "strongest" is simply the last.
    at_issue = missing[0] if missing else (granted[-1] if granted else "INSERT")
    emit_event(
        ctx,
        GovernanceEvent(
            principal=principal.name,
            roles=tuple(sorted(principal.roles)),
            table=table,
            visible=() if missing else tuple(columns),
            denied=missing,
            masked=(),
            row_filters=(),
            privilege=at_issue,
        ),
    )
    if not missing:
        return
    held = sorted(principal.roles)
    raise AccessDeniedError(
        f"Principal {principal.name!r} may not write to {table!r}: missing {', '.join(missing)}.",
        table=table,
        hint=(
            f"Grant it with catalog.grant(<role>, on={table!r}, "
            f"privilege={missing[0]!r}), or give the principal a role that already "
            f"holds it (it holds {held or 'no roles'})."
        ),
    )


#: A merge clause's action, as the privilege it exercises. The three actions a
#: `MergeClause` may take are exactly the three row-level privileges, which is not a
#: coincidence: SQL's ``MERGE`` is defined as the composition of those three statements.
_ACTION_PRIVILEGE = {"insert": "INSERT", "update": "UPDATE", "delete": "DELETE"}


def merge_privileges(clauses: Sequence[object]) -> tuple[str, ...]:
    """The privileges a ``MERGE`` with these `clauses` needs, in `PRIVILEGES` order.

    Derived from what the clauses actually do rather than from the fact that a merge was
    used, so a merge that only inserts needs only ``INSERT``. Requiring all three would
    make the privilege useless for the common insert-only upsert, and a privilege nobody
    can grant narrowly is one every role ends up holding.

    A clause whose action is unrecognized contributes every privilege rather than none.
    That is deliberate and is the one place here that over-requires: a new merge action
    must fail closed until someone classifies it.

    Examples:
        .. doctest::

            >>> from batcher.api.merge.clauses import MergeClause
            >>> from batcher.api.security._write import merge_privileges
            >>> merge_privileges([MergeClause("not_matched", "insert")])
            ('INSERT',)
            >>> merge_privileges(
            ...     [MergeClause("matched", "delete"), MergeClause("not_matched", "insert")]
            ... )
            ('INSERT', 'DELETE')

    Args:
        clauses: The merge's ordered ``WHEN …`` clauses.

    Returns:
        The privileges needed, ordered as `PRIVILEGES` orders them.
    """
    needed: set[str] = set()
    for clause in clauses:
        action = getattr(clause, "action", None)
        privilege = _ACTION_PRIVILEGE.get(action) if isinstance(action, str) else None
        if privilege is None:
            needed.update(p for p in PRIVILEGES if p != "SELECT")
        else:
            needed.add(privilege)
    return tuple(p for p in PRIVILEGES if p in needed)


def refuse_governed_rewrite(path: str, operation: str) -> None:
    """Refuse `operation` when a policy governs `path`, because it rewrites the table.

    A maintenance rewrite reads a table and writes the result back over it. Inside a
    `security()` block the read is the *principal's* view: masked columns hold their mask
    and columns the principal cannot select are absent. Writing that back replaces the
    table with it.

    That is not a hypothetical. Compacting a two-file table under a catalog that masked
    `email` and withheld `ssn` left the table holding ``'XXXXXXX'`` for every address and
    no `ssn` column at all -- the real values destroyed, no error raised, and nothing in
    the result to suggest anything had gone wrong. A governed *read* narrows what one
    principal sees; a governed read fed back into a write narrows the table itself, and
    permanently.

    Maintenance is therefore refused rather than made to work. Making it work means giving
    the operation the authority to read the raw table from inside a block whose whole
    purpose is to prevent that, which is a governance bypass with a compaction attached --
    a much worse thing to own than an operator running maintenance outside the block, which
    is how every warehouse runs `OPTIMIZE` anyway.

    The refusal is unconditional on the table being governed, not conditional on the view
    actually being narrower. Deciding "narrower" needs the table's raw column list, which is
    the one thing this context cannot obtain without the bypass the refusal exists to avoid.

    Args:
        path: The table the operation would rewrite, in any spelling.
        operation: What the caller is doing, named in the message (e.g. ``"compact"``).

    Raises:
        AccessDeniedError: If a `security()` block governs `path`.
    """
    ctx = current_security()
    if ctx is None:
        return
    table = canonical_path(path)
    if not table or not ctx.catalog.governs(table):
        return
    raise AccessDeniedError(
        f"{operation}() will not rewrite {table!r} from inside a security() block: a "
        "policy governs this table, so the rewrite would write the principal's masked and "
        "column-pruned view back over the real data.",
        table=table,
        hint=(
            f"Run {operation}() outside the security() block, with the engine's own "
            "authority over the table."
        ),
    )
