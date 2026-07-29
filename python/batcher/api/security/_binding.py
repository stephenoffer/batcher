"""Where the governance subsystem meets a scan: naming a table, governing it, auditing it.

**Why enforcement happens at the read, not at the terminal operation.** A `Dataset` is
a lazy handle to a plan. If governance ran at `collect()`, it would also have to run at
`count()`, `is_empty()`, `write()`, `iter_batches()`, the streaming path, the distributed
path, and each metadata-answered fast path that skips execution entirely — and a single
missed entry point would be a silent, total bypass. Applying the rewrite when the scan
is created means a `Dataset` never holds an ungoverned plan, so there is nothing to
bypass. It also matches how a database resolves a masking policy: at the moment the
column is read, against the role in effect then.

The same call site is where the audit record is emitted, because it is where the
decision is made — including the denial, which never produces a plan at all.
"""

from __future__ import annotations

import warnings

from batcher._internal.errors import AccessDeniedError, SecurityWarning
from batcher._internal.logging import get_logger
from batcher.api.security._context import SecurityContext, current_security
from batcher.governance import GovernanceEvent, Principal, enforce
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["govern_scan", "table_name"]

_log = get_logger("governance")

# Source identity prefixes that name no durable table: in-memory batches and live
# streams. A policy cannot be written about them because there is nothing to write it
# against before the data exists.
_EPHEMERAL = frozenset({"mem", "stream"})


def table_name(source: Source) -> str:
    """Return the name `source` is governed by: the path it is read from.

    A built-in source's `identity` is ``"<format>:<path>"``, and governance keys on the
    bare path — that is what a policy author knows *before* the table has ever been
    read, and a policy must be declarable ahead of the first read. A custom source whose
    identity carries no format prefix is governed under that identity verbatim, so an
    unrecognized naming scheme fails *closed* (governable) rather than open.

    Only the identities that name no durable table — in-memory batches and live streams —
    are ungovernable, which is honest: there is no name to write a policy about a dict
    you are already holding.

    Args:
        source: The source backing a scan.

    Returns:
        The table name, or ``""`` when the source has no durable identity.
    """
    identity_fn = getattr(source, "identity", None)
    identity = identity_fn() if callable(identity_fn) else ""
    fmt, sep, path = identity.partition(":")
    if sep:
        return "" if fmt in _EPHEMERAL else path
    return identity


def _emit(ctx: SecurityContext, event: GovernanceEvent) -> None:
    """Log the decision, then hand it to the caller's sink.

    The log line is unconditional: an audit trail that a caller can switch off by not
    passing a sink is not an audit trail. A sink that raises is not swallowed — a
    compliance pipeline that cannot record an access should stop the access.
    """
    _log.info("%s", event)
    if ctx.audit is not None:
        ctx.audit(event)


def _denial_event(principal: Principal, exc: AccessDeniedError) -> GovernanceEvent:
    """The event for a table the principal could not open at all."""
    return GovernanceEvent(
        principal=principal.name,
        roles=tuple(sorted(principal.roles)),
        table=exc.table,
        visible=(),
        denied=exc.columns,
        masked=(),
        row_filters=(),
    )


def govern_scan(plan: LogicalPlan, source: Source) -> LogicalPlan:
    """Apply the active security policy to a freshly-built single-source scan.

    Called by `batcher.api.session._scan` for every `Dataset` built from a source.
    Returns `plan` unchanged when no `security()` block is active, when the source has
    no durable name, or when the catalog declares no policy about it.

    Args:
        plan: The `Scan`-rooted plan just built for `source`.
        source: The source the scan reads.

    Returns:
        The governed plan, or `plan` itself when nothing governs it.

    Raises:
        AccessDeniedError: If the principal may select no column of the table. The
            denial is audited before it is raised.
    """
    ctx = current_security()
    if ctx is None:
        _require_governed(source, reason="no security() block is active")
        return plan
    table = table_name(source)
    if not table:
        # An in-memory table or a live stream: there is no durable name to write a policy
        # about, so it cannot be governed. Strict mode refuses it rather than exempting it,
        # which is the honest answer — silently passing it through is how an ungoverned
        # read hides inside a governed pipeline.
        _require_governed(source, reason="the source has no durable name to govern")
        return plan
    try:
        governed, events = enforce(plan, [table], ctx.principal, ctx.catalog)
    except AccessDeniedError as exc:
        _emit(ctx, _denial_event(ctx.principal, exc))
        raise
    for event in events:
        _emit(ctx, event)
    return governed


def _require_governed(source: Source, *, reason: str) -> None:
    """Enforce `governance.mode` for a read that no policy covers.

    ``off`` (the default) does nothing, so an existing deployment is untouched.
    ``strict`` refuses. ``advisory`` warns and proceeds — and that middle setting is not
    padding: it is the only way an operator can find every ungoverned read in a real
    workload *before* switching to strict. Without it strict mode cannot be adopted
    incrementally, and a security control nobody can adopt protects nobody.

    Args:
        source: The source being read, named in the message.
        reason: Why this read is ungoverned.

    Raises:
        AccessDeniedError: Under ``strict``.
    """
    from batcher.config import active_config

    mode = active_config().governance.mode
    if mode == "off":
        return
    identity = getattr(source, "identity", lambda: "<unnamed>")()
    message = f"Refusing an ungoverned read of {identity!r}: {reason}."
    if mode == "strict":
        raise AccessDeniedError(
            message,
            hint=(
                "Wrap the read in `with bt.security(catalog, principal):`, or set "
                "`governance.mode` to 'advisory' to warn instead of refusing."
            ),
        )
    warnings.warn(f"{message} (governance.mode='advisory')", SecurityWarning, stacklevel=3)
