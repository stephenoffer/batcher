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

import re
import warnings

from batcher._internal.errors import AccessDeniedError, SecurityWarning
from batcher._internal.logging import get_logger
from batcher.api.security._context import SecurityContext, current_security
from batcher.governance import GovernanceEvent, Principal, enforce
from batcher.governance.audit_log import record_governance_event
from batcher.io.filesystem import canonical_path
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["emit_event", "govern_scan", "table_name"]

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

    **A source that can name its table says so itself** (`governed_name`), and that is
    asked first. `identity` names a *relation*, not a table: it carries a digest of any
    subset the source is pinned to, so that a capped or file-pinned read does not inherit
    the whole table's cached statistics. Reading the table name off it made an ordinary
    keyword argument a total governance bypass — under a catalog masking `email` and
    withholding `ssn`, `read.parquet(path)` returned the mask and no `ssn` while
    `read.parquet(path, n_rows=2)` returned the raw address and the whole `ssn` column,
    because `<path>#<digest>` is a name no policy mentions and `catalog.governs()` was
    therefore False. `columns=[...]` did the same, and so did the file list a pruned
    MERGE reads its target through.

    Args:
        source: The source backing a scan.

    Returns:
        The table name, or ``""`` when the source has no durable identity.
    """
    named = getattr(source, "governed_name", None)
    if callable(named):
        return named()
    identity_fn = getattr(source, "identity", None)
    identity = identity_fn() if callable(identity_fn) else ""
    fmt, sep, path = identity.partition(":")
    if sep:
        return "" if fmt in _EPHEMERAL else path
    return identity


def emit_event(ctx: SecurityContext, event: GovernanceEvent) -> None:
    """Log the decision, then hand it to the caller's sink.

    The log line is unconditional: an audit trail that a caller can switch off by not
    passing a sink is not an audit trail. A sink that raises is not swallowed — a
    compliance pipeline that cannot record an access should stop the access.

    Three sinks, in widening order of durability: the log, the configured
    `governance.audit_path` file, and the caller's own callback. The file comes before the
    callback so a caller-supplied sink that raises still leaves the durable record behind —
    the opposite order loses exactly the events a failing pipeline most needs to explain.
    """
    from batcher.config import active_config

    _log.info("%s", event)
    record_governance_event(event, active_config().governance.audit_path)
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


def _nearest_governed(tables: frozenset[str], path: str) -> str:
    """The closest table at or above `path` that is in `tables`, or ``""``.

    Walks upward because a policy names a *table* and a pinned path names a *file* inside
    it: a rule on ``/data/orders`` has to cover ``/data/orders/dt=2024/part-0.parquet``,
    which is two levels down and shares no exact name with it.

    Takes the governed set rather than the catalog so the caller can build it once. Asking
    the catalog per path rescans every policy list, which over a source pinned to thousands
    of files is quadratic work at plan time.
    """
    current = canonical_path(path)
    while current:
        if current in tables:
            return current
        parent = current.rsplit("/", 1)[0] if "/" in current else ""
        # `s3://bucket` splits to `s3:/`, and a local path bottoms out at `""`; both are
        # the end of the walk rather than another table to test.
        if parent == current or parent.endswith(":/") or not parent:
            return ""
        current = parent
    return ""


def _refuse_a_split_policy(ctx: SecurityContext, source: Source, table: str) -> None:
    """Refuse a read whose individual files are governed by something else.

    A source pinned to explicit paths — ``read.parquet([a, b])`` — is modelled as their
    **common parent** plus a file list, so its table name is that parent. When the parent is
    not itself a governed table, the policies on the files underneath it were simply not
    consulted: reading ``[secret.parquet, other.parquet]`` as one relation returned every
    column of `secret.parquet`, including the ones a grant withheld, because their shared
    directory is a name nobody wrote a policy about.

    Governing it properly means resolving several policies into one scan, which `enforce`
    is not shaped for: it takes one table per scan and applies that table's masks and row
    filters. Refusing is the answer that cannot be wrong — the caller reads the tables
    separately and the engine governs each — and it is narrow, because it fires only when a
    file the read pins lies under a policy the scan is not already being governed by.

    Raises:
        AccessDeniedError: If a pinned path is governed by a table other than `table`.
    """
    pinned = getattr(source, "_pinned", None)
    if not pinned:
        return
    # The governed set is built once and each path is then a set membership per level.
    # Deduplicating by *directory* would be faster still and is wrong: a policy may name an
    # individual file, and two files with different policies sit in one directory. So every
    # pinned path is tested, and it is the per-test cost that is made O(1) instead.
    tables = ctx.catalog._governed_tables()
    if not tables:
        return
    others = sorted(
        {
            found
            for found in (_nearest_governed(tables, p) for p in pinned)
            if found and found != table
        }
    )
    if not others:
        return
    raise AccessDeniedError(
        f"Refusing to read {len(pinned)} explicit path(s) as one relation: "
        f"{', '.join(repr(o) for o in others)} carries a policy that this read would not "
        "apply, because the paths are governed as their shared parent "
        f"{table or '<unnamed>'!r}.",
        table=table,
        hint="Read each governed table separately, so each is governed by its own policy.",
    )


def govern_scan(
    plan: LogicalPlan, source: Source, *, governed_as: str | None = None
) -> LogicalPlan:
    """Apply the active security policy to a freshly-built single-source scan.

    Called by `batcher.api.session._scan` for every `Dataset` built from a source.
    Returns `plan` unchanged when no `security()` block is active, when the source has
    no durable name, or when the catalog declares no policy about it.

    `governed_as` is the name a caller *declares* for a source that cannot name its own
    table: a query-based warehouse read, or an in-memory table. The declared name is then
    what the catalog is asked about, so a policy on ``analytics.public.users`` governs a
    ``SELECT ... FROM users`` read the caller declared as that table. A source that already
    names its table cannot be re-declared under a different name, since that would be a
    one-keyword way out of its policy.

    Args:
        plan: The `Scan`-rooted plan just built for `source`.
        source: The source the scan reads.
        governed_as: The table name to govern an unnamed source by.

    Returns:
        The governed plan, or `plan` itself when nothing governs it.

    Raises:
        AccessDeniedError: If the principal may select no column of the table, if
            `governed_as` contradicts the source's own name, or if an undeclared query
            reads a table the active catalog governs. A denial is audited before it is
            raised.
    """
    ctx = current_security()
    if ctx is None:
        if governed_as is not None:
            _declared_name(source, governed_as)  # validate the declaration regardless
        _require_governed(source, reason="no security() block is active")
        return plan
    table = _declared_name(source, governed_as)
    if not table:
        _refuse_an_undeclared_governed_query(ctx, source)
        # An in-memory table or a live stream: there is no durable name to write a policy
        # about, so it cannot be governed. Strict mode refuses it rather than exempting it,
        # which is the honest answer — silently passing it through is how an ungoverned
        # read hides inside a governed pipeline.
        _require_governed(source, reason="the source has no durable name to govern")
        return plan
    if governed_as is not None:
        _refuse_a_misspelled_declaration(ctx, table)
    _refuse_a_split_policy(ctx, source, table)
    try:
        governed, events = enforce(plan, [table], ctx.principal, ctx.catalog)
    except AccessDeniedError as exc:
        emit_event(ctx, _denial_event(ctx.principal, exc))
        raise
    for event in events:
        emit_event(ctx, event)
    return governed


def _declared_name(source: Source, governed_as: str | None) -> str:
    """The name `source` is governed by: its own, or the one the caller declared for it.

    Raises:
        AccessDeniedError: If `source` names its own table and `governed_as` names another.
    """
    own = table_name(source)
    if governed_as is None:
        return own
    if not isinstance(governed_as, str) or not governed_as.strip():
        raise AccessDeniedError(
            f"governed_as must be a non-empty table name, got {governed_as!r}.",
            hint="Pass the name the governing policy is written against.",
        )
    if own and canonical_path(own) != canonical_path(governed_as):
        raise AccessDeniedError(
            f"Refusing governed_as={governed_as!r}: this source already names its table "
            f"{own!r}, and governance keys on that name.",
            table=own,
            hint="Drop governed_as; it is for sources that name no table of their own.",
        )
    return governed_as


def _sql_names(name: str) -> tuple[str, ...]:
    """The spellings of governed table `name` that a SQL query could reference it by.

    A warehouse name is ``db.schema.table`` in a policy and very often plain ``table`` in a
    query, because the connection's default database and schema supply the rest. So both
    the full name and its last component are candidates. Quoting (`"`, backtick, brackets)
    is stripped because it does not change which table is named.
    """
    bare = re.sub(r'["`\[\]]', "", name).lower()
    if "/" in bare:
        # A path. A query reaches it only by writing it out, e.g. DuckDB's read_parquet.
        return (bare,)
    last = bare.rsplit(".", 1)[-1]
    return (bare,) if last == bare else (bare, last)


def _referenced_governed_tables(query: str, tables: frozenset[str]) -> list[str]:
    """The governed tables whose name appears as a whole identifier in `query`.

    Deliberately over-inclusive: it is used only to *refuse*, and a false positive costs a
    caller one ``governed_as=`` keyword while a false negative costs an ungoverned read.
    """
    text = re.sub(r'["`\[\]]', "", query).lower()
    found = []
    for table in sorted(tables):
        for spelling in _sql_names(table):
            if re.search(rf"(?<!\w){re.escape(spelling)}(?!\w)", text):
                found.append(table)
                break
    return found


def _refuse_an_undeclared_governed_query(ctx: SecurityContext, source: Source) -> None:
    """Refuse a query read that references a governed table without declaring which it is.

    A query-based warehouse read (Snowflake, a BigQuery or SQL ``query=``) names no table,
    so a policy keyed on a table name cannot match it. Inside a `security()` block that
    meant ``SELECT * FROM users`` returned every column of a table whose policy masked half
    of them, and nothing said so: the catalog was never asked, because there was no name
    to ask about. Declaring the name (``governed_as=``) is what applies the policy; this
    check is what stops the undeclared read from passing silently when the query plainly
    touches a table the catalog protects.

    Raises:
        AccessDeniedError: If the query references a table the active catalog governs.
    """
    query = getattr(source, "query", None)
    if not isinstance(query, str) or not query:
        return
    tables = ctx.catalog._governed_tables()
    if not tables:
        return
    hits = _referenced_governed_tables(query, tables)
    if not hits:
        return
    raise AccessDeniedError(
        f"Refusing an undeclared query read inside security(): the query references "
        f"{', '.join(repr(h) for h in hits)}, which carries a policy that a query read "
        "cannot match, because a query names no table.",
        table=hits[0],
        hint=(
            "Pass governed_as='<table>' naming the table the query reads, so its policy "
            "is applied to the result, or read the table with table= where the connector "
            "supports it."
        ),
    )


def _refuse_a_misspelled_declaration(ctx: SecurityContext, table: str) -> None:
    """Refuse a declared name that matches a governed table in everything but case.

    Governed names keep their case (an S3 key is case-sensitive), while a warehouse
    identifier usually is not. A caller who declares ``analytics.users`` against a policy
    on ``ANALYTICS.USERS`` would otherwise get the policy of neither and no error at all.

    Raises:
        AccessDeniedError: If `table` is not governed but a case-variant of it is.
    """
    tables = ctx.catalog._governed_tables()
    canonical = canonical_path(table)
    if canonical in tables:
        return
    variants = sorted(t for t in tables if t.lower() == canonical.lower())
    if variants:
        raise AccessDeniedError(
            f"Refusing governed_as={table!r}: no policy names it, but {variants[0]!r} "
            "differs from it only in case, so the declared read would be ungoverned.",
            table=table,
            hint=f"Declare governed_as={variants[0]!r}, the spelling the policy uses.",
        )


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
