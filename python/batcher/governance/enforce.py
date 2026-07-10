"""`enforce` — rewrite a plan so a principal can only read what the catalog allows.

A pure `LogicalPlan → LogicalPlan` function. It runs *before* the optimizer, and it is
the only place governance touches a plan.

Every governed `Scan` becomes::

    Project(visible columns, masked)     <- what the principal may read
      Filter(row-access predicate)       <- which rows it may see
        Scan(table)                      <- the raw table

Two properties follow from that shape, and both are load-bearing:

* **The filter sits below the projection**, so a row-access predicate may reference
  columns the principal cannot select. A policy runs with the catalog's authority.
* **The projection sits at the leaf**, so masking is applied before any filter, join,
  aggregate, or sort the *user* wrote. A principal cannot recover a masked value by
  filtering on it, grouping by it, or joining against it — the raw value never exists
  above the scan.

A column the principal may not select is *removed from the scan's output*, not flagged.
Any later reference to it fails as an unknown column, which is both fail-closed and the
right disclosure boundary: an error saying "you may not read `salary`" would confirm
that `salary` exists. Losing access to every column is different — there is nothing to
return and nothing to leak — so that raises `AccessDeniedError` directly.

The rewrite also emits a `GovernanceEvent` per governed table. It comes out of the same
traversal that builds the plan, so an audit log cannot drift from what was enforced.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

from batcher._internal.errors import AccessDeniedError
from batcher.governance.audit import GovernanceEvent
from batcher.governance.catalog import SecurityCatalog
from batcher.governance.principal import Principal
from batcher.plan.expr_ir import Col, Expr
from batcher.plan.logical import Filter, LogicalPlan, Project, Projection, Scan
from batcher.plan.visitor import transform_up

__all__ = ["enforce"]


def enforce(
    plan: LogicalPlan,
    tables: Sequence[str],
    principal: Principal,
    catalog: SecurityCatalog,
) -> tuple[LogicalPlan, tuple[GovernanceEvent, ...]]:
    """Rewrite `plan` so `principal` sees only what `catalog` permits, and say what it did.

    The governed plan and the audit record come out of one traversal, so what is logged
    is by construction what was enforced. A denial raises before any event is returned;
    the exception carries the same facts.

    Args:
        plan: The plan to govern.
        tables: The table name of each source, indexed by a `Scan`'s ``source_id``.
        principal: The identity the query runs as.
        catalog: The declared policies.

    Returns:
        A pair of the governed plan and one `GovernanceEvent` per governed table. The
        plan is the *same object* when no scan in it is governed, so installing a
        catalog leaves unrelated queries structurally untouched.

    Raises:
        AccessDeniedError: If the principal may select no column of a governed table.
    """
    events: list[GovernanceEvent] = []

    def govern(node: LogicalPlan) -> LogicalPlan:
        if not isinstance(node, Scan):
            return node
        table = tables[node.source_id] if node.source_id < len(tables) else ""
        if not table or not catalog.governs(table):
            return node
        governed, event = _govern_scan(node, table, principal, catalog)
        events.append(event)
        return governed

    return transform_up(plan, govern), tuple(events)


def _event(
    principal: Principal,
    table: str,
    visible: Sequence[str],
    columns: Sequence[str],
    masked: Sequence[str] = (),
    row_filters: Sequence[str] = (),
) -> GovernanceEvent:
    return GovernanceEvent(
        principal=principal.name,
        roles=tuple(sorted(principal.roles)),
        table=table,
        visible=tuple(visible),
        denied=tuple(c for c in columns if c not in set(visible)),
        masked=tuple(masked),
        row_filters=tuple(row_filters),
    )


def _govern_scan(
    scan: Scan, table: str, principal: Principal, catalog: SecurityCatalog
) -> tuple[LogicalPlan, GovernanceEvent]:
    """Wrap one governed `Scan` in its row filter and its masked projection."""
    columns = scan.available_columns()
    visible = catalog.visible_columns(table, columns, principal)
    if not visible:
        raise AccessDeniedError(
            f"principal {principal.name!r} may not read table {table!r}: "
            f"no column is granted to roles {sorted(principal.roles)}",
            table=table,
            columns=tuple(columns),
        )

    governed: LogicalPlan = scan
    filters = catalog.row_filters_for(table, principal)
    if filters:
        # Conjoined: every applicable filter must hold. Built over the raw scan, so a
        # predicate may reference columns pruned by the projection above.
        predicate = reduce(lambda a, b: a & b, (f.predicate(principal) for f in filters))
        governed = Filter(governed, predicate)

    masks = {c: catalog.mask_for(table, c, principal) for c in visible}
    event = _event(
        principal,
        table,
        visible,
        columns,
        masked=[c for c, m in masks.items() if m is not None],
        row_filters=[f.name for f in filters],
    )
    if visible == columns and not any(masks.values()):
        # Nothing to prune and nothing to mask: skip the projection entirely rather than
        # inserting an identity `Project` the optimizer would have to see through.
        return governed, event

    items = tuple(Projection(alias=c, expr=_masked(c, masks[c])) for c in visible)
    return Project(governed, items), event


def _masked(column: str, mask) -> Expr:
    """The expression to read `column` through: the mask applied to it, or the column."""
    col = Col(column)
    return col if mask is None else mask(col)
