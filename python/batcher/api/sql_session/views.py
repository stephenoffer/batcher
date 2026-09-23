"""Session views bound at query time, and the one case rule every session name follows.

A ``CREATE VIEW`` stores its **query text** (the parsed body), not the plan that text
produced when the view was created. Each query that names the view translates the body
again against the session as it is at that moment, which is what DuckDB, Postgres and Spark
all do. Storing the plan instead froze the view onto the base relations of the day it was
created: an ``INSERT INTO base`` or a re-``register`` of ``base`` never reached it, and a
``DROP TABLE base`` left the view answering from a table that no longer existed.

The names in a session — registered tables, ``CREATE TABLE AS`` results and views — share
one namespace, and it is case-insensitive the way SQL identifiers are: ``MyTab``, ``mytab``
and ``MYTAB`` are one name. Each entry keeps the spelling it was created with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.sql_session.session import Session

__all__ = ["View", "expand", "find", "lists_catalog", "referenced_views", "rename_columns"]


@dataclass(frozen=True)
class View:
    """A view's definition: its parsed body, its column aliases and any per-call bindings.

    `bindings` holds the relations the defining statement was passed as keyword tables
    (``s.sql("CREATE VIEW v AS SELECT * FROM x", x=ds)``). They are not in the session, so
    the view keeps them; everything else in the body resolves against the session at query
    time.
    """

    body: Any
    columns: tuple[str, ...] = ()
    bindings: dict[str, Any] = field(default_factory=dict)


def find(names, name: str) -> str | None:
    """The key in `names` that `name` spells, exact match first, else case-insensitively.

    Args:
        names: The keys to search.
        name: The name as written.

    Returns:
        The stored key, or None when nothing matches.
    """
    if name in names:
        return name
    folded = name.casefold()
    return next((key for key in names if key.casefold() == folded), None)


def referenced_views(session: Session, ast: Any, shadowed) -> list[str]:
    """The session views `ast` names as a table, excluding names `shadowed` covers.

    Args:
        session: The session holding the views.
        ast: A parsed statement.
        shadowed: Names that bind something else in this query — per-call tables, CTEs.

    Returns:
        The stored view keys, each once.
    """
    hidden = {n.casefold() for n in shadowed}
    hidden |= {cte.alias_or_name.casefold() for cte in ast.find_all(exp.CTE)}
    tables = list(ast.find_all(exp.Table))
    if lists_catalog(ast):
        # The catalog listing lists every view, so every view must be in the registry it
        # reads; the caller skips one that no longer plans rather than failing the listing.
        return [k for k in session._views if k.casefold() not in hidden]
    found: list[str] = []
    for node in tables:
        if not isinstance(node.this, exp.Identifier) or node.args.get("db"):
            continue
        if node.name.casefold() in hidden:
            continue
        key = find(session._views, node.name)
        if key is not None and key not in found:
            found.append(key)
    return found


def lists_catalog(ast: Any) -> bool:
    """Whether `ast` reads ``information_schema``, which lists every session name."""
    return any(
        t.args.get("db") and t.db.casefold() == "information_schema"
        for t in ast.find_all(exp.Table)
    )


def expand(session: Session, key: str, active: tuple[str, ...] = ()) -> Dataset:
    """Translate view `key` against the session as it is now.

    Args:
        session: The session the view belongs to.
        key: The stored view name.
        active: Views already being expanded above this one, to refuse a cycle.

    Returns:
        The view's lazy relation, its columns renamed by the view's alias list.

    Raises:
        PlanError: The view (indirectly) references itself.
    """
    if key in active:
        chain = " -> ".join([*active, key])
        raise PlanError(f"view {key!r} references itself ({chain})")
    view = session._views[key]
    ds = session._translate(view.body, dict(view.bindings), active=(*active, key))
    return rename_columns(ds, view.columns, key)


def rename_columns(ds: Dataset, columns: tuple[str, ...], name: str) -> Dataset:
    """Apply a ``CREATE VIEW v(a, b)`` alias list to the first `len(columns)` columns.

    Args:
        ds: The view body's relation.
        columns: The aliases, possibly fewer than the body's columns.
        name: The view name, for the error.

    Returns:
        `ds` with those columns renamed.

    Raises:
        PlanError: More aliases than the body has columns (DuckDB refuses this too).
    """
    if not columns:
        return ds
    current = list(ds.columns)
    if len(columns) > len(current):
        raise PlanError(
            f"view {name!r} names {len(columns)} columns but its query returns {len(current)}",
            hint="List at most as many column aliases as the SELECT produces.",
        )
    mapping = {old: new for old, new in zip(current, columns, strict=False) if old != new}
    return ds.rename(mapping) if mapping else ds
