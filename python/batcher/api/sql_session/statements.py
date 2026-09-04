"""SQL statements that change the catalog rather than only reading it.

`CREATE`, `DROP`, and the DML trio are the three forms whose effect outlives the call:
each rebinds a name in the owning `Session` and returns the relation's new lazy state.
They are free functions taking the session rather than methods, so `Session` stays the
catalog plus the query entry point and the statement semantics sit in one place a reader
can take in whole.

Every form is a plan rewrite. Nothing here materializes a row: `CREATE TABLE AS` registers
a lazy `Dataset`, and `INSERT`/`DELETE`/`UPDATE` produce a union, a filter, or a projected
`CASE` that runs only on a later terminal op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.sql_session.session import Session

__all__ = ["create", "dml", "drop"]


def create(session: Session, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
    """Handle ``CREATE [OR REPLACE] {TABLE|VIEW} name AS <select>`` — register lazily.

    Both forms register a *lazy* `Dataset`: Batcher is lazy throughout, so ``CREATE TABLE
    AS`` does not materialize, a terminal op does.

    ``IF NOT EXISTS`` keeps the existing binding and returns it, which is the whole point
    of the clause — refusing it (the old behaviour) made the idempotent spelling the one
    that failed, so a script safe to re-run in every other engine could not be re-run here.

    Args:
        session: The session whose catalog gains the name.
        ast: The parsed ``CREATE`` statement.
        tables: Per-call table bindings visible to the body.

    Returns:
        The registered relation.

    Raises:
        PlanError: The name is taken and neither ``OR REPLACE`` nor ``IF NOT EXISTS`` was
            given, or the statement has no ``AS <select>`` body.
    """
    name = ast.this.name
    if name in session._tables and not bool(ast.args.get("replace")):
        if bool(ast.args.get("exists")):
            return session._tables[name]
        raise PlanError(
            f"table {name!r} already exists; use CREATE OR REPLACE or CREATE IF NOT EXISTS"
        )
    body = ast.expression
    if body is None:
        raise PlanError("CREATE TABLE/VIEW requires an AS <select> body")
    ds = session._translate(body, tables)
    session._rebind(name, ds)
    return ds


def dml(session: Session, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
    """Handle ``INSERT`` / ``DELETE`` / ``UPDATE`` — rebind the target table.

    Per-call `tables` bindings are visible to the rewrite, but the rebind lands on the
    session catalog, matching ``CREATE``.

    Args:
        session: The session whose catalog is rebound.
        ast: The parsed DML statement.
        tables: Per-call table bindings visible to the rewrite.

    Returns:
        The target table's new lazy state.
    """
    from batcher._sql.dml import apply_dml

    registry = {name: session._as_dataset(t) for name, t in {**session._tables, **tables}.items()}
    name, new_state = apply_dml(ast, registry, session._functions)
    session._rebind(name, new_state)
    return new_state


def drop(session: Session, ast: Any) -> Dataset:
    """Handle ``DROP TABLE [IF EXISTS] name[, name...]`` — unregister the tables.

    The parser reports the targets in ``args["tables"]``, a list, because ``DROP`` takes a
    comma-separated set. Older parsers put a single target on ``ast.this`` instead and left
    that list absent, so both shapes are read: dropping one name through a parser that only
    fills the list would otherwise raise `AttributeError` on ``None``.

    Args:
        session: The session whose catalog loses the names.
        ast: The parsed ``DROP`` statement.

    Returns:
        A relation naming what was dropped, one row per name.

    Raises:
        PlanError: No such table, and ``IF EXISTS`` was not given.
    """
    targets = list(ast.args.get("tables") or ([ast.this] if ast.this is not None else []))
    names = [t.name for t in targets]
    if not names:
        raise PlanError("DROP TABLE names no table")
    if not bool(ast.args.get("exists")):
        for name in names:
            if name not in session._tables:
                raise PlanError(f"no table {name!r} to drop")
    for name in names:
        session._unbind(name)
    return session._as_dataset(pa.table({"dropped": pa.array(names, pa.string())}))
