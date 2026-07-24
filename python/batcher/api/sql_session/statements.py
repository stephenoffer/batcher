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

    Args:
        session: The session whose catalog gains the name.
        ast: The parsed ``CREATE`` statement.
        tables: Per-call table bindings visible to the body.

    Returns:
        The registered relation.

    Raises:
        PlanError: The name is taken and ``OR REPLACE`` was not given, or the statement
            has no ``AS <select>`` body.
    """
    name = ast.this.name
    if not bool(ast.args.get("replace")) and name in session._tables:
        raise PlanError(f"table {name!r} already exists; use CREATE OR REPLACE")
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
    """Handle ``DROP TABLE [IF EXISTS] name`` — unregister the table.

    Args:
        session: The session whose catalog loses the name.
        ast: The parsed ``DROP`` statement.

    Returns:
        A one-row relation naming what was dropped.

    Raises:
        PlanError: No such table, and ``IF EXISTS`` was not given.
    """
    name = ast.this.name
    if not bool(ast.args.get("exists")) and name not in session._tables:
        raise PlanError(f"no table {name!r} to drop")
    session._unbind(name)
    return session._as_dataset(pa.table({"dropped": pa.array([name], pa.string())}))
