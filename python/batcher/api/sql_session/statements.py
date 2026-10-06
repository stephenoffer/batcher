"""SQL statements that change the catalog rather than only reading it.

`CREATE`, `DROP`, and the DML trio are the three forms whose effect outlives the call:
each rebinds a name in the owning `Session` and returns the relation's new lazy state.
They are free functions taking the session rather than methods, so `Session` stays the
catalog plus the query entry point and the statement semantics sit in one place a reader
can take in whole.

Every form on a *session* name is a plan rewrite. Nothing there materializes a row: `CREATE
TABLE AS` registers a lazy `Dataset`, `CREATE VIEW` stores its query text, and
`INSERT`/`DELETE`/`UPDATE`/`MERGE` produce a union, a filter, a projected `CASE` or a
composed merge that runs only on a later terminal op. `DELETE`/`UPDATE`/`MERGE` on a
*catalog* table is the exception: a catalog table is storage, so the new contents are
collected and written back immediately.

`EXPLAIN` sits here too: it is served at the session, so the query it explains is parsed in
the session's dialect and resolves the session's views and catalog tables exactly as the
same query without `EXPLAIN` would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._internal.sql_errors import parse_sql

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.sql_session.session import Session

__all__ = ["create", "dml", "drop", "explain"]

# Statements a session performs as it translates them, so `EXPLAIN` must not run them.
_CHANGES = (
    exp.Insert,
    exp.Delete,
    exp.Update,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Use,
    exp.Command,
)


def create(session: Session, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
    """Handle ``CREATE [OR REPLACE] {TABLE|VIEW} name[(cols)] AS <select>``.

    ``CREATE TABLE AS`` registers the *lazy* `Dataset` the body translates to: Batcher is
    lazy throughout, so it does not materialize, a terminal op does. It is bound to the
    relations the body named when it was created, as a table's contents are.

    ``CREATE VIEW`` stores the body itself (`views.View`) and every later query that names
    the view translates it again, so the view reads its base tables as they are then. The
    body is translated once here as well, only to reject a view that cannot be planned at
    all, and that plan is discarded.

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
            given, the statement has no ``AS <select>`` body, or a view references itself
            or lists more column aliases than its query returns.
    """
    from batcher.api.sql_session import views

    target = ast.this
    columns: tuple[str, ...] = ()
    if isinstance(target, exp.Schema):
        columns = tuple(e.name for e in target.expressions)
        target = target.this
    name = target.name
    existing = session._key(name)
    if existing is not None and not bool(ast.args.get("replace")):
        if bool(ast.args.get("exists")):
            return session.table(existing)
        raise PlanError(
            f"table {existing!r} already exists; use CREATE OR REPLACE or CREATE IF NOT EXISTS"
        )
    body = ast.expression
    if body is None:
        raise PlanError("CREATE TABLE/VIEW requires an AS <select> body")
    if str(ast.args.get("kind") or "").upper() != "VIEW":
        ds = views.rename_columns(session._translate(body, tables), columns, name)
        session._rebind(name, ds)
        return ds
    if any(t.name.casefold() == name.casefold() for t in body.find_all(exp.Table)):
        raise PlanError(f"view {name!r} cannot reference itself")
    bound = {t: tables[t] for t in {n.name for n in body.find_all(exp.Table)} if t in tables}
    ds = views.rename_columns(session._translate(body, tables), columns, name)
    session._define_view(name, views.View(body=body.copy(), columns=columns, bindings=bound))
    return ds


def dml(session: Session, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
    """Handle ``INSERT`` / ``DELETE`` / ``UPDATE`` / ``MERGE`` — rebind the target table.

    Per-call `tables` bindings are visible to the rewrite, but the rebind lands on the
    session catalog, matching ``CREATE``. A ``DELETE``, ``UPDATE`` or ``MERGE`` on a
    *catalog* table computes the table's new contents and writes them back with
    ``mode="overwrite"``; it is the one DML form that runs immediately, and it reads the
    whole table through this process to do so.

    Args:
        session: The session whose catalog is rebound.
        ast: The parsed DML statement.
        tables: Per-call table bindings visible to the rewrite.

    Returns:
        The ``RETURNING`` rows when the statement has the clause, else the target table's
        new lazy state.

    Raises:
        PlanError: The target is a view, or a statement on a catalog table carries
            ``RETURNING`` or ``ON CONFLICT``.
    """
    from batcher._sql.dml import apply_dml
    from batcher.api.sql_session.catalog_sql import qualified_name

    target = ast.this.this if isinstance(ast.this, exp.Schema) else ast.this
    written = qualified_name(target)
    key = session._key(written) if "." not in written else None
    if key is not None and key in session._views:
        raise PlanError(
            f"{key!r} is a view; INSERT, UPDATE, DELETE and MERGE change tables",
            hint="Change the tables the view reads; the view reflects them on its next query.",
        )
    registry = {n: session._as_dataset(t) for n, t in {**session._tables, **tables}.items()}
    if key is None and written not in tables and session.catalog.has_table(written):
        return _catalog_dml(session, ast, target, written, registry)
    if key is not None and key != target.name:
        # Rename the target to the stored spelling, so the rewrite finds it (SQL names are
        # case-insensitive) and the rebind keeps the name the table was created with.
        ast = ast.copy()
        (ast.this.this if isinstance(ast.this, exp.Schema) else ast.this).set(
            "this", exp.to_identifier(key)
        )
    result = apply_dml(ast, registry, session._functions)
    session._rebind(result.name, result.state)
    return result.state if result.returning is None else result.returning


def _catalog_dml(session: Session, ast: Any, target: Any, name: str, registry: dict) -> Dataset:
    """``DELETE``/``UPDATE``/``MERGE`` on a catalog table: compute the new rows, then overwrite.

    The same rewrite a session table gets, so the statement means the same thing on both;
    only the last step differs. The new contents are collected through this process and
    written with ``mode="overwrite"``, which a catalog that cannot overwrite refuses.
    """
    from batcher._sql.dml import apply_dml
    from batcher.api.session import from_arrow

    if ast.args.get("returning"):
        raise PlanError(
            f"RETURNING on the catalog table {name!r} is not supported",
            hint="Query the table after the statement.",
        )
    key = "__bc_dml_target"
    ast = ast.copy()
    node = ast.this
    alias = node.args.get("alias") or exp.TableAlias(this=exp.to_identifier(target.name))
    for part in ("db", "catalog"):
        node.set(part, None)
    node.set("this", exp.to_identifier(key))
    node.set("alias", alias)
    result = apply_dml(ast, {**registry, key: session.table(name)}, session._functions)
    # Collected before the write: the new rows are computed *from* the table the overwrite
    # replaces, so reading lazily while writing could observe a half-replaced table.
    from_arrow(result.state.collect()).write.table(name, mode="overwrite", session=session)
    return session.table(name)


def drop(session: Session, ast: Any) -> Dataset:
    """Handle ``DROP {TABLE|VIEW} [IF EXISTS] name[, name...]`` — unregister them.

    The kind is checked the way DuckDB checks it: ``DROP VIEW`` refuses a table and
    ``DROP TABLE`` refuses a view, rather than removing whatever carries the name. Names
    are matched case-insensitively. Dropping a table a view reads is allowed; the view
    then fails when it is next queried, as it does in DuckDB.

    The parser reports the targets in ``args["tables"]``, a list, because ``DROP`` takes a
    comma-separated set. Older parsers put a single target on ``ast.this`` instead and left
    that list absent, so both shapes are read.

    Args:
        session: The session whose catalog loses the names.
        ast: The parsed ``DROP`` statement.

    Returns:
        A relation naming what was dropped, one row per name.

    Raises:
        PlanError: No such table or view and ``IF EXISTS`` was not given, or the name is
            the other kind.
    """
    from batcher.api.sql_session.catalog_sql import qualified_name

    kind = str(ast.args.get("kind") or "TABLE").upper()
    targets = list(ast.args.get("tables") or ([ast.this] if ast.this is not None else []))
    names = [qualified_name(t) for t in targets]
    if not names:
        raise PlanError(f"DROP {kind} names no {kind.lower()}")
    plan = [(name, _drop_target(session, name, kind)) for name in names]
    if not bool(ast.args.get("exists")):
        for name, found in plan:
            if found is None:
                raise PlanError(f"no {kind.lower()} {name!r} to drop")
    dropped = []
    for name, found in plan:
        if found == "catalog":
            session.catalog.drop_table(name)
        elif found is not None:
            session._unbind(found)
        if found is not None:
            dropped.append(name)
    return session._as_dataset(pa.table({"dropped": pa.array(dropped, pa.string())}))


def _drop_target(session: Session, name: str, kind: str) -> str | None:
    """What ``DROP <kind> name`` removes: a session key, ``"catalog"``, or None if nothing.

    Raises:
        PlanError: The name exists but is the other kind of object.
    """
    key = session._key(name) if "." not in name else None
    if key is not None:
        actual = "VIEW" if key in session._views else "TABLE"
    elif session.catalog.has_table(name):
        key, actual = "catalog", "TABLE"
    else:
        return None
    if actual != kind:
        raise PlanError(
            f"{name!r} is a {actual.lower()}, not a {kind.lower()}",
            hint=f"Use DROP {actual} {name}.",
        )
    return key


def explain(session: Session, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset | None:
    """Serve ``EXPLAIN [ANALYZE] <query>``, or return None when `ast` is not one.

    sqlglot does not model ``EXPLAIN``; it parses as a `Command` carrying the rest of the
    statement as text. That text is run back through the session, so it is parsed in the
    session's dialect, a syntax error in it is the same `PlanError` the bare query raises,
    and a view or catalog table in it resolves exactly as it would without ``EXPLAIN``.

    Args:
        session: The session the explained query runs against.
        ast: The parsed statement.
        tables: Per-call table bindings visible to the explained query.

    Returns:
        A one-row relation of ``explain_key``/``explain_value``, DuckDB's shape, or None.
    """
    if not (isinstance(ast, exp.Command) and str(ast.this).upper() == "EXPLAIN"):
        return None
    text = ast.expression.this if ast.expression is not None else ""
    analyze = False
    stripped = text.lstrip()
    if stripped[:8].upper() == "ANALYZE ":
        analyze, text = True, stripped[8:]
    if not text.strip():
        raise PlanError("EXPLAIN needs a query to explain", hint="Write EXPLAIN SELECT ...")
    inner = parse_sql(text, dialect=session._dialect)
    if isinstance(inner, _CHANGES):
        # Running the text would perform the change: the session executes DDL and DML as
        # it translates them. Refused rather than explained by doing it.
        raise PlanError(
            f"EXPLAIN of a {type(inner).__name__.upper()} statement is not supported; "
            "EXPLAIN explains a query",
        )
    plan = session._run(text, tables).explain(analyze=analyze)
    return session._as_dataset(pa.table({"explain_key": ["plan"], "explain_value": [plan]}))
