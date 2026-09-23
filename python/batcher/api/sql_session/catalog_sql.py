"""SQL over a session's catalogs: ``USE``, ``SHOW``, schema DDL, and catalog table references.

Two entry points, both called by `Session` before the SELECT translator sees a statement:

`catalog_statement`
    Serves the statements that are about the catalog itself — ``USE``, ``SHOW TABLES``,
    ``SHOW DATABASES``, ``SHOW SCHEMAS``, ``CREATE``/``DROP SCHEMA``, and ``CREATE TABLE
    AS`` / ``INSERT INTO`` a *catalog* table — and returns None for everything else, so the
    existing session-view statements keep their exact behaviour.
`bind`
    Rewrites a query's table references that name catalog tables into per-call bindings,
    and inlines ``current_catalog()``, ``current_schema()``, ``current_database()`` and
    ``current_user`` as literals. The translator then sees an ordinary query over bound
    names and needs no knowledge of catalogs.

The result shapes follow DuckDB, the dialect and oracle the SQL front-end is measured
against: ``SHOW TABLES`` has one ``name`` column, ``SHOW DATABASES`` a ``database_name``
column, and ``current_database()`` is the *catalog* name. Under the Spark dialect
``current_database()`` is the namespace instead, which is what Spark means by it.
"""

from __future__ import annotations

import getpass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.sql_session.session import Session

__all__ = ["bind", "catalog_statement", "qualified_name"]

_SESSION_FUNCTIONS = (
    exp.CurrentCatalog,
    exp.CurrentDatabase,
    exp.CurrentSchema,
    exp.CurrentUser,
    exp.SessionUser,
)


def qualified_name(node: exp.Expression) -> str:
    """The dotted name a table reference spells (``catalog.db.name``, as far as written).

    Args:
        node: An `exp.Table`, or an `exp.Schema` wrapping one.

    Returns:
        The dotted name.
    """
    if isinstance(node, exp.Schema):
        node = node.this
    return ".".join(part.name for part in node.parts)


def _relation(columns: dict[str, pa.Array]) -> Dataset:
    from batcher.api.session.frames import from_arrow

    return from_arrow(pa.table(columns))


def catalog_statement(session: Session, ast: Any, tables: dict[str, Any]) -> Dataset | None:
    """Serve a catalog statement, or return None when `ast` is not one.

    Args:
        session: The session whose catalogs the statement reads or changes.
        ast: The parsed statement.
        tables: Per-call bindings, visible to a ``CREATE TABLE AS``/``INSERT`` body.

    Returns:
        The statement's result relation, or None.
    """
    if isinstance(ast, exp.Use):
        session.catalog.use(qualified_name(ast.this))
        return _position(session)
    show = _show_kind(ast)
    if show is not None:
        return _show(session, tables, *show)
    kind = str(ast.args.get("kind") or "").upper()
    if isinstance(ast, exp.Create) and kind == "SCHEMA":
        name = qualified_name(ast.this)
        session.catalog.create_namespace(name, if_not_exists=bool(ast.args.get("exists")))
        return _relation({"namespace": pa.array([name], pa.string())})
    if isinstance(ast, exp.Drop) and kind == "SCHEMA":
        names = [qualified_name(t) for t in ast.args.get("tables") or [ast.this]]
        for name in names:
            session.catalog.drop_namespace(
                name, if_exists=bool(ast.args.get("exists")), cascade=bool(ast.args.get("cascade"))
            )
        return _relation({"dropped": pa.array(names, pa.string())})
    if isinstance(ast, exp.Create) and kind == "TABLE" and _creates_catalog_table(session, ast):
        return _create_table(session, ast, tables)
    if isinstance(ast, exp.Insert) and _is_catalog_target(session, ast.this, tables):
        return _insert(session, ast, tables)
    return None


def _creates_catalog_table(session: Session, ast: Any) -> bool:
    """Whether ``CREATE TABLE name AS …`` writes a catalog table rather than a session one.

    A qualified name always does. An unqualified one does once ``USE`` has moved the session
    off its starting ``memory.main``: after ``USE wh.raw`` the user means ``wh.raw.name``,
    exactly as DuckDB resolves it, and quietly binding a session-only name instead made the
    table vanish with the process. Without a ``USE``, an unqualified ``CREATE TABLE AS``
    keeps its long-standing meaning, a lazy session table.
    """
    return "." in qualified_name(ast.this) or session.catalog._is_repositioned()


def _position(session: Session) -> Dataset:
    return _relation(
        {
            "catalog": pa.array([session.catalog.current_catalog()], pa.string()),
            "namespace": pa.array([session.catalog.current_namespace()], pa.string()),
        }
    )


def _show_kind(ast: Any) -> tuple[str, str | None] | None:
    """``("TABLES", "ns")``-style reading of a SHOW statement, or None."""
    if isinstance(ast, exp.Show):
        source = ast.args.get("from_") or ast.args.get("from")
        what = str(ast.name or ast.this).upper()
        return what, (qualified_name(source) if source is not None else None)
    if isinstance(ast, exp.Command) and str(ast.this).upper() == "SHOW":
        words = str(getattr(ast.expression, "this", ast.expression) or "").split()
        if not words:
            return None
        source = words[2] if len(words) == 3 and words[1].upper() in ("FROM", "IN") else None
        return words[0].upper(), source
    return None


def _show(session: Session, tables: dict[str, Any], what: str, source: str | None) -> Dataset:
    catalog = session.catalog
    if what == "TABLES":
        if source is None:
            names = list(dict.fromkeys([*session._tables, *session._views, *tables]))
            namespace = catalog.current_namespace()
            owner = catalog.get_catalog(catalog.current_catalog())
        else:
            names = []
            owner, namespace = catalog._resolve_namespace(source)
        prefix = f"{namespace}."
        for table in owner.list_tables():
            if table.startswith(prefix) and table[len(prefix) :] not in names:
                names.append(table[len(prefix) :])
        return _relation({"name": pa.array(names, pa.string())})
    if what == "DATABASES":
        return _relation({"database_name": pa.array(catalog.list_catalogs(), pa.string())})
    if what in ("SCHEMAS", "NAMESPACES"):
        rows = [
            (name, namespace)
            for name in catalog.list_catalogs()
            for namespace in catalog.get_catalog(name).list_namespaces()
        ]
        current = (catalog.current_catalog(), catalog.current_namespace())
        return _relation(
            {
                "database_name": pa.array([r[0] for r in rows], pa.string()),
                "schema_name": pa.array([r[1] for r in rows], pa.string()),
                "current": pa.array([r == current for r in rows], pa.bool_()),
            }
        )
    raise PlanError(
        f"SHOW {what} is not supported",
        hint="SHOW TABLES, SHOW DATABASES and SHOW SCHEMAS are.",
    )


def _is_catalog_target(session: Session, target: Any, tables: dict[str, Any]) -> bool:
    name = qualified_name(target)
    if name in tables or ("." not in name and session._key(name) is not None):
        return False
    return session.catalog.has_table(name)


def _create_table(session: Session, ast: Any, tables: dict[str, Any]) -> Dataset:
    """``CREATE [OR REPLACE] TABLE [IF NOT EXISTS] ns.t AS <select>`` into a catalog."""
    name = qualified_name(ast.this)
    if ast.expression is None or isinstance(ast.this, exp.Schema):
        raise PlanError(
            f"CREATE TABLE {name} needs an AS <select> body; a column-list definition is not "
            "supported for catalog tables",
            hint="Use session.catalog.create_table(name, pyarrow_schema) for an empty table.",
        )
    if ast.args.get("exists") and session.catalog.has_table(name):
        return session.table(name)
    body = session._translate(ast.expression, tables)
    mode = "overwrite" if ast.args.get("replace") else "error"
    body.write.table(name, mode=mode, session=session)
    return session.table(name)


def _insert(session: Session, ast: Any, tables: dict[str, Any]) -> Dataset:
    """``INSERT INTO ns.t [(cols)] <select|values>`` appended to a catalog table."""
    from batcher._sql.dml import align_insert

    for unsupported in ("conflict", "returning", "overwrite"):
        if ast.args.get(unsupported):
            raise PlanError(
                f"INSERT ... {unsupported.upper()} into a catalog table is not supported"
            )
    if ast.expression is None:
        raise PlanError("INSERT requires a VALUES or SELECT body")
    name = qualified_name(ast.this)
    columns = [c.name for c in ast.this.expressions] if isinstance(ast.this, exp.Schema) else None
    current = session.table(name)
    rows = align_insert(name, current, session._translate(ast.expression, tables), columns)
    rows.write.table(name, mode="append", session=session)
    return session.table(name)


def bind(session: Session, ast: Any, tables: dict[str, Any]) -> tuple[Any, dict[str, Any], bool]:
    """Bind a query's catalog table references and inline session functions.

    Args:
        session: The session whose catalogs resolve the references.
        ast: The parsed query.
        tables: Per-call bindings, which shadow catalog tables as session views do.

    Returns:
        ``(ast, bindings, dynamic)``: the (copied, when changed) AST, the extra table
        bindings it now names, and whether the plan depends on session state and so must
        not be served from the prepared-statement cache.

    Raises:
        PlanError: A qualified reference names no catalog table.
    """
    if not _references(session, ast, tables) and not any(ast.find_all(*_SESSION_FUNCTIONS)):
        return ast, {}, False
    ast = ast.copy()
    bindings: dict[str, Any] = {}
    for node, name in _references(session, ast, tables):
        key = f"__catalog_{len(bindings)}"
        bindings[key] = session.catalog.get_table(name).read()
        alias = node.args.get("alias") or exp.TableAlias(this=exp.to_identifier(node.name))
        node.set("this", exp.to_identifier(key))
        node.set("db", None)
        node.set("catalog", None)
        node.set("alias", alias)
    for node in list(ast.find_all(*_SESSION_FUNCTIONS)):
        _inline(session, node)
    return ast, bindings, True


def _references(session: Session, ast: Any, tables: dict[str, Any]) -> list[tuple[Any, str]]:
    """The ``(table node, dotted name)`` pairs in `ast` that name catalog tables."""
    shadowing = {name.lower() for name in [*session._tables, *session._views, *tables]}
    shadowing |= {cte.alias_or_name.lower() for cte in ast.find_all(exp.CTE)}
    found = []
    for node in ast.find_all(exp.Table):
        if not isinstance(node.this, exp.Identifier):
            continue
        parts = [part.name for part in node.parts]
        if len(parts) == 1 and parts[0].lower() in shadowing:
            continue
        if len(parts) > 1 and parts[-2].lower() == "information_schema":
            continue
        name = ".".join(parts)
        if session.catalog.has_table(name):
            found.append((node, name))
        elif len(parts) > 1:
            raise PlanError(
                f"no catalog table {name!r}",
                available=session.catalog.list_tables(),
                available_label="Tables in the current catalog",
            )
    return found


def _inline(session: Session, node: Any) -> None:
    """Replace one session function call with its value, keeping DuckDB's column name."""
    catalog = session.catalog
    if isinstance(node, exp.CurrentCatalog):
        value, label = catalog.current_catalog(), "current_catalog()"
    elif isinstance(node, exp.CurrentSchema):
        value, label = catalog.current_namespace(), "current_schema()"
    elif isinstance(node, exp.CurrentDatabase):
        spark_like = session._dialect in ("spark", "spark2", "databricks", "hive")
        value = catalog.current_namespace() if spark_like else catalog.current_catalog()
        label = "current_database()"
    else:
        value = getpass.getuser()
        label = "current_user" if isinstance(node, exp.CurrentUser) else "session_user"
    literal = exp.Literal.string(value)
    if isinstance(node.parent, exp.Select) and node.arg_key == "expressions":
        node.replace(exp.alias_(literal, label, quoted=True))
    else:
        node.replace(literal)
