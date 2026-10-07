"""SQL that describes rather than queries: EXPLAIN, SHOW, DESCRIBE, information_schema.

The two statements a SQL client issues *before* it issues a query. A BI tool fills its
table picker from the first; a schema browser and a SQLAlchemy reflection read the second.
Without them a session that can answer any query at all looks empty, and it fails at the
connection rather than at anything the user typed.

Neither needs a catalog to be built. `Session` already holds ``{name: Dataset}`` and a
`Dataset` already knows its schema, so both statements read what is there and shape it as a
relation -- the same thing `translator._explain` does one branch above them.

It lives beside the translator rather than inside it for the reason `clauses` and
`from_clause` do: that file is at its size limit, and a catalog statement is its own
concern. One entry point returning None for anything it does not serve, so the caller's
"cannot translate" error still names the statement.
"""

from __future__ import annotations

import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._internal.sql_errors import parse_sql
from batcher.api.dataset import Dataset
from batcher.api.session import from_arrow

__all__ = [
    "SESSION_CATALOG",
    "SESSION_SCHEMA",
    "describing_statement",
    "information_schema_table",
    "information_schema_view",
    "refuse_unserved",
    "schemata_relation",
    "tables_relation",
    "views_relation",
]


def _as_dataset(t: pa.Table) -> Dataset:
    return from_arrow(t)


def describing_statement(tr, node) -> Dataset | None:
    """Build the relation for a describing statement, or None if `node` is not one.

    Args:
        tr: The translator, for its table registry.
        node: The sqlglot node to dispatch.

    Returns:
        The relation, or None when `node` is not a catalog statement served here.
    """
    if isinstance(node, exp.Command) and str(node.this).upper() == "EXPLAIN":
        # sqlglot does not model EXPLAIN; it parses as a Command carrying the rest of the
        # query as text. Re-parse it, render the *planned* tree (no execution), and hand it
        # back as a one-row relation, the way DuckDB's EXPLAIN reads.
        return _explain(tr, node)
    if isinstance(node, exp.Show) and str(node.name or node.this).upper() == "TABLES":
        return _show_tables(tr)
    if isinstance(node, exp.Describe):
        return _describe(tr, node)
    return None


def _show_tables(tr) -> Dataset:
    """``SHOW TABLES`` as a one-column relation, in registration order.

    Every SQL client issues this on connect, and a BI tool or a SQLAlchemy reflection
    that cannot list tables cannot show the user anything to query. The session already
    holds the answer -- `tr._registry` *is* the table list -- so this reads it rather
    than adding a catalog.

    One column named `name`, which is DuckDB's shape for the same statement, because
    the client parsing it was written against some engine and DuckDB is the one this
    dialect and its differential oracle follow.

    Returns:
        A relation with one `name` column, one row per registered table.
    """
    return _as_dataset(pa.table({"name": pa.array(list(tr._registry), pa.string())}))


def _describe(tr, node) -> Dataset:
    """``DESCRIBE <table>`` as a relation of that table's columns.

    The columns are DuckDB's: `column_name`, `column_type`, `null`, `key`, `default`,
    `extra`. The last three are always null here -- Batcher has no primary keys, column
    defaults, or storage attributes to report, and inventing values for them would be
    the kind of plausible answer that is worse than an empty one.

    **`column_type` carries Batcher's own type names, not DuckDB's.** A `DESCRIBE` on an
    integer column says `int64`, where DuckDB says `INTEGER`. That is deliberate: the
    engine stores Arrow, `Dataset.schema` reports Arrow, and rendering a DuckDB spelling
    here would tell a client something about the storage that is not true. The shape is
    borrowed; the content is this engine's.

    Args:
        node: The `exp.Describe` node naming the table.

    Returns:
        A relation with one row per column of the described table.

    Raises:
        PlanError: If the statement names no table, or names one that is not registered.
    """
    target = node.this
    name = getattr(target, "name", None) or str(target or "")
    if not name:
        raise PlanError(
            "DESCRIBE needs a table name.",
            hint="Write DESCRIBE <table>, naming a table registered on the session.",
        )
    if name not in tr._registry:
        raise PlanError(
            f"unknown table {name!r}; registered: {sorted(tr._registry)}",
            hint="Register it first with Session.register(name, dataset).",
        )
    schema = tr._registry[name].schema
    return _as_dataset(
        pa.table(
            {
                "column_name": pa.array([f.name for f in schema], pa.string()),
                "column_type": pa.array([str(f.type) for f in schema], pa.string()),
                "null": pa.array(["YES" if f.nullable else "NO" for f in schema], pa.string()),
                "key": pa.nulls(len(schema), pa.string()),
                "default": pa.nulls(len(schema), pa.string()),
                "extra": pa.nulls(len(schema), pa.string()),
            }
        )
    )


def _explain(tr, node) -> Dataset:
    """Translate an ``EXPLAIN [ANALYZE] <query>`` command into a plan relation.

    Reached only through the stateless translator: a `Session` serves ``EXPLAIN`` itself
    (`api.sql_session.statements.explain`), in its own dialect. The translator holds no
    dialect, so the inner text is read as DuckDB, the translator's default; a syntax error
    in it is a `PlanError`, as it is everywhere else.
    """
    text = node.args["expression"].this if node.args.get("expression") else ""
    analyze = False
    stripped = text.lstrip()
    if stripped[:8].upper() == "ANALYZE ":
        analyze, text = True, stripped[8:]
    inner = parse_sql(text, dialect="duckdb")
    plan = tr.statement(inner).explain(analyze=analyze)
    return _as_dataset(pa.table({"explain_key": ["plan"], "explain_value": [plan]}))


#: The `information_schema` views this serves, and the ANSI columns each carries.
#:
#: The **core** of each view rather than DuckDB's full width. DuckDB's
#: `information_schema.columns` has twelve columns and most are null for an Arrow relation
#: (`character_octet_length`, `numeric_precision_radix`, `udt_catalog`); padding them out
#: would be inventing a shape rather than reporting one. These are the columns a reflection
#: actually selects -- SQLAlchemy reads `table_name`, `column_name`, `data_type`,
#: `is_nullable`, `column_default` and `ordinal_position` -- so the subset is chosen by what
#: reads it, and `docs/api/relational/sql-statements.md` says which columns exist.
#:
#: Session tables and views are reported under this catalog and schema. They are not in any
#: attached catalog -- a session name lives as long as the session -- so they get a
#: namespace of their own, which `schemata` lists alongside the attached catalogs'.
SESSION_CATALOG = "batcher"
SESSION_SCHEMA = "main"

#: The `information_schema` views served, for the refusal that names them.
SERVED_VIEWS = ("tables", "columns", "views", "schemata")

# Registry entries that are plumbing rather than names a user bound: the per-call bindings
# `catalog_sql.bind` gives catalog tables and catalog listings.
_INTERNAL_PREFIX = "__catalog_"


def information_schema_view(node) -> str | None:
    """The lower-cased view name ``information_schema.<view>`` names, or None.

    Args:
        node: A sqlglot node.

    Returns:
        The view name, or None when `node` is not an `information_schema` table reference.
    """
    if not isinstance(node, exp.Table):
        return None
    db = node.args.get("db") or node.args.get("catalog")
    if db is None or str(getattr(db, "name", db)).lower() != "information_schema":
        return None
    return node.name.lower()


def tables_relation(entries: list[tuple[str, str]]) -> Dataset:
    """``information_schema.tables`` over ``(table_name, table_type)`` pairs.

    Args:
        entries: Each session name and its type, ``BASE TABLE`` or ``VIEW``.

    Returns:
        The relation, one row per entry.
    """
    n = len(entries)
    return _as_dataset(
        pa.table(
            {
                "table_catalog": pa.array([SESSION_CATALOG] * n, pa.string()),
                "table_schema": pa.array([SESSION_SCHEMA] * n, pa.string()),
                "table_name": pa.array([e[0] for e in entries], pa.string()),
                "table_type": pa.array([e[1] for e in entries], pa.string()),
            }
        )
    )


def views_relation(entries: list[tuple[str, str]]) -> Dataset:
    """``information_schema.views`` over ``(table_name, view_definition)`` pairs.

    Args:
        entries: Each session view and the SQL text it stores.

    Returns:
        The relation, one row per view.
    """
    n = len(entries)
    return _as_dataset(
        pa.table(
            {
                "table_catalog": pa.array([SESSION_CATALOG] * n, pa.string()),
                "table_schema": pa.array([SESSION_SCHEMA] * n, pa.string()),
                "table_name": pa.array([e[0] for e in entries], pa.string()),
                "view_definition": pa.array([e[1] for e in entries], pa.string()),
            }
        )
    )


def schemata_relation(entries: list[tuple[str, str]]) -> Dataset:
    """``information_schema.schemata`` over ``(catalog_name, schema_name)`` pairs.

    Args:
        entries: Each namespace, the session's own first.

    Returns:
        The relation, one row per namespace.
    """
    return _as_dataset(
        pa.table(
            {
                "catalog_name": pa.array([e[0] for e in entries], pa.string()),
                "schema_name": pa.array([e[1] for e in entries], pa.string()),
            }
        )
    )


def refuse_unserved(view: str) -> PlanError:
    """The error for an `information_schema` view this engine does not serve.

    Args:
        view: The view name.

    Returns:
        The error to raise.
    """
    return PlanError(
        f"information_schema.{view} is not served; {', '.join(SERVED_VIEWS)} are.",
        hint="SHOW TABLES, SHOW SCHEMAS and DESCRIBE <table> answer the same questions.",
    )


def information_schema_table(tr, node) -> Dataset | None:
    """The relation for ``information_schema.<view>``, or None if `node` is not one.

    The ANSI spelling of what `SHOW TABLES` and `DESCRIBE` answer, and the one a
    SQLAlchemy reflection and several BI tools use instead of either. It reads the same
    registry, so the three cannot disagree about what exists.

    A `Session` answers ``tables``, ``views`` and ``schemata`` before the translator runs
    (`api.sql_session.catalog_sql.bind`), because only it knows which names are views and
    which namespaces exist. What reaches here is ``columns``, which needs every relation's
    schema and so reads the registry the session built, and every view on the stateless
    path, which has no views and no catalogs to report.

    Args:
        tr: The translator, for its table registry.
        node: The table reference to inspect.

    Returns:
        The relation, or None when `node` does not name an `information_schema` view.
    """
    view = information_schema_view(node)
    if view is None:
        return None
    names = [n for n in tr._registry if not n.startswith(_INTERNAL_PREFIX)]
    if view == "tables":
        return tables_relation([(n, "BASE TABLE") for n in names])
    if view == "views":
        return views_relation([])
    if view == "schemata":
        return schemata_relation([(SESSION_CATALOG, SESSION_SCHEMA)])
    if view == "columns":
        cat, sch, tbl, col, pos, default, nullable, dtype = [], [], [], [], [], [], [], []
        for name in names:
            for i, field in enumerate(tr._registry[name].schema, start=1):
                cat.append(SESSION_CATALOG)
                sch.append(SESSION_SCHEMA)
                tbl.append(name)
                col.append(field.name)
                pos.append(i)
                default.append(None)
                nullable.append("YES" if field.nullable else "NO")
                dtype.append(str(field.type))
        return _as_dataset(
            pa.table(
                {
                    "table_catalog": pa.array(cat, pa.string()),
                    "table_schema": pa.array(sch, pa.string()),
                    "table_name": pa.array(tbl, pa.string()),
                    "column_name": pa.array(col, pa.string()),
                    "ordinal_position": pa.array(pos, pa.int64()),
                    "column_default": pa.array(default, pa.string()),
                    "is_nullable": pa.array(nullable, pa.string()),
                    "data_type": pa.array(dtype, pa.string()),
                }
            )
        )
    raise refuse_unserved(view)
