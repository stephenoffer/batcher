"""SQL statements that describe rather than query: ``EXPLAIN``, ``SHOW``, ``DESCRIBE``.

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
from batcher.api.dataset import Dataset
from batcher.api.session import from_arrow

__all__ = ["describing_statement"]


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
    """Translate an ``EXPLAIN [ANALYZE] <query>`` command into a plan relation."""
    import sqlglot

    text = node.args["expression"].this if node.args.get("expression") else ""
    analyze = False
    stripped = text.lstrip()
    if stripped[:8].upper() == "ANALYZE ":
        analyze, text = True, stripped[8:]
    inner = sqlglot.parse_one(text, read="duckdb")
    plan = tr.statement(inner).explain(analyze=analyze)
    return _as_dataset(pa.table({"explain_key": ["plan"], "explain_value": [plan]}))
