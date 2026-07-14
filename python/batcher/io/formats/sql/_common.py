"""Shared helpers for SQL/warehouse sources — query rewriting and import guards.

Every relational backend (ADBC, ConnectorX, Snowflake, BigQuery, ClickHouse,
ODBC) shares the same control-plane shaping: a single logical query, optional
projection (rewrite the SELECT column list), optional predicate (append a
WHERE), and a deferred optional-dependency import that raises a typed
`BackendError` telling the user which extra to install. Centralizing these keeps
each backend module small and the behavior identical across backends.

Nothing here touches a row: query strings are shaped in Python, the data plane
stays Arrow-only on the worker.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from batcher._internal.errors import BackendError

__all__ = [
    "apply_predicate",
    "apply_projection",
    "push_down",
    "require_module",
    "wrap_subquery",
]


def require_module(module: str, *, extra: str) -> ModuleType:
    """Import an optional driver module, or raise a typed install hint.

    Args:
        module: The importable module name (e.g. ``"adbc_driver_manager"``).
        extra: The Batcher extra that provides it (e.g. ``"sql"``).

    Returns:
        The imported module.

    Raises:
        BackendError: If the module is not installed, with a `pip install`
            instruction for the relevant extra.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the driver
        raise BackendError(
            f"{module!r} is required for this source; install it with "
            f"pip install 'batcher-engine[{extra}]'"
        ) from exc


def wrap_subquery(query: str, *, table: str | None = None) -> str:
    """Normalize a logical read into a parenthesized derived table.

    A ``table=`` read becomes ``SELECT * FROM <table>``; an arbitrary query is
    wrapped as ``(<query>) AS _bc`` so projection/predicate can be layered on it
    uniformly without parsing the inner SQL.
    """
    inner = query if query is not None else f"SELECT * FROM {table}"
    return f"(\n{inner}\n) AS _bc"


def apply_projection(query: str, projection: list[str] | None, *, table: str | None = None) -> str:
    """Rewrite a read to select only `projection` columns.

    Returns ``SELECT <cols> FROM (<query>) AS _bc``; with no projection returns
    ``SELECT * FROM (<query>) AS _bc``. Column names are emitted verbatim — the
    caller is responsible for trusted/identifier-safe column names (they come
    from the plan's projection-pushdown, not user free-text).
    """
    cols = ", ".join(projection) if projection else "*"
    return f"SELECT {cols} FROM {wrap_subquery(query, table=table)}"


def apply_predicate(sql: str, predicate: str | None) -> str:
    """Append a SQL predicate as an outer ``WHERE`` on an already-shaped read."""
    if not predicate:
        return sql
    return f"SELECT * FROM (\n{sql}\n) AS _bcp WHERE {predicate}"


def push_down(
    query: str,
    predicate: dict | None = None,
    projection: list[str] | None = None,
    *,
    table: str | None = None,
) -> str:
    """`query` with Kyber's pushed projection and predicate folded into the SQL itself.

    This is the whole of "connect a SQL connector to the optimizer". The `WHERE` and the column
    list have to execute **in the database**, because that is the only place they can avoid
    work: a predicate applied after the result set has crossed the wire has already cost the
    scan, the network, and the driver's memory. On a TB table the difference is not a constant
    factor — it is whether the query runs at all.

    It has to happen at **split-planning** time, not at read time, for the same reason. A split
    is a picklable locator that a worker rebuilds a reader from; if the pushdown lives anywhere
    but inside the split's own query, the worker reconstructs an *unfiltered* read and the
    server never hears about the filter. That is precisely how every SQL connector here was
    behaving on the distributed path: correct results, and the entire table pulled per worker.

    A predicate the translator cannot express returns None from `to_sql_where` and is simply not
    pushed — the engine's `Filter` re-checks every row regardless, so an unpushed predicate is
    always correct and merely slower.

    Args:
        query: The base read (a user query, or `SELECT * FROM table` when `table` is given).
        predicate: The predicate IR Kyber pushed to this scan.
        projection: The columns Kyber pushed to this scan.
        table: The table name, when the read is a plain table rather than a query.

    Returns:
        The SQL to send to the server.
    """
    from batcher.io.predicate import to_sql_where

    sql = apply_projection(query, projection, table=table)
    where = to_sql_where(predicate) if predicate is not None else None
    return apply_predicate(sql, where)
