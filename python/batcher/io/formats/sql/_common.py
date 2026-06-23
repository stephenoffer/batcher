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
            f"pip install 'batcher[{extra}]'"
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
