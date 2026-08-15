"""How a connector asks a catalog a question, and which catalog it is asking.

The two things every other module here needs before it can read anything: the mapping
from a driver module name or URI scheme to the abstract *dialect* whose catalog queries
apply, and the tolerant primitives that run a query and coerce its answer. Every probe in
this package is best-effort — a missing permission, a view rather than a base table, an
un-analyzed table, a dialect mismatch — so failure is a `None` here rather than an
exception reaching the planner.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["RunRows", "RunScalar", "dialect_for_driver", "scalar_count_query"]

#: A callback the connector supplies to run a single-value catalog query.
RunScalar = Callable[[str], Any]
#: A callback the connector supplies to run a catalog query returning rows (each a
#: sequence positional to the SELECT list this package controls).
RunRows = Callable[[str], Sequence[Sequence[Any]]]


def scalar_count_query(table: str) -> str:
    """A portable ``SELECT COUNT(*)`` for an exact count via one round-trip.

    Used when no cheaper catalog estimate is available but an exact count is
    wanted; it scans server-side (cheap for columnar warehouses) and returns the
    authoritative count.
    """
    return f"SELECT COUNT(*) AS n FROM {table}"


# ---------------------------------------------------------------------------
# Driver / URI scheme -> dialect
# ---------------------------------------------------------------------------

#: Substrings of a driver module name or connection scheme -> the catalog dialect.
#:
#: Matched as substrings (longest first) so ``adbc_driver_postgresql``,
#: ``postgresql+psycopg``, and ``psycopg2`` all resolve to ``postgres`` without the
#: connector having to know the dialect itself. Order matters only where one name
#: contains another (``mariadb`` before ``maria`` is not needed, but ``postgres``
#: must beat a bare ``pg`` that could collide with unrelated tokens).
_DRIVER_DIALECTS: tuple[tuple[str, str], ...] = (
    ("postgresql", "postgres"),
    ("postgres", "postgres"),
    ("psycopg", "postgres"),
    ("redshift", "redshift"),
    ("cockroach", "postgres"),
    ("snowflake", "snowflake"),
    ("clickhouse", "clickhouse"),
    ("bigquery", "bigquery"),
    ("databricks", "spark"),
    ("mariadb", "mysql"),
    ("mysql", "mysql"),
    ("pymysql", "mysql"),
    ("sqlserver", "sqlserver"),
    ("mssql", "sqlserver"),
    ("pyodbc", "sqlserver"),
    ("oracle", "oracle"),
    ("cx_oracle", "oracle"),
    ("oracledb", "oracle"),
    ("duckdb", "duckdb"),
    ("sqlite", "sqlite"),
    ("trino", "trino"),
    ("presto", "trino"),
)


def dialect_for_driver(name: str | None) -> str | None:
    """The catalog dialect for a driver module name or connection scheme, or None.

    A connector knows *which driver* it loaded (``psycopg``, ``adbc_driver_sqlite``,
    ``clickhouse_connect``) or *which scheme* its URI carried (``postgresql://``) but
    not necessarily the abstract dialect the catalog queries key on. This maps one to
    the other by substring so every SQL connector can share one statistics path
    instead of re-deriving its own. An unrecognized driver yields None, which turns the
    whole statistics probe into a no-op rather than a wrong query.

    Args:
        name: A driver module name (``"psycopg2"``) or a URI scheme (``"postgresql"``).

    Returns:
        The dialect key (``"postgres"``, ``"mysql"``, …), or None if unrecognized.
    """
    if not name:
        return None
    lowered = name.lower()
    for token, dialect in _DRIVER_DIALECTS:
        if token in lowered:
            return dialect
    return None


# ---------------------------------------------------------------------------
# Small tolerant helpers
# ---------------------------------------------------------------------------


def _safe_scalar(run_scalar: RunScalar, sql: str) -> Any:
    """Run a scalar query, returning None on any failure (permission, dialect, etc.)."""
    try:
        return run_scalar(sql)
    except Exception:
        return None


def _at(row: Sequence[Any], index: int) -> Any:
    """`row[index]`, or None when the row is shorter than the SELECT list."""
    return row[index] if index < len(row) else None


def _to_int(value: Any) -> int | None:
    """`value` as an int, or None. A float row count is floored (a fractional estimate)."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """`value` as a float, or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
