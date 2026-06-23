"""Catalog-derived statistics for SQL warehouses.

Every SQL warehouse maintains table statistics in a system catalog that answers
"how many rows" without scanning: Snowflake `INFORMATION_SCHEMA.TABLES`,
BigQuery `__TABLES__`, ClickHouse `system.tables`, Postgres `pg_class.reltuples`,
MySQL `information_schema.TABLES`. A single catalog query gives the planner a row
count for free.

Counts that the engine maintains exactly (Snowflake/BigQuery/ClickHouse base
tables) are `exact_rows=True`; planner estimates that drift between vacuums
(Postgres `reltuples`, MySQL `TABLE_ROWS`) are `exact_rows=False` — they inform
cost but never answer `count()`. Every probe is best-effort: a failure (no
permission, view not a base table, dialect mismatch) yields None and the planner
falls back to its defaults.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.source_stats import SourceStatistics

__all__ = ["catalog_row_count", "scalar_count_query"]


def scalar_count_query(table: str) -> str:
    """A portable ``SELECT COUNT(*)`` for an exact count via one round-trip.

    Used when no cheaper catalog estimate is available but an exact count is
    wanted; it scans server-side (cheap for columnar warehouses) and returns the
    authoritative count.
    """
    return f"SELECT COUNT(*) AS n FROM {table}"


def catalog_row_count(run_scalar: Any, dialect: str, table: str) -> SourceStatistics | None:
    """Probe a dialect's system catalog for a table's row count.

    `run_scalar(sql) -> int | None` executes a single-value query against the
    live connection (the connector supplies it). `dialect` selects the catalog
    query; `table` is the unqualified table name. Returns None on any failure.
    """
    query = _CATALOG_QUERIES.get(dialect)
    if query is None:
        return None
    sql, exact = query
    try:
        value = run_scalar(sql.format(table=table))
    except Exception:
        return None
    if value is None:
        return None
    try:
        rows = int(value)
    except (TypeError, ValueError):
        return None
    return SourceStatistics(row_count=rows, exact_rows=exact)


# dialect -> (catalog query template, exact?). `{table}` is the bare table name.
_CATALOG_QUERIES: dict[str, tuple[str, bool]] = {
    "snowflake": (
        "SELECT ROW_COUNT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}'",
        True,
    ),
    "clickhouse": (
        "SELECT total_rows FROM system.tables WHERE name = '{table}'",
        True,
    ),
    "postgres": (
        "SELECT reltuples::bigint FROM pg_class WHERE relname = '{table}'",
        False,
    ),
    "mysql": (
        "SELECT TABLE_ROWS FROM information_schema.TABLES WHERE TABLE_NAME = '{table}'",
        False,
    ),
}
