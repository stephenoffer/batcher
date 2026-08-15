"""Table-level catalog figures: how many rows, and how many bytes on disk.

Both are single-scalar probes against a dialect's system catalog, and the distinction
that matters between them is exactness. A count the engine maintains transactionally
(Snowflake, BigQuery, ClickHouse base tables, SQL Server partition stats) is EXACT and may
answer a `count()`; a planner estimate that drifts between vacuums (Postgres ``reltuples``,
MySQL ``TABLE_ROWS``, Oracle ``num_rows``) informs cost only. A byte size never makes an
exactness claim at all — it sizes buffers and predicts read time, so a stale figure
mis-sizes a buffer rather than a result.
"""

from __future__ import annotations

from batcher.io.stats.sql_catalog.probes import RunScalar, _safe_scalar, _to_int
from batcher.plan.source_stats import SourceStatistics

__all__ = ["catalog_byte_size", "catalog_row_count"]


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

# dialect -> (catalog query template, exact?). `{table}` is the bare table name.
# A count the engine maintains transactionally is exact and may answer count();
# a planner estimate that drifts between ANALYZE/vacuum runs is advisory only.
_CATALOG_QUERIES: dict[str, tuple[str, bool]] = {
    "snowflake": (
        "SELECT ROW_COUNT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}'",
        True,
    ),
    "clickhouse": (
        "SELECT total_rows FROM system.tables WHERE name = '{table}'",
        True,
    ),
    "sqlserver": (
        "SELECT SUM(row_count) FROM sys.dm_db_partition_stats "
        "WHERE object_id = OBJECT_ID('{table}') AND index_id IN (0, 1)",
        True,
    ),
    "postgres": (
        "SELECT reltuples::bigint FROM pg_class WHERE relname = '{table}'",
        False,
    ),
    "redshift": (
        "SELECT tbl_rows FROM svv_table_info WHERE \"table\" = '{table}'",
        False,
    ),
    "mysql": (
        "SELECT TABLE_ROWS FROM information_schema.TABLES WHERE TABLE_NAME = '{table}'",
        False,
    ),
    "oracle": (
        "SELECT num_rows FROM all_tables WHERE table_name = '{table}'",
        False,
    ),
    "duckdb": (
        "SELECT estimated_size FROM duckdb_tables() WHERE table_name = '{table}'",
        False,
    ),
}


def catalog_row_count(run_scalar: RunScalar, dialect: str, table: str) -> SourceStatistics | None:
    """Probe a dialect's system catalog for a table's row count.

    `run_scalar(sql) -> int | None` executes a single-value query against the
    live connection (the connector supplies it). `dialect` selects the catalog
    query; `table` is the unqualified table name. Returns None on any failure.

    SQLite has no maintained row-count catalog; its ``sqlite_stat1`` (populated by
    ``ANALYZE``) records a whitespace-joined ``"<nrows> <per-index> …"`` string whose
    first token is the table's estimated row count, so it is parsed specially and
    reported advisory.
    """
    if dialect == "sqlite":
        return _sqlite_row_count(run_scalar, table)
    query = _CATALOG_QUERIES.get(dialect)
    if query is None:
        return None
    sql, exact = query
    rows = _to_int(_safe_scalar(run_scalar, sql.format(table=table)))
    if rows is None:
        return None
    return SourceStatistics(row_count=rows, exact_rows=exact)


def _sqlite_row_count(run_scalar: RunScalar, table: str) -> SourceStatistics | None:
    """SQLite's ``sqlite_stat1`` row-count estimate, or None if the table wasn't analyzed."""
    raw = _safe_scalar(run_scalar, f"SELECT stat FROM sqlite_stat1 WHERE tbl = '{table}' LIMIT 1")
    if raw is None:
        return None
    rows = _to_int(str(raw).split()[0] if str(raw).strip() else None)
    return None if rows is None else SourceStatistics(row_count=rows, exact_rows=False)


# ---------------------------------------------------------------------------
# On-disk byte size
# ---------------------------------------------------------------------------

# dialect -> catalog query template for the table's on-disk (compressed) byte size.
# `byte_size` feeds read-cost prediction and broadcast/spill sizing; it is never an
# exactness claim, so a stale figure only mis-sizes a buffer rather than a result.
_BYTE_SIZE_QUERIES: dict[str, str] = {
    "postgres": "SELECT pg_total_relation_size('{table}')",
    "redshift": "SELECT size * 1024 * 1024 FROM svv_table_info WHERE \"table\" = '{table}'",
    "mysql": (
        "SELECT DATA_LENGTH + INDEX_LENGTH FROM information_schema.TABLES "
        "WHERE TABLE_NAME = '{table}'"
    ),
    "clickhouse": (
        "SELECT sum(bytes_on_disk) FROM system.parts WHERE table = '{table}' AND active"
    ),
    "snowflake": ("SELECT BYTES FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}'"),
    # `reserved_page_count` counts 8 KB pages across every partition of the heap and its
    # indexes, which is the same "what the table occupies" figure the others report.
    "sqlserver": (
        "SELECT SUM(reserved_page_count) * 8192 FROM sys.dm_db_partition_stats "
        "WHERE object_id = OBJECT_ID('{table}')"
    ),
    # A table's storage is one or more segments; summing them covers a partitioned table.
    "oracle": "SELECT SUM(bytes) FROM user_segments WHERE segment_name = UPPER('{table}')",
    # `dbstat` is a virtual table compiled into most SQLite builds but not guaranteed to be
    # present; the query simply fails and `_safe_scalar` returns None where it is not.
    "sqlite": "SELECT SUM(pgsize) FROM dbstat WHERE name = '{table}'",
}


def catalog_byte_size(run_scalar: RunScalar, dialect: str, table: str) -> int | None:
    """The table's on-disk byte size from the catalog, or None.

    Read cost tracks bytes, not rows, so a source that can state its stored size lets
    `predicted_read_seconds` and broadcast/spill sizing reason before the read. Every
    figure is compressed on-disk size (what the reader actually transfers), best-effort.
    """
    query = _BYTE_SIZE_QUERIES.get(dialect)
    if query is None:
        return None
    size = _to_int(_safe_scalar(run_scalar, query.format(table=table)))
    return size if size and size > 0 else None
