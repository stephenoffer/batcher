"""Catalog-derived statistics for SQL warehouses and databases.

Every SQL engine maintains table statistics in a system catalog that answers
"how many rows", "how many bytes", and often "how selective is this column"
*without scanning the data*. A single metadata query gives the planner what it
would otherwise pay a full ``COUNT(*)`` (or worse) to learn:

  - **Row counts** — Snowflake ``INFORMATION_SCHEMA.TABLES``, BigQuery
    ``__TABLES__``, ClickHouse ``system.tables``, Postgres ``pg_class.reltuples``,
    MySQL ``information_schema.TABLES``, SQL Server ``sys.dm_db_partition_stats``,
    Oracle ``all_tables``, Redshift ``svv_table_info``, DuckDB ``duckdb_tables()``,
    SQLite ``sqlite_stat1``.
  - **On-disk size** — Postgres ``pg_total_relation_size``, ClickHouse
    ``system.parts``, MySQL ``DATA_LENGTH + INDEX_LENGTH``, Snowflake ``BYTES``.
  - **Per-column stats** — Postgres ``pg_stats`` records, per column, the null
    fraction, distinct-value estimate, most-common values and their frequencies,
    and a histogram of bounds. These map straight onto `ColumnStat`, so a Postgres
    table reaches Kyber with null counts, distinct estimates, MCV-sharpened
    equality selectivity, and quantile-interpolated range selectivity — the same
    facets a Parquet footer supplies, for a database that has no footer.

Counts the engine maintains exactly (Snowflake/BigQuery/ClickHouse base tables,
SQL Server partition stats) are `exact_rows=True`; planner estimates that drift
between vacuums/analyzes (Postgres ``reltuples``, MySQL ``TABLE_ROWS``, Oracle
``num_rows``, Redshift ``tbl_rows``, SQLite ``sqlite_stat1``) are
`exact_rows=False` — they inform cost but never answer an exact ``count()``.
Every column statistic a catalog gives is a *sampled estimate* (``ANALYZE`` reads
a sample), so it is tagged `SKETCH`/`HISTOGRAM` and likewise never answers an
exact query — only cost and cardinality.

Every probe is best-effort: a failure (no permission, a view rather than a base
table, a dialect mismatch, an un-analyzed table) yields None/empty and the
planner falls back to its defaults. Nothing here touches a row — the queries read
the catalog, and the callback that runs them belongs to the connector.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = [
    "catalog_byte_size",
    "catalog_column_stats",
    "catalog_row_count",
    "dialect_for_driver",
    "scalar_count_query",
    "sql_statistics",
]

#: A callback the connector supplies to run a single-value catalog query.
RunScalar = Callable[[str], Any]
#: A callback the connector supplies to run a catalog query returning rows (each a
#: sequence positional to the SELECT list this module controls).
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


# ---------------------------------------------------------------------------
# Per-column statistics (Postgres pg_stats)
# ---------------------------------------------------------------------------


def catalog_column_stats(
    run_rows: RunRows, dialect: str, table: str, row_count: int | None
) -> dict[str, ColumnStat]:
    """Per-column `ColumnStat` from a dialect's statistics catalog, or an empty dict.

    Only Postgres (and its wire-compatible kin — Redshift, CockroachDB) exposes a
    per-column statistics view cheap enough to read at plan time: ``pg_stats`` records,
    per analyzed column, the null fraction, the distinct-value estimate, the most-common
    values with frequencies, and a histogram of bucket bounds. Each maps onto a
    `ColumnStat` facet:

      - ``null_frac`` x `row_count` -> ``null_count`` (SKETCH, a sampled estimate).
      - ``n_distinct`` → ``ndv``. Postgres encodes a *ratio* as a negative number
        (``-1`` = every value distinct), which is resolved against `row_count`.
      - ``most_common_vals`` / ``most_common_freqs`` → ``mcv`` (Misra-Gries-shaped),
        sharpening equality selectivity far past ``1/ndv`` on a skewed column.
      - ``histogram_bounds`` → ``quantiles`` (an even quantile grid), for interpolating
        range selectivity.

    Every facet is `SKETCH`/`HISTOGRAM` provenance, so none can answer an exact query;
    they only inform cost and cardinality. Array columns (``most_common_vals``,
    ``histogram_bounds``) arrive as driver-specific text and are parsed tolerantly —
    a column whose arrays don't parse still contributes its scalar null/ndv facets.

    Args:
        run_rows: Runs a catalog query and returns its rows, each a positional sequence.
        dialect: The catalog dialect; only ``postgres``/``redshift`` are supported.
        table: The unqualified table name.
        row_count: The table's row count, used to resolve ratios into absolute counts.

    Returns:
        A ``{column_name: ColumnStat}`` mapping (possibly empty).
    """
    if dialect not in ("postgres", "redshift"):
        return {}
    sql = (
        "SELECT attname, null_frac, n_distinct, most_common_vals, "
        f"most_common_freqs, histogram_bounds FROM pg_stats WHERE tablename = '{table}'"
    )
    try:
        rows = run_rows(sql)
    except Exception:
        return {}
    out: dict[str, ColumnStat] = {}
    for row in rows or ():
        parsed = _column_stat_from_pg_row(row, row_count)
        if parsed is not None:
            name, stat = parsed
            out[name] = stat
    return out


def _column_stat_from_pg_row(
    row: Sequence[Any], row_count: int | None
) -> tuple[str, ColumnStat] | None:
    """One ``pg_stats`` row -> ``(column_name, ColumnStat)``, or None if unusable."""
    if not row or row[0] is None:
        return None
    name = str(row[0])
    null_frac = _to_float(_at(row, 1))
    n_distinct = _to_float(_at(row, 2))
    null_count = null_frac * row_count if null_frac is not None and row_count is not None else None
    ndv = _resolve_ndv(n_distinct, row_count)
    mcv = _pg_mcv(_at(row, 3), _at(row, 4))
    quantiles = _pg_histogram(_at(row, 5))
    if null_count is None and ndv is None and mcv is None and quantiles is None:
        return None
    return name, ColumnStat(
        null_count=null_count,
        ndv=ndv,
        mcv=mcv,
        quantiles=quantiles,
        provenance=Provenance.SKETCH,
        # Every catalog column stat is a sampled estimate — never let one answer an
        # exact null_count()/count_distinct(). Both facets carry their own SKETCH tag.
        ndv_provenance=Provenance.SKETCH if ndv is not None else None,
        null_count_provenance=Provenance.SKETCH if null_count is not None else None,
    )


def _resolve_ndv(n_distinct: float | None, row_count: int | None) -> float | None:
    """Postgres ``n_distinct`` -> an absolute distinct-value estimate.

    Postgres records a positive number as the estimated distinct count directly, and a
    negative number as *minus the ratio* of distinct values to rows (``-1`` means every
    value is distinct, ``-0.5`` means half are) — a form that survives the table growing.
    Resolving the ratio needs the row count; without it, only a positive (absolute)
    figure is usable.
    """
    if n_distinct is None or n_distinct == 0.0:
        return None
    if n_distinct > 0:
        return n_distinct
    if row_count is None:
        return None
    return -n_distinct * row_count


def _pg_mcv(vals: Any, freqs: Any) -> dict[str, float] | None:
    """``most_common_vals`` + ``most_common_freqs`` -> ``{str(value): frequency}``."""
    values = _parse_pg_array(vals)
    frequencies = _parse_pg_array(freqs)
    if not values or not frequencies:
        return None
    mcv: dict[str, float] = {}
    for value, freq in zip(values, frequencies, strict=False):
        f = _to_float(freq)
        if f is not None:
            mcv[str(value)] = f
    return mcv or None


def _pg_histogram(bounds: Any) -> dict[str, list[float]] | None:
    """``histogram_bounds`` -> an even quantile grid ``{"probs": …, "values": …}``.

    Postgres stores ``N+1`` bucket boundaries that partition the column into ``N``
    equi-depth buckets, so boundary ``i`` sits at cumulative probability ``i/N`` — an
    ascending quantile grid, exactly the shape `ColumnStat.quantiles` interpolates range
    selectivity from. Non-numeric histograms (text/date bounds) yield None here; their
    range selectivity falls back to the default rather than a mis-parsed grid.
    """
    parsed = _parse_pg_array(bounds)
    if not parsed or len(parsed) < 2:
        return None
    values = [_to_float(v) for v in parsed]
    if any(v is None for v in values):
        return None
    n = len(values) - 1
    probs = [i / n for i in range(len(values))]
    return {"probs": probs, "values": [v for v in values if v is not None]}


def _parse_pg_array(value: Any) -> list[Any] | None:
    """A Postgres array column into a Python list, tolerant of how a driver returns it.

    A driver may hand back a real list (psycopg with array support) or the raw
    ``{a,b,c}`` text form. Both are handled; anything else yields None so a column with an
    unparseable array still contributes its scalar facets.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    body = text[1:-1]
    if not body:
        return []
    return [_unquote(part) for part in body.split(",")]


def _unquote(token: str) -> str:
    """Strip the optional double-quotes Postgres wraps a text array element in."""
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def sql_statistics(
    dialect: str,
    table: str,
    *,
    run_scalar: RunScalar,
    run_rows: RunRows | None = None,
) -> SourceStatistics | None:
    """Everything a SQL catalog can cheaply state about `table`, as one `SourceStatistics`.

    Composes the row count, on-disk byte size, and (where available) per-column statistics
    into a single record for Kyber's estimator. Returns None only when even the row count
    is unavailable — a catalog that gives a count but no column stats still sharpens
    cardinality, so it is worth returning.

    Args:
        dialect: The catalog dialect (see `dialect_for_driver`).
        table: The unqualified table name.
        run_scalar: Runs a single-value catalog query.
        run_rows: Runs a multi-row catalog query, for per-column stats. Omit when the
            connector cannot cheaply run one; row count and byte size still apply.

    Returns:
        The composed statistics, or None when the catalog yields nothing.
    """
    base = catalog_row_count(run_scalar, dialect, table)
    byte_size = catalog_byte_size(run_scalar, dialect, table)
    columns = (
        catalog_column_stats(run_rows, dialect, table, base.row_count if base else None)
        if run_rows is not None
        else {}
    )
    if base is None and byte_size is None and not columns:
        return None
    if base is None:
        # No row count, but a byte size and/or column stats are still worth carrying.
        return SourceStatistics(byte_size=byte_size, columns=columns, exact_rows=False)
    import dataclasses

    return dataclasses.replace(base, byte_size=byte_size, columns=columns)


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
