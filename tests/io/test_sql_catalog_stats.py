"""Catalog-derived SQL statistics: dialect mapping, row/byte/column probes, and wiring.

`io.stats.sql_catalog` reads a table's row count, on-disk size, and per-column
null/ndv/mcv/quantile statistics straight from a database's system catalog, so a
warehouse table reaches Kyber with the same facets a Parquet footer supplies. The pure
extraction functions are exercised with mock query callbacks (they must map every
dialect's catalog shape correctly without a live server), and the end-to-end wiring is
pinned against SQLite — a real SQL engine from the standard library, so
`DBAPISource.statistics()` is a genuine catalog round trip with no optional dependency.

Two invariants are load-bearing and asserted directly:

1. **No catalog stat answers an exact query.** Row-count exactness follows the engine's
   own guarantee (Snowflake maintains it, Postgres `reltuples` drifts), and every column
   stat is a sampled `ANALYZE` estimate, so `exact_rows` and the per-facet provenance
   gates must never claim EXACT for an estimate.
2. **A `query=` read declares nothing.** Only a base `table=` has a catalog entry.
"""

from __future__ import annotations

import sqlite3

import pytest

from batcher.io.formats.sql.dbapi import DBAPISource
from batcher.io.stats import (
    catalog_byte_size,
    catalog_column_stats,
    catalog_row_count,
    dialect_for_driver,
    sql_statistics,
)
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit


# --- dialect resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("psycopg2", "postgres"),
        ("psycopg", "postgres"),
        ("postgresql", "postgres"),
        ("postgresql+psycopg", "postgres"),
        ("adbc_driver_postgresql", "postgres"),
        ("cockroachdb", "postgres"),
        ("sqlite3", "sqlite"),
        ("adbc_driver_sqlite", "sqlite"),
        ("clickhouse_connect", "clickhouse"),
        ("snowflake.connector", "snowflake"),
        ("pymysql", "mysql"),
        ("mariadb", "mysql"),
        ("cx_Oracle", "oracle"),
        ("oracledb", "oracle"),
        ("pyodbc", "sqlserver"),
        ("duckdb", "duckdb"),
        ("trino", "trino"),
        ("redshift_connector", "redshift"),
        ("some_unknown_driver", None),
        ("", None),
        (None, None),
    ],
)
def test_dialect_for_driver(name, expected) -> None:
    """Every supported driver/scheme resolves to its catalog dialect; the rest to None."""
    assert dialect_for_driver(name) == expected


# --- row count ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "value", "expect_rows", "expect_exact"),
    [
        ("snowflake", 999, 999, True),
        ("clickhouse", 500, 500, True),
        ("sqlserver", 1234, 1234, True),
        ("postgres", 12345.0, 12345, False),
        ("mysql", 42, 42, False),
        ("oracle", 88, 88, False),
        ("redshift", 7, 7, False),
        ("duckdb", 64, 64, False),
    ],
)
def test_catalog_row_count(dialect, value, expect_rows, expect_exact) -> None:
    """Each dialect's catalog query yields the right count and exactness."""
    stats = catalog_row_count(lambda _sql: value, dialect, "t")
    assert stats is not None
    assert stats.row_count == expect_rows
    assert stats.exact_rows is expect_exact


def test_catalog_row_count_unknown_dialect_is_none() -> None:
    """An unmapped dialect probes nothing rather than running a wrong query."""
    assert catalog_row_count(lambda _sql: 1, "nosuchdb", "t") is None


def test_catalog_row_count_failure_is_none() -> None:
    """A catalog query that raises (no permission, missing view) degrades to None."""

    def boom(_sql: str) -> int:
        raise RuntimeError("permission denied")

    assert catalog_row_count(boom, "postgres", "t") is None


def test_sqlite_stat1_parsing() -> None:
    """SQLite's whitespace-joined stat string is parsed to its leading row count."""
    stats = catalog_row_count(lambda _sql: "1000 4 1", "sqlite", "t")
    assert stats is not None
    assert stats.row_count == 1000
    assert stats.exact_rows is False  # sqlite_stat1 drifts until re-ANALYZEd


def test_sqlite_unanalyzed_is_none() -> None:
    """A table never ANALYZEd has no sqlite_stat1 row, so no count is claimed."""
    assert catalog_row_count(lambda _sql: None, "sqlite", "t") is None


# --- byte size ------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["postgres", "mysql", "clickhouse", "snowflake", "redshift"])
def test_catalog_byte_size(dialect) -> None:
    """A dialect with a size catalog returns a positive byte count."""
    assert catalog_byte_size(lambda _sql: 8192, dialect, "t") == 8192


def test_catalog_byte_size_zero_is_none() -> None:
    """A zero/empty size is reported as None, not a misleading zero."""
    assert catalog_byte_size(lambda _sql: 0, "postgres", "t") is None


def test_catalog_byte_size_unknown_dialect() -> None:
    """A dialect with no size catalog (oracle here) yields None without a query."""
    called = []
    catalog_byte_size(lambda sql: called.append(sql), "oracle", "t")
    assert called == []


# --- column statistics (Postgres pg_stats) --------------------------------------


def _pg_rows() -> list[tuple]:
    """Two `pg_stats` rows: a skewed text column and a fully-distinct key."""
    return [
        # attname, null_frac, n_distinct, most_common_vals, most_common_freqs, histogram
        ("country", 0.25, 3.0, "{US,FR,DE}", "{0.5,0.3,0.2}", None),
        ("id", 0.0, -1.0, None, None, "{1,250,500,750,1000}"),
    ]


def test_pg_column_stats_facets() -> None:
    """pg_stats maps onto null_count, ndv, mcv, and a quantile grid."""
    cols = catalog_column_stats(lambda _sql: _pg_rows(), "postgres", "t", 1000)

    country = cols["country"]
    assert country.null_count == 250.0  # 0.25 * 1000
    assert country.ndv == 3.0
    assert country.mcv == {"US": 0.5, "FR": 0.3, "DE": 0.2}

    key = cols["id"]
    assert key.ndv == 1000.0  # n_distinct=-1 (all distinct) resolved against row count
    assert key.quantiles == {
        "probs": [0.0, 0.25, 0.5, 0.75, 1.0],
        "values": [1.0, 250.0, 500.0, 750.0, 1000.0],
    }


def test_pg_column_stats_never_exact() -> None:
    """Every catalog column stat is a sample, so no facet may answer an exact query."""
    cols = catalog_column_stats(lambda _sql: _pg_rows(), "postgres", "t", 1000)
    country = cols["country"]
    assert country.provenance is Provenance.SKETCH
    assert not country.ndv_is_exact
    assert not country.null_count_is_exact


def test_pg_column_stats_list_valued_arrays() -> None:
    """A driver that returns real Python lists (not `{..}` text) parses identically."""
    rows = [("c", 0.1, 2.0, ["a", "b"], [0.7, 0.3], None)]
    cols = catalog_column_stats(lambda _sql: rows, "postgres", "t", 100)
    assert cols["c"].mcv == {"a": 0.7, "b": 0.3}


def test_pg_column_stats_non_postgres_is_empty() -> None:
    """No other dialect exposes a cheap per-column catalog, so the map is empty."""
    assert catalog_column_stats(lambda _sql: _pg_rows(), "mysql", "t", 1000) == {}


def test_pg_column_stats_failure_is_empty() -> None:
    """A pg_stats query that raises degrades to an empty map, never propagating."""

    def boom(_sql: str) -> list:
        raise RuntimeError("relation pg_stats does not exist")

    assert catalog_column_stats(boom, "postgres", "t", 1000) == {}


# --- composition ----------------------------------------------------------------


def test_sql_statistics_composes_all_facets() -> None:
    """`sql_statistics` folds row count, byte size, and columns into one record."""

    def run_scalar(sql: str) -> object:
        if "reltuples" in sql:
            return 1000.0
        if "pg_total_relation_size" in sql:
            return 999_999
        return None

    stats = sql_statistics("postgres", "t", run_scalar=run_scalar, run_rows=lambda _sql: _pg_rows())
    assert stats is not None
    assert stats.row_count == 1000
    assert stats.exact_rows is False
    assert stats.byte_size == 999_999
    assert set(stats.columns) == {"country", "id"}


def test_sql_statistics_all_missing_is_none() -> None:
    """When the catalog yields nothing at all, no statistics record is produced."""
    assert sql_statistics("oracle", "t", run_scalar=lambda _sql: None) is None


# --- end-to-end wiring against SQLite -------------------------------------------


@pytest.fixture
def analyzed_db(tmp_path):
    """A SQLite database with a table that has been ANALYZEd (so sqlite_stat1 exists)."""
    path = tmp_path / "w.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE ev (id INTEGER, country TEXT)")
    conn.executemany(
        "INSERT INTO ev VALUES (?, ?)",
        [(i, "US" if i % 2 else "FR") for i in range(200)],
    )
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    return path


def test_dbapi_statistics_from_sqlite_catalog(analyzed_db) -> None:
    """`DBAPISource(table=...)` reads its row count from the live SQLite catalog."""
    src = DBAPISource(module="sqlite3", connect_kwargs={"database": str(analyzed_db)}, table="ev")
    stats = src.statistics()
    assert stats is not None
    assert stats.row_count == 200
    assert stats.exact_rows is False  # sqlite_stat1 is advisory


def test_dbapi_statistics_query_read_is_none(analyzed_db) -> None:
    """A `query=` read has no catalog entry, so it declares no statistics."""
    src = DBAPISource(
        module="sqlite3",
        connect_kwargs={"database": str(analyzed_db)},
        query="SELECT id FROM ev WHERE country = 'US'",
    )
    assert src.statistics() is None


def test_dbapi_statistics_borrowed_connection(analyzed_db) -> None:
    """A borrowed (already-open) connection resolves its dialect from the connection type."""
    conn = sqlite3.connect(analyzed_db)
    try:
        src = DBAPISource(connection=conn, table="ev")
        stats = src.statistics()
        assert stats is not None
        assert stats.row_count == 200
    finally:
        conn.close()  # Batcher never closes a borrowed connection; the test owns it.
