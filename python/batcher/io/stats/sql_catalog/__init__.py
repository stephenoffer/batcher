"""Catalog-derived statistics for SQL warehouses and databases.

Every SQL engine maintains table statistics in a system catalog that answers "how many
rows", "how many bytes", and often "how selective is this column" *without scanning the
data*. A single metadata query gives the planner what it would otherwise pay a full
``COUNT(*)`` (or worse) to learn. The probes are grouped by what kind of fact they read:

  - `probes`      : driver/URI -> dialect, and the tolerant query primitives.
  - `counts`      : row count and on-disk byte size, one scalar each.
  - `columns`     : sampled per-column null/ndv/mcv/histogram facets (``pg_stats``).
  - `constraints` : NOT NULL and single-column PRIMARY KEY/UNIQUE — *declared*, so exact.
  - `compose`     : all of the above as one `SourceStatistics`.

The split that matters is the last two against the rest. A `counts`/`columns` figure is
something ``ANALYZE`` measured from a sample, so it drifts and may only inform cost and
cardinality; a `constraints` figure is enforced on every write, so it is EXACT and may
answer a query outright. Row-count exactness follows the engine's own guarantee — a
transactionally maintained count (Snowflake, ClickHouse, SQL Server partition stats) may
answer `count()`, a planner estimate (Postgres ``reltuples``, MySQL ``TABLE_ROWS``) may not.

Every probe is best-effort: a failure (no permission, a view rather than a base table, a
dialect mismatch, an un-analyzed table) yields None/empty and the planner falls back to
its defaults. Nothing here touches a row — the queries read the catalog, and the callback
that runs them belongs to the connector.
"""

from __future__ import annotations

from batcher.io.stats.sql_catalog.columns import catalog_column_stats
from batcher.io.stats.sql_catalog.compose import sql_statistics
from batcher.io.stats.sql_catalog.constraints import constraint_column_stats
from batcher.io.stats.sql_catalog.counts import catalog_byte_size, catalog_row_count
from batcher.io.stats.sql_catalog.probes import (
    RunRows,
    RunScalar,
    dialect_for_driver,
    scalar_count_query,
)

__all__ = [
    "RunRows",
    "RunScalar",
    "catalog_byte_size",
    "catalog_column_stats",
    "catalog_row_count",
    "constraint_column_stats",
    "dialect_for_driver",
    "scalar_count_query",
    "sql_statistics",
]
