"""`io.formats.sql` — SQL / data-warehouse connectors, behind the registry.

Each module here registers an Arrow-native relational source (and, where it
makes sense, a sink) into the `SOURCES` / `SINKS` registries as an import side
effect, exactly like the file-format modules. Importing this package imports them
all, so the registry names (``adbc``, ``connectorx``, ``snowflake``,
``databricks``, ``bigquery``, ``clickhouse``, ``odbc``, ``dbapi``) become available.

Every connector honors the same contract: one query submission, Arrow-only data
movement, and picklable, connection-free splits that rebuild a fresh connection per
worker from credentials excluded from every ``repr``. Driver imports are deferred —
importing this package never requires an optional driver.

Parallel reads come from one of two places, in this order of preference:

- The backend's **native** result partitioning — FlightSQL's ``adbc_execute_partitions``,
  Snowflake's result chunks, BigQuery's read streams. One submission, N shippable
  handles, no extra query.
- **Range partitioning** (`partition`), for backends with none: N independent queries
  over disjoint slices of an indexed numeric column, in the shape of Spark's JDBC
  reader. Opt-in, and the bounds come from the caller — this package still issues no
  schema or bound probes of its own.

Two shared modules support the connectors rather than registering anything:
`uri` maps a SQLAlchemy-style connection URI onto whichever backend can serve it, and
`partition` builds the disjoint-and-exhaustive range predicates.
"""

from __future__ import annotations

from batcher.io.formats.sql.adbc import ADBCSink, ADBCSource
from batcher.io.formats.sql.bigquery import BigQuerySource
from batcher.io.formats.sql.clickhouse import ClickHouseSource
from batcher.io.formats.sql.connectorx import ConnectorXSource
from batcher.io.formats.sql.databricks import DatabricksSource
from batcher.io.formats.sql.dbapi import DBAPISource
from batcher.io.formats.sql.odbc import ODBCSource
from batcher.io.formats.sql.snowflake import SnowflakeSink, SnowflakeSource

__all__ = [
    "ADBCSink",
    "ADBCSource",
    "BigQuerySource",
    "ClickHouseSource",
    "ConnectorXSource",
    "DBAPISource",
    "DatabricksSource",
    "ODBCSource",
    "SnowflakeSink",
    "SnowflakeSource",
]
