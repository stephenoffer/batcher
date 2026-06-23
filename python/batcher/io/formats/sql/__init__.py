"""`io.formats.sql` — SQL / data-warehouse connectors, behind the registry.

Each module here registers an Arrow-native relational source (and, where it
makes sense, a sink) into the `SOURCES` / `SINKS` registries as an import side
effect, exactly like the file-format modules. Importing this package imports them
all, so the registry names (``adbc``, ``connectorx``, ``snowflake``,
``databricks``, ``bigquery``, ``clickhouse``, ``odbc``) become available.

Every connector honors the same contract: one query submission, distributed
reads via the backend's *native* result partitioning (no schema or bound
probes), Arrow-only data movement, and picklable, connection-free splits that
rebuild a fresh connection per worker from never-logged credentials. Driver
imports are deferred — importing this package never requires an optional driver.
"""

from __future__ import annotations

from batcher.io.formats.sql.adbc import ADBCSink, ADBCSource
from batcher.io.formats.sql.bigquery import BigQuerySource
from batcher.io.formats.sql.clickhouse import ClickHouseSource
from batcher.io.formats.sql.connectorx import ConnectorXSource
from batcher.io.formats.sql.databricks import DatabricksSource
from batcher.io.formats.sql.odbc import ODBCSource
from batcher.io.formats.sql.snowflake import SnowflakeSink, SnowflakeSource

__all__ = [
    "ADBCSink",
    "ADBCSource",
    "BigQuerySource",
    "ClickHouseSource",
    "ConnectorXSource",
    "DatabricksSource",
    "ODBCSource",
    "SnowflakeSink",
    "SnowflakeSource",
]
