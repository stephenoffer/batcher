"""`vendors` — what Batcher states, and enforces, about specific databases and warehouses.

The generic routes (ADBC, ConnectorX, PEP 249, ODBC) reach far more databases than anyone
has run them against. This package is where a vendor stops being "probably fine through the
generic route" and becomes a stated profile: `profiles` names the routes, the packages that
install them, the write modes each carries and the type rules; `types` holds the
conversions and refusals Batcher enforces on problematic vendor values; `connect` holds the
per-driver adjustments made where a connection is opened. The warehouse helpers sit beside
them: `snowflake_auth` (one declared authentication strategy), `bigquery_sink` and
`databricks_sink` (bulk writes returning the remote job's identity), and `athena` (a
connection profile over the DB-API source).

Importing this package registers the ``bigquery`` and ``databricks`` sinks. Nothing here
imports a driver until a read or write needs one.
"""

from __future__ import annotations

from batcher.io.formats.sql.vendors.athena import ATHENA_DRIVER, athena_connect_kwargs
from batcher.io.formats.sql.vendors.bigquery_sink import BigQuerySink
from batcher.io.formats.sql.vendors.databricks_sink import DatabricksSink
from batcher.io.formats.sql.vendors.profiles import (
    PROFILES,
    Route,
    TypeRule,
    VendorProfile,
    capability_matrix,
    install_hint,
    profile_for,
)
from batcher.io.formats.sql.vendors.snowflake_auth import snowflake_options

__all__ = [
    "ATHENA_DRIVER",
    "PROFILES",
    "BigQuerySink",
    "DatabricksSink",
    "Route",
    "TypeRule",
    "VendorProfile",
    "athena_connect_kwargs",
    "capability_matrix",
    "install_hint",
    "profile_for",
    "snowflake_options",
]
