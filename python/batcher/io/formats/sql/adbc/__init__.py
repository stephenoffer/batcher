"""`adbc` — the ADBC / FlightSQL source and sink.

`source` owns reading (connection handling, the schema probe, splits, and both the
server-side FlightSQL and range partitioning paths); `sink` owns bulk ingest, which
carries a distributed-write concern reading does not. They share only `_connect`.

Importing this package registers both, exactly as the single module it replaced did, and
the public import path is unchanged.

One note for tests: patch ``batcher.io.formats.sql.adbc.source._connect``. The sink
reaches that function through its module rather than importing it by name, so that one
target intercepts reads and writes alike.
"""

from __future__ import annotations

from batcher.io.formats.sql.adbc.sink import ADBCSink
from batcher.io.formats.sql.adbc.source import (
    ADBCSource,
    _ADBCPartitionSplit,
    _ADBCQuerySplit,
    _connect,
)

# The split classes are private but re-exported: they were importable from the module
# this package replaced, and tests construct them directly to assert streaming and
# pickling behavior that has no public entry point.
__all__ = ["ADBCSink", "ADBCSource", "_ADBCPartitionSplit", "_ADBCQuerySplit", "_connect"]
