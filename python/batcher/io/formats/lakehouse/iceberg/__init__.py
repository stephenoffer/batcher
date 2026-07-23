"""`io.formats.lakehouse.iceberg` — the Apache Iceberg connector.

Split by responsibility: `source` reads (with manifest-level file skipping, time travel,
and incremental scans), `sink` writes (workers stage data files, the driver commits one
snapshot via ``add_files``), and `maintenance` expires snapshots. `_common` holds the
dependency gate and the per-write token.

Importing this package registers the source, sink, and maintenance backends into their
registries. Every `pyiceberg` import is deferred, so the import stays cheap and never
fails on a missing extra.
"""

from __future__ import annotations

from batcher.io.formats.lakehouse.iceberg.maintenance import IcebergMaintenance
from batcher.io.formats.lakehouse.iceberg.sink import IcebergSink
from batcher.io.formats.lakehouse.iceberg.source import IcebergSource, IcebergTableSplit

__all__ = ["IcebergMaintenance", "IcebergSink", "IcebergSource", "IcebergTableSplit"]
