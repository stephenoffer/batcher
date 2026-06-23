"""`io.stats` — extract cheap `SourceStatistics` from connector metadata.

One module per metadata family, each a pure extraction function a connector's
`statistics()` calls:

  - `columnar_footer`   : Parquet/ORC/Arrow footer column min/max/null/count
  - `lakehouse_manifest`: Delta/Iceberg manifest record counts + column bounds
  - `free_counts`       : NumPy ``.npy`` header row counts
  - `sql_catalog`       : SQL warehouse system-catalog row counts

These read footers/manifests/headers/catalogs — O(1) control-plane metadata I/O,
never a per-row scan. The neutral `SourceStatistics` they return lives in
`batcher.plan.source_stats`.
"""

from __future__ import annotations

from batcher.io.stats.columnar_footer import parquet_statistics
from batcher.io.stats.free_counts import numpy_statistics
from batcher.io.stats.lakehouse_manifest import delta_statistics
from batcher.io.stats.sql_catalog import catalog_row_count

__all__ = [
    "catalog_row_count",
    "delta_statistics",
    "numpy_statistics",
    "parquet_statistics",
]
