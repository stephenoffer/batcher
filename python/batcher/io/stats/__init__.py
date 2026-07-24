"""`io.stats` — extract cheap `SourceStatistics` from connector metadata.

One module per metadata family, each a pure extraction function a connector's
`statistics()` calls:

  - `columnar_footer`   : Parquet column min/max/null/count + ORC row counts
  - `parquet_manifest`  : per-FILE Parquet bounds, for skipping whole files
  - `pruning`           : per-row-group zone-map bounds for file/row-group pruning
  - `lakehouse_manifest`: Delta/Iceberg manifest record counts + column bounds
  - `free_counts`       : NumPy ``.npy`` header row counts
  - `sql_catalog`       : SQL warehouse system-catalog row counts, byte size, and
                          per-column null/ndv/mcv/quantile statistics

These read footers/manifests/headers/catalogs — O(1) control-plane metadata I/O,
never a per-row scan. The neutral `SourceStatistics` they return lives in
`batcher.plan.source_stats`.
"""

from __future__ import annotations

from batcher.io.stats.columnar_footer import orc_statistics, parquet_statistics
from batcher.io.stats.free_counts import numpy_statistics
from batcher.io.stats.lakehouse_manifest import manifest_statistics
from batcher.io.stats.parquet_manifest import parquet_file_manifest
from batcher.io.stats.pruning import (
    RowGroupBounds,
    parquet_row_group_bounds,
    surviving_rows_for_range,
)
from batcher.io.stats.sql_catalog import (
    catalog_byte_size,
    catalog_column_stats,
    catalog_row_count,
    dialect_for_driver,
    sql_statistics,
)

__all__ = [
    "RowGroupBounds",
    "catalog_byte_size",
    "catalog_column_stats",
    "catalog_row_count",
    "dialect_for_driver",
    "manifest_statistics",
    "numpy_statistics",
    "orc_statistics",
    "parquet_file_manifest",
    "parquet_row_group_bounds",
    "parquet_statistics",
    "sql_statistics",
    "surviving_rows_for_range",
]
