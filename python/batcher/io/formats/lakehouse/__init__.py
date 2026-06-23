"""`io.formats.lakehouse` — table-format connectors behind the format registry.

Each module here connects a transactional lakehouse table format (Delta Lake,
Apache Iceberg, Apache Hudi, Delta Sharing) and registers its source/sink classes
into the `SOURCES` / `SINKS` registries as an import side effect. Importing this
package imports every connector, so the registries are populated. All heavy
optional dependencies (deltalake, pyiceberg, hudi, delta-sharing) are imported
lazily inside methods, so this import is cheap and never fails on a missing extra.
"""

from __future__ import annotations

from batcher.io.formats.lakehouse.delta import DeltaSink, DeltaSource
from batcher.io.formats.lakehouse.delta_sharing import (
    DeltaSharingFileSplit,
    DeltaSharingSource,
)
from batcher.io.formats.lakehouse.hudi import HudiSink, HudiSource
from batcher.io.formats.lakehouse.iceberg import (
    IcebergSink,
    IcebergSource,
    IcebergTableSplit,
)

__all__ = [
    "DeltaSharingFileSplit",
    "DeltaSharingSource",
    "DeltaSink",
    "DeltaSource",
    "HudiSink",
    "HudiSource",
    "IcebergSink",
    "IcebergSource",
    "IcebergTableSplit",
]
