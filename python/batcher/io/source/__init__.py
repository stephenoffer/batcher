"""Source connectors — the façade over the source implementation modules.

`_impl` holds the `Source` protocol and the built-in connectors (in-memory, iterator,
materialized, and the file formats registered in `SOURCES`); `inmemory_stats` holds the
lazy EXACT column-statistics helpers an immutable in-memory source answers metadata from.
This package re-exports the connector surface behind the stable `batcher.io.source` import
path. Layer: `io`, the control-plane boundary that hands a plan's leaves to the engine.
"""

from __future__ import annotations

from batcher.io.source._impl import (
    SOURCES,
    Checkpointable,
    CSVSource,
    InMemorySource,
    IteratorSource,
    JSONSource,
    MaterializedSource,
    ParquetSource,
    Source,
    Split,
    is_bounded,
    is_checkpointable,
    iter_source,
    read_source,
    source_statistics,
)

__all__ = [
    "SOURCES",
    "CSVSource",
    "Checkpointable",
    "InMemorySource",
    "IteratorSource",
    "JSONSource",
    "MaterializedSource",
    "ParquetSource",
    "Source",
    "Split",
    "is_bounded",
    "is_checkpointable",
    "iter_source",
    "read_source",
    "source_statistics",
]
