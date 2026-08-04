"""Source connectors — the façade over the source implementation modules.

`base` holds the `Source` protocol (and the `Checkpointable` streaming extension);
`read` the neutral helpers every executor reads a source through; `inmemory`,
`iterator`, and `materialized` the built-in connectors; `inmemory_stats` the lazy
EXACT column-statistics helpers an immutable in-memory source answers metadata from.
The file formats register themselves into `SOURCES` from `io/formats/`. This package
re-exports the connector surface behind the stable `batcher.io.source` import path.
Layer: `io`, the control-plane boundary that hands a plan's leaves to the engine.
"""

from __future__ import annotations

from batcher.io.formats import SOURCES, CSVSource, JSONSource, ParquetSource
from batcher.io.source.base import Checkpointable, Source, is_checkpointable
from batcher.io.source.inmemory import InMemorySource
from batcher.io.source.iterator import IteratorSource
from batcher.io.source.materialized import MaterializedSource
from batcher.io.source.read import (
    continues_across_passes,
    is_bounded,
    iter_source,
    plan_splits,
    read_source,
    source_statistics,
)
from batcher.io.splits import Split

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
    "continues_across_passes",
    "is_bounded",
    "is_checkpointable",
    "iter_source",
    "plan_splits",
    "read_source",
    "source_statistics",
]
