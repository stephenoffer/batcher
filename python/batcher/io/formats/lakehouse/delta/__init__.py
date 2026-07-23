"""`io.formats.lakehouse.delta` — the Delta Lake connector.

Split by responsibility: `source` reads (with transaction-log file skipping and time
travel), `stream` reads the Change Data Feed as an unbounded stream, and `sink` writes
(final data files on the workers, a metadata-only commit on the driver). `_snapshot` is
the one resolved read of the `_delta_log` they all share, and `_commit` builds the
transaction the sink writes.

Importing this package registers the source/sink classes into the `SOURCES`/`SINKS`
registries. Every `deltalake` import is deferred, so the import stays cheap and never
fails on a missing extra.
"""

from __future__ import annotations

from batcher.io.formats.lakehouse.delta._snapshot import require_deltalake
from batcher.io.formats.lakehouse.delta.maintenance import DeltaMaintenance
from batcher.io.formats.lakehouse.delta.sink import DeltaSink
from batcher.io.formats.lakehouse.delta.source import DeltaFileSplit, DeltaSource
from batcher.io.formats.lakehouse.delta.stream import DeltaStreamSource

__all__ = [
    "DeltaFileSplit",
    "DeltaMaintenance",
    "DeltaSink",
    "DeltaSource",
    "DeltaStreamSource",
    "require_deltalake",
]
