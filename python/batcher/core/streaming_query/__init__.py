"""The streaming-query engine — the micro-batch loop behind a unified `ds.write`.

Split along the seam the engine itself is built on: `engine` owns the *cadence* (trigger,
batch counter, offset log, recovery, progress), and `processors` owns what a micro-batch
*becomes* (the per-output-mode transform and the routing that picks one). Neither depends on
the other's internals — the engine holds a `MicroBatchProcessor` and asks it a question —
which is why they read and test apart. The public import path
`batcher.core.streaming_query` is unchanged.
"""

from __future__ import annotations

from batcher.core.streaming_query.engine import StreamingQueryEngine
from batcher.core.streaming_query.processors import (
    AggregateProcessor,
    MicroBatchProcessor,
    StatelessProcessor,
    WindowedAggregateProcessor,
    _distinct_as_aggregate,
    make_processor,
)

#: `_distinct_as_aggregate` is engine-internal but reached for by name across modules, and
#: this is the import path it has always used (see `core.streaming.__init__`).
__all__ = [
    "AggregateProcessor",
    "MicroBatchProcessor",
    "StatelessProcessor",
    "StreamingQueryEngine",
    "WindowedAggregateProcessor",
    "_distinct_as_aggregate",
    "make_processor",
]
