"""`plan.streaming` — neutral streaming-query contract types.

The immutable values the conductor (`api`) and executor (`core`) exchange to run a
streaming query: the `Trigger` cadence and `OutputMode` in `spec`, the per-micro-batch
`StreamingQueryProgress` and its state/source/sink detail in `progress`, and the
`StreamingQueryListener` registry in `listener`. Like all of `plan`, this imports no
subsystem, so both layers share one definition.
"""

from __future__ import annotations

from batcher.plan.streaming.listener import (
    QueryProgressEvent,
    QueryStartedEvent,
    QueryTerminatedEvent,
    StreamingQueryListener,
    add_streaming_listener,
    notify_query_progress,
    notify_query_started,
    notify_query_terminated,
    remove_streaming_listener,
    streaming_listeners,
)
from batcher.plan.streaming.progress import (
    SinkProgress,
    SourceProgress,
    StateOperatorProgress,
    StreamingQueryProgress,
    StreamingQueryStatus,
)
from batcher.plan.streaming.spec import (
    OutputMode,
    Trigger,
    Watermark,
    parse_interval_seconds,
)

__all__ = [
    "OutputMode",
    "QueryProgressEvent",
    "QueryStartedEvent",
    "QueryTerminatedEvent",
    "SinkProgress",
    "SourceProgress",
    "StateOperatorProgress",
    "StreamingQueryListener",
    "StreamingQueryProgress",
    "StreamingQueryStatus",
    "Trigger",
    "Watermark",
    "add_streaming_listener",
    "notify_query_progress",
    "notify_query_started",
    "notify_query_terminated",
    "parse_interval_seconds",
    "remove_streaming_listener",
    "streaming_listeners",
]
