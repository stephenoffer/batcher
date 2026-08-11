"""`plan.streaming` — neutral streaming-query contract types.

The immutable values the conductor (`api`) and executor (`core`) exchange to run a
streaming query: the `Trigger` cadence and `OutputMode` in `spec`, the per-micro-batch
`StreamingQueryProgress` and its state/source/sink detail in `progress`, the
`StreamingQueryListener` registry in `listener`, and the per-partition `WatermarkTracker`
in `tracker` — the one definition of how far event time has actually advanced, shared by
the `core` folds and the `api` streaming drivers. Like all of `plan`, this imports no
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
from batcher.plan.streaming.rate import RateController, RateLimit
from batcher.plan.streaming.spec import (
    OutputMode,
    Trigger,
    Watermark,
    parse_interval_seconds,
)
from batcher.plan.streaming.tracker import WatermarkTracker, event_micros

__all__ = [
    "OutputMode",
    "QueryProgressEvent",
    "QueryStartedEvent",
    "QueryTerminatedEvent",
    "RateController",
    "RateLimit",
    "SinkProgress",
    "SourceProgress",
    "StateOperatorProgress",
    "StreamingQueryListener",
    "StreamingQueryProgress",
    "StreamingQueryStatus",
    "Trigger",
    "Watermark",
    "WatermarkTracker",
    "add_streaming_listener",
    "event_micros",
    "notify_query_progress",
    "notify_query_started",
    "notify_query_terminated",
    "parse_interval_seconds",
    "remove_streaming_listener",
    "streaming_listeners",
]
