"""`plan.streaming` — neutral streaming-query contract types.

The immutable values the conductor (`api`) and executor (`core`) exchange to run a
streaming query: the `Trigger` cadence, the `OutputMode`, and the per-micro-batch
`StreamingQueryProgress` / `StreamingQueryStatus`. Like all of `plan`, this imports
no subsystem, so both layers share one definition. Grouped as a subpackage because
the watermark/window streaming contracts join it as the feature set grows.
"""

from __future__ import annotations

from batcher.plan.streaming.spec import (
    OutputMode,
    StreamingQueryProgress,
    StreamingQueryStatus,
    Trigger,
    Watermark,
    parse_interval_seconds,
)

__all__ = [
    "OutputMode",
    "StreamingQueryProgress",
    "StreamingQueryStatus",
    "Trigger",
    "Watermark",
    "parse_interval_seconds",
]
