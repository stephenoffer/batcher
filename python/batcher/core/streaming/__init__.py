"""Streaming (incremental) aggregation and the bounded-memory operator drivers.

Split along the seam between *state* and *driving*: `folds` owns the running aggregate and
the watermark that bounds it, `drivers` owns the per-operator loops that feed those folds
from a source. The public import path `batcher.core.streaming` is unchanged.
"""

from __future__ import annotations

from batcher.core.streaming.drivers import (
    stream_aggregate,
    stream_distinct,
    stream_limit,
    stream_topn,
    stream_windowed_aggregate,
)
from batcher.core.streaming.folds import (
    _WATERMARK_META,
    _AggFold,
    _rebatch,
    _window_key,
    _WindowedAggFold,
    check_agg_state_bounded,
    empty_global_aggregate,
    streaming_state_budget,
)

# The underscore names are engine-internal but *cross-module* — the `map_batches` streaming
# aggregate, the checkpoint state store, and the watermark drivers all reach for them — and
# `batcher.core.streaming` is the import path they have always used. Listing them here is how
# the façade keeps that path working after the split; it widens no public surface, which is
# defined separately in `tools/public_surface.py`.
__all__ = [
    "_WATERMARK_META",
    "_AggFold",
    "_WindowedAggFold",
    "_rebatch",
    "_window_key",
    "check_agg_state_bounded",
    "empty_global_aggregate",
    "stream_aggregate",
    "stream_distinct",
    "stream_limit",
    "stream_topn",
    "stream_windowed_aggregate",
    "streaming_state_budget",
]
