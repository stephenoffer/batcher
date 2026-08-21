"""`core.streaming.folds` — the two streaming aggregate folds.

Split into a package when the windowed fold grew a disk tier and a changelog: `running` holds
the unwatermarked aggregate, `windowed` the watermarked one, `shared` what neither owns. The
import path is preserved, so every caller and test keeps working.
"""

from __future__ import annotations

from batcher.core.streaming.folds.running import _AggFold
from batcher.core.streaming.folds.shared import (
    _read,
    _rebatch,
    check_agg_state_bounded,
    empty_global_aggregate,
    streaming_state_budget,
)
from batcher.core.streaming.folds.windowed import (
    _EPOCH,
    _EVICTED_META,
    _WATERMARK_META,
    _median_window,
    _scan_filter_ir,
    _td,
    _window_key,
    _WindowedAggFold,
    _WindowKey,
)

__all__ = [
    "_EPOCH",
    "_EVICTED_META",
    "_WATERMARK_META",
    "_AggFold",
    "_WindowKey",
    "_WindowedAggFold",
    "_median_window",
    "_read",
    "_rebatch",
    "_scan_filter_ir",
    "_td",
    "_window_key",
    "check_agg_state_bounded",
    "empty_global_aggregate",
    "streaming_state_budget",
]
