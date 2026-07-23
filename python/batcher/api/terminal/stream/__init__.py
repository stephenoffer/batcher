"""Streaming terminal path for `Dataset.iter_batches` — package façade.

`_iter_batches` picks the most bounded-memory way to yield a plan's result as Arrow
batches: a breaker-free pipeline streamed one source batch at a time, a top-level
aggregate / distinct / top-N / limit folded into a running state, a top-level breaker
streamed from the out-of-core bucket pipeline, or — failing all of those — a
materialize-and-rechunk. An unbounded (streaming) source whose plan must materialize
raises `PlanError` instead of hanging.

Split by responsibility: `dispatch` routes a plan to a strategy and drives the two
plan-shape paths (breaker-free pipeline, exact rebatch), while `watermark` owns the
strategies bounded by event time (watermark dedup, stream-stream interval join). Both
are re-exported so `from batcher.api.terminal.stream import X` keeps working across the
split; `terminal` re-exports `_iter_batches`.
"""

from __future__ import annotations

from batcher.api.terminal.stream.dispatch import _iter_batches, _iter_streaming

__all__ = ["_iter_batches", "_iter_streaming"]
