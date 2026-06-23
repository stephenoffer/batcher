"""Core — the adaptive executor. **Execution and adaptation only.**

Responsibility boundary (enforced by the layer-import contract):
  * Core drives the engine: it hands the physical plan to the native runtime,
    schedules morsels, runs the adaptive control loop (re-optimize triggers,
    batch sizing), and reports `OperatorFeedback`.
  * Core does NOT decide the plan (Kyber) and does NOT own memory/credits
    (Carbonite). It consumes Carbonite's allocation primitives and emits feedback
    Kyber learns from — but never imports `kyber` or `carbonite`.

The bootstrap executor calls the Tier-0 interpreter through `batcher._native`;
the morsel scheduler, JIT tiers, and `bc-adapt` control loop land behind this seam.
"""

from __future__ import annotations

from batcher.core.base import ExecutionContext, Executor
from batcher.core.executor import (
    LocalExecutor,
    execute_local,
    execute_local_metered,
    record_exec_metrics,
)
from batcher.core.runtime import default_hub, reset_default_hub
from batcher.core.stats import column_statistics, heavy_hitters, tail_quantiles
from batcher.core.udf import execute_with_udfs, has_map_batches

__all__ = [
    "ExecutionContext",
    "Executor",
    "LocalExecutor",
    "column_statistics",
    "default_hub",
    "execute_local",
    "execute_local_metered",
    "execute_with_udfs",
    "has_map_batches",
    "heavy_hitters",
    "record_exec_metrics",
    "reset_default_hub",
    "tail_quantiles",
]
