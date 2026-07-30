"""Observability sinks — the terminal reporter, the activity store, and the web dashboard.

Layer 2 (neutral infrastructure, beside `io`): every subsystem publishes to the event bus
in `_internal.events`, and this package holds the things that *consume* it. Neutral by
construction — it imports no subsystem, so `api` can turn it on without any of `kyber`,
`carbonite`, or `core` learning that a dashboard exists.

The public entry points are `start_ui` / `stop_ui` / `ui_url`, re-exported at the top level
as ``bt.start_ui()``. Everything else here is machinery the conductor drives.
"""

from __future__ import annotations

from batcher.observe.console import ConsoleReporter
from batcher.observe.control import ensure_sinks, start_ui, stop_ui, ui_url
from batcher.observe.energy import energy_metrics, format_device_table, format_energy_report
from batcher.observe.inference import InferenceProgress
from batcher.observe.metrics import (
    metrics_snapshot,
    prometheus_text,
    reset_metrics,
    start_metrics,
)
from batcher.observe.store import ActivityStore

__all__ = [
    "ActivityStore",
    "ConsoleReporter",
    "InferenceProgress",
    "energy_metrics",
    "ensure_sinks",
    "format_device_table",
    "format_energy_report",
    "metrics_snapshot",
    "prometheus_text",
    "reset_metrics",
    "start_metrics",
    "start_ui",
    "stop_ui",
    "ui_url",
]
