"""Watching accelerators over a run, rather than reading them once at the end.

`observe.node_metrics` exports the four figures every GPU dashboard has and `observe.energy`
prints what the devices are doing right now. Both read instantaneously, and an instantaneous
reading taken when a query finishes describes an idle fleet — accurately, and uselessly.

This package closes that gap:

* `series` — a daemon sampler folding readings into a bounded rolling window, started
  explicitly and never at import, so a short-lived Ray worker does not pay for it.
* `gauges` — the deep readings as Prometheus series: link throughput and derate, clock
  headroom, codec engines, the memory reserve, BAR1, the integrated energy counter, and the
  DCGM performance counters where they exist.
* `diagnosis` — the sampled window rendered as one verdict and one fix per device.

**Facts leave through the scrape endpoint; verdicts leave through a report.** `observe` is a
neutral layer and must not export a decision, so `gauges` carries only what the hardware said
and `diagnosis` is what a person reads.

A neutral layer: it consumes the event bus and the hardware probes, and imports no subsystem.
"""

from __future__ import annotations

from batcher.observe.accelerators.diagnosis import (
    device_verdicts,
    format_bottleneck_report,
    format_saturation_line,
)
from batcher.observe.accelerators.gauges import (
    accelerator_gauges,
    link_gauges,
    utilization_gauges,
)
from batcher.observe.accelerators.series import (
    device_window,
    reset_device_series,
    sampling_active,
    start_device_series,
    stop_device_series,
)

__all__ = [
    "accelerator_gauges",
    "device_verdicts",
    "device_window",
    "format_bottleneck_report",
    "format_saturation_line",
    "link_gauges",
    "reset_device_series",
    "sampling_active",
    "start_device_series",
    "stop_device_series",
    "utilization_gauges",
]
