"""Accelerator reporting (`accelerators`, `show_accelerators`, `measure_energy`).

The second question on a GPU bug report, after "which build", is "what hardware, and what
was it doing". `versions` answers the first. This answers the second in one call: which
devices this process can see, what the cluster's fleet looks like, what it is drawing, and
whether anything is wrong with any of it. `measure_energy` answers the third — what a
pipeline cost to run.

It reports rather than decides. Every figure comes from a source that already exists — the
device table, live telemetry, the interconnect probes, the cluster topology, the configured
power envelope — and each is omitted when its source cannot answer, so a CPU-only host
produces a small honest report rather than a large one full of zeros.

* `rows` — one row per local device.
* `report` — what a reader is shown, and the call-outs a table would bury.
* `energy` — the measurement scope and the learning fold.
"""

from __future__ import annotations

from batcher.api.session.accelerators.energy import measure_energy
from batcher.api.session.accelerators.report import (
    accelerator_problems,
    accelerators,
    show_accelerators,
)

__all__ = ["accelerator_problems", "accelerators", "measure_energy", "show_accelerators"]
