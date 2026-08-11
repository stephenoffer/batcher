"""The per-domain folds behind the metrics export.

`observe.metrics` owns the process-wide counters and the two shapes they are served in.
What it does *not* own is the arithmetic for each domain it summarizes, and keeping those
apart is what let the export grow past query counts without the module becoming a god file:
one fold per question a metrics backend asks.

- `work` — what the engine spent per operator and per query: CPU, spill volume, real
  block-device I/O, faults, preemption, and which execution tier ran the row work.
- `resources` — what Carbonite is *holding*: the buffer-pool envelope, the spill store's
  tiers, the admission queue, the result cache. Gauges, not counters.
- `streams` — what each continuous query is doing per micro-batch, including how far behind
  its trigger cadence it has fallen.
- `writes` — what a job *produced*: files, rows and bytes on storage, per sink format.

Each fold owns its own lock, its own reset, and its own Prometheus rendering, so a new
domain is a new module here rather than another branch in the collector.
"""

from __future__ import annotations

from batcher.observe.counters.resources import ResourceGauges
from batcher.observe.counters.streams import StreamCounters
from batcher.observe.counters.work import WorkCounters
from batcher.observe.counters.writes import WriteCounters

__all__ = ["ResourceGauges", "StreamCounters", "WorkCounters", "WriteCounters"]
