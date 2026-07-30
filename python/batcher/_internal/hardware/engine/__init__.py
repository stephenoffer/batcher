"""Hardware facts read back from the compiled data plane, rather than probed from Python.

Two questions this package answers that the rest of `hardware` structurally cannot:

* `detected` — the CPU topology and ISA the **engine process** found. It differs from the
  Python-side probe whenever a cgroup lands after `import batcher` (a Ray actor's quota) or
  the work runs on a worker unlike the driver.
* `allocator` — what mimalloc is holding. The engine's allocator retains freed pages by
  design, so process RSS over-counts the data plane; only the allocator knows the split.

Everything degrades to an empty answer when the engine is not built or predates these entry
points, so a caller never has to guard the call.
"""

from __future__ import annotations

from batcher._internal.hardware.engine.allocator import (
    allocator_stats,
    release_retained_memory,
)
from batcher._internal.hardware.engine.detected import (
    engine_hardware,
    engine_numa_map,
    engine_pinning_order,
)

__all__ = [
    "allocator_stats",
    "engine_hardware",
    "engine_numa_map",
    "engine_pinning_order",
    "release_retained_memory",
]
