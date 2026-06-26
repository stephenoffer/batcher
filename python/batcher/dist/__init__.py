"""`dist` — distributed execution (opt-in, Ray-orchestrated).

Ray is used ONLY for scheduling tasks and passing tiny control messages (file
paths). The data plane bypasses the Ray object store entirely: shuffle data moves
as Arrow IPC files (the Spark model — scalable, larger-than-memory, and exactly
the "no RecordBatch in the object store" property the architecture requires).

The orchestrator reuses the *same* mergeable primitives as the single-node
parallel executor (`partial_aggregate` → hash-shuffle → `combine_finalize`), so a
distributed aggregation provably equals the single-node one. Shapes it can't yet
distribute fall back to the multi-core single-node engine.
"""

from __future__ import annotations

from batcher.dist.executor import execute_distributed
from batcher.dist.executors.ray_runtime import cluster_topology, resolve_transport

__all__ = ["cluster_topology", "execute_distributed", "resolve_transport"]
