"""The opt-in GPU execution backend for supported relational shapes.

`collect(backend="gpu")` routes a supported plan to the GPU and falls back to the native CPU
engine for everything else (and when no GPU is present). The plan is optimized first, then
matched from the most specific shape to the general one:

* A **chain over one scan** — filter, project, group-by aggregate, sort, distinct, limit,
  window — whose reducer is mergeable fans out one shard per device, each device reading its own
  shard straight from storage and folding the small partials once at the end.
* A **single join** replicates a small build side and splits the probe across devices; a
  **union** splits each of its inputs.
* **Anything else** — a tree of scans, joins and unions of any depth, which is what a real
  analytical query is — goes to the general tree translator, which splits its largest splittable
  leaf and replicates the rest.

Same result, different *where*. GPU is an accelerator, never a requirement: an unsupported
shape, an expression the kernels cannot compute, a device out of memory, a saturated cluster or
a GPU-less one all return the query to the CPU engine, so `backend="gpu"` is always safe to ask
for. What that tolerance must not hide is the backend being *broken* — see `note_gpu_failure`.

Three modules, because the file outgrew the size limit as the tree translator landed:
`route` decides and records, `translate` matches shapes to executions, `fanout` acquires and
releases the devices.
"""

from __future__ import annotations

from batcher.api.terminal.gpu_backend.route import (
    note_gpu_failure,
    record_cpu_crossover,
    try_gpu_collect,
)

__all__ = ["note_gpu_failure", "record_cpu_crossover", "try_gpu_collect"]
