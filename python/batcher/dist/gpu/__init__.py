"""Multi-GPU *scheduling* for the translated GPU backend.

`core.gpu_plan` decides what a device computes; this package decides where and how many
devices compute it, and what happens when one of them is lost. It is a `dist` concern for the
same reason the CPU shuffle is: distributing an operator is scheduling, not a second
semantics, and the mergeable algebra is what guarantees the two agree.
"""

from __future__ import annotations

from batcher.dist.gpu.aggregate import sharded_gpu_aggregate
from batcher.dist.gpu.dispatch import gpu_chain_on_worker, gpu_join_on_worker
from batcher.dist.gpu.groupby import dispatch_gpu_aggregate, distributed_gpu_aggregate
from batcher.dist.gpu.tasks import gpu_task_options, gpu_task_runtime_env

__all__ = [
    "dispatch_gpu_aggregate",
    "distributed_gpu_aggregate",
    "gpu_chain_on_worker",
    "gpu_join_on_worker",
    "gpu_task_options",
    "gpu_task_runtime_env",
    "sharded_gpu_aggregate",
]
