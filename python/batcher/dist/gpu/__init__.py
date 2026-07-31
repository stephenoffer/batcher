"""Multi-GPU *scheduling* for the translated GPU backend.

`core.gpu_plan` decides what a device computes; this package decides where and how many
devices compute it, and what happens when one of them is lost. It is a `dist` concern for the
same reason the CPU shuffle is: distributing an operator is scheduling, not a second
semantics, and the mergeable algebra is what guarantees the two agree.
"""

from __future__ import annotations

from batcher.dist.gpu.aggregate import sharded_gpu_aggregate
from batcher.dist.gpu.cudf_probe import cluster_has_cudf, mark_cudf_missing
from batcher.dist.gpu.dispatch import (
    gpu_chain_on_worker,
    gpu_join_on_worker,
    gpu_tree_on_worker,
    gpu_union_on_worker,
)
from batcher.dist.gpu.groupby import dispatch_gpu_aggregate, distributed_gpu_aggregate
from batcher.dist.gpu.join import sharded_gpu_join
from batcher.dist.gpu.resources import gpu_shard_options, shard_task_share, share_for_bytes
from batcher.dist.gpu.shards import measured_parts
from batcher.dist.gpu.tasks import gpu_task_options, gpu_task_runtime_env
from batcher.dist.gpu.tree import sharded_gpu_tree
from batcher.dist.gpu.union import sharded_gpu_union

__all__ = [
    "cluster_has_cudf",
    "dispatch_gpu_aggregate",
    "distributed_gpu_aggregate",
    "gpu_chain_on_worker",
    "gpu_join_on_worker",
    "gpu_shard_options",
    "gpu_task_options",
    "gpu_task_runtime_env",
    "gpu_tree_on_worker",
    "gpu_union_on_worker",
    "mark_cudf_missing",
    "measured_parts",
    "shard_task_share",
    "sharded_gpu_aggregate",
    "sharded_gpu_join",
    "sharded_gpu_tree",
    "sharded_gpu_union",
    "share_for_bytes",
]
