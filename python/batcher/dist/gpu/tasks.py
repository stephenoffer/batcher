"""The Ray-side of a GPU fan-out: what a GPU worker runs, and what it is scheduled with.

Scheduling is a `dist` concern, so the remote options and the task bodies live here rather
than in the conductor. Two task bodies, and they are deliberately interchangeable:

* `gpu_shard_partial` reads one shard from storage and reduces it on the device;
* `cpu_shard_partial` reads the same shard and reduces it with the native CPU engine.

They return the *same* mergeable partial for the same shard, which is what lets a lost or
unusable device cost one shard rather than the query: the driver can substitute the second for
the first mid-fan-out and the combined answer is unchanged.

Both read their own shard directly from storage. The driver never materializes the source to
hand it out, so a fan-out over a hundred shards moves no bulk data through the object store.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "cpu_shard_partial",
    "gpu_shard_partial",
    "gpu_task_options",
    "gpu_task_runtime_env",
    "nest_ops",
]


def nest_ops(ops: list[dict], source_id: int = 0) -> dict:
    """The operator chain as one nested `RelOp` IR document over a scan.

    The translator carries a chain bottom-up as a flat list, which is the convenient shape for
    replaying it; the engine wants the nested form. Rebuilding it here is what lets the CPU
    fallback run *the same chain* the device would have — including the partial aggregate the
    mergeable decomposition produced — rather than an approximation of it.

    Args:
        ops: The bottom-up operator IR chain.
        source_id: The scan's source index within the task's input list.

    Returns:
        A nested `RelOp` IR document whose leaf is a `scan`.
    """
    node: dict[str, Any] = {"op": "scan", "source_id": source_id}
    for op in ops:
        node = {**op, "input": node}
    return node


def gpu_shard_partial(descriptor: dict, ops: list[dict]):
    """On a GPU worker: read this shard from storage and replay `ops` on the device.

    Returns `None` for an empty shard, which the driver drops rather than concatenating an
    empty table of possibly-different schema into the partials.
    """
    import pyarrow as pa

    from batcher.core.gpu_plan import DfBackend
    from batcher.core.gpu_plan.execute import run_chain
    from batcher.dist.executors.partition_io import read_partition_descriptor

    batches = read_partition_descriptor(descriptor)
    if not batches:
        return None
    import cudf

    be = DfBackend(cudf)
    return be.to_arrow(run_chain(pa.Table.from_batches(batches), ops, be))


def cpu_shard_partial(descriptor: dict, ops: list[dict], engine_config: str):
    """On any worker: read this shard and replay `ops` with the native CPU engine.

    The substitute for `gpu_shard_partial` when a device is lost or unusable. It goes through
    the engine rather than through the translator's host backend on purpose: the engine is the
    correctness oracle, and a fallback is exactly the moment not to be running the less-tested
    of two paths.
    """
    import pyarrow as pa

    from batcher._internal.native import engine
    from batcher.dist.executors.partition_io import read_partition_descriptor

    batches = read_partition_descriptor(descriptor)
    if not batches:
        return None
    plan = json.dumps(nest_ops(ops))
    out = engine().execute_plan(plan, [batches], engine_config)
    return pa.Table.from_batches(out) if out else None


def gpu_task_runtime_env() -> dict | None:
    """The runtime_env for a GPU task: batcher, plus cuDF when the config asks for it.

    numpy is pinned to the cluster version so arrays returned from the task unpickle on the
    driver — cuDF's install otherwise drags numpy to 2.x underneath the caller.
    """
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime.scheduling import worker_runtime_env

    rt = dict(worker_runtime_env() or {})
    if active_config().distributed.gpu_backend_cudf:
        rt["pip"] = ["cudf-cu13==26.6.0", "numpy==1.26.4"]
    return rt or None


def gpu_task_options() -> dict:
    """Ray remote options for a GPU task: one device, the runtime_env, and a retry budget.

    `max_retries` reruns a task whose worker or node was lost (spot reclamation) on surviving
    capacity. `retry_exceptions` is deliberately left off: a deterministic application error —
    a device OOM, an untranslatable expression — must be handled immediately rather than
    repeated N times to the same conclusion.
    """
    from batcher.config import active_config

    opts: dict = {"num_gpus": 1, "max_retries": int(active_config().distributed.task_max_retries)}
    rt = gpu_task_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    return opts
