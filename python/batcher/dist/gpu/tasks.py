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
from typing import TYPE_CHECKING

from batcher.plan.distribution import nest_ops

if TYPE_CHECKING:
    from batcher.core.gpu_plan import DfBackend

__all__ = [
    "cpu_shard_partial",
    "gpu_join_task",
    "gpu_shard_partial",
    "gpu_task_options",
    "gpu_task_runtime_env",
    "gpu_union_task",
    "run_shard_chain",
    "run_shard_join",
    "run_shard_union",
]


def _read(descriptor: dict):
    """The shard's rows as one Arrow table, or `None` when the shard is empty."""
    import pyarrow as pa

    from batcher.dist.executors.partition_io import read_partition_descriptor

    batches = read_partition_descriptor(descriptor)
    return pa.Table.from_batches(batches) if batches else None


def _device() -> DfBackend:
    """The cuDF backend, imported here so a driver with no RAPIDS can still import this module."""
    import cudf

    from batcher.core.gpu_plan import DfBackend

    return DfBackend(cudf)


def run_shard_chain(descriptor: dict, ops: list[dict], be: DfBackend):
    """Read a shard and replay `ops` on `be`, returning Arrow — the body of the GPU task.

    Parameterized by backend so the *task body* is testable on the host against the CPU engine,
    exactly as the translator it calls is. A task that can only be exercised on a GPU is a task
    nothing checks.

    Returns `None` for an empty shard, which the driver drops rather than concatenating an
    empty table of possibly-different schema into the partials.
    """
    from batcher.core.gpu_plan.execute import run_chain

    table = _read(descriptor)
    return None if table is None else be.to_arrow(run_chain(table, ops, be))


def gpu_shard_partial(descriptor: dict, ops: list[dict]):
    """On a GPU worker: read this shard from storage and replay `ops` on the device."""
    return run_shard_chain(descriptor, ops, _device())


def run_shard_join(
    left_desc: dict,
    right_desc: dict,
    left_ops: list[dict],
    right_ops: list[dict],
    join_ir: dict,
    ops: list[dict],
    be: DfBackend,
):
    """Read both join inputs and run the join on `be` — the body of the GPU join task.

    Returns `None` when either side is empty, which the caller reads as "nothing to join" and
    handles rather than concatenating an empty table of unknown schema.
    """
    from batcher.core.gpu_plan.execute import run_join

    left, right = _read(left_desc), _read(right_desc)
    if left is None or right is None:
        return None
    return be.to_arrow(run_join(left, right, left_ops, right_ops, join_ir, ops, be))


def run_shard_union(
    descriptors: list[dict], input_ops: list[list[dict]], distinct: bool, ops: list[dict], be
):
    """Read every union input and run the union on `be` — the body of the GPU union task.

    Returns `None` when every input was empty, which the caller handles rather than
    concatenating tables of unknown schema.
    """
    from batcher.core.gpu_plan.execute import run_union

    read = [(_read(d), o) for d, o in zip(descriptors, input_ops, strict=True)]
    present = [(t, o) for t, o in read if t is not None]
    if not present:
        return None
    tables = [t for t, _ in present]
    chains = [o for _, o in present]
    return be.to_arrow(run_union(tables, chains, distinct, ops, be))


def gpu_union_task(
    descriptors: list[dict], input_ops: list[list[dict]], distinct: bool, ops: list[dict]
):
    """On a GPU worker: read every union input from storage and run the union on the device."""
    return run_shard_union(descriptors, input_ops, distinct, ops, _device())


def gpu_join_task(
    left_desc: dict,
    right_desc: dict,
    left_ops: list[dict],
    right_ops: list[dict],
    join_ir: dict,
    ops: list[dict],
):
    """On a GPU worker: read both join inputs from storage and run the join on the device."""
    return run_shard_join(left_desc, right_desc, left_ops, right_ops, join_ir, ops, _device())


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
