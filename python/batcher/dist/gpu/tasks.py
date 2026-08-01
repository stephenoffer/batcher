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

from batcher._internal.logging import note_suppressed
from batcher.dist.gpu.cudf_probe import cluster_has_cudf
from batcher.plan.distribution import nest_ops

if TYPE_CHECKING:
    from batcher.core.gpu_plan import DfBackend

__all__ = [
    "cpu_shard_partial",
    "gpu_join_task",
    "gpu_shard_partial",
    "gpu_task_options",
    "gpu_task_runtime_env",
    "gpu_tree_task",
    "gpu_union_task",
    "run_shard_chain",
    "run_shard_join",
    "run_shard_tree",
    "run_shard_union",
]


def _read(descriptor: dict):
    """The shard's rows as one Arrow table, or `None` when the shard is empty."""
    import pyarrow as pa

    from batcher.dist.executors.partition_io import read_partition_descriptor

    batches = read_partition_descriptor(descriptor)
    return pa.Table.from_batches(batches) if batches else None


def _frame(descriptor: dict, be: DfBackend):
    """One descriptor's rows as a frame on `be`, or `None` when it holds no rows.

    The device reads for itself where it can, which skips a CPU Parquet decode and a transfer
    across the bus; where it cannot, the host reader runs and the table is converted. The two
    produce the same rows by construction — the device path declines rather than approximating
    — so which one ran is a question of speed, and every caller can ignore the difference.
    """
    from batcher.dist.gpu.device_read import read_descriptor_on_device

    frame = read_descriptor_on_device(descriptor, be)
    if frame is not None:
        return frame if len(frame) else None
    table = _read(descriptor)
    return None if table is None else be.from_arrow(table)


def _empty_frame(descriptor: dict, be: DfBackend):
    """A zero-row frame carrying the descriptor's own schema, or `None` when it has none.

    A leaf that read nothing is not the same event in a tree as it is in a chain. A chain's
    empty shard contributes nothing and is dropped; a tree's empty *leaf* is still an input to a
    join, and a LEFT join over an empty right side has to emit every left row with nulls rather
    than emit nothing. Handing the join a typed empty frame is what makes that the join's
    decision instead of the reader's.
    """
    splits = descriptor.get("splits")
    if not splits:
        return None
    try:
        schema = splits[0].schema()
    except Exception as exc:
        note_suppressed("dist", "read a shard's schema for an empty leaf", exc)
        return None
    projection = descriptor.get("projection")
    if projection is not None:
        schema = _select_fields(schema, projection)
        if schema is None:
            return None
    return be.from_arrow(schema.empty_table())


def _select_fields(schema, projection: list[str]):
    """`schema` narrowed to `projection`, in that order, or `None` when a name is absent."""
    import pyarrow as pa

    try:
        return pa.schema([schema.field(name) for name in projection])
    except KeyError:
        return None


def _leaf_frame(descriptor: dict, be: DfBackend):
    """One leaf's rows as a frame, falling back to a typed empty frame when it read nothing."""
    frame = _frame(descriptor, be)
    return _empty_frame(descriptor, be) if frame is None else frame


def _device() -> DfBackend:
    """The cuDF backend, imported here so a driver with no RAPIDS can still import this module.

    Configures the worker's device allocator on the way past. It happens here rather than at
    task submission because the allocator is a property of the *process* that will compute,
    and only the worker knows which device it was given and what is already resident on it.
    The call is idempotent, so the second task this worker runs pays nothing and keeps the
    pool the first one built.
    """
    from batcher.carbonite.accel import bind_host_threads_to_device, prepare_device_memory

    # Binding comes first, before anything in this process sizes itself **and before cuDF is
    # imported**. The worker's *host* half — the reader, the decoder, the staging buffer —
    # belongs on the cores next to the device it feeds; left alone on a two-socket node, half
    # the workers land across the inter-socket link and pay for it twice per batch, at full
    # device utilization and with nothing in the timings to say so. Doing it after the
    # allocator would leave every pool sized for the whole node while the process runs on half
    # of it. Refuses itself where the mapping is unreadable or the local core set is too small
    # to decode in.
    #
    # Importing cuDF first — which is what this did — creates the CUDA context, and the
    # context's pinned host staging buffers are allocated on whichever NUMA node the process
    # happened to be on at that moment. Those buffers are what every host-to-device copy for
    # the life of the worker passes through, and they cannot be moved afterwards, so binding
    # after the import placed the threads correctly and left the memory they use on the far
    # socket. On a one-socket development box the two orders are indistinguishable.
    bind_host_threads_to_device()

    import cudf

    from batcher.core.gpu_plan import DfBackend
    from batcher.dist.gpu.resources import task_device_tenants

    # The tenancy Ray actually granted, not an assumption of one. A packed fan-out runs several
    # of these bodies on one board, and each of them is about to reserve a memory pool.
    prepare_device_memory(tenants=task_device_tenants())
    return DfBackend(cudf)


def run_shard_chain(descriptor: dict, ops: list[dict], be: DfBackend):
    """Read a shard and replay `ops` on `be`, returning Arrow — the body of the GPU task.

    Parameterized by backend so the *task body* is testable on the host against the CPU engine,
    exactly as the translator it calls is. A task that can only be exercised on a GPU is a task
    nothing checks.

    The device reads its own shard where it can (`device_read`), which skips a CPU Parquet
    decode and a transfer across the bus; where it cannot, the host reader runs and the same
    operator chain replays on the result. The two produce the same rows by construction — the
    device path declines rather than approximating — so which one ran is a question of speed.

    Returns `None` for an empty shard, which the driver drops rather than concatenating an
    empty table of possibly-different schema into the partials.
    """
    from batcher.core.gpu_plan.execute import run_ops

    frame = _frame(descriptor, be)
    return None if frame is None else be.to_arrow(run_ops(frame, ops, be))


def _measured(run):
    """Run a device task, and if it overflows, re-raise with what the device had actually drawn.

    The subdivision that follows an overflow is decided on the **driver**, which has no device;
    asking its own allocator for the high-water mark — which is what it did — returns nothing on
    every distributed run, so the "measured" division silently degraded to blind halving and a
    shard eight times too large took three failed rounds, each re-reading it from storage, to
    find a size that fits.

    The figure exists only here, in the process that overflowed. Appending it to the error is
    what carries it to the process that needs it, and it is the one channel a task failure is
    guaranteed to travel through intact.

    Re-raised as a `MemoryError` chained to the original, so the classification is unchanged
    (`is_memory_failure` already reads a `MemoryError` as one) and the real traceback is still
    attached to whatever finally reports it.
    """
    from batcher.dist.gpu.shards import device_peak_marker, is_memory_failure

    try:
        return run()
    except Exception as exc:
        marker = device_peak_marker() if is_memory_failure(exc) else ""
        if not marker:
            raise
        raise MemoryError(f"{type(exc).__name__}: {exc}{marker}") from exc


def gpu_shard_partial(descriptor: dict, ops: list[dict]):
    """On a GPU worker: read this shard from storage and replay `ops` on the device."""
    return _measured(lambda: run_shard_chain(descriptor, ops, _device()))


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

    Each side is read on the device where it can be, independently of the other: the two are
    different relations, so one arriving from the device reader and the other from the host one
    costs nothing. A broadcast join's small build side is usually the one that cannot.

    Returns `None` when either side is empty, which the caller reads as "nothing to join" and
    handles rather than concatenating an empty table of unknown schema.
    """
    from batcher.core.gpu_plan.execute import run_join_frames

    left = _frame(left_desc, be)
    right = _frame(right_desc, be)
    if left is None or right is None:
        return None
    return be.to_arrow(run_join_frames(left, right, left_ops, right_ops, join_ir, ops, be))


def run_shard_union(
    descriptors: list[dict], input_ops: list[list[dict]], distinct: bool, ops: list[dict], be
):
    """Read every union input and run the union on `be` — the body of the GPU union task.

    Returns `None` when every input was empty, which the caller handles rather than
    concatenating tables of unknown schema.
    """
    from batcher.core.gpu_plan.execute import run_union_frames

    read = [(_frame(d, be), o) for d, o in zip(descriptors, input_ops, strict=True)]
    present = [(f, o) for f, o in read if f is not None]
    if not present:
        return None
    frames = [f for f, _ in present]
    chains = [o for _, o in present]
    return be.to_arrow(run_union_frames(frames, chains, distinct, ops, be))


def gpu_union_task(
    descriptors: list[dict], input_ops: list[list[dict]], distinct: bool, ops: list[dict]
):
    """On a GPU worker: read every union input from storage and run the union on the device."""
    return _measured(lambda: run_shard_union(descriptors, input_ops, distinct, ops, _device()))


def gpu_join_task(
    left_desc: dict,
    right_desc: dict,
    left_ops: list[dict],
    right_ops: list[dict],
    join_ir: dict,
    ops: list[dict],
):
    """On a GPU worker: read both join inputs from storage and run the join on the device."""
    return _measured(
        lambda: run_shard_join(left_desc, right_desc, left_ops, right_ops, join_ir, ops, _device())
    )


def run_shard_tree(descriptors: list[dict], spec: dict, be: DfBackend):
    """Read every leaf of a plan tree and execute it on `be` — the body of the GPU tree task.

    `descriptors` is positional by leaf index, which is the numbering `gpu_tree_spec` assigns.
    Positional rather than keyed by source, because a self-join has two leaves over one source
    and they read different things: one is this worker's shard, the other is the whole relation.

    Returns `None` when the tree produced no rows, which the driver drops rather than
    concatenating an empty table of possibly-different schema into the partials.
    """
    from batcher.core.gpu_plan.tree import run_tree

    frames = {}
    for leaf, descriptor in enumerate(descriptors):
        frame = _leaf_frame(descriptor, be)
        if frame is None:
            # No rows and no schema to invent one from. Every join above this leaf would be over
            # an unknown-width input, so the shard declines and the driver recovers it.
            return None
        frames[leaf] = frame
    out = run_tree(spec, frames, be)
    return be.to_arrow(out) if len(out) else None


def gpu_tree_task(descriptors: list[dict], spec: dict):
    """On a GPU worker: read every leaf of a plan tree from storage and run it on the device."""
    return _measured(lambda: run_shard_tree(descriptors, spec, _device()))


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
    """The runtime_env for a GPU task: batcher, cuDF when configured, and the node's fabric.

    numpy is pinned to the cluster version so arrays returned from the task unpickle on the
    driver — cuDF's install otherwise drags numpy to 2.x underneath the caller.

    The fabric block (`collective_env`) tells a collective library which NIC each device is
    rail-aligned with, which interfaces carry the fabric, and whether peer-to-peer can help
    here, instead of letting it re-derive all three by probing. It is empty on a node whose
    wires cannot be read, and it never overwrites a variable the deployment set itself, so the
    worst case is exactly the environment the task had before.

    The stability block (`carbonite.resilience.collectives`) answers the other half: what
    happens when one of those wires, or a rank on the end of it, goes away. A collective's
    default there is to wait forever — the surviving ranks hold their GPUs and never raise, so
    the task looks alive and makes no progress, and every recovery mechanism in the engine is
    downstream of a failure being reported. Asynchronous error handling turns that into an
    ordinary task failure. The two blocks set disjoint variables, and both defer to anything
    the deployment set for itself.

    The ordering block (`devices.device_order_env`) is the smallest of the three and the one
    that makes the other two mean anything: it pins the CUDA runtime to PCI-bus device
    numbering, so the ordinal a worker computes on and the NVML index its telemetry, its NUMA
    mapping, and its memory pool are all read from name the same board. Unset, CUDA sorts by
    capability instead, and every one of those lookups silently addresses a different device on
    a node whose GPUs are not identical.
    """
    from batcher._internal.hardware.devices import device_order_env
    from batcher.carbonite.resilience import stability_env
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime.scheduling import worker_runtime_env
    from batcher.dist.gpu.fabric import merge_env, node_collective_env

    rt = dict(worker_runtime_env() or {})
    if active_config().distributed.gpu_backend_cudf and not cluster_has_cudf():
        rt["pip"] = ["cudf-cu13==26.6.0", "numpy==1.26.4"]
    block = {**device_order_env(), **stability_env(), **node_collective_env()}
    if block:
        rt["env_vars"] = merge_env(rt.get("env_vars"), block)
    return rt or None


def gpu_task_options(num_gpus: float = 1.0) -> dict:
    """Ray remote options for a GPU task: a device share, the runtime_env, and a retry budget.

    `max_retries` reruns a task whose worker or node was lost (spot reclamation) on surviving
    capacity. `retry_exceptions` is deliberately left off: a deterministic application error —
    a device OOM, an untranslatable expression — must be handled immediately rather than
    repeated N times to the same conclusion.

    `max_calls=0` is the one that costs real time to omit. **Ray does not reuse a worker
    between GPU tasks by default** — it tears the process down after each one to guarantee the
    device memory is released — so every shard of a fan-out started a new Python process,
    imported cuDF again, and built the RMM pool again. Measured on a T4 against one 7.3M-row
    shard of TPC-H `lineitem`: 1.07 s to import cuDF, 0.98 s to configure the allocator, 0.26 s
    to read the shard onto the device and **0.15 s to run the kernels**. Two seconds of set-up
    for a sixth of a second of work, paid per shard, on every query.

    It also made a comment elsewhere in this module untrue: `prepare_device_memory` is
    documented as idempotent "so a worker reused across tasks keeps the pool the first one paid
    for", and no worker was ever reused. With reuse, that is finally the behaviour.

    The device memory Ray's default protects against is memory this path does not leak: each
    task builds cuDF frames and drops them, and the RMM async pool returns freed blocks to the
    driver. A fleet running something that does leak can set `gpu_worker_reuse=False` and get
    the old process-per-task isolation back.

    Args:
        num_gpus: The device share to request. `1.0` — the default, and what every caller that
            has not measured its shards passes — is one whole device, which is what this path
            asked for unconditionally before. A fraction from `resources.shard_task_share` lets
            several shards of a deliberately oversubscribed fan-out run on one device instead
            of queueing behind each other. Values at or below zero are refused rather than
            passed through: Ray reads `num_gpus=0` as a CPU task, so a mis-derived share would
            schedule a cuDF kernel onto a node with no device at all.

    Returns:
        The options dict, ready for `ray.remote(**opts)`.
    """
    from batcher.config import active_config

    dc = active_config().distributed
    opts: dict = {
        "num_gpus": num_gpus if num_gpus > 0 else 1.0,
        # No CPU reservation. Ray gives a task `num_cpus=1` by default, and that default is
        # what deadlocks a distributed GPU query: the shuffle fleet takes its workers in a
        # placement group, so on a cluster fanned out to one worker per core the group holds
        # **every** CPU — and a GPU shard task submitted outside it then waits for a core that
        # will never come free. It does not fail; the query hangs with all four devices idle
        # and `ray status` reporting the cluster fully reserved. Measured on four T4s: TPC-H
        # q1 at 32 partitions hung indefinitely, and completed in 11.5 s at 8.
        #
        # Zero is also the honest figure. The work is on the device; the host thread submits
        # kernels and moves buffers, and concurrency is already bounded by `num_gpus` — the
        # device share is the resource being contended, not the core. It is what Ray's own
        # guidance gives a GPU stage, and what Ray Data requests for its own.
        "num_cpus": 0,
        "max_retries": int(dc.task_max_retries),
    }
    if dc.gpu_worker_reuse:
        opts["max_calls"] = 0
    rt = gpu_task_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    return opts
