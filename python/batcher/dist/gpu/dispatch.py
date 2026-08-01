"""Get a single-device GPU run's *input* to the device without staging it on the driver.

A translated chain that does not shard still has to read its source from somewhere. It used to
be the driver: `list(source.read())` pulled the whole relation into the driver's memory and
then shipped that table to a GPU worker as a task argument. For the queries a GPU is worth
using for, that is the wrong end of the machine — the driver is routinely a small head node,
and it was being asked to hold a dataset chosen precisely because it is large, then move it
twice across the network to compute on it once.

Sending a **partition descriptor** instead moves the read to the worker. The descriptor is a
small manifest of splits with the projection and predicate already pushed into it, so the
worker reads only the columns and files it needs, straight from storage, and the driver never
sees a row. It is the same mechanism the sharded path uses, with a shard count of one.

An in-memory source has no splits to describe, and there the caller's ship-the-table path is
both correct and the only option — the rows are already on the driver by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = [
    "await_gpu_admission",
    "gpu_chain_on_worker",
    "gpu_join_on_worker",
    "gpu_tree_on_worker",
    "gpu_union_on_worker",
    "whole_source_descriptor",
]


def await_gpu_admission(devices: float = 1.0) -> bool:
    """Whether the cluster can actually start a GPU task now, waiting a bounded time for it.

    A GPU task that cannot be placed does not fail — it pends, and `ray.get` on a pending task
    waits for as long as the query is willing to. Declining here is what turns that into a
    slower answer instead of no answer.

    **Only the device is checked.** This gate was written when a GPU shard task took Ray's
    default `num_cpus=1`, and it therefore also required a free core. That requirement outlived
    the task option it described: `gpu_task_options` now requests `num_cpus=0`, because the work
    is on the device and concurrency is already bounded by the device share. Left in, the check
    refused admission on exactly the cluster the zero-CPU request was introduced to rescue — a
    shuffle fleet holding every core in a placement group — so the deadlock it once prevented
    came back as something quieter and no better: the query stalled for the whole admission
    budget and then abandoned the GPU backend, on a fleet whose devices were entirely idle.
    Asking for a resource the task does not request can only ever refuse work that would have
    run.

    Checked against *available* resources rather than the cluster's totals, which is the
    distinction `await_autoscale` deliberately does not make: it asks whether the fleet is big
    enough, and this asks whether any of it is free.

    Args:
        devices: Device share one task needs, so a packed fan-out asking for a quarter of a GPU
            is admitted by a device that is three-quarters busy.

    Returns:
        True when a task could be placed — including whenever the answer cannot be determined,
        because refusing the GPU on an unreadable cluster would disable the backend on every
        deployment whose resource view differs from this one. False only on a positive reading
        that no device is free.
    """
    from batcher.config import active_config

    budget = float(active_config().distributed.gpu_admission_wait_s)
    if budget <= 0:
        return True
    try:
        import ray

        if not ray.is_initialized():
            return True
    except ImportError as exc:  # the `[ray]` extra is not installed
        note_suppressed("dist", "import ray for the GPU admission check", exc)
        return True
    import time

    deadline = time.monotonic() + budget
    poll = min(0.5, max(0.05, budget / 20.0))
    while True:
        free = _free_resources()
        if free is None or free.get("GPU", 0.0) >= devices:
            return True
        if time.monotonic() >= deadline:
            note_suppressed(
                "dist",
                "admit a GPU stage",
                TimeoutError(
                    f"no device free after {budget:.0f}s "
                    f"({free.get('GPU', 0.0):.2f} of a needed {devices:.2f} GPU available)"
                ),
            )
            return False
        time.sleep(poll)


def _free_resources() -> dict | None:
    """Ray's currently *available* resources, or `None` when they cannot be read."""
    try:
        import ray

        return dict(ray.available_resources())
    except Exception as exc:
        note_suppressed("dist", "read the cluster's free resources", exc)
        return None


def whole_source_descriptor(source: Source, projection: list[str] | None = None) -> dict | None:
    """One descriptor covering all of `source`, or `None` when it cannot be described.

    `None` means the source is in-memory (its rows are already on the driver, so there is
    nothing to save) or the cluster is unreadable.

    `projection` narrows the read to the columns the plan uses. It matters most exactly here,
    on a relation every worker reads a whole copy of: the wasted columns are paid for once per
    device rather than once per query.
    """
    # Only Ray is optional; the batcher imports stay outside the `try` so a refactor that moves
    # one fails loudly instead of reading as "this source cannot be described to a worker".
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.executors.partition_io._sources import _scan_splits
    from batcher.io.splits import WholeSourceSplit

    try:
        import ray
    except ImportError as exc:  # the `[ray]` extra is not installed
        note_suppressed("dist", "import ray for the GPU dispatch", exc)
        return None
    if not ray.is_initialized():
        return None
    splits = _scan_splits(source, 1)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return None
    descriptors = partition_descriptors(source, 1, projection=projection)
    return descriptors[0] if descriptors else None


def gpu_chain_on_worker(source: Source, ops: list[dict]) -> pa.Table | None:
    """Run a translated chain on one GPU worker that reads `source` itself.

    Args:
        source: The scan's source.
        ops: The bottom-up operator IR chain.

    Returns:
        The chain's result, or `None` when the source cannot be described to a worker or the
        dispatch failed — the caller then ships the table itself, or uses the CPU engine.
    """
    descriptor = whole_source_descriptor(source)
    if descriptor is None:
        return None
    from batcher.dist.gpu.tasks import gpu_shard_partial

    return _remote(gpu_shard_partial, descriptor, ops)


def gpu_join_on_worker(
    left: Source,
    right: Source,
    left_ops: list[dict],
    right_ops: list[dict],
    join_ir: dict,
    ops: list[dict],
) -> pa.Table | None:
    """Run a translated join on one GPU worker that reads **both** sides itself.

    A join was the worst case for driver staging, because it staged two relations rather than
    one — and the join's whole purpose is that at least one of them is large.

    Args:
        left: The left input's source.
        right: The right input's source.
        left_ops: The left input chain's operator IR.
        right_ops: The right input chain's operator IR.
        join_ir: The join node's IR.
        ops: The operator chain above the join.

    Returns:
        The join's result, or `None` when either side cannot be described to a worker or the
        dispatch failed.
    """
    ldesc = whole_source_descriptor(left)
    rdesc = whole_source_descriptor(right)
    if ldesc is None or rdesc is None:
        return None
    from batcher.dist.gpu.tasks import gpu_join_task

    return _remote(gpu_join_task, ldesc, rdesc, left_ops, right_ops, join_ir, ops)


def gpu_union_on_worker(
    sources: list[Source], input_ops: list[list[dict]], distinct: bool, ops: list[dict]
) -> pa.Table | None:
    """Run a translated union on one GPU worker that reads every input itself.

    Args:
        sources: Each union input's source.
        input_ops: Each input chain's operator IR, positionally matching `sources`.
        distinct: Whether the union deduplicates.
        ops: The operator chain above the union.

    Returns:
        The union's result, or `None` when any input cannot be described to a worker or the
        dispatch failed.
    """
    descriptors = [whole_source_descriptor(s) for s in sources]
    if any(d is None for d in descriptors):
        return None
    from batcher.dist.gpu.tasks import gpu_union_task

    return _remote(gpu_union_task, descriptors, input_ops, distinct, ops)


def gpu_tree_on_worker(spec: dict, sources: list) -> pa.Table | None:
    """Run a whole plan tree on one GPU worker that reads every leaf itself.

    The fallback behind the tree fan-out, for a tree with no splittable leaf or one whose
    replicated side the fan-out would not fit across devices — and, at small scale, the cheaper
    answer outright, since one device that reads four small relations beats sixteen that each
    read three of them.

    Args:
        spec: A GPU plan-tree spec from `gpu_tree_spec`.
        sources: The query's sources, indexed by a leaf's `source_id`.

    Returns:
        The tree's result, or `None` when any leaf cannot be described to a worker (an in-memory
        source, whose rows are on the driver already) or the dispatch failed.
    """
    from batcher.core.gpu_plan.pruning import prune_tree
    from batcher.core.gpu_plan.tree import tree_leaves

    spec, projections = prune_tree(spec)
    descriptors: list[dict] = []
    for leaf in tree_leaves(spec):
        descriptor = whole_source_descriptor(
            sources[leaf["source_id"]], projections.get(leaf["leaf"])
        )
        if descriptor is None:
            return None
        descriptors.append(descriptor)
    from batcher.dist.gpu.tasks import gpu_tree_task

    return _remote(gpu_tree_task, descriptors, spec)


def _remote(task, *args) -> pa.Table | None:
    """Run `task` on one GPU worker, returning `None` on any failure.

    Every failure here — no GPU node, a worker without cuDF, a device OOM, an untranslatable
    expression — has the same correct answer: use the CPU engine. Distinguishing them would
    only let a caller act on a distinction that does not change what it should do.

    The imports stay OUTSIDE that judgement. A moved symbol is not a dispatch failure, and
    letting it read as one is how a whole accelerated path disables itself in silence.
    """
    from batcher.dist.executors.ray_runtime import _ensure_ray
    from batcher.dist.gpu.tasks import gpu_task_options

    try:
        import ray
    except ImportError as exc:  # the `[ray]` extra is not installed
        note_suppressed("dist", "import ray for the GPU dispatch", exc)
        return None
    from batcher.dist.gpu.cudf_probe import mark_cudf_missing

    try:
        _ensure_ray(1)
        if not await_gpu_admission():
            # Nothing is free and nothing is going to be within the budget. Submitting anyway
            # would leave the task PENDING and this `ray.get` waiting on it without a deadline.
            return None
        return ray.get(ray.remote(**gpu_task_options())(task).remote(*args))
    except Exception as exc:
        if mark_cudf_missing(exc):
            # The cuDF probe guessed present and was wrong. It has now recorded otherwise, so
            # a second attempt carries the pip block that installs it — once per session, on
            # the fleets that actually need it rather than on all of them.
            try:
                return ray.get(ray.remote(**gpu_task_options())(task).remote(*args))
            except Exception as retry_exc:
                note_suppressed("dist", "dispatch a GPU chain with cuDF installed", retry_exc)
                return None
        note_suppressed("dist", "dispatch a GPU chain to a worker", exc)
        return None
