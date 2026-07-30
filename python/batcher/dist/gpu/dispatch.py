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

__all__ = ["gpu_chain_on_worker", "gpu_join_on_worker", "whole_source_descriptor"]


def whole_source_descriptor(source: Source) -> dict | None:
    """One descriptor covering all of `source`, or `None` when it cannot be described.

    `None` means the source is in-memory (its rows are already on the driver, so there is
    nothing to save) or the cluster is unreadable.
    """
    try:
        import ray

        from batcher.dist.executors.partition_io import (
            WholeSourceSplit,
            _scan_splits,
            partition_descriptors,
        )
    except Exception as exc:
        note_suppressed("dist", "import the GPU dispatch dependencies", exc)
        return None
    if not ray.is_initialized():
        return None
    splits = _scan_splits(source, 1)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return None
    descriptors = partition_descriptors(source, 1)
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


def _remote(task, *args) -> pa.Table | None:
    """Run `task` on one GPU worker, returning `None` on any failure.

    Every failure here — no GPU node, a worker without cuDF, a device OOM, an untranslatable
    expression — has the same correct answer: use the CPU engine. Distinguishing them would
    only let a caller act on a distinction that does not change what it should do.
    """
    try:
        import ray

        from batcher.dist.executors.ray_runtime import _ensure_ray
        from batcher.dist.gpu.tasks import gpu_task_options

        _ensure_ray(1)
        return ray.get(ray.remote(**gpu_task_options())(task).remote(*args))
    except Exception as exc:
        note_suppressed("dist", "dispatch a GPU chain to a worker", exc)
        return None
