"""Match a plan to a translated GPU execution, from the most specific shape to the general one.

Four attempts, and the order is the point: the three fixed shapes — a chain over one scan, one
join, one union — each have a fan-out purpose-built for them, so they are tried first and get
the better plan. Anything else goes to the general tree translator, which handles a plan of any
depth. Measured on TPC-H, the fixed matchers claim nine of the twenty-two queries and the tree
claims the other thirteen, every one of which was previously refused for joining three relations
instead of two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

import contextlib

from batcher.api.terminal.gpu_backend.fanout import (
    _try_sharded_aggregate,
    _try_sharded_join,
    _try_sharded_union,
    _try_tree,
)
from batcher.dist.gpu import gpu_task_options
from batcher.dist.gpu.groupby import dispatch_gpu_aggregate, distributed_gpu_aggregate

_GPU_AGGS = ("sum", "count", "mean", "min", "max")


def _gpu_agg_spec(plan: LogicalPlan):
    """`(key_alias, key_col, {alias: (col, func)}, scan)` if `plan` is a GPU-executable
    single-key group-by aggregate directly over a scan, else `None`."""
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Aggregate, Scan

    if not isinstance(plan, Aggregate) or not isinstance(plan.input, Scan):
        return None
    if len(plan.group_keys) != 1:
        return None
    gk = plan.group_keys[0]
    if not isinstance(gk.expr, Col):
        return None
    aggs: dict[str, tuple[str, str]] = {}
    for spec in plan.aggregates:
        ae = spec.agg
        if getattr(ae, "func", None) not in _GPU_AGGS or not isinstance(ae.input, Col):
            return None
        aggs[spec.alias] = (ae.input.name, ae.func)
    return gk.alias, gk.expr.name, aggs, plan.input


def _translated(plan: LogicalPlan, sources: list[Source], gpu_count: int, decision):
    """Run `plan` through the plan translator, or `None` when it does not apply.

    Three attempts in descending order of ambition, each a fallback for the one before:

    1. **Fan out.** Every chain whose shape divides — a mergeable reducer to fold, or a
       row-local chain to reassemble — runs a shard per device, each device reading its own
       shard from storage.
    2. **One worker, reading for itself.** Reached when the fan-out declined or failed, so this
       is a retry as a single shard rather than the usual path. It still keeps the source off
       the driver, which is the point: staging a large relation on the driver to send it to a
       GPU is the wrong end of the machine, and the driver is routinely the smallest node.
    3. **Ship the table.** Only for an in-memory source, whose rows are on the driver by
       construction, or a process that owns a device but has no Ray to schedule with.
    """
    import pyarrow as pa

    from batcher.core.gpu_plan import gpu_join_spec, gpu_plan_ops, gpu_union_spec
    from batcher.dist.gpu import gpu_chain_on_worker, gpu_join_on_worker, gpu_union_on_worker

    plan_spec = gpu_plan_ops(plan)
    if plan_spec is not None:
        scan, ops = plan_spec
        source = sources[scan.source_id]
        fanned = _try_sharded_aggregate(source, ops, gpu_count, decision)
        if fanned is not None:
            return fanned
        on_worker = gpu_chain_on_worker(source, ops)
        if on_worker is not None:
            return on_worker
        batches = list(source.read())  # in-memory source: the rows are on the driver already
        if not batches:
            return None
        return _dispatch_cudf_plan(pa.Table.from_batches(batches), ops)

    # A `[ops] over Join(chain, chain)`. A join the planner marked `broadcast` splits its probe
    # side across every device, each reading the whole build side itself; anything else runs on
    # one GPU, which reads both sides itself.
    join_spec = gpu_join_spec(plan)
    if join_spec is not None:
        (lscan, lops), (rscan, rops), join_ir, ops = join_spec
        left, right = sources[lscan.source_id], sources[rscan.source_id]
        fanned = _try_sharded_join(left, right, lops, rops, join_ir, ops, gpu_count, decision)
        if fanned is not None:
            return fanned
        on_worker = gpu_join_on_worker(left, right, lops, rops, join_ir, ops)
        if on_worker is not None:
            return on_worker
        lb, rb = list(left.read()), list(right.read())
        if not lb or not rb:
            return None
        return _dispatch_cudf_join(
            pa.Table.from_batches(lb), pa.Table.from_batches(rb), lops, rops, join_ir, ops
        )

    # A `[ops] over Union(chains)` — concat (+ optional dedup) + a chain. A `UNION ALL` shards
    # each of its inputs across every device; anything else runs on one GPU, which reads every
    # input itself.
    union_spec = gpu_union_spec(plan)
    if union_spec is not None:
        inputs, distinct, ops = union_spec
        usources = [sources[sc.source_id] for sc, _ in inputs]
        input_ops = [o for _, o in inputs]
        fanned = _try_sharded_union(usources, input_ops, distinct, ops, gpu_count, decision)
        if fanned is not None:
            return fanned
        on_worker = gpu_union_on_worker(usources, input_ops, distinct, ops)
        if on_worker is not None:
            return on_worker
        read = [(list(sources[sc.source_id].read()), iops) for sc, iops in inputs]
        tables = [(pa.Table.from_batches(b), iops) for b, iops in read if b]
        if not tables:
            return None
        return _dispatch_cudf_union([t for t, _ in tables], [o for _, o in tables], distinct, ops)

    # Anything else: a tree of scans, joins and unions of any depth. The three matchers above
    # are the shapes with a fan-out purpose-built for them; this is the general form, and it is
    # what a multi-way analytical query actually is. Measured on TPC-H, the fixed matchers claim
    # nine of the twenty-two queries and this claims the other thirteen — every one of which was
    # refused for joining three relations instead of two.
    return _translated_tree(plan, sources, gpu_count, decision)


def _translated_tree(plan: LogicalPlan, sources: list[Source], gpu_count: int, decision):
    """Run any translatable plan tree on the GPUs, or `None` when it cannot be run there."""
    from batcher.core.gpu_plan import gpu_tree_spec

    matched = gpu_tree_spec(plan)
    if matched is None:
        return None
    spec, _scans = matched
    fanned = _try_tree(spec, sources, gpu_count, decision)
    if fanned is not None:
        return fanned
    from batcher.dist.gpu.dispatch import gpu_tree_on_worker

    return gpu_tree_on_worker(spec, sources)


def _legacy_groupby(plan: LogicalPlan, sources: list[Source], decision):
    """The single-key group-by fan-out, for a fleet whose workers have torch but not cuDF.

    Reached only when the translator declined or could not run — which on a normal cluster is
    never, since cuDF ships with the task's runtime_env. Its kernel falls back to a torch
    scatter-reduce, so it is the difference between an accelerated group-by and none at all on
    a RAPIDS-less fleet.
    """
    import pyarrow as pa

    spec = _gpu_agg_spec(plan)
    if spec is None:
        return None
    key_out, key_src, aggs, scan = spec
    result = distributed_gpu_aggregate(
        sources[scan.source_id], key_src, aggs, sharded=decision.distributed
    )
    if result is None:
        batches = list(sources[scan.source_id].read())
        if not batches:
            return None
        result = dispatch_gpu_aggregate(pa.Table.from_batches(batches), key_src, aggs)
    if key_out != key_src and key_src in result.column_names:
        return result.rename_columns([key_out if n == key_src else n for n in result.column_names])
    return result


def _on_device():
    """Configure this worker's device allocator before it computes.

    Runs inside the task body rather than at submission: only the process that was handed a
    device knows how much of it is free, and the pool it builds has to live in that process.
    Idempotent, so a worker reused across tasks keeps the pool the first one paid for.
    """
    from batcher.carbonite.accel import prepare_device_memory

    prepare_device_memory()


def _cudf_plan_worker(table, ops):
    from batcher.core.gpu_plan import execute_cudf_plan

    _on_device()
    return execute_cudf_plan(table, ops)


def _cudf_join_worker(left_t, right_t, left_ops, right_ops, join_ir, ops):
    from batcher.core.gpu_plan import execute_cudf_join

    _on_device()
    return execute_cudf_join(left_t, right_t, left_ops, right_ops, join_ir, ops)


def _cudf_union_worker(tables, input_ops, distinct, ops):
    from batcher.core.gpu_plan import execute_cudf_union

    _on_device()
    return execute_cudf_union(tables, input_ops, distinct, ops)


def _dispatch_cudf_union(
    tables: list, input_ops: list[list[dict]], distinct: bool, ops: list[dict]
) -> pa.Table | None:
    """Run a union (+ op chain) on one GPU via cuDF; `None` on failure → CPU fallback."""
    return _dispatch_on_gpu(_cudf_union_worker, tables, input_ops, distinct, ops)


def _dispatch_cudf_join(
    left_t, right_t, left_ops: list[dict], right_ops: list[dict], join_ir: dict, ops: list[dict]
) -> pa.Table | None:
    """Run a join + op chain on one GPU via cuDF (in-process if this process owns a GPU with
    cuDF, else a GPU worker). `None` on any failure → CPU fallback."""
    return _dispatch_on_gpu(_cudf_join_worker, left_t, right_t, left_ops, right_ops, join_ir, ops)


def _dispatch_cudf_plan(table: pa.Table, ops: list[dict]) -> pa.Table | None:
    """Run a translated op chain on ONE GPU via cuDF; `None` on failure → CPU fallback."""
    return _dispatch_on_gpu(_cudf_plan_worker, table, ops)


def _dispatch_on_gpu(worker, *args) -> pa.Table | None:
    """Run `worker(*args)` on a GPU — in-process when this process owns one with cuDF, else on
    a GPU worker (cuDF shipped in the runtime_env).

    Returns `None` on any failure — an untranslatable expression, a cuDF-less worker, or a GPU
    OOM — so the caller uses the CPU engine. GPU is an accelerator, never a requirement, and
    every dispatch shape (chain, join, union) needs exactly this behavior, so they share it
    rather than restating it three times with three chances to drift."""
    from batcher.core.gpu_transform import gpu_available

    try:
        if gpu_available():
            with contextlib.suppress(Exception):
                return worker(*args)  # GPU-equipped process with cuDF
        import ray

        from batcher.dist.executors.ray_runtime import _ensure_ray

        _ensure_ray(1)
        return ray.get(ray.remote(**gpu_task_options())(worker).remote(*args))
    except Exception:
        return None
