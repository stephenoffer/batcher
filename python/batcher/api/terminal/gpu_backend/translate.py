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

from batcher.plan.ir_tags import RUNNING_AGGREGATES

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

import contextlib

from batcher._internal.logging import note_suppressed
from batcher.api.terminal.gpu_backend.fanout import (
    _cluster_gpu_count,
    _try_sharded_aggregate,
    _try_sharded_join,
    _try_sharded_union,
    _try_tree,
)
from batcher.core.gpu_plan.backend import Unsupported
from batcher.dist.gpu import gpu_task_options
from batcher.dist.gpu.groupby import dispatch_gpu_aggregate, distributed_gpu_aggregate


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
        if getattr(ae, "func", None) not in RUNNING_AGGREGATES or not isinstance(ae.input, Col):
            return None
        aggs[spec.alias] = (ae.input.name, ae.func)
    return gk.alias, gk.expr.name, aggs, plan.input


def _one_device_is_enough(decision, read_bytes: float = 0.0) -> bool:
    """Whether running this plan on a *single* device is a defensible place to put it.

    The gate on the translator's second rung. When a fan-out declines — a join whose build side
    is too large to replicate, a chain with no mergeable reducer — the next thing tried is one
    worker running the entire query on one board. For a small plan that is right, and the tree
    dispatcher says why: one device reading four small relations beats sixteen that each read
    three of them. For a large one it is one board against forty-eight cores.

    Two of the three conditions come from Kyber, which has already decided how much of the fleet
    this plan wants: `desired_gpus` is the devices that would hold the working set in one wave,
    and `distributed` is whether spreading is worth it at all. When either says more than one, a
    single device is knowingly a fraction of what the plan was routed to the accelerator *for*.

    The third is `read_bytes` — what one device would actually have to **read**, measured from
    the source's own row count and the pruned projection rather than estimated. The rule is that
    one device may do the work of *one wave of the fan-out*: at most `gpu_min_shard_bytes` per
    device, times the devices. It needs no new constant, and it says something coherent — the
    rung is a substitute for a fan-out, so it is worth taking while it is no larger than the
    fan-out's own first wave.

    That threshold is where the measurements separate, and they separate sharply. Holding
    everything else constant on a six-T4 fleet, with the single-device rung's projection fixed:

    | query | one device reads | on one device | declined to the CPU |
    |---|---:|---:|---:|
    | ClickBench q08 | 0.10 GB | **2.21x** | 0.89x |
    | ClickBench q10 | 0.58 GB | **1.37x** | 0.87x |
    | ClickBench q13 | 0.58 GB | 0.88x | 1.00x |
    | ClickBench q22 | 1.60 GB | **0.04x** | 0.86x |

    An earlier version of this gate used "the plan is not shardable" instead, on the strength of
    ClickBench q08 through q14 measuring 10.5 s on one device. That was the right observation
    and the wrong cause: those queries were slow because the rung read all **105** columns of
    `hits` to answer a two-column question, and once that was fixed the same queries were 2.21x
    *wins* on one device. Shardability turned out to be uncorrelated — it also declined TPC-H
    q16 (2.95x on one device) and q20 (1.48x).

    Args:
        decision: Kyber's verdict for this plan.
        read_bytes: What one device would read, or `0.0` when it cannot be measured — which is
            treated as "small", keeping the behaviour this rung had before the size test.

    Returns:
        True when one device is a defensible place for this plan.
    """
    if decision.distributed or int(decision.desired_gpus) > 1:
        return False
    return read_bytes <= _one_wave_bytes()


def _one_wave_bytes() -> float:
    """What one wave of the fan-out would read: the shard floor, times the fleet's devices.

    The shard floor is `gpu_min_shard_bytes` — the size below which "the Ray dispatch that
    delivers a shard costs more than the shard's own compute" — so a wave of them is the
    smallest amount of work the fan-out considers worth spreading. A single device doing that
    much is doing what the fan-out would have done, without the dispatches.
    """
    from batcher.config import active_config

    dc = active_config().distributed
    return float(max(1, int(dc.gpu_min_shard_bytes))) * max(1, _cluster_gpu_count())


def _chain_read_bytes(source, ops: list[dict]) -> float:
    """Bytes one device would read for a chain over `source`, from its footer and projection."""
    from batcher.core.gpu_plan.pruning import chain_projection
    from batcher.dist.gpu.shards import source_bytes

    try:
        return float(source_bytes(source, chain_projection(ops)))
    except Exception as exc:
        note_suppressed("api", "measure what one device would read", exc)
        return 0.0


def _rehearsal_reason(plan: LogicalPlan, sources: list[Source]) -> str | None:
    """Why this plan will not translate, found on the driver rather than on a device.

    A GPU worker discovers an untranslatable expression by raising `Unsupported` *after* it has
    been started, given a shard and read it — and the caller then tries the next rung of the
    ladder, which is another worker, which raises the same thing. Measured on a six-T4 fleet at
    TPC-H sf10: q13 and q16 spent 0.9 s per attempt to be told `LIKE '%special%requests%'` was
    not translated, and q7 and q17 spent 1.9 s across two attempts, against CPU answers of
    0.44 s and 0.26 s. All of it was asking a device a question the driver could answer.

    The rehearsal replays the same translation on **zero-row** frames built from the sources'
    own schemas, on the host backend, which costs a schema walk and an operator pass.

    It is one-directional in both senses, and both matter:

    * only an `Unsupported` is a decline — any other exception means the rehearsal reached no
      conclusion and the caller proceeds exactly as it would have. A rehearsal may save a round
      trip; it may not *cause* a fallback;
    * pandas' API is a superset of cuDF's, so a plan that rehearses cleanly can still decline on
      a device. That costs what it costs today.

    Args:
        plan: The optimized logical plan.
        sources: The query's sources, indexed by a leaf's `source_id`.

    Returns:
        The reason, or `None` to proceed — including whenever the rehearsal could not run.
    """
    try:
        import pandas as pd
        import pyarrow as pa

        from batcher.core.gpu_plan import DfBackend, gpu_tree_spec
        from batcher.core.gpu_plan.execute import empty_frame, rehearsal_decline
        from batcher.core.gpu_plan.pruning import prune_tree
        from batcher.core.gpu_plan.tree import run_tree, tree_leaves
    except ImportError as exc:  # a driver without pandas rehearses nothing and loses nothing
        note_suppressed("api", "rehearse the GPU translation on the driver", exc)
        return None
    matched = gpu_tree_spec(plan)
    if matched is None:
        # Not translatable as a tree at all, which the matchers below establish for themselves
        # in microseconds. Saying so here would duplicate that decision, not accelerate it.
        return None
    spec, _scans = matched
    be = DfBackend(pd)
    try:
        # **Pruned**, because that is what a worker runs. Rehearsing the unpruned tree gives
        # every leaf its whole schema, so a column the pruning drops is still there and the one
        # class of failure that most needs catching — an operator reading a column the pruning
        # decided nobody reads — cannot occur in the rehearsal. Both of the tree failures found
        # on this fleet (TPC-H q7's `c_nationkey`, q17's `p_partkey`) were exactly that, and an
        # unpruned rehearsal reported both as translating cleanly.
        pruned, projections = prune_tree(spec)
        frames = {}
        for leaf in tree_leaves(pruned):
            schema = sources[leaf["source_id"]].schema()
            columns = projections.get(leaf["leaf"])
            if columns is not None:
                schema = pa.schema([schema.field(name) for name in columns])
            frames[leaf["leaf"]] = empty_frame(schema, be)
    except Exception as exc:
        note_suppressed("api", "build empty frames for the GPU rehearsal", exc)
        return None
    return rehearsal_decline(lambda: run_tree(pruned, frames, be))


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
       construction, or a process that owns a device but has no Ray to schedule with. That
       "only" is now *checked* rather than inferred from step 2 declining, which is the whole
       point: step 2 returns `None` both when the source is in-memory and when the dispatch
       **failed**, so a device that ran out of memory on a 15M-row Parquet relation was answered
       by staging the whole thing on the driver. On a 30 GB head node that is a SIGKILL, and
       TPC-H q4 and q14 at sf10 were exactly that — no traceback, no fallback, the query gone.
       `rows_already_on_driver` asks the question directly.
    """
    import pyarrow as pa

    # Ask on the driver, before any device is started, whether this plan translates at all.
    reason = _rehearsal_reason(plan, sources)
    if reason is not None:
        note_suppressed(
            "api",
            "translate this plan for the GPU (rehearsed on the driver)",
            Unsupported(reason),
        )
        return None

    from batcher.core.gpu_plan import gpu_join_spec, gpu_plan_ops, gpu_union_spec
    from batcher.dist.gpu import gpu_chain_on_worker, gpu_join_on_worker, gpu_union_on_worker
    from batcher.dist.gpu.dispatch import rows_already_on_driver

    plan_spec = gpu_plan_ops(plan)
    if plan_spec is not None:
        scan, ops = plan_spec
        source = sources[scan.source_id]
        fanned = _try_sharded_aggregate(source, ops, gpu_count, decision)
        if fanned is not None:
            return fanned
        if not _one_device_is_enough(decision, _chain_read_bytes(source, ops)):
            return None
        on_worker = gpu_chain_on_worker(source, ops)
        if on_worker is not None:
            return on_worker
        if not rows_already_on_driver(source):
            return None
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
        both = _chain_read_bytes(left, lops) + _chain_read_bytes(right, rops)
        if not _one_device_is_enough(decision, both):
            return None
        on_worker = gpu_join_on_worker(left, right, lops, rops, join_ir, ops)
        if on_worker is not None:
            return on_worker
        if not (rows_already_on_driver(left) and rows_already_on_driver(right)):
            return None
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
        total = sum(_chain_read_bytes(s, o) for s, o in zip(usources, input_ops, strict=True))
        if not _one_device_is_enough(decision, total):
            return None
        on_worker = gpu_union_on_worker(usources, input_ops, distinct, ops)
        if on_worker is not None:
            return on_worker
        if not all(rows_already_on_driver(s) for s in usources):
            return None
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
    if not _one_device_is_enough(decision, _tree_read_bytes(spec, sources)):
        return None
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


def _tree_read_bytes(spec: dict, sources: list[Source]) -> float:
    """Bytes one device would read for a whole plan tree: every leaf, at its pruned width."""
    from batcher.core.gpu_plan.pruning import prune_tree
    from batcher.core.gpu_plan.tree import tree_leaves
    from batcher.dist.gpu.shards import source_bytes

    try:
        pruned, projections = prune_tree(spec)
        return float(
            sum(
                source_bytes(sources[leaf["source_id"]], projections.get(leaf["leaf"]))
                for leaf in tree_leaves(pruned)
            )
        )
    except Exception as exc:
        note_suppressed("api", "measure what one device would read for this tree", exc)
        return 0.0
