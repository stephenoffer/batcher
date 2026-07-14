"""The `distributed="auto"` routing decision for terminal operations.

Whether a terminal (`collect`, `iter_batches`, …) runs single-node or fans out across the
Ray cluster. The decision is size-aware, not just topology-aware: distributing has a ~2 s
fixed fan-out cost, so a small query is far faster single-node even on a big cluster. This
is the control-plane "make it adaptive" answer to the sub-second-small-query mandate — it
decides *where* to run, never *what* the result is (single-node == distributed is byte
-identical).
"""

from __future__ import annotations

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["resolve_distributed"]


def resolve_distributed(
    distributed: bool | str,
    plan: LogicalPlan | None = None,
    sources: list[Source] | None = None,
) -> bool:
    """Resolve ``distributed="auto"``: distribute only when it PAYS.

    On a multi-node cluster the Ray fan-out is a ~2 s fixed cost, so a small query is far
    faster single-node. "auto" therefore distributes only when connected to a multi-node
    cluster AND either the plan has a GPU stage (which must reach the cluster's GPUs) or the
    estimated input is at least ``distributed.distribute_min_rows`` (an unknown size
    distributes, staying safe for large data). The result is identical either way. Never
    forces a Ray init for a local query; an explicit ``True``/``False`` always wins.
    """
    if distributed != "auto":
        return bool(distributed)
    try:
        import ray

        if not ray.is_initialized():
            return False
        from batcher import dist

        topology = dist.cluster_topology()
        if topology["nodes"] <= 1:
            return False
        # GPU work must reach the cluster's accelerators regardless of size — but only if the
        # cluster HAS any. Routing a `num_gpus=1` stage to a GPU-less cluster asks Ray for a
        # resource no node can ever offer, and the task simply never schedules:
        # `TaskUnschedulableError`, or a hang, from a query that would have run fine on this
        # process. The same plan already runs locally when Ray is not up, so falling through
        # to the size decision here is the consistent answer, not a special case.
        has_gpus = topology.get("gpus", 0.0) > 0
        if has_gpus and plan is not None and _plan_has_gpu_stage(plan):
            return True
        from batcher.config import active_config

        min_rows = active_config().distributed.distribute_min_rows
        # Prefer the *measured* size this exact shape produced on past runs over a first-run
        # source estimate: a recurring query that proved small stays single-node (dodging the
        # fan-out tax) even when a source can't cheaply report a row count, and one that proved
        # large distributes. Cold (or no plan) → the source-estimate path below, unchanged.
        learned = _learned_size(plan)
        if learned is not None:
            return learned >= min_rows
        if sources is None:
            return True
        rows = _estimated_input_rows(sources)
        return rows is None or rows >= min_rows
    except Exception:
        return False


def _learned_size(plan: LogicalPlan | None) -> float | None:
    """The measured output rows learned for `plan`'s signature, or `None` cold. Best-effort."""
    if plan is None:
        return None
    from batcher import core
    from batcher.api.tuning import learned_output_rows

    return learned_output_rows(core.default_hub(), plan)


def _plan_has_gpu_stage(plan: LogicalPlan) -> bool:
    """Whether any `map_batches` stage requests a GPU (so the query must distribute to
    reach the cluster's accelerators, whatever its input size)."""
    from batcher.plan.logical import MapBatches

    node: LogicalPlan | None = plan
    while node is not None:
        if isinstance(node, MapBatches) and getattr(node, "num_gpus", 0) > 0:
            return True
        node = getattr(node, "input", None)
    return False


def _estimated_input_rows(sources: list[Source]) -> int | None:
    """Total estimated input rows across `sources` (cheap — a Parquet footer read), or
    `None` if any source can't cheaply report one (→ distribute, staying safe for large
    or unknown data)."""
    total = 0
    for s in sources:
        rc = s.row_count() if hasattr(s, "row_count") else None
        if rc is None:
            return None
        total += rc
    return total
