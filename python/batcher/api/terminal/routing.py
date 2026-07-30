"""The `distributed="auto"` routing decision for terminal operations.

Whether a terminal (`collect`, `iter_batches`, …) runs single-node or fans out across the
Ray cluster. The decision is size-aware, not just topology-aware: distributing has a ~2 s
fixed fan-out cost, so a small query is far faster single-node even on a big cluster. This
is the control-plane "make it adaptive" answer to the sub-second-small-query mandate — it
decides *where* to run, never *what* the result is (single-node == distributed is byte
-identical).
"""

from __future__ import annotations

import logging
import sys

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["resolve_distributed"]


def _ray_already_live() -> bool:
    """Whether Ray could possibly be initialized in this process — a `sys.modules` lookup.

    ``distributed="auto"`` runs on every terminal op, and its first act was ``import ray``
    so it could ask ``ray.is_initialized()``. Where Ray is *installed but not running* —
    every single-node script on a Ray or Anyscale image, which is the environment this
    engine ships into — that import cost **444 ms on the first `collect()` of the process**
    to compute the answer "no". A 3-row local query paid the entire Ray runtime import to
    be told to run locally.

    Checking `sys.modules` is not an approximation of that question, it is the same
    question: ``ray.is_initialized()`` can only return True if this process called
    ``ray.init()`` (or is a Ray worker), and both require ``ray`` to have been imported
    already. So an absent `sys.modules` entry proves the import would return False, and the
    only thing skipping it loses is the cost. Ray installed but never imported, Ray not
    installed, and Ray broken all reach the same single-node answer they did before.
    """
    return "ray" in sys.modules


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
    if not _ray_already_live():
        return False  # nothing imported Ray, so nothing initialized it — single-node
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
        # Data already resident in THIS process never distributes on `auto`, at any size.
        # The row-count threshold below asks "is there enough work to spread?", which is the
        # wrong question for resident data: the work is not the problem, the *data movement*
        # is. Distributing it ships every batch out of the driver and gathers the result
        # back, and that costs far more than the compute it parallelizes — a 6M-row grouped
        # SUM over an in-memory table measures 45 ms single-node vs 1031 ms distributed
        # (23x) on a 128-CPU cluster. Distribution pays when the workers do the *reading*
        # themselves (the same cluster turns a 4.94x loss into a 0.60x win on an S3-backed
        # sf10 scan), which is exactly the file-backed case this leaves alone.
        #
        # This is not a rare shape: any process where something else has initialized Ray —
        # a Daft or Ray Data comparison in the same script, a Ray-using library, an
        # Anyscale workspace — flips `auto` to True and silently pays the 23x. A GPU stage
        # still distributes (checked above): it must reach the cluster's accelerators, and
        # that is a capability need, not a throughput bet.
        if sources and all(getattr(s, "resident", False) for s in sources):
            return False
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
    except Exception as e:
        # "Auto" is a best-effort *routing* decision, so any failure here degrades to the
        # always-correct single-node answer rather than failing the query. But the failure
        # modes are not all equivalent: "no cluster" and "the configured cluster is
        # unreachable / its topology could not be read" both land here and both look like a
        # query that merely chose not to distribute. Log the cause so a real
        # misconfiguration is diagnosable instead of silently costing the user their
        # cluster. Debug level — on a machine with no Ray this is the expected path, not a
        # problem worth warning about on every query.
        from batcher._internal.logging import get_logger, log_kv

        log_kv(
            get_logger("api"),
            logging.DEBUG,
            "distributed=auto resolved to single-node",
            reason=type(e).__name__,
            detail=str(e),
        )
        return False


def _learned_size(plan: LogicalPlan | None) -> float | None:
    """The measured output rows learned for `plan`'s signature, or `None` cold. Best-effort."""
    if plan is None:
        return None
    from batcher import core
    from batcher.api.tuning import learned_output_rows

    return learned_output_rows(core.default_hub(), plan)


def _plan_has_gpu_stage(plan: LogicalPlan) -> bool:
    """Whether any `map_batches` stage requests an accelerator (so the query must distribute
    to reach the cluster's accelerators, whatever its input size).

    Both request forms count. Ray reports NVIDIA/AMD/Intel/MetaX as the `GPU` resource, which
    is `num_gpus`; every other accelerator — `TPU`, `neuron_cores` (Trainium/Inferentia),
    `HPU` (Gaudi), `NPU` — is a *custom* resource and carries `num_gpus == 0`. Checking only
    `num_gpus` therefore made exactly the non-CUDA accelerators invisible here, so a TPU or
    Trainium stage was routed by input size alone and ran locally on the CPU-only driver,
    never reaching the accelerator nodes it asked for.
    """
    from batcher.plan.logical import MapBatches

    node: LogicalPlan | None = plan
    while node is not None:
        if isinstance(node, MapBatches) and (
            getattr(node, "num_gpus", 0) > 0 or getattr(node, "resources", ())
        ):
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
