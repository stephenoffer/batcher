"""Checks a UDF stage must pass against the live Ray cluster before it is submitted.

Two distributed failures of `map_batches`/`map`/`flat_map`/`filter(fn)` used to surface
badly, and both are decided by the cluster rather than by the plan, so neither can be
caught when the stage is defined:

- **A device request no node can meet.** ``num_gpus=1`` (or a custom accelerator in
  ``resources=``) on a cluster without that resource asks Ray for a worker that can never
  be placed, and the query waited forever with nothing said. Single-node has no scheduler
  to wait on, so there the stage runs its `fn` in this process with a `PerformanceWarning`
  (`api.terminal.routing`). On a cluster the request is binding, so it is refused here,
  naming the request and what the cluster offers.
- **A `fn` Ray cannot pickle.** A closure over a lock, a socket or an open client failed
  with Ray's 3 KB serialization dump, which led with a `repr` of the whole plan.
  `unpicklable_udf_error` names the stage and the captured variable instead.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.logical import LogicalPlan, MapBatches

__all__ = ["require_placeable_accelerators", "unpicklable_udf_error"]


def _map_stages(plan: LogicalPlan) -> Iterator[MapBatches]:
    """Every UDF stage in `plan`, across every branch."""
    from batcher.plan.visitor import children

    stack = [plan]
    while stack:
        node = stack.pop()
        if isinstance(node, MapBatches):
            yield node
        stack.extend(children(node))


def _label(fn: object) -> str:
    """A readable name for a stage's `fn`, for an error message."""
    inner = getattr(fn, "fn", None)  # the row / filter / binding adapters wrap the user's fn
    target = inner if callable(inner) else fn
    return getattr(target, "__qualname__", None) or type(target).__name__


def _requests(stage: MapBatches) -> list[tuple[str, str, float]]:
    """The stage's device requests as ``(argument, Ray resource, amount per worker)``."""
    out = [("num_gpus", "GPU", stage.num_gpus)] if stage.num_gpus > 0 else []
    out += [(f"resources[{name!r}]", name, amount) for name, amount in stage.resources]
    return out


def _cluster_is_fixed(ray: Any) -> bool:
    """Whether no autoscaler can add a node: a local cluster, or no autoscaling signal.

    A cluster this process started with `ray.init()` has no autoscaler. An attached cluster
    is given the benefit of the doubt when the environment says it autoscales, because a
    GPU node that is not up *yet* is exactly what the distributed path's autoscale request
    exists to bring up.
    """
    from batcher.config.profiles import detect_autoscaling_environment

    try:
        started_here = bool(ray._private.worker._global_node.head)
    except AttributeError:
        started_here = False
    return started_here or not detect_autoscaling_environment()


def require_placeable_accelerators(plan: LogicalPlan) -> None:
    """Refuse a UDF stage whose device request no alive node of a fixed cluster can meet.

    Args:
        plan: The plan about to be submitted to the cluster.

    Raises:
        PlanError: If a stage asks for more of a resource per worker than any alive node
            offers, on a cluster that cannot grow. The message names the argument, the
            amount, and the cluster's resources.
    """
    stages = [(stage, req) for stage in _map_stages(plan) for req in _requests(stage)]
    if not stages:
        return
    import ray

    if not ray.is_initialized() or not _cluster_is_fixed(ray):
        return
    nodes = [n.get("Resources", {}) for n in ray.nodes() if n.get("Alive")]
    for stage, (arg, resource, amount) in stages:
        best = max((float(r.get(resource, 0.0)) for r in nodes), default=0.0)
        if best >= amount:
            continue
        totals = ray.cluster_resources()
        offered = ", ".join(
            f"{k}={v:g}" for k, v in sorted(totals.items()) if k in ("CPU", "GPU", resource)
        )
        raise PlanError(
            f"map_batches stage {_label(stage.fn)!r} asks for {arg}={amount:g} per worker, "
            f"but no node of this Ray cluster offers that much {resource} (the most any "
            f"node has is {best:g}; the cluster offers {offered or 'no such resource'}), "
            "so no worker could ever be placed and the query would wait forever. Drop "
            f"{arg} to run the fn on CPU, run on a cluster with {resource} nodes, or collect "
            "with distributed=False to run it in this process."
        )


def unpicklable_udf_error(plan: LogicalPlan, exc: BaseException) -> PlanError | None:
    """A short, actionable error for a UDF `fn` Ray could not serialize, or `None`.

    Args:
        plan: The plan whose submission failed.
        exc: What the submission raised.

    Returns:
        A `PlanError` naming the stage and the captured variable when a stage's `fn` is what
        failed to pickle; `None` when the failure is something else, for the caller to
        re-raise unchanged.
    """
    if "serializ" not in str(exc):
        return None
    from ray import cloudpickle

    for stage in _map_stages(plan):
        try:
            cloudpickle.dumps(stage.fn)
        except Exception as err:
            names = _captured_names(stage.fn)
            captured = f"the variable(s) {names}" if names else "an object"
            reason = str(err).splitlines()[0][:160] if str(err) else type(err).__name__
            return PlanError(
                f"the map_batches fn {_label(stage.fn)!r} cannot be sent to the Ray workers: "
                f"it captures {captured} that cannot be pickled ({reason}). Create that "
                "object inside the fn, or in a class fn's __init__ so each worker builds its "
                "own, or collect with distributed=False to run it in this process."
            )
    return None


def _captured_names(fn: object) -> list[str]:
    """The names of the unpicklable variables `fn` captures, as Ray's inspector finds them.

    `inspect_serializability` prints its whole traversal, so its output is discarded. Its
    failure records name the variable as `name` on older Ray and as the last element of
    `path` on newer Ray; either is read, and an inspector that fails answers no names.
    """
    from ray.util import inspect_serializability

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, failures = inspect_serializability(fn, print_file=io.StringIO())
    except Exception:  # the names only decorate the message; never mask the real failure
        return []
    names = set()
    for failure in failures:
        path = getattr(failure, "path", None) or ()
        name = getattr(failure, "name", None) or (path[-1] if path else None)
        if name:
            names.add(str(name))
    return sorted(names)
