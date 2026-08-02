"""Does a plan need an accelerator? — the one predicate, for every layer that asks.

Two layers ask the same question for different reasons, and both must get the same answer.
The `api` router asks it to decide whether `distributed="auto"` should fan out at all (GPU
work has to reach the cluster's devices, whatever the input size). The `dist` dispatcher
asks it when it is about to fall back to single-node, because a fallback that is harmless
for CPU work silently runs a model on the driver's CPU when the stage asked for a device.

It lives here, in the neutral contracts layer, because `api` and `dist` cannot share
anything else — and a predicate over `LogicalPlan` copied into both is exactly the drift
that makes one of them wrong later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.plan.logical import LogicalPlan

__all__ = ["plan_requests_accelerator"]


def plan_requests_accelerator(plan: LogicalPlan | None) -> bool:
    """Whether any `map_batches` stage in `plan` asks for an accelerator.

    Both request forms count. Ray reports NVIDIA/AMD/Intel/MetaX devices as the ``GPU``
    resource, which is what `num_gpus` carries; every other accelerator — ``TPU``,
    ``neuron_cores`` (Trainium/Inferentia), ``HPU`` (Gaudi), ``NPU`` — is a *custom*
    resource and leaves ``num_gpus == 0``. Checking only `num_gpus` therefore makes exactly
    the non-CUDA accelerators invisible, which is how a TPU stage ends up routed by input
    size alone and running on a CPU-only driver.

    The walk is over `plan.visitor.children`, so it descends **every** branch. Following the
    single `input` chain instead — which is what the router did — cannot see a stage on the
    build side of a join, and a pipeline that embeds its inference under a join is precisely
    the shape that most needs to reach the cluster's devices.

    Args:
        plan: The plan to inspect; ``None`` is not an accelerator plan.

    Returns:
        ``True`` if some stage requested a device.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.accelerator import plan_requests_accelerator
            >>> ds = bt.from_pydict({"x": [1]})
            >>> plan_requests_accelerator(ds._plan)
            False
            >>> plan_requests_accelerator(ds.ml.map_batches(lambda b: b, num_gpus=1)._plan)
            True
    """
    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import children

    if plan is None:
        return False
    stack: list[LogicalPlan] = [plan]
    while stack:
        node = stack.pop()
        if isinstance(node, MapBatches) and (
            getattr(node, "num_gpus", 0) > 0 or getattr(node, "resources", ())
        ):
            return True
        stack.extend(children(node))
    return False
