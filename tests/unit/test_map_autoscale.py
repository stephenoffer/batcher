"""Actor-pool autoscaling + accelerator pinning for `map_batches` (no Ray needed).

Unit-tests the pure scheduling helpers and the API-edge validation: a ``(min, max)``
`concurrency` resolves to a workload-clamped pool size, `accelerator_type` flows onto
the `MapBatches` node and into Ray `.options(...)` only alongside a GPU request.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def test_resolve_pool_size():
    from batcher.dist.executors.map import _resolve_pool_size

    assert _resolve_pool_size(None, 10, 4) == 4  # default to worker count
    assert _resolve_pool_size(3, 10, 4) == 3  # fixed int
    assert _resolve_pool_size((2, 8), 5, 4) == 5  # workload within [2, 8]
    assert _resolve_pool_size((2, 8), 20, 4) == 8  # capped at max
    assert _resolve_pool_size((4, 8), 2, 4) == 4  # floored at min


def test_autoscale_action_decisions():
    from batcher.dist.executors.map import _autoscale_action

    # backlog with headroom → grow
    assert _autoscale_action(pending=5, n_actors=2, n_idle=0, min_size=2, max_size=8) == "up"
    # backlog but already at max → hold
    assert _autoscale_action(pending=5, n_actors=8, n_idle=0, min_size=2, max_size=8) == "hold"
    # drained backlog, idle actors above the floor → shrink
    assert _autoscale_action(pending=0, n_actors=5, n_idle=2, min_size=2, max_size=8) == "down"
    # drained at the floor → hold
    assert _autoscale_action(pending=0, n_actors=2, n_idle=2, min_size=2, max_size=8) == "hold"
    # drained but all actors busy (none idle) → hold
    assert _autoscale_action(pending=0, n_actors=5, n_idle=0, min_size=2, max_size=8) == "hold"


def test_merge_concurrency():
    from batcher.dist.executors.map import _merge_concurrency

    assert _merge_concurrency(None, (2, 5)) == (2, 5)
    assert _merge_concurrency(3, None) == 3
    assert _merge_concurrency((1, 4), 8) == (8, 8)  # take the larger bounds


def test_gpu_options_pins_only_with_gpu():
    from batcher.dist.executors.map import _gpu_options

    assert _gpu_options(0.0, "NVIDIA_A100") == {}  # no GPU -> no accelerator_type
    assert _gpu_options(1.0, None) == {"num_gpus": 1.0}
    assert _gpu_options(1.0, "NVIDIA_A100") == {"num_gpus": 1.0, "accelerator_type": "NVIDIA_A100"}


def test_task_options_accelerator_type():
    from batcher.dist.executors.ray_runtime import task_options
    from batcher.plan.resource import SchedulingEnvelope

    gpu = SchedulingEnvelope(num_gpus=1.0, accelerator_type="NVIDIA_A100")
    assert task_options(gpu)["accelerator_type"] == "NVIDIA_A100"
    cpu = SchedulingEnvelope(num_gpus=0.0, accelerator_type="NVIDIA_A100")
    assert "accelerator_type" not in task_options(cpu)


def test_map_batches_stores_autoscale_and_accelerator():
    ds = bt.from_pydict({"x": [1, 2, 3]})
    plan = ds.ml.map_batches(lambda b: b, concurrency=(2, 8), accelerator_type="NVIDIA_A100")._plan
    assert plan.concurrency == (2, 8)
    assert plan.accelerator_type == "NVIDIA_A100"


@pytest.mark.parametrize("bad", [0, -1, (3, 1), (1, 2, 3), (0, 4)])
def test_invalid_concurrency_rejected(bad):
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match="concurrency"):
        ds.ml.map_batches(lambda b: b, concurrency=bad)
