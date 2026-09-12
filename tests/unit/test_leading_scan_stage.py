"""A `scan -> GPU map` chain runs its read inside the GPU actor, and on a mixed fleet it must not.

`split_into_resource_stages` folds a leading scan-only group into the stage above it, because
"streaming an unprocessed partition over Flight is not worth the hop". That is a claim about
**bytes** and it is right whenever the read has nowhere better to run.

On a heterogeneous fleet it is not the whole story, and nothing fails to make that visible.
Folded, the Parquet decode runs inside the accelerator actor -- on the GPU node's own cores, in
the GPU node's own memory, the two resources that stage needs to keep its device fed -- while
every accelerator-free node in the cluster does nothing. Sampled per node through an image
inference run on an 8-GPU / 9-CPU-node fleet: **CPU nodes 2.1% busy, GPU nodes 24.6% CPU and
36.9% device**.

So the fold became the caller's decision, and the caller asks the fleet. These tests pin both
halves: the plan-shape function is pure and still folds by default, and the scheduler's
predicate says "split" only when there are accelerator-free nodes to split onto.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.dist.executors.plan_analysis import split_into_resource_stages

pytestmark = pytest.mark.unit


class _Model:
    """A load-once class UDF: its own pool, and the reason there is a boundary at all."""

    def __call__(self, batch):
        return batch


def _scan_then_gpu():
    return bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(_Model, num_gpus=1.0)._plan


def test_a_scan_feeding_a_gpu_stage_is_one_stage_when_folded():
    """The default, and the behaviour on every homogeneous cluster: no hand-off."""
    assert split_into_resource_stages(_scan_then_gpu()) is None


def test_the_same_plan_becomes_two_stages_when_the_scan_is_not_folded():
    """The regression this exists for: the read gets a stage of its own, so it can be placed."""
    stages = split_into_resource_stages(_scan_then_gpu(), fold_leading_scan=False)
    assert stages is not None, "unfolded, the scan is a host stage and the model is a GPU stage"
    assert len(stages) == 2
    host, device = stages
    assert host.num_gpus == 0.0 and not host.wants_pool, "the read asks for no accelerator"
    assert device.num_gpus == 1.0 and device.wants_pool


def test_a_cpu_prefix_is_unaffected_by_the_flag():
    """With a real CPU map before the model there was never a scan-only group to fold."""
    plan = (
        bt.from_pydict({"x": [1, 2, 3]})
        .ml.map_batches(lambda b: b)
        .ml.map_batches(_Model, num_gpus=1.0)
        ._plan
    )
    folded = split_into_resource_stages(plan)
    unfolded = split_into_resource_stages(plan, fold_leading_scan=False)
    assert folded is not None and unfolded is not None
    assert len(folded) == len(unfolded) == 2


def test_a_chain_with_no_pool_stage_still_declines_either_way():
    plan = bt.from_pydict({"x": [1]}).ml.map_batches(lambda b: b)._plan
    assert split_into_resource_stages(plan) is None
    assert split_into_resource_stages(plan, fold_leading_scan=False) is None


# --------------------------------------------------------------------------- #
# The scheduler's half: which fleets get the split
# --------------------------------------------------------------------------- #
def test_a_cpu_only_pipeline_always_folds(monkeypatch):
    """No accelerator stage means no node class to keep the read off, so nothing changes."""
    from batcher.dist.streaming.pipeline import driver

    called = []
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.scaling.cpu_only_can_host",
        lambda *a, **k: called.append(a) or True,
    )
    plan = bt.from_pydict({"x": [1]}).ml.map_batches(_Model)._plan
    assert driver.fold_leading_scan(plan, workers=8) is True
    assert called == [], "the fleet is not even consulted for a chain with no accelerator"


def test_a_gpu_pipeline_splits_only_when_cpu_nodes_can_host_the_read(monkeypatch):
    """`cpu_only_can_host` is already False on a homogeneous or accelerator-less cluster."""
    from batcher.dist.streaming.pipeline import driver

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.scaling.cpu_only_can_host", lambda *a, **k: True
    )
    assert driver.fold_leading_scan(_scan_then_gpu(), workers=8) is False

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.scaling.cpu_only_can_host", lambda *a, **k: False
    )
    assert driver.fold_leading_scan(_scan_then_gpu(), workers=8) is True


def test_an_unreadable_fleet_folds_rather_than_failing(monkeypatch):
    """Placement is a courtesy; a fleet that cannot be read keeps today's behaviour."""
    from batcher.dist.streaming.pipeline import driver

    def _boom(*_a, **_k):
        raise RuntimeError("ray is down")

    monkeypatch.setattr("batcher.dist.executors.ray_runtime.scaling.cpu_only_can_host", _boom)
    assert driver.fold_leading_scan(_scan_then_gpu(), workers=8) is True


def test_a_scan_whose_neighbour_is_a_host_stage_folds_even_on_a_mixed_fleet(monkeypatch):
    """The split exists to keep the read off the accelerator nodes. A CPU stage already does.

    `scan -> map(concurrency=N) -> map(num_gpus=1)` is the ordinary two-stage inference shape,
    and the CPU stage's explicit `concurrency` makes it a pool of its own -- so the scan is a
    "leading scan with no CPU map before the first pool stage" and was split off on any mixed
    fleet. That produced **three** stages: eight actors reading the shards and republishing
    every row over Flight to the sixty-four that could have read them directly, at the cost of
    a whole extra hop and a read pool sized like the device fleet rather than the host one.

    Folding here puts the read in the CPU stage's own actors, which are on the CPU nodes. The
    node-placement argument is satisfied and the hop is not paid.
    """
    from batcher.dist.streaming.pipeline import driver

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.scaling.cpu_only_can_host", lambda *a, **k: True
    )
    plan = (
        bt.from_pydict({"x": [1, 2, 3]})
        .ml.map_batches(_Model, concurrency=4)
        .ml.map_batches(_Model, num_gpus=1.0)
        ._plan
    )
    assert driver.fold_leading_scan(plan, workers=8) is True
    stages = split_into_resource_stages(plan, fold_leading_scan=True)
    assert stages is not None and len(stages) == 2, "the read belongs with the host stage"
    host, device = stages
    assert not host.num_gpus and device.num_gpus == 1.0


def test_a_scan_whose_neighbour_is_the_device_stage_still_splits(monkeypatch):
    """The control: with no host stage between, folding really does put the read on the GPU."""
    from batcher.dist.streaming.pipeline import driver

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.scaling.cpu_only_can_host", lambda *a, **k: True
    )
    assert driver.fold_leading_scan(_scan_then_gpu(), workers=8) is False
