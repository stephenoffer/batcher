"""Warming a fleet's devices has to leave something warm.

A GPU worker's first task pays three fixed costs before it touches a row. Measured on a T4
against one shard of TPC-H `lineitem`: **3.04 s to import cuDF** and **0.30 s to build the RMM
pool**, against 0.09 s to read the shard onto the device and 0.02 s to run the kernels. Paying
them off the critical path is the point of `warm_devices`.

It only works if the warmed **process survives**. Ray's default for a GPU task is to tear the
worker down after every call — `gpu_task_options` sets `max_calls=0` precisely to stop that —
so a warm-up that built its own options destroyed the process that had just imported cuDF, and
its only lasting effect was to make the first shard wait for the device to be released before
paying the import itself. A warm-up that is not reused is strictly worse than none.
"""

from __future__ import annotations

import pytest

from batcher.dist.gpu import tasks

pytestmark = pytest.mark.unit


@pytest.fixture
def submitted(monkeypatch):
    """Capture the options every warm-up task is submitted with, without a cluster."""
    calls = []

    class _Node:
        def __init__(self, node_id):
            self.node_id = node_id

    class _Handle:
        def remote(self):
            return object()

    def _remote(**options):
        calls.append(options)
        return lambda fn: _Handle()

    calls = []

    import ray

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "remote", _remote)
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.gpu_node_topology",
        # A tuple, as the real `gpu_node_topology` returns — a generator would be
        # consumed by the first pass and is not what the code under test is handed.
        lambda: tuple(_Node(f"{i:056x}") for i in range(3)),
    )
    tasks.reset_device_warmup()

    class _Warmups(list):
        """Only the *pinned* submissions are warm-ups.

        Building the shard options runs the cuDF probe, which submits its own tiny task through
        the same `ray.remote` — with no `scheduling_strategy`, because it does not care which
        node answers. Counting it as a warm-up made this file's first assertion fail on a task
        it was not about.
        """

        @property
        def warmups(self):
            return [o for o in self if "scheduling_strategy" in o]

    out = _Warmups()
    calls = out
    yield out
    tasks.reset_device_warmup()


def test_the_warm_up_asks_for_a_reusable_worker(submitted):
    """`max_calls=0`. Without it Ray destroys the process that just imported cuDF."""
    assert tasks.warm_devices() == 3
    assert all(o.get("max_calls") == 0 for o in submitted.warmups), submitted.warmups


def test_the_warm_up_takes_a_sliver_of_a_device(submitted):
    """It must neither block a real shard nor hold a board."""
    tasks.warm_devices()
    assert all(0 < o["num_gpus"] < 0.1 for o in submitted.warmups)


def test_the_warm_up_carries_the_shards_runtime_env(submitted):
    """Ray pools workers by runtime_env, so a different one warms a process no shard will get."""
    shard_env = tasks.gpu_task_options().get("runtime_env")
    tasks.warm_devices()
    assert all(o.get("runtime_env") == shard_env for o in submitted.warmups)


def test_one_task_per_node(submitted):
    """A fractional request lets Ray put every warm-up on one node, warming it three times."""
    tasks.warm_devices()
    assert len({o["scheduling_strategy"].node_id for o in submitted.warmups}) == 3


def test_a_node_that_went_away_must_not_strand_the_task(submitted):
    """`soft=True`: a warm-up that lands elsewhere is still a warm worker; one that cannot be
    placed pends against a dead id for the life of the driver."""
    tasks.warm_devices()
    assert all(o["scheduling_strategy"].soft for o in submitted.warmups)


def test_the_fleet_is_warmed_once_per_process(submitted):
    """What it pays for survives for the life of the worker, so a second request would submit
    tasks that find everything already done."""
    assert tasks.warm_devices() == 3
    assert tasks.warm_devices() == 0
    assert len(submitted.warmups) == 3
