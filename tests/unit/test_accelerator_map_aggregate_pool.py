"""An accelerator `map_batches -> aggregate` stage runs on its actor pool, not on tasks.

`map_batches(Model, num_gpus=1) -> agg(...)` is the shape every batch-inference workload has,
because reducing on the workers is what stops the engine being charged a transfer its rivals
do not pay. It had no pool: `_agg_actor_pool` declines an accelerator stage, and the route
then fell through to `_map_agg_task`, which reserves **no device** and rebuilds the class UDF
once per partition -- a model load, a CUDA context and a weight upload apiece.

Nothing failed. The answer was right, the query ran, and on eight T4s over a 10,000-image
corpus it took 10.50 s where the identical dense network on the *host* took 1.15 s and the
read alone took 0.90 s. That is the whole failure mode: a device that is reacquired per
partition reads as a slow device.

These run with no cluster and no data. `_run_pool_stage` is the single point every pooled
stage goes through and `partition_descriptors` the single point partitions come from, so
stubbing the two accounts for the whole route without a Ray connection.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.executors import map as M

pytestmark = pytest.mark.unit


class Model:
    """A load-once class UDF -- the thing an actor pool exists to build exactly once."""

    def __call__(self, batch):
        return {"pred": batch["v"]}


def _plan(**map_kwargs):
    """`scan -> map_batches(Model, **map_kwargs) -> agg(sum)`, as the dispatcher sees it."""
    ds = bt.from_arrow(pa.table({"v": pa.array([1.0, 2.0, 3.0])}))
    ds = ds.map_batches(Model, output_columns=["pred"], batch_format="numpy", **map_kwargs)
    return ds.agg(s=bt.col("pred").sum())._plan


@pytest.fixture
def routed(monkeypatch):
    """Record which pooled stage ran and what it dispatched, building no actors."""
    seen: dict = {}

    def fake_pool_stage(plan, plan0, partitions, opts, workers, hub, **kw):
        seen["opts"] = opts
        seen["partitions"] = len(partitions)
        seen["kw"] = kw

        # Drive the caller's launcher against a recording stand-in for one actor, so the
        # method the route dispatches is observed rather than assumed.
        class _Actor:
            class _Method:
                def __init__(self, name, calls):
                    self._name, self._calls = name, calls

                def remote(self, *args):
                    self._calls.append((self._name, args))
                    return None

            def __init__(self, calls):
                self.run = self._Method("run", calls)
                self.run_agg = self._Method("run_agg", calls)

        calls: list = []
        seen["calls"] = calls
        kw["launch"](_Actor(calls), partitions[0], 0)
        return [None] * len(partitions)

    # No cluster: `_ensure_ray` re-wraps the module's remote functions, which a stubbed
    # `_map_agg_task` is not one of.
    monkeypatch.setattr(M, "_ensure_ray", lambda workers: None)
    monkeypatch.setattr(M, "_run_pool_stage", fake_pool_stage)
    monkeypatch.setattr(
        M, "partition_descriptors", lambda source, workers, **kw: [{"rows": []}] * 4
    )

    def refuse(*a, **k):
        raise AssertionError("the CPU aggregate pool must not be consulted for a device stage")

    seen["refuse"] = refuse
    return seen


def test_a_gpu_map_under_an_aggregate_reaches_the_actor_pool(routed, monkeypatch):
    monkeypatch.setattr(M, "_agg_actor_pool", routed["refuse"])
    agg = _plan(num_gpus=1, concurrency=8)
    M._distributed_map_aggregate(None, agg, [_source()], workers=4)
    # The pool ran, it reserved the device, and it took no core (see `_gpu_options`).
    assert routed["opts"]["num_gpus"] == 1
    assert routed["opts"]["num_cpus"] == 0
    assert routed["kw"]["num_gpus"] == 1


def test_it_dispatches_the_aggregating_entry_point(routed, monkeypatch):
    # `run` would return the mapped rows and move every pixel to the driver; `run_agg`
    # partial-aggregates in the actor, which is what keeps the mergeable contract intact.
    monkeypatch.setattr(M, "_agg_actor_pool", routed["refuse"])
    M._distributed_map_aggregate(None, _plan(num_gpus=1, concurrency=8), [_source()], workers=4)
    assert [name for name, _ in routed["calls"]] == ["run_agg"]


def test_the_explicit_concurrency_gets_partitions_to_fill_it(routed, monkeypatch):
    # `_drive_actor_pool` clamps the pool to `min(max_size, len(partitions))`, so a partition
    # count below the caller's `concurrency` silently shrinks their pool -- and on this route
    # the count used to come from the data, which let a small corpus decide how many devices
    # were allowed to run.
    monkeypatch.setattr(M, "_agg_actor_pool", routed["refuse"])
    counts: list = []
    monkeypatch.setattr(
        M,
        "partition_descriptors",
        lambda source, workers, **kw: counts.append(kw.get("max_partitions")) or [{"rows": []}] * 4,
    )
    M._distributed_map_aggregate(None, _plan(num_gpus=1, concurrency=8), [_source()], workers=4)
    assert counts == [8]


def test_a_cpu_class_udf_still_takes_the_cpu_aggregate_pool(monkeypatch):
    # The accelerator branch must not swallow the CPU route: a class UDF with no device is
    # exactly what `_agg_actor_pool`'s own warm registry is for.
    called: list = []
    monkeypatch.setattr(M, "_ensure_ray", lambda workers: None)
    monkeypatch.setattr(M, "_agg_actor_pool", lambda plan0, workers: called.append(1) or None)
    monkeypatch.setattr(
        M, "partition_descriptors", lambda source, workers, **kw: [{"rows": []}] * 2
    )
    monkeypatch.setattr(M, "_run_pool_stage", lambda *a, **k: pytest.fail("not a device stage"))
    monkeypatch.setattr(M, "_gather_with_pool_recovery", lambda *a, **k: None)
    M._distributed_map_aggregate(None, _plan(concurrency=2), [_source()], workers=2)
    assert called == [1]


def _source():
    """A stand-in source: the route only asks it for partitions, which are stubbed."""
    from batcher.io.source import InMemorySource

    return InMemorySource([pa.record_batch({"v": pa.array([1.0])})])


def test_a_plain_function_asking_for_a_device_still_reserves_one(monkeypatch):
    # No pool: a plain function has no model to load once, so tasks are right for it. The
    # device is not optional though -- without it the task runs wherever Ray has a core, and
    # a UDF that reaches for CUDA fails on every node that has none.
    options: list = []

    class _Task:
        def options(self, **kw):
            options.append(kw)
            return self

        def remote(self, *a, **k):
            return None

    monkeypatch.setattr(M, "_ensure_ray", lambda workers: None)
    monkeypatch.setattr(M, "_map_agg_task", _Task())
    monkeypatch.setattr(M, "_agg_actor_pool", lambda plan0, workers: None)
    monkeypatch.setattr(
        M, "partition_descriptors", lambda source, workers, **kw: [{"rows": []}] * 2
    )
    monkeypatch.setattr(M, "_adaptive_task_cpus", lambda partitions, plan: [1.0, 1.0])
    monkeypatch.setattr(M, "_map_scheduling_options", lambda env, shares: {})
    monkeypatch.setattr(M, "_shared_arg", lambda v: v)
    monkeypatch.setattr(M, "_gather_with_pool_recovery", lambda g, a, t, p, actors: t(0))

    ds = bt.from_arrow(pa.table({"v": pa.array([1.0, 2.0])}))
    ds = ds.map_batches(
        lambda b: {"pred": b["v"]},
        output_columns=["pred"],
        batch_format="numpy",
        num_gpus=1,
    )
    M._distributed_map_aggregate(None, ds.agg(s=bt.col("pred").sum())._plan, [_source()], 2)
    assert options and options[0]["num_gpus"] == 1
    # And it takes no core, for the reason `_gpu_options` gives: an actor or task naming any
    # resource otherwise queues behind a CPU the shuffle fleet may already hold.
    assert options[0]["num_cpus"] == 0
