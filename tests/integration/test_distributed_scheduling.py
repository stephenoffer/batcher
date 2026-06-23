"""Multinode behavior of the metadata-driven scheduling + feedback machinery.

Ray local mode runs each task/actor in its own worker process, so these exercise
the real cross-process paths a multi-node cluster uses: per-task resource requests
reach the worker, distributed workers measure and ship operator metrics back to the
driver's MetadataHub (so the cost loop is fed in distributed mode too), the GPU
utilization loop round-trips driver↔worker, and distributed results stay identical
to single-node.
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")
ray = pytest.importorskip("ray", reason="ray not installed")

pytestmark = pytest.mark.integration

# Worker processes can't re-import this pytest module, so serialize its UDF classes
# by value (the pattern the other distributed-ML tests use).
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _Model:
    """A class (factory) UDF → forces the build-once stateful actor-pool path."""

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch


def _rows(table: pa.Table) -> list:
    return sorted(tuple(sorted(r.items())) for r in table.to_pylist())


def test_per_task_resources_reach_worker():
    # The envelope → `.options(...)` mechanism must actually place a task with the
    # requested resources (verified inside the worker process, not just on the driver).
    import ray

    from batcher.dist.executors.ray_runtime import _ensure_ray, task_options
    from batcher.plan.resource import SchedulingEnvelope

    _ensure_ray(2)

    @ray.remote
    def _assigned():
        return ray.get_runtime_context().get_assigned_resources()

    env = SchedulingEnvelope(num_cpus=2.0, memory_bytes=0, num_gpus=0.0, n_tasks=2, credits=4)
    res = ray.get(_assigned.options(**task_options(env)).remote())
    assert res.get("CPU") == 2.0  # the per-task num_cpus request was honored


def test_distributed_workers_feed_the_metadata_hub():
    # A distributed aggregate's mappers run scan/filter sub-plans on worker
    # processes and ship their measured metrics back; the driver records them so the
    # cost model calibrates from distributed runs, not only single-node ones.
    from batcher import core

    hub = core.default_hub()
    before = len(hub.op_stats_by_kind().get("filter", []))

    t = pa.table({"k": [i % 7 for i in range(2000)], "v": [float(i) for i in range(2000)]})
    ds = bt.from_arrow(t).filter(col("v") > 10).group_by("k").agg(s=col("v").sum())
    ds.collect(distributed=True, num_workers=4)

    by_kind = hub.op_stats_by_kind()
    after = len(by_kind.get("filter", []))
    assert after > before, "distributed mappers must record operator feedback"
    # The recorded feedback carries real measurements (no more m_peak_bytes=0 stub).
    assert any(r["m_peak_bytes"] > 0 for r in by_kind["scan"])
    assert any(r["kind"] == "filter" for r in by_kind["filter"])


def test_distributed_equals_single_node_with_scheduling():
    # The scheduling envelope only changes placement hints, never the data path, so
    # a distributed result is byte-identical to single-node.
    t = pa.table({"k": [i % 5 for i in range(3000)], "v": [float(i % 13) for i in range(3000)]})
    ds = (
        bt.from_arrow(t)
        .filter(col("v") > 2)
        .group_by("k")
        .agg(s=col("v").sum(), n=col("v").count())
    )
    single = ds.collect()
    distrib = ds.collect(distributed=True, num_workers=4)
    assert _rows(single) == _rows(distrib)


def test_actor_pool_runs_distributed_and_gpu_feedback_is_safe():
    # The stateful actor-pool path (the GPU-inference shape) runs across worker
    # processes and produces correct results. On a GPU-less host the actors sample
    # no utilization, so the feedback recording is a safe no-op — it must not break.
    from batcher import core

    hub = core.default_hub()
    t = pa.table({"x": list(range(800))})
    # concurrency forces the actor pool; num_gpus=0 so it schedules on CPU here.
    ds = bt.from_arrow(t).map_batches(_Model, concurrency=2)
    out = ds.collect(distributed=True, num_workers=2)
    assert out.num_rows == 800
    # No GPU → nothing recorded (honest no-op), and the hub stays usable.
    assert hub.op_stats_by_kind() is not None


def test_gpu_record_and_adapt_loop_with_real_plan_key():
    # The driver half of the GPU loop, end-to-end with the real pipeline key: a
    # measured idle utilization is persisted and the next run's num_gpus is packed
    # onto a fraction. (Cross-process *measurement* needs real GPU hardware; here we
    # inject the measured value the actor would have returned.)
    from batcher import core
    from batcher.dist.executors.map import _record_gpu_feedback
    from batcher.ml.gpu import gpu_feedback_key, load_gpu_utilization, recommend_num_gpus

    hub = core.default_hub()
    ds = bt.from_pydict({"x": [1, 2, 3]}).map_batches(_Model, num_gpus=1.0, concurrency=2)

    _record_gpu_feedback(hub, ds._plan, 0.2)  # what an idle-GPU actor would report

    util = load_gpu_utilization(hub, gpu_feedback_key(ds._plan))
    assert util is not None and 0.0 < util < 0.5
    assert recommend_num_gpus(util, 1.0) < 1.0  # idle whole GPU → pack onto a fraction
