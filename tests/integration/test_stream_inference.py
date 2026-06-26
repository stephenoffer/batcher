"""The streaming heterogeneous inference pipeline equals the single-node result.

A CPU preprocess `map_batches` feeding a load-once inference `map_batches` is split
into two overlapped stages that hand off over Arrow Flight. Per the mergeable /
single-node-fallback invariant, the overlapped distributed result MUST equal both the
single-node run and the non-overlapped distributed map.
"""

from __future__ import annotations

import sys

import pytest

import batcher as bt
from batcher.config import Config, DistributedConfig, config_context

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

import ray  # noqa: E402

ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _double(batch):
    d = batch.to_pydict()
    d["x2"] = [v * 2 for v in d["x"]]
    return d


class _AddOne:
    """A load-once class UDF — the inference stage's own resource class."""

    def __call__(self, batch):
        d = batch.to_pydict()
        d["y"] = [v + 1 for v in d["x2"]]
        return d


def _postprocess(batch):
    d = batch.to_pydict()
    d["z"] = [v + 100 for v in d["y"]]
    return d


def _streaming():
    return config_context(Config().replace(distributed=DistributedConfig(stream_inference=True)))


def test_streaming_pipeline_equals_single_node():
    out = bt.from_pydict({"id": list(range(200)), "x": list(range(200))})
    out = out.ml.map_batches(_double).ml.map_batches(_AddOne)
    single = out.collect().sort_by("id").to_pydict()
    with _streaming():
        dist = out.collect(distributed=True, num_workers=2).sort_by("id").to_pydict()
    assert single == dist
    assert dist["y"] == [i * 2 + 1 for i in range(200)]


def test_streaming_three_stage_equals_single_node():
    # CPU preprocess → load-once inference → CPU postprocess: the producer is the CPU
    # prefix, the consumer runs inference AND postprocess. Result must still match.
    out = bt.from_pydict({"id": list(range(180)), "x": list(range(180))})
    out = out.ml.map_batches(_double).ml.map_batches(_AddOne).ml.map_batches(_postprocess)
    single = out.collect().sort_by("id").to_pydict()
    with _streaming():
        dist = out.collect(distributed=True, num_workers=3).sort_by("id").to_pydict()
    assert single == dist
    assert dist["z"] == [i * 2 + 1 + 100 for i in range(180)]


def test_streaming_pipeline_equals_non_overlapped_map():
    out = bt.from_pydict({"id": list(range(150)), "x": list(range(150))})
    out = out.ml.map_batches(_double).ml.map_batches(_AddOne)
    plain = out.collect(distributed=True, num_workers=3).sort_by("id").to_pydict()
    with _streaming():
        streamed = out.collect(distributed=True, num_workers=3).sort_by("id").to_pydict()
    assert plain == streamed


def test_streaming_pipeline_empty_input():
    out = bt.from_pydict({"id": [], "x": []})
    out = out.ml.map_batches(_double).ml.map_batches(_AddOne)
    with _streaming():
        dist = out.collect(distributed=True, num_workers=2).to_pydict()
    assert dist.get("y", []) == []


def test_production_window_bounds_producer_memory():
    # Drive many morsels per partition through one producer with a small credit window
    # and assert the producer never retained more than `credits` published morsels —
    # the production-window memory bound, independent of partition size.
    from batcher.config import Config, DistributedConfig, FlowControlConfig
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.executors.plan_analysis import split_at_first_pool_boundary
    from batcher.dist.executors.ray_runtime import _ensure_ray
    from batcher.dist.flight_worker import new_plan_id
    from batcher.dist.streaming.pipeline import _ProducerActor, _run_streamed
    from batcher.io.source import InMemorySource

    credits = 2
    cfg = Config().replace(
        distributed=DistributedConfig(stream_inference=True),
        flow_control=FlowControlConfig(default_credits=credits),
    )
    with config_context(cfg):
        ds = bt.from_pydict({"id": list(range(120)), "x": list(range(120))})
        plan = ds.ml.map_batches(_double).ml.map_batches(_AddOne)._plan
        cpu_stage, gpu_stage = split_at_first_pool_boundary(plan)
        _ensure_ray(1)
        from batcher.dist.executors.map import _MapActor

        # Force many small morsels (one row each) so a 120-row partition is 120 morsels.
        src = InMemorySource([_one_row(i) for i in range(120)])
        partitions = partition_descriptors(src, 1)
        producer = _ProducerActor.remote(cpu_stage.sub_plan, credits)
        consumer = _MapActor.remote(gpu_stage.sub_plan)
        try:
            results = _run_streamed([producer], [consumer], partitions, new_plan_id(), credits)
            peak = ray.get(producer.peak_retained.remote())
        finally:
            ray.kill(producer)
            ray.kill(consumer)
        assert 0 < peak <= credits  # never held more than the credit window
        # Every morsel landed (120 rows, computed correctly), independent of the window.
        ys = sorted(
            v for _k, out in results.items() if out for b in out for v in b.to_pydict()["y"]
        )
        assert ys == [i * 2 + 1 for i in range(120)]


def _one_row(i: int):
    import pyarrow as pa

    return pa.RecordBatch.from_pydict({"id": [i], "x": [i]})
