"""Distributed batch inference: GPU-aware scheduling + model-once actor pools.

The Ray Data parity surface — `Dataset.infer`/`embed` with `num_gpus`/`concurrency`
fan a model across a heterogeneous CPU+GPU actor pool, loading the model once per
worker. These tests run on CPU (num_gpus=0); the GPU path is the same code with a
resource reservation, asserted at the plan level so it needs no GPU.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.executors.map import _map_resources
from batcher.plan.logical import MapBatches

pytest.importorskip("batcher._native", reason="native engine not built")
ray = pytest.importorskip("ray", reason="ray not installed")

pytestmark = pytest.mark.integration

# pytest imports test files as top-level modules the Ray workers can't re-import,
# so serialize this module's UDF classes by value instead of by reference.
import sys  # noqa: E402

ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _TaggingModel:
    """A stateful model: built once per worker, tags every row with its instance id
    so we can count how many times the model was actually constructed."""

    def __init__(self) -> None:
        import uuid

        self.tag = uuid.uuid4().hex

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        x = batch.column("x").to_pylist()
        return pa.RecordBatch.from_pydict(
            {"x": x, "y": [v * 2 for v in x], "tag": [self.tag] * len(x)}
        )


def _double(batch: pa.RecordBatch) -> pa.RecordBatch:
    x = batch.column("x").to_pylist()
    return pa.RecordBatch.from_pydict({"x": x, "y": [v * 2 for v in x]})


def test_map_resources_extracts_gpu_and_pool():
    # Plan-level assertion of the GPU/actor-pool plumbing — no GPU or Ray needed.
    plan = MapBatches(
        input=bt.from_pydict({"x": [1, 2, 3]})._plan,
        fn=_TaggingModel,  # a class → needs build-once → actor pool
        num_gpus=2.0,
        concurrency=4,
    )
    num_gpus, wants_pool, concurrency, accelerator_type = _map_resources(plan)
    assert num_gpus == 2.0
    assert wants_pool is True
    assert concurrency == 4
    assert accelerator_type is None


def test_class_udf_implies_pool_without_explicit_concurrency():
    plan = MapBatches(input=bt.from_pydict({"x": [1]})._plan, fn=_TaggingModel)
    _gpus, wants_pool, concurrency, _accel = _map_resources(plan)
    assert wants_pool is True  # a factory UDF forces the build-once actor pool
    assert concurrency is None  # sized to the worker count at run time


def test_distributed_infer_loads_model_once_per_actor():
    n = 2000
    ds = bt.from_pydict({"x": list(range(n))})
    out = ds.ml.infer(
        _TaggingModel,
        output_columns=["x", "y", "tag"],
        concurrency=2,
    ).collect(distributed=True, num_workers=4)

    rows = out.to_pylist()
    # Correctness: y == 2x for every row, no rows lost.
    assert len(rows) == n
    assert all(r["y"] == r["x"] * 2 for r in rows)
    # Model-once: at most `concurrency` distinct instances built, regardless of how
    # many partitions/batches flowed through them.
    assert len({r["tag"] for r in rows}) <= 2


def test_autoscaling_pool_correct_results():
    # concurrency=(min, max) runs the dynamic-autoscaling pool: it must still produce
    # every row exactly once, with the model built at most `max` times.
    n = 1500
    ds = bt.from_pydict({"x": list(range(n))})
    out = ds.ml.infer(
        _TaggingModel,
        output_columns=["x", "y", "tag"],
        concurrency=(1, 3),
    ).collect(distributed=True, num_workers=4)

    rows = out.to_pylist()
    assert sorted(r["x"] for r in rows) == list(range(n))  # complete, no dup/loss
    assert all(r["y"] == r["x"] * 2 for r in rows)
    assert len({r["tag"] for r in rows}) <= 3  # never exceeded max actors


def test_dataset_infer_and_embed_single_node():
    ds = bt.from_pydict({"x": [1, 2, 3, 4]})

    inferred = ds.ml.infer(_double, output_columns=["x", "y"]).collect()
    assert inferred.column("y").to_pylist() == [2, 4, 6, 8]

    # embed is the same fluent path specialized for embedding models.
    embedded = ds.ml.embed(_double, output_columns=["x", "y"]).collect()
    assert embedded.column("y").to_pylist() == [2, 4, 6, 8]
