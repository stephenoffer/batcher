"""`split_at_first_pool_boundary` splits a linear map pipeline at the first model stage.

Pure plan inspection (no engine, no Ray): a CPU preprocess `map_batches` feeding a
load-once/GPU inference `map_batches` splits into a stateless-CPU producer + a
model consumer; a chain with no model stage (or no CPU prefix) returns None (so the
caller keeps the non-overlapped distributed-map path).
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.dist.executors.plan_analysis import split_at_first_pool_boundary
from batcher.plan.logical import Scan

pytestmark = pytest.mark.unit


def _double(batch):
    d = batch.to_pydict()
    d["x2"] = [v * 2 for v in d["x"]]
    return d


def _post(batch):
    d = batch.to_pydict()
    d["z"] = [v + 100 for v in d["y"]]
    return d


class _AddOne:
    """A load-once class (factory) UDF — the model stage (`_is_pool_class`)."""

    def __call__(self, batch):
        d = batch.to_pydict()
        d["y"] = [v + 1 for v in d["x2"]]
        return d


def _ds():
    return bt.from_pydict({"id": list(range(8)), "x": list(range(8))})


def test_cpu_then_inference_splits():
    out = _ds().ml.map_batches(_double).ml.map_batches(_AddOne)
    producer, consumer = split_at_first_pool_boundary(out._plan)
    assert producer.wants_pool is False
    assert consumer.wants_pool is True
    assert _leaf(producer.sub_plan).source_id == 0
    assert _leaf(consumer.sub_plan).source_id == 0


def test_cpu_infer_postprocess_splits_at_first_model_stage():
    # CPU → GPU/load-once → CPU postprocess: producer is the CPU prefix, the consumer
    # is the model stage AND the postprocess (no second hand-off).
    out = _ds().ml.map_batches(_double).ml.map_batches(_AddOne).ml.map_batches(_post)
    producer, consumer = split_at_first_pool_boundary(out._plan)
    assert producer.wants_pool is False
    assert consumer.wants_pool is True
    # The consumer carries two map stages (inference + postprocess).
    assert _count_maps(consumer.sub_plan) == 2
    assert _count_maps(producer.sub_plan) == 1


def test_no_model_stage_returns_none():
    out = _ds().ml.map_batches(_double)
    assert split_at_first_pool_boundary(out._plan) is None


def test_model_first_no_cpu_prefix_returns_none():
    # The model stage is first (no CPU map to overlap) → no split worth a hand-off.
    out = _ds().ml.map_batches(_AddOne)
    assert split_at_first_pool_boundary(out._plan) is None


def _leaf(plan):
    node = plan
    while not isinstance(node, Scan):
        node = node.input
    return node


def _count_maps(plan):
    from batcher.plan.logical import MapBatches

    n, node = 0, plan
    while not isinstance(node, Scan):
        n += isinstance(node, MapBatches)
        node = node.input
    return n
