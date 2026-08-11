"""`split_into_resource_stages` cuts a linear map pipeline at every resource boundary.

Pure plan inspection (no engine, no Ray). Consecutive stateless-CPU maps group into one
stage — a Flight hop between two host transforms costs more than it saves — and every
pool-class map (a GPU stage, an explicit `concurrency`, or a class UDF that loads a model
once) becomes a stage of its own. A chain with no pool-class stage returns `None`, so the
caller keeps the non-overlapped distributed-map path.

This used to pin a *single* cut: the CPU prefix, and then the first model stage plus
everything above it. That grouping is what put a CPU postprocess on the GPU actor and two
chained models in one actor, and the cases below are written against the shapes where that
showed.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.dist.executors.plan_analysis import split_into_resource_stages
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
    """A load-once class (factory) UDF — a model stage (`_is_pool_class`)."""

    def __call__(self, batch):
        d = batch.to_pydict()
        d["y"] = [v + 1 for v in d["x2"]]
        return d


class _Rerank:
    """A *second* load-once model, so a chain can hold two pool stages."""

    def __call__(self, batch):
        d = batch.to_pydict()
        d["r"] = [v * 10 for v in d["y"]]
        return d


def _ds():
    return bt.from_pydict({"id": list(range(8)), "x": list(range(8))})


def test_cpu_then_inference_splits():
    stages = split_into_resource_stages(_ds().ml.map_batches(_double).ml.map_batches(_AddOne)._plan)
    assert [s.wants_pool for s in stages] == [False, True]
    assert all(_leaf(s.sub_plan).source_id == 0 for s in stages)


def test_a_postprocess_gets_its_own_stage_instead_of_riding_the_model():
    """CPU → model → CPU is three pools, not two.

    The postprocess used to run inside the model's actor, which spent device time on host
    work and tied the two stages' pool sizes to one number.
    """
    out = _ds().ml.map_batches(_double).ml.map_batches(_AddOne).ml.map_batches(_post)
    stages = split_into_resource_stages(out._plan)
    assert [s.wants_pool for s in stages] == [False, True, False]
    assert [_count_maps(s.sub_plan) for s in stages] == [1, 1, 1]


def test_two_chained_models_get_a_pool_each():
    """Otherwise both models load into one actor, share a device, and take turns."""
    out = _ds().ml.map_batches(_double).ml.map_batches(_AddOne).ml.map_batches(_Rerank)
    stages = split_into_resource_stages(out._plan)
    assert [s.wants_pool for s in stages] == [False, True, True]


def test_consecutive_cpu_maps_stay_in_one_stage():
    """A hand-off between two host transforms costs more than the overlap is worth."""
    out = _ds().ml.map_batches(_double).ml.map_batches(_double).ml.map_batches(_AddOne)
    stages = split_into_resource_stages(out._plan)
    assert [_count_maps(s.sub_plan) for s in stages] == [2, 1]


def test_no_model_stage_returns_none():
    assert split_into_resource_stages(_ds().ml.map_batches(_double)._plan) is None


def test_a_lone_model_stage_returns_none():
    """One pool and nothing to overlap it with is the non-overlapped map, spelled longer."""
    assert split_into_resource_stages(_ds().ml.map_batches(_AddOne)._plan) is None


def test_a_bare_scan_prefix_folds_into_the_model_rather_than_becoming_a_hop():
    """Streaming an unprocessed partition over Flight is not worth its own stage.

    The model reads the partition itself, and the postprocess above it is the hand-off that
    does pay. The old single cut declined this shape outright and ran it non-overlapped.
    """
    stages = split_into_resource_stages(_ds().ml.map_batches(_AddOne).ml.map_batches(_post)._plan)
    assert [s.wants_pool for s in stages] == [True, False]


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


# --- the corpus's flagship shape ------------------------------------------------------
# The field guides' RAG indexing pipeline is four stages across three resource classes —
# `extract` (CPU-heavy) -> `chunk` (CPU-light) -> `embed` (GPU) -> `write` — and they report
# ~86% cost reduction and ~69% faster end-to-end from getting that placement right versus an
# all-GPU cluster. The gap recorded against Batcher was that it split at *one* boundary, so
# the write stage rode the GPU pool. It does not any more, and this is what pins that.


def _chunk(batch):
    d = batch.to_pydict()
    d["x2"] = [v * 3 for v in d["x"]]
    return d


def test_the_rag_four_stage_shape_gives_the_write_stage_its_own_pool():
    plan = (
        _ds()
        .map_batches(_double, output_columns=["id", "x", "x2"])  # extract  (CPU)
        .map_batches(_chunk, output_columns=["id", "x", "x2"])  # chunk    (CPU)
        .ml.map_batches(_AddOne, output_columns=["id", "x", "x2", "y"])  # embed (model)
        .map_batches(_post, output_columns=["id", "x", "x2", "y", "z"])  # write (CPU)
        ._plan
    )
    stages = split_into_resource_stages(plan)
    assert stages is not None, "the CPU prefix must be separated from the model stage"
    # Counting rather than naming, because the *grouping* is the contract: the two CPU
    # stages group, the model is alone, and the write stage no longer rides the GPU pool.
    assert [_count_maps(s.sub_plan) for s in stages] == [2, 1, 1]
    assert [s.wants_pool for s in stages] == [False, True, False]


def test_the_producer_carries_no_gpu_request():
    """The whole point of the split is that the CPU prefix does not hold a GPU while it
    decodes — the guides' named #1 cause of low GPU utilization."""
    plan = (
        _ds()
        .map_batches(_double, output_columns=["id", "x", "x2"])
        .ml.map_batches(_AddOne, num_gpus=1, output_columns=["id", "x", "x2", "y"])
        ._plan
    )
    producer, consumer = split_into_resource_stages(plan)
    assert producer.num_gpus == 0.0, "a decode stage must not reserve a GPU"
    assert consumer.num_gpus == 1.0


def test_a_postprocess_stage_carries_no_gpu_request_either():
    """The same property one hop further up, which the single cut could not give it."""
    plan = (
        _ds()
        .map_batches(_double, output_columns=["id", "x", "x2"])
        .ml.map_batches(_AddOne, num_gpus=1, output_columns=["id", "x", "x2", "y"])
        .map_batches(_post, output_columns=["id", "x", "x2", "y", "z"])
        ._plan
    )
    assert [s.num_gpus for s in split_into_resource_stages(plan)] == [0.0, 1.0, 0.0]


# --- pool bounds -------------------------------------------------------------------------


def _stage(concurrency, *, num_gpus: float = 0.0):
    """A `StageSpec` carrying just the fields the pool sizing reads."""
    from batcher.dist.executors.plan_analysis import StageSpec

    return StageSpec(
        sub_plan=_ds()._plan,
        num_gpus=num_gpus,
        accelerator_type=None,
        wants_pool=True,
        concurrency=concurrency,
    )


def test_a_range_concurrency_opens_at_its_minimum_and_may_grow_to_its_maximum():
    """`concurrency=(min, max)` is documented as autoscaling and must behave as one.

    The streaming pipeline used to resolve the range *statically* — the partition count
    clamped into it — so the same public argument meant "autoscale" on the actor-pool path
    and "a fixed pool sized from a number you never mentioned" here.
    """
    from batcher.dist.streaming.consumers import consumer_pool_bounds

    assert consumer_pool_bounds(_stage((2, 8)), workers=4, num_partitions=16) == (2, 8)


def test_a_range_minimum_above_its_maximum_is_clamped_not_inverted():
    """A caller writing `(8, 2)` must still get a runnable pool rather than an empty one."""
    from batcher.dist.streaming.consumers import consumer_pool_bounds

    start, ceiling = consumer_pool_bounds(_stage((8, 2)), workers=4, num_partitions=16)
    assert ceiling >= start >= 1


def test_a_fixed_concurrency_reports_no_headroom():
    """`start == ceiling` is how a stage says "do not scale", and an int must say it.

    If a plain int read as headroom every existing pipeline would start spawning actors.
    """
    from batcher.dist.streaming.consumers import consumer_pool_bounds

    assert consumer_pool_bounds(_stage(3), workers=4, num_partitions=16) == (3, 3)


def test_an_absent_concurrency_keeps_the_default_size_and_no_headroom():
    """The default path is unchanged: today's size, and nothing to grow into."""
    from batcher.dist.streaming.consumers import consumer_pool_bounds, consumer_pool_size

    stage = _stage(None)
    size = consumer_pool_size(stage, 4, 16)
    assert consumer_pool_bounds(stage, workers=4, num_partitions=16) == (size, size)
