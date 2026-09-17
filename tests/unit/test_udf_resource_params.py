"""Every Ray Data resource parameter on the UDF verbs reaches the scheduler or raises.

`map_batches`, `map`, `flat_map` and a callable `filter` take Ray Data's whole resource
parameter set. A parameter accepted and then dropped reads as a scheduled job and runs as an
unscheduled one, so each is pinned here to one of two outcomes: a `MapBatches` field the map
scheduler (`dist.executors.map._map_resources`) reads, or a `PlanError` that names it. The
removed second spellings (`Dataset.query`, `Dataset.to_torch`, the `ds.ml` UDF verbs) are
pinned to an `AttributeError` that names the kept spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher._internal.errors import PerformanceWarning, PlanError
from batcher.plan.logical import MapBatches

pytestmark = pytest.mark.unit


def _ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3, 4]})


class _Model:
    def __init__(self, offset=0):
        self.offset = offset

    def __call__(self, batch):
        return batch


def _identity(batch):
    return batch


def _keep(batch):
    return pc.greater(batch["x"], 0)


#: One call per verb, taking the resource options as keywords; each returns the stage.
VERBS = {
    "map_batches": lambda ds, fn, **kw: ds.map_batches(fn, **kw),
    "map": lambda ds, fn, **kw: ds.map(lambda row: row, **kw),
    "flat_map": lambda ds, fn, **kw: ds.flat_map(lambda row: [row], **kw),
    "filter": lambda ds, fn, **kw: ds.filter(_keep, **kw),
}


def _stage(verb: str, **kw) -> MapBatches:
    plan = VERBS[verb](_ds(), _identity, **kw)._plan
    assert isinstance(plan, MapBatches), type(plan)
    return plan


# --- honoured: the value lands on the field the scheduler reads ------------------------


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_num_gpus_reaches_the_stage(verb):
    with pytest.warns(PerformanceWarning, match="re-created on every batch"):
        assert _stage(verb, num_gpus=0.5).num_gpus == 0.5


@pytest.mark.parametrize("verb", sorted(VERBS))
@pytest.mark.parametrize(
    ("concurrency", "expected"),
    [(3, 3), ((1, 4), (1, 4)), ((2, 5, 2), (2, 5)), ((3, 3, 3), 3)],
)
def test_concurrency_reaches_the_stage(verb, concurrency, expected):
    assert _stage(verb, concurrency=concurrency).concurrency == expected


@dataclass
class _ActorPoolStrategy:
    """Ray Data's `ActorPoolStrategy` attributes, as `ray.data` 2.58 sets them."""

    min_size: int
    max_size: float
    initial_size: int
    max_tasks_in_flight_per_actor: int | None = None


@dataclass
class _TaskPoolStrategy:
    size: int | None = None


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_an_actor_pool_strategy_becomes_the_pool_size(verb):
    assert _stage(verb, compute=_ActorPoolStrategy(2, 6, 2)).concurrency == (2, 6)
    assert _stage(verb, compute=_ActorPoolStrategy(4, 4, 4)).concurrency == 4


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_a_task_pool_strategy_leaves_the_stage_on_tasks(verb):
    assert _stage(verb, compute=_TaskPoolStrategy()).concurrency is None
    assert _stage(verb, compute="tasks").concurrency is None


def test_compute_actors_takes_the_pool_size_from_concurrency():
    assert _stage("map_batches", compute="actors", concurrency=2).concurrency == 2


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_ray_remote_args_resources_and_accelerator_reach_the_stage(verb):
    stage = _stage(
        verb, ray_remote_args={"resources": {"TPU": 2}, "accelerator_type": "NVIDIA_A100"}
    )
    assert stage.resources == (("TPU", 2.0),)
    assert stage.accelerator_type == "NVIDIA_A100"


def test_ray_remote_args_num_gpus_reaches_the_stage():
    with pytest.warns(PerformanceWarning, match="re-created on every batch"):
        assert _stage("map_batches", ray_remote_args={"num_gpus": 1}).num_gpus == 1


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_batch_size_reaches_the_stage(verb):
    assert _stage(verb, batch_size=2).batch_size == 2


def test_batch_format_reaches_a_map_batches_stage():
    assert _stage("map_batches", batch_format="numpy").batch_format == "numpy"


def test_fn_constructor_args_build_the_class_once_per_worker():
    plan = _ds().map_batches(_Model, fn_constructor_args=(1,))._plan
    assert isinstance(plan.fn, type)
    assert plan.fn()._inner.offset == 1


def test_fn_args_and_kwargs_are_forwarded_to_every_call():
    def add(batch, a, *, b):
        return {"x": pc.add(pc.add(batch["x"], a), b)}

    assert _ds().map_batches(add, fn_args=(1,), fn_kwargs={"b": 10}).to_pydict() == {
        "x": [12, 13, 14, 15]
    }


def test_zero_copy_batch_false_makes_the_batch_writable_and_true_does_not():
    def zero_first(batch):
        batch["x"][0] = 0
        return batch

    ds = _ds()
    assert ds.map_batches(zero_first, batch_format="numpy", zero_copy_batch=False).to_pydict() == {
        "x": [0, 2, 3, 4]
    }
    with pytest.raises(ValueError, match="read-only"):
        ds.map_batches(zero_first, batch_format="numpy").to_pydict()


# --- refused: the parameter is named ----------------------------------------------------


@pytest.mark.parametrize("verb", sorted(VERBS))
@pytest.mark.parametrize(
    ("kw", "named"),
    [
        ({"num_cpus": 2}, "num_cpus"),
        ({"memory": 1 << 30}, "memory"),
        ({"ray_remote_args_fn": dict}, "ray_remote_args_fn"),
        ({"ray_remote_args": {"num_cpus": 1}}, "ray_remote_args\\['num_cpus'\\]"),
        ({"ray_remote_args": {"memory": 10}}, "ray_remote_args\\['memory'\\]"),
        ({"ray_remote_args": {"max_restarts": 3}}, "ray_remote_args\\['max_restarts'\\]"),
        ({"concurrency": (1, 4, 2)}, "concurrency"),
        ({"compute": _ActorPoolStrategy(1, float("inf"), 1)}, "compute"),
        ({"compute": _ActorPoolStrategy(1, 4, 1, 2)}, "compute"),
        ({"compute": _TaskPoolStrategy(size=4)}, "compute"),
    ],
)
def test_a_parameter_the_scheduler_cannot_honour_raises_naming_it(verb, kw, named):
    with pytest.raises(PlanError, match=rf"{verb}\({named}=.*cannot be honoured"):
        _stage(verb, **kw)


def test_the_default_of_each_refused_parameter_is_accepted():
    """`None` is what Ray Data passes when the caller said nothing, so it must not raise."""
    for verb in VERBS:
        stage = _stage(verb, num_cpus=None, memory=None, ray_remote_args_fn=None, compute=None)
        assert stage.concurrency is None


def test_compute_tasks_with_a_class_is_refused():
    with pytest.raises(PlanError, match=r"compute.*actor pool"):
        _ds().map_batches(_Model, compute="tasks")


def test_compute_actors_on_a_function_needs_a_size():
    with pytest.raises(PlanError, match="needs a pool size"):
        _ds().map_batches(_identity, compute="actors")


def test_compute_and_a_different_concurrency_conflict():
    with pytest.raises(PlanError, match="pass one"):
        _ds().map_batches(_identity, compute=_ActorPoolStrategy(2, 2, 2), concurrency=3)


def test_an_explicit_value_and_its_ray_remote_args_twin_conflict():
    with pytest.raises(PlanError, match="pass it once"):
        _ds().map_batches(_Model, num_gpus=1, ray_remote_args={"num_gpus": 2})


def test_zero_copy_batch_must_be_a_bool():
    with pytest.raises(PlanError, match="zero_copy_batch"):
        _ds().map_batches(_identity, zero_copy_batch="no")


@pytest.mark.parametrize("verb", ["map", "flat_map"])
def test_a_row_verb_takes_python_or_numpy_rows_only(verb):
    with pytest.raises(PlanError, match="batch_format"):
        _stage(verb, batch_format="pandas")


def test_forwarded_udf_options_are_checked_against_the_dataset_verbs():
    """`@bt.udf` validates its options against `Dataset.map_batches`, so the Ray set is legal."""
    bt.udf(num_gpus=0, concurrency=(1, 2, 1), zero_copy_batch=False)(_identity)
    with pytest.raises(PlanError, match="not an option of map_batches"):
        bt.udf(concurency=2)(_identity)


# --- removed spellings raise with guidance ----------------------------------------------


@pytest.mark.parametrize(
    ("target", "name", "replacement"),
    [
        ("dataset", "query", 'ds.filter("x > 1")'),
        ("dataset", "to_torch", "ds.ml.iter_torch_batches"),
        ("ml", "map_batches", "ds.map_batches"),
        ("ml", "map", "ds.map"),
        ("ml", "flat_map", "ds.flat_map"),
        ("ml", "filter", "ds.filter"),
        ("ml", "to_torch", "ds.ml.iter_torch_batches"),
    ],
)
def test_a_removed_spelling_raises_attribute_error_naming_the_kept_one(target, name, replacement):
    obj = _ds() if target == "dataset" else _ds().ml
    with pytest.raises(AttributeError) as caught:
        getattr(obj, name)
    assert replacement in str(caught.value)
    assert not hasattr(obj, name)


def test_the_loaders_that_stay_are_still_there():
    ml = _ds().ml
    for name in ("iter_torch_batches", "to_torch_dataloader", "to_tf"):
        assert callable(getattr(ml, name))
