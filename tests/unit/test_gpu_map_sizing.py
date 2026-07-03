"""Kyber sizes a GPU inference stage's `num_gpus` / `batch_size` from the model footprint.

`decide_gpu_map_params` is the pure decision; `size_gpu_map_batches` is the SELECTION-phase rule
that applies it to a `MapBatches` node. Kyber fills only what the user left unset, and only when
`model_memory_gb` is known: pack light models several per GPU, reserve whole GPUs for a heavy
one, and seed a VRAM-aware initial batch size. Head-runnable, no GPU.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher.config import active_config, set_config
from batcher.kyber.gpu.policy import decide_gpu_map_params
from batcher.kyber.gpu.sizing import size_gpu_map_batches
from batcher.plan.logical import MapBatches

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_config():
    saved = active_config()
    yield
    set_config(saved)


def _set_gpu(**overrides):
    cfg = active_config()
    set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, **overrides)))


def test_light_model_packs_a_gpu_fraction(restore_config):
    _set_gpu(gpu_memory_gb=12.0)
    p = decide_gpu_map_params(3.0, 0.0, None)  # 3GB of ~10.2 usable -> ~0.29 -> quantum 0.5
    assert 0 < p.num_gpus <= 1.0
    assert p.batch_size and p.batch_size > 0


def test_heavy_model_reserves_whole_gpus(restore_config):
    _set_gpu(gpu_memory_gb=12.0)
    p = decide_gpu_map_params(30.0, 0.0, None)  # > one GPU -> ceil(30 / 10.2) = 3
    assert p.num_gpus == 3.0
    # a heavy model leaves little VRAM headroom -> a smaller initial batch than a light one
    assert p.batch_size and p.batch_size < decide_gpu_map_params(3.0, 0.0, None).batch_size


def test_user_pinned_values_are_honored():
    p = decide_gpu_map_params(3.0, 1.0, 128)
    assert p.num_gpus == 1.0 and p.batch_size == 128


def test_unknown_model_is_left_untouched():
    p = decide_gpu_map_params(0.0, 0.0, None)
    assert p.num_gpus == 0.0 and p.batch_size is None


def _ctx(ds):
    from batcher.kyber.pass_base import OptimizerContext
    from batcher.kyber.stats.estimator import StatsEstimator

    return OptimizerContext(
        config=active_config(),
        sources=ds._sources,
        hub=None,
        estimator=StatsEstimator(ds._sources, learned={}),
    )


def test_rule_fills_unset_num_gpus_and_batch_size(restore_config):
    _set_gpu(gpu_memory_gb=12.0)

    class Model:
        def __call__(self, b):
            return b

    ds = bt.from_pydict({"x": [1, 2, 3]})
    node = MapBatches(input=ds._plan, fn=Model, model_memory_gb=3.0)
    out = size_gpu_map_batches(node, _ctx(ds))
    assert isinstance(out, MapBatches)
    assert out.num_gpus > 0.0 and out.batch_size is not None


def test_rule_is_a_noop_when_user_pinned_both(restore_config):
    class Model:
        def __call__(self, b):
            return b

    ds = bt.from_pydict({"x": [1, 2, 3]})
    node = MapBatches(input=ds._plan, fn=Model, model_memory_gb=3.0, num_gpus=1.0, batch_size=64)
    assert size_gpu_map_batches(node, _ctx(ds)) is None


def test_rule_ignores_a_non_gpu_stage(restore_config):
    ds = bt.from_pydict({"x": [1, 2, 3]})
    node = MapBatches(input=ds._plan, fn=lambda b: b)  # no model_memory_gb
    assert size_gpu_map_batches(node, _ctx(ds)) is None
