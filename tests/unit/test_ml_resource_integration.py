"""ML stages integrate with Kyber's cost model and the resource (Carbonite) envelope.

A GPU inference stage must cost far more per row than a trivial map (so Kyber pushes
filters/sampling below it), and the scheduling envelope must VRAM-pack a small model
onto a fractional GPU, budget host memory for the model, and pin the accelerator —
the resource safety the relational path gets, now for the map/inference path.
"""

from __future__ import annotations

import batcher as bt
from batcher.api.executors import _map_scheduling_envelope
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.plan.logical import MapBatches


class _Model:
    def __call__(self, batch):
        return batch


def _cost_model(ds):
    return CostModel(CardinalityEstimator(ds._sources))


def test_gpu_inference_costs_far_more_than_cpu_map():
    ds = bt.from_pydict({"x": list(range(1000))})
    cm = _cost_model(ds)
    cpu = cm.op_cost(MapBatches(ds._plan, lambda b: b)).cpu
    gpu = cm.op_cost(MapBatches(ds._plan, _Model, num_gpus=1.0)).cpu
    assert gpu >= 100 * cpu  # GPU forward pass is the costed bottleneck


def test_gpu_cost_scales_with_model_size():
    ds = bt.from_pydict({"x": list(range(1000))})
    cm = _cost_model(ds)
    small = cm.op_cost(MapBatches(ds._plan, _Model, num_gpus=1.0, model_memory_gb=1.0)).cpu
    big = cm.op_cost(MapBatches(ds._plan, _Model, num_gpus=1.0, model_memory_gb=14.0)).cpu
    assert big > small


def test_envelope_vram_packs_small_model(monkeypatch):
    # A 24GB GPU detected → a 2GB model packs onto a fraction.
    monkeypatch.setattr("batcher.ml.gpu.gpu_vram_gb", lambda: 24.0)
    ds = bt.from_pydict({"x": [1, 2, 3]}).ml.infer(
        _Model, num_gpus=1.0, model_memory_gb=2.0, accelerator_type="NVIDIA_A100"
    )
    env = _map_scheduling_envelope(ds._plan, 4, None)
    assert 0 < env.num_gpus < 1.0  # packed onto a fraction
    assert env.accelerator_type == "NVIDIA_A100"
    assert env.memory_bytes == int(2.0 * 1.5 * (1 << 30))  # host budget for the model


def test_envelope_honors_declared_gpus_without_detectable_vram(monkeypatch):
    # GPU-less driver can't detect VRAM → can't VRAM-pack, so the declared request stands.
    monkeypatch.setattr("batcher.ml.gpu.gpu_vram_gb", lambda: None)
    ds = bt.from_pydict({"x": [1, 2, 3]}).ml.infer(_Model, num_gpus=1.0, model_memory_gb=2.0)
    env = _map_scheduling_envelope(ds._plan, 4, None)
    assert env.num_gpus == 1.0


def test_envelope_no_gpu_no_memory_budget():
    ds = bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(lambda b: b)
    env = _map_scheduling_envelope(ds._plan, 2, None)
    assert env.num_gpus == 0.0
    assert env.memory_bytes == 0
