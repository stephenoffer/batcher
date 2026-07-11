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


def test_envelope_tightens_packing_from_learned_peak_vram(monkeypatch):
    # A prior run's MEASURED peak VRAM refines packing beyond the declared model size: an
    # actor that really peaked at 60% of VRAM only fits ~1 per GPU, so it gets a whole GPU
    # even though the declared 2 GB would pack onto a fraction. Measurement only tightens.
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend
    from batcher.ml.gpu import gpu_feedback_key, record_gpu_peak_vram

    monkeypatch.setattr("batcher.ml.gpu.gpu_vram_gb", lambda: 24.0)
    ds = bt.from_pydict({"x": [1, 2, 3]}).ml.infer(_Model, num_gpus=1.0, model_memory_gb=2.0)
    hub = MetadataHub(InProcessBackend())
    cold = _map_scheduling_envelope(ds._plan, 4, hub).num_gpus  # declared-only packing
    assert 0 < cold < 1.0
    record_gpu_peak_vram(hub, gpu_feedback_key(ds._plan), 0.6)  # measured 60% VRAM peak
    warm = _map_scheduling_envelope(ds._plan, 4, hub).num_gpus
    assert warm == 1.0  # 60% peak → 1 actor/GPU → whole GPU; never packs looser than cold
    assert warm >= cold


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
