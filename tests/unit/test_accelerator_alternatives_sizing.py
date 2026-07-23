"""TPU / Trainium / Inferentia inference stages must size like accelerators, not GPUs.

A non-GPU accelerator stage carries `num_gpus == 0` and a custom resource (`{"TPU": 4}`). The
GPU sizing rule fired on any stage with a declared `model_memory_gb`, and the policy assigned a
`num_gpus` fraction whenever `num_gpus <= 0` — so such a stage silently acquired a GPU request
it must never make, and on a GPU-less accelerator fleet that gang never schedules. These pin
that it keeps `num_gpus == 0` while still getting a memory-aware batch seed from its own HBM.
"""

from __future__ import annotations

import pytest

from batcher._internal.accelerators import accelerator_memory_bytes
from batcher.kyber.gpu.policy import decide_gpu_map_params

pytestmark = pytest.mark.unit

_GIB = 1 << 30


@pytest.mark.parametrize(
    ("name", "gib"),
    [
        ("TPU-V2", 8),
        ("TPU-V3", 16),
        ("TPU-V4", 32),
        ("TPU-V5E", 16),
        ("TPU-V5P", 95),
        ("TPU-V6E", 32),
    ],
)
def test_tpu_device_memory_is_known_per_chip(name, gib):
    assert accelerator_memory_bytes(name) == gib * _GIB


@pytest.mark.parametrize("name", ["aws-neuron-core", "Intel-GAUDI"])
def test_ambiguous_vendor_labels_report_unknown(name):
    """One Ray label covers several generations with different memory, so the label alone can't
    determine it. Unknown (0) is safer than a fabricated figure per this module's contract."""
    assert accelerator_memory_bytes(name) == 0


def test_a_tpu_stage_never_acquires_a_gpu_request():
    """The regression: `num_gpus <= 0` triggered a GPU-fraction assignment, so a TPU stage asked
    for a device the fleet hasn't got."""
    params = decide_gpu_map_params(8.0, 0.0, None, gpu_memory_gb=16.0, assign_num_gpus=False)
    assert params.num_gpus == 0.0  # stays off the GPU
    assert params.batch_size is not None and params.batch_size >= 1  # but still seeded


def test_a_tpu_stage_batch_seed_scales_with_its_hbm():
    """A v5p (95 GB) has far more headroom after the model than a v2 (8 GB), so it seeds a larger
    batch — the same memory-aware sizing a GPU gets, keyed on the accelerator's own memory."""
    big = decide_gpu_map_params(4.0, 0.0, None, gpu_memory_gb=95.0, assign_num_gpus=False)
    small = decide_gpu_map_params(4.0, 0.0, None, gpu_memory_gb=8.0, assign_num_gpus=False)
    assert (big.batch_size or 0) > (small.batch_size or 0)


def test_the_gpu_path_is_unchanged():
    """A GPU stage (assign_num_gpus=True, the default) still packs onto a fraction as before."""
    params = decide_gpu_map_params(4.0, 0.0, None, gpu_memory_gb=80.0)
    assert 0.0 < params.num_gpus <= 1.0


def _ds_and_node(**kw):
    import batcher as bt

    ds = bt.from_pydict({"x": [1]})
    return ds, ds.ml.map_batches(lambda b: b, **kw)._plan


def _ctx(ds, hardware=None):
    from batcher.config import active_config
    from batcher.kyber.cardinality import StatsEstimator
    from batcher.kyber.pass_base import OptimizerContext
    from batcher.plan.resource import HardwareProfile

    return OptimizerContext(
        config=active_config(),
        sources=ds._sources,
        hub=None,
        estimator=StatsEstimator(ds._sources),
        hardware=hardware or HardwareProfile.local(),
    )


def test_the_sizing_rule_keeps_a_tpu_stage_off_the_gpu():
    """End to end through the Kyber rule: a TPU stage gets a batch size but no num_gpus."""
    from batcher.kyber.gpu.sizing import size_gpu_map_batches

    ds, node = _ds_and_node(resources={"TPU": 4}, accelerator_type="TPU-V5P", model_memory_gb=8.0)
    out = size_gpu_map_batches(node, _ctx(ds))
    assert out is not None
    assert out.num_gpus == 0.0
    assert out.batch_size is not None and out.batch_size >= 1
    assert out.resources == (("TPU", 4),)  # the accelerator request is preserved


def test_the_sizing_rule_still_packs_a_gpu_stage():
    from batcher.kyber.gpu.sizing import size_gpu_map_batches
    from batcher.plan.resource import HardwareProfile

    ds, node = _ds_and_node(num_gpus=1, model_memory_gb=4.0)
    hw = HardwareProfile.for_cluster(
        cpu_cores=8, memory_bytes=64 << 30, worker_count=2, gpu_count=2, gpu_memory_bytes=80 << 30
    )
    out = size_gpu_map_batches(node, _ctx(ds, hw))
    assert out is not None
    assert 0.0 < out.num_gpus <= 1.0  # a GPU stage still gets a GPU fraction


def _pool(monkeypatch, *, num_gpus, resources, cluster, workers=8, parts=1000):
    import types

    import batcher.ml.gpu as mlgpu

    fake_ray = types.SimpleNamespace(cluster_resources=lambda: cluster)
    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)
    return mlgpu.gpu_aware_pool_default(num_gpus, workers, parts, None, resources=resources)


def test_a_tpu_pool_fills_the_clusters_chips(monkeypatch):
    """The foot-gun the GPU path avoids applies to TPUs too: a stage asking 4 chips on a
    32-chip pod should spawn 8 replicas, not fall back to the worker count."""
    pool = _pool(monkeypatch, num_gpus=0.0, resources={"TPU": 4.0}, cluster={"TPU": 32.0, "CPU": 8})
    assert pool == 8


def test_a_tpu_pool_is_bounded_by_the_scarcest_resource(monkeypatch):
    """A stage needing two custom resources fills the scarcer of them."""
    pool = _pool(
        monkeypatch,
        num_gpus=0.0,
        resources={"TPU": 4.0, "special": 1.0},
        cluster={"TPU": 32.0, "special": 3.0},
    )
    assert pool == 3  # special/1 = 3 constrains below TPU/4 = 8


def test_falls_back_to_workers_when_the_accelerator_is_absent(monkeypatch):
    """A requested accelerator the cluster doesn't advertise → don't guess a pool; keep the
    worker-count fallback rather than sizing to a resource that isn't there."""
    pool = _pool(
        monkeypatch, num_gpus=0.0, resources={"TPU": 4.0}, cluster={"CPU": 16.0}, workers=5
    )
    assert pool == 5


def test_a_plain_cpu_stage_still_returns_the_worker_count(monkeypatch):
    pool = _pool(monkeypatch, num_gpus=0.0, resources={}, cluster={"CPU": 16.0}, workers=7)
    assert pool == 7


def _recommend(monkeypatch, model_gb, nodes):
    import types

    import batcher.dist.executors.ray_runtime.accelerators as accel
    from batcher.dist.executors.ray_runtime import scaling

    monkeypatch.setitem(
        __import__("sys").modules, "ray", types.SimpleNamespace(is_initialized=lambda: True)
    )
    monkeypatch.setattr(scaling, "node_classes", lambda: nodes)
    return accel.recommend_accelerator_type(model_gb)


def _mixed():
    return [
        {"cpus": 8.0, "gpus": 4.0, "accelerators": 0.0, "accelerator_type": "NVIDIA_TESLA_T4"},
        {"cpus": 8.0, "gpus": 8.0, "accelerators": 0.0, "accelerator_type": "NVIDIA_A100_80G"},
    ]


def test_a_large_model_is_pinned_to_the_device_that_fits(monkeypatch):
    """A 40 GB model can't load on a 16 GB T4, so an unpinned stage that lands there OOMs. It's
    pinned to the A100, the smallest class that fits."""
    assert _recommend(monkeypatch, 40.0, _mixed()) == "NVIDIA_A100_80G"


def test_a_small_model_is_left_unpinned(monkeypatch):
    """When every device fits, a pin only constrains placement for no benefit."""
    assert _recommend(monkeypatch, 4.0, _mixed()) is None


def test_the_smallest_fitting_class_is_chosen(monkeypatch):
    """Among several classes that fit, pin to the smallest so the scarce big devices stay free."""
    nodes = [
        *_mixed(),
        {"cpus": 8.0, "gpus": 8.0, "accelerators": 0.0, "accelerator_type": "NVIDIA_L40S"},  # 48 GB
    ]
    # A 20 GB model fits L40S (48*0.85=40.8) and A100 (68); pick L40S, the smaller.
    assert _recommend(monkeypatch, 20.0, nodes) == "NVIDIA_L40S"


def test_a_homogeneous_cluster_is_never_pinned(monkeypatch):
    nodes = [
        {"cpus": 8.0, "gpus": 4.0, "accelerators": 0.0, "accelerator_type": "NVIDIA_TESLA_T4"},
        {"cpus": 8.0, "gpus": 4.0, "accelerators": 0.0, "accelerator_type": "NVIDIA_TESLA_T4"},
    ]
    assert _recommend(monkeypatch, 40.0, nodes) is None


def test_a_model_too_big_for_every_device_is_not_pinned(monkeypatch):
    """Nothing fits one device → don't pin (the sizing path shards across devices instead)."""
    assert _recommend(monkeypatch, 500.0, _mixed()) is None
