"""Non-NVIDIA accelerators must be reachable, not just requestable.

The scheduling surface has long been vendor-neutral, but the *decision and execution* surface
was written against CUDA. These pin the places where a host with a working Intel (`xpu`) or
Apple (`mps`) device was told it had no accelerator, or where a device this module cannot
actually drive was claimed as usable. Each failure is silent: the query runs correctly on the
CPU engine, just without the device the host paid for.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _FakeMps:
    def __init__(self, ok: bool) -> None:
        self._ok = ok

    def is_available(self) -> bool:
        return self._ok


def _probe(monkeypatch, backend: str, *, cuda=False, xpu=None, mps=None) -> bool:
    import types

    import batcher.core.gpu_transform as gt

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=_FakeMps(mps) if mps is not None else None),
    )
    if xpu is not None:
        torch.xpu = _FakeMps(xpu)
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setattr(gt, "gpu_devices_absent", lambda: False)
    monkeypatch.setattr(gt, "accelerator_backend", lambda: backend)
    gt.gpu_available.cache_clear()
    try:
        return gt.gpu_available()
    finally:
        gt.gpu_available.cache_clear()


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_cuda_and_rocm_are_unchanged(monkeypatch, backend):
    """ROCm speaks the CUDA API, so both resolve through `torch.cuda`."""
    assert _probe(monkeypatch, backend, cuda=True) is True
    assert _probe(monkeypatch, backend, cuda=False) is False


def test_an_intel_xpu_host_is_not_reported_as_gpu_less(monkeypatch):
    """The regression: the gate asked `torch.cuda.is_available()`, which is False on Intel —
    so the one function deciding whether to use the device said "no accelerator" on a host
    with a working one, and the xpu kernel below it was unreachable."""
    assert _probe(monkeypatch, "xpu", cuda=False, xpu=True) is True
    assert _probe(monkeypatch, "xpu", cuda=False, xpu=False) is False


def test_an_apple_mps_host_is_not_reported_as_gpu_less(monkeypatch):
    assert _probe(monkeypatch, "mps", cuda=False, mps=True) is True
    assert _probe(monkeypatch, "mps", cuda=False, mps=False) is False


def test_a_device_these_kernels_cannot_drive_stays_unavailable(monkeypatch):
    """A TPU host has an accelerator, but not one this module's torch kernels can drive.
    Claiming it would route work to a device with no kernel behind it."""
    assert _probe(monkeypatch, "tpu", cuda=False) is False
    assert _probe(monkeypatch, "cpu", cuda=False) is False


def _clamp(monkeypatch, nodes, workers, num_cpus, num_gpus=0.0):
    import ray

    from batcher.dist.executors.ray_runtime import scaling

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    # `placeable_workers` imports `node_classes` from `scaling` at call time, so patching it
    # here reaches both the clamp and the capacity check.
    monkeypatch.setattr(scaling, "node_classes", lambda: nodes)
    monkeypatch.setattr(
        scaling,
        "cluster_topology",
        lambda: {
            "nodes": len(nodes),
            "cpus": sum(n["cpus"] for n in nodes),
            "gpus": sum(n["gpus"] for n in nodes),
            "memory": 0.0,
            "min_node_memory": 0.0,
        },
    )
    return scaling.clamp_workers(workers, num_cpus, num_gpus)


def test_fanout_is_bounded_by_what_a_single_node_can_host(monkeypatch):
    """The regression: capacity came from `total_cpus / num_cpus`. Four 8-core nodes total 32
    cores, so the sum admitted two `num_cpus=16` workers while no node could host even one.
    The fleet is gang-scheduled, so that over-count leaves the placement group unsatisfiable
    and the job hangs rather than failing."""
    nodes = [{"cpus": 8.0, "gpus": 0.0, "accelerator_type": None} for _ in range(4)]
    assert _clamp(monkeypatch, nodes, workers=2, num_cpus=16.0) == 1  # floors at 1, never 2


def test_a_homogeneous_cluster_is_unchanged(monkeypatch):
    """Totals and per-node counts agree when nodes are equal, so nothing moves."""
    nodes = [{"cpus": 8.0, "gpus": 0.0, "accelerator_type": None} for _ in range(4)]
    assert _clamp(monkeypatch, nodes, workers=4, num_cpus=8.0) == 4
    assert _clamp(monkeypatch, nodes, workers=8, num_cpus=4.0) == 8


def test_a_skewed_cluster_counts_each_node_separately(monkeypatch):
    """One 16-core node hosts two 8-core workers; three 4-core nodes host none."""
    nodes = [{"cpus": 16.0, "gpus": 0.0, "accelerator_type": None}] + [
        {"cpus": 4.0, "gpus": 0.0, "accelerator_type": None} for _ in range(3)
    ]
    # Totals say 28/8 = 3; only the big node can host, and it holds 2.
    assert _clamp(monkeypatch, nodes, workers=3, num_cpus=8.0) == 2


def test_a_tpu_node_is_recognized_as_an_accelerator_node():
    """A TPU/Trainium node carries `gpus == 0` plus a custom resource, so classifying by GPUs
    alone counted it CPU-only — and a CPU shuffle could then steal its cores from an inference
    stage. `is_accelerator_node` recognizes it."""
    from batcher._internal.accelerators import accelerator_units, is_accelerator_node

    assert accelerator_units({"TPU": 4.0, "CPU": 8.0}) == 4.0
    assert accelerator_units({"neuron_cores": 2.0}) == 2.0
    assert accelerator_units({"CPU": 8.0}) == 0.0
    assert is_accelerator_node({"gpus": 0.0, "accelerators": 4.0}) is True
    assert is_accelerator_node({"gpus": 1.0, "accelerators": 0.0}) is True
    assert is_accelerator_node({"gpus": 0.0, "accelerators": 0.0}) is False


def test_cpu_fleet_isolation_triggers_on_a_pure_tpu_plus_cpu_cluster(monkeypatch):
    """The regression: isolation keyed on GPU nodes, so a TPU-plus-CPU cluster (no GPUs) got no
    isolation at all and its TPU-node cores were offered to a CPU shuffle."""
    from batcher.dist.executors.ray_runtime import scaling

    nodes = [
        {"cpus": 16.0, "gpus": 0.0, "accelerators": 4.0, "accelerator_type": "TPU-V5P"},
        {"cpus": 32.0, "gpus": 0.0, "accelerators": 0.0, "accelerator_type": None},
    ]
    monkeypatch.setattr(scaling, "node_classes", lambda: nodes)
    # 8 single-core workers fit on the 32 CPU-only cores, keeping them off the 16 TPU-node cores.
    assert scaling.cpu_only_can_host(8, 1.0) is True
    # 40 would need the TPU node's cores too, so the restriction is dropped (don't under-provision).
    assert scaling.cpu_only_can_host(40, 1.0) is False


def test_a_pure_cpu_cluster_still_gets_no_isolation(monkeypatch):
    from batcher.dist.executors.ray_runtime import scaling

    nodes = [{"cpus": 16.0, "gpus": 0.0, "accelerators": 0.0, "accelerator_type": None}]
    monkeypatch.setattr(scaling, "node_classes", lambda: nodes)
    assert scaling.cpu_only_can_host(4, 1.0) is False  # nothing to keep off → no restriction
