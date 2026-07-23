"""A Trainium or Gaudi host must self-identify, not report itself as a plain CPU box.

`accelerator_backend` detected NVIDIA/AMD/Intel/Apple/TPU but fell through to ``cpu`` on AWS
Neuron (Trainium/Inferentia) and Intel Gaudi — so a host with a working accelerator looked
device-less to diagnostics, device-keyed learning, and `torch_device`. Detection uses the
device nodes because the vendor frameworks initialize their runtime on import.
"""

from __future__ import annotations

import pytest

import batcher._internal.hardware as hw
from batcher.ml.gpu import torch_device, vram_context_overhead

pytestmark = pytest.mark.unit


def _no_torch(monkeypatch):
    """Force the torch-less path so detection turns on the device-node probes alone."""
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_a_neuron_host_reports_neuron(monkeypatch):
    import batcher._internal.accelerators as accel

    _no_torch(monkeypatch)
    monkeypatch.setattr(hw, "gpu_devices_absent", lambda: False)
    monkeypatch.setattr(hw, "_tpu_available", lambda: False)
    monkeypatch.setattr(accel, "has_neuron_device", lambda: True)
    monkeypatch.setattr(accel, "has_gaudi_device", lambda: False)
    assert hw.accelerator_backend() == "neuron"


def test_a_gaudi_host_reports_hpu(monkeypatch):
    import batcher._internal.accelerators as accel

    _no_torch(monkeypatch)
    monkeypatch.setattr(hw, "gpu_devices_absent", lambda: False)
    monkeypatch.setattr(hw, "_tpu_available", lambda: False)
    monkeypatch.setattr(accel, "has_neuron_device", lambda: False)
    monkeypatch.setattr(accel, "has_gaudi_device", lambda: True)
    assert hw.accelerator_backend() == "hpu"


def test_a_bare_cpu_host_still_reports_cpu(monkeypatch):
    monkeypatch.setattr(hw, "gpu_devices_absent", lambda: True)
    monkeypatch.setattr(hw, "_tpu_available", lambda: False)
    assert hw.accelerator_backend() == "cpu"


def test_device_nodes_disambiguate_neuron_and_gaudi(monkeypatch):
    """Neuron uses /dev/neuron*, Gaudi uses /dev/hl* — specific, unlike the shared /dev/accel*."""
    import batcher._internal.accelerators as accel

    def only(substr, hit):
        return lambda pattern, **k: [hit] if substr in pattern else []

    monkeypatch.setattr(accel.glob, "glob", only("neuron", "/dev/neuron0"))
    assert accel.has_neuron_device() is True
    assert accel.has_gaudi_device() is False
    monkeypatch.setattr(accel.glob, "glob", only("hl", "/dev/hl0"))
    assert accel.has_gaudi_device() is True
    assert accel.has_neuron_device() is False


@pytest.mark.parametrize(
    ("backend", "device"),
    [("neuron", "xla"), ("hpu", "hpu"), ("tpu", "xla"), ("cuda", "cuda"), ("rocm", "cuda")],
)
def test_torch_device_maps_the_new_backends(backend, device):
    assert torch_device(backend) == device


def test_context_overhead_is_defined_for_the_new_backends():
    """A lookup miss would fall to the 0.4 CUDA default, over-reserving on a device with none."""
    assert vram_context_overhead("neuron") == 0.0
    assert vram_context_overhead("hpu") == 0.0
