"""Accelerators that are neither NVIDIA nor AMD, and the two ways they were mis-counted.

Batcher runs wherever the capacity is, and on a growing share of it the accelerator is a
Trainium, a Gaudi, a TPU or an Ascend NPU. None of them is enumerable through NVML or
`torch.cuda`, so they are read from their device nodes — and reading device nodes has two
failure modes that only appear on real hardware:

* **Lexicographic ordering.** A `trn1.32xlarge` has sixteen `/dev/neuron*`. Sorted as strings
  they run 0, 1, 10, 11, ..., 15, 2, 3 — so device 10 is reported as index 2 and every
  consumer lining an index up against a framework's ordering addresses the wrong chip.
* **A vendor nobody enumerated.** Ascend's Ray resource (`NPU`) was already recognized while
  its device nodes were not, so an Ascend node reported no accelerator at all.
"""

from __future__ import annotations

import pytest

import batcher._internal.accelerators as accel
import batcher._internal.hardware as hw
from batcher.ml.devices import resolve_device
from batcher.ml.gpu import torch_device, vram_context_overhead

pytestmark = pytest.mark.unit


def _fake_nodes(monkeypatch, nodes: dict[str, list[str]]):
    """Answer `glob` from a pattern-substring to device-node map, in an arbitrary order.

    Returned unsorted on purpose: the inventory is what must impose an order, and a fixture
    that hands it a sorted list cannot see it fail to.
    """

    def fake_glob(pattern, **_kwargs):
        for token, hits in nodes.items():
            if token in pattern:
                return list(hits)
        return []

    monkeypatch.setattr(accel.glob, "glob", fake_glob)
    accel.reset_accelerator_probes()


def test_sixteen_neuron_devices_are_indexed_by_number_not_by_string(monkeypatch):
    devices = [f"/dev/neuron{i}" for i in range(16)]
    _fake_nodes(monkeypatch, {"neuron": sorted(devices)})  # the order `glob` really returns
    inventory = accel.gpu_inventory()
    assert [d["index"] for d in inventory] == list(range(16))
    assert [d["name"] for d in inventory] == [f"Neuron (neuron{i})" for i in range(16)]
    accel.reset_accelerator_probes()


def test_an_ascend_host_is_an_accelerator_host(monkeypatch):
    _fake_nodes(monkeypatch, {"davinci": ["/dev/davinci0", "/dev/davinci1"]})
    assert accel.has_ascend_device() is True
    assert accel.gpu_devices_absent() is False, "an Ascend node is not a device-less one"
    assert [d["name"] for d in accel.gpu_inventory()] == [
        "Ascend (davinci0)",
        "Ascend (davinci1)",
    ]
    accel.reset_accelerator_probes()


def test_an_ascend_host_reports_the_npu_backend(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.setattr(accel, "gpu_devices_absent", lambda: False)
    monkeypatch.setattr(accel, "_tpu_available", lambda: False)
    monkeypatch.setattr(accel, "has_neuron_device", lambda: False)
    monkeypatch.setattr(accel, "has_gaudi_device", lambda: False)
    monkeypatch.setattr(accel, "has_ascend_device", lambda: True)
    assert hw.accelerator_backend() == "npu"


@pytest.mark.parametrize(("name", "device"), [("npu", "npu"), ("ascend", "npu")])
def test_an_ascend_device_can_be_asked_for_by_name(name, device):
    # Auto-detection is not the only way onto a device: a user naming it explicitly must
    # reach the same backend, the way `gaudi` and `trainium` already do.
    assert resolve_device(name) == device


def test_the_npu_backend_maps_to_the_torch_npu_device_string():
    assert torch_device("npu") == "npu"
    assert vram_context_overhead("npu") == 0.0


def test_indices_stay_continuous_across_vendors(monkeypatch):
    # Restarting the count per vendor broke `gpu_inventory()[i]["index"] == i`, which every
    # other probe here holds to.
    _fake_nodes(monkeypatch, {"neuron": ["/dev/neuron0"], "hl": ["/dev/hl0", "/dev/hl1"]})
    assert [d["index"] for d in accel.gpu_inventory()] == [0, 1, 2]
    accel.reset_accelerator_probes()


def test_the_two_visibility_vocabularies_cannot_drift():
    # `accelerators.VISIBLE_DEVICE_ENVS` renumbers an NVML- or AMD-probed device list;
    # `scheduler.VISIBLE_DEVICE_COUNT_ENVS` counts a grant across every vendor. They answer
    # different questions about the same variables, and `accelerators` says why a second copy
    # is dangerous: "a vendor variable added to one copy and not the others is a silent
    # disagreement about which devices a process owns".
    from batcher._internal.accelerators import VISIBLE_DEVICE_ENVS
    from batcher._internal.site.scheduler import VISIBLE_DEVICE_COUNT_ENVS

    assert VISIBLE_DEVICE_COUNT_ENVS[: len(VISIBLE_DEVICE_ENVS)] == VISIBLE_DEVICE_ENVS
