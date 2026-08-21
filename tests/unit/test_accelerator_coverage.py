"""Every accelerator answers the four questions the adaptive loops steer by.

The packing loop targets 80% utilization and the autobatcher caps on VRAM, and both are
**no-ops when their reading is `None`** — `recommend_num_gpus` returns the declared request
unchanged, and `ThroughputController` climbs with no ceiling. So a backend missing from the
memory or utilization layer does not fail, it silently loses both the utilization target and
the predictive out-of-memory guard. These assert coverage rather than values, with fake torch
namespaces, so they run on a CPU-only host.
"""

from __future__ import annotations

import sys
import types

import pytest

from batcher._internal.hardware.devices import (
    TORCH_NAMESPACE,
    accelerator_namespace,
    device_memory_used_fraction,
    device_total_memory_bytes,
    device_utilization,
    is_device_oom,
    release_device_cache,
)

pytestmark = pytest.mark.unit

#: Every accelerator `accelerator_backend()` can return. `tpu`/`neuron` run through XLA, which
#: has no caching allocator, and `cpu` is not an accelerator — the rest must all be readable.
ALL_BACKENDS = ("cuda", "rocm", "xpu", "mps", "tpu", "neuron", "hpu", "npu", "cpu")
XLA_BACKENDS = ("tpu", "neuron")


def _fake_namespace(*, total=8 << 30, reserved=2 << 30, util=42, ordinal_arg=True):
    ns = types.SimpleNamespace()
    ns.is_available = lambda: True
    ns.is_initialized = lambda: True
    ns.get_device_properties = lambda i=0: types.SimpleNamespace(total_memory=total)
    ns.memory_reserved = lambda i=0: reserved
    ns.memory_allocated = lambda i=0: reserved // 2
    ns.empty_cache = lambda: None
    ns.set_per_process_memory_fraction = lambda f: None
    ns.utilization = (lambda i=0: util) if ordinal_arg else (lambda: util)
    return ns


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a fake `torch` exposing every accelerator namespace at once."""
    torch = types.ModuleType("torch")
    for module_name in dict.fromkeys(TORCH_NAMESPACE.values()):
        setattr(torch, module_name, _fake_namespace())
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_the_backend_detector_and_the_memory_table_agree_on_what_is_readable():
    # A backend the detector can return but the memory table has never heard of is a backend
    # whose VRAM cap is inert, and nothing anywhere says so.
    unreadable = set(ALL_BACKENDS) - set(TORCH_NAMESPACE) - set(XLA_BACKENDS) - {"cpu"}
    assert not unreadable, f"no memory reading for {sorted(unreadable)}"


@pytest.mark.parametrize("backend", sorted(TORCH_NAMESPACE))
def test_every_backend_resolves_a_namespace(backend, fake_torch):
    assert accelerator_namespace(backend) is not None


@pytest.mark.parametrize("backend", sorted(TORCH_NAMESPACE))
def test_every_backend_reports_its_total_memory(backend, fake_torch):
    assert device_total_memory_bytes(backend) == 8 << 30


@pytest.mark.parametrize("backend", sorted(TORCH_NAMESPACE))
def test_every_backend_reports_the_share_its_allocator_holds(backend, fake_torch):
    assert device_memory_used_fraction(backend) == pytest.approx(0.25)


@pytest.mark.parametrize("backend", ["xpu", "hpu", "npu"])
def test_the_namespace_backends_report_utilization(backend, fake_torch):
    # These three have no SMI library the control plane can link, so the torch counter is the
    # only route to the 80% packing target on them.
    assert device_utilization(backend) == pytest.approx(0.42)


@pytest.mark.parametrize("backend", ["xpu", "hpu", "npu"])
def test_the_utilization_registry_covers_them_end_to_end(backend, fake_torch):
    from batcher.ml.gpu import _UTILIZATION, sample_gpu_utilization

    assert backend in _UTILIZATION
    assert sample_gpu_utilization(backend) == pytest.approx(0.42)


def test_a_utilization_counter_that_takes_no_ordinal_still_reports(monkeypatch):
    # A namespace whose `utilization()` reports for the current device raises TypeError when
    # handed an ordinal. Treating that as unsupported would lose the reading entirely.
    torch = types.ModuleType("torch")
    torch.npu = _fake_namespace(util=77, ordinal_arg=False)
    torch.backends = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch)
    from batcher.ml.gpu import sample_gpu_utilization

    assert sample_gpu_utilization("npu") == pytest.approx(0.77)


def test_an_unavailable_namespace_reads_as_absent_not_as_zero(monkeypatch):
    # A torch build can carry `torch.xpu` on a machine with no Intel GPU. Answering from it
    # would make a plausible wrong number out of an absent one.
    torch = types.ModuleType("torch")
    torch.xpu = _fake_namespace()
    torch.xpu.is_available = lambda: False
    torch.backends = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert accelerator_namespace("xpu") is None
    assert device_total_memory_bytes("xpu") is None
    assert device_memory_used_fraction("xpu") is None


def test_the_fragmentation_signal_is_not_cuda_only(fake_torch, monkeypatch):
    # Fragmentation is what tells "the device is full" from "the memory is there in pieces",
    # and the two want opposite responses from the OOM ladder.
    from batcher._internal.hardware.devices import fragmentation_ratio

    monkeypatch.setattr(
        "batcher._internal.accelerators.accelerator_backend", lambda: "hpu", raising=False
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.devices.torch_memory._torch", lambda: fake_torch
    )
    assert fragmentation_ratio() == pytest.approx(0.5)


def test_the_per_process_cap_applies_on_every_backend_that_has_one(fake_torch, monkeypatch):
    # The cap is what makes packing several actors onto one board safe rather than merely
    # dense: a stage that misjudges its footprint fails its own allocation instead of taking
    # every co-tenant on the device down with it.
    from batcher._internal.hardware.devices import set_memory_fraction

    applied = []
    for backend in ("cuda", "xpu", "hpu", "npu"):
        ns = getattr(fake_torch, TORCH_NAMESPACE[backend])
        ns.set_per_process_memory_fraction = lambda f, b=backend: applied.append(b)
        monkeypatch.setattr(
            "batcher._internal.accelerators.accelerator_backend", lambda b=backend: b
        )
        set_memory_fraction(0.5)
    assert applied == ["cuda", "xpu", "hpu", "npu"]


def test_a_backend_with_no_cap_declines_rather_than_claiming_it_applied(monkeypatch):
    from batcher._internal.hardware.devices import set_memory_fraction

    torch = types.ModuleType("torch")
    torch.npu = _fake_namespace()
    del torch.npu.set_per_process_memory_fraction
    torch.backends = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("batcher._internal.accelerators.accelerator_backend", lambda: "npu")
    assert set_memory_fraction(0.5) is False


def test_the_cache_release_reaches_every_backend(fake_torch):
    emptied = []
    for module_name in dict.fromkeys(TORCH_NAMESPACE.values()):
        ns = getattr(fake_torch, module_name)
        ns.empty_cache = lambda n=module_name: emptied.append(n)
    assert release_device_cache() is True
    assert set(emptied) == set(TORCH_NAMESPACE.values())


class TestOomVocabulary:
    """An exhaustion the classifier does not recognize loses the halving retry entirely."""

    @pytest.mark.parametrize(
        "message",
        [
            # An unambiguous exhaustion stands on its own, in any separator style. The marker
            # this replaced read `hip_error_out_of_memory` and matched none of these.
            "CUDA out of memory. Tried to allocate 2.00 GiB",
            "HIP out of memory",
            "hipErrorOutOfMemory",
            "hip error: out-of-memory",
            "NPU out of memory. Tried to allocate 512.00 MiB",
            "XPU out of memory",
            "RESOURCE_EXHAUSTED: XLA ran out of memory",
        ],
    )
    def test_an_unambiguous_exhaustion_is_recognized_in_any_spelling(self, message):
        assert is_device_oom(RuntimeError(message))

    @pytest.mark.parametrize(
        "message",
        [
            "hipErrorMemoryAllocation: failed to allocate device memory (hipError)",
            "synStatus 26: failed to allocate memory on device",
            "std::bad_alloc raised by the rmm pool",
        ],
    )
    def test_a_looser_phrasing_is_accepted_when_it_names_a_device_allocator(self, message):
        assert is_device_oom(RuntimeError(message))

    @pytest.mark.parametrize(
        "message",
        [
            "bad column name",
            "failed to allocate a socket",
            "failed to allocate a device handle",
            # The one that matters most: this is the shape a *host* allocation failure takes,
            # and accepting it would send the batch round the halving ladder — log2(rows)
            # re-executions of something that was never going to succeed — before reporting
            # the wrong diagnosis.
            "Failed to allocate memory for the output buffer",
            "cannot allocate memory for the arena",
        ],
    )
    def test_a_loose_phrasing_without_a_device_is_not_one(self, message):
        assert not is_device_oom(RuntimeError(message))

    def test_a_host_exhaustion_is_still_not_a_device_one(self):
        # They want opposite responses: halving a batch relieves the device and does nothing
        # for the host.
        assert not is_device_oom(MemoryError("Unable to allocate 1.2 GiB for an array"))

    @pytest.mark.parametrize("marker", ["npu", "synapse", "xpu", "hpu", "rmm"])
    def test_a_device_allocators_memory_error_is_one(self, marker):
        assert is_device_oom(MemoryError(f"std::bad_alloc: out_of_memory ({marker})"))
