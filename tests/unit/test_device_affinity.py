"""Binding a GPU worker's host half to the cores next to its device, and shared-device sizing.

The bug this guards against is a silent one: CUDA renumbers devices from zero inside every
worker, so a process handed the host's device 5 calls it device 0. Code that asks NVML about
"device 0" then gets a different board's NUMA node — and on a node where every device is the
same model, the wrong answer is indistinguishable from the right one until someone measures
host-to-device bandwidth. The other property under test is that binding refuses itself
wherever it would make things worse: an unreadable mapping, a tiny core set, or a platform
with no affinity control all leave the scheduler alone.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.hardware import nvml
from batcher._internal.hardware.devices import scope
from batcher._internal.hardware.fabric import device_links
from batcher.carbonite.accel import affinity

pytestmark = pytest.mark.unit


class _FakeNvml:
    def __init__(self, uuids):
        self._uuids = uuids

    def nvmlDeviceGetCount(self):
        return len(self._uuids)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return self._uuids[handle]

    def nvmlDeviceGetPciInfo(self, handle):
        return type("Info", (), {"busId": f"0000:{handle:02x}:00.0"})()


@pytest.fixture
def eight_devices(monkeypatch):
    """A host with eight devices, four on each NUMA node — the dense two-socket shape.

    The driver is faked at `hardware.devices.scope`, which is the single place that resolves
    the visibility environment. `device_links.visible_device_indices` is an alias for it: there
    used to be two implementations, and they disagreed about ROCm, about an empty value, and
    about MIG handles — so a worker's affinity binding was read off a different device set from
    its own memory pool. Faking the canonical source is what keeps these tests exercising the
    resolution the engine actually performs rather than a second copy of it.
    """
    uuids = [f"GPU-{i}" for i in range(8)]
    telemetry = tuple(nvml.DeviceTelemetry(index=i, uuid=uuid) for i, uuid in enumerate(uuids))
    monkeypatch.setattr(scope, "device_telemetry", lambda: telemetry)
    monkeypatch.setattr(device_links, "_nvml", lambda: _FakeNvml(uuids))
    monkeypatch.setattr(device_links, "_device_count", lambda nv: nv.nvmlDeviceGetCount())
    cpus = {i: tuple(range(0, 48)) if i < 4 else tuple(range(48, 96)) for i in range(8)}
    monkeypatch.setattr(
        device_links,
        "device_cpu_affinity",
        lambda address: cpus[int(address.split(":")[1], 16)],
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("BATCHER_MPS_CLIENTS", raising=False)
    return uuids


def test_the_two_visibility_resolvers_are_one(monkeypatch, eight_devices):
    """`device_links` and `hardware.devices` must never answer this differently.

    They did, in three ways that all landed on the fabric probes: `device_links` consulted
    `CUDA_VISIBLE_DEVICES` alone (so a ROCm node reported every device to the affinity path),
    read an empty value as "the whole node" rather than "no devices", and resolved no MIG
    handle. A worker therefore bound its host threads, chose its rail and read its PCIe link
    against one device set while its VRAM pool and OOM guard used another.
    """
    from batcher._internal.hardware.devices import visible_device_indices

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "6")
    assert device_links.visible_device_indices() == visible_device_indices() == (6,)

    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert device_links.visible_device_indices() == visible_device_indices() == ()


# --- Visible-device translation -----------------------------------------------------------


def test_an_unset_variable_means_every_device(eight_devices):
    assert device_links.visible_device_indices() == tuple(range(8))


def test_a_worker_given_one_device_sees_it_as_ordinal_zero(monkeypatch, eight_devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assert device_links.visible_device_indices() == (5,)
    # The whole point: ordinal 0 is host device 5, so its cores are the far socket's.
    assert affinity.feeder_cpus_for_device(0) == tuple(range(48, 96))
    assert affinity.device_affinity_summary(0)["device_index"] == 5


def test_devices_named_by_uuid_are_resolved(monkeypatch, eight_devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-6,GPU-1")
    assert device_links.visible_device_indices() == (6, 1)
    assert affinity.feeder_cpus_for_device(1) == tuple(range(0, 48))


def test_an_unresolvable_entry_truncates_the_list_as_cuda_does(monkeypatch, eight_devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,nonsense,2")
    assert device_links.visible_device_indices() == (1,)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,99,2")
    assert device_links.visible_device_indices() == (1,)


def test_an_ordinal_past_the_visible_set_binds_nothing(monkeypatch, eight_devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert affinity.feeder_cpus_for_device(1) == ()
    assert affinity.bind_host_threads_to_device(1) == ()


def test_no_driver_reports_no_visible_devices(monkeypatch):
    monkeypatch.setattr(device_links, "_nvml", lambda: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert device_links.visible_device_indices() == ()
    assert affinity.feeder_cpus_for_device(0) == ()


# --- Binding ------------------------------------------------------------------------------


def test_binding_applies_the_local_core_set(monkeypatch, eight_devices):
    applied = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: applied.append(set(cpus)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    assert affinity.bind_host_threads_to_device() == tuple(range(48, 96))
    assert applied == [set(range(48, 96))]


def test_binding_an_already_bound_worker_touches_nothing(monkeypatch, eight_devices):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(48, 96)))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: pytest.fail("rebound an already-bound worker")
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    assert affinity.bind_host_threads_to_device() == tuple(range(48, 96))


def test_a_core_set_too_small_to_decode_in_is_refused(monkeypatch, eight_devices):
    # Pinning a decode pipeline into two cores to save a memory hop trades a bandwidth
    # problem for a worse throughput one.
    monkeypatch.setattr(device_links, "device_cpu_affinity", lambda address: (0, 1))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: pytest.fail("bound into too few cores")
    )
    assert affinity.feeder_cpus_for_device(0) == ()
    assert affinity.bind_host_threads_to_device(0) == ()


def test_an_unreadable_mapping_leaves_the_scheduler_alone(monkeypatch, eight_devices):
    monkeypatch.setattr(device_links, "device_cpu_affinity", lambda address: ())
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: pytest.fail("bound on no information")
    )
    assert affinity.bind_host_threads_to_device(0) == ()


def test_a_refused_syscall_is_not_an_error(monkeypatch, eight_devices):
    def _deny(pid, cpus):
        raise OSError("operation not permitted")

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)))
    monkeypatch.setattr(os, "sched_setaffinity", _deny)
    assert affinity.bind_host_threads_to_device(0) == ()


# --- Shared devices -----------------------------------------------------------------------


def test_an_unshared_device_is_sized_whole(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    assert affinity.mps_active() is False
    assert affinity.mps_client_share() == 1.0


def test_a_shared_device_is_sized_by_its_tenancy(monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    assert affinity.mps_active() is True
    # Tenancy unpublished: the daemon knows and the client does not, so nothing is assumed.
    monkeypatch.delenv("BATCHER_MPS_CLIENTS", raising=False)
    assert affinity.mps_client_share() == 1.0
    monkeypatch.setenv("BATCHER_MPS_CLIENTS", "4")
    assert affinity.mps_client_share() == pytest.approx(0.25)
    monkeypatch.setenv("BATCHER_MPS_CLIENTS", "not-a-number")
    assert affinity.mps_client_share() == 1.0


# --- Sizing a shared device ---------------------------------------------------------------


def test_a_declared_share_bounds_what_the_vram_pool_will_plan_for():
    # The case measurement cannot cover: four co-tenants starting together, each seeing an
    # empty device, each sizing to all of it, all four failing at once.
    from batcher.carbonite.accel import VramPool

    gib = 1 << 30
    whole = VramPool(capacity_bytes=80 * gib, headroom=0.0)
    quarter = VramPool(capacity_bytes=80 * gib, headroom=0.0, share=0.25)
    assert whole.usable_bytes() == 80 * gib
    assert quarter.usable_bytes() == 20 * gib
    assert quarter.fits(19 * gib) is True
    assert quarter.fits(21 * gib) is False


def test_the_share_composes_with_headroom_and_another_tenant():
    from batcher.carbonite.accel import VramPool

    gib = 1 << 30
    pool = VramPool(capacity_bytes=80 * gib, headroom=0.5, share=0.5, external_bytes={0: 5 * gib})
    # Half the device, half of that held back, less what someone else already resident holds.
    assert pool.usable_bytes() == 15 * gib


def test_a_nonsensical_share_cannot_grow_the_device():
    from batcher.carbonite.accel import VramPool

    gib = 1 << 30
    assert VramPool(capacity_bytes=80 * gib, headroom=0.0, share=4.0).usable_bytes() == 80 * gib
    assert VramPool(capacity_bytes=80 * gib, headroom=0.0, share=-1.0).usable_bytes() == 0


def test_one_unreadable_address_does_not_renumber_the_other_devices(monkeypatch):
    # The alignment bug this guards: compacting the list would shift every device after the
    # unreadable one, so a caller indexing by device attributes one board's NUMA node, host
    # link, and degradation ratio to a different board — and on a node of identical devices
    # the wrong answer looks right.
    class _Refusing(_FakeNvml):
        def nvmlDeviceGetPciInfo(self, handle):
            if handle == 1:
                raise RuntimeError("not published")
            return super().nvmlDeviceGetPciInfo(handle)

    monkeypatch.setattr(device_links, "_nvml", lambda: _Refusing([f"GPU-{i}" for i in range(4)]))
    monkeypatch.setattr(device_links, "_device_count", lambda nv: 4)
    addresses = device_links.gpu_pci_addresses()
    assert len(addresses) == 4
    assert addresses[1] == ""
    assert addresses[2] == "0000:02:00.0"
    # And the device with no address binds nothing rather than borrowing a neighbour's cores.
    # The visible set comes from `hardware.devices`, the one resolver, so the driver has to be
    # faked there as well as at the PCI probe this test's first half exercises.
    monkeypatch.setattr(
        scope,
        "device_telemetry",
        lambda: tuple(nvml.DeviceTelemetry(index=i, uuid=f"GPU-{i}") for i in range(4)),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(device_links, "device_cpu_affinity", lambda a: tuple(range(16)))
    assert affinity.feeder_cpus_for_device(1) == ()
    assert affinity.feeder_cpus_for_device(2) == tuple(range(16))


def test_placement_skips_a_quarantined_device_however_empty_it_looks():
    # A board that has fallen off the bus reports all of its memory free, which makes it the
    # most attractive placement on the node and the only one guaranteed to fail.
    from batcher.carbonite.accel import VramPool

    gib = 1 << 30
    pool = VramPool(capacity_bytes=80 * gib, device_count=4, headroom=0.0)
    pool.reserve(70 * gib, device=0)
    pool.reserve(60 * gib, device=1)
    pool.reserve(50 * gib, device=2)
    assert pool.best_device() == 3
    assert pool.best_device(exclude=[3]) == 2
    assert pool.best_device(exclude=[2, 3]) == 1


def test_excluding_every_device_still_returns_a_placement():
    # Refusing to place work at all is worse than placing it on the least-bad device and
    # letting the reservation fail with a reason.
    from batcher.carbonite.accel import VramPool

    pool = VramPool(capacity_bytes=1 << 30, device_count=2, headroom=0.0)
    assert pool.best_device(exclude=[0, 1]) == 0


# --- Attributing device memory to the process that holds it --------------------------------


class _ProcNvml:
    """NVML with a per-process memory listing, which a shared device needs and a total hides."""

    def __init__(self, procs, refuse=False):
        self._procs, self._refuse = procs, refuse

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        if self._refuse:
            raise RuntimeError("not permitted in this namespace")
        return [type("P", (), {"pid": p, "usedGpuMemory": m})() for p, m in self._procs]


def test_this_process_share_is_measured_not_accounted(monkeypatch):
    import os

    from batcher._internal.hardware import nvml

    mine = os.getpid()
    monkeypatch.setattr(nvml, "_nvml", lambda: _ProcNvml([(mine, 10 << 30), (999, 30 << 30)]))
    assert nvml.device_processes(0) == ((mine, 10 << 30), (999, 30 << 30))
    assert nvml.own_device_memory(0) == 10 << 30


def test_an_unattributable_device_reports_none_not_zero(monkeypatch):
    # `0` would mean "this process holds nothing", which is a claim; `None` means "the driver
    # would not say", which is the truth inside a container that hides other namespaces.
    from batcher._internal.hardware import nvml

    monkeypatch.setattr(nvml, "_nvml", lambda: _ProcNvml([], refuse=True))
    assert nvml.own_device_memory(0) is None
    monkeypatch.setattr(nvml, "_nvml", lambda: None)
    assert nvml.own_device_memory(0) is None
    assert nvml.device_processes(0) == ()


def test_the_pool_reserves_against_the_neighbour_not_against_itself():
    # The pool admits; the framework allocates. Subtracting what was admitted leaves the
    # difference — the allocator's own pool, the CUDA context — charged to the co-tenant.
    from batcher.carbonite.accel import VramPool

    gib = 1 << 30
    pool = VramPool(capacity_bytes=80 * gib, headroom=0.0)
    pool.reserve(10 * gib)
    pool.observe_external(0, 50 * gib)
    assert pool.external_bytes[0] == 40 * gib  # accounting: total minus what we admitted
    pool.observe_external(0, 50 * gib, own_bytes=20 * gib)
    assert pool.external_bytes[0] == 30 * gib  # measured: total minus what we actually hold


def test_binding_invalidates_the_probes_that_report_the_mask(monkeypatch, eight_devices):
    # The hazard the ordering exists for: `available_cpu_count` is memoized, so a worker
    # bound to half a node's cores would keep sizing its pools to the whole node — the
    # oversubscription that probe exists to prevent, reintroduced by the call meant to place
    # the work well.
    reset_calls = []
    monkeypatch.setattr(
        "batcher._internal.hardware.reset_hardware_probes", lambda: reset_calls.append(1)
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    assert affinity.bind_host_threads_to_device() == tuple(range(48, 96))
    assert reset_calls == [1]


def test_a_worker_already_bound_does_not_invalidate_anything(monkeypatch, eight_devices):
    # The second task on a worker must not throw away probes that are still correct.
    monkeypatch.setattr(
        "batcher._internal.hardware.reset_hardware_probes",
        lambda: pytest.fail("re-probed without changing the mask"),
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(48, 96)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    assert affinity.bind_host_threads_to_device() == tuple(range(48, 96))
