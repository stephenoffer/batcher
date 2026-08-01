"""Device identity on a real cluster: PID namespaces, device ordering, and packed tenancy.

Every defect covered here is invisible on the machine the GPU path was developed on and
certain on the machines it is meant to run on. They share one shape — a fact that is true of a
single GPU in a single process on a bare-metal host, and false of a fractional GPU in a
container on a mixed node — so none of them can be caught by a test that has a device and
nothing else.

* **PID namespaces.** NVML reports the host's PID; a containerized worker knows itself by the
  namespace's. Nothing matched, and every per-process attribution silently answered the most
  dangerous value it could: zero memory held, zero utilization used, every device contended.
* **Device ordering.** NVML enumerates by PCI bus, CUDA by capability. On identical boards the
  two agree; on a mixed node every ordinal-to-index translation names a different device.
* **Packed tenancy.** The fan-out deliberately packs several shards per device, and each of
  them sized its memory pool for the whole board.

The fakes below are deliberately shaped like the real failures rather than like the APIs:
a process list with no matching PID *is* the container, and that is what is asserted on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from batcher._internal.device_share import MAX_COTENANTS
from batcher._internal.hardware import nvml
from batcher._internal.hardware.devices import (
    DEVICE_ORDER_ENV,
    PCI_BUS_ORDER,
    current_ordinal,
    device_order_env,
)
from batcher._internal.hardware.telemetry import processes as procmod

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Sample:
    """The shape `device_process_utilization` returns, without needing a driver."""

    index: int
    pid: int
    sm: float = 0.5
    memory: float = 0.0
    encoder: float = 0.0
    decoder: float = 0.0
    timestamp_us: int = 1

    @property
    def active(self) -> bool:
        return self.sm > 0


def test_the_host_pid_is_read_from_sched_not_from_status():
    """`/proc/self/sched` carries the global PID; `status` carries the namespace-local one.

    Outside a container the two agree, which is exactly why reading the wrong one was never
    noticed. The assertion that matters is only that a number comes back at all on Linux, since
    this test cannot itself be inside a foreign namespace.
    """
    nvml.reset_nvml_probe()
    assert nvml.host_pid() == os.getpid()


def test_own_process_ids_always_offers_the_local_pid():
    """Both candidates are tried, and the local one is never dropped.

    A pod with `hostPID: true`, and every bare-metal worker, is listed by NVML under exactly the
    PID `os.getpid()` reports. Resolving a host PID must not cost those deployments their match.
    """
    nvml.reset_nvml_probe()
    assert os.getpid() in nvml.own_process_ids()


def test_unmatched_processes_report_unattributable_rather_than_zero(monkeypatch):
    """The container case: NVML lists processes and none of them is this one.

    `0` was the old answer and is the harmful one — the VRAM pool subtracts it, concludes the
    whole device belongs to a neighbour, and plans no allocator at all, so the worker spends the
    query on the synchronizing driver allocator the pool exists to replace. `None` says "cannot
    attribute", which every caller already handles by keeping its own accounting.
    """
    monkeypatch.setattr(nvml, "device_processes", lambda index: ((999_001, 8 << 30),))
    monkeypatch.setattr(nvml, "own_process_ids", lambda: (4242,))
    assert nvml.own_device_memory(0) is None


def test_a_matching_host_pid_is_summed(monkeypatch):
    """With the namespaces resolved, attribution works and `0` becomes an honest answer."""
    monkeypatch.setattr(
        nvml, "device_processes", lambda index: ((999_001, 8 << 30), (4242, 2 << 30))
    )
    monkeypatch.setattr(nvml, "own_process_ids", lambda: (4242,))
    assert nvml.own_device_memory(0) == 2 << 30


def test_an_idle_device_with_no_processes_stays_unknown(monkeypatch):
    """No process list at all is unknowable, as it always was — not zero."""
    monkeypatch.setattr(nvml, "device_processes", lambda index: ())
    assert nvml.own_device_memory(0) is None


def test_an_unrecognized_pid_set_does_not_report_contention(monkeypatch):
    """Every active sample belonging to a stranger is a namespace mismatch, not a busy device.

    Reporting True here tells autobatching that the device-wide utilization reading is a sum it
    must not trust — permanently, on every containerized worker in the fleet, which is all of
    them. That is the reading that quietly disables the control loop.
    """
    monkeypatch.setattr(
        procmod, "device_process_utilization", lambda since_us=0: (_Sample(0, 999_001),)
    )
    monkeypatch.setattr(procmod, "own_process_ids", lambda: (4242,))
    assert procmod.device_shared_with_others(0) is None


def test_a_device_held_only_by_this_process_reads_as_exclusive(monkeypatch):
    """The answer the control loop actually wants, and could not previously get."""
    monkeypatch.setattr(
        procmod, "device_process_utilization", lambda since_us=0: (_Sample(0, 4242),)
    )
    monkeypatch.setattr(procmod, "own_process_ids", lambda: (4242,))
    assert procmod.device_shared_with_others(0) is False


def test_a_real_neighbour_is_still_reported(monkeypatch):
    """Suppressing the false positive must not suppress the true one."""
    monkeypatch.setattr(
        procmod,
        "device_process_utilization",
        lambda since_us=0: (_Sample(0, 4242), _Sample(0, 999_001)),
    )
    monkeypatch.setattr(procmod, "own_process_ids", lambda: (4242,))
    assert procmod.device_shared_with_others(0) is True


def test_own_utilization_matches_the_host_pid(monkeypatch):
    """The same correction on the utilization half."""
    monkeypatch.setattr(
        procmod, "device_process_utilization", lambda since_us=0: (_Sample(0, 4242, sm=0.7),)
    )
    monkeypatch.setattr(procmod, "own_process_ids", lambda: (4242,))
    assert procmod.own_utilization(0) == pytest.approx(0.7)


def test_a_worker_is_pinned_to_pci_bus_device_order(monkeypatch):
    """Without this the ordinal a worker computes on and the NVML index it reads telemetry
    from are different boards on any node whose GPUs are not identical."""
    monkeypatch.delenv(DEVICE_ORDER_ENV, raising=False)
    assert device_order_env() == {DEVICE_ORDER_ENV: PCI_BUS_ORDER}


def test_an_operators_own_device_order_is_never_overwritten():
    """A deployment that pinned the ordering has a reason no probe can see."""
    assert device_order_env({DEVICE_ORDER_ENV: "FASTEST_FIRST"}) == {}


def test_the_gpu_task_environment_carries_the_device_order(monkeypatch):
    """The variable has to reach the worker, and it has to reach it before CUDA initializes —
    which means in the `runtime_env`, not in the task body."""
    monkeypatch.delenv(DEVICE_ORDER_ENV, raising=False)
    from batcher.dist.gpu.tasks import gpu_task_runtime_env

    env = (gpu_task_runtime_env() or {}).get("env_vars", {})
    assert env.get(DEVICE_ORDER_ENV) == PCI_BUS_ORDER


def test_the_current_ordinal_declines_without_a_loaded_framework():
    """Never imports one to answer: that would cost seconds and create a CUDA context as a
    side effect, on a path that is deciding where to put work."""
    assert current_ordinal() is None


def test_task_tenancy_is_one_when_ray_is_not_scheduling():
    """Every unknown resolves to the unpacked behavior this path had before."""
    from batcher.dist.gpu.resources import task_device_tenants

    assert task_device_tenants() == 1


@pytest.mark.parametrize(
    ("granted", "expected"),
    [(1.0, 1), (0.5, 2), (0.25, 4), (0.33, 3), (0.0, 1), (0.001, MAX_COTENANTS)],
)
def test_a_fractional_grant_becomes_a_co_tenant_count(monkeypatch, granted, expected):
    """Ray reports the fraction back as a float, so a third arrives as 0.33 and must round to
    three rather than ceiling to four. The last case is the co-tenancy ceiling, which stops a
    mis-derived share from dividing the pool into nothing."""
    from batcher.dist.gpu import resources

    class _Ctx:
        @staticmethod
        def get_assigned_resources():
            return {"GPU": granted}

    fake = type(
        "_Ray",
        (),
        {"is_initialized": staticmethod(lambda: True), "get_runtime_context": staticmethod(_Ctx)},
    )
    monkeypatch.setitem(__import__("sys").modules, "ray", fake)
    assert resources.task_device_tenants() == expected


def test_a_packed_device_plans_a_proportionally_smaller_pool(monkeypatch):
    """The defect this exists for: four shard tasks on one board each sized a pool for the
    whole of it, and at the default `pool_initial_fraction` of 0.5 they reserved 200% of the
    device between them.

    Faked down to an 80 GiB device with no telemetry, so the assertion is about the division
    and not about whatever hardware the test happens to run on.
    """
    from batcher._internal import accelerators
    from batcher.carbonite.accel import allocator

    monkeypatch.setattr(
        accelerators,
        "gpu_inventory",
        lambda: [{"index": 0, "name": "H100", "memory_bytes": 80 << 30}],
    )
    monkeypatch.setattr(nvml, "device_telemetry", lambda: ())

    alone = allocator._visible_device_usable_bytes(0.15, 1)
    packed = allocator._visible_device_usable_bytes(0.15, 4)
    assert alone == pytest.approx((80 << 30) * 0.85, rel=0.01)
    assert packed * 4 == pytest.approx(alone, rel=0.01)


def test_routing_cupy_and_numba_through_rmm_imports_neither_of_them(monkeypatch):
    """A relational worker with neither library loaded must be left exactly as it was — and
    must not pay an import to find out.

    `sys.modules` is faked empty of all three rather than merely inspected, so the assertion
    holds whatever else the suite has already imported into this interpreter.
    """
    import sys

    from batcher.carbonite.accel.device import adopt_rmm_everywhere

    for name in ("cupy", "numba", "rmm"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    adopt_rmm_everywhere()  # must not raise
    assert not {"cupy", "numba", "rmm"} & set(sys.modules)
