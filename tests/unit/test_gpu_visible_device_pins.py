"""How a device pin is spelled, and what the inventory does with each spelling.

A scheduler pins a process's devices through `CUDA_VISIBLE_DEVICES`, and it does not always
write ordinals. Ray writes indices; the Kubernetes device plugin writes **UUIDs**; MIG writes
partition handles. Index-only parsing treats the last two as unreadable, and the fallback for
unreadable is "every device on the node" — so on exactly the fleets that pin hardest, a pod
granted one board had its pool sized, its health judged, and its report written for eight.

The fallback itself is right and is asserted here too: resolving a UUID needs the driver, an
AMD host has no NVML to ask, and hiding every device on an Instinct node would be a worse
error than reporting too many.
"""

from __future__ import annotations

import pytest

from batcher._internal import accelerators

pytestmark = pytest.mark.unit


@pytest.fixture
def devices():
    """Four probed devices, as NVML or the AMD sysfs tree would report them."""
    return [{"index": i, "name": f"gpu{i}", "memory_bytes": (i + 1) << 30} for i in range(4)]


@pytest.fixture(autouse=True)
def _unpinned(monkeypatch):
    """Start every case from an unpinned process, whatever the host actually sets."""
    for var in accelerators._VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)


def test_an_unpinned_process_sees_every_device(devices):
    """The pre-existing answer, and the one a driver or a monitor needs."""
    assert accelerators._visible_devices(devices) == devices


def test_ordinals_select_and_renumber_from_zero(monkeypatch, devices):
    """CUDA renumbers the visible set from zero, so `gpu_inventory()[0]` must be the device the
    framework calls zero rather than a physical slot the process cannot address."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    picked = accelerators._visible_devices(devices)
    assert [d["name"] for d in picked] == ["gpu2", "gpu3"]
    assert [d["index"] for d in picked] == [0, 1]


def test_an_explicitly_empty_pin_hides_everything(monkeypatch, devices):
    """`CUDA_VISIBLE_DEVICES=""` is how a scheduler says "no devices", and is distinct from
    unset."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert accelerators._visible_devices(devices) == []


def test_a_uuid_pin_is_resolved_against_the_driver(monkeypatch, devices):
    """The Kubernetes device-plugin case, which index-only parsing could not read at all."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-4f2a0000-1111-2222-3333-444455556666")
    monkeypatch.setattr(
        "batcher._internal.hardware.devices.scope.visible_device_indices", lambda: (2,)
    )
    picked = accelerators._visible_devices(devices)
    assert [d["name"] for d in picked] == ["gpu2"]
    assert picked[0]["index"] == 0


def test_a_mig_handle_resolves_to_its_parent_board(monkeypatch, devices):
    """A MIG partition draws its memory from the parent device, which is the board whose
    capacity and residency bound the process."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-GPU-4f2a0000-1111-2222-3333-444455556666/1/0")
    monkeypatch.setattr(
        "batcher._internal.hardware.devices.scope.visible_device_indices", lambda: (1,)
    )
    assert [d["name"] for d in accelerators._visible_devices(devices)] == ["gpu1"]


def test_an_unresolvable_uuid_still_reports_every_device(monkeypatch, devices):
    """An AMD host has no NVML to resolve a UUID against. Reporting too many devices is what
    every caller already handles; reporting none would make an Instinct node look CPU-only."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-unknown")
    monkeypatch.setattr(
        "batcher._internal.hardware.devices.scope.visible_device_indices", lambda: ()
    )
    assert accelerators._visible_devices(devices) == devices


def test_a_driver_scope_covering_the_whole_node_carries_no_information(monkeypatch, devices):
    """`visible_device_indices` returns every device both when a process really is unpinned and
    when it could not resolve the pin. A scope as wide as the node cannot distinguish those, so
    it is read as "could not resolve" rather than as a selection."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-ambiguous")
    monkeypatch.setattr(
        "batcher._internal.hardware.devices.scope.visible_device_indices", lambda: (0, 1, 2, 3)
    )
    assert accelerators._visible_devices(devices) == devices


def test_the_driver_lookup_is_not_reached_for_an_ordinal_pin(monkeypatch, devices):
    """The common pin — every Ray worker — must not pay an NVML handshake to be parsed."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    def _boom():
        raise AssertionError("the driver must not be consulted for an ordinal pin")

    monkeypatch.setattr("batcher._internal.hardware.devices.scope.visible_device_indices", _boom)
    assert [d["name"] for d in accelerators._visible_devices(devices)] == ["gpu1"]


@pytest.mark.parametrize("masked", ["-1", "none", "void"])
def test_an_explicit_no_device_pin_does_not_hand_over_the_whole_node(masked):
    """`-1` is CUDA's own documented "no devices", and what a framework or CI script writes.

    `_resolve` truncates at the first entry naming no live device and falls back to the whole
    host when *nothing* resolved — the right reading of a value it does not understand. `-1` is
    not a digit, not a UUID and not a MIG handle, so it resolved to nothing and took that
    fallback: a process explicitly denied every device was told it owned all four, and the pool
    sized itself, attributed telemetry and picked a NUMA affinity accordingly.

    Note the truncation rule already handled `-1` correctly *anywhere but first* — only the
    leading position hit the fallback, which is the position anything disabling the GPU writes.
    """
    from batcher._internal.hardware.devices import scope

    telemetry = [_Telemetry(index=i, uuid=f"GPU-{i}") for i in range(4)]
    assert scope._resolve(masked, telemetry) == ()


def test_a_trailing_no_device_entry_still_truncates(devices):
    """`"0,-1,1"` exposes exactly one device, as the CUDA runtime does — the pre-existing
    behavior, which the leading-position fix must not disturb."""
    from batcher._internal.hardware.devices import scope

    telemetry = [_Telemetry(index=i, uuid=f"GPU-{i}") for i in range(4)]
    assert scope._resolve("0,-1,1", telemetry) == (0,)


class _Telemetry:
    """The two fields `_resolve` reads off a probed device."""

    def __init__(self, index: int, uuid: str) -> None:
        self.index = index
        self.uuid = uuid
