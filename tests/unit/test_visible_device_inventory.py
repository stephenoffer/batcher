"""The device inventory must report what this process may use, not what the node has.

NVML and the AMD sysfs tree enumerate every accelerator physically present and neither honors
the visibility variable a runtime uses to hand out a subset. Ray sets that variable on every
task holding a `num_gpus` grant, so on a multi-device node the normal case was the broken one:
measured on a four-T4 node, an actor granted one GPU saw `CUDA_VISIBLE_DEVICES=0` and
`torch.cuda.device_count() == 1` while `gpu_inventory()` reported four devices and 60 GiB --
for a process entitled to one device and 15.

The module's own torch fallback already returned the visible set, so the two backends
disagreed with each other and with the docstring's word "visible", depending only on which one
answered first.
"""

from __future__ import annotations

import pytest

from batcher._internal.accelerators import _visible_devices

pytestmark = pytest.mark.unit

_GIB = 1 << 30


def _devices(n: int) -> list[dict[str, object]]:
    """`n` physically-probed T4s, as the NVML path reports them."""
    return [{"index": i, "name": "Tesla T4", "memory_bytes": 15 * _GIB} for i in range(n)]


def test_an_unset_variable_leaves_every_device_visible(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    assert _visible_devices(_devices(4)) == _devices(4)


def test_one_granted_device_reports_one(monkeypatch) -> None:
    """The measured case: a Ray actor with `num_gpus=1` on a four-device node."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    visible = _visible_devices(_devices(4))
    assert len(visible) == 1
    assert sum(d["memory_bytes"] for d in visible) == 15 * _GIB


def test_two_granted_devices_report_two(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    visible = _visible_devices(_devices(4))
    assert [d["index"] for d in visible] == [0, 1]
    assert sum(d["memory_bytes"] for d in visible) == 30 * _GIB


def test_indices_are_renumbered_from_zero(monkeypatch) -> None:
    """CUDA renumbers the visible set, so index `i` here must be torch's device `i`."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert [d["index"] for d in _visible_devices(_devices(4))] == [0, 1]


def test_the_listed_order_is_preserved(monkeypatch) -> None:
    """`3,1` means device 0 is physical 3 — the runtime's order, not the driver's."""
    devices = [{"index": i, "name": f"d{i}", "memory_bytes": _GIB} for i in range(4)]
    visible = _visible_devices(devices)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    visible = _visible_devices(devices)
    assert [d["name"] for d in visible] == ["d3", "d1"]
    assert [d["index"] for d in visible] == [0, 1]


def test_an_explicitly_empty_variable_hides_everything(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _visible_devices(_devices(4)) == []


def test_an_out_of_range_index_truncates(monkeypatch) -> None:
    """CUDA stops enumerating at the first entry it cannot resolve."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,9")
    assert len(_visible_devices(_devices(4))) == 1


def test_a_uuid_form_is_left_alone(monkeypatch) -> None:
    """Mapping a UUID back to a slot needs a lookup the probe did not record, so don't guess."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fe4c4a1b-2c3d-4e5f-8a9b-0c1d2e3f4a5b")
    assert _visible_devices(_devices(4)) == _devices(4)


def test_a_mig_id_is_left_alone(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-fe4c4a1b-2c3d-4e5f-8a9b-0c1d2e3f4a5b")
    assert _visible_devices(_devices(4)) == _devices(4)


def test_the_amd_variables_are_honored_too(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert len(_visible_devices(_devices(4))) == 1


def test_filtering_does_not_mutate_the_probe_result(monkeypatch) -> None:
    """The probe is memoized, so a caller must never see a neighbour's renumbering."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    probed = _devices(4)
    _visible_devices(probed)
    assert [d["index"] for d in probed] == [0, 1, 2, 3]


def test_no_devices_stays_no_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert _visible_devices([]) == []


# --- The documented "no devices" spellings ----------------------------------------------------


@pytest.mark.parametrize("masked", ["-1", "", "none", "void"])
def test_a_masked_off_process_sees_no_devices(monkeypatch, masked):
    """`-1` is CUDA's own documented "no devices" and what a framework or CI script writes.

    Only the empty string was recognized, so `-1` fell through the ordinal parse (it is not
    `isdigit`), failed to resolve as a UUID, and hit the "could not resolve" fallback — which
    returns EVERY device on the node. A pod explicitly denied the GPU was reported as owning
    all eight of them, and the pool sized itself accordingly.
    """
    from batcher._internal import accelerators as acc

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", masked)
    devices = [{"index": i, "name": "NVIDIA H100", "memory_bytes": 80 << 30} for i in range(8)]
    assert acc._visible_devices(devices) == []


@pytest.mark.parametrize("var", ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"])
def test_the_cheap_negative_honours_a_mask(monkeypatch, var):
    """A run told to stay off the GPU must not pay the ~2 s `import torch` to discover that."""
    from batcher._internal import accelerators as acc

    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(var, "-1" if var == "CUDA_VISIBLE_DEVICES" else "none")
    acc.gpu_devices_absent.cache_clear()
    assert acc.gpu_devices_absent() is True
    acc.gpu_devices_absent.cache_clear()


def test_an_ordinary_pin_is_unaffected(monkeypatch):
    from batcher._internal import accelerators as acc

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    devices = [{"index": i, "name": "NVIDIA H100", "memory_bytes": 80 << 30} for i in range(8)]
    assert [d["index"] for d in acc._visible_devices(devices)] == [0, 1]
