"""The measured host link, and whether a file's bytes can reach a device directly.

Both of these correct a *nameplate* assumption. The GPU cost model charged every host copy at
the link the device's datasheet lists, so a board that renegotiated to half width kept winning
decisions it now loses at twice the transfer cost; and a device-native read is only the fast
path when the bytes can actually get there without a bounce through host memory. The tests
below pin the direction each is allowed to be wrong in when it cannot see: unreadable links
report full efficiency (the assumption already in force), and an unrecognized filesystem
reports ineligible (a fallback to a path that works).
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric import device_links, pcie
from batcher.io.splits import gds
from batcher.kyber.gpu.energy import device_energy_advice

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    """Re-probe around every test; a name a test replaced has no cache to clear."""

    def _clear():
        gds.reset_gds_probe()
        clear = getattr(pcie.pcie_link, "cache_clear", None)
        if clear is not None:
            clear()

    _clear()
    yield
    _clear()


# --- The measured host link ---------------------------------------------------------------


class _FakeNvml:
    def __init__(self, addresses):
        self._addresses = addresses

    def nvmlDeviceGetCount(self):
        return len(self._addresses)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetPciInfo(self, handle):
        return type("Info", (), {"busId": self._addresses[handle]})()


def _use(monkeypatch, addresses, links):
    monkeypatch.setattr(device_links, "_nvml", lambda: _FakeNvml(addresses))
    monkeypatch.setattr(device_links, "_device_count", lambda nv: nv.nvmlDeviceGetCount())
    # Patched in the module that owns it: `degraded_device_links` delegates to the generic
    # `degraded_pcie_links`, which reads the probe from its own module rather than through
    # this one.
    monkeypatch.setattr(device_links, "pcie_link", lambda a: links[a])
    monkeypatch.setattr(pcie, "pcie_link", lambda a: links[a])


def test_no_driver_reports_full_efficiency_not_a_penalty(monkeypatch):
    monkeypatch.setattr(device_links, "_nvml", lambda: None)
    assert device_links.gpu_pci_addresses() == ()
    assert device_links.device_pcie_links() == ()
    assert device_links.degraded_device_links() == ()
    assert device_links.device_link_efficiency() == 1.0


def test_a_healthy_fleet_reports_full_efficiency(monkeypatch):
    full = pcie.PcieLink("0000:0c:00.0", gen=5, width=16, max_gen=5, max_width=16)
    _use(monkeypatch, ("0000:0C:00.0",), {"0000:0c:00.0": full})
    assert device_links.gpu_pci_addresses() == ("0000:0c:00.0",)
    assert device_links.device_link_efficiency() == pytest.approx(1.0)
    assert device_links.degraded_device_links() == ()


def test_the_worst_device_sets_the_efficiency_not_the_mean(monkeypatch):
    # A stage runs across the devices it is given and goes at the rate of the slowest link;
    # averaging a degraded device away produces an estimate no device on the node meets.
    good = pcie.PcieLink("0000:0c:00.0", gen=5, width=16, max_gen=5, max_width=16)
    bad = pcie.PcieLink("0000:1a:00.0", gen=4, width=8, max_gen=5, max_width=16)
    _use(
        monkeypatch,
        ("0000:0c:00.0", "0000:1a:00.0"),
        {"0000:0c:00.0": good, "0000:1a:00.0": bad},
    )
    assert device_links.device_link_efficiency() == pytest.approx(0.25)
    assert [link.address for link in device_links.degraded_device_links()] == ["0000:1a:00.0"]


def test_a_degraded_link_makes_the_host_copy_cost_more():
    # The whole point: the copy term is what decides whether a stage is worth a device, and a
    # link at a quarter of nameplate stretches it fourfold.
    kwargs = {"bytes_per_row": 4096.0, "flops_per_row": 8.0}
    full = device_energy_advice("NVIDIA_H100", **kwargs, link_efficiency=1.0)
    quarter = device_energy_advice("NVIDIA_H100", **kwargs, link_efficiency=0.25)
    assert quarter.speedup < full.speedup
    assert quarter.transfer_share > full.transfer_share


def test_a_resident_stage_pays_no_copy_however_bad_the_link():
    kwargs = {"bytes_per_row": 4096.0, "flops_per_row": 8.0, "resident": True}
    assert device_energy_advice("NVIDIA_H100", **kwargs, link_efficiency=0.25).transfer_share == 0.0


def test_a_nonsensical_efficiency_cannot_make_the_copy_free_or_infinite():
    kwargs = {"bytes_per_row": 4096.0, "flops_per_row": 8.0}
    for ratio in (0.0, -1.0, 5.0):
        advice = device_energy_advice("NVIDIA_H100", **kwargs, link_efficiency=ratio)
        assert advice.speedup >= 0.0
        assert 0.0 <= advice.transfer_share <= 1.0


# --- GPUDirect Storage eligibility --------------------------------------------------------


def test_a_remote_uri_is_never_storage_to_device(monkeypatch):
    monkeypatch.setattr(gds, "cufile_available", lambda: True)
    for uri in ("s3://bucket/a.parquet", "gs://b/a.parquet", "https://h/a.parquet"):
        verdict = gds.gds_eligible(uri)
        assert verdict.eligible is False
        assert verdict.reason == "remote"


def test_without_cufile_nothing_is_eligible(monkeypatch):
    monkeypatch.setattr(gds, "cufile_available", lambda: False)
    assert gds.gds_eligible("/data/a.parquet").reason == "no_cufile"


def test_a_supported_local_filesystem_is_eligible(monkeypatch):
    monkeypatch.setattr(gds, "cufile_available", lambda: True)
    monkeypatch.setattr(gds, "filesystem_type", lambda p: "ext4")
    verdict = gds.gds_eligible("/nvme/data/a.parquet")
    assert verdict.eligible is True
    assert verdict.filesystem == "ext4"
    assert verdict.reason == ""


def test_a_file_uri_is_treated_as_the_local_path_it_names(monkeypatch):
    monkeypatch.setattr(gds, "cufile_available", lambda: True)
    seen = []
    monkeypatch.setattr(gds, "filesystem_type", lambda p: seen.append(p) or "xfs")
    assert gds.gds_eligible("file:///nvme/a.parquet").eligible is True
    assert seen == ["/nvme/a.parquet"]


def test_the_container_overlay_and_tmpfs_are_declined_by_name(monkeypatch):
    monkeypatch.setattr(gds, "cufile_available", lambda: True)
    for kind in ("overlay", "tmpfs", "fuse"):
        monkeypatch.setattr(gds, "filesystem_type", lambda p, k=kind: k)
        verdict = gds.gds_eligible("/workdir/a.parquet")
        assert verdict.eligible is False
        assert (verdict.reason, verdict.filesystem) == ("filesystem", kind)


def test_an_unrecognized_filesystem_is_declined_not_assumed(monkeypatch):
    # Wrong this way costs a fallback to a path that works; wrong the other way costs a read
    # that silently bounces every byte through host memory.
    monkeypatch.setattr(gds, "cufile_available", lambda: True)
    monkeypatch.setattr(gds, "filesystem_type", lambda p: "somefs9000")
    assert gds.gds_eligible("/x/a.parquet").eligible is False


def test_an_unreadable_mount_table_reports_missing_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(gds, "cufile_available", lambda: True)
    monkeypatch.setattr(gds, "filesystem_type", lambda p: "")
    assert gds.gds_eligible("/x/a.parquet").reason == "missing"


def test_the_mount_lookup_takes_the_containing_mount_not_a_prefix(monkeypatch, tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/root / ext4 rw 0 0\noverlay /mnt overlay rw 0 0\n/dev/nvme0n1 /mnt/nvme xfs rw 0 0\n"
    )
    monkeypatch.setattr("builtins.open", _opener(str(mounts)))
    gds.reset_gds_probe()
    assert gds.filesystem_type("/mnt/nvme/data/a.parquet") == "xfs"
    assert gds.filesystem_type("/mnt/other/a.parquet") == "overlay"
    assert gds.filesystem_type("/etc/hosts") == "ext4"


def _opener(path: str):
    """An `open` that serves the fake mount table and defers everything else to the real one."""
    import builtins

    real = builtins.open

    def opener(target, *a, **k):
        return real(path if target == "/proc/self/mounts" else target, *a, **k)

    return opener


def test_the_summary_counts_the_reasons_a_read_was_not_direct(monkeypatch):
    monkeypatch.setattr(gds, "cufile_available", lambda: True)
    monkeypatch.setattr(gds, "filesystem_type", lambda p: "ext4" if "nvme" in p else "overlay")
    summary = gds.gds_summary(("/nvme/a.parquet", "/root/b.parquet", "s3://c/d.parquet"))
    assert summary["paths"] == 3
    assert summary["eligible"] == 1
    assert summary["reasons"] == {"filesystem": 1, "remote": 1}


def test_a_device_read_reports_the_transfer_path_it_took(monkeypatch):
    # The failure this makes visible: with cuFile absent the same read still works and still
    # returns the right rows, having bounced every byte through host memory.
    from batcher._internal import events
    from batcher.dist.gpu import device_read

    seen = []
    unsubscribe = events.subscribe(seen.append)
    try:
        monkeypatch.setattr(gds, "cufile_available", lambda: False)
        device_read._publish_transfer_path([gds_spec("/nvme/a.parquet")])
    finally:
        unsubscribe()
    assert [e.fields["event"] for e in seen] == ["transfer_path"]
    assert seen[0].fields["eligible"] == 0
    assert seen[0].fields["reasons"] == {"no_cufile": 1}


def test_nothing_is_probed_when_nobody_is_listening(monkeypatch):
    from batcher.dist.gpu import device_read

    monkeypatch.setattr(gds, "cufile_available", lambda: pytest.fail("probed with no subscriber"))
    device_read._publish_transfer_path([gds_spec("/nvme/a.parquet")])


def gds_spec(path: str):
    """A stand-in for `io.splits.device.DeviceReadSpec` carrying only what the probe reads."""
    from batcher.io.splits.device import DeviceReadSpec

    return DeviceReadSpec(path=path)


# --- Pairing a device with the NIC it should leave the node through -------------------------


def test_the_nearest_nic_is_the_one_on_the_same_root_complex(monkeypatch):
    # A transfer routed via a NIC on the other root complex crosses the inter-socket link
    # twice on its way *off* the node, which is the opposite of the point.
    from batcher._internal.hardware.fabric import device_links
    from batcher._internal.hardware.fabric.rdma import RdmaDevice

    monkeypatch.setattr(device_links, "visible_device_indices", lambda: (0,))
    monkeypatch.setattr(device_links, "gpu_pci_addresses", lambda: ("0000:0c:00.0",))
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma.active_rdma_devices",
        lambda: (
            RdmaDevice("mlx5_far", 1, state="ACTIVE", pci_address="0000:aa:00.0"),
            RdmaDevice("mlx5_near", 1, state="ACTIVE", pci_address="0000:0d:00.0"),
        ),
    )
    classes = {("0000:0c:00.0", "0000:0d:00.0"): "pix", ("0000:0c:00.0", "0000:aa:00.0"): "sys"}
    monkeypatch.setattr(device_links, "pcie_class", lambda a, b: classes.get((a, b), "sys"))
    assert device_links.nearest_rdma_device(0) == "mlx5_near"


def test_the_pairing_is_stable_when_two_nics_are_equally_close(monkeypatch):
    # Two workers that disagree about which NIC is nearest would each be right and would
    # still contend, so ties break on a name rather than on enumeration order.
    from batcher._internal.hardware.fabric import device_links
    from batcher._internal.hardware.fabric.rdma import RdmaDevice

    monkeypatch.setattr(device_links, "visible_device_indices", lambda: (0,))
    monkeypatch.setattr(device_links, "gpu_pci_addresses", lambda: ("0000:0c:00.0",))
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma.active_rdma_devices",
        lambda: (
            RdmaDevice("mlx5_9", 1, state="ACTIVE", pci_address="0000:0e:00.0"),
            RdmaDevice("mlx5_1", 1, state="ACTIVE", pci_address="0000:0d:00.0"),
        ),
    )
    monkeypatch.setattr(device_links, "pcie_class", lambda a, b: "pix")
    assert device_links.nearest_rdma_device(0) == "mlx5_1"


def test_no_fabric_or_no_address_pairs_with_nothing(monkeypatch):
    from batcher._internal.hardware.fabric import device_links

    monkeypatch.setattr(device_links, "visible_device_indices", lambda: (0,))
    monkeypatch.setattr(device_links, "gpu_pci_addresses", lambda: ("0000:0c:00.0",))
    monkeypatch.setattr("batcher._internal.hardware.fabric.rdma.active_rdma_devices", lambda: ())
    assert device_links.nearest_rdma_device(0) == ""
    # A device whose own address the driver would not publish pairs with nothing either.
    monkeypatch.setattr(device_links, "gpu_pci_addresses", lambda: ("",))
    assert device_links.nearest_rdma_device(0) == ""
    assert device_links.nearest_rdma_device(3) == ""


def test_the_degraded_link_helpers_are_one_implementation(monkeypatch):
    # Two functions answering "which of these links came up low" is the duplication the
    # subsystem rules forbid; the device-specific one delegates to the generic primitive.
    from batcher._internal.hardware.fabric import device_links, pcie

    bad = pcie.PcieLink("0000:1a:00.0", gen=3, width=8, max_gen=5, max_width=16)
    monkeypatch.setattr(device_links, "gpu_pci_addresses", lambda: ("0000:1a:00.0",))
    monkeypatch.setattr(pcie, "pcie_link", lambda a: bad)
    assert [link.address for link in device_links.degraded_device_links()] == ["0000:1a:00.0"]
