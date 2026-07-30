"""The interconnect probes — RDMA ports, PCIe links, and NVLink state.

These read `/sys` and NVML, neither of which exists on the machine running the suite, so every
test here builds the tree or the driver it wants and asserts on what the probe makes of it.
Two properties matter more than the parsing: an unreadable fabric reports *nothing* rather
than a plausible default, because a fabricated NIC rate produces a confident transfer estimate
that is wrong by an order of magnitude; and a port that is cabled but not `ACTIVE` contributes
nothing, because counting it over-states the fabric by an integer factor.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.hardware import fabric
from batcher._internal.hardware.fabric import nvlink, pcie, rdma

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_fabric():
    """Re-probe around every test, so one test's fake tree never leaks into the next."""
    rdma.reset_fabric_probes()
    yield
    rdma.reset_fabric_probes()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _fake_ib(root: str, name: str, port: int, *, rate: str, state: str, layer: str) -> None:
    """Build one `/sys/class/infiniband` port under `root`."""
    device = os.path.join(root, name)
    _write(os.path.join(device, "node_guid"), "0011:2233:4455:6677\n")
    port_dir = os.path.join(device, "ports", str(port))
    _write(os.path.join(port_dir, "rate"), rate)
    _write(os.path.join(port_dir, "state"), state)
    _write(os.path.join(port_dir, "link_layer"), layer)
    _write(os.path.join(port_dir, "pkeys", "0"), "0xffff\n")


# --- RDMA ---------------------------------------------------------------------------------


def test_no_rdma_tree_reports_no_fabric(monkeypatch, tmp_path):
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", str(tmp_path / "absent"))
    assert rdma.rdma_devices() == ()
    assert rdma.rdma_available() is False
    assert rdma.fabric_bandwidth_gbps() == 0.0
    assert rdma.rdma_link_layers() == {}
    assert rdma.fabric_partition() == ""


def test_active_ports_sum_to_the_node_fabric_bandwidth(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    for i in range(4):
        _fake_ib(
            root, f"mlx5_{i}", 1, rate="400 Gb/sec (4X NDR)", state="4: ACTIVE", layer="InfiniBand"
        )
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    assert len(rdma.rdma_devices()) == 4
    assert rdma.rdma_available() is True
    assert rdma.fabric_bandwidth_gbps() == pytest.approx(1600.0)
    assert rdma.rdma_link_layers() == {"InfiniBand": 4}
    assert rdma.fabric_partition() == "0xffff"


def test_a_cabled_but_down_port_contributes_nothing(monkeypatch, tmp_path):
    # The failure this guards: counting ports rather than *active* ports over-states the
    # fabric by an integer factor, in the direction that turns a plan into a stall.
    root = str(tmp_path / "ib")
    _fake_ib(root, "mlx5_0", 1, rate="200 Gb/sec (4X HDR)", state="4: ACTIVE", layer="InfiniBand")
    _fake_ib(root, "mlx5_1", 1, rate="200 Gb/sec (4X HDR)", state="1: DOWN", layer="InfiniBand")
    _fake_ib(root, "mlx5_2", 1, rate="200 Gb/sec (4X HDR)", state="2: INIT", layer="InfiniBand")
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    assert len(rdma.rdma_devices()) == 3
    assert len(rdma.active_rdma_devices()) == 1
    assert rdma.fabric_bandwidth_gbps() == pytest.approx(200.0)


def test_roce_and_infiniband_are_counted_apart(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    _fake_ib(root, "mlx5_0", 1, rate="200 Gb/sec (4X HDR)", state="4: ACTIVE", layer="InfiniBand")
    _fake_ib(root, "mlx5_1", 1, rate="100 Gb/sec (4X EDR)", state="4: ACTIVE", layer="Ethernet")
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    assert rdma.rdma_link_layers() == {"InfiniBand": 1, "Ethernet": 1}
    assert [d.roce for d in rdma.active_rdma_devices()] == [False, True]


def test_an_unparseable_rate_reads_as_unknown_not_as_slow(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    _fake_ib(root, "mlx5_0", 1, rate="unknown\n", state="4: ACTIVE", layer="InfiniBand")
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    assert rdma.rdma_available() is True
    assert rdma.fabric_bandwidth_gbps() == 0.0


def test_a_node_spanning_two_partitions_reports_no_single_fabric(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    _fake_ib(root, "mlx5_0", 1, rate="200 Gb/sec", state="4: ACTIVE", layer="InfiniBand")
    _fake_ib(root, "mlx5_1", 1, rate="200 Gb/sec", state="4: ACTIVE", layer="InfiniBand")
    _write(os.path.join(root, "mlx5_1", "ports", "1", "pkeys", "0"), "0x7fff\n")
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    assert rdma.fabric_partition() == ""


def test_multi_port_devices_are_enumerated_per_port(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    for port in (1, 2):
        _fake_ib(root, "mlx5_0", port, rate="100 Gb/sec", state="4: ACTIVE", layer="InfiniBand")
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    assert [(d.name, d.port) for d in rdma.rdma_devices()] == [("mlx5_0", 1), ("mlx5_0", 2)]
    assert rdma.fabric_bandwidth_gbps() == pytest.approx(200.0)


def test_rdma_summary_is_flat_and_json_shaped(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    _fake_ib(root, "mlx5_0", 1, rate="400 Gb/sec", state="4: ACTIVE", layer="InfiniBand")
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    summary = rdma.rdma_summary()
    assert summary["rdma_available"] is True
    assert summary["active_ports"] == 1
    assert summary["bandwidth_gbps"] == pytest.approx(400.0)
    assert summary["devices"] == ["mlx5_0"]


# --- PCIe ---------------------------------------------------------------------------------


def _fake_pci(
    root: str,
    address: str,
    *,
    speed: str,
    width: str,
    max_speed: str,
    max_width: str,
    numa: str = "0",
) -> None:
    base = os.path.join(root, address)
    _write(os.path.join(base, "current_link_speed"), speed)
    _write(os.path.join(base, "current_link_width"), width)
    _write(os.path.join(base, "max_link_speed"), max_speed)
    _write(os.path.join(base, "max_link_width"), max_width)
    _write(os.path.join(base, "numa_node"), numa)


def test_pcie_bandwidth_uses_payload_rates_not_raw_transfer_rates():
    # Gen-4 x16 is ~252 Gb/s of payload, not 16 GT/s * 16.
    assert pcie.pcie_bandwidth_gbps(4, 16) == pytest.approx(252.06, rel=1e-3)
    assert pcie.pcie_bandwidth_gbps(5, 16) == pytest.approx(504.13, rel=1e-3)
    assert pcie.pcie_bandwidth_gbps(3, 0) == 0.0
    assert pcie.pcie_bandwidth_gbps(9, 16) == 0.0


def test_a_link_that_renegotiated_low_is_reported_degraded(monkeypatch, tmp_path):
    root = str(tmp_path / "pci")
    _fake_pci(
        root,
        "0000:0c:00.0",
        speed="8.0 GT/s PCIe",
        width="8",
        max_speed="32.0 GT/s PCIe",
        max_width="16",
    )
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", root)
    link = pcie.pcie_link("0000:0c:00.0")
    assert (link.gen, link.width, link.max_gen, link.max_width) == (3, 8, 5, 16)
    assert link.degraded is True
    # gen3 x8 against gen5 x16 is a quarter of the capable bandwidth.
    assert link.degradation_ratio == pytest.approx(0.125, rel=1e-2)
    assert pcie.degraded_pcie_links(("0000:0c:00.0",)) == (link,)


def test_a_link_at_full_capability_is_not_degraded(monkeypatch, tmp_path):
    root = str(tmp_path / "pci")
    _fake_pci(
        root,
        "0000:0c:00.0",
        speed="32.0 GT/s PCIe",
        width="16",
        max_speed="32.0 GT/s PCIe",
        max_width="16",
    )
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", root)
    link = pcie.pcie_link("0000:0c:00.0")
    assert link.degraded is False
    assert link.degradation_ratio == pytest.approx(1.0)
    assert pcie.degraded_pcie_links(("0000:0c:00.0",)) == ()


def test_an_unreadable_link_is_not_evidence_of_a_degraded_one(monkeypatch, tmp_path):
    # Reporting an unreadable link as degraded sends an operator to inspect healthy hardware.
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", str(tmp_path / "absent"))
    link = pcie.pcie_link("0000:0c:00.0")
    assert (link.gen, link.max_gen) == (0, 0)
    assert link.degraded is False
    assert link.bandwidth_gbps == 0.0


def test_a_malformed_address_degrades_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", str(tmp_path))
    assert pcie.pcie_link("not-an-address").address == "not-an-address"
    assert pcie.device_numa_node("not-an-address") == -1


def test_numa_node_minus_one_means_no_preference(monkeypatch, tmp_path):
    root = str(tmp_path / "pci")
    _fake_pci(root, "0000:0c:00.0", speed="", width="", max_speed="", max_width="", numa="-1")
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", root)
    assert pcie.device_numa_node("0000:0c:00.0") == -1


def test_pcie_class_orders_from_same_device_to_cross_socket(monkeypatch, tmp_path):
    root = tmp_path / "pci"
    tree = tmp_path / "devices"
    # Two devices under one switch, one under a second switch on the same root complex, and
    # one on a different root complex entirely.
    layout = {
        "0000:0c:00.0": tree / "pci0000:00" / "0000:00:01.0" / "0000:05:00.0" / "0000:0c:00.0",
        "0000:0d:00.0": tree / "pci0000:00" / "0000:00:01.0" / "0000:05:00.0" / "0000:0d:00.0",
        "0000:1a:00.0": tree / "pci0000:00" / "0000:00:01.0" / "0000:06:00.0" / "0000:1a:00.0",
        "0000:aa:00.0": tree / "pci0000:80" / "0000:80:01.0" / "0000:aa:00.0",
    }
    for address, target in layout.items():
        target.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        os.symlink(target, root / address)
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", str(root))
    assert pcie.pcie_class("0000:0c:00.0", "0000:0c:00.0") == "pix"
    assert pcie.pcie_class("0000:0c:00.0", "0000:0d:00.0") == "pix"
    assert pcie.pcie_class("0000:0c:00.0", "0000:1a:00.0") == "pxb"
    assert pcie.pcie_class("0000:0c:00.0", "0000:aa:00.0") == "sys"


def test_unresolvable_addresses_take_the_coarsest_class(monkeypatch, tmp_path):
    # Assuming proximity would route a "peer-to-peer" transfer through the host silently.
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", str(tmp_path / "absent"))
    assert pcie.pcie_class("0000:0c:00.0", "0000:aa:00.0") == "sys"
    assert pcie.PCIE_CLASSES[-1] == "sys"


# --- NVLink -------------------------------------------------------------------------------


class _FakeNvml:
    """The handful of NVML entry points the NVLink probe uses, over a scripted fabric."""

    def __init__(self, states: dict[int, list[int]], errors: int = 0):
        self._states = states
        self._errors = errors

    def nvmlDeviceGetCount(self):
        return len(self._states)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return f"GPU-{handle}"

    def nvmlDeviceGetNvLinkState(self, handle, link):
        states = self._states[handle]
        if link >= len(states):
            raise RuntimeError("invalid link")
        return states[link]

    def nvmlDeviceGetNvLinkErrorCounter(self, handle, link, counter):
        return self._errors

    def nvmlDeviceGetNvLinkRemotePciInfo(self, handle, link):
        raise RuntimeError("not published")


def _use_nvml(monkeypatch, fake) -> None:
    monkeypatch.setattr(nvlink, "_nvml", lambda: fake)
    monkeypatch.setattr(nvlink, "_device_count", lambda nv: nv.nvmlDeviceGetCount())


def test_nvlink_walk_stops_at_the_first_rejected_link(monkeypatch):
    _use_nvml(monkeypatch, _FakeNvml({0: [1] * 18, 1: [1] * 12}))
    status = nvlink.nvlink_status()
    assert [s.links for s in status] == [18, 12]
    assert [s.active_links for s in status] == [18, 12]
    assert all(not s.degraded for s in status)


def test_a_device_with_links_down_is_degraded(monkeypatch):
    _use_nvml(monkeypatch, _FakeNvml({0: [1, 1, 0, 0], 1: [1, 1, 1, 1]}))
    degraded = nvlink.nvlink_degraded_devices()
    assert [s.index for s in degraded] == [0]
    assert degraded[0].active_links == 2


def test_a_device_with_no_nvlink_is_not_degraded(monkeypatch):
    # Absence of a fabric is not a fault; flagging it would bury the devices that are broken.
    _use_nvml(monkeypatch, _FakeNvml({0: [], 1: []}))
    status = nvlink.nvlink_status()
    assert [s.links for s in status] == [0, 0]
    assert nvlink.nvlink_degraded_devices() == ()


def test_error_counters_sum_across_links_and_gate_on_a_threshold(monkeypatch):
    _use_nvml(monkeypatch, _FakeNvml({0: [1, 1]}, errors=5))
    status = nvlink.nvlink_status()
    # Four counters on two links, five each.
    assert status[0].errors == dict.fromkeys(nvlink.NVLINK_ERROR_COUNTERS, 10)
    assert status[0].total_errors == 40
    assert nvlink.nvlink_degraded_devices(status) == ()
    assert nvlink.nvlink_degraded_devices(status, error_threshold=10)[0].index == 0


def test_nvlink_summary_aggregates_the_node(monkeypatch):
    _use_nvml(monkeypatch, _FakeNvml({0: [1, 0], 1: [1, 1]}, errors=1))
    summary = nvlink.nvlink_summary()
    assert summary["devices"] == 2
    assert summary["links"] == 4
    assert summary["active_links"] == 3
    assert summary["degraded_devices"] == 1
    assert summary["errors"]["replay"] == 4


def test_no_driver_reports_nothing_rather_than_raising(monkeypatch):
    monkeypatch.setattr(nvlink, "_nvml", lambda: None)
    assert nvlink.nvlink_status() == ()
    assert nvlink.nvlink_degraded_devices() == ()
    assert nvlink.p2p_pairs() == ()
    assert nvlink.nvlink_summary()["devices"] == 0


def test_the_fabric_facade_exports_every_entry_point():
    for name in fabric.__all__:
        assert hasattr(fabric, name), name


def test_rdma_devices_expose_the_interface_a_socket_would_dial(monkeypatch, tmp_path):
    # An RDMA device (`mlx5_0`) and the interface an IP socket uses to reach the same wire
    # (`ib0`) are different names for one piece of hardware, and only the kernel relates
    # them. Without the join, a node's fast fabric is invisible to anything that dials.
    root = str(tmp_path / "ib")
    _fake_ib(root, "mlx5_0", 1, rate="400 Gb/sec", state="4: ACTIVE", layer="InfiniBand")
    _fake_ib(root, "mlx5_1", 1, rate="400 Gb/sec", state="1: DOWN", layer="InfiniBand")
    os.makedirs(os.path.join(root, "mlx5_0", "device", "net", "ib0"), exist_ok=True)
    os.makedirs(os.path.join(root, "mlx5_1", "device", "net", "ib1"), exist_ok=True)
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    # Only the active port's interface: a down port's address is one nothing answers on.
    assert rdma.rdma_net_interfaces() == ("ib0",)


def test_no_fabric_interface_means_no_address(monkeypatch, tmp_path):
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", str(tmp_path / "absent"))
    assert rdma.rdma_net_interfaces() == ()
    assert rdma.fabric_interface_address() == ""


def test_a_configured_but_down_interface_is_not_advertised(monkeypatch, tmp_path):
    # Advertising an address on a down interface is worse than advertising none: a peer
    # dialing it waits for a timeout instead of failing immediately.
    import socket

    root = str(tmp_path / "ib")
    _fake_ib(root, "mlx5_0", 1, rate="400 Gb/sec", state="4: ACTIVE", layer="InfiniBand")
    os.makedirs(os.path.join(root, "mlx5_0", "device", "net", "ib0"), exist_ok=True)
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)

    class _Addr:
        family = socket.AF_INET
        address = "10.10.0.5"

    class _Stats:
        def __init__(self, up):
            self.isup = up

    import psutil

    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {"ib0": [_Addr()]})
    monkeypatch.setattr(psutil, "net_if_stats", lambda: {"ib0": _Stats(False)})
    assert rdma.fabric_interface_address() == ""
    monkeypatch.setattr(psutil, "net_if_stats", lambda: {"ib0": _Stats(True)})
    assert rdma.fabric_interface_address() == "10.10.0.5"
