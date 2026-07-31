"""How far apart two devices sit on the bus, and the collective that pays for it.

On an NVLink node the fabric hides the bus topology. On the PCIe-only nodes that make up most
rented GPU capacity it is the whole story: two devices under one switch exchange peer-to-peer
without the transfer reaching the CPU, and two under different root complexes cross the
inter-socket link, which is slower and contended with every other socket-crossing access on the
machine. Nothing measured that, so a tensor-parallel group was placed blind.

These build a fake `/sys/bus/pci` tree, since the real one here has no accelerators in it.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric import device_links, pcie

pytestmark = pytest.mark.unit


def _pci_tree(root, layout):
    """Write a PCI tree from `{address: parent_path_components}`.

    `pcie_class` reads the chain of bridges above a device, so the *shape* of the tree is what
    decides the answer, not any attribute on the leaf.
    """
    for address, parents in layout.items():
        path = root
        for component in parents:
            path = path / component
            path.mkdir(exist_ok=True)
        device = path / address
        device.mkdir()
        (device / "numa_node").write_text("0\n")
        (root / address).symlink_to(device)


@pytest.fixture
def topology(tmp_path, monkeypatch):
    """A fake PCI tree, with NVML absent so the AMD/no-vendor paths are not consulted."""
    root = tmp_path / "pci"
    root.mkdir()
    monkeypatch.setattr(pcie, "PCIE_SYSFS_ROOT", str(root))
    monkeypatch.setattr(pcie.pcie_link, "cache_clear", lambda: None, raising=False)
    return root


def _addresses(monkeypatch, *addresses):
    monkeypatch.setattr(device_links, "gpu_pci_addresses", lambda: tuple(addresses))


def test_no_devices_is_no_opinion(monkeypatch):
    # Empty must read as "no opinion" rather than as a refusal: a caller falls back to the
    # choice it already had, which is what keeps this from changing behavior where it cannot
    # see anything.
    _addresses(monkeypatch)
    assert device_links.device_topology() == ()
    assert device_links.tightest_device_group(4) == ()
    assert device_links.group_topology_class((0, 1)) == ""


def test_the_diagonal_is_the_closest_class(monkeypatch, topology):
    _pci_tree(topology, {"0000:01:00.0": ("pci0000:00", "0000:00:01.0")})
    _addresses(monkeypatch, "0000:01:00.0")
    assert device_links.device_topology() == (("pix",),)


def test_the_matrix_is_square_and_symmetric(monkeypatch, topology):
    _pci_tree(
        topology,
        {
            "0000:01:00.0": ("pci0000:00", "0000:00:01.0", "0000:01:00.1"),
            "0000:41:00.0": ("pci0000:40", "0000:40:01.0", "0000:41:00.1"),
        },
    )
    _addresses(monkeypatch, "0000:01:00.0", "0000:41:00.0")
    matrix = device_links.device_topology()
    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
    assert matrix[0][1] == matrix[1][0], "distance is not directional"


def test_an_unknown_address_is_assumed_far_not_near(monkeypatch, topology):
    # The safe direction. An unknown distance treated as a short one would place a collective
    # across a socket and report that it had placed it under a switch.
    _pci_tree(topology, {"0000:01:00.0": ("pci0000:00",)})
    _addresses(monkeypatch, "0000:01:00.0", "")
    matrix = device_links.device_topology()
    assert matrix[0][1] == "sys"
    assert matrix[1][1] == "pix", "a device is still as close to itself as anything can be"


def test_a_group_prefers_the_devices_under_one_switch(monkeypatch):
    # Four devices, two per switch. A two-device group must take a pair from one switch, not
    # one from each — which is the choice a naive "first N" makes half the time.
    monkeypatch.setattr(
        device_links,
        "device_topology",
        lambda: (
            ("pix", "pix", "sys", "sys"),
            ("pix", "pix", "sys", "sys"),
            ("sys", "sys", "pix", "pix"),
            ("sys", "sys", "pix", "pix"),
        ),
    )
    assert device_links.tightest_device_group(2) == (0, 1)
    assert device_links.group_topology_class((0, 1)) == "pix"
    assert device_links.group_topology_class((1, 2)) == "sys"


def test_a_group_that_cannot_fit_reports_the_boundary_it_crosses(monkeypatch):
    # Asking for three from two switches of two: the group necessarily spans them, and the
    # honest answer is the worst pair rather than the average.
    monkeypatch.setattr(
        device_links,
        "device_topology",
        lambda: (
            ("pix", "pix", "sys", "sys"),
            ("pix", "pix", "sys", "sys"),
            ("sys", "sys", "pix", "pix"),
            ("sys", "sys", "pix", "pix"),
        ),
    )
    group = device_links.tightest_device_group(3)
    assert len(group) == 3
    assert device_links.group_topology_class(group) == "sys"


def test_a_group_is_bounded_by_its_worst_pair_not_its_average(monkeypatch):
    # Seven devices under one switch plus one across the socket is a socket-crossing group: a
    # collective runs at the rate of its slowest edge.
    monkeypatch.setattr(
        device_links,
        "device_topology",
        lambda: (("pix", "pix", "phb"), ("pix", "pix", "phb"), ("phb", "phb", "pix")),
    )
    assert device_links.group_topology_class((0, 1, 2)) == "phb"
    assert device_links.group_topology_class((0, 1)) == "pix"


def test_the_choice_is_the_same_on_every_run(monkeypatch):
    # A group that changes between two identical processes is not a placement, it is a coin
    # toss, and it makes a run's throughput unreproducible.
    matrix = (
        ("pix", "pix", "pix", "pix"),
        ("pix", "pix", "pix", "pix"),
        ("pix", "pix", "pix", "pix"),
        ("pix", "pix", "pix", "pix"),
    )
    monkeypatch.setattr(device_links, "device_topology", lambda: matrix)
    assert {device_links.tightest_device_group(2) for _ in range(20)} == {(0, 1)}


def test_asking_for_more_devices_than_exist_gets_no_opinion(monkeypatch):
    monkeypatch.setattr(device_links, "device_topology", lambda: (("pix",),))
    assert device_links.tightest_device_group(4) == ()
    assert device_links.tightest_device_group(0) == ()


def test_a_group_of_one_has_no_pair_to_compare(monkeypatch):
    monkeypatch.setattr(device_links, "device_topology", lambda: (("pix",),))
    assert device_links.group_topology_class((0,)) == ""


def test_an_index_outside_the_matrix_is_unknown_rather_than_a_crash(monkeypatch):
    monkeypatch.setattr(device_links, "device_topology", lambda: (("pix",),))
    assert device_links.group_topology_class((0, 9)) == ""


def test_this_host_answers_without_raising():
    assert isinstance(device_links.device_topology(), tuple)
    assert isinstance(device_links.tightest_device_group(2), tuple)


# --- Reaching the tensor-parallel advice ----------------------------------------------------


def test_a_tight_group_adds_nothing_to_the_advice(monkeypatch):
    from batcher.ml.llm.engines import parallelism

    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.group_topology_class", lambda group: "pxb"
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.tightest_device_group", lambda size: (0, 1)
    )
    assert parallelism.group_spread(2) == "", "below one host bridge is peer-to-peer already"


def test_a_socket_crossing_group_is_worth_saying(monkeypatch):
    from batcher.ml.llm.engines import parallelism

    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.group_topology_class", lambda group: "sys"
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.tightest_device_group", lambda size: (0, 3)
    )
    assert parallelism.group_spread(2) == "sys"


def test_an_unreadable_topology_says_nothing(monkeypatch):
    from batcher.ml.llm.engines import parallelism

    monkeypatch.setattr("batcher._internal.hardware.fabric.tightest_device_group", lambda size: ())
    monkeypatch.setattr("batcher._internal.hardware.fabric.group_topology_class", lambda g: "")
    assert parallelism.group_spread(8) == ""


def test_the_boundary_reaches_the_warning_a_user_sees(monkeypatch, recwarn):
    from batcher._internal.errors import PerformanceWarning
    from batcher.ml.llm.engines import parallelism

    monkeypatch.setattr(parallelism, "_TP_WARNED", False)
    monkeypatch.setattr(parallelism, "group_spread", lambda size: "sys")
    parallelism.warn_about_tensor_parallelism(
        declared=4, model_gb=1.0, vram_gb=48.0, device_name="NVIDIA L40S"
    )
    (warning,) = [w for w in recwarn if issubclass(w.category, PerformanceWarning)]
    message = str(warning.message)
    assert "PCIe-only card" in message
    assert "cannot place 4 devices closer than `sys`" in message
