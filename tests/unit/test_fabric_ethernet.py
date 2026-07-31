"""Ethernet is the fabric on most rented GPU capacity, and it was reading as no fabric at all.

`fabric.rdma` covers InfiniBand and RoCE, which the top tier is wired with. Below that tier a
node reaches the network over ordinary Ethernet, and `fabric_bandwidth_gbps()` reported zero
there — not "slow" but "unknown", which sent the cost model back to a constant. These pin the
three readings that have a plausible wrong answer: a container's virtual interfaces are not
network capacity, a bond is one link rather than two, and an unpublished rate is unknown rather
than zero.

The tree is faked in a tmp directory. The real `/sys/class/net` on this machine has one
interface that publishes no speed at all, which exercises exactly one of these cases.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric import ethernet

pytestmark = pytest.mark.unit


def _iface(
    root, name, *, speed=None, operstate="up", carrier="1", physical=True, master=None, bond=False
):
    """Write one fake interface into a `/sys/class/net`-shaped tree."""
    path = root / name
    path.mkdir()
    if speed is not None:
        (path / "speed").write_text(f"{speed}\n")
    (path / "operstate").write_text(f"{operstate}\n")
    (path / "carrier").write_text(f"{carrier}\n")
    if physical:
        device = root / "_pci" / "0000:00:1f.6"
        device.mkdir(parents=True, exist_ok=True)
        (path / "device").symlink_to(device)
    if master is not None:
        (path / "master").symlink_to(root / master)
    if bond:
        (path / "bonding").mkdir()
    return path


@pytest.fixture
def net(tmp_path, monkeypatch):
    root = tmp_path / "net"
    root.mkdir()
    monkeypatch.setattr(ethernet, "ETHERNET_SYSFS_ROOT", str(root))
    return root


def test_a_single_hundred_gig_nic_reads_its_rate(net):
    _iface(net, "ens5", speed=100_000)
    (link,) = ethernet.ethernet_links()
    assert link.name == "ens5"
    assert link.speed_gbps == 100.0, "the kernel publishes megabits; a caller wants gigabits"
    assert link.up is True
    assert link.address == "0000:00:1f.6"
    assert ethernet.ethernet_bandwidth_gbps() == 100.0


def test_loopback_and_a_container_bridge_are_not_network_capacity(net):
    # The failure this prevents: counting `docker0` and a veth pair as fabric, which on a
    # node with no real NIC visible produces a confident, entirely fictional bandwidth.
    _iface(net, "ens5", speed=25_000)
    _iface(net, "lo", speed=None, physical=False)
    _iface(net, "docker0", speed=10_000, physical=False)
    _iface(net, "veth9a1b2c3", speed=10_000, physical=False)
    assert [link.name for link in ethernet.ethernet_links()] == ["ens5"]
    assert ethernet.ethernet_bandwidth_gbps() == 25.0


def test_a_bond_counts_once_not_once_per_member(net):
    # Both the bond and its slaves appear in the tree. Summing all three reports 400 Gb/s on a
    # node that can move 200, and the optimizer then prices a shuffle at half what it costs.
    _iface(net, "bond0", speed=200_000, physical=False, bond=True)
    _iface(net, "eth0", speed=100_000, master="bond0")
    _iface(net, "eth1", speed=100_000, master="bond0")
    links = ethernet.ethernet_links()
    assert [link.name for link in links] == ["bond0"]
    assert links[0].bonded is True
    assert ethernet.ethernet_bandwidth_gbps() == 200.0


def test_two_separate_nics_do_add_up(net):
    _iface(net, "ens5", speed=100_000)
    _iface(net, "ens6", speed=100_000)
    assert ethernet.ethernet_bandwidth_gbps() == 200.0


def test_an_unpublished_rate_is_unknown_rather_than_zero(net):
    # The kernel returns -1 from an interface whose driver does not implement the query. That
    # is not a zero-bandwidth link, and the distinction is what keeps a real NIC from pricing
    # a shuffle at the maximum weight.
    _iface(net, "ens5", speed=-1)
    (link,) = ethernet.ethernet_links()
    assert link.speed_gbps == 0.0
    assert link.up is True, "unknown speed, known state: the link is still up"
    assert ethernet.ethernet_bandwidth_gbps() == 0.0


def test_a_missing_speed_file_is_unknown_too(net):
    _iface(net, "ens5", speed=None)
    assert ethernet.ethernet_bandwidth_gbps() == 0.0


def test_a_garbled_speed_file_is_unknown_too(net):
    _iface(net, "ens5", speed="not a number")
    assert ethernet.ethernet_bandwidth_gbps() == 0.0


def test_a_down_link_carries_nothing(net):
    _iface(net, "ens5", speed=100_000, operstate="down", carrier="0")
    (link,) = ethernet.ethernet_links()
    assert link.up is False
    assert link.speed_gbps == 100.0, "reported, because the cable is there"
    assert ethernet.ethernet_bandwidth_gbps() == 0.0, "and not counted, because it carries none"


def test_an_up_link_with_no_carrier_carries_nothing(net):
    # A cable pulled out of a running interface leaves `operstate` lagging on some drivers.
    _iface(net, "ens5", speed=100_000, operstate="up", carrier="0")
    assert ethernet.ethernet_bandwidth_gbps() == 0.0


def test_the_summary_names_the_interface_a_transfer_will_take(net):
    _iface(net, "eth0", speed=1_000)
    _iface(net, "ens5f0", speed=200_000)
    assert ethernet.ethernet_summary() == {
        "interfaces": 2,
        "up": 2,
        "total_gbps": 201.0,
        "fastest": "ens5f0",
    }


def test_no_interfaces_is_an_empty_summary_not_a_crash(net):
    assert ethernet.ethernet_summary() == {
        "interfaces": 0,
        "up": 0,
        "total_gbps": 0.0,
        "fastest": "",
    }


def test_no_sysfs_tree_at_all_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ethernet, "ETHERNET_SYSFS_ROOT", str(tmp_path / "absent"))
    assert ethernet.ethernet_links() == ()
    assert ethernet.ethernet_bandwidth_gbps() == 0.0


def test_this_host_answers_without_raising():
    assert isinstance(ethernet.ethernet_links(), tuple)
    assert ethernet.ethernet_bandwidth_gbps() >= 0.0


# --- Reaching the cost model --------------------------------------------------------------


def test_rdma_wins_when_the_node_has_both(monkeypatch):
    # A node with a fast fabric shuffles over it. Adding the management NIC's rate to the
    # InfiniBand rate would price the shuffle against bandwidth no batch will use.
    from batcher.kyber.cost import fabric as cost_fabric

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 3200.0)
    monkeypatch.setattr("batcher._internal.hardware.fabric.ethernet_bandwidth_gbps", lambda: 25.0)
    assert cost_fabric.measured_fabric_gbps() == 3200.0


def test_ethernet_prices_the_shuffle_when_there_is_no_rdma(monkeypatch):
    from batcher.kyber.cost import fabric as cost_fabric

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 0.0)
    monkeypatch.setattr("batcher._internal.hardware.fabric.ethernet_bandwidth_gbps", lambda: 100.0)
    assert cost_fabric.measured_fabric_gbps() == 100.0
    # 100 Gb/s is 12.5 GB/s against a 20 GB/s local reference: a shuffled byte costs 1.6 local
    # ones, which is *cheaper* than the 2.0 default the node used to be ranked against.
    assert cost_fabric.fabric_net_weight(100.0) == pytest.approx(1.6)


def test_a_slow_link_is_priced_as_expensive_but_stays_inside_the_clamp(monkeypatch):
    from batcher.kyber.cost import fabric as cost_fabric

    assert cost_fabric.fabric_net_weight(10.0) == 16.0
    assert cost_fabric.fabric_net_weight(0.1) == 32.0, "clamped, not unbounded"


def test_an_unreadable_network_keeps_the_configured_weight(monkeypatch):
    # The whole safety property: a container that can see neither fabric must rank plans the
    # way it always did, not the way a zero-bandwidth node would.
    from batcher.kyber.cost import fabric as cost_fabric

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 0.0)
    monkeypatch.setattr("batcher._internal.hardware.fabric.ethernet_bandwidth_gbps", lambda: 0.0)
    assert cost_fabric.fabric_net_weight() is None
