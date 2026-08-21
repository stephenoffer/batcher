"""IPv6-only clusters, where the shuffle could not address itself.

An IPv6-only Kubernetes cluster (IPv6-only EKS and GKE, and several on-prem builds) has no
IPv4 address to fall back to, so two IPv4 assumptions in the shuffle path were fatal rather
than merely limiting:

* the fabric-address probe read only `AF_INET`, so an IPv6-only InfiniBand or RoCE fabric
  reported no address and the entire shuffle silently went back over the management NIC — two
  orders of magnitude slower, with nothing to say so;
* the advertised address was built as `f"{host}:{port}"`, which for an IPv6 literal is not an
  authority at all, and the listener bound `0.0.0.0`, which has no IPv6 interface.

The bracketing and the bind live in Rust (`bc-transport`, `bc-py`) and are tested there. This
file covers the Python half and the address contract the two sides share.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from batcher._internal.hardware.fabric import rdma
from batcher.carbonite.transfer.lifecycle import host_of

pytestmark = pytest.mark.unit


def _addr(family, address):
    return SimpleNamespace(family=family, address=address)


@pytest.fixture
def fabric(monkeypatch):
    """Install a fake fabric interface set, and return the dict the test fills in."""
    ifaces: dict[str, list] = {}
    monkeypatch.setattr(rdma, "rdma_net_interfaces", lambda: tuple(ifaces))

    class _FakePsutil:
        @staticmethod
        def net_if_addrs():
            return ifaces

        @staticmethod
        def net_if_stats():
            return {name: SimpleNamespace(isup=True) for name in ifaces}

    monkeypatch.setitem(__import__("sys").modules, "psutil", _FakePsutil)
    return ifaces


def test_an_ipv6_only_fabric_is_addressable(fabric):
    fabric["ibp0"] = [_addr(socket.AF_INET6, "fd00:1234::5")]
    assert rdma.fabric_interface_address() == "fd00:1234::5"


def test_ipv4_is_preferred_when_the_fabric_has_both(fabric):
    # The rest of the stack is IPv4-centric and a dual-stack fabric routes either, so the
    # more universally dialable one wins.
    fabric["ibp0"] = [_addr(socket.AF_INET6, "fd00::5"), _addr(socket.AF_INET, "10.1.2.3")]
    assert rdma.fabric_interface_address() == "10.1.2.3"


def test_ipv4_on_a_second_interface_still_beats_ipv6_on_the_first(fabric):
    fabric["ibp0"] = [_addr(socket.AF_INET6, "fd00::5")]
    fabric["ibp1"] = [_addr(socket.AF_INET, "10.1.2.3")]
    assert rdma.fabric_interface_address() == "10.1.2.3"


@pytest.mark.parametrize("link_local", ["fe80::1", "fe80::a1b2%ibp0", "::1"])
def test_a_link_local_or_loopback_address_is_not_advertised(fabric, link_local):
    # `fe80::` is dialable only with the *peer's* zone index appended, which this node
    # cannot know, so advertising one gives every reducer a connect timeout.
    fabric["ibp0"] = [_addr(socket.AF_INET6, link_local)]
    assert rdma.fabric_interface_address() == ""


def test_a_zone_suffix_is_stripped_from_a_global_address(fabric):
    fabric["ibp0"] = [_addr(socket.AF_INET6, "fd00::9%ibp0")]
    assert rdma.fabric_interface_address() == "fd00::9"


def test_a_down_interface_is_never_advertised(fabric, monkeypatch):
    fabric["ibp0"] = [_addr(socket.AF_INET6, "fd00::5")]
    monkeypatch.setattr(
        __import__("sys").modules["psutil"],
        "net_if_stats",
        staticmethod(lambda: {"ibp0": SimpleNamespace(isup=False)}),
    )
    assert rdma.fabric_interface_address() == ""


def test_the_node_identity_of_a_bracketed_ipv6_address_is_the_host(monkeypatch):
    # `host_of` splits at the last colon, which is only correct because the Rust side
    # brackets the literal. This pins the two halves of that contract together.
    assert host_of("[fd00::1]:50051") == "[fd00::1]"
    assert host_of("10.0.0.4:50051") == "10.0.0.4"
