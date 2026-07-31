"""The fabric-over-bus device topology: classes, islands, groups, and bisection.

Every case describes a node rather than reading one, because the property under test is what
the *decision* does with a topology and no CI host has the topologies that matter.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric.p2p import (
    NVLINK_CLASS,
    P2P_CLASSES,
    bandwidth_matrix,
    bisection_gbps,
    host_staged_pairs,
    island_of,
    p2p_capable,
    peer_bandwidth_gbps,
    peer_class,
    peer_group_class,
    peer_islands,
    peer_matrix,
    peer_summary,
    tightest_peer_group,
)

# A two-socket, four-device PCIe node: 0-1 under one switch, 2-3 under another, the two
# halves across the socket from each other.
PCIE_NODE = (
    ("pix", "pix", "sys", "sys"),
    ("pix", "pix", "sys", "sys"),
    ("sys", "sys", "pix", "pix"),
    ("sys", "sys", "pix", "pix"),
)


def test_peer_matrix_overlays_the_fabric_on_the_bus() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 2), (1, 3)])
    assert m[0][2] == NVLINK_CLASS
    assert m[2][0] == NVLINK_CLASS
    assert m[1][3] == NVLINK_CLASS
    # A pair with no link keeps whatever the bus said.
    assert m[0][1] == "pix"
    assert m[2][3] == "pix"


def test_a_fabric_node_reports_its_diagonal_as_fabric() -> None:
    """A device is as close to itself as the node's best class allows."""
    assert peer_matrix(PCIE_NODE, [(0, 1)])[3][3] == NVLINK_CLASS
    assert peer_matrix(PCIE_NODE, [])[3][3] == "pix"


def test_a_fabric_pair_naming_an_absent_device_is_ignored() -> None:
    """Two probes disagreeing about the device count is a reason to distrust the extra."""
    m = peer_matrix(PCIE_NODE, [(0, 9), (2, 3)])
    assert len(m) == 4
    assert m[2][3] == NVLINK_CLASS


def test_an_unreadable_bus_yields_no_matrix() -> None:
    assert peer_matrix((), [(0, 1)]) == ()


def test_an_index_off_the_matrix_reports_the_coarsest_class() -> None:
    assert peer_class(0, 9, PCIE_NODE) == "sys"
    assert peer_class(-1, 0, PCIE_NODE) == "sys"


def test_the_class_order_puts_the_fabric_first() -> None:
    assert P2P_CLASSES[0] == NVLINK_CLASS
    assert P2P_CLASSES.index("pix") < P2P_CLASSES.index("sys")


@pytest.mark.parametrize(
    ("peer_cls", "expected"),
    [(NVLINK_CLASS, 450.0), ("pix", 50.0), ("phb", 25.0), ("sys", 12.5)],
)
def test_bandwidth_derates_by_how_much_of_the_machine_is_crossed(
    peer_cls: str, expected: float
) -> None:
    assert peer_bandwidth_gbps(peer_cls, nvlink_gbps=450.0, pcie_gbps=50.0) == expected


def test_an_unpriced_link_reports_no_opinion_rather_than_a_guess() -> None:
    assert peer_bandwidth_gbps(NVLINK_CLASS, pcie_gbps=50.0) == 0.0
    assert peer_bandwidth_gbps("pix", nvlink_gbps=450.0) == 0.0


def test_bandwidth_matrix_prices_every_pair() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 2)])
    rates = bandwidth_matrix(m, nvlink_gbps=450.0, pcie_gbps=50.0)
    assert rates[0][2] == 450.0
    assert rates[0][1] == 50.0
    assert rates[1][2] == pytest.approx(12.5)


def test_direct_pairs_are_the_fabric_and_below_one_switch() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 2)])
    assert p2p_capable(0, 2, m)  # fabric
    assert p2p_capable(0, 1, m)  # one switch
    assert not p2p_capable(1, 2, m)  # across the socket
    assert p2p_capable(3, 3, m)  # itself


def test_staged_pairs_are_the_ones_that_bounce_through_the_host() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 2)])
    assert host_staged_pairs(m) == ((0, 3), (1, 2), (1, 3))


def test_islands_are_the_fabric_connected_components() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 1), (2, 3)])
    assert peer_islands(m) == ((0, 1), (2, 3))


def test_a_device_with_no_link_is_its_own_island() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 1)])
    assert peer_islands(m) == ((0, 1), (2,), (3,))
    assert island_of(2, peer_islands(m)) == (2,)
    assert island_of(9, peer_islands(m)) == ()


def test_islands_can_be_asked_of_the_bus_instead() -> None:
    """The same question of switch-local pairs, which is what a PCIe-only node has."""
    m = peer_matrix(PCIE_NODE, [])
    assert peer_islands(m, classes=(NVLINK_CLASS, "pix")) == ((0, 1), (2, 3))


def test_the_tightest_group_follows_the_fabric_not_the_bus() -> None:
    """The defect this module exists to remove: a bus-ranked group ignores NVLink."""
    m = peer_matrix(PCIE_NODE, [(0, 2), (0, 3), (2, 3)])
    assert tightest_peer_group(3, m) == (0, 2, 3)


def test_the_tightest_group_matches_the_bus_answer_with_no_fabric() -> None:
    m = peer_matrix(PCIE_NODE, [])
    assert tightest_peer_group(2, m) == (0, 1)


def test_a_group_larger_than_the_node_is_no_opinion() -> None:
    assert tightest_peer_group(9, PCIE_NODE) == ()
    assert tightest_peer_group(0, PCIE_NODE) == ()


def test_a_group_is_bounded_by_its_worst_pair() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 1)])
    assert peer_group_class((0, 1), m) == NVLINK_CLASS
    assert peer_group_class((0, 1, 2), m) == "sys"
    assert peer_group_class((0,), m) == ""


def test_bisection_sums_the_crossing_pairs() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 2), (1, 3)])
    # Cut {0,1} against {2,3}: two fabric pairs at 450 and two socket pairs at 12.5.
    assert bisection_gbps((0, 1, 2, 3), m, nvlink_gbps=450.0, pcie_gbps=50.0) == pytest.approx(
        925.0
    )
    assert bisection_gbps((0,), m, nvlink_gbps=450.0) == 0.0


def test_summary_reports_the_node_shape() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 1), (2, 3)])
    summary = peer_summary(m)
    assert summary["devices"] == 4
    assert summary["largest_island"] == 2
    assert summary["fabric_pairs"] == 2
    assert summary["class"] == "sys"


def test_summary_of_an_unreadable_node_is_zeroed() -> None:
    assert peer_summary(())["devices"] == 0
