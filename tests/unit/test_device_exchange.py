"""Device-to-device redistribution: the schedule, the ring, the cost, and the refusals.

A pairing that lets two copies share one device is not a slower plan, it is the plan the
caller already had — so disjointness per round is the property every case here turns on.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric.p2p import NVLINK_CLASS, fabric_fraction, peer_matrix
from batcher.carbonite.transfer.device_exchange import (
    ExchangePlan,
    all_reduce_seconds,
    exchange_seconds,
    pairwise_rounds,
    plan_exchange,
    ring_bandwidth_gbps,
    ring_order,
    staged_pairs_in,
    worth_device_exchange,
)
from batcher.carbonite.transfer.locality import TransferMode, select_device_mode

PCIE_NODE = (
    ("pix", "pix", "sys", "sys"),
    ("pix", "pix", "sys", "sys"),
    ("sys", "sys", "pix", "pix"),
    ("sys", "sys", "pix", "pix"),
)
FABRIC = peer_matrix(PCIE_NODE, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def test_every_pair_exchanges_exactly_once() -> None:
    steps = pairwise_rounds(range(4))
    seen = [pair for step in steps for pair in step.pairs]
    assert sorted(seen) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(seen) == len(set(seen))


def test_no_device_appears_twice_in_one_round() -> None:
    """The property that makes a round run at link rate instead of half of it."""
    for step in pairwise_rounds(range(8)):
        members = [d for pair in step.pairs for d in pair]
        assert len(members) == len(set(members))


def test_an_even_node_takes_n_minus_one_full_rounds() -> None:
    steps = pairwise_rounds(range(8))
    assert len(steps) == 7
    assert all(step.width == 4 for step in steps)


def test_an_odd_node_sits_one_device_out_per_round() -> None:
    steps = pairwise_rounds(range(5))
    assert len(steps) == 5
    assert all(step.width == 2 for step in steps)
    seen = [pair for step in steps for pair in step.pairs]
    assert len(seen) == 10  # every pair of five, still exactly once


def test_fewer_than_two_devices_has_nothing_to_exchange() -> None:
    assert pairwise_rounds([3]) == ()
    assert pairwise_rounds([]) == ()


def test_duplicates_and_order_do_not_change_the_schedule() -> None:
    assert pairwise_rounds([2, 0, 1, 0]) == pairwise_rounds([0, 1, 2])


def test_the_ring_follows_the_fabric_rather_than_the_index() -> None:
    """0-2 and 1-3 are linked; index order would cross the socket on every second hop."""
    m = peer_matrix(PCIE_NODE, [(0, 2), (2, 1), (1, 3), (3, 0)])
    assert ring_order(range(4), m) == (0, 2, 1, 3)


def test_the_ring_is_index_order_on_a_node_with_no_fabric() -> None:
    assert ring_order(range(2), peer_matrix(PCIE_NODE, [])) == (0, 1)


def test_a_ring_runs_at_its_worst_hop_including_the_one_that_closes_it() -> None:
    m = peer_matrix(PCIE_NODE, [(0, 1), (1, 2), (2, 3)])
    # The closing hop 3->0 is a socket crossing, so the ring is bounded by it, not by NVLink.
    assert ring_bandwidth_gbps((0, 1, 2, 3), m, nvlink_gbps=450.0, pcie_gbps=50.0) == pytest.approx(
        12.5
    )


def test_a_ring_of_one_has_no_rate() -> None:
    assert ring_bandwidth_gbps((0,), FABRIC, nvlink_gbps=450.0) == 0.0


def test_staged_pairs_are_the_ones_the_bus_keeps_apart() -> None:
    m = peer_matrix(PCIE_NODE, [])
    assert staged_pairs_in(range(4), m) == ((0, 2), (0, 3), (1, 2), (1, 3))
    assert staged_pairs_in(range(4), FABRIC) == ()


def test_a_round_costs_its_slowest_pair() -> None:
    steps = pairwise_rounds(range(2))
    seconds = exchange_seconds(2_000_000_000, steps, FABRIC, nvlink_gbps=500.0, pcie_gbps=50.0)
    assert seconds == pytest.approx(2_000_000_000 / 500e9)


def test_nothing_to_move_costs_nothing() -> None:
    assert exchange_seconds(0, pairwise_rounds(range(4)), FABRIC, nvlink_gbps=500.0) == 0.0
    assert exchange_seconds(1000, (), FABRIC, nvlink_gbps=500.0) == 0.0


def test_an_unpriced_node_reports_no_duration() -> None:
    """And `worth_device_exchange` then refuses, rather than spending the optimism."""
    assert exchange_seconds(1_000_000, pairwise_rounds(range(4)), FABRIC) == 0.0


def test_all_reduce_follows_the_two_n_minus_one_over_n_bound() -> None:
    assert all_reduce_seconds(1_000_000_000, 4, 500.0) == pytest.approx(2 * 3 / 4 * 1e9 / 500e9)
    assert all_reduce_seconds(1_000_000_000, 1, 500.0) == 0.0
    assert all_reduce_seconds(0, 4, 500.0) == 0.0
    assert all_reduce_seconds(1_000_000_000, 4, 0.0) == 0.0


def test_a_fabric_plan_beats_the_host_path_and_is_taken() -> None:
    plan = plan_exchange(
        range(4), 8_000_000_000, FABRIC, nvlink_gbps=450.0, pcie_gbps=50.0, host_gbps=25.0
    )
    assert plan.rounds == 3
    assert plan.fully_direct
    assert plan.speedup > 1.25
    assert worth_device_exchange(plan)


def test_a_bus_only_plan_that_ties_is_refused() -> None:
    """The host path already moves these bytes correctly; a tie does not buy the new one."""
    plan = ExchangePlan(steps=pairwise_rounds(range(2)), seconds=1.0, host_seconds=1.1)
    assert not worth_device_exchange(plan)
    assert plan.speedup == pytest.approx(1.1)


def test_an_unpriced_plan_is_refused_rather_than_assumed_favorable() -> None:
    assert not worth_device_exchange(ExchangePlan(steps=pairwise_rounds(range(2))))
    assert not worth_device_exchange(ExchangePlan(seconds=1.0, host_seconds=9.0))


def test_a_margin_below_one_is_clamped() -> None:
    """A plan that is slower is never worth running, whatever margin was asked for."""
    plan = ExchangePlan(steps=pairwise_rounds(range(2)), seconds=2.0, host_seconds=1.0)
    assert not worth_device_exchange(plan, margin=0.1)


def test_a_single_device_plan_is_empty_rather_than_free() -> None:
    plan = plan_exchange([0], 1_000_000, FABRIC, nvlink_gbps=450.0, host_gbps=25.0)
    assert plan.rounds == 0
    assert not worth_device_exchange(plan)


def test_the_plan_summarizes_flat() -> None:
    plan = plan_exchange(
        range(4), 4_000_000_000, FABRIC, nvlink_gbps=450.0, pcie_gbps=50.0, host_gbps=25.0
    )
    summary = plan.summary()
    assert summary["rounds"] == 3
    assert summary["pairs"] == 6
    assert summary["staged"] == 0
    assert summary["speedup"] > 1.0


def test_fabric_fraction_separates_a_domain_from_a_chassis() -> None:
    assert fabric_fraction(range(4), FABRIC) == 1.0
    assert fabric_fraction(range(4), peer_matrix(PCIE_NODE, [])) == 0.0
    assert fabric_fraction(range(4), peer_matrix(PCIE_NODE, [(0, 1), (2, 3)])) == pytest.approx(
        2 / 6
    )
    assert fabric_fraction([0], FABRIC) == 0.0


def test_the_matrix_and_the_mode_selector_agree_on_a_direct_pair() -> None:
    """`p2p_capable` is what `select_device_mode` is meant to be handed."""
    assert FABRIC[0][3] == NVLINK_CLASS
    assert select_device_mode(0, 3, direct=True) is TransferMode.DEVICE_P2P
    assert select_device_mode(0, 0, direct=True) is TransferMode.DEVICE_LOCAL


def test_a_pair_the_bus_keeps_apart_falls_back_to_the_host_mode() -> None:
    assert (
        select_device_mode(0, 2, host_mode=TransferMode.SHARED_MEMORY, direct=False)
        is TransferMode.SHARED_MEMORY
    )


def test_a_host_side_endpoint_is_not_a_device_transfer() -> None:
    assert select_device_mode(-1, 0, host_mode=TransferMode.NETWORK) is TransferMode.NETWORK
    assert select_device_mode(-1, -1, host_mode=TransferMode.NETWORK) is TransferMode.NETWORK


def test_the_device_modes_rank_below_the_network_and_stay_local() -> None:
    assert TransferMode.DEVICE_LOCAL.rank < TransferMode.DIRECT_MEMORY.rank
    assert TransferMode.DEVICE_P2P.rank < TransferMode.SHARED_MEMORY.rank
    assert TransferMode.DEVICE_P2P.is_local
    assert TransferMode.DEVICE_LOCAL.is_local
