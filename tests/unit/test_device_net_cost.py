"""What a byte off a device costs, and how wide a stage may fan out before it.

The node's summed port rate is optimistic for device-resident data in two directions at once:
a device uses one rail, and its bytes cross the host link first. Both errors point the same
way, so a stage planned against the host figure expects bandwidth it does not have.
"""

from __future__ import annotations

import pytest

from batcher.kyber.gpu.exchange import (
    DeviceFabric,
    device_exchange_gbps,
    device_net_gbps,
    device_net_weight,
    fabric_bounded_width,
    widest_fabric_island,
)

pytestmark = pytest.mark.unit


def test_the_off_node_rate_is_the_narrower_of_the_two_wires() -> None:
    """400 Gb/s of rail is 50 GB/s, and the host link is 25: the byte gets 25."""
    wires = DeviceFabric(rail_gbps=400.0, host_link_gbps=25.0)
    assert device_net_gbps(wires) == 25.0


def test_a_shared_rail_bounds_the_byte_even_behind_a_wide_link() -> None:
    wires = DeviceFabric(rail_gbps=50.0, host_link_gbps=64.0)
    assert device_net_gbps(wires) == pytest.approx(6.25)


def test_a_partially_readable_node_is_priced_against_the_half_it_knows() -> None:
    assert device_net_gbps(DeviceFabric(host_link_gbps=25.0)) == 25.0
    assert device_net_gbps(DeviceFabric(rail_gbps=400.0)) == 50.0


def test_an_unreadable_node_has_no_rate_and_no_weight() -> None:
    wires = DeviceFabric()
    assert not wires.readable
    assert device_net_gbps(wires) == 0.0
    assert device_net_weight(wires) is None


def test_the_weight_is_local_bandwidth_over_the_device_rate() -> None:
    wires = DeviceFabric(rail_gbps=400.0, host_link_gbps=25.0)
    assert device_net_weight(wires, local_gbps=25.0) == pytest.approx(1.0)
    assert device_net_weight(wires, local_gbps=50.0) == pytest.approx(2.0)


def test_the_weight_never_makes_a_device_byte_cheaper_than_a_local_one() -> None:
    fast = DeviceFabric(rail_gbps=4000.0, host_link_gbps=500.0)
    assert device_net_weight(fast, local_gbps=20.0) == 1.0


def test_the_weight_is_capped_where_the_ranking_stops_changing() -> None:
    slow = DeviceFabric(rail_gbps=1.0, host_link_gbps=0.1)
    assert device_net_weight(slow, local_gbps=20.0) == 32.0


def test_an_exchange_inside_one_island_runs_on_the_fabric() -> None:
    wires = DeviceFabric(host_link_gbps=25.0, island=8)
    assert device_exchange_gbps(4, wires, nvlink_gbps=450.0) == 450.0


def test_an_exchange_past_the_island_is_bounded_by_the_host_link() -> None:
    wires = DeviceFabric(host_link_gbps=25.0, island=4)
    assert device_exchange_gbps(8, wires, nvlink_gbps=450.0) == 25.0


def test_an_exchange_of_one_device_is_not_an_exchange() -> None:
    assert device_exchange_gbps(1, DeviceFabric(island=8), nvlink_gbps=450.0) == 0.0


def test_an_unknown_device_model_falls_back_to_the_host_link() -> None:
    wires = DeviceFabric(host_link_gbps=25.0, island=8)
    assert device_exchange_gbps(4, wires) == 25.0


def test_a_stage_that_exchanges_is_capped_at_the_island() -> None:
    """The ninth device on an eight-wide fabric makes the collective slower than eight."""
    assert fabric_bounded_width(16, 8) == 8


def test_a_stage_of_independent_shards_is_not_capped() -> None:
    assert fabric_bounded_width(16, 8, exchanges=False) == 16


def test_a_request_that_already_fits_is_untouched() -> None:
    assert fabric_bounded_width(4, 8) == 4


def test_an_unreadable_topology_leaves_the_request_alone() -> None:
    assert fabric_bounded_width(16, 0) == 16


def test_the_widest_island_is_read_from_the_groups_it_is_given() -> None:
    assert widest_fabric_island(((0, 1, 2, 3), (4, 5))) == 4
    assert widest_fabric_island(()) == 0


def test_the_summary_carries_the_derived_rate() -> None:
    summary = DeviceFabric(rail_gbps=400.0, host_link_gbps=25.0, island=8).summary()
    assert summary["net_gbps"] == 25.0
    assert summary["island"] == 8
