"""The collective environment derived from measured wires, and what it refuses to set.

Two properties carry the whole module: a variable is set only when a probe answered, and a
variable the deployment set itself is never replaced.
"""

from __future__ import annotations

from batcher.dist.gpu.fabric.collective_env import (
    COLLECTIVE_VARS,
    collective_env,
    gdr_level,
    ib_hca_list,
    merge_env,
    p2p_disabled,
    socket_ifnames,
)
from batcher.dist.gpu.fabric.placement import (
    adaptive_shard_factor,
    device_shard_counts,
    fleet_spread,
    shard_device_assignment,
)

RAILS = {0: "mlx5_0", 1: "mlx5_1", 2: "mlx5_2", 3: "mlx5_3"}


def test_the_hca_list_is_the_rail_map_written_out() -> None:
    assert ib_hca_list(RAILS) == "mlx5_0,mlx5_1,mlx5_2,mlx5_3"


def test_the_hca_list_is_in_device_order_whatever_the_map_iterates_in() -> None:
    """Entry `i` is device `i`'s NIC, so an out-of-order map would misalign every rail."""
    assert ib_hca_list({2: "c", 0: "a", 1: "b"}) == "a,b,c"


def test_no_rail_map_leaves_the_library_to_choose() -> None:
    assert ib_hca_list({}) == ""


def test_the_socket_interfaces_are_the_fabrics_own() -> None:
    assert socket_ifnames(["ib0", "ib1"]) == "ib0,ib1"
    assert socket_ifnames([]) == ""


def test_the_gdr_level_is_named_only_where_the_dma_path_pays() -> None:
    assert gdr_level("pix") == "PIX"
    assert gdr_level("pxb") == "PXB"
    assert gdr_level("phb") == "PHB"
    assert gdr_level("sys") == ""  # past the point where the DMA path helps
    assert gdr_level("") == ""  # unmeasured: keep the library's default


def test_peer_to_peer_is_disabled_only_when_no_pair_can_use_it() -> None:
    assert p2p_disabled(staged_pairs=6, devices=4) == "1"  # every pair of four is staged


def test_a_node_with_some_direct_pairs_keeps_peer_to_peer() -> None:
    assert p2p_disabled(staged_pairs=4, devices=4) == ""


def test_an_unread_topology_gets_no_opinion_about_peer_to_peer() -> None:
    assert p2p_disabled(staged_pairs=0, devices=0) == ""
    assert p2p_disabled(staged_pairs=0, devices=4) == ""


def test_a_measured_node_gets_the_whole_block() -> None:
    env = collective_env(
        assignment=RAILS, interfaces=["ib0"], device_class="pix", staged_pairs=6, devices=4
    )
    assert env == {
        "NCCL_IB_HCA": "mlx5_0,mlx5_1,mlx5_2,mlx5_3",
        "NCCL_CROSS_NIC": "0",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_NET_GDR_LEVEL": "PIX",
        "NCCL_P2P_DISABLE": "1",
    }


def test_an_unreadable_node_gets_nothing_at_all() -> None:
    """The whole degradation path: the library then probes exactly as it did before."""
    assert collective_env(assignment={}, interfaces=[]) == {}


def test_cross_nic_is_only_claimed_where_an_alignment_exists() -> None:
    assert "NCCL_CROSS_NIC" not in collective_env(assignment={}, interfaces=["ib0"])


def test_every_variable_set_is_one_this_module_declares() -> None:
    env = collective_env(
        assignment=RAILS, interfaces=["ib0"], device_class="pix", staged_pairs=6, devices=4
    )
    assert set(env) <= set(COLLECTIVE_VARS)


def test_an_operator_setting_in_the_runtime_env_is_never_replaced() -> None:
    merged = merge_env({"NCCL_IB_HCA": "mlx5_9"}, {"NCCL_IB_HCA": "mlx5_0"}, process_env={})
    assert merged["NCCL_IB_HCA"] == "mlx5_9"


def test_an_operator_setting_in_the_process_is_never_replaced() -> None:
    merged = merge_env(None, {"NCCL_IB_HCA": "mlx5_0"}, process_env={"NCCL_IB_HCA": "mlx5_9"})
    assert merged == {}


def test_merging_does_not_mutate_its_inputs() -> None:
    base = {"OTHER": "1"}
    merged = merge_env(base, {"NCCL_IB_HCA": "mlx5_0"}, process_env={})
    assert base == {"OTHER": "1"}
    assert merged == {"OTHER": "1", "NCCL_IB_HCA": "mlx5_0"}


def test_shards_are_dealt_in_proportion_to_measured_throughput() -> None:
    """A device twice as fast takes twice the shards, or the stage runs at the slow one."""
    assert device_shard_counts(9, [2.0, 1.0]) == (6, 3)


def test_the_counts_sum_exactly_to_the_shard_count() -> None:
    for n in range(1, 20):
        assert sum(device_shard_counts(n, [3.0, 1.0, 1.5])) == n


def test_an_unmeasured_device_is_treated_as_average_not_as_idle() -> None:
    """Giving an unmeasured device nothing guarantees it stays unmeasured."""
    counts = device_shard_counts(8, [2.0, 2.0, 0.0])
    assert counts[2] > 0


def test_a_fleet_with_no_measurements_is_dealt_evenly() -> None:
    assert device_shard_counts(8, [0.0, 0.0]) == (4, 4)


def test_no_devices_and_no_shards_are_handled() -> None:
    assert device_shard_counts(8, []) == ()
    assert device_shard_counts(0, [1.0, 1.0]) == (0, 0)


def test_shards_are_interleaved_so_the_fast_device_starts_its_second_early() -> None:
    assert shard_device_assignment(3, [2.0, 1.0]) == (0, 1, 0)


def test_every_shard_is_placed_exactly_once() -> None:
    placed = shard_device_assignment(11, [3.0, 1.0, 2.0])
    assert len(placed) == 11
    assert set(placed) <= {0, 1, 2}


def test_no_devices_places_nothing() -> None:
    assert shard_device_assignment(4, []) == ()


def test_a_uniform_fleet_keeps_the_configured_shard_factor() -> None:
    """Every existing deployment runs this path; a measured fleet must not disturb it."""
    assert adaptive_shard_factor(4, [1000.0, 1000.0, 1020.0]) == 4


def test_an_unmeasured_fleet_keeps_the_configured_factor() -> None:
    assert adaptive_shard_factor(4, []) == 4
    assert adaptive_shard_factor(4, [0.0, 0.0]) == 4
    assert adaptive_shard_factor(4, [1000.0]) == 4


def test_an_uneven_fleet_is_divided_more_finely() -> None:
    """Ray runs one task per device, so equal shard counts end at the slowest device."""
    assert adaptive_shard_factor(4, [4000.0, 1000.0]) == 16


def test_the_multiplier_is_capped_so_one_sick_device_is_not_a_scheduler_problem() -> None:
    assert adaptive_shard_factor(4, [100_000.0, 100.0]) == 16


def test_the_factor_never_falls_below_the_configured_one() -> None:
    assert adaptive_shard_factor(0, [4000.0, 1000.0]) >= 1
    assert adaptive_shard_factor(8, [4000.0, 1000.0]) >= 8


def test_the_spread_ignores_unmeasured_devices() -> None:
    assert fleet_spread([1000.0, 0.0, 1000.0]) == 1.0
    assert fleet_spread([]) == 1.0
    assert fleet_spread([2000.0, 1000.0]) == 2.0
