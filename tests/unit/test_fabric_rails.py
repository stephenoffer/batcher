"""Multi-rail NIC assignment: each device on its closest NIC, balanced across equals.

The defect these pin is the one a per-device probe cannot see. Asking "what is my nearest
NIC?" eight times independently is right eight times and wrong once, as a node: every device
can name the same NIC and then one rail carries the whole shuffle.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.fabric.rails import (
    Rail,
    assign_rails,
    device_rail_bandwidth_gbps,
    rail_aligned,
    rail_for_device,
    rail_imbalance,
    rail_summary,
)

# Four devices and two NICs, one per socket. Devices 0-1 are near NIC "a", 2-3 near "b".
DEVICES = ("d0", "d1", "d2", "d3")
NICS = (("a", "na"), ("b", "nb"))
_NEAR = {("d0", "na"), ("d1", "na"), ("d2", "nb"), ("d3", "nb")}


def near(dev: str, nic: str) -> int:
    return 0 if (dev, nic) in _NEAR else 4


def flat(_dev: str, _nic: str) -> int:
    """A node where every NIC is equidistant from every device — the balancing case."""
    return 1


def test_each_device_takes_its_closest_nic() -> None:
    assert assign_rails(DEVICES, NICS, near) == {0: "a", 1: "a", 2: "b", 3: "b"}


def test_equidistant_devices_are_spread_rather_than_piled_on_one_nic() -> None:
    """The whole point: a greedy per-device probe would answer `a` four times."""
    assert assign_rails(DEVICES, NICS, flat) == {0: "a", 1: "b", 2: "a", 3: "b"}


def test_balance_never_overrides_distance() -> None:
    """Crossing a socket to balance a rail costs more than the imbalance saves."""
    devices = ("d0", "d1", "d2")
    nics = (("a", "na"), ("b", "nb"))

    def lopsided(dev: str, nic: str) -> int:
        return 0 if nic == "na" else 4

    assert assign_rails(devices, nics, lopsided) == {0: "a", 1: "a", 2: "a"}


def test_a_constrained_device_is_placed_before_a_free_one() -> None:
    """Device 1 has only one good rail; it must not lose it to a device that had two."""
    devices = ("d0", "d1")
    nics = (("a", "na"), ("b", "nb"))

    def mixed(dev: str, nic: str) -> int:
        if dev == "d1":
            return 0 if nic == "na" else 8
        return 1

    assert assign_rails(devices, nics, mixed) == {0: "b", 1: "a"}


def test_a_device_with_no_address_is_left_unassigned() -> None:
    """An absent entry means no opinion, which leaves the caller's own selection alone."""
    assert assign_rails(("d0", "", "d2"), NICS, near) == {0: "a", 2: "b"}


def test_a_node_with_no_nics_assigns_nothing() -> None:
    assert assign_rails(DEVICES, (), near) == {}


def test_the_assignment_is_stable_across_processes() -> None:
    """Two workers that disagree about the rail map would each be right and still contend."""
    assert assign_rails(DEVICES, NICS, flat) == assign_rails(DEVICES, NICS, flat)


def test_a_rails_share_is_its_rate_over_its_tenants() -> None:
    rail = Rail("a", rate_gbps=400.0, devices=(0, 1, 2, 3))
    assert rail.share_gbps == 100.0
    assert rail.loaded


def test_an_empty_rail_has_no_share_rather_than_a_division_by_zero() -> None:
    rail = Rail("a", rate_gbps=400.0)
    assert rail.share_gbps == 0.0
    assert not rail.loaded


def test_imbalance_is_zero_when_the_rails_are_even() -> None:
    records = [Rail("a", devices=(0, 1)), Rail("b", devices=(2, 3))]
    assert rail_imbalance(records) == 0.0


def test_imbalance_flags_a_node_that_piled_onto_one_rail() -> None:
    records = [Rail("a", devices=(0, 1, 2, 3)), Rail("b", devices=(4,))]
    assert rail_imbalance(records) == pytest.approx(0.375)


def test_a_single_rail_cannot_be_unbalanced_against_itself() -> None:
    assert rail_imbalance([Rail("a", devices=(0, 1, 2))]) == 0.0
    assert rail_imbalance([]) == 0.0


def test_a_device_gets_its_own_rails_share_not_the_nodes_total() -> None:
    records = [Rail("a", rate_gbps=400.0, devices=(0, 1)), Rail("b", rate_gbps=400.0, devices=(2,))]
    assert device_rail_bandwidth_gbps(0, records) == 200.0
    assert device_rail_bandwidth_gbps(2, records) == 400.0
    assert device_rail_bandwidth_gbps(9, records) == 0.0


def test_alignment_is_true_where_there_is_no_rail_to_be_off() -> None:
    assert rail_aligned(0, "a", {0: "a"})
    assert not rail_aligned(0, "b", {0: "a"})
    assert rail_aligned(9, "b", {0: "a"})  # no assignment: no opinion, not a fault
    assert rail_aligned(0, "", {0: "a"})


def test_the_map_and_the_devices_agree() -> None:
    assert rail_for_device(1, {0: "a", 1: "b"}) == "b"
    assert rail_for_device(7, {0: "a"}) == ""


def test_summary_carries_the_imbalance_and_the_layout() -> None:
    # Both devices on one of two rails, which is the fault this metric exists to catch. It
    # used to report `0.0` for it: the imbalance was computed over *loaded* rails only, so a
    # node with one loaded rail had a single-entry load list and took the "cannot be unbalanced
    # against itself" branch. The empty rail is the evidence, and `rails()` keeps a record for
    # it precisely so this figure can see it.
    records = [Rail("a", rate_gbps=400.0, devices=(0, 1)), Rail("b", rate_gbps=400.0)]
    summary = rail_summary(records)
    assert summary == {
        "rails": 2,
        "loaded_rails": 1,
        "devices": 2,
        "imbalance": 0.5,
        "total_gbps": 400.0,
        "assignment": {"a": [0, 1]},
    }


def test_every_device_on_one_rail_of_eight_is_the_worst_answer_not_the_best() -> None:
    """The headline failure, which the metric used to score as perfectly balanced."""
    from batcher._internal.hardware.fabric.rails import rail_imbalance

    one_rail = [Rail("a", devices=tuple(range(8)))] + [Rail(f"r{i}") for i in range(7)]
    assert rail_imbalance(one_rail) == pytest.approx(0.875)
    balanced = [Rail(f"r{i}", devices=(i,)) for i in range(8)]
    assert rail_imbalance(balanced) == 0.0
