"""Counting the accelerators of one model, without believing Ray's per-node marker.

Ray publishes a custom resource called `accelerator_type:<MODEL>`, which reads exactly like a
device count and is not one: it is a marker worth 1.0 on each node carrying that model. On a
four-node, sixteen-device T4 fleet `accelerator_type:T4` totals 4.0 while `GPU` totals 16.0.

Anything sizing against the marker gets the number of *nodes*, and the under-count is silent
and largest on the densest hardware. It reached the inference actor pool: a stage pinned to a
model -- the documented way to target a device class on a mixed fleet -- got four actors for
sixteen devices, so three quarters of them never received one.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.fabric.topology import GpuNodeTopology, devices_of_class

pytestmark = pytest.mark.unit


def _node(node_id: str, gpus: int, model: str) -> GpuNodeTopology:
    return GpuNodeTopology(node_id=node_id, gpus=gpus, accelerator_type=model)


#: This repo's own fleet: four nodes, four T4s each.
_T4_FLEET = tuple(_node(f"n{i}", 4, "T4") for i in range(4))

#: A mixed fleet: two dense A100 nodes beside four T4 nodes.
_MIXED = (*_T4_FLEET, _node("a0", 8, "A100"), _node("a1", 8, "A100"))


def test_it_counts_devices_not_nodes() -> None:
    """The whole point: 16 devices on 4 nodes is 16, not 4."""
    assert devices_of_class("T4", _T4_FLEET) == 16


def test_a_mixed_fleet_counts_only_the_model_asked_for() -> None:
    assert devices_of_class("T4", _MIXED) == 16
    assert devices_of_class("A100", _MIXED) == 16


def test_no_model_counts_every_accelerator() -> None:
    assert devices_of_class(None, _MIXED) == 32
    assert devices_of_class("", _MIXED) == 32


def test_a_model_the_fleet_has_not_got_counts_zero() -> None:
    """Zero means "none of these here", which is a different answer from "unknown"."""
    assert devices_of_class("H100", _MIXED) == 0


def test_matching_ignores_case_and_padding() -> None:
    """A label's spelling is written by whichever tool set it."""
    assert devices_of_class("t4", _T4_FLEET) == 16
    assert devices_of_class("  T4 ", _T4_FLEET) == 16


def test_an_unlabelled_node_matches_nothing_by_model() -> None:
    fleet = (_node("a", 4, ""), _node("b", 4, "T4"))
    assert devices_of_class("T4", fleet) == 4
    assert devices_of_class(None, fleet) == 8, "but it is still an accelerator"


def test_an_empty_fleet_counts_zero() -> None:
    assert devices_of_class("T4", ()) == 0
    assert devices_of_class(None, ()) == 0


def test_heterogeneous_density_sums_rather_than_averages() -> None:
    fleet = (_node("a", 1, "T4"), _node("b", 4, "T4"), _node("c", 8, "T4"))
    assert devices_of_class("T4", fleet) == 13
