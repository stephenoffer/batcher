"""A converged credit window belongs to the machine that converged it.

One credit is one in-flight batch slot: the ceiling is `credit_byte_budget` held under a
share of `total_memory_bytes()`, and the value the controller converges to is the path's
`BtlBw x RTprop`. That is the node's RAM and the fabric's bandwidth and delay --
machine units in exactly the sense `metadata.hardware_scope` defines, which names "batch
sizes chosen against them" outright.

It was the only machine-unit learned scalar in Carbonite left unscoped, beside
`io.throughput_mbps` and `carbonite.pressure_flap` which both carry the fingerprint.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.policies.flow_control import (
    load_shuffle_window,
    record_shuffle_window,
    shuffle_window_is_stable,
)
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hardware_scope import planning_for

pytestmark = pytest.mark.unit


def test_unlike_fleets_do_not_share_a_converged_window() -> None:
    """A fat-NIC fleet converges wide and a thin one narrow, on the same shuffle signature."""
    hub = MetadataHub(InProcessBackend())
    with planning_for("fat-nic"):
        for _ in range(6):
            record_shuffle_window(hub, "sig", 64)
    with planning_for("thin-nic"):
        for _ in range(6):
            record_shuffle_window(hub, "sig", 4)

    with planning_for("fat-nic"):
        assert load_shuffle_window(hub, "sig") == 64
    with planning_for("thin-nic"):
        assert load_shuffle_window(hub, "sig") == 4


def test_a_class_that_never_measured_stays_cold_rather_than_borrowing() -> None:
    """Cold is the honest answer, and it keeps the controller's own search switched on.

    `shuffle_window_is_stable` is what decides whether slow start is *skipped*, so a window
    borrowed from another fleet is not merely a poor warm start -- it can pin a badly-sized
    channel with no ramp to escape on.
    """
    hub = MetadataHub(InProcessBackend())
    with planning_for("fat-nic"):
        for _ in range(6):
            record_shuffle_window(hub, "sig", 64)
    with planning_for("never-ran-here"):
        assert load_shuffle_window(hub, "sig") is None
        assert shuffle_window_is_stable(hub, "sig") is False
