"""Carbonite transfer-mode selection (pure logic, no engine).

`select_mode`/`locality_ratio` decide and measure how much of a shuffle stays off
the network. These are pure functions, so they pin the policy without the native
engine — the integration suite exercises the modes end to end.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer.locality import TransferMode, locality_ratio, select_mode

pytestmark = pytest.mark.unit


def test_same_address_is_direct_memory():
    assert select_mode("127.0.0.1:5", "127.0.0.1:5") is TransferMode.DIRECT_MEMORY


def test_same_node_other_process_is_shared_memory():
    mode = select_mode("h:1", "h:2", source_node="nodeA", local_node="nodeA")
    assert mode is TransferMode.SHARED_MEMORY


def test_different_node_is_network():
    assert (
        select_mode("h:1", "h:2", source_node="nodeA", local_node="nodeB") is TransferMode.NETWORK
    )


def test_unknown_node_defaults_to_network():
    # Without node identity, a different address is conservatively remote.
    assert select_mode("h:1", "h:2") is TransferMode.NETWORK


def test_locality_ratio_counts_off_network_fraction():
    assert locality_ratio([]) == 1.0  # nothing moved → trivially local
    assert locality_ratio([TransferMode.DIRECT_MEMORY]) == 1.0
    assert locality_ratio([TransferMode.NETWORK]) == 0.0
    mixed = [TransferMode.DIRECT_MEMORY, TransferMode.SHARED_MEMORY, TransferMode.NETWORK]
    assert locality_ratio(mixed) == pytest.approx(2 / 3)
