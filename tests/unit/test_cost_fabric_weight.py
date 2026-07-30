"""Pricing a shuffled byte against the fabric the cluster actually has.

`CostWeights.net` was the constant 2.0 everywhere. These tests pin the two directions that
constant is wrong in — a 400 Gb/s InfiniBand node where a shuffled byte costs about what a
local one does, and a 10 Gb/s VM where it costs an order of magnitude more — and, more
importantly, pin the cases where nothing may change: an unreadable fabric and an operator who
set the weight themselves must both rank plans exactly as before.
"""

from __future__ import annotations

import pytest

from batcher.config import CostWeights
from batcher.kyber.cost import Cost
from batcher.kyber.cost import fabric as cost_fabric

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_weight():
    cost_fabric.reset_fabric_weight()
    yield
    cost_fabric.reset_fabric_weight()


def test_an_unreadable_fabric_keeps_the_configured_weight():
    assert cost_fabric.fabric_net_weight(0.0) is None
    assert cost_fabric.fabric_net_weight(None, local_gbps=0.0) is None
    weights = CostWeights()
    assert cost_fabric.fabric_adjusted_weights(weights) is weights


def test_a_datacenter_fabric_prices_a_shuffled_byte_near_a_local_one():
    # Eight 400 Gb/s ports is 400 GB/s on the wire, past what a host reaches against its own
    # memory, so the weight floors at 1.0 rather than dropping below it.
    assert cost_fabric.fabric_net_weight(3200.0) == pytest.approx(1.0)
    # A single 200 Gb/s port is 25 GB/s, still faster than the 20 GB/s local reference.
    assert cost_fabric.fabric_net_weight(200.0) == pytest.approx(1.0)


def test_a_commodity_nic_prices_a_shuffled_byte_far_above_a_local_one():
    # 10 Gb/s is 1.25 GB/s against a 20 GB/s local reference: sixteen times.
    assert cost_fabric.fabric_net_weight(10.0) == pytest.approx(16.0)
    # 25 Gb/s is 3.125 GB/s: six and a half times, above the 2.0 the constant assumed.
    assert cost_fabric.fabric_net_weight(25.0) == pytest.approx(6.4)


def test_the_weight_is_clamped_at_both_ends():
    assert cost_fabric.fabric_net_weight(1_000_000.0) == pytest.approx(1.0)
    assert cost_fabric.fabric_net_weight(0.001) == pytest.approx(32.0)


def test_an_explicit_operator_weight_outranks_the_measurement(monkeypatch):
    monkeypatch.setattr(cost_fabric, "_measured_weight", lambda: 1.0)
    chosen = CostWeights(net=8.0)
    assert cost_fabric.fabric_adjusted_weights(chosen) is chosen


def test_a_measured_fabric_reaches_the_scalar_cost(monkeypatch):
    monkeypatch.setattr(cost_fabric, "_measured_weight", lambda: 1.0)
    cost = Cost(cpu=10.0, io=5.0, net=100.0)
    # 10 + 5 + 100 * 1.0 on the fast fabric, against 10 + 5 + 100 * 2.0 at the default.
    assert cost.total() == pytest.approx(115.0)
    monkeypatch.setattr(cost_fabric, "_measured_weight", lambda: 16.0)
    assert cost.total() == pytest.approx(1615.0)


def test_a_single_node_plan_is_ranked_exactly_as_before(monkeypatch):
    # The `net` axis is zero by construction on one node, so no fabric measurement can move
    # a single-node ranking however fast or slow the NIC turns out to be.
    cost = Cost(cpu=10.0, io=5.0)
    for weight in (1.0, 2.0, 32.0):
        monkeypatch.setattr(cost_fabric, "_measured_weight", lambda w=weight: w)
        assert cost.total() == pytest.approx(15.0)


def test_the_summary_reports_what_was_derived_and_what_is_in_force(monkeypatch):
    monkeypatch.setattr(cost_fabric, "_measured_weight", lambda: 4.0)
    summary = cost_fabric.net_weight_summary()
    assert summary["derived_net_weight"] == pytest.approx(4.0)
    assert summary["net_weight"] == pytest.approx(4.0)


def test_the_derived_weight_is_computed_once(monkeypatch):
    calls = []

    def _probe():
        calls.append(1)
        return 100.0

    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", _probe)
    cost_fabric.reset_fabric_weight()
    for _ in range(50):
        Cost(net=1.0).total()
    assert len(calls) == 1
