"""Per-rail shuffle throughput: the finding a node-wide utilization ratio hides.

A node whose devices all landed on one NIC reports the same summed utilization as a shuffle
that is simply slow, and the two have opposite fixes. The cases here drive the counters
directly, because no CI host has eight rails.
"""

from __future__ import annotations

import time

import pytest

from batcher._internal.hardware.fabric.counters import PortCounters
from batcher.carbonite.transfer.fabric_usage import fabric_baseline, rail_usage

pytestmark = pytest.mark.unit


def _sample(*octets: int) -> tuple[PortCounters, ...]:
    return tuple(
        PortCounters(device=f"mlx5_{i}", port=1, rcv_bytes=value, xmit_bytes=0)
        for i, value in enumerate(octets)
    )


def _usage(monkeypatch, before, after, seconds: float = 1.0) -> dict:
    import batcher._internal.hardware.fabric as fabric

    monkeypatch.setattr(fabric, "port_counters", lambda: after, raising=True)
    return rail_usage(before, time.monotonic() - seconds)


def test_a_node_that_used_one_rail_of_four_is_unmistakable(monkeypatch) -> None:
    before = _sample(0, 0, 0, 0)
    after = _sample(4_000_000_000, 0, 0, 0)
    usage = _usage(monkeypatch, before, after)
    assert usage["idle_rails"] == 3
    assert usage["spread"] == 4.0
    assert usage["busiest_gbps"] > 0.0


def test_an_evenly_used_fabric_has_a_spread_of_one(monkeypatch) -> None:
    before = _sample(0, 0, 0, 0)
    after = _sample(1_000_000_000, 1_000_000_000, 1_000_000_000, 1_000_000_000)
    usage = _usage(monkeypatch, before, after)
    assert usage["spread"] == 1.0
    assert usage["idle_rails"] == 0


def test_every_rail_is_reported_by_its_port_key(monkeypatch) -> None:
    usage = _usage(monkeypatch, _sample(0, 0), _sample(1_000_000_000, 0))
    assert sorted(usage["rails"]) == ["mlx5_0:1", "mlx5_1:1"]


def test_a_single_rail_node_has_no_spread_to_report(monkeypatch) -> None:
    assert _usage(monkeypatch, _sample(0), _sample(1_000_000_000)) == {}


def test_an_idle_rail_is_counted_even_though_the_delta_omits_it(monkeypatch) -> None:
    """The delta drops a port that moved nothing, which is exactly the port that matters."""
    usage = _usage(monkeypatch, _sample(0, 0, 0, 0), _sample(4_000_000_000, 0, 0, 0))
    assert sorted(usage["rails"]) == ["mlx5_0:1", "mlx5_1:1", "mlx5_2:1", "mlx5_3:1"]
    assert usage["rails"]["mlx5_3:1"] == 0.0


def test_a_node_with_no_fabric_reports_nothing() -> None:
    assert rail_usage((), 0.0) == {}


def test_a_counter_probe_failure_never_fails_a_shuffle(monkeypatch) -> None:
    import batcher._internal.hardware.fabric as fabric

    def boom():
        raise OSError("sysfs went away")

    monkeypatch.setattr(fabric, "port_counters", boom, raising=True)
    assert rail_usage(_sample(0, 0), time.monotonic() - 1.0) == {}


def test_a_baseline_on_a_host_without_rdma_is_empty() -> None:
    sample, started = fabric_baseline()
    assert (sample, started) == ((), 0.0) or sample
