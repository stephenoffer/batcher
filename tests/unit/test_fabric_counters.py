"""Port counters — what the fabric carried, and what it got wrong carrying it.

A fabric cable does not fail cleanly. Symbol errors climb, the link retrains itself, and
throughput sags while the port stays `ACTIVE` and every scheduler above it keeps placing
cross-node stages on the node. These tests pin the reading, the unit conversion that is easy
to get wrong in a way that still looks plausible, and the two distinctions that decide whether
the numbers can be trusted: a counter the driver does not publish is not zero, and a counter
that went backwards is a driver reset rather than negative throughput.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.hardware.fabric import counters, rdma

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    rdma.reset_fabric_probes()
    yield
    rdma.reset_fabric_probes()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _fake_port(root: str, name: str, port: int, *, active: bool = True, **counter_values) -> None:
    """One `/sys/class/infiniband` port with a counters directory."""
    port_dir = os.path.join(root, name, "ports", str(port))
    _write(os.path.join(port_dir, "rate"), "400 Gb/sec (4X NDR)")
    _write(os.path.join(port_dir, "state"), "4: ACTIVE" if active else "1: DOWN")
    _write(os.path.join(port_dir, "link_layer"), "InfiniBand")
    for filename, value in counter_values.items():
        _write(os.path.join(port_dir, "counters", filename), f"{value}\n")


def test_no_fabric_means_no_counters(monkeypatch, tmp_path):
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", str(tmp_path / "absent"))
    assert counters.port_counters() == ()
    assert counters.fabric_error_total() == {}


def test_data_counters_are_four_octet_words_not_bytes(monkeypatch, tmp_path):
    # The unit slip this guards is the dangerous kind: it produces a figure four times too
    # small that still looks like a plausible measurement.
    root = str(tmp_path / "ib")
    _fake_port(root, "mlx5_0", 1, port_xmit_data=1000, port_rcv_data=250)
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    (sample,) = counters.port_counters()
    assert sample.xmit_bytes == 4000
    assert sample.rcv_bytes == 1000
    assert sample.key == "mlx5_0:1"


def test_an_unpublished_counter_is_absent_not_zero(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    _fake_port(root, "mlx5_0", 1, symbol_error=3)
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    (sample,) = counters.port_counters()
    assert sample.errors == {"symbol_errors": 3}
    assert "link_downed" not in sample.errors
    assert sample.xmit_bytes is None
    assert sample.readable is True


def test_a_port_publishing_nothing_reads_as_unreadable(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    _fake_port(root, "mlx5_0", 1)
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    (sample,) = counters.port_counters()
    assert sample.readable is False
    assert sample.total_errors == 0


def test_a_down_port_is_omitted_rather_than_reported_stale(monkeypatch, tmp_path):
    # Its counters are frozen at whatever they held when the link dropped; printing them
    # beside a live port's invites reading a stale total as a current one.
    root = str(tmp_path / "ib")
    _fake_port(root, "mlx5_0", 1, port_xmit_data=10)
    _fake_port(root, "mlx5_1", 1, active=False, port_xmit_data=999999)
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    assert [s.device for s in counters.port_counters()] == ["mlx5_0"]


def test_errors_sum_across_the_node(monkeypatch, tmp_path):
    root = str(tmp_path / "ib")
    _fake_port(root, "mlx5_0", 1, symbol_error=2, link_downed=1)
    _fake_port(root, "mlx5_1", 1, symbol_error=5, port_rcv_errors=7)
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", root)
    total = counters.fabric_error_total()
    assert total == {"symbol_errors": 7, "link_downed": 1, "rcv_errors": 7}
    assert sum(s.total_errors for s in counters.port_counters()) == 15


def test_throughput_is_the_difference_between_two_samples():
    before = (counters.PortCounters("mlx5_0", 1, xmit_bytes=0, rcv_bytes=0),)
    after = (counters.PortCounters("mlx5_0", 1, xmit_bytes=50_000_000_000, rcv_bytes=0),)
    # 50 GB in 1 s is 400 Gb/s.
    assert counters.throughput_delta(before, after, 1.0)["mlx5_0:1"] == pytest.approx(400.0)
    assert counters.throughput_delta(before, after, 2.0)["mlx5_0:1"] == pytest.approx(200.0)


def test_a_counter_that_went_backwards_is_a_reset_not_negative_throughput():
    before = (counters.PortCounters("mlx5_0", 1, xmit_bytes=1_000_000, rcv_bytes=0),)
    after = (counters.PortCounters("mlx5_0", 1, xmit_bytes=10, rcv_bytes=0),)
    assert counters.throughput_delta(before, after, 1.0) == {}


def test_a_port_missing_from_either_sample_is_skipped():
    before = (counters.PortCounters("mlx5_0", 1, xmit_bytes=0),)
    after = (
        counters.PortCounters("mlx5_0", 1, xmit_bytes=1_000_000),
        counters.PortCounters("mlx5_9", 1, xmit_bytes=1_000_000),
    )
    assert list(counters.throughput_delta(before, after, 1.0)) == ["mlx5_0:1"]


def test_a_zero_window_yields_nothing_rather_than_dividing_by_it():
    before = (counters.PortCounters("mlx5_0", 1, xmit_bytes=0),)
    after = (counters.PortCounters("mlx5_0", 1, xmit_bytes=10),)
    assert counters.throughput_delta(before, after, 0.0) == {}
    assert counters.throughput_delta(before, after, -1.0) == {}


def test_a_dropped_link_puts_a_node_on_the_drain_list():
    # Distinguished from symbol errors on purpose: a drop cost a stage its in-flight
    # transfers, while accumulating symbol errors is a warning about the next one.
    from batcher.dist.executors.ray_runtime import hardware_probe

    clean = {"node_id": "a", "quarantined": [], "fabric_errors": {"symbol_errors": 40}}
    dropped = {"node_id": "b", "quarantined": [], "fabric_errors": {"link_downed": 1}}
    assert [r["node_id"] for r in hardware_probe.unhealthy_nodes((clean, dropped))] == ["b"]


# --- What a shuffle reports about the wire it ran on ---------------------------------------


def test_a_node_with_no_fabric_adds_nothing_to_the_shuffle_stats(monkeypatch, tmp_path):
    # Most nodes. The measurement must cost nothing and say nothing there.
    monkeypatch.setattr(rdma, "RDMA_SYSFS_ROOT", str(tmp_path / "absent"))
    from batcher.carbonite.transfer import ShuffleSession

    stats = ShuffleSession(4).stats()
    assert not [k for k in stats if k.startswith("fabric_")]


def test_a_shuffle_reports_what_the_fabric_carried(monkeypatch):
    from batcher.carbonite.transfer import ShuffleSession

    baseline = (counters.PortCounters("mlx5_0", 1, xmit_bytes=0, rcv_bytes=0),)
    monkeypatch.setattr("batcher._internal.hardware.fabric.port_counters", lambda: baseline)
    shuffle = ShuffleSession(4)
    # A tenth of a 400 Gb/s fabric, which is the shape of "the shuffle never reached the
    # fast wire" — the failure every other statistic in the session reads the same through.
    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 400.0)
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.throughput_delta",
        lambda before, after, seconds: {"mlx5_0:1": 40.0},
    )
    stats = shuffle.stats()
    assert stats["fabric_gbps_observed"] == pytest.approx(40.0)
    assert stats["fabric_gbps_capable"] == pytest.approx(400.0)
    assert stats["fabric_utilization"] == pytest.approx(0.1)


def test_a_failing_probe_never_breaks_the_stats(monkeypatch):
    from batcher.carbonite.transfer import ShuffleSession

    baseline = (counters.PortCounters("mlx5_0", 1, xmit_bytes=0),)
    monkeypatch.setattr("batcher._internal.hardware.fabric.port_counters", lambda: baseline)
    shuffle = ShuffleSession(4)

    def _boom():
        raise RuntimeError("sysfs went away")

    monkeypatch.setattr("batcher._internal.hardware.fabric.port_counters", _boom)
    stats = shuffle.stats()
    assert "fetches" in stats
    assert "fabric_gbps_observed" not in stats
