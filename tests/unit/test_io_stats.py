"""Observed per-source I/O throughput capture (measure on read, predict on re-read)."""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.metadata.io_stats import load_source_throughput_mbps, record_source_io

pytestmark = pytest.mark.unit

_MB = 1024 * 1024


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_records_and_smooths_throughput():
    hub = _hub()
    record_source_io(hub, "s3://b/x", byte_count=100 * _MB, elapsed_ms=1000.0)  # 100 MB/s
    assert load_source_throughput_mbps(hub, "s3://b/x") == pytest.approx(100.0)
    # A second, faster observation exp-smooths toward it (stays strictly between).
    record_source_io(hub, "s3://b/x", byte_count=200 * _MB, elapsed_ms=1000.0)  # 200 MB/s
    smoothed = load_source_throughput_mbps(hub, "s3://b/x")
    assert 100.0 < smoothed < 200.0


def test_explain_surfaces_learned_read_throughput():
    # The measured throughput becomes a `Decision` in explain(), so a slow (small-files)
    # source is visible before running — the observe half of the io measurement chain.
    from batcher.api.terminal.profile import _io_throughput_decisions

    class _Src:
        def identity(self) -> str:
            return "s3://b/x"

    hub = _hub()
    record_source_io(hub, "s3://b/x", 100 * _MB, 1000.0)  # 100 MB/s
    decisions = _io_throughput_decisions([_Src()], hub)
    assert len(decisions) == 1
    assert decisions[0].category == "io"
    assert "MB/s" in decisions[0].summary
    assert decisions[0].detail["throughput_mbps"] == pytest.approx(100.0)
    # A source with no learned throughput yields no decision.
    assert _io_throughput_decisions([_Src()], _hub()) == []
    # With a known byte size, the decision predicts the read cost (200 MB @ 100 MB/s ≈ 2s).
    from batcher.metadata.source_stats_store import save_source_stats
    from batcher.plan.source_stats import SourceStatistics

    save_source_stats(hub, "s3://b/x", SourceStatistics(row_count=1000, byte_size=200 * _MB))
    d = _io_throughput_decisions([_Src()], hub)[0]
    assert d.detail["predicted_read_seconds"] == pytest.approx(2.0)
    assert "to read" in d.summary


def test_predicted_read_seconds():
    from batcher.metadata.io_stats import predicted_read_seconds

    hub = _hub()
    record_source_io(hub, "s3://b/x", 100 * _MB, 1000.0)  # 100 MB/s learned
    assert predicted_read_seconds(hub, "s3://b/x", 200 * _MB) == pytest.approx(2.0)
    assert predicted_read_seconds(hub, "cold-source", 100 * _MB) is None  # never measured
    assert predicted_read_seconds(hub, "s3://b/x", 0) is None  # no bytes


def test_bad_inputs_and_cold_store_are_safe():
    hub = _hub()
    record_source_io(hub, "id", 0, 100.0)  # zero bytes → dropped, not recorded
    record_source_io(hub, "id", 100 * _MB, 0.0)  # zero time → dropped
    record_source_io(None, "id", 100 * _MB, 100.0)  # no hub → no-op
    assert load_source_throughput_mbps(hub, "id") is None
    assert load_source_throughput_mbps(None, "id") is None
