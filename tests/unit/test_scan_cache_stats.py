"""Worker scan-cache effectiveness counters (hit-rate visibility for warm reads)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_scan_cache_stats_track_hit_rate():
    from batcher.dist.executors import scan_read as sr

    # Reset the module-global cache + counters for an isolated measurement.
    sr._SCAN_CACHE.clear()
    sr._SCAN_CACHE_HITS = 0
    sr._SCAN_CACHE_MISSES = 0
    sr._SCAN_CACHE_BYTES = 0

    key = ("src", "proj", "pred")
    assert sr._scan_cache_get(key) is None  # cold miss
    sr._scan_cache_put(key, ["batch"], 100)  # admit under the (large) default budget
    assert sr._scan_cache_get(key) == ["batch"]  # warm hit
    assert sr._scan_cache_get(key) == ["batch"]  # warm hit

    s = sr.scan_cache_stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["used_bytes"] == 100
