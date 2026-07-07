"""The `"auto"` runtime-bloom-join policy engages only for selective large joins.

The distributed bloom reduction is result-invariant (no false negatives), so the
`"auto"` policy is a pure cardinality-driven performance decision: engage when the
probe side is much larger than the build side and large enough to repay the bloom.
"""

from __future__ import annotations

import types

import pytest

from batcher.dist.executors.join import (
    _BLOOM_AUTO_MIN_PROBE_ROWS,
    _bloom_beneficial,
    _bloom_engaged,
)

pytestmark = pytest.mark.unit


class _Join:
    join_type = "inner"
    left_keys = ("a",)
    right_keys = ("b",)
    left = "PROBE"
    right = "BUILD"


def _patch_estimator(monkeypatch, *, probe: int, build: int) -> None:
    """Make the estimator report fixed left(probe)/right(build) row counts."""
    sizes = {"PROBE": probe, "BUILD": build}

    class _FakeEstimator:
        def __init__(self, *_a, **_k):
            pass

        def estimate(self, node):
            return types.SimpleNamespace(rows=sizes[node])

    monkeypatch.setattr("batcher.kyber.cardinality.CardinalityEstimator", _FakeEstimator)


def test_force_true_and_false_ignore_estimate():
    assert _bloom_engaged(True, _Join(), []) is True
    assert _bloom_engaged(False, _Join(), []) is False


def test_auto_engages_for_selective_large_join(monkeypatch):
    _patch_estimator(monkeypatch, probe=100 * _BLOOM_AUTO_MIN_PROBE_ROWS, build=1000)
    assert _bloom_engaged("auto", _Join(), ["src"]) is True


def test_auto_skips_balanced_join(monkeypatch):
    # Probe only ~1.1x the build → below the ratio floor → keep the plain shuffle.
    n = 10 * _BLOOM_AUTO_MIN_PROBE_ROWS
    _patch_estimator(monkeypatch, probe=n, build=int(n / 1.1))
    assert _bloom_beneficial(_Join(), ["src"]) is False


def test_auto_skips_small_probe(monkeypatch):
    # Very selective ratio but a tiny probe → not worth building the bloom.
    _patch_estimator(monkeypatch, probe=_BLOOM_AUTO_MIN_PROBE_ROWS // 4, build=10)
    assert _bloom_beneficial(_Join(), ["src"]) is False


def test_auto_unknown_estimate_is_safe(monkeypatch):
    # A build estimate of 0 (unknown) must not divide-engage; keep the plain shuffle.
    _patch_estimator(monkeypatch, probe=10 * _BLOOM_AUTO_MIN_PROBE_ROWS, build=0)
    assert _bloom_beneficial(_Join(), ["src"]) is False
