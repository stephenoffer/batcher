"""The executor re-checks the measured build side before giving every device a copy.

Kyber decides whether a join broadcasts from *estimates*, and an estimate of a join's inputs is
what this engine's cardinality model is least reliable about: TPC-H q14 and q17 at sf10 estimate
**one row** for relations of tens of millions. The device fan-out acted on that estimate and
never looked again.

The CPU path does look. `OptimizerConfig.resolved_broadcast_max_bytes` documents that "the
executor re-checks the *measured* build side against this same number before replicating it, so
a planner under-estimate costs a fallback rather than a cluster-wide OOM". This is that check
for the device tier, with the extra term a device fan-out has and a CPU shuffle does not: every
device *reads* its own copy, so replication costs `build x devices` against the probe side it
splits. Measured on six T4s, TPC-H q12 at sf10 took **24.1 s** on the wrong side of that
comparison, against the CPU engine's 1.3 s.
"""

from __future__ import annotations

import pytest

from batcher.dist.gpu.join import _replication_measures_up

pytestmark = pytest.mark.unit

GB = 1e9


class _Source:
    """A source that reports a size, which is all `source_bytes` reads from it."""

    def __init__(self, nbytes: float):
        self._nbytes = nbytes


@pytest.fixture
def probe_bytes(monkeypatch):
    def _use(nbytes):
        monkeypatch.setattr("batcher.dist.gpu.shards.source_bytes", lambda s, p=None: nbytes)

    return _use


def test_a_small_dimension_against_a_large_fact_still_replicates(probe_bytes):
    """The shape the fan-out exists for, and it must survive the guard untouched."""
    probe_bytes(20 * GB)
    assert _replication_measures_up(0.001 * GB, _Source(20 * GB), None, 6) is True


def test_a_build_side_the_probe_barely_exceeds_still_fits(probe_bytes):
    """The executor asks only about **fit**. Whether the fan-out buys enough to be worth running
    is decided on the plan, and asking it again here in *aggregate bytes* was wrong: the devices
    read concurrently, so replicating is never slower than the single device it replaces. That
    rule refused TPC-H q14 — 0.48 GB replicated against a 1.68 GB probe — and gave up 12.4x."""
    probe_bytes(2 * GB)
    assert _replication_measures_up(1 * GB, _Source(2 * GB), None, 6) is True


def test_the_fleet_width_does_not_change_the_fit(probe_bytes):
    """Each device holds one copy however many devices there are."""
    probe_bytes(10 * GB)
    assert _replication_measures_up(1 * GB, _Source(10 * GB), None, 4) is True
    assert _replication_measures_up(1 * GB, _Source(10 * GB), None, 32) is True


def test_a_build_side_past_the_device_budget_is_refused(probe_bytes):
    """Whatever the ratio says, it has to fit beside the shard being joined."""
    probe_bytes(10_000 * GB)
    assert _replication_measures_up(500 * GB, _Source(10_000 * GB), None, 6) is False


def test_an_unmeasurable_build_side_keeps_kybers_decision(probe_bytes):
    """`0` means unknown. Overriding a planner decision on no evidence would decline exactly
    the fan-out this guard exists to protect."""
    probe_bytes(10 * GB)
    assert _replication_measures_up(0.0, _Source(10 * GB), None, 6) is True


def test_an_unmeasurable_probe_side_keeps_kybers_decision(probe_bytes):
    probe_bytes(0.0)
    assert _replication_measures_up(1 * GB, _Source(0.0), None, 6) is True


def test_a_single_device_fleet_has_nothing_to_weigh(probe_bytes):
    """With one device there is no split to buy and no second copy to pay for."""
    probe_bytes(1 * GB)
    assert _replication_measures_up(100 * GB, _Source(1 * GB), None, 1) is True
