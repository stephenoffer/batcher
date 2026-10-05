"""A broadcast join's output bound is the node's to set, not a spill estimate's.

The worker's grant is its *spill threshold* -- an estimate of the plan's peak divided across
tasks -- and as the hard bound on a probe task's joined output it declined broadcasts that fit
easily: TPC-H q9 at SF1000 gave up at 0.1 GiB per node and ran a 28-minute single-worker
shuffle. The bound is now at least the node share, and what stops a broadcast from taking a
node down is the engine's live headroom reading.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist import flight_broadcast as fb

pytestmark = pytest.mark.unit

_GB = 1 << 30


class _Engine:
    def __init__(self, reading):
        self.reading = reading

    def memory_headroom(self):
        return self.reading


def test_a_small_grant_does_not_become_the_bound(monkeypatch):
    import psutil

    total = psutil.virtual_memory().total
    share = int(total * fb._OUTPUT_BUDGET_FRACTION)
    assert fb._output_budget(100 << 20) == max(100 << 20, share)
    assert fb._output_budget(total) == total  # a large grant still stands


def test_output_is_refused_past_the_bound_and_when_the_node_runs_short(monkeypatch):
    batch = pa.record_batch({"x": pa.array(range(1000), pa.int64())})
    monkeypatch.setattr(fb, "engine", lambda: _Engine((40 * _GB, 4 * _GB)))
    held = fb._charge(0, 1 << 40, [batch])  # positive control: roomy node, large bound
    assert held > 0
    with pytest.raises(fb.BroadcastOutputTooLarge):
        fb._charge(0, 1, [batch])  # over the bound
    monkeypatch.setattr(fb, "engine", lambda: _Engine((7 * _GB, 4 * _GB)))
    with pytest.raises(fb.BroadcastOutputTooLarge):
        fb._charge(0, 1 << 40, [batch])  # within twice the floor
    monkeypatch.setattr(fb, "engine", lambda: _Engine(None))
    assert fb._charge(0, 1 << 40, [batch]) > 0  # unreadable: the bound alone decides
