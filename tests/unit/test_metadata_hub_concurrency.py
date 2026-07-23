"""Concurrent pipelines must not lose each other's feedback in the shared hub.

`MetadataHub` is a process singleton (`core.default_hub`), so every pipeline folds its
measurements into the same object. `record` derives half its storage key from `self._seq`,
and `self._seq += 1` is a read-modify-write: without serialization two queries can take the
same sequence number, and one query's row silently overwrites the other's.

Nothing raises and no result is wrong — the plans just quietly stop improving, which is the
"Core measures, Kyber decides" failure mode the engineering contract calls out by name.
"""

from __future__ import annotations

import sys
import threading

import pytest

from batcher.metadata.backends import InProcessBackend
from batcher.metadata.hub import MetadataHub
from batcher.plan.feedback import OperatorFeedback


@pytest.fixture
def _preemptive(monkeypatch):
    """Force aggressive thread switching, so a read-modify-write race is reproducible.

    At the default 5 ms switch interval the window inside `record` is small enough that the
    race almost never fires, which is exactly what lets it survive in a test suite.
    """
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)
    yield
    sys.setswitchinterval(original)


def _feedback(signature: str) -> OperatorFeedback:
    return OperatorFeedback(
        op_id=1,
        kind="aggregate",
        n_actual=5,
        t_op_ms=1.0,
        m_peak_bytes=0,
        selectivity=1.0,
        batch_size=1024,
        signature=signature,
    )


def test_concurrent_pipelines_do_not_overwrite_each_others_feedback(_preemptive):
    """Every recorded row must land under its own key, whoever else is recording."""
    hub = MetadataHub(InProcessBackend())
    keys: list[object] = []
    guard = threading.Lock()
    original_put = hub._backend.put

    def _tracking_put(space, key, value, *args, **kwargs):
        with guard:
            keys.append(key)
        return original_put(space, key, value, *args, **kwargs)

    hub._backend.put = _tracking_put

    n_pipelines, per_pipeline = 8, 1500

    def pipeline(i: int) -> None:
        for _ in range(per_pipeline):
            hub.record(_feedback(f"sig{i}"))

    threads = [threading.Thread(target=pipeline, args=(i,)) for i in range(n_pipelines)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    expected = n_pipelines * per_pipeline
    assert len(keys) == expected, "a feedback row never reached the backend"
    collisions = expected - len(set(keys))
    assert collisions == 0, f"{collisions} feedback rows overwrote another pipeline's"
