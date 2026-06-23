"""Carbonite shuffle recovery: the recompute-on-failure loop and lineage epochs.

The recovery coordinator is pure (caller supplies attempt/recompute closures), so
its convergence and give-up semantics are pinned here without Ray or the engine.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ResourceError
from batcher.carbonite.resilience import RecoveryPolicy, ShuffleLineage, ShuffleRecovery

pytestmark = pytest.mark.unit


def test_clean_run_does_not_recompute():
    rec = ShuffleRecovery()
    result = rec.run(lambda: ("done", set()), lambda failed: None)
    assert result == "done"
    assert rec.recomputes == 0


def test_recovers_after_one_recompute():
    healed = {"ok": False}

    def attempt():
        return ("done", set()) if healed["ok"] else (None, {2})

    def recompute(failed):
        assert failed == {2}
        healed["ok"] = True

    rec = ShuffleRecovery()
    assert rec.run(attempt, recompute) == "done"
    assert rec.recomputes == 1


def test_exhausts_attempts_and_raises():
    rec = ShuffleRecovery(RecoveryPolicy(max_attempts=2))
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        return (None, {1})  # never recovers

    with pytest.raises(ResourceError, match="did not recover"):
        rec.run(attempt, lambda failed: None)
    assert calls["n"] == 2  # tried exactly max_attempts times


def test_lineage_reincarnate_bumps_epoch_immutably():
    lin = ShuffleLineage(stage=0, src_partition=3)
    assert lin.epoch == 0
    nxt = lin.reincarnate()
    assert (nxt.stage, nxt.src_partition, nxt.epoch) == (0, 3, 1)
    assert lin.epoch == 0  # original unchanged


def test_ticket_epoch_fences_recomputed_partition():
    """A recomputed source publishes under a new epoch, so its ticket differs from
    the stale one a lost worker left — a reducer asking for the new epoch can never
    resolve the old partial. Epoch 0 keeps the clean-run ticket unchanged."""
    from batcher.dist.flight_worker import _ticket

    stale = _ticket(0, 2, 1)  # epoch defaults to 0 — the original publish
    fresh = _ticket(0, 2, 1, ShuffleLineage(0, 2).reincarnate().epoch)
    assert str(stale).endswith("/0")
    assert str(fresh).endswith("/1")
    assert str(stale) != str(fresh)
