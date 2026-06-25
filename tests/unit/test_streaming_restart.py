"""Phase-4: a streaming query survives a transient mid-stream fault by restarting
from its checkpoint, instead of one preempted worker killing the whole stream.

These drive `_run_resilient` directly with an injected `_loop`, so the restart
policy (transient ⇒ restart-from-checkpoint, bounded by consecutive no-progress
attempts; non-transient or no-checkpoint ⇒ surface immediately) is tested without a
live source/sink/cluster.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import ResourceError
from batcher.config import active_config
from batcher.core.streaming_query import StreamingQueryEngine
from batcher.io.formats.streaming.checkpoint import CheckpointStore


def _engine(checkpoint) -> StreamingQueryEngine:
    return StreamingQueryEngine(
        name="t",
        source=object(),
        sink=object(),
        processor=object(),
        trigger=object(),
        output_mode="append",
        checkpoint=checkpoint,
    )


def test_transient_fault_restarts_from_checkpoint(tmp_path):
    eng = _engine(CheckpointStore(str(tmp_path / "ckpt")))
    calls = {"loop": 0, "recover": 0}

    def fake_loop():
        calls["loop"] += 1
        if calls["loop"] == 1:
            eng._batches += 1  # committed a batch (progress) before faulting
            raise ResourceError("worker lost mid-stream")
        # second attempt drains cleanly

    eng._loop = fake_loop
    eng._recover = lambda: calls.__setitem__("recover", calls["recover"] + 1)

    eng._run_resilient()  # must not raise
    assert calls["loop"] == 2  # faulted once, then succeeded
    assert calls["recover"] == 1  # rolled back to the checkpoint before retrying


def test_non_transient_error_surfaces_immediately(tmp_path):
    eng = _engine(CheckpointStore(str(tmp_path / "ckpt")))

    def fake_loop():
        raise ValueError("a real logic bug")

    eng._loop = fake_loop
    eng._recover = lambda: None
    with pytest.raises(ValueError, match="logic bug"):
        eng._run_resilient()


def test_no_checkpoint_means_no_restart():
    eng = _engine(None)  # no durable restore point

    def fake_loop():
        raise ResourceError("worker lost")

    eng._loop = fake_loop
    with pytest.raises(ResourceError):
        eng._run_resilient()


def test_consecutive_restarts_are_bounded(tmp_path):
    eng = _engine(CheckpointStore(str(tmp_path / "ckpt")))
    calls = {"loop": 0}

    def fake_loop():
        calls["loop"] += 1
        raise ResourceError("never recovers")  # no progress between faults

    eng._loop = fake_loop
    eng._recover = lambda: None
    with pytest.raises(ResourceError):
        eng._run_resilient()
    # initial attempt + recovery_max_attempts consecutive restarts, then it surfaces
    assert calls["loop"] == active_config().distributed.recovery_max_attempts + 1


def test_progress_resets_the_restart_budget(tmp_path):
    # A long stream that faults repeatedly but makes progress each time must never
    # exhaust the budget — the counter resets whenever a batch commits.
    eng = _engine(CheckpointStore(str(tmp_path / "ckpt")))
    budget = active_config().distributed.recovery_max_attempts
    calls = {"loop": 0}

    def fake_loop():
        calls["loop"] += 1
        if calls["loop"] <= budget + 3:  # more faults than the raw budget
            eng._batches += 1  # but each makes progress
            raise ResourceError("flaky but advancing")

    eng._loop = fake_loop
    eng._recover = lambda: None
    eng._run_resilient()  # must not raise despite > budget faults
    assert calls["loop"] == budget + 4


def test_checkpoint_durability_warning_under_spot(monkeypatch):
    # Phase 4c: under resilience="spot", a node-local checkpoint location warns (a
    # reclaimed node would lose exactly-once recovery); a durable URI does not.
    import dataclasses
    import warnings

    from batcher.api.streaming import _warn_if_checkpoint_not_durable
    from batcher.config import Config, config_context

    spot = Config().replace(
        distributed=dataclasses.replace(Config().distributed, resilience="spot")
    )
    with config_context(spot):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_checkpoint_not_durable("/local/scratch/ckpt")
            _warn_if_checkpoint_not_durable("s3://bucket/ckpt")
        msgs = [str(w.message) for w in caught]
        assert len(msgs) == 1 and "node-local" in msgs[0]

    # Default profile never warns (no spot-cluster assumption).
    with config_context(Config()), warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_checkpoint_not_durable("/local/scratch/ckpt")
