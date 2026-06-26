"""Phase-0 fault-tolerance foundations: the ``spot`` resilience profile and the
proactive preemption monitor.

The profile must harden the distributed retry/recovery budgets as a bundle while
keeping the default profile (and every value the user pinned) untouched. The monitor
must flip a sticky draining flag and fire its drain hook exactly once.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.carbonite.resilience import PreemptionMonitor
from batcher.config import Config, DistributedConfig
from batcher.config.profiles import apply_resilience_profile
from batcher.config.validation import validate_config


def _with_resilience(**overrides: object) -> Config:
    return Config().replace(distributed=dataclasses.replace(Config().distributed, **overrides))


def test_default_profile_is_identity():
    cfg = apply_resilience_profile(Config())
    base = DistributedConfig()
    assert cfg.distributed.actor_max_restarts == base.actor_max_restarts
    assert cfg.distributed.recovery_max_attempts == base.recovery_max_attempts
    assert cfg.distributed.flight_keepalive_s == base.flight_keepalive_s
    assert cfg.distributed.speculation_max_backups == base.speculation_max_backups


def test_spot_profile_hardens_the_budgets():
    cfg = apply_resilience_profile(_with_resilience(resilience="spot"))
    d = cfg.distributed
    # Each knob is strictly stronger than its conservative default.
    assert d.actor_max_restarts > DistributedConfig().actor_max_restarts
    assert d.recovery_max_attempts > DistributedConfig().recovery_max_attempts
    assert d.recovery_backoff_base_s > 0
    assert d.flight_keepalive_s is not None  # keepalive on
    assert d.speculation_max_backups >= 1  # one straggler backup
    assert d.fleet_max_attempts > DistributedConfig().fleet_max_attempts  # more fleet retries


def test_explicit_override_wins_over_profile():
    # A pinned knob survives; the rest of the profile still applies.
    cfg = apply_resilience_profile(_with_resilience(resilience="spot", actor_max_restarts=9))
    assert cfg.distributed.actor_max_restarts == 9
    assert cfg.distributed.recovery_max_attempts == 6


def test_profile_is_idempotent():
    once = apply_resilience_profile(_with_resilience(resilience="spot"))
    assert apply_resilience_profile(once) == once


def test_unknown_profile_is_rejected():
    with pytest.raises(Exception, match="resilience"):
        validate_config(_with_resilience(resilience="bogus"))


def test_profile_resolves_through_env_entry_point():
    cfg = Config.from_env({"BATCHER_DISTRIBUTED_RESILIENCE": "spot"})
    assert cfg.distributed.actor_max_restarts > 1


def test_monitor_starts_undrained_and_triggers_once():
    fired: list[str] = []
    mon = PreemptionMonitor(probe=lambda: False)
    mon.on_drain(lambda: fired.append("flush"))
    assert mon.is_draining() is False
    mon.trigger()
    mon.trigger()  # idempotent — the hook fires exactly once
    assert mon.is_draining() is True
    assert fired == ["flush"]


def test_monitor_late_registration_fires_immediately_when_draining():
    fired: list[str] = []
    mon = PreemptionMonitor(probe=lambda: False)
    mon.trigger()
    mon.on_drain(lambda: fired.append("late"))
    assert fired == ["late"]


def test_monitor_poll_loop_detects_drain():
    # A probe that flips to True drives the sticky flag without an explicit trigger.
    states = iter([False, True])
    mon = PreemptionMonitor(probe=lambda: next(states, True), poll_interval_s=0.01)
    fired: list[str] = []
    mon.on_drain(lambda: fired.append("drain"))
    mon.start()
    try:
        for _ in range(200):
            if mon.is_draining():
                break
            import time

            time.sleep(0.01)
        assert mon.is_draining() is True
        assert fired == ["drain"]
    finally:
        mon.stop()


def test_monitor_drain_hook_exception_never_escapes():
    def boom() -> None:
        raise RuntimeError("hook failed")

    mon = PreemptionMonitor(probe=lambda: False)
    mon.on_drain(boom)
    mon.trigger()  # must not raise
    assert mon.is_draining() is True


def test_detect_spot_environment_from_flag(monkeypatch):
    from batcher.config.profiles import detect_spot_environment

    monkeypatch.delenv("BATCHER_SPOT", raising=False)
    assert detect_spot_environment() is False
    monkeypatch.setenv("BATCHER_SPOT", "1")
    assert detect_spot_environment() is True


def test_detect_spot_environment_from_lifecycle(monkeypatch):
    from batcher.config.profiles import detect_spot_environment

    monkeypatch.delenv("BATCHER_SPOT", raising=False)
    monkeypatch.setenv("INSTANCE_LIFECYCLE", "spot")
    assert detect_spot_environment() is True


def test_spot_env_auto_applies_profile(monkeypatch):
    # On a detected spot node, a config with the default profile auto-upgrades to "spot"
    # (stronger retries) without the user choosing it.

    from batcher.config import Config
    from batcher.config.config import _resolved

    monkeypatch.setenv("BATCHER_SPOT", "1")
    resolved = _resolved(Config())
    assert resolved.distributed.resilience == "spot"
    assert resolved.distributed.actor_max_restarts == 4  # the spot profile value

    # Off a spot node, the default profile stands.
    monkeypatch.delenv("BATCHER_SPOT", raising=False)
    assert _resolved(Config()).distributed.resilience == "default"
