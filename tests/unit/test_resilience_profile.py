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
from batcher.config.profiles import AUTOSCALE_WAIT_AUTO, apply_resilience_profile
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


def test_spot_profile_enables_autoscale_wait():
    # A spot cluster is an autoscaling one: the profile turns on a bounded wait so a
    # stage that over-asks briefly waits for replacement nodes instead of clamping to
    # the shrunken current capacity and running under-provisioned.
    cfg = apply_resilience_profile(_with_resilience(resilience="spot"))
    assert cfg.distributed.autoscale_wait_s > 0.0
    # The default profile leaves it at the "auto" sentinel — the config layer resolves
    # that per-environment (a bounded wait on an autoscaling cluster, off on a fixed one).
    assert apply_resilience_profile(Config()).distributed.autoscale_wait_s == AUTOSCALE_WAIT_AUTO


def test_spot_profile_upgrades_metadata_to_object_storage(monkeypatch):
    # A discoverable shared location upgrades the still-default in-process store to
    # durable object storage, so learning survives a driver moving between spot nodes.
    monkeypatch.setenv("BATCHER_METADATA_URI", "s3://bkt/prefix")
    cfg = apply_resilience_profile(_with_resilience(resilience="spot"))
    assert cfg.metadata.backend == "object_storage"
    assert cfg.metadata.uri == "s3://bkt/prefix"


def test_spot_profile_uses_managed_artifact_storage(monkeypatch):
    # Absent an explicit BATCHER_METADATA_URI, a managed cluster's persistent artifact
    # storage is discovered and namespaced so batcher's objects don't collide with others.
    monkeypatch.delenv("BATCHER_METADATA_URI", raising=False)
    monkeypatch.setenv("ANYSCALE_ARTIFACT_STORAGE", "s3://org/artifacts/")
    cfg = apply_resilience_profile(_with_resilience(resilience="spot"))
    assert cfg.metadata.backend == "object_storage"
    assert cfg.metadata.uri == "s3://org/artifacts/batcher-metadata"


def test_spot_metadata_no_location_stays_in_process(monkeypatch):
    # With no shared location, the store stays in-process rather than pretend a
    # driver-local file is durable on a cluster where the driver can move.
    monkeypatch.delenv("BATCHER_METADATA_URI", raising=False)
    monkeypatch.delenv("ANYSCALE_ARTIFACT_STORAGE", raising=False)
    cfg = apply_resilience_profile(_with_resilience(resilience="spot"))
    assert cfg.metadata.backend == "in_process"
    assert cfg.metadata.uri is None


def test_spot_metadata_respects_explicit_backend(monkeypatch):
    # A user who pinned a backend keeps it — the profile never overrides an explicit choice.
    import dataclasses as _dc

    monkeypatch.setenv("BATCHER_METADATA_URI", "s3://bkt/prefix")
    base = _with_resilience(resilience="spot")
    pinned = base.replace(metadata=_dc.replace(base.metadata, backend="sqlite"))
    cfg = apply_resilience_profile(pinned)
    assert cfg.metadata.backend == "sqlite"


def _autoscale_env_vars() -> tuple[str, ...]:
    """Every env var the detectors read, taken from the module itself.

    Listed by hand this drifts the moment a platform is added — and it drifts *silently*,
    because a var the fixture forgets to clear is simply inherited from the host. That is
    a real failure mode here: these tests run on Ray/KubeRay CI, where the managed-cluster
    markers are genuinely set in the environment."""
    from batcher.config import profiles

    return (
        *profiles._AUTOSCALE_FLAG_VARS,
        *profiles._SPOT_FLAG_VARS,
        *profiles._SPOT_LIFECYCLE_VARS,
        *profiles._MANAGED_AUTOSCALE_VARS,
    )


_AUTOSCALE_ENV_VARS = _autoscale_env_vars()


@pytest.fixture
def _clean_autoscale_env(monkeypatch):
    """A cluster with no autoscaling/spot/managed signal — the fixed-cluster baseline."""
    for var in _AUTOSCALE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_detect_autoscaling_off_on_fixed_cluster(_clean_autoscale_env):
    from batcher.config.profiles import detect_autoscaling_environment

    assert detect_autoscaling_environment() is False


def test_detect_autoscaling_managed_cluster(_clean_autoscale_env):
    # An Anyscale marker implies an autoscaler is present.
    from batcher.config.profiles import detect_autoscaling_environment

    _clean_autoscale_env.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    assert detect_autoscaling_environment() is True


def test_detect_autoscaling_flag_is_authoritative_both_ways(_clean_autoscale_env):
    # The explicit flag overrides even a managed-cluster marker — the power-user opt-out.
    from batcher.config.profiles import detect_autoscaling_environment

    _clean_autoscale_env.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    _clean_autoscale_env.setenv("BATCHER_AUTOSCALE", "0")
    assert detect_autoscaling_environment() is False
    _clean_autoscale_env.setenv("BATCHER_AUTOSCALE", "1")
    assert detect_autoscaling_environment() is True


def test_detect_autoscaling_spot_implies_autoscaling(_clean_autoscale_env):
    from batcher.config.profiles import detect_autoscaling_environment

    _clean_autoscale_env.setenv("BATCHER_SPOT", "1")
    assert detect_autoscaling_environment() is True


@pytest.mark.parametrize(
    ("body", "draining"),
    [
        ('{"Events": []}', False),  # the steady state — always 200, usually empty
        ('{"Events": [{"EventType": "Preempt"}]}', True),  # spot eviction
        ('{"Events": [{"EventType": "Terminate"}]}', True),  # announced shutdown
        ('{"Events": [{"EventType": "Reboot"}]}', False),  # maintenance, not reclamation
        ('{"Events": [{"EventType": "Freeze"}]}', False),
        ('{"Events": [{"EventType": "Freeze"}, {"EventType": "Preempt"}]}', True),
        ("not json", False),  # a malformed body must never read as a drain
        ("null", False),
        ("{}", False),
    ],
)
def test_azure_scheduled_events_drain_detection(body, draining):
    """Azure always answers 200, so the payload — not the status — decides.

    Without this probe the `spot` profile is a silent no-op on Azure: the budgets harden
    but nothing ever notices a preemption, so eviction costs a full recompute instead of
    the proactive migration the profile exists to buy."""
    from batcher.carbonite.resilience.preemption import _azure_is_draining

    assert _azure_is_draining(body) is draining


def test_detect_managed_cluster_off_with_no_signal(_clean_autoscale_env):
    from batcher.config.profiles import detect_managed_cluster

    assert detect_managed_cluster() is False


@pytest.mark.parametrize(
    "var",
    [
        "ANYSCALE_SESSION_ID",  # Anyscale
        "RAY_CLUSTER_NAME",  # KubeRay, on any cloud or on-prem Kubernetes
        "RAY_CLUSTER_NAMESPACE",
        "RAY_USAGE_STATS_KUBERAY_IN_USE",
        "BATCHER_RAY_CLUSTER",  # the escape hatch for an unnamed platform
    ],
)
def test_detect_managed_cluster_is_platform_neutral(_clean_autoscale_env, var):
    """Every platform marker is equally authoritative — batcher must not privilege one
    vendor, or "attach to the running cluster" works on that vendor and silently strands a
    distributed job on one node everywhere else."""
    from batcher.config.profiles import detect_managed_cluster

    _clean_autoscale_env.setenv(var, "x")
    assert detect_managed_cluster() is True


def test_managed_cluster_implies_autoscaling_on_kuberay(_clean_autoscale_env):
    from batcher.config.profiles import detect_autoscaling_environment

    _clean_autoscale_env.setenv("RAY_CLUSTER_NAME", "raycluster-prod")
    assert detect_autoscaling_environment() is True


def test_resolve_autoscale_wait_auto_enables_on_autoscaling(_clean_autoscale_env):
    from batcher.config.profiles import AUTOSCALE_WAIT_DEFAULT_S, resolve_autoscale_wait

    _clean_autoscale_env.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    # A still-default (sentinel) config auto-enables the bounded wait — no tuning.
    assert Config().distributed.autoscale_wait_s == AUTOSCALE_WAIT_AUTO
    resolved = resolve_autoscale_wait(Config())
    assert resolved.distributed.autoscale_wait_s == AUTOSCALE_WAIT_DEFAULT_S


def test_resolve_autoscale_wait_off_on_fixed_cluster(_clean_autoscale_env):
    # The sentinel resolves to a concrete `0` (non-blocking) on a fixed cluster, so the
    # runtime never sees `-1` and single-node stays immediate.
    from batcher.config.profiles import resolve_autoscale_wait

    assert resolve_autoscale_wait(Config()).distributed.autoscale_wait_s == 0.0


def test_resolve_autoscale_wait_honors_explicit_off(_clean_autoscale_env):
    # An explicit `0` is honored even on an autoscaling cluster (distinct from the sentinel).
    from batcher.config.profiles import resolve_autoscale_wait

    _clean_autoscale_env.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    cfg = _with_resilience(autoscale_wait_s=0.0)
    assert resolve_autoscale_wait(cfg).distributed.autoscale_wait_s == 0.0


def test_resolve_autoscale_wait_honors_explicit_budget(_clean_autoscale_env):
    from batcher.config.profiles import resolve_autoscale_wait

    _clean_autoscale_env.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    cfg = _with_resilience(autoscale_wait_s=42.0)
    assert resolve_autoscale_wait(cfg).distributed.autoscale_wait_s == 42.0


def test_spot_profile_beats_autoscale_default(_clean_autoscale_env):
    # Spot resolves through the profile (concrete 180) before the auto step sees it, so the
    # two mechanisms don't double-apply; the spot value stands.
    from batcher.config.profiles import AUTOSCALE_WAIT_DEFAULT_S, resolve_autoscale_wait

    _clean_autoscale_env.setenv("BATCHER_SPOT", "1")
    spotted = apply_resilience_profile(_with_resilience(resilience="spot"))
    assert resolve_autoscale_wait(spotted).distributed.autoscale_wait_s == AUTOSCALE_WAIT_DEFAULT_S


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
