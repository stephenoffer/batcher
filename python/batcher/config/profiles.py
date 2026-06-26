"""Named fault-tolerance profiles for the distributed engine.

A profile is a bundle of fault-tolerance overrides applied on top of the library
defaults but *below* any value the user set explicitly. The ``"spot"`` profile
hardens the distributed retry/recovery budgets and failure-detection knobs for a
churning spot-node cluster, where the conservative defaults (tuned for a stable
on-demand cluster) retry too few times and notice a dropped peer too slowly.

Resolution runs once at every `Config` entry point (`batcher.config.config`), after
the env/file/programmatic layers are merged, so precedence is
``explicit override > profile > default``: a managed field still equal to its
default is the profile's to set; a field the user changed is left alone. Applying a
profile is idempotent.
"""

from __future__ import annotations

import dataclasses

from batcher.config.config import Config, DistributedConfig

__all__ = ["RESILIENCE_PROFILES", "apply_resilience_profile", "detect_spot_environment"]

# Env vars whose presence marks a preemptible/spot node. The first group is an explicit
# opt-in (set by the launcher/Dockerfile); the second is a node-lifecycle hint some
# orchestrators surface. Detection is env-var only — never a metadata-service network
# call on a hot path — so a deployment with no signal sets `BATCHER_SPOT=1` (or passes
# `resilience="spot"`). Truthy means spot.
_SPOT_TRUE = frozenset({"1", "true", "yes", "on", "spot", "preemptible", "preempt"})
_SPOT_FLAG_VARS = ("BATCHER_SPOT", "RAY_SPOT")
_SPOT_LIFECYCLE_VARS = ("RAY_NODE_TYPE_NAME", "NODE_LIFECYCLE", "INSTANCE_LIFECYCLE")


def detect_spot_environment() -> bool:
    """Best-effort detection of a preemptible/spot environment from cheap local signals.

    True when an explicit spot flag env var is truthy, or a node-lifecycle env var names
    a spot/preemptible instance. No network call (the cloud metadata service is avoided
    on the hot path). When detected and the user has not chosen a resilience profile, the
    config layer auto-selects ``"spot"`` so a job rides out preemption without tuning —
    while ``recovery_max_attempts`` etc. already give every job baseline recovery.
    """
    import os

    if any(os.environ.get(v, "").strip().lower() in _SPOT_TRUE for v in _SPOT_FLAG_VARS):
        return True
    return any(
        "spot" in os.environ.get(v, "").lower() or "preempt" in os.environ.get(v, "").lower()
        for v in _SPOT_LIFECYCLE_VARS
    )


# The set of valid `DistributedConfig.resilience` values (validated in
# `config.validation`). ``"default"`` is the identity profile.
RESILIENCE_PROFILES: frozenset[str] = frozenset({"default", "spot"})

# The ``"spot"`` profile's distributed fault-tolerance overrides. Each is stronger
# than the on-demand default: enough actor restarts and recompute rounds to ride out
# repeated preemption, a backoff base that spaces recovery so a preemption *wave* is
# not retried in a tight loop, HTTP/2 keepalive on to detect a silently-dropped
# connection well before the idle timeout, and one speculative backup so a
# degraded-but-alive node cannot stall a shuffle barrier.
_SPOT_DISTRIBUTED: dict[str, object] = {
    "actor_max_restarts": 4,
    "actor_max_task_retries": 3,
    "task_max_retries": 4,
    "recovery_max_attempts": 6,
    "recovery_backoff_base_s": 1.0,
    "flight_keepalive_s": 20.0,
    "speculation_max_backups": 1,
    "fleet_max_attempts": 6,
}


def apply_resilience_profile(cfg: Config) -> Config:
    """Overlay the selected resilience profile's defaults onto `cfg`.

    Returns `cfg` unchanged for the ``"default"`` profile. For ``"spot"``, each
    managed field still at its library default is raised to the profile's value while
    a field the user set explicitly is preserved (``explicit > profile > default``).
    Idempotent — applying twice yields the same config.

    Examples:
        .. doctest::

            >>> import dataclasses
            >>> from batcher.config import Config
            >>> from batcher.config.profiles import apply_resilience_profile
            >>> spot = Config().replace(
            ...     distributed=dataclasses.replace(
            ...         Config().distributed, resilience="spot"
            ...     )
            ... )
            >>> apply_resilience_profile(spot).distributed.actor_max_restarts
            4
    """
    if cfg.distributed.resilience == "default":
        return cfg
    baseline = DistributedConfig()
    overrides = {
        name: value
        for name, value in _SPOT_DISTRIBUTED.items()
        if getattr(cfg.distributed, name) == getattr(baseline, name)
    }
    if not overrides:
        return cfg
    return cfg.replace(distributed=dataclasses.replace(cfg.distributed, **overrides))
