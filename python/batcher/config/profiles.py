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
import os

from batcher.config.config import (
    AUTOSCALE_WAIT_AUTO,
    Config,
    DistributedConfig,
    MetadataConfig,
)

__all__ = [
    "AUTOSCALE_WAIT_AUTO",
    "AUTOSCALE_WAIT_DEFAULT_S",
    "RESILIENCE_PROFILES",
    "apply_resilience_profile",
    "detect_autoscaling_environment",
    "detect_managed_cluster",
    "detect_spot_environment",
    "resolve_autoscale_wait",
]

# Default bounded autoscale wait (seconds) turned on for an autoscaling-capable cluster —
# the spot profile and the out-of-the-box auto-enable share this single value. Longer than
# a cloud node's boot time so a genuinely-launching node is not abandoned; the
# `autoscale_stall_s` grace window still bails fast on a fixed cluster that will not grow.
AUTOSCALE_WAIT_DEFAULT_S = 180.0

# Env vars whose presence marks a preemptible/spot node. The first group is an explicit
# opt-in (set by the launcher/Dockerfile); the second is a node-lifecycle hint some
# orchestrators surface. Detection is env-var only — never a metadata-service network
# call on a hot path — so a deployment with no signal sets `BATCHER_SPOT=1` (or passes
# `resilience="spot"`). Truthy means spot.
_SPOT_TRUE = frozenset({"1", "true", "yes", "on", "spot", "preemptible", "preempt"})
_SPOT_FLAG_VARS = ("BATCHER_SPOT", "RAY_SPOT")
_SPOT_LIFECYCLE_VARS = ("RAY_NODE_TYPE_NAME", "NODE_LIFECYCLE", "INSTANCE_LIFECYCLE")

# Explicit autoscaling opt-in/out flag (truthy → autoscaling; falsey → force fixed, even on
# a managed cluster) and the managed-cluster markers that imply an autoscaling-capable
# control plane. Even a rare fixed-size managed cluster bails fast via the stall window, so
# treating "managed cluster" as "can grow" is safe.
_AUTOSCALE_FLAG_VARS = ("BATCHER_AUTOSCALE", "RAY_AUTOSCALING")
# Managed Ray control planes, by platform. Batcher must behave the same on any of them, so
# this is a list of *equally-weighted* vendor markers, not a primary plus special cases:
#
#   * Anyscale sets `ANYSCALE_SESSION_ID`/`ANYSCALE_CLUSTER_ID` on every cluster.
#   * The KubeRay operator sets `RAY_CLUSTER_NAME` and `RAY_CLUSTER_NAMESPACE` on every
#     Ray pod it creates, and `RAY_USAGE_STATS_KUBERAY_IN_USE` to mark itself — this is
#     the on-prem / self-hosted / any-cloud Kubernetes case (EKS, GKE, AKS, OpenShift,
#     bare metal), which is where most non-managed Ray actually runs.
#   * `BATCHER_RAY_CLUSTER` is the escape hatch for a platform none of the above name —
#     a hand-rolled on-prem cluster, a vendor we have not seen. Set it to any non-empty
#     value and batcher attaches to the running cluster instead of starting a local one.
#
# Env-var only, no network call (see `detect_spot_environment`). Extend this tuple for a
# new platform; nothing else needs to change.
_MANAGED_AUTOSCALE_VARS = (
    "BATCHER_RAY_CLUSTER",
    "ANYSCALE_SESSION_ID",
    "ANYSCALE_CLUSTER_ID",
    "RAY_CLUSTER_NAME",
    "RAY_CLUSTER_NAMESPACE",
    "RAY_USAGE_STATS_KUBERAY_IN_USE",
)


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


def detect_autoscaling_environment() -> bool:
    """Best-effort detection of an autoscaling-capable cluster from cheap local signals.

    An explicit `BATCHER_AUTOSCALE` (or `RAY_AUTOSCALING`) flag is authoritative in *both*
    directions — truthy forces autoscaling on, falsey forces it off even on a managed
    cluster (the power-user opt-out). Absent that, a spot node is autoscaling by definition,
    and a managed-cluster marker (Anyscale) implies an autoscaler is present. No network
    call — env vars only, like `detect_spot_environment`. Used to auto-enable the bounded
    autoscale wait so a query fills the cluster it triggers a scale-up for, out of the box.
    """
    for var in _AUTOSCALE_FLAG_VARS:
        raw = os.environ.get(var)
        if raw is not None and raw.strip():
            return raw.strip().lower() in _SPOT_TRUE
    if detect_spot_environment():
        return True
    return detect_managed_cluster()


def detect_managed_cluster() -> bool:
    """Best-effort detection of a managed Ray control plane from cheap local signals.

    Covers Anyscale, any KubeRay-operated cluster (the on-prem / self-hosted / any-cloud
    Kubernetes case), and an explicit `BATCHER_RAY_CLUSTER` for a platform we do not name —
    see `_MANAGED_AUTOSCALE_VARS`. No platform is privileged: the signals are equivalent,
    so batcher behaves identically wherever it runs.

    When true and no Ray address is configured, batcher attaches to the *running* cluster
    (`ray.init(address="auto")`) instead of starting a local single-node Ray — the fix for
    a cluster that exports neither `RAY_ADDRESS` nor Ray's current-cluster pointer, where a
    bare `ray.init()` would silently strand a distributed job on one node. Env-var only (no
    network call), like `detect_spot_environment`; `_ensure_ray` still falls back to a local
    start if no cluster turns out to be reachable, so a false positive degrades rather than
    fails.
    """
    return any(os.environ.get(v, "").strip() for v in _MANAGED_AUTOSCALE_VARS)


def resolve_autoscale_wait(cfg: Config) -> Config:
    """Resolve the `AUTOSCALE_WAIT_AUTO` sentinel to a concrete wait budget, so out of the
    box a query fills the cluster it triggers a scale-up for with no tuning.

    Only the sentinel (`-1`, the library default) is resolved — an explicit value wins,
    including `0` to disable the wait even on an autoscaling cluster (the spot profile
    likewise sets a concrete value upstream). The sentinel becomes the bounded default wait
    on an autoscaling-capable cluster (`detect_autoscaling_environment`) and `0` (off) on a
    fixed or non-cloud one, so the runtime only ever sees a concrete `>= 0` and single-node
    / CI runs stay non-blocking. Pure scheduling — the result is identical either way; the
    power-user opt-out is an explicit `autoscale_wait_s=0` or `BATCHER_AUTOSCALE=0`.
    """
    if cfg.distributed.autoscale_wait_s != AUTOSCALE_WAIT_AUTO:
        return cfg
    wait = AUTOSCALE_WAIT_DEFAULT_S if detect_autoscaling_environment() else 0.0
    return cfg.replace(distributed=dataclasses.replace(cfg.distributed, autoscale_wait_s=wait))


# The set of valid `DistributedConfig.resilience` values (validated in
# `config.validation`). ``"default"`` is the identity profile.
RESILIENCE_PROFILES: frozenset[str] = frozenset({"default", "spot"})

# The ``"spot"`` profile's distributed fault-tolerance overrides. Each is stronger
# than the on-demand default: enough actor restarts and recompute rounds to ride out
# repeated preemption, a backoff base that spaces recovery so a preemption *wave* is
# not retried in a tight loop, HTTP/2 keepalive on to detect a silently-dropped
# connection well before the idle timeout, and one speculative backup so a
# degraded-but-alive node cannot stall a shuffle barrier. `autoscale_wait_s` is turned
# on because a spot cluster *is* an autoscaling one — a stage that over-asks should
# briefly wait for the autoscaler to bring up replacement nodes (spot capacity churns)
# rather than immediately clamping to the shrunken current capacity and running
# under-provisioned. Longer than a node's boot time.
_SPOT_DISTRIBUTED: dict[str, object] = {
    "actor_max_restarts": 4,
    "actor_max_task_retries": 3,
    "task_max_retries": 4,
    "recovery_max_attempts": 6,
    "recovery_backoff_base_s": 1.0,
    "flight_keepalive_s": 20.0,
    "speculation_max_backups": 1,
    "fleet_max_attempts": 6,
    "autoscale_wait_s": AUTOSCALE_WAIT_DEFAULT_S,
    # Keep a second copy of every mapper's shuffle output on another node, so a
    # preemption — the expected event here, not an exceptional one — is served from the
    # replica instead of paying a full map recompute (re-reading the source from object
    # storage). One extra copy of the (pre-aggregated, small) partial state buys the
    # single largest reduction in recovery cost on a churning cluster.
    "shuffle_replication": 2,
}

# Env vars naming a durable, cross-node location for learned stats, in priority order. The
# first is batcher's explicit override (any fsspec URL); the second is a managed cluster's
# persistent artifact storage, which survives cluster restarts and driver moves. On a spot
# cluster the driver itself can be preempted and reschedule onto a different node, so an
# in-process (or driver-local) store loses everything learned; pointing the store at shared
# object storage is what makes "adapt from *past* runs" hold across the churn.
_METADATA_URI_VARS = ("BATCHER_METADATA_URI", "ANYSCALE_ARTIFACT_STORAGE")


def _durable_metadata_uri() -> str | None:
    """A cross-node object-storage root for learned stats, discovered from the env.

    Returns an explicit `BATCHER_METADATA_URI` (any fsspec URL) verbatim, or the managed
    cluster's artifact-storage root with a `batcher-metadata` sub-prefix so batcher's objects
    don't collide with other artifacts written under the same root. `None` when no shared
    location is known — in which case the store stays in-process rather than pretend a
    driver-local file is durable on a cluster where the driver can move.
    """
    for var in _METADATA_URI_VARS:
        val = os.environ.get(var, "").strip()
        if val:
            root = val.rstrip("/")
            return root if var == "BATCHER_METADATA_URI" else f"{root}/batcher-metadata"
    return None


def _spot_metadata(cfg: Config) -> Config:
    """Upgrade a still-default in-process learned-stats store to durable object storage.

    Only fires when the store is still the library default (`in_process`, no `uri`) so an
    explicit choice is preserved, and only when a shared location is discoverable — so a
    dev run with no object store keeps its zero-config in-process behavior.
    """
    default_meta = MetadataConfig()
    if cfg.metadata.backend != default_meta.backend or cfg.metadata.uri is not None:
        return cfg
    uri = _durable_metadata_uri()
    if uri is None:
        return cfg
    return cfg.replace(
        metadata=dataclasses.replace(cfg.metadata, backend="object_storage", uri=uri)
    )


def apply_resilience_profile(cfg: Config) -> Config:
    """Overlay the selected resilience profile's defaults onto `cfg`.

    Returns `cfg` unchanged for the ``"default"`` profile. For ``"spot"``, each
    managed field still at its library default is raised to the profile's value while
    a field the user set explicitly is preserved (``explicit > profile > default``).
    This also upgrades a still-default in-process learned-stats store to durable object
    storage when a shared location is discoverable (see `_durable_metadata_uri`), so
    cross-run learning survives a preempted driver rescheduling onto another node.
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
    if overrides:
        cfg = cfg.replace(distributed=dataclasses.replace(cfg.distributed, **overrides))
    # Durable cross-run learning: only for a still-default store, and only when a shared
    # location is discoverable (no-op otherwise, so single-node/dev runs are unchanged).
    return _spot_metadata(cfg)
