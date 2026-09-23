"""Range and combination checks for the `distributed` section and its shuffle TLS block.

Split out of `sections.py` for the reason `gpu.py` was: that module is at its size limit.
This is the largest single section by some margin, and it divides cleanly in three -- the
failure budgets a job retries under, where its tasks are allowed to land, and the TLS
material the shuffle needs before it will carry a byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.config.profiles import AUTOSCALE_WAIT_AUTO, RESILIENCE_PROFILES
from batcher.config.validation.check import check as _check
from batcher.config.validation.gpu import check_gpu_packing

if TYPE_CHECKING:
    from batcher.config.config import DistributedConfig, ShuffleTlsConfig

__all__ = ["check_distributed"]


def check_distributed(d: DistributedConfig) -> None:
    """Every distributed tunable: failure budgets, placement, and the shuffle's TLS."""
    _check_distributed_faults(d)
    _check_distributed_placement(d)
    check_gpu_packing(d)
    _check_shuffle_tls(d.tls)


def _check_distributed_faults(d: DistributedConfig) -> None:
    """Retry budgets, backoff, timeouts, and shuffle replication.

    Retries and restarts may be zero (a fleet that never retries is a legitimate choice),
    but an *attempt* count may not: zero attempts means the work never runs at all, which
    is a misconfiguration rather than a policy.
    """
    _check(
        d.task_max_retries >= 0,
        f"distributed.task_max_retries must be >= 0, got {d.task_max_retries}",
    )
    _check(
        d.actor_max_restarts >= 0,
        f"distributed.actor_max_restarts must be >= 0, got {d.actor_max_restarts}",
    )
    _check(
        d.actor_max_task_retries >= 0,
        f"distributed.actor_max_task_retries must be >= 0, got {d.actor_max_task_retries}",
    )
    _check(
        d.recovery_max_attempts >= 1,
        f"distributed.recovery_max_attempts must be >= 1, got {d.recovery_max_attempts}",
    )
    _check(
        d.recovery_backoff_base_s >= 0,
        f"distributed.recovery_backoff_base_s must be >= 0, got {d.recovery_backoff_base_s}",
    )
    _check(
        d.drain_lead_s >= 0,
        f"distributed.drain_lead_s must be >= 0, got {d.drain_lead_s}",
    )
    _check(
        d.flight_idle_timeout_s > 0,
        f"distributed.flight_idle_timeout_s must be positive, got {d.flight_idle_timeout_s}",
    )
    _check(
        d.flight_keepalive_s is None or d.flight_keepalive_s > 0,
        f"distributed.flight_keepalive_s must be positive or None, got {d.flight_keepalive_s}",
    )
    _check(
        d.placement_timeout_s > 0,
        f"distributed.placement_timeout_s must be positive, got {d.placement_timeout_s}",
    )
    _check(
        d.cluster_connect_timeout_s >= 0,
        f"distributed.cluster_connect_timeout_s must be >= 0, got {d.cluster_connect_timeout_s}",
    )
    _check(
        d.autoscale_wait_s >= 0 or d.autoscale_wait_s == AUTOSCALE_WAIT_AUTO,
        f"distributed.autoscale_wait_s must be >= 0 (or {AUTOSCALE_WAIT_AUTO} for auto), "
        f"got {d.autoscale_wait_s}",
    )
    _check(
        d.autoscale_poll_s > 0,
        f"distributed.autoscale_poll_s must be positive, got {d.autoscale_poll_s}",
    )
    _check(
        d.autoscale_stall_s >= 0,
        f"distributed.autoscale_stall_s must be >= 0, got {d.autoscale_stall_s}",
    )
    _check(
        d.fleet_max_attempts >= 1,
        f"distributed.fleet_max_attempts must be >= 1, got {d.fleet_max_attempts}",
    )
    _check(
        d.speculation_max_backups >= 0,
        f"distributed.speculation_max_backups must be >= 0, got {d.speculation_max_backups}",
    )
    _check(
        d.shuffle_replication >= 1,
        f"distributed.shuffle_replication must be >= 1 (1 = no replica), "
        f"got {d.shuffle_replication}",
    )
    _check(
        d.resilience in RESILIENCE_PROFILES,
        f"distributed.resilience must be one of {sorted(RESILIENCE_PROFILES)}, "
        f"got {d.resilience!r}",
    )
    _check(
        d.skew_join_salt >= -1,
        f"distributed.skew_join_salt must be >= -1 (-1 = never salt, 0 = salt on measured "
        f"skew, >0 = force this fan-out), got {d.skew_join_salt}",
    )
    _check(
        0.0 <= d.skew_join_fraction <= 1.0,
        f"distributed.skew_join_fraction must be in [0, 1], got {d.skew_join_fraction}",
    )


def _check_distributed_placement(d: DistributedConfig) -> None:
    """Transport choice, speculation thresholds, and how tasks spread across the cluster."""
    _check(
        d.mode in {"auto", "always", "never"},
        f"distributed.mode must be one of {{'auto', 'always', 'never'}}, got {d.mode!r}",
    )
    _check(
        d.transport in {"auto", "flight", "disk"},
        f"distributed.transport must be one of {{'auto', 'flight', 'disk'}}, got {d.transport!r}",
    )
    _check(
        d.on_read_error in {"error", "skip"},
        f"distributed.on_read_error must be one of {{'error', 'skip'}}, got {d.on_read_error!r}",
    )
    _check(
        d.speculation_straggler_factor >= 1.0,
        f"distributed.speculation_straggler_factor must be >= 1, "
        f"got {d.speculation_straggler_factor}",
    )
    _check(
        0.0 < d.speculation_min_finished_frac <= 1.0,
        f"distributed.speculation_min_finished_frac must be in (0, 1], "
        f"got {d.speculation_min_finished_frac}",
    )
    _check(
        d.session_fleet_idle_s >= 0,
        f"distributed.session_fleet_idle_s must be >= 0, got {d.session_fleet_idle_s}",
    )
    _check(
        d.object_store_memory_bytes is None or d.object_store_memory_bytes > 0,
        f"distributed.object_store_memory_bytes must be positive or None, "
        f"got {d.object_store_memory_bytes}",
    )
    _check(
        d.map_partition_multiplier >= 1,
        f"distributed.map_partition_multiplier must be >= 1 (1 = one partition per worker), "
        f"got {d.map_partition_multiplier}",
    )
    _check(
        d.max_pending_tasks >= 0,
        f"distributed.max_pending_tasks must be >= 0 (0 = derive), got {d.max_pending_tasks}",
    )
    _check(
        d.pending_window_factor >= 1,
        f"distributed.pending_window_factor must be >= 1, got {d.pending_window_factor}",
    )
    _check(
        d.map_spread in {"auto", "always", "never"},
        f"distributed.map_spread must be one of {{'auto', 'always', 'never'}}, "
        f"got {d.map_spread!r}",
    )
    _check(
        d.runtime_bloom_join in (True, False, "auto"),
        "distributed.runtime_bloom_join must be True, False, or 'auto', "
        f"got {d.runtime_bloom_join!r}",
    )
    _check(
        d.task_events in {"auto", "always", "never"},
        f"distributed.task_events must be one of {{'auto', 'always', 'never'}}, "
        f"got {d.task_events!r}",
    )
    _check(
        d.task_events_fanout_cap >= 1,
        f"distributed.task_events_fanout_cap must be >= 1, got {d.task_events_fanout_cap}",
    )
    _check(
        d.map_spread_node_cap >= 1,
        f"distributed.map_spread_node_cap must be >= 1, got {d.map_spread_node_cap}",
    )
    _check(
        d.map_spread_pack_share > 0,
        f"distributed.map_spread_pack_share must be positive, got {d.map_spread_pack_share}",
    )
    _check(
        d.map_inflight_depth >= 1,
        f"distributed.map_inflight_depth must be >= 1, got {d.map_inflight_depth}",
    )


def _check_shuffle_tls(t: ShuffleTlsConfig) -> None:
    """With TLS on, the server identity and trust root must all be present.

    The only *combination* check in this module, and the reason it exists: a
    half-configured deployment must fail at config time, not at its first fetch, when the
    fleet is already up and the failure looks like a network fault.
    """
    if not t.enabled:
        return
    _check(
        bool(t.ca_cert_path),
        "distributed.tls.enabled requires ca_cert_path (the peer trust root)",
    )
    _check(
        bool(t.server_cert_path) and bool(t.server_key_path),
        "distributed.tls.enabled requires server_cert_path and server_key_path",
    )
    _check(
        not t.require_client_auth or bool(t.ca_cert_path),
        "distributed.tls.require_client_auth (mTLS) requires ca_cert_path to verify "
        "client certificates against",
    )
    _check(
        bool(t.client_cert_path) == bool(t.client_key_path),
        "distributed.tls client_cert_path and client_key_path must be set together",
    )
    _check(
        bool(t.server_name),
        "distributed.tls.enabled requires server_name (the peer certificate SAN)",
    )
