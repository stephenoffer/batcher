"""The range and consistency checks themselves, one function per `Config` section.

Pure: each takes its section and raises `ConfigError` on the first bad value. The order
here follows the order the sections appear on `Config`, so a reader looking for "what
constrains `flow_control.aimd_beta`" has exactly one place to look, and adding a tunable
has exactly one place to touch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import ConfigError
from batcher.config.accelerator import validate_accelerator
from batcher.config.config import VERBOSITY_LEVELS
from batcher.config.profiles import AUTOSCALE_WAIT_AUTO, RESILIENCE_PROFILES

if TYPE_CHECKING:
    from batcher.config.config import (
        Config,
        DistributedConfig,
        ExecutionConfig,
        FlowControlConfig,
        MemoryConfig,
        MetadataConfig,
        ObservabilityConfig,
        OptimizerConfig,
        PIDConfig,
        ShuffleTlsConfig,
    )

__all__ = ["run_checks"]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def run_checks(cfg: Config) -> None:
    """Run every section's checks. Pure; raises `ConfigError` on the first bad value.

    One function per config section, in the order the sections appear on `Config`, so a
    reader looking for "what constrains `flow_control.aimd_beta`" has exactly one place to
    look and adding a tunable has exactly one place to touch.
    """
    _check_memory(cfg.memory)
    _check_execution(cfg.execution)
    _check_distributed(cfg.distributed)
    _check_flow_control(cfg.flow_control)
    _check_optimizer(cfg.optimizer)
    validate_accelerator(cfg.accelerator)
    _check_pid(cfg.pid)
    _check_metadata(cfg.metadata)
    _check_observability(cfg.observability)


def _check_memory(m: MemoryConfig) -> None:
    """The memory envelope: fractions ordered and in (0, 1], caps and budgets positive."""
    _check(
        0.0 < m.soft_limit <= m.hard_limit <= 1.0,
        f"memory limits must satisfy 0 < soft_limit ({m.soft_limit}) <= "
        f"hard_limit ({m.hard_limit}) <= 1",
    )
    _check(
        m.max_memory_bytes is None or m.max_memory_bytes > 0,
        f"memory.max_memory_bytes must be positive or None, got {m.max_memory_bytes}",
    )
    _check(
        m.default_total_bytes > 0,
        f"memory.default_total_bytes must be positive, got {m.default_total_bytes}",
    )

    _check(
        m.streaming_state_max_bytes >= 0,
        f"memory.streaming_state_max_bytes must be >= 0, got {m.streaming_state_max_bytes}",
    )
    _check(
        m.result_cache_max_bytes >= 0 and m.file_cache_max_bytes >= 0,
        "memory result/file cache budgets must be >= 0, got "
        f"{m.result_cache_max_bytes}, {m.file_cache_max_bytes}",
    )
    _check(
        m.spill_bucket_max_bytes > 0,
        f"memory.spill_bucket_max_bytes must be positive, got {m.spill_bucket_max_bytes}",
    )
    _check(
        m.spill_local_budget_bytes is None or m.spill_local_budget_bytes >= 0,
        f"memory.spill_local_budget_bytes must be non-negative or None, "
        f"got {m.spill_local_budget_bytes}",
    )


def _check_execution(e: ExecutionConfig) -> None:
    """Execution sizing: morsels, CPU shares, splits, bloom, thresholds, skew buckets."""
    _check(e.parallelism >= 0, f"execution.parallelism must be >= 0, got {e.parallelism}")
    _check(e.morsel_rows > 0, f"execution.morsel_rows must be positive, got {e.morsel_rows}")
    _check(e.morsel_bytes > 0, f"execution.morsel_bytes must be positive, got {e.morsel_bytes}")
    _check(e.cpus_per_task > 0, f"execution.cpus_per_task must be positive, got {e.cpus_per_task}")
    _check(e.cpu_share_io > 0, f"execution.cpu_share_io must be positive, got {e.cpu_share_io}")
    _check(e.cpu_share_min > 0, f"execution.cpu_share_min must be positive, got {e.cpu_share_min}")

    _check(e.split_bytes > 0, f"execution.split_bytes must be positive, got {e.split_bytes}")
    _check(
        0.0 < e.bloom_fp_rate < 1.0,
        f"execution.bloom_fp_rate must be in (0, 1), got {e.bloom_fp_rate}",
    )
    _check(
        e.bloom_min_build_rows >= 0,
        f"execution.bloom_min_build_rows must be >= 0, got {e.bloom_min_build_rows}",
    )
    _check(
        e.window_parallel_row_threshold >= 0,
        f"execution.window_parallel_row_threshold must be >= 0, "
        f"got {e.window_parallel_row_threshold}",
    )
    _check(
        e.radix_parallel_threshold >= 0,
        f"execution.radix_parallel_threshold must be >= 0, got {e.radix_parallel_threshold}",
    )
    _check(
        e.sort_merge_fanin >= 2,
        f"execution.sort_merge_fanin must be >= 2, got {e.sort_merge_fanin}",
    )
    _check(
        e.skew_bucket_factor >= 1,
        f"execution.skew_bucket_factor must be >= 1, got {e.skew_bucket_factor}",
    )
    _check(
        e.skew_min_bucket_rows >= 0 and e.skew_min_bucket_bytes >= 0,
        "execution.skew_min_bucket_{rows,bytes} must be >= 0, got "
        f"{e.skew_min_bucket_rows}, {e.skew_min_bucket_bytes}",
    )


def _check_distributed(d: DistributedConfig) -> None:
    """Every distributed tunable: failure budgets, placement, and the shuffle's TLS."""
    _check_distributed_faults(d)
    _check_distributed_placement(d)
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
        d.skew_join_salt >= 0, f"distributed.skew_join_salt must be >= 0, got {d.skew_join_salt}"
    )
    _check(
        0.0 <= d.skew_join_fraction <= 1.0,
        f"distributed.skew_join_fraction must be in [0, 1], got {d.skew_join_fraction}",
    )


def _check_distributed_placement(d: DistributedConfig) -> None:
    """Transport choice, speculation thresholds, and how tasks spread across the cluster."""
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


def _check_flow_control(fc: FlowControlConfig) -> None:
    """Credit-window sizing and the AIMD control law's coefficients."""
    _check(
        fc.default_credits >= 1,
        f"flow_control.default_credits must be >= 1, got {fc.default_credits}",
    )
    _check(
        fc.credit_ceiling_factor >= 1,
        f"flow_control.credit_ceiling_factor must be >= 1, got {fc.credit_ceiling_factor}",
    )
    _check(
        fc.credit_byte_budget > 0,
        f"flow_control.credit_byte_budget must be positive, got {fc.credit_byte_budget}",
    )
    _check(fc.aimd_alpha >= 1, f"flow_control.aimd_alpha must be >= 1, got {fc.aimd_alpha}")
    _check(
        0.0 < fc.aimd_beta < 1.0,
        f"flow_control.aimd_beta (multiplicative decrease) must be in (0, 1), got {fc.aimd_beta}",
    )
    _check(
        0.0 <= fc.backpressure_low <= fc.backpressure_high <= 1.0,
        "flow_control backpressure thresholds must satisfy 0 <= backpressure_low "
        f"({fc.backpressure_low}) <= backpressure_high ({fc.backpressure_high}) <= 1",
    )
    _check(
        fc.shuffle_fan_in >= 2,
        f"flow_control.shuffle_fan_in must be >= 2 (a combiner tree needs fan-in), "
        f"got {fc.shuffle_fan_in}",
    )


def _check_optimizer(o: OptimizerConfig) -> None:
    """Task sizing, join-planning thresholds, learning rates, and cardinality fallbacks."""
    card = o.cardinality
    _check(
        o.target_rows_per_task >= 1,
        f"optimizer.target_rows_per_task must be >= 1, got {o.target_rows_per_task}",
    )
    _check(
        o.target_bytes_per_task >= 1,
        f"optimizer.target_bytes_per_task must be >= 1, got {o.target_bytes_per_task}",
    )
    _check(
        o.broadcast_max_bytes >= 0,
        f"optimizer.broadcast_max_bytes must be >= 0, got {o.broadcast_max_bytes}",
    )
    _check(
        o.fixpoint_iterations >= 1,
        f"optimizer.fixpoint_iterations must be >= 1, got {o.fixpoint_iterations}",
    )
    _check(o.row_bytes >= 1, f"optimizer.row_bytes must be >= 1, got {o.row_bytes}")
    _check(
        0.0 <= o.learning_smoothing_alpha <= 1.0,
        f"optimizer.learning_smoothing_alpha must be in [0, 1], got {o.learning_smoothing_alpha}",
    )
    _check(
        o.reoptimize_error > 0,
        f"optimizer.reoptimize_error must be positive, got {o.reoptimize_error}",
    )
    _check(
        1 <= o.join_dp_max_tables <= o.greedy_max_tables,
        "optimizer join thresholds must satisfy 1 <= join_dp_max_tables "
        f"({o.join_dp_max_tables}) <= greedy_max_tables ({o.greedy_max_tables})",
    )
    _check(
        o.cost_calibration_min_samples >= 1,
        f"optimizer.cost_calibration_min_samples must be >= 1, "
        f"got {o.cost_calibration_min_samples}",
    )
    _check(
        o.cost_calibration_clamp > 0,
        f"optimizer.cost_calibration_clamp must be positive, got {o.cost_calibration_clamp}",
    )

    # Cardinality — Selinger fallbacks: a probability in [0, 1], MCV fraction in (0, 1].
    _check(
        card.unknown_rows > 0, f"cardinality.unknown_rows must be positive, got {card.unknown_rows}"
    )
    for name, val in (
        ("default_filter_selectivity", card.default_filter_selectivity),
        ("eq_selectivity", card.eq_selectivity),
        ("range_selectivity", card.range_selectivity),
        ("null_selectivity", card.null_selectivity),
    ):
        _check(0.0 <= val <= 1.0, f"cardinality.{name} must be in [0, 1], got {val}")
    _check(
        0.0 < card.mcv_min_fraction <= 1.0,
        f"cardinality.mcv_min_fraction must be in (0, 1], got {card.mcv_min_fraction}",
    )


def _check_pid(pid: PIDConfig) -> None:
    """Controller gains: non-negative, since a negative gain inverts the control law."""
    _check(
        pid.kp >= 0 and pid.ki >= 0 and pid.kd >= 0,
        f"pid gains must be >= 0, got kp={pid.kp}, ki={pid.ki}, kd={pid.kd}",
    )
    _check(pid.integral_clamp > 0, f"pid.integral_clamp must be positive, got {pid.integral_clamp}")
    _check(
        0.0 < pid.max_step_fraction <= 1.0,
        f"pid.max_step_fraction must be in (0, 1], got {pid.max_step_fraction}",
    )


def _check_metadata(md: MetadataConfig) -> None:
    """Metadata store: the backend name and the per-day decay fraction."""
    _check(
        md.backend in {"in_process", "sqlite", "redis", "object_storage"},
        "metadata.backend must be one of {'in_process', 'sqlite', 'redis', "
        f"'object_storage'}}, got {md.backend!r}",
    )
    _check(
        0.0 <= md.decay_per_day <= 1.0,
        f"metadata.decay_per_day must be in [0, 1], got {md.decay_per_day}",
    )


def _check_observability(ob: ObservabilityConfig) -> None:
    """Verbosity, log level, progress, and log-file rotation.

    `None` is valid for `log_level` and `progress`: it means "derive from verbosity", and
    is their default, so only an explicitly-set value is enum-checked.
    """
    # `None` is valid for `log_level` and `progress`: it means "derive from verbosity", and
    # is their default. Only an explicitly-set value is enum-checked.
    _check(
        ob.log_level is None or ob.log_level in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"},
        "observability.log_level must be None or one of CRITICAL/ERROR/WARNING/INFO/DEBUG, "
        f"got {ob.log_level!r}",
    )
    _check(
        ob.progress is None or ob.progress in {"auto", "on", "off"},
        f"observability.progress must be None or 'auto'/'on'/'off', got {ob.progress!r}",
    )
    _check(
        _valid_verbosity(ob.verbosity),
        "observability.verbosity must be one of "
        f"{'/'.join(level.name for level in VERBOSITY_LEVELS)} or 0-{len(VERBOSITY_LEVELS) - 1}, "
        f"got {ob.verbosity!r}",
    )
    _check(
        ob.log_format in {"human", "json"},
        f"observability.log_format must be 'human' or 'json', got {ob.log_format!r}",
    )
    _check(
        ob.log_file_max_bytes > 0 and ob.log_file_backups >= 0,
        "observability log-file rotation must satisfy log_file_max_bytes > 0 and "
        f"log_file_backups >= 0, got {ob.log_file_max_bytes}, {ob.log_file_backups}",
    )


def _valid_verbosity(value: object) -> bool:
    """Whether `value` names a verbosity rung, by name or by index.

    `bool` is rejected explicitly: it is an `int` in Python, so `verbosity=True` would
    otherwise silently validate as rung 1 ("quiet") — a confusing way to spell something the
    user almost certainly did not mean.
    """
    if isinstance(value, bool):
        return False
    names = {level.name for level in VERBOSITY_LEVELS}
    if isinstance(value, int):
        return 0 <= value < len(VERBOSITY_LEVELS)
    text = str(value).strip().lower()
    if text.isdigit():
        return 0 <= int(text) < len(VERBOSITY_LEVELS)
    return text in names


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
