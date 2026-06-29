"""Config range/consistency validation.

Lives next to `config.py` (not inside it) so the single-source-of-truth `Config`
module stays focused on the dataclasses and the active-config plumbing. `Config`
calls `validate_config(self)` at every entry point (`set_config`, `config_context`,
`from_env`, `from_file`) so an out-of-range or inconsistent tunable raises
`ConfigError` early and clearly instead of surfacing as a confusing runtime failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import ConfigError
from batcher.config.profiles import RESILIENCE_PROFILES

if TYPE_CHECKING:
    from batcher.config.config import Config

__all__ = ["validate_config"]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate_config(cfg: Config) -> None:
    """Raise `ConfigError` if any tunable is out of range or inconsistent.

    Pure (no side effects); covers the memory envelope, execution sizing, the
    distributed fault-tolerance budgets/timeouts, and flow-control credits.
    """
    m, e, d, fc = cfg.memory, cfg.execution, cfg.distributed, cfg.flow_control
    o, card, pid, md, ob = (
        cfg.optimizer,
        cfg.optimizer.cardinality,
        cfg.pid,
        cfg.metadata,
        cfg.observability,
    )

    # Memory envelope: fractions ordered and in (0, 1]; caps positive.
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

    # Execution sizing.
    _check(e.parallelism >= 0, f"execution.parallelism must be >= 0, got {e.parallelism}")
    _check(e.morsel_rows > 0, f"execution.morsel_rows must be positive, got {e.morsel_rows}")
    _check(e.morsel_bytes > 0, f"execution.morsel_bytes must be positive, got {e.morsel_bytes}")
    _check(e.cpus_per_task > 0, f"execution.cpus_per_task must be positive, got {e.cpus_per_task}")
    _check(e.cpu_share_io > 0, f"execution.cpu_share_io must be positive, got {e.cpu_share_io}")
    _check(e.cpu_share_min > 0, f"execution.cpu_share_min must be positive, got {e.cpu_share_min}")

    # Distributed fault tolerance: non-negative budgets, >= 1 attempt, positive timeouts.
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
        d.autoscale_wait_s >= 0,
        f"distributed.autoscale_wait_s must be >= 0, got {d.autoscale_wait_s}",
    )
    _check(
        d.autoscale_poll_s > 0,
        f"distributed.autoscale_poll_s must be positive, got {d.autoscale_poll_s}",
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

    # Flow control credits.
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

    # Execution sizing (extended) — splits, bloom, parallel thresholds, skew buckets.
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

    # Memory (extended) — caches and spill bucket sizing.
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
        m.spill_local_budget_bytes is None or m.spill_local_budget_bytes > 0,
        f"memory.spill_local_budget_bytes must be positive or None, "
        f"got {m.spill_local_budget_bytes}",
    )

    # Distributed (extended) — transport enum, speculation thresholds, object-store knob.
    _check(
        d.transport in {"auto", "flight", "disk"},
        f"distributed.transport must be one of {{'auto', 'flight', 'disk'}}, got {d.transport!r}",
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

    # Optimizer — task sizing, join-planning thresholds, learning, cardinality.
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

    # PID controller gains — non-negative (a negative gain inverts control), bounds positive.
    _check(
        pid.kp >= 0 and pid.ki >= 0 and pid.kd >= 0,
        f"pid gains must be >= 0, got kp={pid.kp}, ki={pid.ki}, kd={pid.kd}",
    )
    _check(pid.integral_clamp > 0, f"pid.integral_clamp must be positive, got {pid.integral_clamp}")
    _check(
        0.0 < pid.max_step_fraction <= 1.0,
        f"pid.max_step_fraction must be in (0, 1], got {pid.max_step_fraction}",
    )

    # Metadata store — backend enum and a decay fraction in [0, 1].
    _check(
        md.backend in {"in_process", "sqlite", "redis", "object_storage"},
        "metadata.backend must be one of {'in_process', 'sqlite', 'redis', "
        f"'object_storage'}}, got {md.backend!r}",
    )
    _check(
        0.0 <= md.decay_per_day <= 1.0,
        f"metadata.decay_per_day must be in [0, 1], got {md.decay_per_day}",
    )

    # Observability — log-level / format enums and positive file-rotation sizing.
    _check(
        ob.log_level in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"},
        "observability.log_level must be one of CRITICAL/ERROR/WARNING/INFO/DEBUG, "
        f"got {ob.log_level!r}",
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
