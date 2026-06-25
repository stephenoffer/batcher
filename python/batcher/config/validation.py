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
