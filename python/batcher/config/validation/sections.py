"""The range and consistency checks themselves, one function per `Config` section.

Pure: each takes its section and raises `ConfigError` on the first bad value. The order
here follows the order the sections appear on `Config`, so a reader looking for "what
constrains `flow_control.aimd_beta`" has exactly one place to look, and adding a tunable
has exactly one place to touch. The `distributed` section is the exception: it is large
enough to have its own module, and `run_checks` calls into it in section order like the
rest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.config.accelerator import validate_accelerator
from batcher.config.config import VERBOSITY_LEVELS
from batcher.config.fault_tolerance import validate_fault_tolerance
from batcher.config.validation.check import check as _check
from batcher.config.validation.distributed import check_distributed

if TYPE_CHECKING:
    from batcher.config.config import (
        Config,
        ExecutionConfig,
        FlowControlConfig,
        GovernanceConfig,
        MemoryConfig,
        MetadataConfig,
        ObservabilityConfig,
        OptimizerConfig,
        PIDConfig,
    )

__all__ = ["run_checks"]


def run_checks(cfg: Config) -> None:
    """Run every section's checks. Pure; raises `ConfigError` on the first bad value.

    One function per config section, in the order the sections appear on `Config`, so a
    reader looking for "what constrains `flow_control.aimd_beta`" has exactly one place to
    look and adding a tunable has exactly one place to touch.
    """
    _check_memory(cfg.memory)
    _check_execution(cfg.execution)
    check_distributed(cfg.distributed)
    _check_flow_control(cfg.flow_control)
    _check_optimizer(cfg.optimizer)
    validate_accelerator(cfg.accelerator)
    validate_fault_tolerance(cfg.fault_tolerance)
    _check_pid(cfg.pid)
    _check_metadata(cfg.metadata)
    _check_governance(cfg.governance)
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
        m.result_cache_max_bytes >= 0
        and m.file_cache_max_bytes >= 0
        and m.result_cache_disk_max_bytes >= 0,
        "memory result/file cache budgets must be >= 0, got "
        f"{m.result_cache_max_bytes}, {m.file_cache_max_bytes}, "
        f"{m.result_cache_disk_max_bytes}",
    )
    _check(
        m.shared_cache_ttl_seconds >= 0,
        f"memory.shared_cache_ttl_seconds must be >= 0, got {m.shared_cache_ttl_seconds}",
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
    _check(
        0.0 < m.oom_kill_backoff <= 1.0,
        f"memory.oom_kill_backoff must be in (0, 1], got {m.oom_kill_backoff}",
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
    # The *floor* on the same blend, and it needs the same bound for a sharper reason. Two of
    # its three consumers (`kyber.learning._smooth`, `learned_tuning.priors`) use it as
    # `alpha = max(floor, 1/(n+1))` and then `alpha*observed + (1-alpha)*prior` with no upper
    # clamp, so a floor above 1 makes `(1 - alpha)` negative: the estimate moves *past* the
    # observation, away from the prior, and diverges instead of converging. Blending 100
    # toward 200 at a floor of 3.0 yields 400. `metadata.smoothed` clamps its step to 1.0 and
    # so escapes it; the other two do not, and nothing stopped the value being set.
    _check(
        0.0 <= o.learned_scalar_alpha_floor <= 1.0,
        "optimizer.learned_scalar_alpha_floor must be in [0, 1], got "
        f"{o.learned_scalar_alpha_floor}",
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


#: The only values `governance.mode` may take. Enforcement reads it by equality
#: (`== "off"`, `== "strict"`) and treats everything else as `advisory`, so an unrecognized
#: value does not fail -- it *downgrades*, which is the one direction a security control
#: must never move on its own.
GOVERNANCE_MODES = ("off", "advisory", "strict")


def _check_governance(g: GovernanceConfig) -> None:
    """Reject a governance setting that cannot be honored, rather than honoring less of it.

    Both checks here exist because this section fails **open**. `_refuse_ungoverned_read`
    selects on `mode == "off"` and `mode == "strict"` and lets every other string fall
    through to the advisory warning, so ``mode="Strict"`` -- a capitalization, not even a
    typo -- is accepted by config, reported nowhere, and silently turns a deployment that
    refuses ungoverned reads into one that logs them and proceeds. A wrong value for a
    security switch has to be an error at the point it is set; there is no later point
    where anything notices.

    `default_deny` is the same failure one step further along: it is declared, documented
    in `docs/configuration/options.md`, and read by nothing at all, so setting it grants a
    deny-by-default catalog that does not exist. Until something implements it, refusing
    the value is the only honest answer -- silently ignoring a request to deny more is
    strictly worse than saying it is unavailable.
    """
    _check(
        g.mode in GOVERNANCE_MODES,
        f"governance.mode must be one of {', '.join(map(repr, GOVERNANCE_MODES))}, got "
        f"{g.mode!r}. Enforcement treats an unrecognized mode as 'advisory', so this would "
        f"have downgraded a strict deployment to warnings instead of refusals.",
    )
    _check(
        not g.default_deny,
        "governance.default_deny is not implemented: no code path reads it, so setting it "
        "True would leave every table a grant does not mention readable, exactly as False "
        "does. Restrict access with explicit grants and governance.mode='strict' instead.",
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
